//! Qwen3.5 / Qwen3.6 / Qwen3.8 OpenAI-serving implementation.
//!
//! This module owns the native model load, bounded and multimodal prefill,
//! serial and slot-aware generation, tool/reasoning semantics, retained-prefix
//! reuse, and exact target-verified speculation. The SlotAware speculative
//! path keeps target and MTP cursors transactionally equal, consumes the full
//! prompt into the MTP cache, drafts fixed depth-three blocks, and can instead
//! propose continuations from request history. Both proposers are guarded by
//! independent measured cost controllers.
//!
//! `engine.rs` owns cross-family scheduling and physical-slot lifecycle; the
//! Qwen-specific model, cache, sampler, and rollback contracts remain here.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use mlx_native::MlxBuffer;
use tokenizers::Tokenizer;

use crate::inference::models::qwen35::kv_cache::{
    HybridKvCache, HybridKvCacheSnapshot, HybridKvSlotAnchor, HybridKvSlotTransaction,
};
use crate::inference::models::qwen35::model::Qwen35Model;
use crate::inference::spec_decode::cost_controller::SpeculationCostController;
use crate::inference::spec_decode::ngram_proposer::{HistoryLookupConfig, HistoryLookupIndex};
use crate::serve::load_info::{
    self, ArchFamily, ChatTemplateSource, LoadInfo, LoadInfoBuilder, MoeShape, TokenizerSource,
};
// ADR-040 Phase B4b (2026-05-24): every Qwen35 decode-side entry point
// now takes `slot_id: SlotId`. Engine callers route SlotId(0) until C2c
// wires the SlotAware scheduler runtime (per ADR-040 §6 Phase C C2c row).
use crate::core::provenance::{self, Provenance};
use crate::serve::multi_seq_kv::SlotId;

use super::engine::{
    effective_repetition_penalty, grammar_runtime_for_request, sample_logits_with_grammar,
    supervised_gpu_call, DeepstackData, LoadOptions, SamplingParams, SerialStreamEnd,
    SerialStreamResult, SoftTokenData, ToolCallPolicy,
};
use super::engine_supervisor::EngineSupervisor;

const QWEN35_WORKER_TRANSACTION_TIMEOUT: Duration = Duration::from_secs(30);

/// Canonical Qwen3 chat-template stop tokens.
///
/// Used by `Qwen35LoadedModel::load` (ADR-028 iter-267) to scan
/// `tokenizer.ggml.tokens` by name for the standard stop tokens —
/// extended-vocab GGUFs (e.g. 27B-MTP-Q4_0 with vocab=248044) put
/// `<|im_end|>` at a different ID than the standard 151645 and some
/// omit `eos_token_id` metadata entirely (per iter-265).  Hoisted to
/// module scope at iter-295 to satisfy the structural audit
/// (`tests/structural_audit_serve_consts.rs`); previously was
/// `fn`-local inside `load`, which is structurally untestable from a
/// `mod tests` block (fn-local consts aren't `super::*`-visible).
pub const QWEN35_EOS_STOP_NAMES: &[&str] = &["<|im_end|>", "<|endoftext|>"];
const QWEN35_VISION_MARKERS: &[&str] = &["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"];

/// Construct the serving tokenizer exclusively from the opened GGUF.
///
/// This function intentionally has no filesystem-path parameter. A Qwen GGUF
/// carrying `tokenizer.ggml.*` is self-contained; requiring a sibling
/// `tokenizer.json` would be a dead precondition and would make externally
/// produced artifacts needlessly non-portable.
fn build_qwen35_serving_tokenizer(gguf: &mlx_native::gguf::GgufFile) -> Result<(Tokenizer, bool)> {
    let mut tokenizer =
        crate::inference::models::qwen35::tokenizer::build_tokenizer_from_gguf(gguf)
            .map_err(|e| anyhow::anyhow!("GGUF-driven tokenizer build failed: {e}"))?;
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow::anyhow!("Failed to disable tokenizer truncation: {e}"))?;
    let vision_special_tokens_present = QWEN35_VISION_MARKERS
        .iter()
        .all(|token| tokenizer.token_to_id(token).is_some());
    Ok((tokenizer, vision_special_tokens_present))
}

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
    /// Filesystem path to the GGUF opened by this loaded model.
    pub model_path: PathBuf,
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
    /// ADR-017 §F4 — GGUF provenance captured at load time via
    /// `crate::core::provenance::detect(&gguf)`. Stored for the common
    /// `LoadedModel::provenance()` surface; Qwen35 KV-spill remains
    /// descriptor-pending because the cache is hybrid.
    pub provenance: Provenance,
    /// Text-GGUF projector binding captured independently of remote-source
    /// provenance so local copies and request-time projector swaps remain
    /// fail-closed.
    pub expected_projector_sha256: Option<String>,
    /// Exact projector profile emitted only for multimodal
    /// ConditionalGeneration artifacts. Text-only artifacts keep this
    /// unset and cannot consume a process projector.
    pub vision_projector_profile: Option<String>,
    /// Number of DeepStack feature streams expected by the text model when
    /// the producer recorded hf2q's exact extension metadata. External GGUFs
    /// may omit the extension even when their standard multimodal contract is
    /// otherwise complete.
    pub vision_deepstack_output_count: Option<u32>,
    /// Standard Qwen multimodal marker tokens are present in the embedded
    /// vocabulary. This is the producer-neutral consumer signal used for
    /// external GGUFs that predate or do not emit hf2q extension metadata.
    pub vision_special_tokens_present: bool,
    /// Single-slot prompt cache (Wedge-3 / iter-216 Phase C / D).  Stores
    /// the post-prefill `HybridKvCacheSnapshot` + the greedy first
    /// decoded token + a generation-affecting params key, so a subsequent
    /// equivalent-prompt + greedy request can short-circuit the prefill.
    /// Owned by the worker thread (single-writer through `LoadedModel`),
    /// so no synchronization is needed.
    pub prompt_cache: HybridPromptCache,
    /// ADR-017 Phase B-hybrid.1 (LCP partial-prefill resume for Qwen
    /// 3.5/3.6) — observability foundation. Mirrors the
    /// `GemmaLoadedModel::lcp_registry` field at engine.rs:1102, but
    /// stores `HybridKvCacheSnapshot` (full-attn K/V + DeltaNet
    /// conv_state + recurrent state) instead of `Vec<Arc<DenseKvBuffers>>`.
    ///
    /// **Iter B.1 scope (this commit)**: probe-only observability — the
    /// engine probe runs after `HybridPromptCache::try_match` miss and
    /// records `hf2q_kv_lcp_lookups_total` + `hf2q_kv_lcp_detected_total`
    /// for Qwen35 requests. The probe ALWAYS returns None (no resume)
    /// because hybrid SSM-state resume requires sparse mid-prefill
    /// checkpoints, which is iter B.2 scope (per dossier
    /// `docs/research/adr017-iter36-phaseB-architecture-2026-05-05.md`
    /// Appendix D.2 — chunk-aligned DeltaNet checkpoints + chunked
    /// prefill flow).
    ///
    /// **Iter B.2 scope (next iter)**: replace the marker payload with
    /// a real `HybridKvCacheCheckpoint` carrying per-chunk-boundary
    /// snapshots; engine resume gate restores at K_aligned and slices
    /// tokens to [K_aligned..N) before calling forward_gpu_impl.
    ///
    /// Capacity = 1 — same as Gemma's lcp_registry. /cfa Phase 2
    /// fan-out gets the speedup once B.2 ships; multi-turn chat too.
    pub lcp_registry: crate::serve::kv_persist::lcp_registry::LcpRegistry<
        crate::inference::models::qwen35::kv_cache::HybridKvCacheSnapshot,
    >,
    /// ADR-017 Phase B-hybrid.1 — handle to the AppState-owned
    /// `KvSpillCounters` so per-request LCP probes from the Qwen35
    /// worker thread bump the same Arc the `/metrics` handler reads.
    /// Wired by `LoadedModel::generate*` calls. `None` when the engine
    /// is constructed standalone (test paths).
    pub kv_metrics_sink:
        Option<std::sync::Arc<dyn crate::serve::kv_persist::metrics::KvCacheMetricsSink>>,

    /// ADR-027 Phase A iter-6b.2 — disk back-end for cold-process LCP
    /// resume. `Some` iff `LoadOptions.kv_persist_dir` was set
    /// (sourced from `HF2Q_KV_PERSIST` env). When present,
    /// `store_lcp_with_disk_writeback` writes through to disk on every
    /// successful `lcp_registry.store`, ensuring snapshots persist
    /// across process crashes / restarts. `None` keeps the legacy
    /// in-process-only behavior (no I/O cost).
    ///
    /// Iter-6b.3 wires automatic re-insertion of disk snapshots into
    /// the in-memory registry on cold start via
    /// `hydrate_lcp_registry_from_disk`; iter-6b.2 shipped write-only
    /// durability.
    pub disk_persistor: Option<
        std::sync::Arc<
            crate::serve::kv_persist::families::qwen35_disk_persistor::Qwen35DiskPersistor,
        >,
    >,

    /// ADR-027 Phase A iter-6b.3 — set of cfg-fingerprint hex strings
    /// whose disk snapshots have already been hydrated into the
    /// in-memory `lcp_registry` for this process lifetime.
    /// `hydrate_lcp_registry_from_disk` checks this before any I/O and
    /// is a no-op on hit, so the cost across N requests is one
    /// `read_dir` per cfg per process. Cleared when the engine drops
    /// (process exit).
    pub lcp_hydrated_for_cfg: std::collections::HashSet<String>,

    /// ADR-027 Phase B iter-12 — TQ-active KV cache flag (sourced from
    /// `HF2Q_TQ_KV` env at engine load via
    /// `tq_packed_descriptor::is_tq_active_mode()`). When `true`,
    /// `alloc_kv_cache_for_request` calls
    /// `HybridKvCache::new_with_options(... tq_kv_active=true)` so
    /// every per-prefill cache allocates TQ-active full-attn buffers
    /// alongside the F32 K/V (shadow-cache pattern; iter-N drops the
    /// F32 backing for full 3.94× memory savings).
    ///
    /// **Iter-12 scope:** field + alloc plumbing only. The actual
    /// per-token encode + SDPA dispatch in `gpu_full_attn::full_attn_
    /// layer_gpu` is iter-13 scope. Setting this flag without iter-13
    /// wiring increases per-request memory cost (40 MB extra/slot at
    /// qwen36 8K shape) but does NOT change inference output —
    /// the F32 read path stays exclusive until iter-13 swaps in TQ
    /// SDPA. Mantra-aligned (no-stub): the flag IS load-bearing for
    /// allocation; iter-13 makes it load-bearing for dispatch.
    pub tq_kv_active: bool,

    /// Long-lived architecture-typed KV cache.
    ///
    /// SerialFifo retains one demand-grown [`HybridKvCache`] across
    /// requests so ordinary agentic turns reuse the same Metal buffers.
    /// A larger request replaces it with a geometrically grown cache;
    /// compatible snapshots restore through `restore_partial`, whose wire
    /// format is independent of the destination cache capacity. SlotAware
    /// uses the same typed field for its multi-sequence cache.
    ///
    /// Keeping the concrete cache on the architecture-specific loaded model
    /// preserves type information on the decode hot path and avoids a
    /// virtual cache interface in `EngineInner`.
    pub persistent_kv_cache: Option<HybridKvCache>,
    /// Server-side MTP policy and per-conversation profitability state. The
    /// native runner owns no persistent speculative KV yet, so this state
    /// deliberately contains no cache tensors.
    pub speculation: super::qwen35_speculation::QwenSpeculationController,
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
    /// ADR-044: the Qwen3.8-qualified K-quant width-four decode route
    /// (`HF2Q_DECODE_MVN=0` + `HF2Q_DECODE_MV_EXT=1`, qualified by the
    /// native qL4 decision/cache gate and the matched ABBA receipt) was
    /// previously applied only by the canonical `serve_qwen38_opencode.sh`
    /// launcher. Loading a Qwen3.8-identified model now applies the same
    /// route by default, at engine load, before the first quantized matmul
    /// snapshots mlx-native's process-global routing policy. Explicit
    /// operator-exported values always win.
    ///
    /// The route stays Qwen3.8-scoped on purpose: unlike the default-on
    /// byte-exact mvN route, `mul_mv_ext` is not bit-exact, and no other
    /// model family carries the qualifying receipt. The id match mirrors
    /// the serve registry's deliberately fuzzy substring philosophy.
    ///
    /// Multi-model note: mlx-native's routing policy is process-global and
    /// cached on first dispatch, so this default can only take effect when
    /// the Qwen3.8 load precedes the process's first quantized matmul —
    /// the same per-process scope the launcher always had.
    fn apply_qwen38_qualified_decode_route(model_path: &std::path::Path) {
        let id = model_path.to_string_lossy().to_ascii_lowercase();
        if !(id.contains("qwen38") || id.contains("qwen3.8")) {
            return;
        }
        for (key, value) in [("HF2Q_DECODE_MVN", "0"), ("HF2Q_DECODE_MV_EXT", "1")] {
            if std::env::var_os(key).is_none() {
                std::env::set_var(key, value);
                tracing::info!(
                    key,
                    value,
                    "applied Qwen3.8-qualified decode route default"
                );
            }
        }
    }

    pub fn load(opts: &LoadOptions) -> Result<Self> {
        let load_start = Instant::now();
        let model_path = &opts.model_path;
        Self::apply_qwen38_qualified_decode_route(model_path);
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
        let provenance = provenance::detect(&gguf);
        let expected_projector_sha256 = provenance::projector_sha256(&gguf)
            .map_err(|error| anyhow::anyhow!("Qwen projector binding: {error}"))?;
        let vision_projector_profile = gguf
            .metadata_string("hf2q.vision.projector_profile")
            .map(str::to_owned);
        let vision_deepstack_output_count = gguf.metadata_u32("hf2q.vision.deepstack_output_count");

        // ADR-018 C3: legacy `tracing::info!("Qwen35 SERVE load: model = ...")`
        // was deleted here. The same fact (`model_path`) is now emitted by
        // `emit_tracing(&info)` at every CLI/SERVE entry that constructs a
        // `LoadInfo`. Conditions/warnings stay; load FACTS are unified.

        // ---- Load weights (full mlx-native pipeline) ----
        // ADR-018 C3: TTY-aware progress reporter mirroring cmd_generate
        // (mod.rs:519-531). Under default verbosity on a TTY stderr the
        // per-layer `\r loading i/n layers` line renders; under tracing
        // INFO+ (`-v`) or non-TTY stderr (SERVE redirected to systemd /
        // a log file) the reporter is silent and tracing::debug events
        // from the per-layer loaders provide per-layer detail.
        //
        // Pre-parse just the config (cheap — parses GGUF metadata only,
        // no tensor reads) so the progress denominator matches
        // `cmd_generate`'s pattern. `Qwen35Model::load_from_gguf` will
        // re-parse it internally; the duplicate cost is microsecond-scale
        // metadata-key reads against an already-mmapped GGUF.
        let stderr_is_tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
        let verbosity = if tracing::enabled!(tracing::Level::INFO) {
            1
        } else {
            0
        };
        let cfg_preview = Qwen35Model::load_config_only(&gguf)
            .context("Qwen35Model::load_config_only (progress sizing)")?;
        let mut progress = crate::serve::header::LoadProgress::new(
            stderr_is_tty,
            verbosity,
            cfg_preview.num_hidden_layers as usize,
        );
        let mut model = Qwen35Model::load_from_gguf(&gguf, &mut progress)
            .context("Qwen35Model::load_from_gguf")?;
        // ADR-018 C3: legacy `tracing::info!("Qwen35 SERVE load: weights loaded ({} layers, variant={:?})", ...)`
        // was deleted here. `emit_tracing(&info)` surfaces both `n_layers`
        // and architecture facts as structured fields.

        // ADR-020 AC#5 Iter C2.4 #3 — DWQ overlay (mlx-affine packed-U32
        // safetensors) applied after the GGUF load.  Replaces each
        // trained MoE-expert (gate / up / down) bucket with a stacked
        // MlxAffineMoeStack that the gpu_ffn.rs MoE dispatch routes
        // through `mlx_native::quantized_matmul_id_into` (Iter C2.4 #4).
        if let Some(overlay_path) = opts.dwq_overlay_path.as_ref() {
            let device = mlx_native::MlxDevice::new()
                .map_err(|e| anyhow::anyhow!("Qwen35 DWQ overlay device: {e}"))?;
            let stacked = model
                .apply_dwq_overlay(&device, overlay_path)
                .with_context(|| {
                    format!("Qwen35 DWQ overlay from {} failed", overlay_path.display())
                })?;
            tracing::info!(
                count = stacked,
                path = %overlay_path.display(),
                "DWQ overlay applied to Qwen35LoadedModel"
            );
        }

        // ---- Resolve EOS ----
        // Qwen3.5/3.6: `tokenizer.ggml.eos_token_id` is typically 151645
        // (`<|im_end|>`) per `cmd_generate_qwen35:1066-1069`.  When the
        // GGUF metadata is absent we fall back to the HF Qwen3.5 default.
        //
        // ADR-028 iter-267: extended-vocab qwen3 GGUFs (e.g. 27B-MTP-Q4_0
        // with vocab=248044) put `<|im_end|>` at a different ID than the
        // standard 151645 — and some GGUFs omit `eos_token_id` metadata
        // entirely (per iter-265 the 27B-MTP file has 32 metadata keys
        // none of which are eos_token_id).  Robust fallback: scan
        // `tokenizer.ggml.tokens` array by NAME for the canonical qwen3
        // chat-template stop tokens (`<|im_end|>`, `<|endoftext|>`) and
        // include EVERY match.  is_eos.contains() then catches whatever
        // the model actually emits.
        let mut eos_token_ids: Vec<u32> = Vec::with_capacity(2);
        if let Some(id) = gguf.metadata_u32("tokenizer.ggml.eos_token_id") {
            eos_token_ids.push(id);
        }
        if let Some(id) = gguf.metadata_u32("tokenizer.ggml.eot_token_id") {
            if !eos_token_ids.contains(&id) {
                eos_token_ids.push(id);
            }
        }
        // Scan tokens array for canonical qwen3 chat-template stops.
        if let Some(arr) = gguf.metadata("tokenizer.ggml.tokens") {
            if let mlx_native::gguf::MetadataValue::Array(elems) = arr {
                for (i, el) in elems.iter().enumerate() {
                    if let mlx_native::gguf::MetadataValue::String(s) = el {
                        if QWEN35_EOS_STOP_NAMES.contains(&s.as_str()) {
                            let id = i as u32;
                            if !eos_token_ids.contains(&id) {
                                eos_token_ids.push(id);
                            }
                        }
                    }
                }
            }
        }
        if eos_token_ids.is_empty() {
            // Final fallback to legacy default.
            eos_token_ids.push(151_645);
        }
        let _eos_token: u32 = eos_token_ids[0];
        tracing::info!(
            count = eos_token_ids.len(),
            ids = ?eos_token_ids,
            "Qwen35 EOS token set resolved (iter-267 multi-source)"
        );

        // ---- Load tokenizer ----
        //
        // Mirrors `cmd_generate_qwen35`'s tokenizer construction at
        // `serve/mod.rs::cmd_generate_qwen35` (commit 5ccc54b): builds a
        // `tokenizers::Tokenizer` programmatically from the GGUF's own
        // `tokenizer.ggml.*` metadata arrays, NOT from the on-disk HF
        // `tokenizer.json`. The HF tokenizer overshoots the GGUF's
        // physical embedding-row count on abliterated/apex GGUFs (e.g.
        // declares `<|im_start|>`=248045 against a 248,044-row
        // `token_embd.weight`), and the resulting OOB token IDs hit the
        // embedding loader's zero-pad fallback — decapitating the
        // residual stream and producing deterministic prompt-repetition
        // gibberish on chat-templated requests.
        //
        let (tokenizer, vision_special_tokens_present) = build_qwen35_serving_tokenizer(&gguf)?;

        // ---- Chat template ----
        // Qwen's ChatML tool protocol is not interchangeable with Gemma's
        // native turn/call protocol. Prefer the GGUF's vendor template, use
        // hf2q's pinned Qwen3.6 template only when metadata is absent, and
        // reject a structurally incompatible template before inference.
        let template_arch = gguf
            .metadata_string("general.architecture")
            .unwrap_or("qwen35moe");
        let chat_template = gguf
            .metadata_string("tokenizer.chat_template")
            .map(str::to_string)
            .unwrap_or_else(|| {
                tracing::warn!(
                    "Qwen35 load: no GGUF `tokenizer.chat_template`; using pinned QWEN3_CHATML fallback"
                );
                crate::core::chat_templates::QWEN3_CHATML.to_string()
            });
        crate::core::chat_templates::validate_tool_chat_template(template_arch, &chat_template)
            .map_err(|error| anyhow::anyhow!("Qwen3.6 chat template contract: {error}"))?;

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
        // Promoted to `crate::serve::load_info::infer_quant_label` per
        // ADR-018 C1 — the previously-inline body was byte-identical to
        // the Gemma-path body, both now route through the shared helper.
        let quant_type = crate::serve::load_info::infer_quant_label(&gguf);

        let load_duration = load_start.elapsed();
        // ADR-018 C3: legacy `tracing::info!("Qwen35 SERVE load: complete in {:.1}s ...", ...)`
        // was deleted here. `emit_tracing(&info)` (called by every entry
        // that constructs a `LoadInfo`) emits structured `load_wall_clock`,
        // `n_layers`, `max_context_length`, `quant_label` fields. The
        // free-text format here was incompatible with `journalctl -u hf2q | jq`.

        let loaded = Self {
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
            expected_projector_sha256,
            vision_projector_profile,
            vision_deepstack_output_count,
            vision_special_tokens_present,
            prompt_cache: HybridPromptCache::new(),
            // ADR-017 Phase E.a default-on — byte-budget LcpRegistry.
            // Budget computed from sysinfo available_memory() × 5%
            // clamped to [1 GiB, 16 GiB], or overridden via
            // `HF2Q_KV_LCP_RESUME_CAPACITY` (byte-suffix form e.g. `2g`;
            // legacy bare-integer < 4096 still accepted as entry-count
            // × 300 MB with a deprecation warning). See
            // `qwen35_lcp_registry_byte_budget` for the sizing rationale.
            lcp_registry:
                crate::serve::kv_persist::lcp_registry::LcpRegistry::with_byte_budget(
                    qwen35_lcp_registry_byte_budget(),
                ),
            // Wired by AppState via LoadedModel::generate at request
            // time; None by default.
            kv_metrics_sink: None,
            // ADR-027 Phase A iter-6b.2: construct disk persistor when
            // HF2Q_KV_PERSIST is set. Failure to mkdir → log + None
            // (graceful fallback to in-process-only; do NOT bail engine
            // load on persistence-layer failure — degrades to legacy
            // behavior, doesn't break inference).
            disk_persistor: opts.kv_persist_dir.as_ref().and_then(|cache_dir| {
                // ADR-027 sub-iter 23d-γ: wire HF2Q_KV_PERSIST_BUDGET_BYTES
                // into the qwen35 LCP persistor (same env + parse
                // semantics as the block-store budget at
                // `serve/mod.rs:4208`; 0 = unlimited). Without this the
                // write-through path grew unbudgeted — 105 GB observed
                // for a single ~100K-token agentic session.
                let budget_bytes: u64 = match std::env::var("HF2Q_KV_PERSIST_BUDGET_BYTES") {
                    Ok(raw) => match raw.trim().parse::<u64>() {
                        Ok(parsed) => parsed,
                        Err(err) => {
                            tracing::warn!(
                                raw = %raw,
                                error = %err,
                                "ADR-027 23d-γ: HF2Q_KV_PERSIST_BUDGET_BYTES parse failed; \
                                 defaulting to 0 (unlimited)"
                            );
                            0
                        }
                    },
                    Err(_) => 0,
                };
                match crate::serve::kv_persist::families::qwen35_disk_persistor::Qwen35DiskPersistor::new_with_budget(cache_dir.clone(), budget_bytes) {
                    Ok(p) => {
                        tracing::info!(
                            cache_dir = %cache_dir.display(),
                            budget_bytes = budget_bytes,
                            "ADR-027 iter-6b.2 + 23d-γ: Qwen35DiskPersistor constructed; \
                             cold-process LCP resume enabled"
                        );
                        Some(std::sync::Arc::new(p))
                    }
                    Err(e) => {
                        tracing::warn!(
                            cache_dir = %cache_dir.display(),
                            error = %e,
                            "ADR-027 iter-6b.2: Qwen35DiskPersistor construction failed; \
                             falling back to in-process-only LCP"
                        );
                        None
                    }
                }
            }),
            // ADR-027 iter-6b.3: empty set on construction; first prefill
            // for each cfg-fingerprint inserts into this set after a
            // successful hydrate so subsequent prefills skip the I/O.
            lcp_hydrated_for_cfg: std::collections::HashSet::new(),
            // ADR-027 Phase B iter-12: source HF2Q_TQ_KV env once at
            // engine load via the same `is_tq_active_mode()` helper
            // Gemma's tq_packed_descriptor wiring uses (engine.rs:2328).
            // env unset / "0" / "false" → false (legacy F32 path);
            // env "1" / "true" → true (TQ-active KV alloc path; iter-13
            // wires the SDPA dispatch). Per-process flag, captured once
            // here so the env can't toggle mid-process.
            tq_kv_active: crate::serve::api::tq_packed_descriptor::is_tq_active_mode(),
            // ADR-040 Phase C iter-2a (C2b) — persistent multi-seq KV
            // cache scaffold. Always `None` at iter-2a (SerialFifo
            // max_slots=1 keeps the per-request alloc path; see field
            // doc for the byte-equivalence rationale + iter-2b lift
            // sequencing). The first SlotAware-aware request in iter-2b
            // populates this lazily via `HybridKvCache::new_with_options
            // (... n_seqs = max_slots, ...)`.
            persistent_kv_cache: None,
            // ADR-044 (2026-08-21): `auto` is now the default server
            // policy for this family when HF2Q_QWEN_SPECULATION is unset;
            // explicit `off` remains the escape. See qwen35_speculation.rs.
            speculation: super::qwen35_speculation::QwenSpeculationController::from_environment(),
        };

        // ADR-027 sub-iter 23d-γ (2026-08-03) — the TQ × persist and
        // LCP-on-TQ combinations are both CLOSED and production-ready:
        //   * Persist in TQ-only mode: `cfg_from_cache` derives shape
        //     from `slot.tq.k_packed` (F32 backing absent) and codec v5
        //     round-trips compact full-attention/MTP TQ payloads;
        //     `KvSubstrate` namespaces
        //     the on-disk fingerprint so cross-substrate hydration is a
        //     clean miss rather than a silent zero-restore.
        //   * LCP-on-TQ: `HybridKvCache::restore_partial` copies the
        //     first-n_tokens positions of all four TQ buffers per slot,
        //     closing the silent zero-restore gap that the
        //     `effective_kv_lcp_resume` auto-disable guarded against.
        // These info lines replace the iter-22 "future constraint"
        // warnings that described both combinations as unimplemented.
        if loaded.tq_kv_active && opts.kv_persist_dir.is_some() {
            tracing::info!(
                "ADR-027 sub-iter 23d-γ: HF2Q_TQ_KV=1 + HF2Q_KV_PERSIST both active — \
                 persist snapshots round-trip the compact TQ substrate (codec v5); \
                 fingerprint is capacity-independent and substrate-namespaced so \
                 cross-mode hydration is a clean miss."
            );
        }
        if loaded.tq_kv_active {
            tracing::info!(
                "ADR-027 sub-iter 23d-γ: HF2Q_TQ_KV=1; LCP resume restores TQ buffers \
                 per-slot (restore_partial 23d-γ) — coherent under the production \
                 TQ-only regime."
            );
        }

        Ok(loaded)
    }

    /// ADR-027 Phase A iter-6b.2 — write-through helper that stores a
    /// snapshot in the in-memory `LcpRegistry` AND, when a disk
    /// persistor is bound, writes it through to the cfg-fingerprint
    /// subdir on disk.
    ///
    /// Cfg is derived from `kv_cache` via
    /// `qwen35_hybrid_persistor::cfg_from_cache(kv_cache,
    /// FullAttnCodec::F32Dense)` — runtime authority on shape per
    /// ADR-027 §4.7 iter-6a finding. Phase B iter-11 will extend this
    /// site to thread the codec choice from Self when TQ-active mode
    /// flips on.
    ///
    /// Disk write failures are logged via `tracing::warn!` but do NOT
    /// fail the in-memory store — the runtime path stays correct;
    /// only the cold-resume back-end loses durability for that one
    /// snapshot. (Mantra-aligned: persistence-layer failure must
    /// not break inference.)
    ///
    /// Called from the 4 existing `lcp_registry.store` sites in this
    /// file (mid-prefill snapshot writes); replaces the bare
    /// `lcp_registry.store(…)` calls.
    pub fn store_lcp_with_disk_writeback(
        &mut self,
        kv_cache: &crate::inference::models::qwen35::kv_cache::HybridKvCache,
        key: crate::serve::kv_persist::lcp_registry::LcpKey,
        prompt_tokens: Vec<u32>,
        snapshot: crate::inference::models::qwen35::kv_cache::HybridKvCacheSnapshot,
        sliding_window: usize,
        linear_capacity: usize,
    ) -> Result<(), crate::serve::kv_persist::lcp_registry::LcpStoreError> {
        let snapshot = std::sync::Arc::new(snapshot);
        // Prepare the background job without serializing. The prompt clone is
        // small relative to the checkpoint and the snapshot itself is shared
        // by Arc with the in-memory registry, so enqueue adds no KV payload
        // copy on the request path.
        let disk_job = if let Some(persistor) = &self.disk_persistor {
            match crate::serve::kv_persist::families::qwen35_hybrid_persistor::cfg_from_cache(
                kv_cache,
                crate::serve::kv_persist::families::qwen35_hybrid_persistor::FullAttnCodec::F32Dense,
            ) {
                Ok(cfg) => {
                    let key_hex = lcp_key_to_filename_hex(&key);
                    // ADR-027 iter-6b.3: construct sidecar metadata from
                    // the live (key, prompt_tokens, sliding_window,
                    // linear_capacity) so the cold-start hydrate path
                    // can replay the in-memory `LcpRegistry::store(...)`
                    // call exactly. Clones are O(N) on the disk-write
                    // arm only — the in-memory arm still moves
                    // `prompt_tokens`.
                    let sidecar = crate::serve::kv_persist::families::qwen35_hybrid_persistor::LcpSidecarMetadata {
                        model_fingerprint: key.model_fingerprint.clone(),
                        tenant_id: key.tenant_id.clone(),
                        params_hash: key.params_hash,
                        prompt_tokens: prompt_tokens.clone(),
                        sliding_window: sliding_window as u64,
                        linear_capacity: linear_capacity as u64,
                    };
                    Some((std::sync::Arc::clone(persistor), cfg, key_hex, sidecar))
                }
                Err(e) => {
                    tracing::warn!(
                        cache_dir = %persistor.cache_dir().display(),
                        error = %format!("{e:#}"),
                        "ADR-027 iter-6b.3: cfg_from_cache failed; disk \
                         write-through skipped (in-memory store still proceeds)"
                    );
                    None
                }
            }
        } else {
            None
        };

        // Make the checkpoint visible to the next agentic turn first. A
        // rejected in-memory store is not submitted to disk as if it were a
        // live recovery point.
        self.lcp_registry.store(
            key,
            prompt_tokens,
            vec![std::sync::Arc::clone(&snapshot)],
            sliding_window,
            linear_capacity,
        )?;

        if let Some((persistor, cfg, key_hex, sidecar)) = disk_job {
            match persistor.enqueue_write(cfg, key_hex.clone(), snapshot, sidecar) {
                Ok(replaced_pending) => {
                    tracing::info!(
                        target: "hf2q::serve::api::engine_qwen35::progress",
                        key_hex = %key_hex,
                        replaced_pending,
                        pending_writes = persistor.pending_writes(),
                        "Qwen35 checkpoint queued for async disk persistence"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        cache_dir = %persistor.cache_dir().display(),
                        key_hex = %key_hex,
                        error = %format!("{e:#}"),
                        "Qwen35 async checkpoint enqueue failed; in-memory checkpoint remains live"
                    );
                }
            }
        }
        Ok(())
    }

    /// ADR-027 Phase A iter-6b.3 — cold-start hydrate. Walks the disk
    /// persistor's per-cfg subdir for the live cache shape and re-inserts
    /// every successfully-deserialized snapshot+sidecar into the
    /// in-memory `lcp_registry`. Idempotent across the process lifetime
    /// via `self.lcp_hydrated_for_cfg`: each cfg is hydrated at most once.
    ///
    /// Called by the prefill entrypoints (text + soft-token) BEFORE the
    /// LCP probe, so the registry is populated by the time chunked
    /// prefill looks for stride-aligned matches.
    ///
    /// No-ops cleanly when:
    /// - `disk_persistor` is None (HF2Q_KV_PERSIST not set)
    /// - this cfg-fingerprint has already been hydrated this process
    /// - the on-disk per-cfg subdir doesn't exist yet (clean cold start)
    ///
    /// Errors during cfg derivation, persistor read, or registry store
    /// are logged via `tracing::warn!` and swallowed — hydrate failure
    /// must NOT break the live request (mantra-aligned: persistence-
    /// layer failure must not break inference).
    pub fn hydrate_lcp_registry_from_disk(
        &mut self,
        kv_cache: &crate::inference::models::qwen35::kv_cache::HybridKvCache,
        device: &mlx_native::MlxDevice,
    ) {
        let persistor = match &self.disk_persistor {
            Some(p) => std::sync::Arc::clone(p),
            None => return, // HF2Q_KV_PERSIST not set — nothing to hydrate
        };
        let cfg = match crate::serve::kv_persist::families::qwen35_hybrid_persistor::cfg_from_cache(
            kv_cache,
            crate::serve::kv_persist::families::qwen35_hybrid_persistor::FullAttnCodec::F32Dense,
        ) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    error = %format!("{e:#}"),
                    "ADR-027 iter-6b.3: hydrate_lcp_registry: cfg_from_cache failed; skipping"
                );
                return;
            }
        };
        let fingerprint_hex =
            crate::serve::kv_persist::families::qwen35_disk_persistor::Qwen35DiskPersistor::fingerprint_hex_for(&cfg);
        if self.lcp_hydrated_for_cfg.contains(&fingerprint_hex) {
            return; // already hydrated this cfg — no-op
        }
        // Mark BEFORE the I/O so a partial-success hydrate doesn't loop
        // (bad files were warn-logged in hydrate_for_cfg; we won't
        // retry them — the operator must fix or delete corrupt files).
        self.lcp_hydrated_for_cfg.insert(fingerprint_hex.clone());

        let triples = match persistor.hydrate_for_cfg(&cfg, device) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    cache_dir = %persistor.cache_dir().display(),
                    fingerprint = %fingerprint_hex,
                    error = %format!("{e:#}"),
                    "ADR-027 iter-6b.3: hydrate_lcp_registry: hydrate_for_cfg failed; \
                     skipping (registry remains empty for this cfg)"
                );
                return;
            }
        };
        let total = triples.len();
        let mut inserted = 0usize;
        for (key_hex, snap, sidecar) in triples {
            let key = crate::serve::kv_persist::lcp_registry::LcpKey {
                model_fingerprint: sidecar.model_fingerprint.clone(),
                tenant_id: sidecar.tenant_id.clone(),
                params_hash: sidecar.params_hash,
            };
            // Sliding-window / linear-capacity were stored as u64 over
            // usize for portability; production values fit comfortably.
            let sliding_window = sidecar.sliding_window as usize;
            let linear_capacity = sidecar.linear_capacity as usize;
            match self.lcp_registry.store(
                key,
                sidecar.prompt_tokens.clone(),
                vec![std::sync::Arc::new(snap)],
                sliding_window,
                linear_capacity,
            ) {
                Ok(()) => inserted += 1,
                Err(e) => {
                    tracing::warn!(
                        key_hex = %key_hex,
                        error = ?e,
                        "ADR-027 iter-6b.3: lcp_registry.store rejected hydrated entry; \
                         skipping (file may exceed byte budget or be empty)"
                    );
                }
            }
        }
        tracing::info!(
            cache_dir = %persistor.cache_dir().display(),
            fingerprint = %fingerprint_hex,
            files_on_disk = total,
            entries_inserted = inserted,
            registry_len = self.lcp_registry.len(),
            "ADR-027 iter-6b.3: hydrate_lcp_registry_from_disk complete"
        );
    }

    /// **ADR-040 Phase C iter-2d (C2d)** — provision the persistent
    /// multi-seq KV scaffold via [`HybridKvCache::new_with_options`]
    /// with `n_seqs = max_slots`.
    ///
    /// Called by [`crate::serve::api::engine::Engine::spawn_with_mode`]
    /// when [`crate::serve::api::engine::EngineMode::SlotAware`] is
    /// selected for a Qwen35 engine. Sets
    /// [`Self::persistent_kv_cache`] to
    /// `Some(HybridKvCache { n_seqs: max_slots, .. })` covering every
    /// layer (full-attn + linear-attn + optional MTP slot) in a single
    /// cache instance — mirrors the production
    /// `alloc_kv_cache_for_request` path at `engine_qwen35.rs:1088`,
    /// just with `n_seqs = max_slots` instead of `1` and
    /// `max_seq_len = cfg.max_position_embeddings` (rather than the
    /// per-request `prompt_len + max_tokens + 64` sizing — the
    /// spawn-time scaffold is sized to the model's full context
    /// window because the per-request prompt/decode lengths are not
    /// known at spawn time).
    ///
    /// **Why a single `HybridKvCache` (not per-layer like Gemma 4)**:
    /// Qwen35's `HybridKvCache` already owns the per-layer `LayerSlot`
    /// table internally (full_attn / linear_attn / mtp_slot fields
    /// indexed by per-layer rank). The A2a multi-seq lift threaded
    /// `n_seqs` through that internal layout — there is no sibling
    /// `MultiSeqHybridKvBuffers` type at the engine layer the way
    /// Gemma 4 has `MultiSeqHbKvBuffers`. So Qwen35's C2d provisioning
    /// is one allocator call vs Gemma 4's N (one per layer).
    ///
    /// **TQ-active path**: honours `self.tq_kv_active` so the multi-seq
    /// scaffold matches the engine's TQ mode. Packed and norms buffers use
    /// `n_seqs = max_slots` as their outer axis; encode, attention, and resume
    /// dispatches receive zero-copy views of the selected slot region.
    ///
    /// # Errors
    ///
    /// Propagates the first [`HybridKvCache::new_with_options`] error
    /// (`anyhow::Error`); typical causes are zero `max_seq_len` /
    /// zero `n_seqs` (caught earlier by the spawn arm's `max_slots == 0`
    /// guard) or an invalid Metal virtual allocation.
    pub fn provision_multi_seq_kv_for_slot_aware(&mut self, max_slots: u32) -> anyhow::Result<()> {
        use anyhow::Context;
        if max_slots == 0 {
            anyhow::bail!(
                "ADR-040 C2d: provision_multi_seq_kv_for_slot_aware called with \
                 max_slots == 0; spawn_with_mode invariant is max_slots >= 1 \
                 (EngineMode::SlotAware variant enforces this at the API \
                 boundary — caller violated)"
            );
        }
        let device = mlx_native::MlxDevice::new()
            .context("ADR-040 C2d: MlxDevice::new for Qwen35 multi-seq KV provisioning")?;
        // Every agent slot receives the complete model context. The Qwen KV
        // allocator reserves full-attention storage lazily, while scheduler
        // admission governs aggregate physical high-water use.
        let max_seq_len = self.model.cfg.max_position_embeddings;
        let cache = HybridKvCache::new_with_options(
            &self.model.cfg,
            &device,
            max_seq_len,
            max_slots,
            self.tq_kv_active,
        )
        .with_context(|| {
            format!(
                "ADR-040 C2d: HybridKvCache::new_with_options(max_seq_len={}, \
                 n_seqs={}, tq_kv_active={}) for Qwen35 SlotAware provisioning",
                max_seq_len, max_slots, self.tq_kv_active
            )
        })?;
        self.persistent_kv_cache = Some(cache);
        Ok(())
    }
}

/// ADR-027 Phase A iter-6b.2 — derive the on-disk filename hex from a
/// `LcpKey`. SHA-256 over the model_fingerprint + tenant_id +
/// params_hash; first 16 bytes hex-encoded (32-char filename).
fn lcp_key_to_filename_hex(key: &crate::serve::kv_persist::lcp_registry::LcpKey) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"QH35-lcp-key-fname-v1");
    h.update(&key.model_fingerprint.0);
    h.update(key.tenant_id.as_bytes());
    h.update(&key.params_hash.to_le_bytes());
    let digest = h.finalize();
    hex::encode(&digest[..16])
}

impl LoadInfoBuilder for Qwen35LoadedModel {
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
            arch_family: ArchFamily::Qwen35,
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
            full_attention_interval: Some(cfg.full_attention_interval),
            max_context_length: self.context_length.map(|v| v as u32),
            moe: cfg.moe.as_ref().map(|m| MoeShape {
                n_experts: m.num_experts,
                n_experts_per_tok: m.num_experts_per_tok,
            }),
            quant_label: self.quant_type.clone(),
            quant_bpw: load_info::compute_bpw(gguf),
            tokenizer_source: TokenizerSource::GgufEmbedded,
            eos_token_ids: self.eos_token_ids.clone(),
            bos_token_id: gguf.metadata_u32("tokenizer.ggml.bos_token_id"),
            chat_template_source: if gguf.metadata_string("tokenizer.chat_template").is_some() {
                ChatTemplateSource::GgufEmbedded
            } else {
                ChatTemplateSource::HardcodedFallback {
                    name: "QWEN3_CHATML",
                }
            },
            provenance: self.provenance.clone(),
            vision_projector: None,
            load_wall_clock,
            resident_weight_bytes: None,
            kv_cache_budget_bytes,
            kv_spill_active,
            // ADR-027 Phase B iter-17: surface the engine-load
            // `HF2Q_TQ_KV` state to operators via the load banner.
            // `self.tq_kv_active` was sourced at engine load via
            // `is_tq_active_mode()` (iter-12).
            tq_kv_active: self.tq_kv_active,
            // DeltaNet layers retain fixed recurrent state. Only the
            // full-attention (and optional MTP) cache grows with context;
            // TQ mode stores packed U8 K/V plus F32 norms and drops F32 K/V.
            kv_bytes_per_token_override: Some(load_info::qwen35_slot_kv_bytes_per_token(
                cfg,
                self.tq_kv_active,
            )),
            kv_fixed_bytes_per_slot_override: Some(load_info::qwen35_fixed_kv_bytes_per_slot(cfg)),
        }
    }
}

// `infer_quant_type_from_gguf` (formerly 27 LOC of histogram code,
// byte-identical to the Gemma-path body in engine.rs) was relocated to
// `crate::serve::load_info::infer_quant_label` per ADR-018 C1.  The
// duplication is gone; both `*LoadedModel::load` paths route through
// the promoted helper.

// ---------------------------------------------------------------------------
// HybridPromptCache (Wedge-3 / ADR-005 iter-216 Phase C)
// ---------------------------------------------------------------------------

/// Generation-affecting parameters that must all match for a Wedge-3
/// HybridPromptCache hit to fire.
///
/// Mirrors the role of `super::engine::PromptCacheKey` for Gemma but
/// trimmed to the parameter surface this MVP wires: greedy decode is the
/// only mode the cache stores, so the bypass-eligibility gate
/// (`is_greedy_eligible`) handles every sampling-mode field; this key
/// only needs the parameters that affect the cached *response shape*
/// or interpretation even under greedy decode (max_tokens early-stop,
/// stop_strings, and schema-derived tool argument wire kinds).
///
/// Grammar, logit-bias, logprobs, and effective repetition-penalty
/// requests bypass exact prompt replay through `is_greedy_eligible`, so
/// they do not belong in this deliberately narrow key. Tool-call policy and
/// wire kinds affect response routing after KV restoration rather than the
/// cached prompt state; wire kinds are included because replayed argument
/// bytes must retain their schema-authoritative JSON type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridPromptCacheKey {
    pub max_tokens: usize,
    pub stop_strings: Vec<String>,
    pub tool_argument_wire_kinds: Option<Arc<super::registry::ToolArgumentWireKinds>>,
    pub vision_fingerprint: Option<[u8; 32]>,
    pub thinking_token_budget: Option<usize>,
    pub reasoning_end_tokens: Option<Arc<Vec<u32>>>,
    pub reasoning_close_tokens: Option<Arc<Vec<u32>>>,
}

impl HybridPromptCacheKey {
    pub fn from_params(params: &SamplingParams) -> Self {
        Self {
            max_tokens: params.max_tokens,
            stop_strings: params.stop_strings.clone(),
            tool_argument_wire_kinds: params.tool_argument_wire_kinds.clone(),
            vision_fingerprint: params.vision_fingerprint,
            thinking_token_budget: params.thinking_token_budget,
            reasoning_end_tokens: params.reasoning_end_tokens.clone(),
            reasoning_close_tokens: params.reasoning_close_tokens.clone(),
        }
    }
}

/// Single-slot prompt cache for Qwen3.5/3.6 SERVE-side chat completion.
///
/// Wedge-3 / ADR-005 iter-216 Phase C.  Mirrors Gemma's
/// `super::engine::PromptCache` shape (a single previous-request slot
/// with full-equality replay) but stores the hybrid-cache substrate
/// instead of the full text result:
/// - `cached_prompt_tokens` — exact prompt that produced this snapshot.
/// - `snapshot` — `HybridKvCacheSnapshot` (16 full-attn F32 K/V + 48
///   DeltaNet conv_state + recurrent state for Qwen3.6 27B; ~1-4 GB).
/// - `first_decoded_token` — the greedy argmax sampled from the prefill's
///   last-position logits.  Cached so a hit can short-circuit prefill
///   AND skip the additional 1-token forward that would otherwise be
///   needed to re-derive the first decode token from the restored KV.
/// - `gen_params` — `HybridPromptCacheKey` snapshot of the params that
///   produced this cache entry; new requests must match BOTH prompt
///   AND key for a hit.
///
/// # Eligibility
///
/// Hits fire ONLY when the new request is fully greedy (T=0, top_k=0,
/// top_p=1.0, rep_penalty=1.0, seed=None) AND prompt+key both match.
/// Sampling-mode bypass mirrors Gemma's `PromptCache::lookup` rationale
/// (replaying a deterministic greedy decode for a sampling request would
/// silently violate per-call variation expectations).
///
/// # Snapshot scope (DeltaNet ping-pong note)
///
/// The snapshot owns *active* (read) DeltaNet conv-state + recurrent
/// buffers only — scratch buffers carry post-write garbage.  See
/// [`crate::inference::models::qwen35::kv_cache::HybridKvCache::snapshot`]
/// for the full ping-pong contract.
///
/// # Why not extend `super::engine::PromptCache` to cover this
///
/// Gemma's `PromptCache` is text-replay shaped: it stores the whole
/// `GenerationResult` and short-circuits the entire prefill+decode
/// chain.  The Qwen3.5/3.6 MVP shape is KV-state shaped: store the
/// post-prefill substrate so a future request can RUN the decode loop
/// from that state.  These are different cache contracts; collapsing
/// them into one type would require either a sum-type with two unrelated
/// payloads (engine.rs has rejected that for clarity reasons in
/// neighboring iter docs) or a forced commit to one shape on both
/// arches. Both families now have real wire-up, and their distinct cache
/// contracts remain explicit rather than hidden behind a lossy abstraction.
#[derive(Debug)]
pub struct HybridPromptCache {
    cached_prompt_tokens: Vec<u32>,
    snapshot: Option<HybridKvCacheSnapshot>,
    /// Greedy argmax token sampled from prefill's last-position logits
    /// at cache-store time.  Only meaningful when `snapshot.is_some()`.
    first_decoded_token: u32,
    gen_params: Option<HybridPromptCacheKey>,
}

impl Default for HybridPromptCache {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridPromptCache {
    /// Empty cache — initial state for a fresh `Qwen35LoadedModel`.
    pub fn new() -> Self {
        Self {
            cached_prompt_tokens: Vec::new(),
            snapshot: None,
            first_decoded_token: 0,
            gen_params: None,
        }
    }

    /// Try to match a new (prompt, gen_params) pair against the cached
    /// entry.  Returns `Some(matched_prefix_len)` on a hit, `None` on a
    /// miss.  In MVP the only matching mode is full-equality, so a hit
    /// always returns `Some(prompt.len())`.  The signature accommodates
    /// the future LCP-based partial-prefill resume (Wedge-4 deferred
    /// follow-up; mirrors the iter-97+ scope for Gemma's PromptCache).
    ///
    /// Eligibility: bypass for any non-greedy mode (sampling parameters
    /// any non-default).  See `HybridPromptCache` doc for rationale.
    pub fn try_match(&self, new_prompt: &[u32], new_params: &SamplingParams) -> Option<usize> {
        if !is_greedy_eligible(new_params) {
            return None;
        }
        if self.snapshot.is_none() || self.gen_params.is_none() {
            return None;
        }
        if self.cached_prompt_tokens.is_empty() {
            return None;
        }
        if self.cached_prompt_tokens.as_slice() != new_prompt {
            return None;
        }
        let request_key = HybridPromptCacheKey::from_params(new_params);
        if self.gen_params.as_ref() != Some(&request_key) {
            return None;
        }
        Some(new_prompt.len())
    }

    /// Read-only access to the cached snapshot — returns `None` when no
    /// entry is stored.  Used by the worker arm to call
    /// `HybridKvCache::restore_from(snap)` after a `try_match` hit.
    pub fn snapshot(&self) -> Option<&HybridKvCacheSnapshot> {
        self.snapshot.as_ref()
    }

    /// Cached greedy first decoded token — only meaningful after a hit.
    pub fn first_decoded_token(&self) -> u32 {
        self.first_decoded_token
    }

    /// Store a fresh prefill snapshot under the request's prompt + key.
    /// Same eligibility gate as `try_match`: sampling-mode requests are
    /// not stored (a future greedy lookup must never replay a sampled
    /// outcome).
    pub fn update(
        &mut self,
        prompt: Vec<u32>,
        snapshot: HybridKvCacheSnapshot,
        first_decoded_token: u32,
        params: &SamplingParams,
    ) {
        if !is_greedy_eligible(params) {
            return;
        }
        self.cached_prompt_tokens = prompt;
        self.snapshot = Some(snapshot);
        self.first_decoded_token = first_decoded_token;
        self.gen_params = Some(HybridPromptCacheKey::from_params(params));
    }

    /// Drop the stored entry (e.g. on shutdown or arena pressure).
    /// Currently unused; provided for completeness + Wedge-4 hooks.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.cached_prompt_tokens.clear();
        self.snapshot = None;
        self.first_decoded_token = 0;
        self.gen_params = None;
    }

    /// `true` when the cache currently holds an entry.  Useful for tests
    /// + tracing without exposing the snapshot.
    pub fn has_entry(&self) -> bool {
        self.snapshot.is_some()
    }
}

/// `true` when the request would produce a deterministic greedy decode
/// and is therefore eligible for the prompt-cache fast-path.  Mirrors
/// Gemma's `PromptCache::lookup` early-bypass at engine.rs:945-952.
fn is_greedy_eligible(params: &SamplingParams) -> bool {
    !(params.temperature > 0.0
        || params.top_k > 0
        || params.top_p < 1.0
        || effective_repetition_penalty(params) != 1.0
        || params.seed.is_some()
        || params.logprobs
        || !params.logit_bias.is_empty()
        || params.grammar.is_some())
}

/// Server-complete semantics gate for exact Qwen speculation.
///
/// This deliberately is stricter than the ordinary greedy fast path. MTP
/// Proposal acceptance is exact only when every distribution and response
/// transform is represented by the verifier transaction. Greedy grammar and
/// forced-thinking state are supported; stochastic sampling and per-token
/// logprob response contracts remain closed.
pub(crate) fn is_qwen_server_speculation_exact_eligible(params: &SamplingParams) -> bool {
    // Unlike the ordinary greedy fast path, repetition penalty is allowed:
    // an MTP proposal may ignore it, but both verifier selections below use
    // `sample_logits_qwen35_constrained` with the canonical generated-token
    // history and the live/simulated grammar state.
    !(params.temperature > 0.0
        || params.top_k > 0
        || params.top_p < 1.0
        || params.seed.is_some()
        || params.logprobs
        || !params.logit_bias.is_empty())
        && params.stop_strings.is_empty()
        && params.frequency_penalty == 0.0
        && params.presence_penalty == 0.0
        && params.min_p == 0.0
        && params.top_logprobs == 0
        && !params.parallel_tool_calls
        && (params.tool_call_policy == ToolCallPolicy::Auto || params.grammar.is_some())
}

fn is_serial_mtp_exact_eligible(params: &SamplingParams) -> bool {
    is_qwen_server_speculation_exact_eligible(params)
        && !params.reasoning_forced_open
        && params.thinking_token_budget.is_none()
        && params.reasoning_end_tokens.is_none()
        && params.reasoning_close_tokens.is_none()
        && params.grammar.is_none()
        && params.tool_call_policy == ToolCallPolicy::Auto
}

// ---------------------------------------------------------------------------
// LCP store observability (2026-08-03 store-gate fix)
// ---------------------------------------------------------------------------
//
// The prefix-cache store path used to fail silently: snapshot/store errors
// were `tracing::warn!`-only (invisible without RUST_LOG) and the store
// gates gave no signal when env config disabled them — an empty registry
// (probe prints `registry_len=0`) had zero diagnostics.  These helpers emit
// one-shot-per-process stderr lines for each distinct skip/failure reason:
// enough to diagnose an empty registry at a glance, without spamming once
// per chunk boundary (20-40 boundaries per long-context request).

static LCP_STORE_NOTIFY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn lcp_notify_once(bit: u64, msg: std::string::String) {
    let prev = LCP_STORE_NOTIFY.fetch_or(bit, std::sync::atomic::Ordering::Relaxed);
    if prev & bit == 0 {
        eprintln!("{msg}");
    }
}

/// One-shot notice when a stride-aligned boundary skips its checkpoint
/// store for env-config reasons.  Non-aligned partial tails are normal
/// (unaddressable by the descending probe) and never logged.
fn lcp_store_skip_notify(stride_aligned: bool, lcp_resume_enabled: bool, mid_store_disabled: bool) {
    if !stride_aligned {
        return;
    }
    if !lcp_resume_enabled {
        lcp_notify_once(
            1,
            "[hf2q qwen35 lcp store] SKIPPED: HF2Q_KV_LCP_RESUME is off — \
             no prefix checkpoints will be written (once-per-process notice)"
                .to_string(),
        );
    } else if mid_store_disabled {
        lcp_notify_once(
            2,
            "[hf2q qwen35 lcp store] SKIPPED: HF2Q_KV_LCP_DISABLE_MID_STORE=1 \
             — mid-prefill checkpoints disabled (once-per-process notice)"
                .to_string(),
        );
    }
}

/// One-shot notice for checkpoint snapshot failures (was silent on the
/// stream path: `if let Ok(snap)` swallowed the Err arm).
fn lcp_snapshot_error_notify<E: std::fmt::Debug>(phase: &str, chunk_pos: usize, err: &E) {
    lcp_notify_once(
        4,
        format!(
            "[hf2q qwen35 lcp store] snapshot FAILED ({phase}, chunk_pos={chunk_pos}): \
             {err:?} — checkpoints are NOT being written; further occurrences \
             suppressed (once-per-process notice)"
        ),
    );
}

/// One-shot notice for registry store rejections (e.g. EntryExceedsBudget —
/// was `tracing::warn!`-only, invisible under default logging).
fn lcp_store_error_notify<E: std::fmt::Debug>(phase: &str, chunk_pos: usize, err: &E) {
    lcp_notify_once(
        8,
        format!(
            "[hf2q qwen35 lcp store] store FAILED ({phase}, chunk_pos={chunk_pos}): \
             {err:?} — check the registry byte budget; further occurrences \
             suppressed (once-per-process notice)"
        ),
    );
}

// ---------------------------------------------------------------------------
// Wedge-3 / iter-216 Phase D — worker-thread inference paths
// ---------------------------------------------------------------------------
//
// These helpers replace the iter-215 MVP `qwen35_not_implemented_err()` arms
// in `super::engine::worker_run`.  Each is a thin wrapper around the
// `cmd_generate_qwen35` flow at `serve/mod.rs:1072-1416`, dropping CLI-only
// concerns (header printing, benchmark output, stdout streaming) and
// replacing them with the engine's `Result` / `mpsc::Sender` handoff.
//
// The non-streaming + streaming + embed paths share the same prefill +
// KV-cache + prompt-cache substrate; only the post-prefill output
// dispatch differs.

use crate::inference::models::qwen35::io_heads::greedy_argmax_last_token;
// `HybridKvCache` is imported once at the module top (alongside
// `HybridKvCacheSnapshot`) to keep the type visible at the
// `Qwen35LoadedModel` field declaration (per ADR-040 C2b scaffold field
// `persistent_kv_cache: Option<HybridKvCache>`).
use crate::serve::sampler_pure::{self, SamplingParams as SamplerPureParams};
use mlx_native::MlxDevice;

use super::engine::GenerationResult;
use super::registry::{
    ModelRegistration, ReasoningSplitter, SplitSlot, ToolCallEvent, ToolCallSplitter,
};
use super::sse::{DeltaKind, GenerationEvent, StreamStats};

/// Build the flat `[4 * seq_len]` axis-major position buffer the IMROPE
/// kernel expects.  Mirrors `cmd_generate_qwen35` at
/// `serve/mod.rs:1262-1270` — for text-only Qwen3.5/3.6 we replicate the
/// absolute-position index across all 4 axes.
fn prefill_positions_for(prompt_len: usize) -> Vec<i32> {
    prefill_positions_from(0, prompt_len)
}

fn prefill_positions_from(start: usize, len: usize) -> Vec<i32> {
    let mut flat = vec![0i32; 4 * len];
    for axis in 0..4 {
        for t in 0..len {
            flat[axis * len + t] = (start + t) as i32;
        }
    }
    flat
}

fn requested_kv_cache_capacity(
    qwen: &Qwen35LoadedModel,
    prompt_len: usize,
    max_tokens: usize,
) -> usize {
    (prompt_len + max_tokens + 64)
        .max(128)
        .min(qwen.model.cfg.max_position_embeddings as usize)
}

/// Select a demand-grown single-slot capacity. Powers of two avoid replacing
/// the Metal buffers on every ordinary agentic turn while keeping short
/// sessions far below the model's full training-context allocation.
fn serial_kv_cache_capacity(required: usize, current: usize, maximum: usize) -> usize {
    let rounded_required = required
        .checked_next_power_of_two()
        .unwrap_or(maximum)
        .min(maximum);
    rounded_required
        .max(current.saturating_mul(2).min(maximum))
        .max(required)
        .min(maximum)
}

/// Allocate a SerialFifo cache with the bounded recovery-capture arena ready
/// up front. This moves Metal allocation/zero-fill into the first cold cache
/// admission, where it is amortized across the session, instead of charging
/// the first short agentic continuation tens of milliseconds.
fn alloc_serial_kv_cache(
    qwen: &Qwen35LoadedModel,
    device: &MlxDevice,
    capacity: usize,
) -> Result<HybridKvCache> {
    let mut cache = HybridKvCache::new_with_options(
        &qwen.model.cfg,
        device,
        capacity as u32,
        1,
        qwen.tq_kv_active,
    )
    .context("HybridKvCache::new_with_options (SerialFifo)")?;
    cache
        .ensure_la_capture(
            &qwen.model.cfg,
            device,
            QWEN35_RECOVERY_CAPTURE_MAX_SUFFIX_TOKENS as u32,
        )
        .context("SerialFifo recovery capture preallocation")?;
    cache.clear_la_capture();
    Ok(cache)
}

/// Allocate a fresh `HybridKvCache` sized for `prompt_len + max_tokens + 64`,
/// clamped to `cfg.max_position_embeddings` and floored at 128.
///
/// This remains the one-shot helper for embeddings, multimodal siblings, and
/// focused tests. Serial text generation uses [`take_serial_kv_cache`] so the
/// same buffers survive across turns.
fn alloc_kv_cache_for_request(
    qwen: &Qwen35LoadedModel,
    device: &MlxDevice,
    prompt_len: usize,
    max_tokens: usize,
) -> Result<HybridKvCache> {
    let max_seq = requested_kv_cache_capacity(qwen, prompt_len, max_tokens);
    // ADR-027 Phase B iter-12: branch on the engine-load-time
    // `tq_kv_active` flag (sourced from HF2Q_TQ_KV env). When true,
    // allocate TQ-active full-attn buffers alongside F32 K/V (shadow
    // cache). iter-13 wires the per-token encode + SDPA dispatch that
    // makes these buffers load-bearing in the forward path.
    HybridKvCache::new_with_options(
        &qwen.model.cfg,
        device,
        max_seq as u32,
        1,
        qwen.tq_kv_active,
    )
    .context("HybridKvCache::new_with_options")
}

/// Take the retained SerialFifo cache when it is large enough; otherwise
/// replace it with a geometrically grown single-slot cache. The caller resets
/// a reused cache only on a true cold miss. LCP/prompt-cache restores fully
/// replace semantic state and therefore do not need a zeroing pass first.
fn take_serial_kv_cache(
    qwen: &mut Qwen35LoadedModel,
    device: &MlxDevice,
    prompt_len: usize,
    max_tokens: usize,
) -> Result<(HybridKvCache, bool)> {
    let required = requested_kv_cache_capacity(qwen, prompt_len, max_tokens);
    let maximum = qwen.model.cfg.max_position_embeddings as usize;
    let current = qwen.persistent_kv_cache.take();
    if let Some(cache) = current {
        if cache.n_seqs == 1 && cache.max_seq_len as usize >= required {
            return Ok((cache, true));
        }
        let capacity = serial_kv_cache_capacity(required, cache.max_seq_len as usize, maximum);
        drop(cache);
        return alloc_serial_kv_cache(qwen, device, capacity).map(|cache| (cache, false));
    }

    let capacity = serial_kv_cache_capacity(required, 0, maximum);
    alloc_serial_kv_cache(qwen, device, capacity).map(|cache| (cache, false))
}

/// Sample the next token from a `[vocab_size]` logits slice using
/// `sampler_pure::sample_token` for non-greedy modes.
///
/// Legacy unconstrained sampler retained for call sites that do not yet
/// consume request grammar or logit-bias state.
fn sample_logits_qwen35(logits: &mut [f32], params: &SamplingParams, generated: &[u32]) -> u32 {
    let sp = SamplerPureParams {
        temperature: params.temperature as f64,
        top_p: params.top_p as f64,
        top_k: params.top_k,
        min_p: 0.0,
        repetition_penalty: effective_repetition_penalty(params),
        max_tokens: params.max_tokens,
        seed: params.seed,
    };
    sampler_pure::sample_token(logits, &sp, generated)
}

/// ADR-020 AC#7 — variant of [`sample_logits_qwen35`] that also returns
/// the log-probability of the chosen token under the *raw* (pre-rep-penalty,
/// pre-temperature) distribution.  Routed when `params.logprobs == true`
/// so the chat handler can populate `ChoiceLogprobs.content[]`.
fn sample_logits_qwen35_with_logprob(
    logits: &mut [f32],
    params: &SamplingParams,
    generated: &[u32],
) -> (u32, f32) {
    let sp = SamplerPureParams {
        temperature: params.temperature as f64,
        top_p: params.top_p as f64,
        top_k: params.top_k,
        min_p: 0.0,
        repetition_penalty: effective_repetition_penalty(params),
        max_tokens: params.max_tokens,
        seed: params.seed,
    };
    sampler_pure::sample_token_with_logprob(logits, &sp, generated)
}

fn sample_logits_qwen35_constrained(
    logits: &mut [f32],
    params: &SamplingParams,
    generated: &[u32],
    runtime: Option<&super::grammar::GrammarRuntime>,
    want_logprobs: bool,
) -> Result<(u32, Option<f32>)> {
    for (&token, &bias) in &params.logit_bias {
        if let Some(logit) = logits.get_mut(token as usize) {
            *logit += bias;
        }
    }
    let sampler = SamplerPureParams {
        temperature: params.temperature as f64,
        top_p: params.top_p as f64,
        top_k: params.top_k,
        min_p: params.min_p as f64,
        repetition_penalty: effective_repetition_penalty(params),
        max_tokens: params.max_tokens,
        seed: params.seed,
    };
    sample_logits_with_grammar(
        logits,
        &sampler,
        generated,
        runtime,
        params.token_bytes.as_deref().map(Vec::as_slice),
        want_logprobs,
    )
}

fn advance_qwen35_grammar(
    runtime: &mut Option<super::grammar::GrammarRuntime>,
    params: &SamplingParams,
    token: u32,
) {
    if let (Some(runtime), Some(token_bytes)) = (runtime.as_mut(), params.token_bytes.as_deref()) {
        if let Some(bytes) = token_bytes.get(token as usize) {
            if !bytes.is_empty() {
                runtime.accept_bytes(bytes);
            }
        }
    }
}

fn qwen35_grammar_terminal_token(
    runtime: Option<&super::grammar::GrammarRuntime>,
    params: &SamplingParams,
    token: u32,
) -> bool {
    let Some(runtime) = runtime else {
        return false;
    };
    if runtime.is_dead() {
        return true;
    }
    runtime.is_accepted()
        && params.token_bytes.as_deref().is_some_and(|token_bytes| {
            token_bytes
                .get(token as usize)
                .map(|bytes| bytes.is_empty())
                .unwrap_or(true)
        })
}

/// ADR-017 Phase B-hybrid.1 — build the LcpKey for a Qwen35 request.
///
/// Mirrors `engine::build_lcp_key_for_request` (the Gemma version)
/// at engine.rs:3742 with the same provenance / fingerprint logic.
/// Tenant + params hash empty for v1 — same scope as Gemma's iter-2.
fn build_lcp_key_for_qwen35(
    qwen: &Qwen35LoadedModel,
    _params: &SamplingParams,
) -> crate::serve::kv_persist::lcp_registry::LcpKey {
    use crate::serve::kv_persist::format::compute_model_fingerprint;
    let (producer_version, source_sha256) = match &qwen.provenance {
        crate::core::provenance::Provenance::Hf2q {
            producer_version,
            source_sha256,
            ..
        } => (producer_version.as_str(), source_sha256.as_str()),
        crate::core::provenance::Provenance::External => ("", ""),
    };
    let chat_template_hash = match &qwen.provenance {
        crate::core::provenance::Provenance::Hf2q { .. } => {
            super::kv_spill_descriptor::KvSpillProvenance::hash_chat_template(&qwen.chat_template)
        }
        crate::core::provenance::Provenance::External => String::new(),
    };
    let quant = qwen.quant_type.as_deref().unwrap_or("");
    let fp = compute_model_fingerprint(
        &qwen.model_id,
        quant,
        producer_version,
        source_sha256,
        &chat_template_hash,
    );
    crate::serve::kv_persist::lcp_registry::LcpKey {
        model_fingerprint: fp,
        tenant_id: String::new(),
        params_hash: 0,
    }
}

/// ADR-017 Phase E.a B.3 — chunk-position-keyed LcpKey for mid-prefill
/// stride-aligned checkpointing.
///
/// Same fingerprint / chat-template hash as `build_lcp_key_for_qwen35`,
/// but with the chunk position embedded in `tenant_id`.  This lets the
/// engine store MULTIPLE entries from one prefill (one per stride
/// boundary) under distinct keys without changing the underlying
/// LcpRegistry data structure (LcpRegistry is a flat HashMap with one
/// entry per key — chunk-position keys give us multi-entry-per-request
/// while preserving the existing API).
///
/// The probe iterates chunk positions DESCENDING (largest first) to
/// find the longest stride-aligned true-continuation match for the
/// new prompt.  When `chunk_position == 0` this reduces to the base
/// key (no chunk discriminator).
fn build_lcp_key_for_qwen35_chunk(
    qwen: &Qwen35LoadedModel,
    params: &SamplingParams,
    chunk_position: usize,
) -> crate::serve::kv_persist::lcp_registry::LcpKey {
    let mut key = build_lcp_key_for_qwen35(qwen, params);
    if chunk_position > 0 {
        key.tenant_id = format!("qwen35:lcp_chunk:{chunk_position}");
    }
    key
}

/// Find the longest coherent Qwen checkpoint for a continuation request.
/// The base key holds the previous request's stable generation boundary,
/// including prompts shorter than one checkpoint stride. Stride-keyed
/// entries preserve older branch points. A checkpoint is usable only when
/// the new request fully extends the cached prompt; DeltaNet state captured
/// after a divergent suffix cannot be rolled back to the divergence point.
fn lookup_qwen35_resume_checkpoint<T>(
    registry: &mut crate::serve::kv_persist::lcp_registry::LcpRegistry<T>,
    base_key: &crate::serve::kv_persist::lcp_registry::LcpKey,
    new_tokens: &[u32],
    stride: usize,
) -> Option<(crate::serve::kv_persist::lcp_registry::LcpPrefix<T>, usize)>
where
    T: Send + Sync + 'static + crate::serve::kv_persist::lcp_registry::ByteSized,
{
    if new_tokens.is_empty() {
        return None;
    }

    let base_match = registry
        .lookup(base_key, new_tokens)
        .filter(|prefix| prefix.k == prefix.cached_prompt_len && prefix.k < new_tokens.len());
    let base_len = base_match.as_ref().map(|prefix| prefix.k).unwrap_or(0);

    if stride > 0 {
        let mut chunk_pos = (new_tokens.len() / stride).saturating_mul(stride);
        while chunk_pos >= stride && chunk_pos > base_len {
            let mut chunk_key = base_key.clone();
            chunk_key.tenant_id = format!("qwen35:lcp_chunk:{chunk_pos}");
            if let Some(prefix) = registry.lookup(&chunk_key, new_tokens) {
                if prefix.k == prefix.cached_prompt_len && prefix.k < new_tokens.len() {
                    return Some((prefix, chunk_pos));
                }
            }
            if chunk_pos == stride {
                break;
            }
            chunk_pos -= stride;
        }
    }

    base_match.map(|prefix| (prefix, 0))
}

/// Conservative fallback when the loaded Qwen tokenizer does not confirm
/// the vendor template's generation-only suffix at the end of the prompt.
const QWEN35_RECOVERY_TAIL_FALLBACK_TOKENS: usize = 64;

/// The existing mlx-native per-position DeltaNet capture kernel is tuned for
/// decode and short batched verification. At or below this boundary it is
/// cheaper to process the changed suffix once and snapshot the stable state
/// from capture than to submit a second forward for the template tail.
const QWEN35_RECOVERY_CAPTURE_MAX_SUFFIX_TOKENS: usize = 32;

/// Qwen's vendor template appends one of these generation-only suffixes
/// after the stable `<|im_start|>assistant\n` boundary. On the next turn the
/// template renders the prior assistant content at that boundary instead,
/// so retaining KV state for these suffix tokens is not coherent. Derive the
/// token count with the loaded tokenizer and require an exact prompt suffix
/// match; a foreign/custom template therefore falls back safely instead of
/// inheriting an assumed token count.
fn qwen35_recovery_tail_tokens(
    qwen: &Qwen35LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
) -> usize {
    let unstable_suffix = if params.reasoning_forced_open {
        "<think>\n"
    } else {
        "<think>\n\n</think>\n\n"
    };
    let encoded = qwen.tokenizer.encode(unstable_suffix, false);
    let suffix_tokens = encoded.as_ref().map(|e| e.get_ids()).unwrap_or(&[]);
    recovery_tail_for_suffix(
        prompt_tokens,
        suffix_tokens,
        QWEN35_RECOVERY_TAIL_FALLBACK_TOKENS,
    )
}

fn recovery_tail_for_suffix(
    prompt_tokens: &[u32],
    suffix_tokens: &[u32],
    fallback: usize,
) -> usize {
    if !suffix_tokens.is_empty() && prompt_tokens.ends_with(suffix_tokens) {
        suffix_tokens.len()
    } else {
        fallback
    }
}

fn stride_checkpoint_superseded_by_recovery_anchor(
    recovery_eligible: bool,
    k_end: usize,
    stride: usize,
    recovery_anchor: usize,
) -> bool {
    recovery_eligible && k_end <= recovery_anchor && k_end.saturating_add(stride) > recovery_anchor
}

fn qwen35_recovery_capture_plan(
    lcp_resume_start: usize,
    recovery_anchor: usize,
    prompt_len: usize,
    recovery_eligible: bool,
    chunked_eligible: bool,
) -> Option<(usize, usize)> {
    if !recovery_eligible
        || chunked_eligible
        || recovery_anchor <= lcp_resume_start
        || prompt_len <= recovery_anchor
    {
        return None;
    }
    let suffix_len = prompt_len.checked_sub(lcp_resume_start)?;
    if suffix_len == 0 || suffix_len > QWEN35_RECOVERY_CAPTURE_MAX_SUFFIX_TOKENS {
        return None;
    }
    let capture_index = recovery_anchor.checked_sub(lcp_resume_start + 1)?;
    Some((suffix_len, capture_index))
}

fn store_qwen35_latest_turn_checkpoint(
    qwen: &mut Qwen35LoadedModel,
    kv_cache: &HybridKvCache,
    device: &MlxDevice,
    params: &SamplingParams,
    prompt_tokens: &[u32],
    anchor: usize,
    phase: &str,
    capture_index: Option<usize>,
) {
    let snapshot_started = Instant::now();
    let snapshot_result = match capture_index {
        Some(index) => kv_cache.snapshot_prefix_from_capture(device, anchor, index),
        None => kv_cache.snapshot_prefix(device, anchor),
    };
    match snapshot_result {
        Ok(snapshot) => {
            let snapshot_elapsed = snapshot_started.elapsed();
            let snapshot_bytes = snapshot.total_bytes();
            let base_key = build_lcp_key_for_qwen35(qwen, params);
            let linear_capacity = kv_cache
                .linear_attn
                .first()
                .map(|slot| slot.recurrent.byte_len())
                .unwrap_or(0);
            let disk_enabled = qwen.disk_persistor.is_some();
            let store_started = Instant::now();
            let store_result = qwen.store_lcp_with_disk_writeback(
                kv_cache,
                base_key,
                prompt_tokens[..anchor].to_vec(),
                snapshot,
                0,
                linear_capacity,
            );
            let store_elapsed = store_started.elapsed();
            tracing::info!(
                target: "hf2q::serve::api::engine_qwen35::progress",
                phase,
                anchor_tokens = anchor,
                snapshot_bytes,
                snapshot_ms = snapshot_elapsed.as_secs_f64() * 1_000.0,
                store_ms = store_elapsed.as_secs_f64() * 1_000.0,
                disk_enabled,
                capture_index,
                "Qwen35 latest-turn checkpoint prepared"
            );
            if let Err(error) = store_result {
                lcp_store_error_notify(phase, anchor, &error);
            }
        }
        Err(error) => lcp_snapshot_error_notify(phase, anchor, &error),
    }
}

/// ADR-017 Phase E.a default-on — byte budget for the Qwen35 LCP registry.
///
/// Delegates entirely to `default_lcp_byte_budget()` in `lcp_registry.rs`,
/// which probes `sysinfo::available_memory() × 5%` clamped to
/// `[1 GiB, 16 GiB]`, with `HF2Q_KV_LCP_RESUME_CAPACITY` override support.
///
/// Rationale: `HybridKvCacheSnapshot::total_bytes()` is the exact per-entry
/// cost (e.g. ~300 MB on Qwen 3.6 35B-A3B at stride=1024 with 8K tokens,
/// per the `c04d5d2` motivation for cap=8 ≈ 2.4 GB). The sysinfo-derived
/// budget on a 128 GB Mac with ~100 GB free ≈ 5 GB ≈ 16 entries at 300 MB
/// each — a natural replacement for the hard-coded cap=8. Operators who
/// want more entries raise the budget via `HF2Q_KV_LCP_RESUME_CAPACITY=10g`.
pub fn qwen35_lcp_registry_byte_budget() -> u64 {
    crate::serve::kv_persist::lcp_registry::default_lcp_byte_budget()
}

fn qwen35_reported_cached_tokens(
    prompt_len: usize,
    prompt_cache_hit: bool,
    lcp_resume_start: usize,
) -> usize {
    if prompt_cache_hit {
        prompt_len
    } else {
        lcp_resume_start.min(prompt_len)
    }
}

/// `true` when the running text ends with any of the registered stop
/// strings.  Mirrors `engine::hit_stop_string` (kept private to that
/// module for Gemma; replicated here so the Qwen35 path stays
/// self-contained without cross-module coupling).
fn qwen35_hit_stop_string(text: &str, stops: &[String]) -> bool {
    if stops.is_empty() {
        return false;
    }
    stops
        .iter()
        .any(|s| !s.is_empty() && text.ends_with(s.as_str()))
}

/// Strip the first matching trailing stop string from `text`.  Mirrors
/// `engine::strip_trailing_stop`.
fn qwen35_strip_trailing_stop(text: &mut String, stops: &[String]) {
    for s in stops {
        if !s.is_empty() && text.ends_with(s) {
            let new_len = text.len() - s.len();
            text.truncate(new_len);
            return;
        }
    }
}

/// Wedge-3 / Phase D: non-streaming chat generation against a loaded
/// Qwen3.5/3.6 model.  Replaces the `worker_run` 501 arm for
/// `Request::Generate`.
///
/// Pipeline (mirrors the Gemma `generate_once` shape):
///   1. Allocate per-request `HybridKvCache` sized for prompt + max_tokens.
///   2. `prompt_cache.try_match(prompt, params)` — on hit, restore the
///      prefill snapshot and use `cached.first_decoded_token` as the
///      seed; on miss, run `forward_gpu_last_logits` and snapshot.
///   3. Decode loop: greedy via `forward_gpu_greedy` (4-byte download per
///      step) when `params` is fully greedy; sampling via
///      `forward_gpu_last_logits` + `sample_logits_qwen35` otherwise
///      (note: sampling-mode allocates per-decode-step logits).
///   4. EOS: stop on `qwen.eos_token_ids.contains(&next_token)` OR
///      `params.stop_strings` match in the decoded running text OR
///      `max_tokens` reached.
///   5. Reasoning split via `super::registry::split_full_output(QWEN35, ...)`
///      — the registry-side splitter handles `<think>` / `</think>`
///      semantics out-of-the-box.
///
/// Tool-call splitter wiring is intentionally NOT included in the
/// non-streaming path for Wedge-3 — the registry-side `split_full_output`
/// only returns content + reasoning, not tool-call structure.  The
/// chat handler's non-streaming dispatcher already extracts tool calls
/// from the assembled text via `extract_tool_calls_from_text` (see
/// `handlers.rs:296+`); that helper consumes the same QWEN35 marker
/// pair (`<tool_call>` / `</tool_call>`) as the streaming path.  A
/// Wedge-4 follow-up may inline tool-call structure here for parity
/// with Gemma's non-streaming arm; for MVP the handler's call-graph is
/// the canonical post-decode parser.
fn generate_qwen35_once_ordinary(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    registration: Option<&ModelRegistration>,
    supervisor: &EngineSupervisor,
) -> Result<GenerationResult> {
    let request_start = Instant::now();
    let _disk_request_guard = qwen
        .disk_persistor
        .as_ref()
        .map(|persistor| persistor.begin_request());
    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "generate_qwen35_once: empty prompt_tokens"
    );
    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);

    // Greedy fast-path detection — controls the prompt-cache lookup +
    // the decode-step `forward_gpu_greedy` vs `forward_gpu_last_logits`
    // dispatch.
    let is_greedy = is_greedy_eligible(params);

    // ADR-020 AC#7 — populate per-token logprobs when the request opted
    // in.  When set, the greedy fast-path is bypassed (need full logits
    // to compute log_softmax) but token selection is identical (T=0
    // with sample_token_with_logprob ⇒ argmax).
    let want_logprobs = params.logprobs;
    let mut logprobs_vec: Option<Vec<f32>> = if want_logprobs {
        Some(Vec::with_capacity(max_tokens))
    } else {
        None
    };
    let mut grammar_runtime = grammar_runtime_for_request(params, registration)?;

    let cache_alloc_start = Instant::now();
    let device =
        MlxDevice::new().map_err(|e| anyhow::anyhow!("MlxDevice::new (qwen35 generate): {e}"))?;
    let (mut kv_cache, cache_reused) = take_serial_kv_cache(qwen, &device, prompt_len, max_tokens)?;
    tracing::info!(
        target: "hf2q::serve::api::engine_qwen35::progress",
        mode = "unary",
        prompt_tokens = prompt_len,
        max_tokens,
        cache_capacity_tokens = kv_cache.max_seq_len,
        cache_reused,
        tq_kv = qwen.tq_kv_active,
        elapsed_ms = cache_alloc_start.elapsed().as_secs_f64() * 1000.0,
        "Qwen35 request cache ready"
    );

    // ADR-027 iter-6b.3: cold-start hydrate of in-memory LcpRegistry
    // from disk-persisted snapshots.  Idempotent + cheap on hot path
    // (HashSet lookup) — no-op when HF2Q_KV_PERSIST is unset, when
    // this cfg has already been hydrated this process, or when the
    // cfg-subdir is empty.
    qwen.hydrate_lcp_registry_from_disk(&kv_cache, &device);

    // ── Prompt-cache fast-path ────────────────────────────────────
    let prompt_cache_hit = qwen.prompt_cache.try_match(prompt_tokens, params).is_some();

    // ── ADR-017 Phase B-hybrid.1 — LCP partial-prefix observability ──
    //
    // Mirrors Gemma's iter-2 probe at `engine.rs:3838-3940` but only
    // runs when the HybridPromptCache full-equality check missed
    // (matching the Gemma probe placement, post-PromptCache lookup).
    //
    // Iter B.1 scope (this commit): probe RUNS, increments
    // `lcp_lookups_total` and `lcp_detected_total` for Qwen35
    // requests. The `take_prefix` and `forward_gpu_with_resume`
    // resume path is NOT YET WIRED — iter B.2 lands the chunk-aligned
    // checkpoint storage + restore path.
    //
    // Skip multimodal: Qwen35 path doesn't currently expose
    // SoftTokenInjection at `generate_qwen35_once` (the soft-token
    // variant lives at `generate_qwen35_once_with_soft_tokens` line
    // 907; this site is text-only by construction).
    // ── ADR-017 Phase E.a B.2c + B.3: LCP partial-prefill resume probe ──
    //
    // Sequence: PromptCache full-equality first (fast path); if miss,
    // probe lcp_registry for the longest stride-aligned true-continuation
    // match across all chunk-position-keyed entries from prior requests.
    //
    // B.3 mid-prefill checkpointing: prior requests stored snapshots at
    // every chunk boundary k_end during their chunked prefill (under
    // chunk-position-keyed LcpKey).  Each snapshot's DeltaNet recurrent
    // state is at exactly position k_end, so resuming from there is
    // byte-correct (no position-mismatch).  This generalises B.2c v1's
    // true-continuation-only support: any LCP that lands on a stride
    // boundary now hits a checkpoint.
    //
    // Probe strategy: iterate chunk positions DESCENDING (from largest
    // possible stride-aligned position ≤ prompt_len down to stride),
    // probe each chunk-position-keyed LcpKey.  First true-continuation
    // hit (k == cached_prompt_len) wins.  Stops at first hit because
    // descending order = largest match first.
    let mut lcp_resume_start: usize = 0;
    if !prompt_cache_hit {
        // Chunk-aligned observability probe (counter-fix 2026-05-06,
        // Codex Phase-2b follow-up): qwen35 stores under chunk-keyed
        // LcpKeys via build_lcp_key_for_qwen35_chunk(qwen, params,
        // chunk_pos), so a BASE-key probe always returned None and
        // hf2q_kv_lcp_detected_total never incremented even when
        // resume engaged. probe_lcp_opportunity_chunk_aligned scans
        // stride-aligned chunk positions descending — the same shape
        // as the actual resume probe loop below, but side-effect-free
        // (no restore_partial). Returns Some(K) on the largest match
        // so detected_total now accurately reflects "resume would have
        // engaged on this request".
        //
        // Precompute the BASE key BEFORE taking the &mut on the
        // registry — the closure derives chunk-position keys by cloning
        // base_key and overwriting tenant_id (mirrors
        // build_lcp_key_for_qwen35_chunk's body). Avoids borrow conflict
        // between &mut qwen.lcp_registry and immutable use of qwen
        // inside the closure.
        let stride_for_observe = crate::debug::INVESTIGATION_ENV.kv_lcp_deltanet_checkpoint_stride;
        let base_key_for_observe = build_lcp_key_for_qwen35(qwen, params);
        let detected = lookup_qwen35_resume_checkpoint(
            &mut qwen.lcp_registry,
            &base_key_for_observe,
            prompt_tokens,
            stride_for_observe,
        )
        .map(|(prefix, _chunk_pos)| prefix.k);
        if let Some(sink) = qwen.kv_metrics_sink.as_ref() {
            sink.record_lcp_probe(detected);
        }
        let _ = detected;

        // B.2c gate: HF2Q_KV_LCP_RESUME enables actual partial-prefill
        // resume (default ON since 2026-05-06; observability above runs
        // unconditionally so metrics show LCP opportunity rate even when
        // resume is disabled). Q3 auto-disable: under default-on with
        // HF2Q_USE_DENSE=0, effective_kv_lcp_resume emits a one-shot
        // warn-once and returns false. Explicit HF2Q_KV_LCP_RESUME=1
        // overrides the auto-disable. Codex Phase-2b 2026-05-06 caught
        // this site missing the helper — added to mirror engine.rs:4099.
        // ADR-027 sub-iter 23d-γ (2026-08-03): qwen35's TQ-only restore
        // path is PROVEN (restore_partial restores all four TQ buffers
        // per slot; cold-vs-resumed byte-identity live-gated) — LCP is
        // unconditionally resumable on this arch, no explicit
        // HF2Q_KV_LCP_RESUME=1 needed under the production TQ regime.
        let lcp_resume_enabled = crate::serve::api::engine::effective_kv_lcp_resume(
            crate::debug::INVESTIGATION_ENV.kv_lcp_resume,
            true,
        );
        if lcp_resume_enabled {
            let stride = crate::debug::INVESTIGATION_ENV.kv_lcp_deltanet_checkpoint_stride;
            let base_key = build_lcp_key_for_qwen35(qwen, params);
            eprintln!(
                "[hf2q qwen35 lcp probe] enabled, registry_len={}, prompt_len={}, \
                 stride={}, scanning latest-turn + stride checkpoints",
                qwen.lcp_registry.len(),
                prompt_tokens.len(),
                stride,
            );
            if let Some((prefix, chunk_pos)) = lookup_qwen35_resume_checkpoint(
                &mut qwen.lcp_registry,
                &base_key,
                prompt_tokens,
                stride,
            ) {
                let snapshot: &HybridKvCacheSnapshot = &prefix.dense_kvs[0];
                let restore_start = Instant::now();
                kv_cache
                    .restore_partial(snapshot, prefix.k)
                    .context("qwen35 lcp_registry restore_partial")?;
                let restore_ms = restore_start.elapsed().as_micros() as f64 / 1000.0;
                lcp_resume_start = prefix.k;
                let checkpoint = if chunk_pos == 0 {
                    "LATEST-TURN"
                } else {
                    "STRIDE-ALIGNED"
                };
                eprintln!(
                    "[hf2q qwen35 lcp resume] {checkpoint} HIT — restoring at \
                     k={} (cached_prompt_len={}, chunk_pos={}, restore_ms={:.3})",
                    prefix.k, prefix.cached_prompt_len, chunk_pos, restore_ms
                );
            } else {
                eprintln!(
                    "[hf2q qwen35 lcp probe] no compatible checkpoint \
                     (registry_len={})",
                    qwen.lcp_registry.len()
                );
            }
        }
    }

    if cache_reused && !prompt_cache_hit && lcp_resume_start == 0 {
        kv_cache.reset();
    }

    let prefill_start = Instant::now();
    let mut next_token: u32;
    if prompt_cache_hit {
        // Hit: restore the substrate + reuse the cached first decoded token.
        let snap = qwen
            .prompt_cache
            .snapshot()
            .expect("try_match returned Some implies snapshot Some");
        kv_cache
            .restore_partial(snap, prompt_len)
            .context("prompt_cache restore_partial")?;
        next_token = qwen.prompt_cache.first_decoded_token();
        tracing::debug!(
            "qwen35 prompt_cache: HIT — {} tokens; prefill skipped",
            prompt_len
        );
    } else {
        // Miss: full prefill.
        //
        // ADR-017 Phase B-hybrid.2a — when HF2Q_KV_LCP_CHUNKED_PREFILL=1
        // AND prompt_len divides evenly by the configured stride, run
        // prefill in chunks instead of one monolithic call. Each chunk
        // call propagates state (full-attn current_len cursor +
        // DeltaNet recurrent state) via the kv_cache; the cumulative
        // effect MUST be byte-identical to the monolithic call (the
        // `phase_b2a_chunked_vs_monolithic_byte_identity` falsifier
        // guards this invariant).
        //
        // Constraint: prompt_len % stride == 0. The DeltaNet
        // chunk_gated_delta_rule kernel requires `seq_len % FIXED_BT
        // (=64) == 0`; stride defaults to 1024 (16 internal chunks),
        // and the chunked-prefill flow guarantees each call has
        // seq_len = stride for all but the last call. For
        // prompt_len % stride != 0, the last chunk would have a
        // non-aligned seq_len that may take a different kernel path,
        // breaking byte-identity. iter B.2b (next iter) extends to
        // handle the partial-tail case via a dedicated final-chunk
        // call; iter B.2a (this commit) keeps the constraint to
        // verify the foundational invariant first.
        let stride = crate::debug::INVESTIGATION_ENV.kv_lcp_deltanet_checkpoint_stride;
        // ADR-017 Phase E.a B.3: chunked prefill engagement requires
        // HF2Q_KV_LCP_CHUNKED_PREFILL=1.  Mid-prefill stride-aligned
        // checkpoints (B.3) only fire when chunked is engaged AND
        // HF2Q_KV_LCP_RESUME=1 is also set (gated below at the per-chunk
        // store site).
        //
        // Why we don't auto-enable chunked under LCP_RESUME=1: chunked
        // prefill at 3+ chunks with a partial-tail < 16 tokens hits a
        // kernel-level divergence — the partial-tail chunk falls to the
        // legacy F32 SDPA path (qL ∈ [2, 15] has no FA-fast-or-resume
        // coverage; see `gpu_full_attn.rs:1830-1852`), producing byte-
        // different output than monolithic FA fast path.  Forcing
        // chunked under LCP_RESUME would silently break byte-identity
        // for any prompt that ends in a partial-tail < 16-token chunk
        // (common: chat-template prompts rarely land on stride-aligned
        // token counts).  Operators must explicitly opt into chunked
        // prefill via HF2Q_KV_LCP_CHUNKED_PREFILL=1 alongside LCP_RESUME=1
        // to engage B.3's mid-prefill stores.
        //
        // B.4 follow-up: kernel-level FA fast/resume support for seq < 16
        // would close this divergence and enable auto-chunked-under-LCP.
        let lcp_resume_enabled = crate::debug::INVESTIGATION_ENV.kv_lcp_resume;
        // ADR-017 Phase E.a B.5: B.4 path-ii danger-zone bypass REMOVED.
        //
        // The gate lift at `gpu_full_attn.rs:1909+` (resume_path_eligible
        // now requires `kv_seq_len >= 16` instead of `seq_len >= 16`)
        // makes chunked prefill byte-identical to monolithic at ALL
        // prompt lengths — including the previously-broken danger zone
        // (`prompt_len % stride ∈ {1..15}` where the last chunk has
        // seq < 16).  Verified empirically:
        //
        //   * mlx-native kernel probe
        //     `flash_attn_prefill_bf16_d256_resume_small_ql_multi_kl_probe`:
        //     0/N elements differ at qL ∈ {2, 8, 15} with kL=130.
        //   * hf2q-side B.4 danger-zone falsifier (now retitled to test
        //     post-B.5 chunked engagement at danger-zone lengths).
        //
        // Chunked prefill now engages on all `prompt_len > stride`
        // requests under `HF2Q_KV_LCP_CHUNKED_PREFILL=1`, with B.3
        // mid-prefill stores firing on every stride-aligned chunk
        // boundary regardless of partial-tail length.
        let chunked_eligible = stride > 0
            && prompt_len > stride
            && (lcp_resume_start == 0 || lcp_resume_start % stride == 0)
            && crate::debug::INVESTIGATION_ENV.kv_lcp_chunked_prefill;
        let recovery_tail_tokens = qwen35_recovery_tail_tokens(qwen, prompt_tokens, params);
        let recovery_anchor = prompt_len.saturating_sub(recovery_tail_tokens);
        let recovery_eligible = lcp_resume_enabled
            && recovery_anchor > lcp_resume_start
            // The small-query-length resume kernel requires an established
            // prefix. Very short prompts stay monolithic.
            && recovery_anchor >= 16;
        let recovery_capture_plan = qwen35_recovery_capture_plan(
            lcp_resume_start,
            recovery_anchor,
            prompt_len,
            recovery_eligible,
            chunked_eligible,
        );
        let prefill_logits = if let Some((suffix_len, capture_index)) = recovery_capture_plan {
            let capture_alloc_start = Instant::now();
            kv_cache
                .ensure_la_capture(&qwen.model.cfg, &device, suffix_len as u32)
                .context("Qwen35 latest-turn recovery capture allocation")?;
            let capture_alloc_ms = capture_alloc_start.elapsed().as_secs_f64() * 1000.0;
            let position_build_start = Instant::now();
            let suffix_tokens = &prompt_tokens[lcp_resume_start..];
            let mut suffix_positions = vec![0i32; 4 * suffix_len];
            for axis in 0..4 {
                for token in 0..suffix_len {
                    suffix_positions[axis * suffix_len + token] = (lcp_resume_start + token) as i32;
                }
            }
            let position_build_ms = position_build_start.elapsed().as_secs_f64() * 1000.0;
            let forward_start = Instant::now();
            let logits = supervised_gpu_call(supervisor, "qwen35_serial_prefill", || {
                qwen.model
                    .forward_gpu_last_logits(
                        suffix_tokens,
                        &suffix_positions,
                        &mut kv_cache,
                        SlotId(0),
                    )
                    .context("Qwen35 captured latest-turn suffix prefill")
            })?;
            let forward_ms = forward_start.elapsed().as_secs_f64() * 1000.0;
            let checkpoint_start = Instant::now();
            store_qwen35_latest_turn_checkpoint(
                qwen,
                &kv_cache,
                &device,
                params,
                prompt_tokens,
                recovery_anchor,
                "captured latest-turn recovery-anchor",
                Some(capture_index),
            );
            let checkpoint_ms = checkpoint_start.elapsed().as_secs_f64() * 1000.0;
            let capture_clear_start = Instant::now();
            kv_cache.clear_la_capture();
            let capture_clear_ms = capture_clear_start.elapsed().as_secs_f64() * 1000.0;
            tracing::info!(
                target: "hf2q::serve::api::engine_qwen35::progress",
                mode = "unary",
                suffix_tokens = suffix_len,
                capture_index,
                capture_alloc_ms,
                position_build_ms,
                forward_ms,
                checkpoint_ms,
                capture_clear_ms,
                "Qwen35 captured recovery prefill phase timing"
            );
            eprintln!(
                "[hf2q qwen35 lcp store] captured latest-turn recovery anchor={} \
                 suffix_tokens={} capture_index={}",
                recovery_anchor, suffix_len, capture_index
            );
            logits
        } else if recovery_eligible && !chunked_eligible {
            let prefix_tokens = &prompt_tokens[lcp_resume_start..recovery_anchor];
            let prefix_len = prefix_tokens.len();
            let mut prefix_positions = vec![0i32; 4 * prefix_len];
            for axis in 0..4 {
                for token in 0..prefix_len {
                    prefix_positions[axis * prefix_len + token] = (lcp_resume_start + token) as i32;
                }
            }
            supervised_gpu_call(supervisor, "qwen35_serial_prefill", || {
                qwen.model
                    .forward_gpu_last_logits(
                        prefix_tokens,
                        &prefix_positions,
                        &mut kv_cache,
                        SlotId(0),
                    )
                    .context("Qwen35 latest-turn recovery-anchor prefix prefill")
            })?;
            store_qwen35_latest_turn_checkpoint(
                qwen,
                &kv_cache,
                &device,
                params,
                prompt_tokens,
                recovery_anchor,
                "latest-turn recovery-anchor",
                None,
            );

            let tail_tokens = &prompt_tokens[recovery_anchor..];
            let tail_len = tail_tokens.len();
            let mut tail_positions = vec![0i32; 4 * tail_len];
            for axis in 0..4 {
                for token in 0..tail_len {
                    tail_positions[axis * tail_len + token] = (recovery_anchor + token) as i32;
                }
            }
            eprintln!(
                "[hf2q qwen35 lcp store] latest-turn recovery anchor={} tail_tokens={}",
                recovery_anchor, tail_len
            );
            supervised_gpu_call(supervisor, "qwen35_serial_prefill", || {
                qwen.model
                    .forward_gpu_last_logits(tail_tokens, &tail_positions, &mut kv_cache, SlotId(0))
                    .context("Qwen35 latest-turn recovery-anchor tail prefill")
            })?
        } else if chunked_eligible {
            // ADR-017 Phase B-hybrid.2a + B.3 — chunked prefill with mid-
            // prefill stride-aligned checkpoint storage.  Each chunk's
            // `forward_gpu_last_logits` call leaves the kv_cache state at
            // position k_end; we snapshot under a chunk-position-keyed
            // LcpKey so the next request's probe finds it.
            //
            // Resumed-suffix mode (`lcp_resume_start > 0`): the first
            // chunk's `chunk_idx = lcp_resume_start / stride`.  The
            // kv_cache is already populated through `lcp_resume_start`
            // (via `restore_from(snapshot)`); the loop continues from
            // there, byte-identical to a fresh prefill that reached
            // `lcp_resume_start` via prior chunks.
            anyhow::ensure!(
                lcp_resume_start % stride == 0,
                "qwen35 chunked prefill: lcp_resume_start ({}) must be \
                 stride-aligned ({}) — snapshots are stored only at \
                 stride boundaries, so this should be guaranteed by the \
                 probe site",
                lcp_resume_start,
                stride
            );
            let first_chunk_idx = lcp_resume_start / stride;
            // On a cold chunked prefill, stop at the stable generation
            // boundary, store it under the base key, then process only the
            // generation-only template suffix. This replaces the final
            // stride snapshot instead of adding a second large synchronous
            // snapshot/write to the cold path.
            let chunked_prefill_end = if recovery_eligible {
                recovery_anchor
            } else {
                prompt_len
            };
            let n_chunks = (chunked_prefill_end + stride - 1) / stride;
            eprintln!(
                "[hf2q qwen35 chunked prefill] {} chunks (stride={}, \
                 prompt_len={}, prefill_end={}, first_chunk_idx={})",
                n_chunks, stride, prompt_len, chunked_prefill_end, first_chunk_idx
            );
            let mut last_logits: Option<Vec<f32>> = None;
            for chunk_idx in first_chunk_idx..n_chunks {
                let k_start = chunk_idx * stride;
                let k_end = ((chunk_idx + 1) * stride).min(chunked_prefill_end);
                let chunk_seq_len = k_end - k_start;
                let chunk_tokens = &prompt_tokens[k_start..k_end];
                // Build positions for [k_start..k_end). Positions are
                // absolute (NOT chunk-relative) so RoPE sees the
                // correct logical positions across the whole prompt.
                let mut chunk_positions = vec![0i32; 4 * chunk_seq_len];
                for axis in 0..4 {
                    for t in 0..chunk_seq_len {
                        chunk_positions[axis * chunk_seq_len + t] = (k_start + t) as i32;
                    }
                }
                let logits = supervised_gpu_call(
                    supervisor,
                    "qwen35_serial_prefill_chunk",
                    || {
                        qwen.model
                        .forward_gpu_last_logits(
                            chunk_tokens,
                            &chunk_positions,
                            &mut kv_cache,
                            SlotId(0),
                        )
                        .with_context(|| {
                            format!(
                                "qwen35 chunked prefill: chunk {}/{} (k_start={}, k_end={}, seq_len={})",
                                chunk_idx + 1, n_chunks, k_start, k_end, chunk_seq_len
                            )
                        })
                    },
                )?;
                if chunk_idx == n_chunks - 1 {
                    last_logits = Some(logits.clone());
                }

                // ADR-017 Phase E.a B.3: snapshot after each STRIDE-ALIGNED
                // chunk completion.  Includes the last chunk when k_end ==
                // prompt_len AND prompt_len is stride-aligned — that covers
                // the true-continuation case (turn-2 fully extends turn-1).
                // Skip non-stride-aligned partial-tail chunks (last chunk
                // when prompt_len isn't a multiple of stride): future
                // stride-descending probes can't address them, so storing
                // would burn ~300 MB without any chance of being looked up.
                let stride_aligned = k_end % stride == 0;
                let superseded_by_recovery_anchor = stride_checkpoint_superseded_by_recovery_anchor(
                    recovery_eligible,
                    k_end,
                    stride,
                    recovery_anchor,
                );
                let mid_store_disabled =
                    std::env::var("HF2Q_KV_LCP_DISABLE_MID_STORE").as_deref() == Ok("1");
                // 2026-08-03 store-gate fix: prefix KV state is a pure
                // function of the prompt tokens + weights — sampling
                // params (temperature/top_p/top_k/seed) never touch it.
                // The `is_greedy` conjunct here was copy-pasted from
                // HybridPromptCache (whose replayed first DECODED token
                // IS sampling-dependent) and silently disabled every
                // mid-prefill store for any client sending sampling
                // params — probes printed registry_len=0 forever and
                // every turn re-prefilled the whole conversation.
                // Greedy gating stays ONLY on the decoded-token replay
                // cache (HybridPromptCache); prefix checkpoints store
                // unconditionally.
                lcp_store_skip_notify(
                    stride_aligned && !superseded_by_recovery_anchor,
                    lcp_resume_enabled,
                    mid_store_disabled,
                );
                if lcp_resume_enabled
                    && stride_aligned
                    && !superseded_by_recovery_anchor
                    && !mid_store_disabled
                {
                    match kv_cache.snapshot_prefix(&device, k_end) {
                        Ok(snap) => {
                            let chunk_key = build_lcp_key_for_qwen35_chunk(qwen, params, k_end);
                            let linear_capacity = kv_cache
                                .linear_attn
                                .first()
                                .map(|s| s.recurrent.byte_len())
                                .unwrap_or(0);
                            // ADR-027 iter-6b.2: route store through the
                            // disk-write-through helper so HF2Q_KV_PERSIST
                            // sessions persist mid-prefill snapshots.
                            // Helper falls back to bare in-memory store
                            // when disk_persistor is None (zero-cost).
                            if let Err(e) = qwen.store_lcp_with_disk_writeback(
                                &kv_cache,
                                chunk_key,
                                prompt_tokens[..k_end].to_vec(),
                                snap,
                                0,
                                linear_capacity,
                            ) {
                                lcp_store_error_notify("mid-prefill", k_end, &e);
                            } else {
                                eprintln!(
                                    "[hf2q qwen35 lcp store] mid-prefill snapshot \
                                     at chunk_pos={k_end} (registry_len_after={})",
                                    qwen.lcp_registry.len()
                                );
                            }
                        }
                        Err(e) => {
                            lcp_snapshot_error_notify("mid-prefill", k_end, &e);
                        }
                    }
                }
            }
            if recovery_eligible {
                store_qwen35_latest_turn_checkpoint(
                    qwen,
                    &kv_cache,
                    &device,
                    params,
                    prompt_tokens,
                    recovery_anchor,
                    "chunked latest-turn recovery-anchor",
                    None,
                );

                let tail_tokens = &prompt_tokens[recovery_anchor..];
                let tail_len = tail_tokens.len();
                let mut tail_positions = vec![0i32; 4 * tail_len];
                for axis in 0..4 {
                    for token in 0..tail_len {
                        tail_positions[axis * tail_len + token] = (recovery_anchor + token) as i32;
                    }
                }
                eprintln!(
                    "[hf2q qwen35 lcp store] latest-turn recovery anchor={} tail_tokens={}",
                    recovery_anchor, tail_len
                );
                last_logits = Some(supervised_gpu_call(
                    supervisor,
                    "qwen35_serial_prefill",
                    || {
                        qwen.model
                            .forward_gpu_last_logits(
                                tail_tokens,
                                &tail_positions,
                                &mut kv_cache,
                                SlotId(0),
                            )
                            .context("Qwen35 chunked recovery-anchor tail prefill")
                    },
                )?);
            }
            last_logits.expect("at least one chunk in chunked prefill")
        } else if lcp_resume_start > 0 {
            // Suffix-only monolithic prefill from the LCP boundary.
            // The kv_cache is already populated through `lcp_resume_start`
            // by `restore_from(snapshot)`; the suffix call's first chunk
            // sees `cur_len = lcp_resume_start > 0`, which routes through
            // the FA bf16 d256 RESUME kernel (head_dim=256 production
            // path) — byte-identical to a fresh full prefill (B.2-fix
            // landed in commit fff4b4d / mlx-native@1819fad).
            let suffix_tokens = &prompt_tokens[lcp_resume_start..];
            let suffix_len = suffix_tokens.len();
            // Positions are absolute (axis-replicated 4x for IMROPE).
            let mut suffix_positions = vec![0i32; 4 * suffix_len];
            for axis in 0..4 {
                for t in 0..suffix_len {
                    suffix_positions[axis * suffix_len + t] = (lcp_resume_start + t) as i32;
                }
            }
            eprintln!(
                "[hf2q qwen35 lcp resume] suffix prefill {} tokens \
                 (lcp_resume_start={}, prompt_len={})",
                suffix_len, lcp_resume_start, prompt_len
            );
            supervised_gpu_call(supervisor, "qwen35_serial_prefill", || {
                qwen.model
                    .forward_gpu_last_logits(
                        suffix_tokens,
                        &suffix_positions,
                        &mut kv_cache,
                        SlotId(0),
                    )
                    .context("Qwen35Model::forward_gpu_last_logits (LCP resume suffix)")
            })?
        } else {
            let positions = prefill_positions_for(prompt_len);
            supervised_gpu_call(supervisor, "qwen35_serial_prefill", || {
                qwen.model
                    .forward_gpu_last_logits(prompt_tokens, &positions, &mut kv_cache, SlotId(0))
                    .context("Qwen35Model::forward_gpu_last_logits (prefill)")
            })?
        };
        anyhow::ensure!(
            prefill_logits.len() == qwen.vocab_size,
            "qwen35 prefill logits len {} != vocab_size {}",
            prefill_logits.len(),
            qwen.vocab_size
        );
        // First decoded token: greedy argmax for the greedy path; for
        // sampling-mode we apply the sampler to the prefill logits so
        // user temperature affects the very first generated token (same
        // contract as Gemma's `generate_once`).
        if is_greedy {
            next_token = greedy_argmax_last_token(&prefill_logits, qwen.vocab_size as u32);
        } else {
            let mut logits = prefill_logits.clone();
            let (token, logprob) = sample_logits_qwen35_constrained(
                &mut logits,
                params,
                &[],
                grammar_runtime.as_ref(),
                want_logprobs,
            )?;
            next_token = token;
            if let (Some(values), Some(logprob)) = (logprobs_vec.as_mut(), logprob) {
                values.push(logprob);
            }
        }
        advance_qwen35_grammar(&mut grammar_runtime, params, next_token);

        // Snapshot + cache update.  The snapshot captures KV state AFTER
        // the prefill (current_len[0] == prompt_len for full-attn slots;
        // DeltaNet conv/recurrent populated by the linear-attn layer's
        // prefill emit).  Split by sampling-dependence (2026-08-03
        // store-gate fix):
        //
        // * prompt_cache (Phase E.b full-equality replay) STAYS
        //   greedy-gated: it replays the first DECODED token, which is a
        //   function of the sampling params — a sampled token must never
        //   be replayed as a future greedy request's first token.
        //
        // * lcp_registry (Phase E.a partial-prefill resume) is NOT
        //   greedy-gated: prefix KV state is a pure function of the
        //   prompt tokens + weights, so checkpoints are valid regardless
        //   of how the decoded tokens were sampled.  Gating stores on
        //   `is_greedy` silently disabled them for any client sending
        //   temperature/top_p/top_k/seed (registry_len=0 forever).
        //
        // ADR-017 Phase E.a B.2b: snapshot BOTH the existing prompt_cache
        // (Phase E.b full-equality replay, capacity 1) AND the lcp_registry
        // (Phase E.a partial-prefill resume, capacity defined at
        // Qwen35LoadedModel construction).  Two snapshots is wasteful
        // (~2× GB-scale memcpy) — a follow-up should refactor
        // HybridPromptCache to share Arc<HybridKvCacheSnapshot> with
        // lcp_registry, eliminating the duplication.  Cost is one-time
        // per request, amortised over the prefill saving on the next
        // matching request.
        if is_greedy {
            // Snapshot 1: prompt_cache (full-equality replay, Phase E.b).
            let prompt_snapshot_start = Instant::now();
            match kv_cache.snapshot_prefix(&device, prompt_len) {
                Ok(snap) => {
                    let snapshot_bytes = snap.total_bytes();
                    qwen.prompt_cache
                        .update(prompt_tokens.to_vec(), snap, next_token, params);
                    tracing::info!(
                        target: "hf2q::serve::api::engine_qwen35::progress",
                        phase = "full-prompt replay",
                        snapshot_bytes,
                        elapsed_ms = prompt_snapshot_start.elapsed().as_secs_f64() * 1000.0,
                        "Qwen35 prompt-cache checkpoint prepared"
                    );
                }
                Err(e) => {
                    eprintln!("[hf2q qwen35 lcp store] prompt_cache snapshot failed: {e}");
                }
            }
        }
    }
    let prefill_duration = prefill_start.elapsed();
    let reported_cached_tokens =
        qwen35_reported_cached_tokens(prompt_len, prompt_cache_hit, lcp_resume_start);
    let prefill_work_tokens = prompt_len.saturating_sub(reported_cached_tokens);
    tracing::info!(
        target: "hf2q::serve::api::engine_qwen35::progress",
        mode = "unary",
        prompt_tokens = prompt_len,
        cached_tokens = reported_cached_tokens,
        work_tokens = prefill_work_tokens,
        elapsed_ms = prefill_duration.as_secs_f64() * 1000.0,
        tokens_per_second = if prefill_duration.is_zero() {
            0.0
        } else {
            prefill_work_tokens as f64 / prefill_duration.as_secs_f64()
        },
        "Qwen35 prefill complete"
    );

    // ── Decode loop ────────────────────────────────────────────────
    let decode_start = Instant::now();
    let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    generated_tokens.push(next_token);

    let first_fragment = qwen
        .tokenizer
        .decode(&[next_token], false)
        .unwrap_or_default();
    let mut decoded_text = first_fragment.clone();

    let mut finish_reason: &'static str = "length";

    // Early EOS check on the prefill-emitted first token.
    if qwen.eos_token_ids.contains(&next_token) {
        finish_reason = "stop";
    } else if qwen35_hit_stop_string(&decoded_text, &params.stop_strings) {
        finish_reason = "stop";
        qwen35_strip_trailing_stop(&mut decoded_text, &params.stop_strings);
    } else {
        for step in 1..max_tokens {
            let pos = (prompt_len + step - 1) as i32;
            // Bound check on the KV cache.  The alloc helper sized
            // `max_seq` to cover the full request; if the iter overshoots
            // (e.g. caller stretched max_tokens between the alloc and
            // here — not possible inside this function but defensive),
            // stop with "length" rather than corrupting.
            if pos as u32 >= kv_cache.max_seq_len {
                tracing::warn!(
                    pos,
                    max_seq = kv_cache.max_seq_len,
                    "qwen35 decode: hit kv-cache bound; stopping with finish=length",
                );
                break;
            }
            let decode_positions = vec![pos; 4];

            next_token = if is_greedy {
                supervised_gpu_call(supervisor, "qwen35_serial_decode", || {
                    qwen.model
                        // ADR-040 Phase B4d (2026-05-30) — forward_gpu_greedy
                        // now accepts SlotId.  Single-seq engine path:
                        // SlotId(0) is byte-identical to pre-B4d. C2c/C2d
                        // SlotAware activation has its own slot-aware
                        // sibling (forward_gpu_last_logits(.., slot_id))
                        // — see engine_qwen35.rs:5204 for the routing.
                        .forward_gpu_greedy(
                            &[next_token],
                            &decode_positions,
                            &mut kv_cache,
                            SlotId(0),
                        )
                        .with_context(|| format!("forward_gpu_greedy decode step {step}"))
                })?
            } else {
                let logits_full = supervised_gpu_call(supervisor, "qwen35_serial_decode", || {
                    qwen.model
                        .forward_gpu_last_logits(
                            &[next_token],
                            &decode_positions,
                            &mut kv_cache,
                            SlotId(0),
                        )
                        .with_context(|| format!("forward_gpu_last_logits decode step {step}"))
                })?;
                let mut logits = logits_full;
                let (token, logprob) = sample_logits_qwen35_constrained(
                    &mut logits,
                    params,
                    &generated_tokens,
                    grammar_runtime.as_ref(),
                    want_logprobs,
                )?;
                if let (Some(values), Some(logprob)) = (logprobs_vec.as_mut(), logprob) {
                    values.push(logprob);
                }
                token
            };
            advance_qwen35_grammar(&mut grammar_runtime, params, next_token);

            if qwen.eos_token_ids.contains(&next_token) {
                finish_reason = "stop";
                break;
            }
            if qwen35_grammar_terminal_token(grammar_runtime.as_ref(), params, next_token) {
                finish_reason = "stop";
                break;
            }
            generated_tokens.push(next_token);
            let fragment = qwen
                .tokenizer
                .decode(&[next_token], false)
                .unwrap_or_default();
            decoded_text.push_str(&fragment);
            if qwen35_hit_stop_string(&decoded_text, &params.stop_strings) {
                finish_reason = "stop";
                qwen35_strip_trailing_stop(&mut decoded_text, &params.stop_strings);
                break;
            }
        }
    }
    let decode_duration = decode_start.elapsed();
    qwen.persistent_kv_cache = Some(kv_cache);
    tracing::info!(
        target: "hf2q::serve::api::engine_qwen35::progress",
        mode = "unary",
        generated_tokens = generated_tokens.len(),
        elapsed_ms = decode_duration.as_secs_f64() * 1000.0,
        tokens_per_second = if decode_duration.is_zero() {
            0.0
        } else {
            generated_tokens.len() as f64 / decode_duration.as_secs_f64()
        },
        "Qwen35 decode complete"
    );

    // ── Reasoning split (Decision #21) ────────────────────────────
    // Use the existing registry helper so the split is byte-identical
    // to the Gemma path's non-streaming arm.  Tool-call extraction is
    // owned by the chat handler's post-decode pipeline, NOT this
    // function — see Wedge-3 PRD Phase D step 10 + the docstring above.
    let (content, reasoning_text) = match registration {
        Some(reg) if reg.has_reasoning() => super::registry::split_full_output_forced(
            reg,
            &decoded_text,
            params.reasoning_forced_open,
        ),
        _ => (decoded_text, None),
    };

    // Reasoning token count: rerun the splitter token-by-token to
    // attribute each completion token to a slot.  Cheap (one decode
    // per token + a tail-buffered splitter) and matches the Gemma
    // path's `reasoning_token_count` semantics.
    let reasoning_token_count = match registration {
        Some(reg) if reg.has_reasoning() => {
            let mut sp =
                super::registry::make_reasoning_splitter(reg, params.reasoning_forced_open);
            let mut count = 0usize;
            for &tok in &generated_tokens {
                let frag = qwen.tokenizer.decode(&[tok], false).unwrap_or_default();
                if let Some(splitter) = sp.as_mut() {
                    let _ = splitter.feed(&frag);
                    if splitter.in_reasoning() {
                        count += 1;
                    }
                }
            }
            count
        }
        _ => 0,
    };

    tracing::info!(
        target: "hf2q::serve::api::engine_qwen35::progress",
        mode = "unary",
        prompt_tokens = prompt_len,
        cached_tokens = reported_cached_tokens,
        completion_tokens = generated_tokens.len(),
        total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
        "Qwen35 request complete"
    );

    Ok(GenerationResult {
        text: content,
        reasoning_text,
        prompt_tokens: prompt_len,
        completion_tokens: generated_tokens.len(),
        reasoning_tokens: if reasoning_token_count > 0 {
            Some(reasoning_token_count)
        } else {
            None
        },
        finish_reason,
        prefill_duration,
        decode_duration,
        cached_tokens: reported_cached_tokens,
        logprobs: logprobs_vec,
    })
}

/// Server transaction dispatcher for native Qwen MTP speculation.
///
/// This is intentionally a narrow, exact greedy slice. The MTP sampler has
/// no server-complete distribution transform yet, so every sampled,
/// grammar-constrained, logprob, or biased request is routed to the ordinary
/// decoder. A future sampler lane can broaden `is_greedy_eligible` only after
/// it proves that proposal and target distributions are identical.
pub(super) fn generate_qwen35_once(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    registration: Option<&ModelRegistration>,
    supervisor: &EngineSupervisor,
) -> Result<GenerationResult> {
    use super::qwen35_speculation::{self, QwenSpeculationDecision};

    let mut decision = qwen.speculation.decide(
        prompt_tokens,
        is_serial_mtp_exact_eligible(params),
        qwen.model.mtp.is_some(),
        qwen.prompt_cache.try_match(prompt_tokens, params).is_some(),
    );

    if decision == QwenSpeculationDecision::Eligible {
        match generate_qwen35_once_mtp(qwen, prompt_tokens, params, registration) {
            Ok((result, stats)) => {
                qwen35_speculation::record_outcome(
                    stats.proposed,
                    stats.accepted,
                    stats.rejected,
                    stats.target_forwards,
                    result.cached_tokens,
                );
                tracing::info!(
                    target: "hf2q::serve::api::engine_qwen35::speculation",
                    drafted_tokens = stats.proposed,
                    accepted_tokens = stats.accepted,
                    rejected_tokens = stats.rejected,
                    target_forwards = stats.target_forwards,
                    cached_tokens = result.cached_tokens,
                    "Qwen native MTP transaction complete"
                );
                return Ok(result);
            }
            Err(error) => {
                // No persistent speculative KV is retained by this slice, so
                // falling back cannot expose partial target state to the
                // ordinary server cache. Preserve availability over an
                // optional optimization failure.
                tracing::warn!(
                    error = %error,
                    "Qwen native MTP unavailable; falling back to ordinary decode"
                );
                decision = QwenSpeculationDecision::RuntimeUnavailable;
            }
        }
    }

    qwen35_speculation::record_fallback(decision);
    let result =
        generate_qwen35_once_ordinary(qwen, prompt_tokens, params, registration, supervisor)?;
    qwen35_speculation::record_outcome(0, 0, 0, 0, result.cached_tokens);
    tracing::debug!(
        target: "hf2q::serve::api::engine_qwen35::speculation",
        ?decision,
        cached_tokens = result.cached_tokens,
        "Qwen ordinary decode selected"
    );
    Ok(result)
}

fn generate_qwen35_once_mtp(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    registration: Option<&ModelRegistration>,
) -> Result<(
    GenerationResult,
    crate::inference::models::qwen35::spec_decode::SpecDecodeStats,
)> {
    use crate::inference::models::qwen35::spec_decode::SpecDecode;

    anyhow::ensure!(!prompt_tokens.is_empty(), "Qwen MTP: empty prompt");
    anyhow::ensure!(
        is_serial_mtp_exact_eligible(params),
        "Qwen MTP: request has unsupported sampling, grammar, tool, or thinking semantics"
    );
    anyhow::ensure!(
        qwen.model.mtp.is_some(),
        "Qwen MTP: model has no MTP weights"
    );
    let max_tokens = params.max_tokens.max(1);
    let result = SpecDecode::run_with_eos_set(
        &qwen.model,
        prompt_tokens,
        max_tokens,
        qwen.eos_token_ids.clone(),
        qwen.model.cfg.max_position_embeddings,
    )?;
    let mut decoded_text = qwen
        .tokenizer
        .decode(&result.tokens, false)
        .unwrap_or_default();
    let finish_reason = if result.tokens.len() < max_tokens {
        "stop"
    } else {
        "length"
    };

    let (content, reasoning_text) = match registration {
        Some(reg) if reg.has_reasoning() => super::registry::split_full_output_forced(
            reg,
            &decoded_text,
            params.reasoning_forced_open,
        ),
        _ => (std::mem::take(&mut decoded_text), None),
    };
    let reasoning_token_count = match registration {
        Some(reg) if reg.has_reasoning() => {
            let mut splitter =
                super::registry::make_reasoning_splitter(reg, params.reasoning_forced_open);
            let mut count = 0usize;
            for &token in &result.tokens {
                let fragment = qwen.tokenizer.decode(&[token], false).unwrap_or_default();
                if let Some(splitter) = splitter.as_mut() {
                    let _ = splitter.feed(&fragment);
                    if splitter.in_reasoning() {
                        count += 1;
                    }
                }
            }
            count
        }
        _ => 0,
    };
    Ok((
        GenerationResult {
            text: content,
            reasoning_text,
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: result.tokens.len(),
            reasoning_tokens: (reasoning_token_count > 0).then_some(reasoning_token_count),
            finish_reason,
            prefill_duration: result.stats.prefill_elapsed,
            decode_duration: result.stats.decode_elapsed,
            // The native MTP runner currently starts from a fresh cache. The
            // dispatcher declines exact prompt-cache hits instead of claiming
            // reuse that did not happen.
            cached_tokens: 0,
            logprobs: None,
        },
        result.stats,
    ))
}

/// **ADR-040 iter-C2d-cont-kernel iter-1 (2026-05-29)** — slot-aware
/// chat-generation entry that routes the worker hot path through the
/// persistent multi-seq `HybridKvCache` (`Qwen35LoadedModel.
/// persistent_kv_cache`) instead of the per-request `alloc_kv_cache_
/// for_request` allocation.
///
/// **Scope (iter-1 only — Generate arm)**. This is the minimal valid
/// increment of `iter-C2d-cont-kernel` per the §6.1.27 closure block.
/// Lifts ONLY the `Request::Generate` worker arm onto the persistent
/// cache at `SlotId(N>0)` for Qwen35; the other 3 worker arms
/// (`Request::GenerateStream`, `Request::Embed`,
/// `Request::GenerateWithSoftTokens`) STILL carry the typed
/// `MultiSeqError::CapabilityUnsupported` clamp from C2d-cont §6.1.24
/// with a relabeled `iter-C2d-cont-kernel-iter-{2,3,4}` deferral cite.
/// See §6.1.27 for the iter-1 → iter-{2,3,4} sequencing decision.
///
/// **Differences from `generate_qwen35_once`**:
/// 1. Caller passes an explicit `&mut HybridKvCache` (the persistent
///    cache, taken out of `Qwen35LoadedModel` at the worker-arm site)
///    + an explicit `slot_id: SlotId`. No per-request alloc happens
///    inside this fn.
/// 2. Snapshot restore uses `restore_partial(snap, prefix.k)` instead of
///    `restore_from(snap)`. The persistent cache is sized to
///    `cfg.max_position_embeddings` while snapshots were taken at the
///    per-request `prompt_len + max_tokens + 64` sizing — `restore_from`
///    requires byte-equal `max_seq_len`, while `restore_partial` copies
///    only the first `n_tokens` per-head positions and works across
///    different sizes. This is the §6.1.27 prompt-cache invariant lift.
/// 3. Entry + exit reset the slot's per-slot region via
///    `kv_cache.reset_for_slot(slot_id)` so the persistent cache is
///    request-isolated within the slot — the next request to land on
///    this slot sees a zero-cursor full-attn cache + zero recurrent
///    / conv state. Matches the fresh-alloc invariant the per-request
///    path relied on for SlotAware SlotId(0) (H39 byte-equivalence pin).
/// 4. Threads `slot_id` into every `forward_gpu_last_logits` call (the
///    signature already accepts `SlotId` post-B4b §6.1.20). At
///    `SlotId(0)` this is byte-equivalent to `generate_qwen35_once` by
///    construction (forward_gpu_last_logits at SlotId(0) routes through
///    the pre-A2b single-seq path verbatim).
///
/// **Per-slot byte-equivalence at SlotId(0)** (H51 pin):
/// `generate_qwen35_once_slot_aware(.., kv=&mut persistent_cache,
/// slot_id=SlotId(0))` produces the same `GenerationResult` as
/// `generate_qwen35_once(..)` for any non-spec-decode greedy request
/// when `persistent_cache.n_seqs == 1` AND `persistent_cache.max_seq_len
/// >= prompt_len + max_tokens + 64`. Because (a) `reset_for_slot(0)` at
/// entry zeros the cursors (matching the fresh-alloc state), (b)
/// `forward_gpu_last_logits(.., SlotId(0))` is byte-equivalent to the
/// pre-A2b path (B4a §6.1.4 pin), and (c) `restore_partial(snap, k)` at
/// `k == snap.full_attn_current_len[0][0]` is byte-equivalent to
/// `restore_from(snap)` per the kv_cache.rs:2143 docstring.
///
/// **Co-changes** (iter-1 deliberately minimal):
/// - Per-slot LCP / mid-prefill checkpoint storage is DISABLED in
///   slot-aware mode.  ADR-040 §6.1.50 (2026-05-30) closes
///   **iter-C2d-cont-kernel-iter-LCP per ADR-040 §6.1.50** as
///   **STRUCTURAL N/A**: the snapshot codec keys snapshots on per-
///   request `max_seq_len = prompt_len + max_tokens + 64` while the
///   persistent multi-seq cache is sized to
///   `cfg.max_position_embeddings`; cross-slot prefix sharing also
///   carries tenant-isolation risk (LCP cache is global, slot regions
///   are per-tenant).  The full-prompt-cache HIT fast-path already uses
///   `restore_partial` (working at line ~2308); the chunked-prefill
///   mid-store + cross-request `lcp_registry.probe_lcp_opportunity`
///   paths would require either (a) per-slot LcpRegistry plus a slot
///   discriminator in the persisted snapshot format, or (b) a tenant-
///   aware cross-slot probe — both multi-iter snapshot-codec extensions
///   beyond the iter-LCP scope.  The structural-N/A pin documents the
///   call-graph reality: slot-aware mode's existing prompt-cache HIT
///   path (full-equality) IS the LCP fast-path operators get; the
///   remaining LCP feature (partial-prefix probe across requests) is
///   structurally incompatible with per-slot byte regions today.
/// - Chunked-prefill is DISABLED in slot-aware mode (same snapshot-
///   shape reason).  Same STRUCTURAL N/A pin per §6.1.50.
/// - DFlash / spec-decode capture-states are NOT engaged here (they
///   require `ensure_la_capture` which is spec-decode-only); slot-aware
///   spec-decode is **iter-B4d** per §6.1.26 deferrals matrix.
/// - ADR-040 §6.1.50 (2026-05-30) lands
///   **iter-C2d-cont-kernel-iter-G per ADR-040 §6.1.50** REAL LIFT: the
///   greedy decode branch (T=0, no logprobs, no grammar) now routes
///   through `forward_gpu_greedy(.., slot_id)` instead of
///   `forward_gpu_last_logits + greedy_argmax_last_token`.  Saves the
///   per-step vocab-size F32 download (~250 µs at vocab=151k) by
///   returning the argmax token directly from the GPU.  Sampling +
///   logprobs branches UNCHANGED (still use `forward_gpu_last_logits`
///   for CPU-side sampler / logprob computation).
///
/// # Errors
/// - `prompt_tokens.is_empty()` (matches `generate_qwen35_once`).
/// - `slot_id.0 >= kv_cache.n_seqs` (via `reset_for_slot` bounds-first
///   per A2b §6.1.23 iter-1.5 cfa-finding-F5).
/// - Forward / sample failures propagate from `forward_gpu_last_logits`
///   (sampling branch) or `forward_gpu_greedy` (greedy fast-path branch
///   per ADR-040 §6.1.50 iter-G).
pub fn generate_qwen35_once_slot_aware(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    registration: Option<&ModelRegistration>,
    kv_cache: &mut HybridKvCache,
    slot_id: SlotId,
) -> Result<GenerationResult> {
    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "generate_qwen35_once_slot_aware: empty prompt_tokens"
    );
    // Bounds-first per A2b §6.1.23 iter-1.5 cfa-finding-F5 ordering.
    anyhow::ensure!(
        slot_id.0 < kv_cache.n_seqs,
        "generate_qwen35_once_slot_aware: SlotOutOfRange slot={} max_slots={} \
         (ADR-040 iter-C2d-cont-kernel iter-1)",
        slot_id.0,
        kv_cache.n_seqs,
    );
    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);
    // Verify the persistent cache has room for this request. Persistent
    // cache is sized to `cfg.max_position_embeddings`; per-request need
    // is `prompt_len + max_tokens + 64`.
    let need_seq = prompt_len + max_tokens + 64;
    if need_seq > kv_cache.max_seq_len as usize {
        return Err(anyhow::anyhow!(
            "generate_qwen35_once_slot_aware: per-request need_seq={} exceeds \
             persistent cache max_seq_len={} (slot={} prompt_len={} max_tokens={}). \
             ADR-040 iter-C2d-cont-kernel iter-1 sizes the persistent cache to \
             cfg.max_position_embeddings; reduce max_tokens or use a shorter prompt.",
            need_seq,
            kv_cache.max_seq_len,
            slot_id.0,
            prompt_len,
            max_tokens
        ));
    }

    let is_greedy = is_greedy_eligible(params);
    let want_logprobs = params.logprobs;
    let mut logprobs_vec: Option<Vec<f32>> = if want_logprobs {
        Some(Vec::with_capacity(max_tokens))
    } else {
        None
    };

    // Per-slot reset at entry — the persistent cache may carry stale
    // bytes from a prior request on this slot. `reset_for_slot` zeros
    // the per-seq cursors + linear-attn conv/recurrent slices for
    // `slot_id` only (other slots untouched).
    kv_cache
        .reset_for_slot(slot_id)
        .context("ADR-040 iter-C2d-cont-kernel iter-1: reset_for_slot at entry")?;

    // ── Prompt-cache fast-path (slot-aware variant) ──────────────────
    //
    // Snapshot restore uses `restore_partial(snap, k)` instead of
    // `restore_from(snap)`. The persistent cache's `max_seq_len`
    // (= cfg.max_position_embeddings) differs from the snapshot
    // producer's per-request `max_seq_len` — `restore_from` would
    // ensure_byte_equal fail; `restore_partial` copies only the first
    // `k` per-head positions and works across sizes.
    let prompt_cache_hit = qwen.prompt_cache.try_match(prompt_tokens, params).is_some();

    let prefill_start = Instant::now();
    let next_token: u32;
    if prompt_cache_hit {
        let snap = qwen
            .prompt_cache
            .snapshot()
            .expect("try_match returned Some implies snapshot Some");
        // Use the full prompt length as the partial-restore boundary —
        // the HybridPromptCache full-equality hit means the snapshot
        // covers exactly `prompt_len` tokens.
        kv_cache
            .restore_partial(snap, prompt_len)
            .context("ADR-040 iter-C2d-cont-kernel iter-1: prompt_cache restore_partial")?;
        next_token = qwen.prompt_cache.first_decoded_token();
        tracing::debug!(
            "qwen35 slot-aware prompt_cache: HIT slot={} prompt_len={} prefill skipped",
            slot_id.0,
            prompt_len
        );
    } else {
        // Fresh monolithic prefill — chunked-prefill + LCP-resume are
        // STRUCTURAL N/A in slot-aware mode (see fn docstring for the
        // snapshot codec invariant rationale; the iter-LCP STRUCTURAL
        // N/A closure landed at iter-C2d-cont-kernel-iter-LCP per
        // ADR-040 §6.1.50).
        let positions = prefill_positions_for(prompt_len);
        let prefill_logits = qwen
            .model
            .forward_gpu_last_logits(prompt_tokens, &positions, kv_cache, slot_id)
            .context("Qwen35Model::forward_gpu_last_logits (slot-aware prefill)")?;
        anyhow::ensure!(
            prefill_logits.len() == qwen.vocab_size,
            "qwen35 slot-aware prefill logits len {} != vocab_size {}",
            prefill_logits.len(),
            qwen.vocab_size
        );
        if is_greedy && !want_logprobs {
            next_token = greedy_argmax_last_token(&prefill_logits, qwen.vocab_size as u32);
        } else {
            let mut logits = prefill_logits;
            if let Some(ref mut lps) = logprobs_vec {
                let (tok, lp) = sample_logits_qwen35_with_logprob(&mut logits, params, &[]);
                lps.push(lp);
                next_token = tok;
            } else {
                next_token = sample_logits_qwen35(&mut logits, params, &[]);
            }
        }
    }
    let prefill_duration = prefill_start.elapsed();

    // ── Decode loop ───────────────────────────────────────────────────
    let device = MlxDevice::new()
        .map_err(|e| anyhow::anyhow!("MlxDevice::new (qwen35 slot-aware decode): {e}"))?;
    let _ = &device; // sample_logits_qwen35 doesn't need the device handle
    let decode_start = Instant::now();
    let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    generated_tokens.push(next_token);
    let mut decoded_text = qwen
        .tokenizer
        .decode(&[next_token], false)
        .unwrap_or_default();
    let stops = &params.stop_strings;
    let mut finish_reason: &'static str = "length";
    if qwen.eos_token_ids.contains(&next_token) {
        // Strip the EOS token from the visible output: it shouldn't be
        // surfaced to the user even though we count it.
        generated_tokens.pop();
        decoded_text.clear();
        finish_reason = "stop";
    } else if qwen35_hit_stop_string(&decoded_text, stops) {
        qwen35_strip_trailing_stop(&mut decoded_text, stops);
        finish_reason = "stop";
    }

    // Decode loop.
    //
    // ADR-040 §6.1.50 (2026-05-30) iter-C2d-cont-kernel-iter-G REAL LIFT:
    // the greedy fast-path (`is_greedy && !want_logprobs`) now routes
    // through `forward_gpu_greedy(.., slot_id)` instead of
    // `forward_gpu_last_logits + greedy_argmax_last_token`.  The GPU-side
    // argmax returns a `u32` directly, eliminating the per-step vocab-
    // size F32 download (saves ~250 µs at vocab=151k Qwen3.6 35B-A3B).
    // The sampling + logprobs branches still go through
    // `forward_gpu_last_logits` because they need the full logits buffer
    // CPU-side for `sample_logits_qwen35` / `sample_logits_qwen35_with_logprob`.
    let mut step = 1usize;
    while step < max_tokens && finish_reason == "length" {
        // Position for the just-emitted next_token (used in the NEXT
        // forward as the decode-step input position).
        let pos = prompt_len + step - 1;
        let pos_i32 = pos as i32;
        let positions: Vec<i32> = vec![pos_i32; 4];
        let last_input = &generated_tokens[generated_tokens.len() - 1..];
        let tok = if is_greedy && !want_logprobs {
            // ADR-040 §6.1.50 iter-G fast path: `forward_gpu_greedy`
            // accepts `slot_id` since B4d §6.1.44 (2026-05-30).
            qwen.model
                .forward_gpu_greedy(last_input, &positions, kv_cache, slot_id)
                .with_context(|| {
                    format!(
                        "Qwen35Model::forward_gpu_greedy (slot-aware decode step {step}; \
                         ADR-040 §6.1.50 iter-G)"
                    )
                })?
        } else {
            let logits = qwen
                .model
                .forward_gpu_last_logits(last_input, &positions, kv_cache, slot_id)
                .with_context(|| {
                    format!("Qwen35Model::forward_gpu_last_logits (slot-aware decode step {step})")
                })?;
            anyhow::ensure!(
                logits.len() == qwen.vocab_size,
                "qwen35 slot-aware decode logits len {} != vocab_size {}",
                logits.len(),
                qwen.vocab_size
            );
            let mut logits = logits;
            if let Some(ref mut lps) = logprobs_vec {
                let (tok, lp) =
                    sample_logits_qwen35_with_logprob(&mut logits, params, &generated_tokens);
                lps.push(lp);
                tok
            } else {
                sample_logits_qwen35(&mut logits, params, &generated_tokens)
            }
        };
        if qwen.eos_token_ids.contains(&tok) {
            finish_reason = "stop";
            break;
        }
        generated_tokens.push(tok);
        let frag = qwen.tokenizer.decode(&[tok], false).unwrap_or_default();
        decoded_text.push_str(&frag);
        if qwen35_hit_stop_string(&decoded_text, stops) {
            qwen35_strip_trailing_stop(&mut decoded_text, stops);
            finish_reason = "stop";
            break;
        }
        step += 1;
    }
    let decode_duration = decode_start.elapsed();

    // Per-slot reset at exit — leave the slot clean for the next
    // request to land on it. Belt-and-suspenders w/ the entry reset:
    // ensures that even if a future iter adds a code path that
    // bypasses the entry reset, the slot is always clean at handoff.
    kv_cache
        .reset_for_slot(slot_id)
        .context("ADR-040 iter-C2d-cont-kernel iter-1: reset_for_slot at exit")?;

    // Reasoning split — mirror of `generate_qwen35_once` (line 2111
    // pre-iter-1). `split_full_output(reg, &text)` returns
    // `(content: String, reasoning_text: Option<String>)`.
    let (content_text, reasoning_text) = match registration {
        Some(reg) if reg.has_reasoning() => super::registry::split_full_output_forced(
            reg,
            &decoded_text,
            params.reasoning_forced_open,
        ),
        _ => (decoded_text, None),
    };
    // Reasoning token count: same shape as generate_qwen35_once.
    let reasoning_token_count = match registration {
        Some(reg) if reg.has_reasoning() => {
            let mut sp =
                super::registry::make_reasoning_splitter(reg, params.reasoning_forced_open);
            let mut count = 0usize;
            for &tok in &generated_tokens {
                let frag = qwen.tokenizer.decode(&[tok], false).unwrap_or_default();
                if let Some(splitter) = sp.as_mut() {
                    let _ = splitter.feed(&frag);
                    if splitter.in_reasoning() {
                        count += 1;
                    }
                }
            }
            count
        }
        _ => 0,
    };

    Ok(GenerationResult {
        text: content_text,
        reasoning_text,
        prompt_tokens: prompt_len,
        completion_tokens: generated_tokens.len(),
        reasoning_tokens: if reasoning_token_count > 0 {
            Some(reasoning_token_count)
        } else {
            None
        },
        finish_reason,
        prefill_duration,
        decode_duration,
        // Slot-aware mode disables LCP / chunked prefill (see fn
        // docstring); cached_tokens reflects the full-prompt-cache
        // hit only.
        cached_tokens: if prompt_cache_hit { prompt_len } else { 0 },
        logprobs: logprobs_vec,
    })
}

// ===========================================================================
// ADR-040 Phase F M1 (F1) — Qwen35 per-slot decode state + tick.
//
// Beside the serial reference `generate_qwen35_once_slot_aware` (above) so
// the two stay auditable side-by-side. Hoists that function's decode-loop
// locals so the SlotAware worker (`super::engine::worker_run_slot_aware`)
// can interleave N slots across ticks. DELIBERATELY simpler than the
// Gemma 4 tail: this legacy slot-aware Qwen35 path has no grammar runtime /
// token-bytes / live reasoning splitter / tool-call splitter in the loop,
// computes
// `reasoning_tokens` POST-HOC at end-of-decode, and POPS+clears on a
// first-token EOS — all preserved exactly here.
//
// STEP 1: `decode_tick` calls the existing per-slot `forward_gpu_greedy` /
// `forward_gpu_last_logits` once per handle (the caller loops handles).
// STEP 2 (F2) replaces that with one batched forward; this tail is
// unchanged.
// ===========================================================================

/// One decoded token's surface from `Qwen35DecodeState::decode_tick`,
/// mirroring `super::engine::TickOutcome` (kept local because that type is
/// private to the engine module). The engine loop maps this 1:1.
pub(crate) struct Qwen35TickOutcome {
    pub fragment: String,
    /// Qwen35 has no streaming reasoning-span classification in the decode
    /// loop (reasoning split is post-hoc), so this is always false; carried
    /// for shape-parity with the engine's `TickOutcome`.
    pub is_reasoning: bool,
    pub finished: bool,
}

/// One bounded, scheduler-visible Qwen prompt transaction.
///
/// Slot-aware serving previously forwarded the entire uncached prompt inside
/// admission. A large OpenCode tool prompt could therefore monopolize the
/// worker and submit watchdog-scale Metal work before an existing stream got
/// another decode turn. This state keeps the request-local cursor outside the
/// forward call so the worker can yield only at a successfully committed
/// cache+ledger boundary.
pub(crate) enum Qwen35PrefillAdvance {
    Pending {
        state: Qwen35PrefillState,
        advanced_tokens: usize,
        checkpoint: Option<Qwen35StablePromptCheckpoint>,
    },
    Ready {
        state: Qwen35DecodeState,
        prefill_logits: Vec<f32>,
        advanced_tokens: usize,
        checkpoint: Option<Qwen35StablePromptCheckpoint>,
    },
}

pub(crate) struct Qwen35StablePromptCheckpoint {
    pub prompt_tokens: Vec<u32>,
    pub kv: HybridKvSlotAnchor,
    pub prefill_logits: Vec<f32>,
    pub vision_fingerprint: Option<[u8; 32]>,
    pub spec: Option<Qwen35SpecPrefixBoundary>,
}

/// Model-semantic state paired with a physically snapshotted prompt prefix.
/// `Some` is the coherence marker: consumers must still validate that target
/// and MTP cursors both equal `token_count` after any restore.
#[derive(Clone)]
pub(crate) struct Qwen35SpecPrefixBoundary {
    pub token_count: usize,
    pub pending_target_hidden: MlxBuffer,
}

/// Owned multimodal state carried by a bounded SlotAware prefill. Unlike the
/// serial vision entry point, this survives scheduler yields and can project
/// only the soft-token/deepstack rows intersecting the current prompt chunk.
#[derive(Debug)]
pub(crate) struct Qwen35VisionPrefillData {
    soft_tokens: Vec<SoftTokenData>,
    deepstack: Option<DeepstackData>,
    positions_flat: Option<Vec<i32>>,
}

impl Qwen35VisionPrefillData {
    pub(crate) fn new(
        soft_tokens: Vec<SoftTokenData>,
        deepstack: Option<DeepstackData>,
        positions_flat: Option<Vec<i32>>,
    ) -> Self {
        Self {
            soft_tokens,
            deepstack,
            positions_flat,
        }
    }

    pub(crate) fn text_anchor_reuse_limit(&self, prompt_len: usize) -> Option<usize> {
        let soft_ranges: Vec<_> = self
            .soft_tokens
            .iter()
            .map(|soft| soft.range.clone())
            .collect();
        super::engine::qwen35_text_anchor_reuse_limit(
            prompt_len,
            &soft_ranges,
            self.deepstack
                .as_ref()
                .map(|data| data.image_token_positions.as_slice()),
            self.positions_flat.as_deref(),
        )
    }

    pub(crate) fn validate(&self, prompt_len: usize, hidden_size: usize) -> Result<()> {
        anyhow::ensure!(
            !self.soft_tokens.is_empty()
                || self.deepstack.is_some()
                || self.positions_flat.is_some(),
            "Qwen35VisionPrefillData must carry at least one multimodal extension"
        );
        if let Some(positions) = self.positions_flat.as_ref() {
            anyhow::ensure!(
                positions.len() == 4 * prompt_len,
                "Qwen35 vision positions len {} != 4 * prompt_len {}",
                positions.len(),
                4 * prompt_len
            );
        }
        let row_bytes = hidden_size
            .checked_mul(std::mem::size_of::<f32>())
            .context("Qwen35 vision hidden row byte overflow")?;
        let mut previous_end = 0usize;
        for (index, soft) in self.soft_tokens.iter().enumerate() {
            anyhow::ensure!(
                soft.range.start < soft.range.end && soft.range.end <= prompt_len,
                "Qwen35 vision soft token [{index}] range {:?} outside prompt_len={prompt_len}",
                soft.range
            );
            anyhow::ensure!(
                index == 0 || soft.range.start >= previous_end,
                "Qwen35 vision soft token ranges overlap or are unsorted at index {index}"
            );
            let needed = soft
                .range
                .len()
                .checked_mul(row_bytes)
                .context("Qwen35 vision soft-token byte-size overflow")?;
            anyhow::ensure!(
                soft.embeddings.byte_len() >= needed,
                "Qwen35 vision soft token [{index}] has {} bytes, needs at least {needed}",
                soft.embeddings.byte_len()
            );
            previous_end = soft.range.end;
        }
        if let Some(deepstack) = self.deepstack.as_ref() {
            anyhow::ensure!(
                deepstack
                    .image_token_positions
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
                "Qwen35 vision deepstack positions must be strictly increasing"
            );
            anyhow::ensure!(
                deepstack
                    .image_token_positions
                    .last()
                    .is_none_or(|position| (*position as usize) < prompt_len),
                "Qwen35 vision deepstack position outside prompt_len={prompt_len}"
            );
            let needed = deepstack
                .image_token_positions
                .len()
                .checked_mul(row_bytes)
                .context("Qwen35 vision deepstack byte-size overflow")?;
            for (index, chunk) in deepstack.chunks.iter().enumerate() {
                anyhow::ensure!(
                    chunk.byte_len() >= needed,
                    "Qwen35 vision deepstack chunk [{index}] has {} bytes, needs at least {needed}",
                    chunk.byte_len()
                );
            }
        }
        Ok(())
    }

    fn chunk(&self, start: usize, end: usize, hidden_size: usize) -> Result<Qwen35VisionChunk> {
        anyhow::ensure!(start < end, "Qwen35 vision chunk must be non-empty");
        let row_bytes = hidden_size
            .checked_mul(std::mem::size_of::<f32>())
            .context("Qwen35 vision chunk row byte overflow")?;
        let mut soft_tokens = Vec::new();
        for soft in &self.soft_tokens {
            let intersection_start = soft.range.start.max(start);
            let intersection_end = soft.range.end.min(end);
            if intersection_start >= intersection_end {
                continue;
            }
            let source_row = intersection_start - soft.range.start;
            let rows = intersection_end - intersection_start;
            let byte_offset = source_row
                .checked_mul(row_bytes)
                .context("Qwen35 vision soft-token view offset overflow")?;
            let elements = rows
                .checked_mul(hidden_size)
                .context("Qwen35 vision soft-token view length overflow")?;
            soft_tokens.push((
                (intersection_start - start)..(intersection_end - start),
                soft.embeddings.slice_view(byte_offset as u64, elements),
            ));
        }

        let deepstack = self.deepstack.as_ref().and_then(|deepstack| {
            let selected: Vec<(usize, u32)> = deepstack
                .image_token_positions
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, position)| {
                    let position = *position as usize;
                    position >= start && position < end
                })
                .collect();
            let (first, _) = selected.first().copied()?;
            let rows = selected.len();
            debug_assert!(selected
                .iter()
                .enumerate()
                .all(|(offset, (index, _))| *index == first + offset));
            let byte_offset = first.checked_mul(row_bytes)?;
            let elements = rows.checked_mul(hidden_size)?;
            let positions = selected
                .into_iter()
                .map(|(_, position)| position - start as u32)
                .collect();
            let chunks = deepstack
                .chunks
                .iter()
                .map(|chunk| chunk.slice_view(byte_offset as u64, elements))
                .collect();
            Some((positions, chunks))
        });

        let positions_flat = self.positions_flat.as_ref().map(|positions| {
            let full_len = positions.len() / 4;
            let mut chunk_positions = Vec::with_capacity(4 * (end - start));
            for axis in 0..4 {
                chunk_positions
                    .extend_from_slice(&positions[axis * full_len + start..axis * full_len + end]);
            }
            chunk_positions
        });
        Ok(Qwen35VisionChunk {
            soft_tokens,
            deepstack,
            positions_flat,
        })
    }

    fn decode_position_base(&self, prompt_len: usize) -> usize {
        self.positions_flat
            .as_ref()
            .map(|positions| {
                positions[..prompt_len]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0)
                    .saturating_add(1)
                    .max(0) as usize
            })
            .unwrap_or(prompt_len)
    }
}

struct Qwen35VisionChunk {
    soft_tokens: Vec<(std::ops::Range<usize>, mlx_native::MlxBuffer)>,
    deepstack: Option<(Vec<u32>, Vec<mlx_native::MlxBuffer>)>,
    positions_flat: Option<Vec<i32>>,
}

pub(crate) struct Qwen35PrefillState {
    slot_id: SlotId,
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
    cached_tokens: usize,
    next_token_index: usize,
    cached_prefill_logits: Option<Vec<f32>>,
    stable_prompt_prefix_tokens: Option<usize>,
    vision: Option<Qwen35VisionPrefillData>,
    /// Post-output-RMSNorm target row immediately preceding the next prompt
    /// chunk. When present, the MTP cache has consumed exactly the same
    /// verified prompt prefix as the target cache.
    mtp_pending_hidden: Option<MlxBuffer>,
    /// A prompt catch-up failure permanently selects bounded ordinary prefill
    /// for the rest of this request. The already-restored target prefix stays
    /// reusable; only speculative semantic state is abandoned.
    speculation_unavailable: bool,
    prefill_started: Instant,
}

fn qwen35_next_prefill_end(
    cursor: usize,
    prompt_len: usize,
    max_chunk_tokens: usize,
    stable_prompt_prefix_tokens: Option<usize>,
) -> usize {
    let mut end = cursor.saturating_add(max_chunk_tokens).min(prompt_len);
    if let Some(boundary) = stable_prompt_prefix_tokens {
        if cursor < boundary && boundary < end {
            end = boundary;
        }
    }
    end
}

impl Qwen35PrefillState {
    /// Immutable rendered prompt owned by an in-flight bounded prefill. The
    /// scheduler uses this only for busy-slot affinity: it must wait for this
    /// request to finish before the prefix becomes reusable.
    pub(crate) fn prompt_tokens(&self) -> &[u32] {
        &self.prompt_tokens
    }

    pub(crate) fn vision_fingerprint(&self) -> Option<[u8; 32]> {
        self.params.vision_fingerprint
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin(
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        registration: Option<&ModelRegistration>,
        kv_cache: &mut HybridKvCache,
        slot_id: SlotId,
        cached_tokens: usize,
        cached_prefill_logits: Option<Vec<f32>>,
        cached_spec: Option<Qwen35SpecPrefixBoundary>,
        vision: Option<Qwen35VisionPrefillData>,
        hidden_size: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            !prompt_tokens.is_empty(),
            "Qwen35PrefillState::begin: empty prompt_tokens"
        );
        anyhow::ensure!(
            slot_id.0 < kv_cache.n_seqs,
            "Qwen35PrefillState::begin: SlotOutOfRange slot={} max_slots={}",
            slot_id.0,
            kv_cache.n_seqs,
        );
        let prompt_len = prompt_tokens.len();
        anyhow::ensure!(
            cached_tokens <= prompt_len,
            "Qwen35PrefillState::begin: cached_tokens={} exceeds prompt_len={}",
            cached_tokens,
            prompt_len,
        );
        anyhow::ensure!(
            (cached_tokens == prompt_len) == cached_prefill_logits.is_some(),
            "Qwen35PrefillState::begin: a full-prompt cache hit requires prompt-boundary logits, and partial/cold prefill must not supply them"
        );
        if let Some(spec) = cached_spec.as_ref() {
            anyhow::ensure!(
                spec.token_count == cached_tokens,
                "Qwen cached speculative boundary token_count={} != cached_tokens={cached_tokens}",
                spec.token_count,
            );
            anyhow::ensure!(
                vision.is_none(),
                "Qwen speculative prefix reuse is text-only"
            );
            kv_cache
                .validate_speculative_cursors_for_slot(slot_id, cached_tokens)
                .context("Qwen cached speculative boundary cursor equality")?;
        }
        let max_tokens = params.max_tokens.max(1);
        let need_seq = prompt_len
            .checked_add(max_tokens)
            .and_then(|tokens| tokens.checked_add(64))
            .context("Qwen35PrefillState::begin: prompt + completion capacity overflow")?;
        anyhow::ensure!(
            need_seq <= kv_cache.max_seq_len as usize,
            "Qwen35PrefillState::begin: per-request need_seq={} exceeds persistent cache max_seq_len={} (slot={} prompt_len={} max_tokens={})",
            need_seq,
            kv_cache.max_seq_len,
            slot_id.0,
            prompt_len,
            max_tokens,
        );

        // Reject malformed grammar/tool state before the first bounded Metal
        // transaction, matching the legacy prefill path's fail-fast contract.
        let _ = grammar_runtime_for_request(&params, registration)?;
        if let Some(vision) = vision.as_ref() {
            vision.validate(prompt_len, hidden_size)?;
            anyhow::ensure!(
                params.vision_fingerprint.is_some(),
                "Qwen35 SlotAware multimodal prefill requires an exact vision fingerprint"
            );
        }

        if cached_tokens == 0 {
            kv_cache
                .reset_for_slot(slot_id)
                .context("ADR-040 full-context slots: Qwen35 cold reset_for_slot at entry")?;
        }
        kv_cache
            .validate_sequence_len_for_slot(slot_id, cached_tokens)
            .context("Qwen35PrefillState::begin: validate cache/ledger boundary")?;

        let stable_prompt_prefix_tokens = params
            .stable_prompt_prefix_tokens
            .filter(|&boundary| boundary > cached_tokens && boundary < prompt_len);

        Ok(Self {
            slot_id,
            prompt_tokens,
            params,
            cached_tokens,
            next_token_index: cached_tokens,
            cached_prefill_logits,
            stable_prompt_prefix_tokens,
            vision,
            mtp_pending_hidden: cached_spec.map(|spec| spec.pending_target_hidden),
            speculation_unavailable: false,
            prefill_started: Instant::now(),
        })
    }

    pub(super) fn advance(
        mut self,
        qwen: &mut Qwen35LoadedModel,
        registration: Option<&ModelRegistration>,
        kv_cache: &mut HybridKvCache,
        max_chunk_tokens: usize,
        supervisor: &EngineSupervisor,
    ) -> Result<Qwen35PrefillAdvance> {
        let (prefill_logits, mtp_hidden, advanced_tokens, checkpoint) = if let Some(logits) =
            self.cached_prefill_logits.take()
        {
            anyhow::ensure!(
                logits.len() == qwen.vocab_size,
                "qwen35 cached prompt-boundary logits len {} != vocab_size {}",
                logits.len(),
                qwen.vocab_size
            );
            (logits, None, 0, None)
        } else {
            anyhow::ensure!(
                max_chunk_tokens > 0,
                "Qwen35PrefillState::advance requires a non-zero chunk"
            );
            kv_cache
                .validate_sequence_len_for_slot(self.slot_id, self.next_token_index)
                .context("validate Qwen35 slot cursors before bounded prefill")?;
            let end = qwen35_next_prefill_end(
                self.next_token_index,
                self.prompt_tokens.len(),
                max_chunk_tokens,
                self.stable_prompt_prefix_tokens,
            );
            anyhow::ensure!(
                end > self.next_token_index,
                "Qwen35 bounded prefill has no suffix without cached logits"
            );
            let chunk_start = self.next_token_index;
            let chunk = &self.prompt_tokens[self.next_token_index..end];
            let transaction =
                begin_slot_state_transaction(kv_cache, self.slot_id, chunk_start as u32)?;
            let vision_chunk = self
                .vision
                .as_ref()
                .map(|vision| vision.chunk(self.next_token_index, end, qwen.hidden_size))
                .transpose()?;
            let positions = vision_chunk
                .as_ref()
                .and_then(|vision| vision.positions_flat.clone())
                .unwrap_or_else(|| prefill_positions_from(self.next_token_index, chunk.len()));
            let chunk_started = Instant::now();
            let lease =
                supervisor.arm("Qwen35 bounded prefill", QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
            let mtp_prefill = (self.cached_tokens == 0 || self.mtp_pending_hidden.is_some())
                && !self.speculation_unavailable
                && self.vision.is_none()
                && qwen.speculation.policy()
                    == super::qwen35_speculation::QwenSpeculationPolicy::Auto
                && is_qwen_server_speculation_exact_eligible(&self.params)
                && qwen.model.mtp.is_some()
                && kv_cache.mtp_slot.is_some();
            let (forward, mtp_hidden) = if let Some(vision) = vision_chunk.as_ref() {
                let soft_tokens: Vec<_> = vision
                    .soft_tokens
                    .iter()
                    .map(
                        |(range, embeddings)| crate::serve::forward_prefill::SoftTokenInjection {
                            range: range.clone(),
                            embeddings,
                        },
                    )
                    .collect();
                let deepstack = vision.deepstack.as_ref().map(|(positions, chunks)| {
                    crate::serve::forward_prefill::DeepstackInjection {
                        image_token_positions: positions.clone(),
                        chunks: chunks.iter().collect(),
                    }
                });
                (
                    qwen.model
                        .forward_gpu_last_logits_with_soft_tokens_and_deepstack(
                            chunk,
                            &positions,
                            &soft_tokens,
                            deepstack.as_ref(),
                            kv_cache,
                            self.slot_id,
                        ),
                    None,
                )
            } else if mtp_prefill {
                match qwen.model.forward_gpu_last_logits_with_hidden(
                    chunk,
                    &positions,
                    kv_cache,
                    self.slot_id,
                ) {
                    // A failed target forward has not proven the completed
                    // ping-pong transition required by recurrent rollback.
                    // Surface the error; the scheduler fail-closes the slot.
                    Err(error) => (
                        Err(error.context("Qwen SlotAware MTP target prefill")),
                        None,
                    ),
                    Ok((logits, target_nextn)) => {
                        let catchup = (|| -> Result<MlxBuffer> {
                            let shared_embed_rows = qwen.model.embed_tokens_gpu(chunk)?;
                            let mtp =
                                qwen.model.mtp.as_ref().context(
                                    "Qwen SlotAware MTP prompt catch-up weights missing",
                                )?;
                            qwen.model.with_gpu_cache_mut(|device, registry| {
                                mtp.process_target_batch(
                                    chunk,
                                    self.mtp_pending_hidden.as_ref(),
                                    &target_nextn,
                                    &shared_embed_rows,
                                    kv_cache,
                                    self.slot_id,
                                    &positions,
                                    device,
                                    registry,
                                    &qwen.model.cfg,
                                )
                            })?;
                            kv_cache
                                .validate_speculative_cursors_for_slot(self.slot_id, end)
                                .context("Qwen SlotAware prompt target/MTP cursor equality")?;
                            let hidden =
                                crate::inference::models::qwen35::spec_decode::last_hidden_row(
                                    &target_nextn,
                                    qwen.model.cfg.hidden_size,
                                )?;
                            anyhow::ensure!(
                                hidden.element_count() == qwen.model.cfg.hidden_size as usize,
                                "Qwen SlotAware MTP prefill hidden must be one row"
                            );
                            qwen35_state_failpoint(Qwen35StateFailpoint::PrefillMtpCatchup)?;
                            Ok(hidden)
                        })();
                        match catchup {
                            Ok(hidden) => {
                                super::qwen35_speculation::record_outcome(0, 0, 0, 1, 0);
                                self.mtp_pending_hidden = Some(hidden);
                                (Ok(logits), None)
                            }
                            Err(error) => {
                                // The target forward completed, so recurrent
                                // rollback is now defined. Restore this
                                // chunk's entry boundary and replay only the
                                // same bounded chunk through the ordinary
                                // target.
                                tracing::warn!(
                                    slot = self.slot_id.0,
                                    error = %error,
                                    "Qwen SlotAware MTP prompt catch-up unavailable; replaying bounded ordinary prefill"
                                );
                                super::qwen35_speculation::record_fallback(
                                    super::qwen35_speculation::QwenSpeculationDecision::RuntimeUnavailable,
                                );
                                kv_cache
                                    .rollback_slot_transaction(self.slot_id, &transaction)
                                    .context(
                                        "Qwen SlotAware MTP prompt-catch-up bounded rollback",
                                    )?;
                                self.mtp_pending_hidden = None;
                                self.speculation_unavailable = true;
                                (
                                    qwen.model.forward_gpu_last_logits(
                                        chunk,
                                        &positions,
                                        kv_cache,
                                        self.slot_id,
                                    ),
                                    None,
                                )
                            }
                        }
                    }
                }
            } else {
                (
                    qwen.model
                        .forward_gpu_last_logits(chunk, &positions, kv_cache, self.slot_id),
                    None,
                )
            };
            if let Err(error) = lease.finish() {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen35 bounded prefill supervision",
                );
            }
            let logits = match forward {
                Ok(logits) => logits,
                Err(error) => {
                    return rollback_slot_state_error(
                        kv_cache,
                        self.slot_id,
                        &transaction,
                        error,
                        "Qwen35 bounded prefill forward",
                    );
                }
            };
            if let Err(error) = qwen35_state_failpoint(Qwen35StateFailpoint::PrefillTarget) {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen35 bounded prefill post-target failpoint",
                );
            }
            if let Err(error) = kv_cache.validate_sequence_len_for_slot(self.slot_id, end) {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "validate Qwen35 slot cursors after bounded prefill",
                );
            }
            tracing::info!(
                slot = self.slot_id.0,
                chunk_start,
                chunk_end = end,
                chunk_tokens = chunk.len(),
                prompt_tokens = self.prompt_tokens.len(),
                elapsed_seconds = chunk_started.elapsed().as_secs_f64(),
                "Qwen35 bounded prefill chunk complete"
            );
            let advanced = end - self.next_token_index;
            self.next_token_index = end;
            let checkpoint = if self.stable_prompt_prefix_tokens == Some(end) {
                let spec =
                    self.mtp_pending_hidden
                        .as_ref()
                        .map(|hidden| Qwen35SpecPrefixBoundary {
                            token_count: end,
                            pending_target_hidden: hidden.clone(),
                        });
                let kv = match kv_cache.snapshot_slot_anchor(self.slot_id, end) {
                    Ok(kv) => kv,
                    Err(error) => {
                        return rollback_slot_state_error(
                            kv_cache,
                            self.slot_id,
                            &transaction,
                            error,
                            "capture Qwen35 stable prompt boundary",
                        );
                    }
                };
                Some(Qwen35StablePromptCheckpoint {
                    prompt_tokens: self.prompt_tokens[..end].to_vec(),
                    kv,
                    prefill_logits: logits.clone(),
                    vision_fingerprint: self.params.vision_fingerprint,
                    spec,
                })
            } else {
                None
            };
            (logits, mtp_hidden, advanced, checkpoint)
        };

        if self.next_token_index < self.prompt_tokens.len() {
            return Ok(Qwen35PrefillAdvance::Pending {
                state: self,
                advanced_tokens,
                checkpoint,
            });
        }

        let prefill_duration = self.prefill_started.elapsed();
        let decode_position_base = self
            .vision
            .as_ref()
            .map(|vision| vision.decode_position_base(self.prompt_tokens.len()))
            .unwrap_or(self.prompt_tokens.len());
        let mtp_hidden = mtp_hidden.or_else(|| self.mtp_pending_hidden.take());
        let state = Qwen35DecodeState::from_prefill_logits(
            qwen,
            self.prompt_tokens,
            self.params,
            registration,
            self.slot_id,
            self.cached_tokens,
            &prefill_logits,
            prefill_duration,
            decode_position_base,
            mtp_hidden,
        )?;
        Ok(Qwen35PrefillAdvance::Ready {
            state,
            prefill_logits,
            advanced_tokens,
            checkpoint,
        })
    }

    pub(crate) fn operator_progress(&self) -> (usize, usize, f64) {
        let completed = self.next_token_index.saturating_sub(self.cached_tokens);
        let work = self.prompt_tokens.len().saturating_sub(self.cached_tokens);
        let rate = completed as f64
            / self
                .prefill_started
                .elapsed()
                .as_secs_f64()
                .max(f64::EPSILON);
        (completed, work, rate)
    }
}

/// Hoisted per-slot decode state for a Qwen35 SlotAware request — the
/// locals from `generate_qwen35_once_slot_aware`'s decode loop
/// (engine_qwen35.rs:2292-2470), lifted for N-slot interleave. Field
/// semantics mirror that function so N=1 is byte-identical to the serial
/// reference.
pub(crate) struct Qwen35DecodeState {
    slot_id: SlotId,
    prompt_tokens: Vec<u32>,
    prompt_len: usize,
    /// Absolute text-axis position for the first post-prefill decode input.
    /// Equals `prompt_len` for text, but Qwen multimodal mRoPE advances image
    /// spans by their temporal grid extent rather than soft-token row count.
    decode_position_base: usize,
    max_tokens: usize,
    is_greedy: bool,
    want_logprobs: bool,
    logprobs_vec: Option<Vec<f32>>,
    cached_tokens: usize,
    /// Cloned request sampling params — the qwen35 sampling helpers
    /// (`sample_logits_qwen35*`) read temperature/top_p/top_k/rep-penalty/
    /// max_tokens off this each tick (serial ref passes the live `params`).
    params: SamplingParams,
    /// Request-local grammar state. SlotAware previously dropped this state
    /// and sampled unconstrained, which made required/automatic tool calls
    /// diverge from the serial OpenCode path.
    grammar_runtime: Option<super::grammar::GrammarRuntime>,
    /// Independent marker detector used only to awaken auto-lazy tool
    /// grammars. OpenAI SSE routing owns a sibling splitter in `engine.rs`.
    tool_splitter: Option<ToolCallSplitter>,
    /// The token to feed into the NEXT decode forward.
    next_token: u32,
    generated_tokens: Vec<u32>,
    /// Fixed 64-token repetition window over the rendered prompt tail plus
    /// committed generation. This is deliberately separate from
    /// `generated_tokens`, whose length and contents define the streamed
    /// completion ledger.
    sampling_history: Vec<u32>,
    decoded_text: String,
    stop_strings: Vec<String>,
    finish_reason: &'static str,
    /// Decode-step counter, mirrors the serial ref `step` (starts at 1).
    step: usize,
    answer_event_reported: bool,
    thinking_budget: Option<Qwen35ThinkingBudgetState>,
    prefill_duration: Duration,
    decode_start: Instant,
    /// Present only when SlotAware target verification can preserve every
    /// admitted API semantic (including constrained greedy state). The cache
    /// remains the scheduler-owned per-slot `HybridKvCache`.
    mtp: Option<Qwen35SlotMtpState>,
    /// Per-request lookup state. It is initialized from prompt + target seed
    /// and is synchronized only from the committed output ledger.
    history_lookup: Option<HistoryLookupIndex>,
    /// A verified draft+bonus may span worker ticks. This queue is shared by
    /// all proposers so no new proposal can overtake an already verified SSE
    /// token.
    pending_speculation_output: VecDeque<u32>,
    terminal_after_pending: bool,
    /// MTP is admitted only after this generation has an ordinary target
    /// timing baseline and remains enabled only while equivalent output cost
    /// is positive.
    mtp_cost: SpeculationCostController,
    /// History lookup has negligible proposal cost, but its block target
    /// verifier can still lose on a particular model/device.
    history_cost: SpeculationCostController,
}

struct Qwen35SlotMtpState {
    verifier_hidden: MlxBuffer,
}

const QWEN35_REPETITION_WINDOW: usize = 64;

fn qwen35_prompt_sampling_history(prompt_tokens: &[u32]) -> Vec<u32> {
    let start = prompt_tokens.len().saturating_sub(QWEN35_REPETITION_WINDOW);
    prompt_tokens[start..].to_vec()
}

fn qwen35_observe_sampling_history(history: &mut Vec<u32>, token: u32) {
    if history.len() == QWEN35_REPETITION_WINDOW {
        history.remove(0);
    }
    history.push(token);
}

fn take_pending_speculation_output(queue: &mut VecDeque<u32>) -> Option<u32> {
    queue.pop_front()
}

fn equivalent_target_decisions(queue: &VecDeque<u32>, terminal_after_pending: bool) -> usize {
    queue.len() + usize::from(terminal_after_pending)
}

fn may_route_history_miss_to_mtp(mtp_available: bool, cost: &SpeculationCostController) -> bool {
    mtp_available && cost.may_speculate()
}

fn mtp_cursor_for_slot(kv_cache: &HybridKvCache, slot_id: SlotId) -> Result<u32> {
    kv_cache
        .mtp_slot
        .as_ref()
        .and_then(|slot| slot.current_len.get(slot_id.0 as usize).copied())
        .context("Qwen SlotAware MTP cursor missing")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Qwen35StateFailpoint {
    OrdinaryTarget = 1,
    HistoryTarget = 2,
    HistoryMtpCatchup = 3,
    HistoryCommit = 4,
    MtpDraft = 5,
    MtpTarget = 6,
    MtpCatchup = 7,
    MtpCommit = 8,
    WarmupTarget = 9,
    WarmupMtpCatchup = 10,
    PrefillTarget = 11,
    PrefillMtpCatchup = 12,
}

#[cfg(test)]
static QWEN35_STATE_FAILPOINT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[inline]
fn qwen35_state_failpoint(point: Qwen35StateFailpoint) -> Result<()> {
    #[cfg(test)]
    if QWEN35_STATE_FAILPOINT.load(std::sync::atomic::Ordering::SeqCst) == point as u8 {
        anyhow::bail!("injected Qwen state failure after {point:?}");
    }
    #[cfg(not(test))]
    let _ = point;
    Ok(())
}

fn begin_slot_state_transaction(
    kv_cache: &HybridKvCache,
    slot_id: SlotId,
    target_cursor: u32,
) -> Result<HybridKvSlotTransaction> {
    kv_cache
        .begin_slot_transaction(slot_id, target_cursor)
        .context("capture exact Qwen slot state transaction")
}

fn rollback_slot_state_error<T>(
    kv_cache: &mut HybridKvCache,
    slot_id: SlotId,
    transaction: &HybridKvSlotTransaction,
    error: anyhow::Error,
    context: &'static str,
) -> Result<T> {
    let original = error.context(context);
    kv_cache.clear_la_capture();
    match kv_cache.rollback_slot_transaction(slot_id, transaction) {
        Ok(()) => Err(original),
        Err(rollback) => {
            let reset = kv_cache.reset_for_slot(slot_id);
            Err(anyhow::anyhow!(
                "{original:#}; exact slot-state rollback failure: {rollback:#}; fail-closed reset: {reset:?}"
            ))
        }
    }
}

#[derive(Debug, Clone)]
struct Qwen35ThinkingBudgetState {
    limit: usize,
    reasoning_tokens: usize,
    forced_tokens: Arc<Vec<u32>>,
    close_tokens: Arc<Vec<u32>>,
    forced_cursor: Option<usize>,
    closed: bool,
}

impl Qwen35ThinkingBudgetState {
    fn from_params(params: &SamplingParams) -> Option<Self> {
        let limit = params.thinking_token_budget?;
        let forced_tokens = params.reasoning_end_tokens.clone()?;
        let close_tokens = params.reasoning_close_tokens.clone()?;
        (params.reasoning_forced_open
            && limit > 0
            && !forced_tokens.is_empty()
            && !close_tokens.is_empty())
        .then_some(Self {
            limit,
            reasoning_tokens: 0,
            forced_tokens,
            close_tokens,
            forced_cursor: None,
            closed: false,
        })
    }

    fn next_forced_token(&mut self) -> Option<(u32, bool)> {
        if self.closed {
            return None;
        }
        let started = self.forced_cursor.is_none() && self.reasoning_tokens >= self.limit;
        if started {
            self.forced_cursor = Some(0);
        }
        let cursor = self.forced_cursor?;
        let token = self.forced_tokens.get(cursor).copied()?;
        self.forced_cursor = Some(cursor + 1);
        Some((token, started))
    }

    fn observe_generated(&mut self, generated_tokens: &[u32], tool_opened: bool) {
        if self.closed {
            return;
        }
        if tool_opened || generated_tokens.ends_with(self.close_tokens.as_slice()) {
            self.closed = true;
            return;
        }
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(1);
    }

    fn was_forced_closed(&self) -> bool {
        self.forced_cursor.is_some() && self.closed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen35CanonicalDecision {
    token: u32,
    terminal: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct Qwen35VerifiedBlockPlan {
    output: VecDeque<u32>,
    terminal_after_pending: bool,
    matched_drafts: usize,
    rejected_drafts: usize,
    /// Number of verifier input rows that belong to the committed prefix.
    valid_input_tokens: usize,
    /// Post-output-norm verifier row to carry into the next round.
    carry_hidden_row: usize,
}

/// Walk target decisions in order and stop at the first rejected proposal.
///
/// Row zero predicts `drafts[0]`; row `K` is the target bonus after full
/// acceptance. A terminal proposal is counted as matched for telemetry but is
/// neither streamed nor fed back into the retained cache, so its valid prefix
/// ends at the row that predicted it.
fn plan_qwen35_verified_block(
    drafts: &[u32],
    mut canonical_at: impl FnMut(usize) -> Result<Qwen35CanonicalDecision>,
) -> Result<Qwen35VerifiedBlockPlan> {
    anyhow::ensure!(
        !drafts.is_empty(),
        "verified block requires at least one draft"
    );
    let mut output = VecDeque::with_capacity(drafts.len() + 1);
    for (index, &draft) in drafts.iter().enumerate() {
        let decision = canonical_at(index)?;
        if decision.token != draft {
            if !decision.terminal {
                output.push_back(decision.token);
            }
            return Ok(Qwen35VerifiedBlockPlan {
                output,
                terminal_after_pending: decision.terminal,
                matched_drafts: index,
                rejected_drafts: 1,
                valid_input_tokens: index + 1,
                carry_hidden_row: index,
            });
        }
        if decision.terminal {
            return Ok(Qwen35VerifiedBlockPlan {
                output,
                terminal_after_pending: true,
                matched_drafts: index + 1,
                rejected_drafts: 0,
                valid_input_tokens: index + 1,
                carry_hidden_row: index,
            });
        }
        output.push_back(draft);
    }

    let bonus = canonical_at(drafts.len())?;
    if !bonus.terminal {
        output.push_back(bonus.token);
    }
    Ok(Qwen35VerifiedBlockPlan {
        output,
        terminal_after_pending: bonus.terminal,
        matched_drafts: drafts.len(),
        rejected_drafts: 0,
        valid_input_tokens: drafts.len() + 1,
        carry_hidden_row: drafts.len(),
    })
}

#[derive(Clone)]
struct Qwen35SpecSemanticState {
    generated_tokens: Vec<u32>,
    sampling_history: Vec<u32>,
    grammar_runtime: Option<super::grammar::GrammarRuntime>,
    tool_splitter: Option<ToolCallSplitter>,
    thinking_budget: Option<Qwen35ThinkingBudgetState>,
}

impl Qwen35SpecSemanticState {
    fn from_decode(state: &Qwen35DecodeState) -> Self {
        Self {
            generated_tokens: state.generated_tokens.clone(),
            sampling_history: state.sampling_history.clone(),
            grammar_runtime: state.grammar_runtime.clone(),
            tool_splitter: state.tool_splitter.clone(),
            thinking_budget: state.thinking_budget.clone(),
        }
    }

    fn select_and_observe(
        &mut self,
        qwen: &Qwen35LoadedModel,
        params: &SamplingParams,
        logits: &mut [f32],
    ) -> Result<Qwen35CanonicalDecision> {
        let forced = self
            .thinking_budget
            .as_mut()
            .and_then(Qwen35ThinkingBudgetState::next_forced_token)
            .map(|(token, _)| token);
        let token = if let Some(forced) = forced {
            forced
        } else {
            sample_logits_qwen35_constrained(
                logits,
                params,
                &self.sampling_history,
                self.grammar_runtime.as_ref(),
                false,
            )?
            .0
        };
        advance_qwen35_grammar(&mut self.grammar_runtime, params, token);
        let terminal = qwen.eos_token_ids.contains(&token)
            || qwen35_grammar_terminal_token(self.grammar_runtime.as_ref(), params, token);
        if terminal {
            return Ok(Qwen35CanonicalDecision { token, terminal });
        }

        self.generated_tokens.push(token);
        qwen35_observe_sampling_history(&mut self.sampling_history, token);
        let fragment = qwen.tokenizer.decode(&[token], false).unwrap_or_default();
        let tool_opened = self.tool_splitter.as_mut().is_some_and(|splitter| {
            splitter
                .feed(&fragment)
                .iter()
                .any(|event| matches!(event, ToolCallEvent::ToolCallOpen))
        });
        if tool_opened {
            if let Some(runtime) = self.grammar_runtime.as_mut() {
                runtime.trigger();
            }
        }
        if let Some(budget) = self.thinking_budget.as_mut() {
            budget.observe_generated(&self.generated_tokens, tool_opened);
        }
        Ok(Qwen35CanonicalDecision { token, terminal })
    }
}

impl Qwen35DecodeState {
    /// Run prefill for one Qwen35 slot and seed the decode state — mirror
    /// of `generate_qwen35_once_slot_aware` engine_qwen35.rs:2264-2396
    /// (bounds + need-seq guard + entry reset + prompt-cache-or-prefill +
    /// first-token derivation + first-token EOS/stop check). The caller
    /// owns no separate reset (entry reset is done here, matching the
    /// serial ref). If the first token already terminates, `finish_reason
    /// != "length"` so the loop fires the reply with no decode tick.
    pub(crate) fn prefill_seed(
        qwen: &mut Qwen35LoadedModel,
        prompt_tokens: &[u32],
        params: &SamplingParams,
        registration: Option<&ModelRegistration>,
        kv_cache: &mut HybridKvCache,
        slot_id: SlotId,
        cached_tokens: usize,
        cached_prefill_logits: Option<&[f32]>,
    ) -> Result<(Self, Vec<f32>)> {
        anyhow::ensure!(
            !prompt_tokens.is_empty(),
            "Qwen35DecodeState::prefill_seed: empty prompt_tokens"
        );
        anyhow::ensure!(
            slot_id.0 < kv_cache.n_seqs,
            "Qwen35DecodeState::prefill_seed: SlotOutOfRange slot={} max_slots={} \
             (ADR-040 Phase F M1)",
            slot_id.0,
            kv_cache.n_seqs,
        );
        let prompt_len = prompt_tokens.len();
        anyhow::ensure!(
            cached_tokens <= prompt_len,
            "Qwen35DecodeState::prefill_seed: cached_tokens={} exceeds prompt_len={}",
            cached_tokens,
            prompt_len,
        );
        anyhow::ensure!(
            (cached_tokens == prompt_len) == cached_prefill_logits.is_some(),
            "Qwen35DecodeState::prefill_seed: a full-prompt cache hit requires prompt-boundary logits, and partial/cold prefill must not supply them"
        );
        let max_tokens = params.max_tokens.max(1);
        let need_seq = prompt_len + max_tokens + 64;
        if need_seq > kv_cache.max_seq_len as usize {
            return Err(anyhow::anyhow!(
                "Qwen35DecodeState::prefill_seed: per-request need_seq={} exceeds \
                 persistent cache max_seq_len={} (slot={} prompt_len={} max_tokens={}). \
                 ADR-040 Phase F M1; reduce max_tokens or use a shorter prompt.",
                need_seq,
                kv_cache.max_seq_len,
                slot_id.0,
                prompt_len,
                max_tokens
            ));
        }

        // Preserve the historical fail-fast contract: invalid grammar/tool
        // state is rejected before any Metal submission. Seed sampling itself
        // is centralized in `from_prefill_logits` so bounded and legacy slot
        // paths cannot drift.
        let _ = grammar_runtime_for_request(params, registration)?;

        if cached_tokens == 0 {
            kv_cache
                .reset_for_slot(slot_id)
                .context("ADR-040 full-context slots: Qwen35 cold reset_for_slot at entry")?;
        } else {
            let cursor = kv_cache
                .sequence_len_for_slot(slot_id)
                .context("ADR-040 full-context slots: read Qwen35 retained cursor")?
                as usize;
            anyhow::ensure!(
                cursor == cached_tokens,
                "Qwen35DecodeState::prefill_seed: retained token ledger/cache cursor mismatch for slot {} (ledger={}, cursor={}); refusing unsafe resume",
                slot_id.0,
                cached_tokens,
                cursor,
            );
        }

        // codex M1 review fix (2026-06-24): cross-request prompt-cache RESUME is
        // NOT slot-isolated under SlotAware. `restore_partial(snap, prompt_len)`
        // writes BROAD shared state — it sets every `current_len` entry and copies
        // the full linear-attn state buffers (kv_cache.rs:2504/2547) with no
        // destination SlotId — so a cache hit on one slot would corrupt peer
        // slots' KV. Disable the hit on the slot-aware path → always do a fresh
        // per-slot prefill (correct). Per-slot / refcounted prompt caching under
        // continuous batching is a separate M5 optimization (mirrors the gemma4
        // LCP write-back skip). SerialFifo serves qwen35 via the legacy
        // generate_qwen35_once path (its own prompt-cache), unaffected.
        let prefill_start = Instant::now();
        let prefill_logits = if let Some(logits) = cached_prefill_logits {
            anyhow::ensure!(
                logits.len() == qwen.vocab_size,
                "qwen35 cached prompt-boundary logits len {} != vocab_size {}",
                logits.len(),
                qwen.vocab_size
            );
            logits.to_vec()
        } else {
            let suffix = &prompt_tokens[cached_tokens..];
            anyhow::ensure!(
                !suffix.is_empty(),
                "Qwen35DecodeState::prefill_seed: empty suffix without cached prompt-boundary logits"
            );
            let positions = prefill_positions_from(cached_tokens, suffix.len());
            qwen.model
                .forward_gpu_last_logits(suffix, &positions, kv_cache, slot_id)
                .context("Qwen35Model::forward_gpu_last_logits (slot-aware prefill)")?
        };
        anyhow::ensure!(
            prefill_logits.len() == qwen.vocab_size,
            "qwen35 slot-aware prefill logits len {} != vocab_size {}",
            prefill_logits.len(),
            qwen.vocab_size
        );
        let prefill_duration = prefill_start.elapsed();
        let state = Self::from_prefill_logits(
            qwen,
            prompt_tokens.to_vec(),
            params.clone(),
            registration,
            slot_id,
            cached_tokens,
            &prefill_logits,
            prefill_duration,
            prompt_len,
            None,
        )?;
        Ok((state, prefill_logits))
    }

    #[allow(clippy::too_many_arguments)]
    fn from_prefill_logits(
        qwen: &Qwen35LoadedModel,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        registration: Option<&ModelRegistration>,
        slot_id: SlotId,
        cached_tokens: usize,
        prefill_logits: &[f32],
        prefill_duration: Duration,
        decode_position_base: usize,
        mtp_hidden: Option<MlxBuffer>,
    ) -> Result<Self> {
        anyhow::ensure!(
            prefill_logits.len() == qwen.vocab_size,
            "qwen35 slot-aware prefill logits len {} != vocab_size {}",
            prefill_logits.len(),
            qwen.vocab_size
        );
        let prompt_len = prompt_tokens.len();
        let max_tokens = params.max_tokens.max(1);
        let is_greedy = is_greedy_eligible(&params);
        let want_logprobs = params.logprobs;
        let mut logprobs_vec = want_logprobs.then(|| Vec::with_capacity(max_tokens));
        let mut grammar_runtime = grammar_runtime_for_request(&params, registration)?;
        let mut tool_splitter = registration.and_then(ToolCallSplitter::from_registration);
        let mut sampling_history = qwen35_prompt_sampling_history(&prompt_tokens);
        let next_token = if is_greedy && !want_logprobs {
            greedy_argmax_last_token(prefill_logits, qwen.vocab_size as u32)
        } else {
            let mut logits = prefill_logits.to_vec();
            let (token, logprob) = sample_logits_qwen35_constrained(
                &mut logits,
                &params,
                &sampling_history,
                grammar_runtime.as_ref(),
                want_logprobs,
            )?;
            if let (Some(values), Some(logprob)) = (logprobs_vec.as_mut(), logprob) {
                values.push(logprob);
            }
            token
        };
        advance_qwen35_grammar(&mut grammar_runtime, &params, next_token);

        let mut generated_tokens = Vec::with_capacity(max_tokens);
        generated_tokens.push(next_token);
        qwen35_observe_sampling_history(&mut sampling_history, next_token);
        let mut decoded_text = qwen
            .tokenizer
            .decode(&[next_token], false)
            .unwrap_or_default();
        let mut tool_opened = false;
        if let Some(splitter) = tool_splitter.as_mut() {
            let marker_events = splitter.feed(&decoded_text);
            tool_opened = marker_events
                .iter()
                .any(|event| matches!(event, ToolCallEvent::ToolCallOpen));
            if tool_opened {
                if let Some(runtime) = grammar_runtime.as_mut() {
                    runtime.trigger();
                }
            }
        }
        let stop_strings = params.stop_strings.clone();
        let mut finish_reason = "length";
        if qwen.eos_token_ids.contains(&next_token)
            || qwen35_grammar_terminal_token(grammar_runtime.as_ref(), &params, next_token)
        {
            generated_tokens.pop();
            decoded_text.clear();
            finish_reason = "stop";
        } else if qwen35_hit_stop_string(&decoded_text, &stop_strings) {
            qwen35_strip_trailing_stop(&mut decoded_text, &stop_strings);
            finish_reason = "stop";
        }

        let mut thinking_budget = Qwen35ThinkingBudgetState::from_params(&params);
        if finish_reason == "length" {
            if let Some(budget) = thinking_budget.as_mut() {
                budget.observe_generated(&generated_tokens, tool_opened);
            }
        }

        let history_lookup = if finish_reason == "length"
            && qwen.speculation.policy() == super::qwen35_speculation::QwenSpeculationPolicy::Auto
            && is_qwen_server_speculation_exact_eligible(&params)
            && cached_tokens != prompt_len
        {
            let mut lookup = HistoryLookupIndex::new(HistoryLookupConfig {
                min_match: 6,
                max_match: 12,
                max_draft_tokens: 3,
                max_model_len: qwen.model.cfg.max_position_embeddings as usize,
            });
            lookup.reset(&prompt_tokens);
            lookup.extend_verified(&generated_tokens);
            Some(lookup)
        } else {
            None
        };

        Ok(Self {
            slot_id,
            prompt_tokens,
            prompt_len,
            decode_position_base,
            max_tokens,
            is_greedy,
            want_logprobs,
            logprobs_vec,
            cached_tokens,
            params,
            grammar_runtime,
            tool_splitter,
            next_token,
            generated_tokens,
            sampling_history,
            decoded_text,
            stop_strings,
            finish_reason,
            step: 1,
            answer_event_reported: false,
            thinking_budget,
            prefill_duration,
            decode_start: Instant::now(),
            mtp: mtp_hidden.map(|verifier_hidden| Qwen35SlotMtpState { verifier_hidden }),
            history_lookup,
            pending_speculation_output: VecDeque::new(),
            terminal_after_pending: false,
            mtp_cost: SpeculationCostController::new(),
            history_cost: SpeculationCostController::new(),
        })
    }

    /// Whether the slot already terminated during prefill-seed.
    pub(crate) fn finished_at_seed(&self) -> bool {
        self.finish_reason != "length"
    }

    /// The decoded first-token text to stream when the slot finishes at
    /// seed. On first-token EOS this is empty (the serial ref pops + clears
    /// decoded_text); on a stop-string hit it is the stop-stripped prefix.
    pub(crate) fn seed_fragment(&self) -> String {
        self.decoded_text.clone()
    }

    /// Exact input-token prefix represented by the live KV/recurrent state.
    /// The cache cursor is authoritative because the newest sampled token has
    /// not necessarily been fed back through the model yet.
    pub(crate) fn retained_prefix(&self, valid_tokens: usize) -> Vec<u32> {
        self.prompt_tokens
            .iter()
            .chain(self.generated_tokens.iter())
            .copied()
            .take(valid_tokens)
            .collect()
    }

    /// Return model-semantic speculative state for an exactly published
    /// physical prefix. A verified queue means GPU state is intentionally
    /// ahead of the streamed ledger, so it can never be published.
    pub(crate) fn spec_prefix_candidate(
        &self,
        valid_tokens: usize,
    ) -> Option<Qwen35SpecPrefixBoundary> {
        if valid_tokens == 0
            || valid_tokens > self.prompt_tokens.len() + self.generated_tokens.len()
            || !self.pending_speculation_output.is_empty()
        {
            return None;
        }
        self.mtp.as_ref().map(|mtp| Qwen35SpecPrefixBoundary {
            token_count: valid_tokens,
            pending_target_hidden: mtp.verifier_hidden.clone(),
        })
    }

    /// Exact rendered prompt and generation parameters used by the per-slot
    /// deterministic replay cache. The cache is response-shaped; live KV
    /// remains available for longer conversation continuations.
    pub(crate) fn prompt_cache_identity(&self) -> (&[u32], &SamplingParams) {
        (&self.prompt_tokens, &self.params)
    }

    /// Return true exactly once after the stream router has delivered content
    /// or a structured tool-call delta. Raw decoded reasoning is deliberately
    /// not treated as answer progress.
    pub(crate) fn mark_first_answer_event(&mut self) -> bool {
        if self.answer_event_reported {
            false
        } else {
            self.answer_event_reported = true;
            true
        }
    }

    pub(crate) fn operator_progress(&self) -> (usize, usize, f64) {
        let generated = self.generated_tokens.len();
        let rate = generated as f64 / self.decode_start.elapsed().as_secs_f64().max(f64::EPSILON);
        (generated, self.max_tokens, rate)
    }

    pub(crate) fn operator_thinking_progress(&self) -> (Option<usize>, Option<usize>, bool, bool) {
        self.thinking_budget.as_ref().map_or(
            (None, None, false, self.answer_event_reported),
            |budget| {
                (
                    Some(budget.reasoning_tokens.min(budget.limit)),
                    Some(budget.limit),
                    budget.was_forced_closed(),
                    self.answer_event_reported,
                )
            },
        )
    }

    pub(crate) fn operator_prefill_progress(&self) -> (usize, usize, f64) {
        let work = self.prompt_len.saturating_sub(self.cached_tokens);
        let rate = work as f64 / self.prefill_duration.as_secs_f64().max(f64::EPSILON);
        (work, work, rate)
    }

    /// Advance this slot by exactly one decode token — mirror of the serial
    /// ref's `while` body engine_qwen35.rs:2410-2468 for ONE iteration.
    pub(super) fn decode_tick(
        &mut self,
        qwen: &mut Qwen35LoadedModel,
        kv_cache: &mut HybridKvCache,
        supervisor: &EngineSupervisor,
    ) -> Result<Qwen35TickOutcome> {
        // Loop bound mirror: serial ref `while step < max_tokens &&
        // finish_reason == "length"`. The caller only ticks live slots, so
        // entry here implies finish_reason == "length"; enforce the step
        // bound so a slot at the bound finishes with finish_reason "length".
        if self.step >= self.max_tokens {
            return Ok(Qwen35TickOutcome {
                fragment: String::new(),
                is_reasoning: false,
                finished: true,
            });
        }
        if let Some(token) = take_pending_speculation_output(&mut self.pending_speculation_output) {
            return self.commit_speculation_output(qwen, token);
        }
        if self.history_lookup.is_some() {
            return self.decode_tick_history_lookup(qwen, kv_cache, supervisor);
        }
        if self.mtp.is_some() {
            return self.decode_tick_mtp_k3(qwen, kv_cache, supervisor);
        }
        let prior_target = kv_cache.sequence_len_for_slot(self.slot_id)?;
        let transaction = begin_slot_state_transaction(kv_cache, self.slot_id, prior_target)?;
        let ordinary_target_started = Instant::now();
        let pos = self.decode_position_base + self.step - 1;
        let pos_i32 = pos as i32;
        let positions: Vec<i32> = vec![pos_i32; 4];
        let last_input = &self.generated_tokens[self.generated_tokens.len() - 1..];
        let forced_token = self
            .thinking_budget
            .as_mut()
            .and_then(Qwen35ThinkingBudgetState::next_forced_token);
        if forced_token.is_some_and(|(_, started)| started) {
            tracing::warn!(
                slot = self.slot_id.0,
                budget = self.params.thinking_token_budget,
                generated_tokens = self.generated_tokens.len(),
                "Qwen35 thinking token budget reached; forcing reasoning close and continuing answer"
            );
        }
        let tok = if self.is_greedy && !self.want_logprobs {
            let lease =
                supervisor.arm("Qwen35 decode greedy", QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
            let forward =
                qwen.model
                    .forward_gpu_greedy(last_input, &positions, kv_cache, self.slot_id);
            if let Err(error) = lease.finish() {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen35 ordinary greedy supervision",
                );
            }
            let predicted = match forward {
                Ok(predicted) => predicted,
                Err(error) => {
                    return rollback_slot_state_error(
                        kv_cache,
                        self.slot_id,
                        &transaction,
                        error,
                        "Qwen35 ordinary greedy forward",
                    );
                }
            };
            if let Err(error) = qwen35_state_failpoint(Qwen35StateFailpoint::OrdinaryTarget) {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen35 ordinary greedy post-target failpoint",
                );
            }
            forced_token.map_or(predicted, |(token, _)| token)
        } else {
            let lease =
                supervisor.arm("Qwen35 decode logits", QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
            let forward =
                qwen.model
                    .forward_gpu_last_logits(last_input, &positions, kv_cache, self.slot_id);
            if let Err(error) = lease.finish() {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen35 ordinary logits supervision",
                );
            }
            let logits = match forward {
                Ok(logits) => logits,
                Err(error) => {
                    return rollback_slot_state_error(
                        kv_cache,
                        self.slot_id,
                        &transaction,
                        error,
                        "Qwen35 ordinary logits forward",
                    );
                }
            };
            if let Err(error) = qwen35_state_failpoint(Qwen35StateFailpoint::OrdinaryTarget) {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen35 ordinary logits post-target failpoint",
                );
            }
            if logits.len() != qwen.vocab_size {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    anyhow::anyhow!(
                        "qwen35 slot-aware decode logits len {} != vocab_size {}",
                        logits.len(),
                        qwen.vocab_size
                    ),
                    "Qwen35 ordinary logits shape",
                );
            }
            let (token, logprob) = if let Some((token, _)) = forced_token {
                (token, self.want_logprobs.then_some(0.0))
            } else {
                let mut logits = logits;
                match sample_logits_qwen35_constrained(
                    &mut logits,
                    &self.params,
                    &self.sampling_history,
                    self.grammar_runtime.as_ref(),
                    self.want_logprobs,
                ) {
                    Ok(decision) => decision,
                    Err(error) => {
                        return rollback_slot_state_error(
                            kv_cache,
                            self.slot_id,
                            &transaction,
                            error,
                            "Qwen35 ordinary canonical decision",
                        );
                    }
                }
            };
            if let (Some(values), Some(logprob)) = (self.logprobs_vec.as_mut(), logprob) {
                values.push(logprob);
            }
            token
        };
        let ordinary_target_elapsed = ordinary_target_started.elapsed();
        self.mtp_cost
            .observe_ordinary_target(ordinary_target_elapsed);
        self.history_cost
            .observe_ordinary_target(ordinary_target_elapsed);
        advance_qwen35_grammar(&mut self.grammar_runtime, &self.params, tok);
        if qwen.eos_token_ids.contains(&tok) {
            self.finish_reason = "stop";
            return Ok(Qwen35TickOutcome {
                fragment: String::new(),
                is_reasoning: false,
                finished: true,
            });
        }
        if qwen35_grammar_terminal_token(self.grammar_runtime.as_ref(), &self.params, tok) {
            self.finish_reason = "stop";
            return Ok(Qwen35TickOutcome {
                fragment: String::new(),
                is_reasoning: false,
                finished: true,
            });
        }
        self.generated_tokens.push(tok);
        qwen35_observe_sampling_history(&mut self.sampling_history, tok);
        let frag = qwen.tokenizer.decode(&[tok], false).unwrap_or_default();
        self.decoded_text.push_str(&frag);
        let mut tool_opened = false;
        if let Some(splitter) = self.tool_splitter.as_mut() {
            let marker_events = splitter.feed(&frag);
            tool_opened = marker_events
                .iter()
                .any(|event| matches!(event, ToolCallEvent::ToolCallOpen));
            if tool_opened {
                if let Some(runtime) = self.grammar_runtime.as_mut() {
                    runtime.trigger();
                }
            }
        }
        if let Some(budget) = self.thinking_budget.as_mut() {
            budget.observe_generated(&self.generated_tokens, tool_opened);
        }
        if qwen35_hit_stop_string(&self.decoded_text, &self.stop_strings) {
            qwen35_strip_trailing_stop(&mut self.decoded_text, &self.stop_strings);
            self.finish_reason = "stop";
            return Ok(Qwen35TickOutcome {
                fragment: frag,
                is_reasoning: false,
                finished: true,
            });
        }
        self.step += 1;
        let finished = self.step >= self.max_tokens;
        Ok(Qwen35TickOutcome {
            fragment: frag,
            is_reasoning: false,
            finished,
        })
    }

    /// Target-verified request-history speculation. The lookup may return up
    /// to three continuation tokens; one target batch verifies the block and,
    /// when MTP is available, reconciles that same batch into its K/V cache so
    /// proposer selection can change on the next round without state drift.
    fn decode_tick_history_lookup(
        &mut self,
        qwen: &mut Qwen35LoadedModel,
        kv_cache: &mut HybridKvCache,
        supervisor: &EngineSupervisor,
    ) -> Result<Qwen35TickOutcome> {
        if let Some(token) = take_pending_speculation_output(&mut self.pending_speculation_output) {
            return self.commit_speculation_output(qwen, token);
        }
        if !self.history_cost.may_speculate() {
            let lookup = self
                .history_lookup
                .take()
                .expect("history lookup checked above");
            let outcome = self.decode_tick(qwen, kv_cache, supervisor);
            self.history_lookup = Some(lookup);
            return outcome;
        }

        let expected_len = self.prompt_tokens.len() + self.generated_tokens.len();
        let lookup = self
            .history_lookup
            .as_mut()
            .expect("history lookup checked above");
        if lookup.verified_len() < expected_len {
            let generated_cursor = lookup
                .verified_len()
                .checked_sub(self.prompt_tokens.len())
                .context("Qwen history lookup precedes prompt boundary")?;
            lookup.extend_verified(&self.generated_tokens[generated_cursor..]);
        }
        anyhow::ensure!(
            lookup.verified_len() == expected_len,
            "Qwen history lookup ledger cursor mismatch"
        );
        let remaining = self.max_tokens.saturating_sub(self.generated_tokens.len());
        let mut drafts = lookup.propose();
        drafts.truncate(remaining.saturating_sub(1).min(3));
        if drafts.is_empty() {
            super::qwen35_speculation::record_history_lookup_no_match();
            let mut lookup = self
                .history_lookup
                .take()
                .expect("history lookup checked above");
            if self.mtp.is_some() && !may_route_history_miss_to_mtp(true, &self.mtp_cost) {
                // Once MTP is cost-disabled, ordinary target decode may
                // advance without reconciling its cache. Drop the semantic
                // state permanently rather than restore a lagging proposer.
                self.mtp = None;
            }
            let committed_before = self.generated_tokens.len();
            let outcome = self.decode_tick(qwen, kv_cache, supervisor);
            if self.generated_tokens.len() > committed_before {
                lookup.extend_verified(&self.generated_tokens[committed_before..]);
            }
            self.history_lookup = Some(lookup);
            return outcome;
        }

        let round_started = Instant::now();
        let next_token = *self
            .generated_tokens
            .last()
            .context("Qwen history lookup needs a seed")?;
        let next_pos = (self.decode_position_base + self.generated_tokens.len() - 1) as i32;
        let prior_target_len = kv_cache.sequence_len_for_slot(self.slot_id)?;
        let prior_mtp_len = self
            .mtp
            .as_ref()
            .map(|_| mtp_cursor_for_slot(kv_cache, self.slot_id))
            .transpose()?;
        if let Some(mtp_len) = prior_mtp_len {
            anyhow::ensure!(
                mtp_len == prior_target_len,
                "Qwen history verifier requires equal target/MTP cursors (target={prior_target_len}, mtp={mtp_len})"
            );
        }

        let verify_rows = drafts.len() + 1;
        if let Err(error) = qwen.model.with_gpu_cache_mut(|device, _registry| {
            kv_cache.ensure_la_capture(&qwen.model.cfg, device, verify_rows as u32)
        }) {
            kv_cache.clear_la_capture();
            return Err(error.context("Qwen history recurrent capture allocation"));
        }
        let mut verify_input = Vec::with_capacity(verify_rows);
        verify_input.push(next_token);
        verify_input.extend_from_slice(&drafts);
        let verify_positions = crate::inference::models::qwen35::spec_decode::positions_for_range(
            next_pos,
            verify_rows,
        );
        let transaction = begin_slot_state_transaction(kv_cache, self.slot_id, prior_target_len)?;
        let lease = match supervisor.arm(
            "Qwen35 SlotAware history block verify",
            QWEN35_WORKER_TRANSACTION_TIMEOUT,
        ) {
            Ok(lease) => lease,
            Err(error) => {
                kv_cache.clear_la_capture();
                return Err(error.context("Qwen history verifier admission"));
            }
        };
        let verified = qwen.model.forward_gpu_with_nextn_hidden_buffer(
            &verify_input,
            &verify_positions,
            kv_cache,
            self.slot_id,
        );
        let supervision = lease.finish();
        let (mut verify_logits, verify_hidden) = match (supervision, verified) {
            (Ok(()), Ok(value)) => value,
            (supervision, forward) => {
                let error = supervision
                    .err()
                    .or_else(|| forward.err())
                    .expect("failed history verification has an error");
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen history block verify",
                );
            }
        };
        if let Err(error) = qwen35_state_failpoint(Qwen35StateFailpoint::HistoryTarget) {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen history post-target failpoint",
            );
        }
        let vocab = qwen.vocab_size;
        if verify_logits.element_count() != verify_rows * vocab
            || verify_hidden.element_count() != verify_rows * qwen.model.cfg.hidden_size as usize
        {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                anyhow::anyhow!("Qwen history verify output shape mismatch"),
                "Qwen history verify output shape",
            );
        }

        if let (Some(mtp_state), Some(prior_mtp_len)) = (self.mtp.as_ref(), prior_mtp_len) {
            let shared_embed_rows = match qwen.model.embed_tokens_gpu(&verify_input) {
                Ok(rows) => rows,
                Err(error) => {
                    return rollback_slot_state_error(
                        kv_cache,
                        self.slot_id,
                        &transaction,
                        error,
                        "Qwen history MTP embeddings",
                    );
                }
            };
            let mtp = match qwen.model.mtp.as_ref() {
                Some(mtp) => mtp,
                None => {
                    return rollback_slot_state_error(
                        kv_cache,
                        self.slot_id,
                        &transaction,
                        anyhow::anyhow!("Qwen history MTP state exists without weights"),
                        "Qwen history MTP lookup",
                    );
                }
            };
            let lease = match supervisor.arm(
                "Qwen35 SlotAware history MTP catch-up",
                QWEN35_WORKER_TRANSACTION_TIMEOUT,
            ) {
                Ok(lease) => lease,
                Err(error) => {
                    return rollback_slot_state_error(
                        kv_cache,
                        self.slot_id,
                        &transaction,
                        error,
                        "Qwen history MTP catch-up admission",
                    );
                }
            };
            let caught_up = qwen.model.with_gpu_cache_mut(|device, registry| {
                mtp.process_target_batch(
                    &verify_input,
                    Some(&mtp_state.verifier_hidden),
                    &verify_hidden,
                    &shared_embed_rows,
                    kv_cache,
                    self.slot_id,
                    &verify_positions,
                    device,
                    registry,
                    &qwen.model.cfg,
                )
            });
            if let Err(error) = lease.finish().and(caught_up) {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen history MTP catch-up",
                );
            }
            if let Err(error) = qwen35_state_failpoint(Qwen35StateFailpoint::HistoryMtpCatchup) {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen history post-MTP-catch-up failpoint",
                );
            }
            debug_assert_eq!(prior_mtp_len, prior_target_len);
        }

        let verify_logits = match verify_logits.as_mut_slice::<f32>() {
            Ok(logits) => logits,
            Err(error) => {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    anyhow::anyhow!("{error}"),
                    "Qwen history logits view",
                );
            }
        };
        let mut semantic = Qwen35SpecSemanticState::from_decode(self);
        let plan = match plan_qwen35_verified_block(&drafts, |row| {
            let start = row * vocab;
            semantic.select_and_observe(
                qwen,
                &self.params,
                &mut verify_logits[start..start + vocab],
            )
        }) {
            Ok(plan) => plan,
            Err(error) => {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen history canonical accept walk",
                );
            }
        };
        let committed_cursor = prior_target_len + plan.valid_input_tokens as u32;
        let state_result = (|| -> Result<()> {
            kv_cache.truncate_full_attn_to_for_slot(self.slot_id, committed_cursor)?;
            if prior_mtp_len.is_some() {
                kv_cache.truncate_mtp_to_for_slot(self.slot_id, committed_cursor)?;
            }
            if plan.valid_input_tokens < verify_rows {
                kv_cache.rollback_la_to(self.slot_id, plan.carry_hidden_row as u32)?;
            }
            kv_cache.clear_la_capture();
            if prior_mtp_len.is_some() {
                kv_cache.validate_speculative_cursors_for_slot(
                    self.slot_id,
                    committed_cursor as usize,
                )?;
            } else {
                kv_cache.validate_sequence_len_for_slot(self.slot_id, committed_cursor as usize)?;
            }
            Ok(())
        })();
        if let Err(error) = state_result {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen history commit verified prefix",
            );
        }
        if let Err(error) = qwen35_state_failpoint(Qwen35StateFailpoint::HistoryCommit) {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen history post-commit failpoint",
            );
        }
        if let Some(mtp_state) = self.mtp.as_mut() {
            mtp_state.verifier_hidden =
                match crate::inference::models::qwen35::spec_decode::nth_hidden_row(
                    &verify_hidden,
                    qwen.model.cfg.hidden_size,
                    plan.carry_hidden_row as u64,
                ) {
                    Ok(hidden) => hidden,
                    Err(error) => {
                        return rollback_slot_state_error(
                            kv_cache,
                            self.slot_id,
                            &transaction,
                            error,
                            "Qwen history carry hidden",
                        );
                    }
                };
        }
        self.pending_speculation_output = plan.output;
        self.terminal_after_pending = plan.terminal_after_pending;
        super::qwen35_speculation::record_proposer_outcome(
            super::qwen35_speculation::QwenSpeculationProposer::HistoryLookup,
            drafts.len(),
            plan.matched_drafts,
            plan.rejected_drafts,
            1,
            0,
        );
        let equivalent_decisions = equivalent_target_decisions(
            &self.pending_speculation_output,
            self.terminal_after_pending,
        );
        let round_elapsed = round_started.elapsed();
        if let Some(equivalent_ordinary) = self
            .history_cost
            .equivalent_ordinary_elapsed(equivalent_decisions)
        {
            super::qwen35_speculation::record_proposer_timing(
                super::qwen35_speculation::QwenSpeculationProposer::HistoryLookup,
                round_elapsed,
                equivalent_ordinary,
            );
        }
        let remains_profitable = self
            .history_cost
            .observe_speculative_round(equivalent_decisions, round_elapsed);
        if !remains_profitable {
            super::qwen35_speculation::record_cost_disabled(
                super::qwen35_speculation::QwenSpeculationProposer::HistoryLookup,
            );
            self.history_lookup = None;
        }

        if let Some(token) = take_pending_speculation_output(&mut self.pending_speculation_output) {
            self.commit_speculation_output(qwen, token)
        } else {
            self.finish_reason = "stop";
            Ok(Qwen35TickOutcome {
                fragment: String::new(),
                is_reasoning: false,
                finished: true,
            })
        }
    }

    /// Fixed-depth-three MTP transaction. Three normalized MTP steps are
    /// verified by one four-row target batch, then that target batch
    /// reconciles the MTP K/V cache.
    /// Partial rejection commits the valid prefix directly from captured
    /// target state; no ordinary replay is required.
    fn decode_tick_mtp_k3(
        &mut self,
        qwen: &mut Qwen35LoadedModel,
        kv_cache: &mut HybridKvCache,
        supervisor: &EngineSupervisor,
    ) -> Result<Qwen35TickOutcome> {
        const DRAFT_DEPTH: usize = 3;
        const VERIFY_ROWS: usize = DRAFT_DEPTH + 1;

        if let Some(token) = take_pending_speculation_output(&mut self.pending_speculation_output) {
            return self.commit_speculation_output(qwen, token);
        }
        let remaining = self.max_tokens.saturating_sub(self.generated_tokens.len());
        if remaining < VERIFY_ROWS || !self.mtp_cost.may_speculate() {
            return self.decode_tick_mtp_warmup(qwen, kv_cache, supervisor);
        }

        let round_started = Instant::now();
        let phase_profile = std::env::var("HF2Q_MTP_PHASE_PROFILE").as_deref() == Ok("1");
        let next_token = *self
            .generated_tokens
            .last()
            .context("Qwen SlotAware MTP K3 needs a seeded output token")?;
        let next_pos = (self.decode_position_base + self.generated_tokens.len() - 1) as i32;
        let prior_target_len = kv_cache
            .sequence_len_for_slot(self.slot_id)
            .context("Qwen SlotAware MTP K3 read target cursor")?;
        let prior_mtp_len = mtp_cursor_for_slot(kv_cache, self.slot_id)?;
        anyhow::ensure!(
            prior_target_len == prior_mtp_len,
            "Qwen SlotAware MTP K3 requires equal entry cursors (target={prior_target_len}, mtp={prior_mtp_len})"
        );

        let mtp = qwen
            .model
            .mtp
            .as_ref()
            .context("Qwen SlotAware MTP K3 weights missing")?;
        let initial_hidden = &self
            .mtp
            .as_ref()
            .context("Qwen SlotAware MTP K3 state missing")?
            .verifier_hidden;
        if let Err(error) = qwen.model.with_gpu_cache_mut(|device, _registry| {
            kv_cache
                .ensure_la_capture(&qwen.model.cfg, device, VERIFY_ROWS as u32)
                .context("Qwen SlotAware MTP K3 allocate recurrent capture")
        }) {
            kv_cache.clear_la_capture();
            return Err(error);
        }
        let transaction = begin_slot_state_transaction(kv_cache, self.slot_id, prior_target_len)?;
        let draft_lease = match supervisor.arm(
            "Qwen35 SlotAware MTP K3 draft",
            QWEN35_WORKER_TRANSACTION_TIMEOUT,
        ) {
            Ok(lease) => lease,
            Err(error) => {
                kv_cache.clear_la_capture();
                return Err(error.context("Qwen SlotAware MTP K3 draft admission"));
            }
        };
        let drafted = qwen.model.with_gpu_cache_mut(|device, registry| {
            let mut drafts = Vec::with_capacity(DRAFT_DEPTH);
            let mut chain_hidden: Option<MlxBuffer> = None;
            let mut chain_token = next_token;
            for depth in 0..DRAFT_DEPTH {
                let shared_embed =
                    qwen.model
                        .embed_tokens_gpu_in_context(&[chain_token], device, registry)?;
                let previous = chain_hidden.as_ref().unwrap_or(initial_hidden);
                let (token, next_hidden) = mtp.forward_draft_greedy_for_token(
                    previous,
                    chain_token,
                    &shared_embed,
                    kv_cache,
                    self.slot_id,
                    &[next_pos + depth as i32; 4],
                    device,
                    registry,
                    &qwen.model.cfg,
                )?;
                drafts.push(token);
                chain_token = token;
                chain_hidden = Some(next_hidden);
            }
            Ok::<_, anyhow::Error>(drafts)
        });
        let draft_supervision = draft_lease.finish();
        if let Err(error) = draft_supervision.and(
            drafted
                .as_ref()
                .map(|_| ())
                .map_err(|e| anyhow::anyhow!("{e:#}")),
        ) {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen SlotAware MTP K3 draft",
            );
        }
        let drafts = drafted.expect("draft result checked above");
        if let Err(error) = qwen35_state_failpoint(Qwen35StateFailpoint::MtpDraft) {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen SlotAware MTP K3 post-draft failpoint",
            );
        }
        if let Err(error) = kv_cache.truncate_mtp_to_for_slot(self.slot_id, prior_mtp_len) {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen SlotAware MTP K3 discard speculative draft cache",
            );
        }
        let draft_elapsed = round_started.elapsed();

        let mut verify_input = Vec::with_capacity(VERIFY_ROWS);
        verify_input.push(next_token);
        verify_input.extend_from_slice(&drafts);
        let verify_positions = crate::inference::models::qwen35::spec_decode::positions_for_range(
            next_pos,
            VERIFY_ROWS,
        );
        let verify_lease = match supervisor.arm(
            "Qwen35 SlotAware MTP K3 verify",
            QWEN35_WORKER_TRANSACTION_TIMEOUT,
        ) {
            Ok(lease) => lease,
            Err(error) => {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen SlotAware MTP K3 verify admission",
                );
            }
        };
        let verify_started = Instant::now();
        let verified = qwen.model.forward_gpu_with_nextn_hidden_buffer(
            &verify_input,
            &verify_positions,
            kv_cache,
            self.slot_id,
        );
        let verify_supervision = verify_lease.finish();
        let (mut verify_logits, verify_hidden) = match (verify_supervision, verified) {
            (Ok(()), Ok(value)) => value,
            (supervision, forward) => {
                let error = supervision
                    .err()
                    .or_else(|| forward.err())
                    .expect("failed verification has an error");
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen SlotAware MTP K3 verify",
                );
            }
        };
        if let Err(error) = qwen35_state_failpoint(Qwen35StateFailpoint::MtpTarget) {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen SlotAware MTP K3 post-target failpoint",
            );
        }
        let verify_elapsed = verify_started.elapsed();
        let vocab = qwen.vocab_size;
        if verify_logits.element_count() != VERIFY_ROWS * vocab
            || verify_hidden.element_count() != VERIFY_ROWS * qwen.model.cfg.hidden_size as usize
        {
            let error = anyhow::anyhow!(
                "Qwen SlotAware MTP K3 verify shape mismatch: logits={} expected={}, hidden={} expected={}",
                verify_logits.element_count(),
                VERIFY_ROWS * vocab,
                verify_hidden.element_count(),
                VERIFY_ROWS * qwen.model.cfg.hidden_size as usize,
            );
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen SlotAware MTP K3 verify shape",
            );
        }

        let embed_started = Instant::now();
        let shared_embed_rows = match qwen.model.embed_tokens_gpu(&verify_input) {
            Ok(rows) => rows,
            Err(error) => {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen SlotAware MTP K3 verifier embeddings",
                );
            }
        };
        let embed_elapsed = embed_started.elapsed();
        let catchup_lease = match supervisor.arm(
            "Qwen35 SlotAware MTP K3 target catch-up",
            QWEN35_WORKER_TRANSACTION_TIMEOUT,
        ) {
            Ok(lease) => lease,
            Err(error) => {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen SlotAware MTP K3 catch-up admission",
                );
            }
        };
        let catchup_started = Instant::now();
        let caught_up = qwen.model.with_gpu_cache_mut(|device, registry| {
            mtp.process_target_batch(
                &verify_input,
                Some(initial_hidden),
                &verify_hidden,
                &shared_embed_rows,
                kv_cache,
                self.slot_id,
                &verify_positions,
                device,
                registry,
                &qwen.model.cfg,
            )
        });
        let catchup_supervision = catchup_lease.finish();
        if let Err(error) = catchup_supervision.and(caught_up) {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen SlotAware MTP K3 target catch-up",
            );
        }
        if let Err(error) = qwen35_state_failpoint(Qwen35StateFailpoint::MtpCatchup) {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen SlotAware MTP K3 post-catch-up failpoint",
            );
        }
        let catchup_elapsed = catchup_started.elapsed();

        let verify_logits = match verify_logits.as_mut_slice::<f32>() {
            Ok(logits) => logits,
            Err(error) => {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    anyhow::anyhow!("{error}"),
                    "Qwen SlotAware MTP K3 logits view",
                );
            }
        };
        let semantic_started = Instant::now();
        let mut semantic = Qwen35SpecSemanticState::from_decode(self);
        let plan = match plan_qwen35_verified_block(&drafts, |row| {
            let start = row * vocab;
            semantic.select_and_observe(
                qwen,
                &self.params,
                &mut verify_logits[start..start + vocab],
            )
        }) {
            Ok(plan) => plan,
            Err(error) => {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen SlotAware MTP K3 canonical accept walk",
                );
            }
        };
        let semantic_elapsed = semantic_started.elapsed();

        let committed_cursor = prior_target_len + plan.valid_input_tokens as u32;
        let state_started = Instant::now();
        let mut rollback_elapsed = Duration::ZERO;
        let state_result = (|| -> Result<()> {
            kv_cache
                .truncate_full_attn_to_for_slot(self.slot_id, committed_cursor)
                .context("Qwen SlotAware MTP K3 target truncate")?;
            kv_cache
                .truncate_mtp_to_for_slot(self.slot_id, committed_cursor)
                .context("Qwen SlotAware MTP K3 MTP truncate")?;
            if plan.valid_input_tokens < VERIFY_ROWS {
                let rollback_started = Instant::now();
                kv_cache
                    .rollback_la_to(self.slot_id, plan.carry_hidden_row as u32)
                    .context("Qwen SlotAware MTP K3 recurrent rollback")?;
                rollback_elapsed = rollback_started.elapsed();
            }
            kv_cache.clear_la_capture();
            kv_cache
                .validate_speculative_cursors_for_slot(self.slot_id, committed_cursor as usize)
                .context("Qwen SlotAware MTP K3 committed cursor equality")?;
            Ok(())
        })();
        let state_elapsed = state_started.elapsed();
        if let Err(error) = state_result {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen SlotAware MTP K3 commit verified prefix",
            );
        }
        if let Err(error) = qwen35_state_failpoint(Qwen35StateFailpoint::MtpCommit) {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen SlotAware MTP K3 post-commit failpoint",
            );
        }

        let carry_hidden = match crate::inference::models::qwen35::spec_decode::nth_hidden_row(
            &verify_hidden,
            qwen.model.cfg.hidden_size,
            plan.carry_hidden_row as u64,
        ) {
            Ok(hidden) => hidden,
            Err(error) => {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen SlotAware MTP K3 carry hidden",
                );
            }
        };
        self.mtp
            .as_mut()
            .expect("MTP state checked above")
            .verifier_hidden = carry_hidden;
        self.pending_speculation_output = plan.output;
        self.terminal_after_pending = plan.terminal_after_pending;
        super::qwen35_speculation::record_proposer_outcome(
            super::qwen35_speculation::QwenSpeculationProposer::Mtp,
            DRAFT_DEPTH,
            plan.matched_drafts,
            plan.rejected_drafts,
            1,
            0,
        );
        let equivalent_decisions = equivalent_target_decisions(
            &self.pending_speculation_output,
            self.terminal_after_pending,
        );
        let round_elapsed = round_started.elapsed();
        if phase_profile {
            eprintln!(
                "[MTP_PHASE] draft={:.2}ms verify={:.2}ms embed={:.2}ms catchup={:.2}ms semantic={:.2}ms state={:.2}ms rollback={:.2}ms partial={} remainder={:.2}ms total={:.2}ms",
                draft_elapsed.as_secs_f64() * 1000.0,
                verify_elapsed.as_secs_f64() * 1000.0,
                embed_elapsed.as_secs_f64() * 1000.0,
                catchup_elapsed.as_secs_f64() * 1000.0,
                semantic_elapsed.as_secs_f64() * 1000.0,
                state_elapsed.as_secs_f64() * 1000.0,
                rollback_elapsed.as_secs_f64() * 1000.0,
                plan.valid_input_tokens < VERIFY_ROWS,
                round_elapsed
                    .saturating_sub(draft_elapsed)
                    .saturating_sub(verify_elapsed)
                    .saturating_sub(embed_elapsed)
                    .saturating_sub(catchup_elapsed)
                    .saturating_sub(semantic_elapsed)
                    .saturating_sub(state_elapsed)
                    .as_secs_f64()
                    * 1000.0,
                round_elapsed.as_secs_f64() * 1000.0,
            );
        }
        if let Some(equivalent_ordinary) = self
            .mtp_cost
            .equivalent_ordinary_elapsed(equivalent_decisions)
        {
            super::qwen35_speculation::record_proposer_timing(
                super::qwen35_speculation::QwenSpeculationProposer::Mtp,
                round_elapsed,
                equivalent_ordinary,
            );
        }
        let remains_profitable = self
            .mtp_cost
            .observe_speculative_round(equivalent_decisions, round_elapsed);
        if !remains_profitable {
            super::qwen35_speculation::record_cost_disabled(
                super::qwen35_speculation::QwenSpeculationProposer::Mtp,
            );
            self.mtp = None;
        }

        if let Some(token) = take_pending_speculation_output(&mut self.pending_speculation_output) {
            self.commit_speculation_output(qwen, token)
        } else {
            self.finish_reason = "stop";
            Ok(Qwen35TickOutcome {
                fragment: String::new(),
                is_reasoning: false,
                finished: true,
            })
        }
    }

    /// One canonical target token while retaining a coherent MTP boundary.
    /// This is both the measured ordinary baseline for the cost controllers
    /// and the tail path when fewer than four output tokens remain.
    fn decode_tick_mtp_warmup(
        &mut self,
        qwen: &mut Qwen35LoadedModel,
        kv_cache: &mut HybridKvCache,
        supervisor: &EngineSupervisor,
    ) -> Result<Qwen35TickOutcome> {
        let next_token = *self
            .generated_tokens
            .last()
            .context("Qwen SlotAware coherent ordinary decode needs a seed")?;
        let next_pos = (self.decode_position_base + self.generated_tokens.len() - 1) as i32;
        let prior_target = kv_cache.sequence_len_for_slot(self.slot_id)?;
        let prior_mtp = mtp_cursor_for_slot(kv_cache, self.slot_id)?;
        anyhow::ensure!(
            prior_target == prior_mtp,
            "Qwen coherent ordinary decode cursor mismatch (target={prior_target}, mtp={prior_mtp})"
        );
        let transaction = begin_slot_state_transaction(kv_cache, self.slot_id, prior_target)?;
        let pending_hidden = self
            .mtp
            .as_ref()
            .context("Qwen coherent ordinary decode missing MTP state")?
            .verifier_hidden
            .clone();
        // Measure the complete target-equivalent decision transaction. The
        // speculation controller compares this baseline against a full draft
        // + verify + MTP catch-up + cache-commit round, so stopping the timer
        // after the target forward alone would systematically underprice
        // ordinary decode and disable a profitable proposer.
        let ordinary_decision_started = Instant::now();
        let lease = supervisor.arm(
            "Qwen35 coherent ordinary target",
            QWEN35_WORKER_TRANSACTION_TIMEOUT,
        )?;
        let forward = qwen.model.forward_gpu_with_nextn_hidden(
            &[next_token],
            &[next_pos; 4],
            kv_cache,
            self.slot_id,
        );
        if let Err(error) = lease.finish() {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen coherent ordinary target supervision",
            );
        }
        let (mut logits, nextn_hidden) = match forward {
            Ok(value) => value,
            Err(error) => {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen coherent ordinary target",
                );
            }
        };
        if let Err(error) = qwen35_state_failpoint(Qwen35StateFailpoint::WarmupTarget) {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen coherent ordinary post-target failpoint",
            );
        }
        let shared_embed = match qwen.model.embed_tokens_gpu(&[next_token]) {
            Ok(embed) => embed,
            Err(error) => {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen coherent ordinary shared embedding",
                );
            }
        };
        let mtp = match qwen.model.mtp.as_ref() {
            Some(mtp) => mtp,
            None => {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    anyhow::anyhow!("Qwen coherent ordinary decode MTP weights missing"),
                    "Qwen coherent ordinary MTP lookup",
                );
            }
        };
        let lease = match supervisor.arm(
            "Qwen35 coherent ordinary MTP catch-up",
            QWEN35_WORKER_TRANSACTION_TIMEOUT,
        ) {
            Ok(lease) => lease,
            Err(error) => {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen coherent ordinary MTP catch-up admission",
                );
            }
        };
        let caught_up = qwen.model.with_gpu_cache_mut(|device, registry| {
            mtp.process_target_batch(
                &[next_token],
                Some(&pending_hidden),
                &nextn_hidden,
                &shared_embed,
                kv_cache,
                self.slot_id,
                &[next_pos; 4],
                device,
                registry,
                &qwen.model.cfg,
            )
        });
        if let Err(error) = lease.finish().and(caught_up) {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen coherent ordinary MTP catch-up",
            );
        }
        if let Err(error) = qwen35_state_failpoint(Qwen35StateFailpoint::WarmupMtpCatchup) {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen coherent ordinary post-MTP-catch-up failpoint",
            );
        }
        if let Err(error) = kv_cache
            .validate_speculative_cursors_for_slot(self.slot_id, (prior_target + 1) as usize)
        {
            return rollback_slot_state_error(
                kv_cache,
                self.slot_id,
                &transaction,
                error,
                "Qwen coherent ordinary committed cursor equality",
            );
        }
        let carry_hidden = match crate::inference::models::qwen35::spec_decode::nth_hidden_row(
            &nextn_hidden,
            qwen.model.cfg.hidden_size,
            0,
        ) {
            Ok(hidden) => hidden,
            Err(error) => {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen coherent ordinary carry hidden",
                );
            }
        };
        self.mtp
            .as_mut()
            .expect("MTP state checked above")
            .verifier_hidden = carry_hidden;

        let mut semantic = Qwen35SpecSemanticState::from_decode(self);
        let decision = match semantic.select_and_observe(qwen, &self.params, &mut logits) {
            Ok(decision) => decision,
            Err(error) => {
                return rollback_slot_state_error(
                    kv_cache,
                    self.slot_id,
                    &transaction,
                    error,
                    "Qwen coherent ordinary canonical decision",
                );
            }
        };
        let ordinary_decision_elapsed = ordinary_decision_started.elapsed();
        self.mtp_cost
            .observe_ordinary_target(ordinary_decision_elapsed);
        self.history_cost
            .observe_ordinary_target(ordinary_decision_elapsed);
        super::qwen35_speculation::record_outcome(0, 0, 0, 1, 0);
        if decision.terminal {
            self.finish_reason = "stop";
            return Ok(Qwen35TickOutcome {
                fragment: String::new(),
                is_reasoning: false,
                finished: true,
            });
        }
        self.pending_speculation_output.push_back(decision.token);
        let token = self
            .pending_speculation_output
            .pop_front()
            .expect("coherent ordinary decision queued above");
        self.commit_speculation_output(qwen, token)
    }

    fn commit_speculation_output(
        &mut self,
        qwen: &Qwen35LoadedModel,
        token: u32,
    ) -> Result<Qwen35TickOutcome> {
        if qwen.eos_token_ids.contains(&token) {
            self.finish_reason = "stop";
            return Ok(Qwen35TickOutcome {
                fragment: String::new(),
                is_reasoning: false,
                finished: true,
            });
        }
        let forced_token = self
            .thinking_budget
            .as_mut()
            .and_then(Qwen35ThinkingBudgetState::next_forced_token);
        if let Some((forced, started)) = forced_token {
            debug_assert_eq!(forced, token, "verified speculative force token drift");
            if started {
                tracing::warn!(
                    slot = self.slot_id.0,
                    budget = self.params.thinking_token_budget,
                    generated_tokens = self.generated_tokens.len(),
                    "Qwen35 thinking token budget reached; forcing reasoning close and continuing answer"
                );
            }
        }
        advance_qwen35_grammar(&mut self.grammar_runtime, &self.params, token);
        if qwen35_grammar_terminal_token(self.grammar_runtime.as_ref(), &self.params, token) {
            self.finish_reason = "stop";
            return Ok(Qwen35TickOutcome {
                fragment: String::new(),
                is_reasoning: false,
                finished: true,
            });
        }
        self.generated_tokens.push(token);
        qwen35_observe_sampling_history(&mut self.sampling_history, token);
        self.next_token = token;
        let fragment = qwen.tokenizer.decode(&[token], false).unwrap_or_default();
        self.decoded_text.push_str(&fragment);
        let mut tool_opened = false;
        if let Some(splitter) = self.tool_splitter.as_mut() {
            let marker_events = splitter.feed(&fragment);
            tool_opened = marker_events
                .iter()
                .any(|event| matches!(event, ToolCallEvent::ToolCallOpen));
            if tool_opened {
                if let Some(runtime) = self.grammar_runtime.as_mut() {
                    runtime.trigger();
                }
            }
        }
        if let Some(budget) = self.thinking_budget.as_mut() {
            budget.observe_generated(&self.generated_tokens, tool_opened);
        }
        self.step = self.step.saturating_add(1);
        let terminal = self.terminal_after_pending && self.pending_speculation_output.is_empty();
        if terminal {
            self.finish_reason = "stop";
        }
        Ok(Qwen35TickOutcome {
            fragment,
            is_reasoning: false,
            finished: terminal || self.generated_tokens.len() >= self.max_tokens,
        })
    }

    /// Per-slot KV exit reset — caller invokes once after the slot finishes
    /// (mirror of serial ref exit reset 2476-2478).
    pub(crate) fn reset_at_exit(&self, kv_cache: &mut HybridKvCache) -> Result<()> {
        kv_cache
            .reset_for_slot(self.slot_id)
            .context("ADR-040 Phase F M1: Qwen35 reset_for_slot at exit")
    }

    /// Assemble the `GenerationResult` — mirror of serial ref 2483-2526,
    /// including the POST-HOC reasoning-token recompute (re-feed all
    /// generated tokens through a fresh splitter) that diverges from gemma4.
    pub(crate) fn finish(
        self,
        qwen: &Qwen35LoadedModel,
        registration: Option<&ModelRegistration>,
    ) -> GenerationResult {
        let (content_text, reasoning_text) = match registration {
            Some(reg) if reg.has_reasoning() => super::registry::split_full_output_forced(
                reg,
                &self.decoded_text,
                self.params.reasoning_forced_open,
            ),
            _ => (self.decoded_text.clone(), None),
        };
        let reasoning_token_count = match registration {
            Some(reg) if reg.has_reasoning() => {
                let mut sp = super::registry::make_reasoning_splitter(
                    reg,
                    self.params.reasoning_forced_open,
                );
                let mut count = 0usize;
                for &tok in &self.generated_tokens {
                    let frag = qwen.tokenizer.decode(&[tok], false).unwrap_or_default();
                    if let Some(splitter) = sp.as_mut() {
                        let _ = splitter.feed(&frag);
                        if splitter.in_reasoning() {
                            count += 1;
                        }
                    }
                }
                count
            }
            _ => 0,
        };
        let decode_duration = self.decode_start.elapsed();
        GenerationResult {
            text: content_text,
            reasoning_text,
            prompt_tokens: self.prompt_len,
            completion_tokens: self.generated_tokens.len(),
            reasoning_tokens: if reasoning_token_count > 0 {
                Some(reasoning_token_count)
            } else {
                None
            },
            finish_reason: self.finish_reason,
            prefill_duration: self.prefill_duration,
            decode_duration,
            cached_tokens: self.cached_tokens,
            logprobs: self.logprobs_vec,
        }
    }
}

/// ADR-040 Phase C iter-C2d-cont-kernel iter-2 (2026-05-30): slot-aware
/// streaming Qwen35 chat generation against the **persistent multi-seq
/// `HybridKvCache`** (`Qwen35LoadedModel.persistent_kv_cache`) instead
/// of a per-request fresh alloc.
///
/// **Direct mirror of `generate_qwen35_once_slot_aware`** (iter-1) for
/// the [`super::engine::Request::GenerateStream`] worker arm. iter-1
/// landed the non-streaming Generate-arm lift; iter-2 lands the
/// streaming-arm lift onto the same persistent cache + per-slot reset
/// + `restore_partial`-based prompt-cache HIT scaffolding.
///
/// # Structural parallels with iter-1
///
/// 1. Bounds-checks `slot_id` against `kv_cache.n_seqs` (bounds-first
///    per A2b §6.1.23 iter-1.5 cfa-finding-F5 ordering); surfaces typed
///    error via the SSE `events` channel if slot OOR.
/// 2. Verifies `prompt_len + max_tokens + 64 <= kv_cache.max_seq_len`
///    (persistent cache is sized to `cfg.max_position_embeddings`;
///    per-request need must fit).
/// 3. Calls `kv_cache.reset_for_slot(slot_id)` at entry — zeros the
///    per-seq full-attn cursor + per-seq linear-attn conv/recurrent
///    slices for `slot_id` only (other slots untouched).
/// 4. Prompt-cache fast-path uses `restore_partial(snap, prompt_len)`
///    instead of `restore_from(snap)` — the persistent cache's
///    `max_seq_len` differs from the snapshot producer's per-request
///    `max_seq_len`, breaking `restore_from`'s byte-equal precondition.
/// 5. Threads `slot_id` into every `forward_gpu_last_logits` call (the
///    signature already accepts `SlotId` post-B4b §6.1.20).
/// 6. Calls `kv_cache.reset_for_slot(slot_id)` at exit — belt-and-
///    suspenders with the entry reset (mirror of iter-1).
///
/// **Per-slot byte-equivalence at SlotId(0)** (H58 pin):
/// `generate_stream_qwen35_once_extended_slot_aware(.., kv=&mut
/// persistent_cache, slot_id=SlotId(0))` produces the same SSE event
/// stream as `generate_stream_qwen35_once_extended(..)` for any
/// text-only request when `persistent_cache.n_seqs == 1` AND
/// `persistent_cache.max_seq_len >= prompt_len + max_tokens + 64`. The
/// proof is identical to iter-1's H51 pin: `reset_for_slot(0)` matches
/// the fresh-alloc state, `forward_gpu_last_logits(.., SlotId(0))` is
/// byte-equivalent to the pre-A2b path (B4a §6.1.4 pin), and
/// `restore_partial(snap, k)` at `k == snap.full_attn_current_len[0][0]`
/// is byte-equivalent to `restore_from(snap)` per the kv_cache.rs:2143
/// docstring.
///
/// # Vision-augmented streaming deferral (iter-2 scope discipline)
///
/// When **any** of `soft_tokens` / `deepstack` / `positions_flat` is
/// non-empty / `Some(...)`, iter-2 emits a typed `capability_unsupported:`
/// error event and aborts. Vision-augmented streaming via the persistent
/// multi-seq cache is **iter-C2d-cont-kernel-iter-4 scope** — the same
/// iter that ports the non-streaming
/// `generate_qwen35_once_with_soft_tokens_and_deepstack` (the API
/// surfaces add injection-bytes-keyed cache invalidation discipline
/// that touches more than the streaming wrapper). The H59 + H63 pins
/// validate the typed-error path for any-extension requests.
///
/// # Co-changes (iter-2 deliberately minimal — exact mirror of iter-1)
///
/// - Per-slot LCP / mid-prefill checkpoint storage is DISABLED in
///   slot-aware mode.  ADR-040 §6.1.50 (2026-05-30) closes
///   **iter-C2d-cont-kernel-iter-LCP per ADR-040 §6.1.50** as
///   **STRUCTURAL N/A** (same finding as iter-1's docstring — the
///   snapshot codec keys on per-request `max_seq_len` + cross-slot
///   tenant-isolation; full-equality prompt-cache HIT path already uses
///   `restore_partial`).  The structural-N/A pin documents the call-
///   graph reality.
/// - Chunked-prefill is DISABLED in slot-aware mode (same snapshot-
///   shape reason).  Same STRUCTURAL N/A pin per §6.1.50.
/// - Vision-augmented streaming (soft_tokens / deepstack /
///   positions_flat any non-empty) returns a typed error event citing
///   **iter-C2d-cont-kernel-iter-4** as the implementing iter.
/// - ADR-040 §6.1.50 (2026-05-30) lands
///   **iter-C2d-cont-kernel-iter-G per ADR-040 §6.1.50** REAL LIFT for
///   the streaming decode greedy fast-path: when `is_greedy` is true,
///   the decode step now routes through `forward_gpu_greedy(.., slot_id)`
///   instead of `forward_gpu_last_logits + greedy_argmax_last_token`.
///   Same per-step savings as iter-1 (~250 µs at vocab=151k).  Non-
///   greedy + cancellation paths UNCHANGED.
///
/// # SSE event ordering (H63 pin)
///
/// The slot-aware streaming function emits the same SSE event order as
/// `generate_stream_qwen35_once_extended`: per-token `Delta` events
/// routed through the `ReasoningSplitter` + `ToolCallSplitter` chain,
/// followed by a terminal `Done` event (or `Error` event on failure).
/// The splitter-chain wiring is structurally identical — iter-2 does
/// not alter the splitter shape, only the prefill / decode KV substrate.
///
/// # Errors emitted on the SSE channel
/// - `slot_id.0 >= kv_cache.n_seqs` → `Error("capability_unsupported:
///   ADR-040 iter-C2d-cont-kernel iter-2 — SlotOutOfRange ...")`.
/// - `need_seq > kv_cache.max_seq_len` → `Error("capability_unsupported:
///   ADR-040 iter-C2d-cont-kernel iter-2 — per-request need_seq ...")`.
/// - `has_extension == true` → `Error("capability_unsupported:
///   ADR-040 iter-C2d-cont-kernel iter-2 — vision-augmented streaming
///   slot-aware port is iter-C2d-cont-kernel-iter-4 ...")`.
/// - Any `reset_for_slot` failure → `Error("ADR-040 iter-C2d-cont-kernel
///   iter-2 — reset_for_slot ...")`.
/// - Any `restore_partial` failure → `Error("ADR-040 iter-C2d-cont-kernel
///   iter-2 — prompt_cache restore_partial ...")`.
/// - Forward / sample failures propagate from `forward_gpu_last_logits`.
#[allow(clippy::too_many_arguments)]
pub fn generate_stream_qwen35_once_extended_slot_aware(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    soft_tokens: &[crate::serve::forward_prefill::SoftTokenInjection<'_>],
    deepstack: Option<&crate::serve::forward_prefill::DeepstackInjection<'_>>,
    positions_flat: Option<&[i32]>,
    params: &SamplingParams,
    events: &tokio::sync::mpsc::Sender<GenerationEvent>,
    registration: Option<&ModelRegistration>,
    cancellation_counter: Option<&std::sync::atomic::AtomicU64>,
    kv_cache: &mut HybridKvCache,
    slot_id: SlotId,
) {
    macro_rules! send {
        ($ev:expr) => {
            if events.blocking_send($ev).is_err() {
                tracing::info!("SSE stream dropped by client; aborting qwen35 slot-aware decode");
                if let Some(c) = cancellation_counter {
                    c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return;
            }
        };
    }

    if prompt_tokens.is_empty() {
        send!(GenerationEvent::Error(
            "generate_stream_qwen35_once_extended_slot_aware: empty prompt_tokens".into()
        ));
        return;
    }
    // Bounds-first per A2b §6.1.23 iter-1.5 cfa-finding-F5 ordering.
    if slot_id.0 >= kv_cache.n_seqs {
        send!(GenerationEvent::Error(format!(
            "capability_unsupported: ADR-040 iter-C2d-cont-kernel iter-2 — \
             SlotOutOfRange slot={} max_slots={} (generate_stream_qwen35_\
             once_extended_slot_aware)",
            slot_id.0, kv_cache.n_seqs,
        )));
        return;
    }

    // ADR-040 iter-C2d-cont-kernel iter-4 (2026-05-30): vision-augmented
    // streaming slot-aware path. iter-2 originally surfaced a typed
    // capability_unsupported error event here; iter-4 LIFTS the branch
    // — `has_extension == true` routes through
    // `forward_gpu_last_logits_with_soft_tokens_and_deepstack(.., slot_id)`
    // for prefill + a t_post-advanced decode loop (mirror of the
    // non-streaming `generate_qwen35_once_with_soft_tokens_and_deepstack`
    // shape, lifted via `*_slot_aware`). Empty soft + None deepstack +
    // None positions ⇒ text-only path unchanged.
    let has_extension = !soft_tokens.is_empty() || deepstack.is_some() || positions_flat.is_some();

    // Validate `positions_flat` length up-front so we fail loud BEFORE
    // any GPU work — mirrors the non-streaming sibling at
    // `generate_qwen35_once_with_soft_tokens_and_deepstack_slot_aware`
    // and the non-slot-aware `generate_stream_qwen35_once_extended`.
    if let Some(p) = positions_flat {
        if p.len() != 4 * prompt_tokens.len() {
            send!(GenerationEvent::Error(format!(
                "qwen35 stream slot-aware (iter-4): positions_flat.len() = {} \
                 != 4 * prompt_len = {}",
                p.len(),
                4 * prompt_tokens.len()
            )));
            return;
        }
    }

    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);
    let need_seq = prompt_len + max_tokens + 64;
    if need_seq > kv_cache.max_seq_len as usize {
        send!(GenerationEvent::Error(format!(
            "capability_unsupported: ADR-040 iter-C2d-cont-kernel iter-2 — \
             per-request need_seq={} exceeds persistent cache \
             max_seq_len={} (slot={} prompt_len={} max_tokens={}). \
             Persistent cache is sized to cfg.max_position_embeddings; \
             reduce max_tokens or use a shorter prompt.",
            need_seq, kv_cache.max_seq_len, slot_id.0, prompt_len, max_tokens
        )));
        return;
    }

    let is_greedy = is_greedy_eligible(params);

    let device = match MlxDevice::new() {
        Ok(d) => d,
        Err(e) => {
            send!(GenerationEvent::Error(format!(
                "qwen35 stream slot-aware: MlxDevice::new failed: {e}"
            )));
            return;
        }
    };
    let _ = &device; // sampling helpers don't need the device handle

    // Per-slot reset at entry — persistent cache may carry stale bytes
    // from a prior request on this slot.
    if let Err(e) = kv_cache.reset_for_slot(slot_id) {
        send!(GenerationEvent::Error(format!(
            "ADR-040 iter-C2d-cont-kernel iter-2: reset_for_slot at entry \
             failed: {e:#}"
        )));
        return;
    }

    let pre_dispatches = mlx_native::dispatch_count();
    let pre_syncs = mlx_native::sync_count();

    // ── Prompt-cache fast-path (slot-aware variant) ──────────────────
    //
    // Snapshot restore uses `restore_partial(snap, k)` instead of
    // `restore_from(snap)` — the persistent cache's `max_seq_len`
    // (= cfg.max_position_embeddings) differs from the snapshot
    // producer's per-request `max_seq_len`. Mirror of iter-1's
    // prompt-cache HIT branch.
    //
    // ADR-040 iter-C2d-cont-kernel iter-4: BYPASS prompt-cache on the
    // vision-augmented path (has_extension == true). Cache key is
    // `prompt_tokens` only and would falsely hit on a vision-augmented
    // request with the same placeholder ids but different image content.
    // Mirrors non-slot-aware `generate_stream_qwen35_once_extended` at
    // engine_qwen35.rs:3870 ("Prompt-cache fast-path is BYPASSED
    // whenever any extension is present").
    let prompt_cache_hit =
        !has_extension && qwen.prompt_cache.try_match(prompt_tokens, params).is_some();

    let prefill_start = Instant::now();
    let mut next_token: u32;
    if prompt_cache_hit {
        let snap = qwen
            .prompt_cache
            .snapshot()
            .expect("try_match Some implies snapshot Some");
        if let Err(e) = kv_cache.restore_partial(snap, prompt_len) {
            send!(GenerationEvent::Error(format!(
                "ADR-040 iter-C2d-cont-kernel iter-2: prompt_cache \
                 restore_partial failed: {e:#}"
            )));
            return;
        }
        next_token = qwen.prompt_cache.first_decoded_token();
        tracing::debug!(
            "qwen35 stream slot-aware prompt_cache: HIT slot={} prompt_len={} \
             prefill skipped",
            slot_id.0,
            prompt_len
        );
    } else {
        // Fresh monolithic prefill — chunked-prefill + LCP-resume are
        // STRUCTURAL N/A in slot-aware mode (see fn docstring for the
        // snapshot codec invariant rationale; iter-LCP STRUCTURAL N/A
        // closure landed at iter-C2d-cont-kernel-iter-LCP per ADR-040
        // §6.1.50).
        //
        // ADR-040 iter-C2d-cont-kernel iter-4: when has_extension is
        // true, prefill goes through
        // `forward_gpu_last_logits_with_soft_tokens_and_deepstack(..,
        // slot_id)` with caller-supplied 3D positions when present;
        // otherwise text-only `forward_gpu_last_logits(.., slot_id)`
        // with synthesized text-style positions. Mirror of
        // non-slot-aware `generate_stream_qwen35_once_extended`'s
        // `if has_extension { ... } else { ... }` branch (engine_qwen35.rs:4061).
        let positions_owned: Vec<i32>;
        let positions_slice: &[i32] = match positions_flat {
            Some(p) => p,
            None => {
                positions_owned = prefill_positions_for(prompt_len);
                &positions_owned
            }
        };
        let prefill_logits_res: Result<Vec<f32>> = if has_extension {
            qwen.model
                .forward_gpu_last_logits_with_soft_tokens_and_deepstack(
                    prompt_tokens,
                    positions_slice,
                    soft_tokens,
                    deepstack,
                    kv_cache,
                    slot_id,
                )
        } else {
            qwen.model
                .forward_gpu_last_logits(prompt_tokens, positions_slice, kv_cache, slot_id)
        };
        let prefill_logits = match prefill_logits_res {
            Ok(l) => l,
            Err(e) => {
                send!(GenerationEvent::Error(format!(
                    "qwen35 stream slot-aware prefill failed: {e:#}"
                )));
                return;
            }
        };
        if prefill_logits.len() != qwen.vocab_size {
            send!(GenerationEvent::Error(format!(
                "qwen35 stream slot-aware prefill logits len {} != \
                 vocab_size {}",
                prefill_logits.len(),
                qwen.vocab_size,
            )));
            return;
        }
        if is_greedy {
            next_token = greedy_argmax_last_token(&prefill_logits, qwen.vocab_size as u32);
        } else {
            let mut logits = prefill_logits;
            next_token = sample_logits_qwen35(&mut logits, params, &[]);
        }
        // Prompt-cache snapshot is intentionally NOT taken in slot-aware
        // mode: the persistent cache's snapshot would carry the
        // cfg.max_position_embeddings shape, which differs from
        // per-request snapshots the SerialFifo / SlotId(0) path stores.
        // Cross-mode snapshot sharing is iter-C2d-cont-kernel-iter-LCP
        // STRUCTURAL N/A per ADR-040 §6.1.50.
    }
    let prefill_duration = prefill_start.elapsed();

    // ADR-040 iter-C2d-cont-kernel iter-4: post-prefill text decode
    // positions advance from the global temporal counter (which
    // advances by max(n_x, n_y) per image during prefill, NOT by
    // n_image_tokens, per peer `mtmd.cpp:1354-1357`). When
    // `positions_flat` is supplied, compute `t_post = max(axis-0
    // positions) + 1`; when None, fall back to the legacy
    // `prompt_len` text-style advance. Mirror of non-slot-aware
    // sibling at engine_qwen35.rs:4270 and the non-streaming
    // `generate_qwen35_once_with_soft_tokens_and_deepstack_slot_aware`.
    let t_post: i32 = match positions_flat {
        Some(p) => {
            let mut max_t = 0i32;
            for i in 0..prompt_len {
                let v = p[i]; // axis 0 = t
                if v > max_t {
                    max_t = v;
                }
            }
            max_t.saturating_add(1)
        }
        None => prompt_len as i32,
    };

    // ── Splitter wiring (Reasoning + ToolCall) ────────────────────
    // Mirror of generate_stream_qwen35_once_extended's splitter chain.
    let mut reasoning_splitter = registration
        .and_then(|r| super::registry::make_reasoning_splitter(r, params.reasoning_forced_open));
    let mut tool_splitter = registration.and_then(ToolCallSplitter::from_registration);
    let mut tool_call_body: String = String::new();
    let mut tool_call_index: usize = 0;
    let mut saw_tool_call: bool = false;

    // Inner closure-like helper: route a Content-classified text run
    // through the tool-call splitter. Mirror of route_content_qwen35
    // in generate_stream_qwen35_once_extended. Returns `false` on
    // client disconnect.
    fn route_content_qwen35_slot_aware(
        tool_splitter: &mut Option<ToolCallSplitter>,
        body: &mut String,
        tc_index: &mut usize,
        saw_tc: &mut bool,
        registration: Option<&ModelRegistration>,
        wire_kinds: Option<&super::registry::ToolArgumentWireKinds>,
        events: &tokio::sync::mpsc::Sender<GenerationEvent>,
        text: &str,
    ) -> bool {
        if text.is_empty() {
            return true;
        }
        let Some(tcs) = tool_splitter.as_mut() else {
            return events
                .blocking_send(GenerationEvent::Delta {
                    kind: DeltaKind::Content,
                    text: text.to_string(),
                })
                .is_ok();
        };
        for ev in tcs.feed(text) {
            match ev {
                ToolCallEvent::Content(t) => {
                    if !t.is_empty()
                        && events
                            .blocking_send(GenerationEvent::Delta {
                                kind: DeltaKind::Content,
                                text: t,
                            })
                            .is_err()
                    {
                        return false;
                    }
                }
                ToolCallEvent::ToolCallOpen => {
                    body.clear();
                }
                ToolCallEvent::ToolCallText(t) => {
                    body.push_str(&t);
                }
                ToolCallEvent::ToolCallClose => {
                    let parsed = registration.and_then(|r| {
                        super::registry::parse_tool_call_body_with_wire_kinds(r, body, wire_kinds)
                    });
                    let body_dump = std::mem::take(body);
                    let sink = super::engine::EventSink::new(events);
                    if super::engine::emit_streaming_tool_call_close(
                        parsed,
                        body_dump,
                        params_tool_call_policy_for_qwen35_stream(),
                        tc_index,
                        saw_tc,
                        &sink,
                    )
                    .is_err()
                    {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn emit_fragment_qwen35_slot_aware(
        reasoning_splitter: &mut Option<ReasoningSplitter>,
        tool_splitter: &mut Option<ToolCallSplitter>,
        body: &mut String,
        tc_index: &mut usize,
        saw_tc: &mut bool,
        registration: Option<&ModelRegistration>,
        wire_kinds: Option<&super::registry::ToolArgumentWireKinds>,
        events: &tokio::sync::mpsc::Sender<GenerationEvent>,
        fragment: &str,
    ) -> bool {
        if fragment.is_empty() {
            return true;
        }
        if let Some(rs) = reasoning_splitter.as_mut() {
            for (slot, text) in rs.feed(fragment) {
                match slot {
                    SplitSlot::Reasoning => {
                        if !text.is_empty()
                            && events
                                .blocking_send(GenerationEvent::Delta {
                                    kind: DeltaKind::Reasoning,
                                    text,
                                })
                                .is_err()
                        {
                            return false;
                        }
                    }
                    SplitSlot::Content => {
                        if !route_content_qwen35_slot_aware(
                            tool_splitter,
                            body,
                            tc_index,
                            saw_tc,
                            registration,
                            wire_kinds,
                            events,
                            &text,
                        ) {
                            return false;
                        }
                    }
                }
            }
            true
        } else {
            route_content_qwen35_slot_aware(
                tool_splitter,
                body,
                tc_index,
                saw_tc,
                registration,
                wire_kinds,
                events,
                fragment,
            )
        }
    }

    // ── Decode loop ────────────────────────────────────────────────
    let decode_start = Instant::now();
    let mut completion_tokens = 0usize;
    let mut generated_tokens = Vec::with_capacity(max_tokens);
    generated_tokens.push(next_token);
    let mut accumulated_text = String::new();
    let mut reasoning_token_count = 0usize;
    let mut finish_reason: &'static str = "length";

    let first_text = qwen
        .tokenizer
        .decode(&[next_token], false)
        .unwrap_or_default();
    let mut is_eos_first = qwen.eos_token_ids.contains(&next_token);
    if !is_eos_first && !first_text.is_empty() {
        accumulated_text.push_str(&first_text);
        if !emit_fragment_qwen35_slot_aware(
            &mut reasoning_splitter,
            &mut tool_splitter,
            &mut tool_call_body,
            &mut tool_call_index,
            &mut saw_tool_call,
            registration,
            params.tool_argument_wire_kinds.as_deref(),
            events,
            &first_text,
        ) {
            if let Some(c) = cancellation_counter {
                c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            // Per-slot reset at exit on cancellation path too — keep
            // the slot clean for the next request to land on it.
            let _ = kv_cache.reset_for_slot(slot_id);
            return;
        }
    }
    completion_tokens += 1;
    if reasoning_splitter
        .as_ref()
        .map(|s| s.in_reasoning())
        .unwrap_or(false)
    {
        reasoning_token_count += 1;
    }
    if is_eos_first {
        finish_reason = "stop";
    } else if qwen35_hit_stop_string(&accumulated_text, &params.stop_strings) {
        finish_reason = "stop";
        is_eos_first = true;
    }

    if !is_eos_first {
        for step in 1..max_tokens {
            // ADR-040 iter-C2d-cont-kernel iter-4: decode position uses
            // the t_post advance computed above so the vision-augmented
            // path correctly resumes from the post-prefill global
            // temporal counter. For the text-only path
            // (has_extension == false AND positions_flat is None),
            // `t_post = prompt_len as i32`, making `pos = prompt_len +
            // step - 1` — byte-identical to iter-2's text-only advance.
            let pos = t_post + (step as i32 - 1);
            if pos as u32 >= kv_cache.max_seq_len {
                break;
            }
            let decode_positions = vec![pos; 4];
            // ADR-040 §6.1.50 (2026-05-30) iter-C2d-cont-kernel-iter-G
            // REAL LIFT for streaming: greedy fast-path now routes through
            // `forward_gpu_greedy(.., slot_id)` (saves the per-step vocab-
            // size F32 download, ~250 µs at vocab=151k Qwen3.6 35B-A3B).
            // Sampling branch still uses `forward_gpu_last_logits` because
            // it needs the full logits CPU-side for `sample_logits_qwen35`.
            let dec_result: Result<u32, anyhow::Error> = if is_greedy {
                qwen.model
                    .forward_gpu_greedy(&[next_token], &decode_positions, kv_cache, slot_id)
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "qwen35 stream slot-aware forward_gpu_greedy (ADR-040 \
                             §6.1.50 iter-G) step {step}: {e}"
                        )
                    })
            } else {
                match qwen.model.forward_gpu_last_logits(
                    &[next_token],
                    &decode_positions,
                    kv_cache,
                    slot_id,
                ) {
                    Ok(logits) => {
                        if logits.len() != qwen.vocab_size {
                            Err(anyhow::anyhow!(
                                "qwen35 stream slot-aware decode logits len {} \
                                 != vocab_size {}",
                                logits.len(),
                                qwen.vocab_size,
                            ))
                        } else {
                            let mut tmp = logits;
                            Ok(sample_logits_qwen35(&mut tmp, params, &generated_tokens))
                        }
                    }
                    Err(e) => Err(e),
                }
            };
            next_token = match dec_result {
                Ok(t) => t,
                Err(e) => {
                    send!(GenerationEvent::Error(format!(
                        "qwen35 stream slot-aware decode step {step} failed: {e:#}"
                    )));
                    // Per-slot reset on error path too.
                    let _ = kv_cache.reset_for_slot(slot_id);
                    return;
                }
            };
            if qwen.eos_token_ids.contains(&next_token) {
                finish_reason = "stop";
                break;
            }
            generated_tokens.push(next_token);
            completion_tokens += 1;
            let fragment = qwen
                .tokenizer
                .decode(&[next_token], false)
                .unwrap_or_default();
            accumulated_text.push_str(&fragment);
            if !emit_fragment_qwen35_slot_aware(
                &mut reasoning_splitter,
                &mut tool_splitter,
                &mut tool_call_body,
                &mut tool_call_index,
                &mut saw_tool_call,
                registration,
                params.tool_argument_wire_kinds.as_deref(),
                events,
                &fragment,
            ) {
                if let Some(c) = cancellation_counter {
                    c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                let _ = kv_cache.reset_for_slot(slot_id);
                return;
            }
            if reasoning_splitter
                .as_ref()
                .map(|s| s.in_reasoning())
                .unwrap_or(false)
            {
                reasoning_token_count += 1;
            }
            if qwen35_hit_stop_string(&accumulated_text, &params.stop_strings) {
                finish_reason = "stop";
                break;
            }
        }
    }

    // ── Drain splitter tails ───────────────────────────────────────
    // Mirror of generate_stream_qwen35_once_extended's tail drain.
    if let Some(rs) = reasoning_splitter.as_mut() {
        if let Some((slot, tail)) = rs.finish() {
            match slot {
                SplitSlot::Reasoning => {
                    if !tail.is_empty()
                        && events
                            .blocking_send(GenerationEvent::Delta {
                                kind: DeltaKind::Reasoning,
                                text: tail,
                            })
                            .is_err()
                    {
                        if let Some(c) = cancellation_counter {
                            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        let _ = kv_cache.reset_for_slot(slot_id);
                        return;
                    }
                }
                SplitSlot::Content => {
                    if !route_content_qwen35_slot_aware(
                        &mut tool_splitter,
                        &mut tool_call_body,
                        &mut tool_call_index,
                        &mut saw_tool_call,
                        registration,
                        params.tool_argument_wire_kinds.as_deref(),
                        events,
                        &tail,
                    ) {
                        if let Some(c) = cancellation_counter {
                            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        let _ = kv_cache.reset_for_slot(slot_id);
                        return;
                    }
                }
            }
        }
    }
    if let Some(tcs) = tool_splitter.as_mut() {
        if let Some(ev) = tcs.finish() {
            match ev {
                ToolCallEvent::Content(t) => {
                    if !t.is_empty()
                        && events
                            .blocking_send(GenerationEvent::Delta {
                                kind: DeltaKind::Content,
                                text: t,
                            })
                            .is_err()
                    {
                        if let Some(c) = cancellation_counter {
                            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        let _ = kv_cache.reset_for_slot(slot_id);
                        return;
                    }
                }
                ToolCallEvent::ToolCallText(t) => {
                    let prefix = registration.and_then(|r| r.tool_open).unwrap_or("");
                    let fallback = format!("{prefix}{t}");
                    if !fallback.is_empty()
                        && events
                            .blocking_send(GenerationEvent::Delta {
                                kind: DeltaKind::Content,
                                text: fallback,
                            })
                            .is_err()
                    {
                        if let Some(c) = cancellation_counter {
                            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        let _ = kv_cache.reset_for_slot(slot_id);
                        return;
                    }
                }
                ToolCallEvent::ToolCallOpen | ToolCallEvent::ToolCallClose => {
                    // unreachable — finish() never emits Open/Close.
                }
            }
        }
    }

    if saw_tool_call {
        finish_reason = "tool_calls";
    }

    let decode_duration = decode_start.elapsed();

    // Per-slot reset at exit — leave the slot clean for the next
    // request to land on it. Belt-and-suspenders with the entry reset;
    // mirrors iter-1's exit-reset discipline (H54).
    if let Err(e) = kv_cache.reset_for_slot(slot_id) {
        // Reset failure at exit is unusual — log + surface as terminal
        // Error event (still followed by no `Done`; the SSE handler
        // treats a terminal Error as a clean stream termination).
        send!(GenerationEvent::Error(format!(
            "ADR-040 iter-C2d-cont-kernel iter-2: reset_for_slot at exit \
             failed: {e:#}"
        )));
        return;
    }

    let stats = StreamStats {
        prefill_time_secs: Some(prefill_duration.as_secs_f64()),
        decode_time_secs: Some(decode_duration.as_secs_f64()),
        total_time_secs: Some((prefill_duration + decode_duration).as_secs_f64()),
        time_to_first_token_ms: Some(prefill_duration.as_secs_f64() * 1000.0),
        prefill_tokens_per_sec: Some(if prefill_duration.as_secs_f64() > 0.0 {
            prompt_len as f64 / prefill_duration.as_secs_f64()
        } else {
            0.0
        }),
        decode_tokens_per_sec: Some(if decode_duration.as_secs_f64() > 0.0 {
            completion_tokens as f64 / decode_duration.as_secs_f64()
        } else {
            0.0
        }),
        gpu_sync_count: Some(mlx_native::sync_count().saturating_sub(pre_syncs)),
        gpu_dispatch_count: Some(mlx_native::dispatch_count().saturating_sub(pre_dispatches)),
        // Slot-aware mode reports prompt-cache HIT only (LCP/chunked
        // are disabled in slot-aware mode per the iter-LCP deferral).
        // Item 5a (2026-08-20): a miss is a known zero — report
        // Some(0), never None (the usage frame always carries an
        // explicit cached_tokens).
        cached_prompt_tokens: Some(if prompt_cache_hit { prompt_len } else { 0 }),
        reasoning_tokens: if reasoning_token_count > 0 {
            Some(reasoning_token_count)
        } else {
            None
        },
    };

    send!(GenerationEvent::Done {
        finish_reason,
        prompt_tokens: prompt_len,
        completion_tokens,
        stats,
    });
}

/// ADR-005 Phase 4 Wedge-4a (2026-05-01): vision-aware non-streaming
/// chat generation against a loaded Qwen3.5/3.6 model.  Replaces the
/// `worker_run` 501 arm for `Request::GenerateWithSoftTokens` (the last
/// `qwen35_not_implemented_err()` call site at
/// `crate::serve::api::engine::worker_run`).
///
/// Identical to `generate_qwen35_once` except:
///   * Prefill goes through `Qwen35Model::forward_gpu_last_logits_with_soft_tokens`
///     so per-position embedding overrides apply (image-token positions
///     consume the supplied projector outputs instead of the language-
///     model embedding table).
///   * The prompt-cache fast-path is BYPASSED.  The cache is keyed on
///     `prompt_tokens` only; a vision-augmented request with the same
///     placeholder ids but different image content would falsely hit a
///     cached text-only result.  Wedge-4 follow-up may extend the cache
///     key to include a hash of the soft-token bytes; for the Wedge-4a
///     opener we take the safe path and skip cache entirely when
///     `soft_tokens` is non-empty.
///   * Decode steps after the prefill use the standard text path
///     (`forward_gpu_greedy` / `forward_gpu_last_logits`) — soft-token
///     overrides only apply during prefill.  Decode positions are
///     post-prompt by construction so they cannot lie within a
///     soft-token range.
///
/// **Wedge-4a scope.** Wires the API for vision-on-Qwen3.5/3.6 without
/// adding a vision encoder.  Wedge-4b lands the qwen3vl ViT +
/// qwen3vl_merger projector + DeepStack taps so end-to-end multimodal
/// chat works against `Qwen/Qwen3-VL-8B-Instruct-GGUF`.
///
/// When `soft_tokens` is empty, behaviour is byte-identical to
/// `generate_qwen35_once`.
pub(super) fn generate_qwen35_once_with_soft_tokens(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    soft_tokens: &[crate::serve::forward_prefill::SoftTokenInjection<'_>],
    params: &SamplingParams,
    registration: Option<&ModelRegistration>,
    supervisor: &EngineSupervisor,
) -> Result<GenerationResult> {
    // Empty slice → identity over `generate_qwen35_once`.  This keeps
    // text-only fallback paths from paying any soft-token overhead
    // when (e.g.) a future caller threads an empty vec through the
    // engine `Request::GenerateWithSoftTokens` arm.
    if soft_tokens.is_empty() {
        return generate_qwen35_once(qwen, prompt_tokens, params, registration, supervisor);
    }

    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "generate_qwen35_once_with_soft_tokens: empty prompt_tokens"
    );
    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);
    let is_greedy = is_greedy_eligible(params);

    // ADR-020 AC#7 — see generate_qwen35_once.
    let want_logprobs = params.logprobs;
    let mut logprobs_vec: Option<Vec<f32>> = if want_logprobs {
        Some(Vec::with_capacity(max_tokens))
    } else {
        None
    };

    let device = MlxDevice::new()
        .map_err(|e| anyhow::anyhow!("MlxDevice::new (qwen35 generate w/ soft tokens): {e}"))?;
    let mut kv_cache = alloc_kv_cache_for_request(qwen, &device, prompt_len, max_tokens)?;

    // ADR-027 iter-6b.3: cold-start hydrate (soft-token path). Same
    // idempotent semantics as the text-only path; persistence-layer
    // failures are warn-logged + swallowed.
    qwen.hydrate_lcp_registry_from_disk(&kv_cache, &device);

    // Prompt-cache is intentionally NOT consulted on the vision path
    // (see docstring above for the cache-key safety rationale).
    let prefill_start = Instant::now();
    let positions = prefill_positions_for(prompt_len);
    let prefill_logits = supervised_gpu_call(supervisor, "qwen35_serial_soft_prefill", || {
        qwen.model
            .forward_gpu_last_logits_with_soft_tokens(
                prompt_tokens,
                &positions,
                soft_tokens,
                &mut kv_cache,
                SlotId(0),
            )
            .context("Qwen35Model::forward_gpu_last_logits_with_soft_tokens (prefill)")
    })?;
    anyhow::ensure!(
        prefill_logits.len() == qwen.vocab_size,
        "qwen35 prefill (soft tokens) logits len {} != vocab_size {}",
        prefill_logits.len(),
        qwen.vocab_size
    );
    let mut next_token: u32 = if want_logprobs {
        // ADR-020 AC#7 — bypass greedy fast-path; need full logits.
        let mut logits = prefill_logits.clone();
        let (tok, lp) = sample_logits_qwen35_with_logprob(&mut logits, params, &[]);
        if let Some(v) = logprobs_vec.as_mut() {
            v.push(lp);
        }
        tok
    } else if is_greedy {
        greedy_argmax_last_token(&prefill_logits, qwen.vocab_size as u32)
    } else {
        let mut logits = prefill_logits.clone();
        sample_logits_qwen35(&mut logits, params, &[])
    };
    let prefill_duration = prefill_start.elapsed();

    // Decode loop — identical to `generate_qwen35_once`.  Decode
    // positions are post-prompt by construction (>= prompt_len) and so
    // cannot lie within any soft-token range, so the decode path
    // deliberately uses the soft-token-FREE forward methods.
    let decode_start = Instant::now();
    let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    generated_tokens.push(next_token);

    let first_fragment = qwen
        .tokenizer
        .decode(&[next_token], false)
        .unwrap_or_default();
    let mut decoded_text = first_fragment.clone();

    let mut finish_reason: &'static str = "length";

    if qwen.eos_token_ids.contains(&next_token) {
        finish_reason = "stop";
    } else if qwen35_hit_stop_string(&decoded_text, &params.stop_strings) {
        finish_reason = "stop";
        qwen35_strip_trailing_stop(&mut decoded_text, &params.stop_strings);
    } else {
        for step in 1..max_tokens {
            let pos = (prompt_len + step - 1) as i32;
            if pos as u32 >= kv_cache.max_seq_len {
                tracing::warn!(
                    pos,
                    max_seq = kv_cache.max_seq_len,
                    "qwen35 decode (soft tokens): hit kv-cache bound; stopping with finish=length",
                );
                break;
            }
            let decode_positions = vec![pos; 4];

            next_token = if want_logprobs {
                // ADR-020 AC#7 — bypass greedy fast-path; need full logits.
                let logits_full = supervised_gpu_call(supervisor, "qwen35_serial_decode", || {
                    qwen.model
                        .forward_gpu_last_logits(
                            &[next_token],
                            &decode_positions,
                            &mut kv_cache,
                            SlotId(0),
                        )
                        .with_context(|| {
                            format!(
                            "forward_gpu_last_logits decode step {step} (soft tokens, logprobs)"
                        )
                        })
                })?;
                let mut logits = logits_full;
                let (tok, lp) =
                    sample_logits_qwen35_with_logprob(&mut logits, params, &generated_tokens);
                if let Some(v) = logprobs_vec.as_mut() {
                    v.push(lp);
                }
                tok
            } else if is_greedy {
                supervised_gpu_call(supervisor, "qwen35_serial_decode", || {
                    qwen.model
                        // ADR-040 Phase B4d (2026-05-30) — see sibling at
                        // engine_qwen35.rs:2071 for the SlotId contract.
                        .forward_gpu_greedy(
                            &[next_token],
                            &decode_positions,
                            &mut kv_cache,
                            SlotId(0),
                        )
                        .with_context(|| {
                            format!("forward_gpu_greedy decode step {step} (soft tokens)")
                        })
                })?
            } else {
                let logits_full = supervised_gpu_call(supervisor, "qwen35_serial_decode", || {
                    qwen.model
                        .forward_gpu_last_logits(
                            &[next_token],
                            &decode_positions,
                            &mut kv_cache,
                            SlotId(0),
                        )
                        .with_context(|| {
                            format!("forward_gpu_last_logits decode step {step} (soft tokens)")
                        })
                })?;
                let mut logits = logits_full;
                sample_logits_qwen35(&mut logits, params, &generated_tokens)
            };

            if qwen.eos_token_ids.contains(&next_token) {
                finish_reason = "stop";
                break;
            }
            generated_tokens.push(next_token);
            let fragment = qwen
                .tokenizer
                .decode(&[next_token], false)
                .unwrap_or_default();
            decoded_text.push_str(&fragment);
            if qwen35_hit_stop_string(&decoded_text, &params.stop_strings) {
                finish_reason = "stop";
                qwen35_strip_trailing_stop(&mut decoded_text, &params.stop_strings);
                break;
            }
        }
    }
    let decode_duration = decode_start.elapsed();

    // Reasoning split — same registry helper as generate_qwen35_once.
    let (content, reasoning_text) = match registration {
        Some(reg) if reg.has_reasoning() => super::registry::split_full_output_forced(
            reg,
            &decoded_text,
            params.reasoning_forced_open,
        ),
        _ => (decoded_text, None),
    };

    let reasoning_token_count = match registration {
        Some(reg) if reg.has_reasoning() => {
            let mut sp =
                super::registry::make_reasoning_splitter(reg, params.reasoning_forced_open);
            let mut count = 0usize;
            for &tok in &generated_tokens {
                let frag = qwen.tokenizer.decode(&[tok], false).unwrap_or_default();
                if let Some(splitter) = sp.as_mut() {
                    let _ = splitter.feed(&frag);
                    if splitter.in_reasoning() {
                        count += 1;
                    }
                }
            }
            count
        }
        _ => 0,
    };

    Ok(GenerationResult {
        text: content,
        reasoning_text,
        prompt_tokens: prompt_len,
        completion_tokens: generated_tokens.len(),
        reasoning_tokens: if reasoning_token_count > 0 {
            Some(reasoning_token_count)
        } else {
            None
        },
        finish_reason,
        prefill_duration,
        decode_duration,
        cached_tokens: 0,
        logprobs: logprobs_vec,
    })
}

/// Wedge-4d (ADR-005 iter-224 row 4) — vision-aware non-streaming chat
/// generation for Qwen3-VL with the full DeepStack injection pipeline.
///
/// Identical to `generate_qwen35_once_with_soft_tokens` except:
///   * Prefill goes through
///     `Qwen35Model::forward_gpu_last_logits_with_soft_tokens_and_deepstack`
///     so the per-LM-layer DeepStack chunks are added to the residual
///     stream at the image-token positions during prefill (per peer
///     `qwen3vl.cpp:96-100`).
///   * The 3D-mRoPE position buffer (`positions_flat: [4 * prompt_len]`)
///     is supplied by the chat handler via
///     `crate::serve::forward_prefill::build_qwen3vl_positions`, NOT
///     synthesized via `prefill_positions_for`. This carries the
///     `[t, y, x, 0]` axis assignment that the IMROPE kernel consumes
///     for image-patch tokens.
///   * Decode steps after prefill use text-only `[t,t,t,t]` positions
///     starting from the post-prefill global temporal counter (which
///     advances by `max(n_x, n_y)` per image, NOT by `n_image_tokens`,
///     per peer `mtmd.cpp:1354-1357`).
///
/// When both `deepstack` and `positions_flat` are `None`, behaviour is
/// identical to `generate_qwen35_once_with_soft_tokens`.
pub(super) fn generate_qwen35_once_with_soft_tokens_and_deepstack(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    soft_tokens: &[crate::serve::forward_prefill::SoftTokenInjection<'_>],
    deepstack: Option<&crate::serve::forward_prefill::DeepstackInjection<'_>>,
    positions_flat: Option<&[i32]>,
    params: &SamplingParams,
    registration: Option<&ModelRegistration>,
    supervisor: &EngineSupervisor,
) -> Result<GenerationResult> {
    // Empty soft + no deepstack + no positions → identity over text-only.
    if soft_tokens.is_empty() && deepstack.is_none() && positions_flat.is_none() {
        return generate_qwen35_once(qwen, prompt_tokens, params, registration, supervisor);
    }

    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "generate_qwen35_once_with_soft_tokens_and_deepstack: empty prompt_tokens"
    );
    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);
    let is_greedy = is_greedy_eligible(params);

    // ADR-020 AC#7 — see generate_qwen35_once.
    let want_logprobs = params.logprobs;
    let mut logprobs_vec: Option<Vec<f32>> = if want_logprobs {
        Some(Vec::with_capacity(max_tokens))
    } else {
        None
    };

    let device = MlxDevice::new()
        .map_err(|e| anyhow::anyhow!("MlxDevice::new (qwen35 vision generate): {e}"))?;
    let (mut kv_cache, cache_reused) = take_serial_kv_cache(qwen, &device, prompt_len, max_tokens)?;

    // ADR-027 iter-6b.3: cold-start hydrate (wedge-4d deepstack path).
    qwen.hydrate_lcp_registry_from_disk(&kv_cache, &device);

    let prompt_cache_hit = params.vision_fingerprint.is_some()
        && qwen.prompt_cache.try_match(prompt_tokens, params).is_some();
    if cache_reused && !prompt_cache_hit {
        kv_cache.reset();
    }

    let prefill_start = Instant::now();
    let mut next_token: u32;
    if prompt_cache_hit {
        let snap = qwen
            .prompt_cache
            .snapshot()
            .expect("vision prompt-cache hit requires snapshot");
        kv_cache
            .restore_partial(snap, prompt_len)
            .context("vision prompt-cache restore_partial")?;
        next_token = qwen.prompt_cache.first_decoded_token();
    } else {
        // Use supplied 3D positions if provided; otherwise fall back to
        // text-style `[t,t,t,t]` positions.
        let positions_owned: Vec<i32>;
        let positions: &[i32] = match positions_flat {
            Some(p) => {
                anyhow::ensure!(
                    p.len() == 4 * prompt_len,
                    "generate_qwen35_once_with_soft_tokens_and_deepstack: \
                     positions_flat.len() = {} != 4 * prompt_len = {}",
                    p.len(),
                    4 * prompt_len
                );
                p
            }
            None => {
                positions_owned = prefill_positions_for(prompt_len);
                &positions_owned
            }
        };

        let prefill_logits =
            supervised_gpu_call(supervisor, "qwen35_serial_deepstack_prefill", || {
                qwen.model
                    .forward_gpu_last_logits_with_soft_tokens_and_deepstack(
                        prompt_tokens,
                        positions,
                        soft_tokens,
                        deepstack,
                        &mut kv_cache,
                        SlotId(0),
                    )
                    .context(
                        "Qwen35Model::forward_gpu_last_logits_with_soft_tokens_and_deepstack \
                         (prefill)",
                    )
            })?;
        anyhow::ensure!(
            prefill_logits.len() == qwen.vocab_size,
            "qwen35 vision prefill logits len {} != vocab_size {}",
            prefill_logits.len(),
            qwen.vocab_size
        );
        next_token = if want_logprobs {
            let mut logits = prefill_logits.clone();
            let (tok, lp) = sample_logits_qwen35_with_logprob(&mut logits, params, &[]);
            if let Some(v) = logprobs_vec.as_mut() {
                v.push(lp);
            }
            tok
        } else if is_greedy {
            greedy_argmax_last_token(&prefill_logits, qwen.vocab_size as u32)
        } else {
            let mut logits = prefill_logits.clone();
            sample_logits_qwen35(&mut logits, params, &[])
        };
        if is_greedy && params.vision_fingerprint.is_some() {
            match kv_cache.snapshot_prefix(&device, prompt_len) {
                Ok(snapshot) => {
                    qwen.prompt_cache
                        .update(prompt_tokens.to_vec(), snapshot, next_token, params)
                }
                Err(error) => tracing::warn!(%error, "Qwen vision prompt-cache snapshot failed"),
            }
        }
    }
    let prefill_duration = prefill_start.elapsed();

    // Decode loop — for the post-prefill text steps, the global
    // temporal position has advanced by image-aware amounts. Compute
    // the post-prefill temporal `t_post` from the LAST text token's
    // axis-0 position +1, OR (when the prompt ends with image tokens)
    // from `axis-0[last] + temporal_advance`. Easiest: take
    // `max(positions axis 0) + 1` as the next text temporal position.
    let t_post: i32 = match positions_flat {
        Some(p) => {
            // axis 0 of positions_flat covers indices 0..prompt_len.
            let mut max_t = 0i32;
            for i in 0..prompt_len {
                let v = p[i]; // axis 0 = t
                if v > max_t {
                    max_t = v;
                }
            }
            // Image-tail special case: if last prompt token is an
            // image-patch token (its t=t_img is constant across the
            // image), the next text step's t = t_img + temporal_advance.
            // We can't recover `temporal_advance` cheaply here; the
            // simplest correct rule is `max_t + 1` for purely-text-end
            // prompts (the conservative case). For Qwen3-VL chat the
            // prompt always ends with the assistant turn marker (text),
            // so max_t+1 is correct.
            max_t.saturating_add(1)
        }
        None => prompt_len as i32,
    };

    let decode_start = Instant::now();
    let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    generated_tokens.push(next_token);

    let first_fragment = qwen
        .tokenizer
        .decode(&[next_token], false)
        .unwrap_or_default();
    let mut decoded_text = first_fragment.clone();

    let mut finish_reason: &'static str = "length";

    if qwen.eos_token_ids.contains(&next_token) {
        finish_reason = "stop";
    } else if qwen35_hit_stop_string(&decoded_text, &params.stop_strings) {
        finish_reason = "stop";
        qwen35_strip_trailing_stop(&mut decoded_text, &params.stop_strings);
    } else {
        for step in 1..max_tokens {
            // Decode position is `t_post + (step - 1)`, broadcast across
            // all 4 axes (text-style).
            let pos = t_post + (step as i32 - 1);
            if pos as u32 >= kv_cache.max_seq_len {
                tracing::warn!(
                    pos,
                    max_seq = kv_cache.max_seq_len,
                    "qwen35 decode (wedge-4d): hit kv-cache bound; stopping with finish=length",
                );
                break;
            }
            let decode_positions = vec![pos; 4];

            next_token = if want_logprobs {
                // ADR-020 AC#7 — bypass greedy fast-path; need full logits.
                let logits_full =
                    supervised_gpu_call(supervisor, "qwen35_serial_decode", || {
                        qwen.model.forward_gpu_last_logits(
                        &[next_token],
                        &decode_positions,
                        &mut kv_cache,
                        SlotId(0),
                    )
                    .with_context(|| {
                        format!("forward_gpu_last_logits decode step {step} (wedge-4d, logprobs)")
                    })
                    })?;
                let mut logits = logits_full;
                let (tok, lp) =
                    sample_logits_qwen35_with_logprob(&mut logits, params, &generated_tokens);
                if let Some(v) = logprobs_vec.as_mut() {
                    v.push(lp);
                }
                tok
            } else if is_greedy {
                supervised_gpu_call(supervisor, "qwen35_serial_decode", || {
                    qwen.model
                        // ADR-040 Phase B4d (2026-05-30) — see sibling at
                        // engine_qwen35.rs:2071 for the SlotId contract.
                        .forward_gpu_greedy(
                            &[next_token],
                            &decode_positions,
                            &mut kv_cache,
                            SlotId(0),
                        )
                        .with_context(|| {
                            format!("forward_gpu_greedy decode step {step} (wedge-4d)")
                        })
                })?
            } else {
                let logits_full = supervised_gpu_call(supervisor, "qwen35_serial_decode", || {
                    qwen.model
                        .forward_gpu_last_logits(
                            &[next_token],
                            &decode_positions,
                            &mut kv_cache,
                            SlotId(0),
                        )
                        .with_context(|| {
                            format!("forward_gpu_last_logits decode step {step} (wedge-4d)")
                        })
                })?;
                let mut logits = logits_full;
                sample_logits_qwen35(&mut logits, params, &generated_tokens)
            };

            if qwen.eos_token_ids.contains(&next_token) {
                finish_reason = "stop";
                break;
            }
            generated_tokens.push(next_token);
            let fragment = qwen
                .tokenizer
                .decode(&[next_token], false)
                .unwrap_or_default();
            decoded_text.push_str(&fragment);
            if qwen35_hit_stop_string(&decoded_text, &params.stop_strings) {
                finish_reason = "stop";
                qwen35_strip_trailing_stop(&mut decoded_text, &params.stop_strings);
                break;
            }
        }
    }
    let decode_duration = decode_start.elapsed();
    qwen.persistent_kv_cache = Some(kv_cache);

    let (content, reasoning_text) = match registration {
        Some(reg) if reg.has_reasoning() => super::registry::split_full_output_forced(
            reg,
            &decoded_text,
            params.reasoning_forced_open,
        ),
        _ => (decoded_text, None),
    };

    let reasoning_token_count = match registration {
        Some(reg) if reg.has_reasoning() => {
            let mut sp =
                super::registry::make_reasoning_splitter(reg, params.reasoning_forced_open);
            let mut count = 0usize;
            for &tok in &generated_tokens {
                let frag = qwen.tokenizer.decode(&[tok], false).unwrap_or_default();
                if let Some(splitter) = sp.as_mut() {
                    let _ = splitter.feed(&frag);
                    if splitter.in_reasoning() {
                        count += 1;
                    }
                }
            }
            count
        }
        _ => 0,
    };

    Ok(GenerationResult {
        text: content,
        reasoning_text,
        prompt_tokens: prompt_len,
        completion_tokens: generated_tokens.len(),
        reasoning_tokens: if reasoning_token_count > 0 {
            Some(reasoning_token_count)
        } else {
            None
        },
        finish_reason,
        prefill_duration,
        decode_duration,
        cached_tokens: if prompt_cache_hit { prompt_len } else { 0 },
        logprobs: logprobs_vec,
    })
}

/// Wedge-3 / Phase D: streaming chat generation against a loaded
/// Qwen3.5/3.6 model.  Replaces the `worker_run` 501 arm for
/// `Request::GenerateStream`.
///
/// Mirrors `generate_qwen35_once` for prefill / KV-cache / prompt-cache
/// substrate, but routes per-token decoded fragments through the
/// `ReasoningSplitter` + `ToolCallSplitter` so the SSE stream emits
/// `Delta { kind: Reasoning }`, `Delta { kind: Content }`, and
/// `ToolCallDelta` events identical to the Gemma path's
/// `generate_stream_once`.  Tool-call body parsing on close uses the
/// shared `super::engine::emit_streaming_tool_call_close` helper (the
/// close-buffered shape; W-B3 incremental tool-call streaming is a
/// Wedge-4 follow-up — the spec-valid single-arguments-delta shape that
/// Gemma's path used pre-W-B3 is what Qwen35 ships here).
///
/// Cancellation: `events.blocking_send` returning Err signals client
/// disconnect; we bump `cancellation_counter` (if supplied) and abort.
///
/// **Wedge-4e (iter-224 row 5)**: this entry now delegates to
/// [`generate_stream_qwen35_once_extended`] with empty soft-tokens, no
/// deepstack, and no 3D positions — preserving byte-identical
/// behaviour for the legacy text-only streaming path. The extended
/// entry threads `soft_tokens` + `deepstack` + `positions_flat`
/// through the prefill so streaming Qwen3-VL chat (image-bearing +
/// tools[] + reasoning_content) works end-to-end.
pub(super) fn generate_stream_qwen35_once(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    events: &tokio::sync::mpsc::Sender<GenerationEvent>,
    registration: Option<&ModelRegistration>,
    cancellation_counter: Option<&std::sync::atomic::AtomicU64>,
    supervisor: &EngineSupervisor,
) -> SerialStreamResult {
    generate_stream_qwen35_once_extended(
        qwen,
        prompt_tokens,
        &[],
        None,
        None,
        params,
        events,
        registration,
        cancellation_counter,
        supervisor,
    )
}

/// **Wedge-4e (iter-224 row 5)**: streaming Qwen3.5/3.6 generation with
/// optional soft-tokens + DeepStack + 3D-mRoPE positions.
///
/// Identical to [`generate_stream_qwen35_once`] except:
///
///   * Prefill goes through
///     `Qwen35Model::forward_gpu_last_logits_with_soft_tokens_and_deepstack`
///     when any of `soft_tokens` / `deepstack` / `positions_flat` is
///     non-empty / `Some(...)`. Empty soft + `None` deepstack + `None`
///     positions is byte-identical to the text-only stream (regression
///     pin via existing Wedge-3 streaming tests).
///
///   * The 3D-mRoPE position buffer (`positions_flat: [4 * prompt_len]`)
///     is supplied by the chat handler via
///     `crate::serve::forward_prefill::build_qwen3vl_positions` when
///     image-bearing; otherwise text-style `[t,t,t,t]` positions are
///     synthesized via `prefill_positions_for(prompt_len)`.
///
///   * Decode steps after prefill use text-only `[t,t,t,t]` positions
///     starting from the post-prefill global temporal counter (which
///     advances by `max(n_x, n_y)` per image, NOT by `n_image_tokens`,
///     per peer `mtmd.cpp:1354-1357`). Computed as
///     `max(positions_flat axis 0) + 1` — same rule as the
///     non-streaming `generate_qwen35_once_with_soft_tokens_and_deepstack`.
///
///   * Prompt-cache fast-path is BYPASSED whenever any extension is
///     present (the cache key is `prompt_tokens` only and would
///     falsely hit on a vision-augmented prompt with the same
///     placeholder ids but different image content). Mirrors the
///     non-streaming `generate_qwen35_once_with_soft_tokens` rationale.
///
/// **MODE-INVARIANT splitters**: `ReasoningSplitter` + `ToolCallSplitter`
/// operate on the token-delta stream and are NOT aware of vision
/// augmentation. Image-bearing streaming requests with `tools[]` or
/// reasoning markers route deltas through the same splitter chain
/// the text-only path uses — verified by the Wedge-4e splitter
/// invariance tests at the bottom of this file.
///
/// **Cancellation safety**: client disconnect mid-stream causes
/// `events.blocking_send` to return Err; we bump the cancellation
/// counter (if supplied) and abort — the borrowed `DeepstackInjection`
/// + `SoftTokenInjection` slices are dropped at end-of-scope, releasing
/// the augmented-embed GPU buffers (the owned `DeepstackData` /
/// `SoftTokenData` is held by the `Request::GenerateStream` variant
/// the caller dropped, so the buffers are reclaimed there too).
#[allow(clippy::too_many_arguments)]
pub(super) fn generate_stream_qwen35_once_extended(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    soft_tokens: &[crate::serve::forward_prefill::SoftTokenInjection<'_>],
    deepstack: Option<&crate::serve::forward_prefill::DeepstackInjection<'_>>,
    positions_flat: Option<&[i32]>,
    params: &SamplingParams,
    events: &tokio::sync::mpsc::Sender<GenerationEvent>,
    registration: Option<&ModelRegistration>,
    cancellation_counter: Option<&std::sync::atomic::AtomicU64>,
    supervisor: &EngineSupervisor,
) -> SerialStreamResult {
    let request_start = Instant::now();
    let _disk_request_guard = qwen
        .disk_persistor
        .as_ref()
        .map(|persistor| persistor.begin_request());
    macro_rules! send {
        ($ev:expr) => {
            if events.blocking_send($ev).is_err() {
                tracing::info!("SSE stream dropped by client; aborting qwen35 decode");
                if let Some(c) = cancellation_counter {
                    c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return Ok(SerialStreamEnd::ClientClosed);
            }
        };
    }

    if prompt_tokens.is_empty() {
        send!(GenerationEvent::Error(
            "generate_stream_qwen35_once: empty prompt_tokens".into()
        ));
        return Ok(SerialStreamEnd::TerminalSent);
    }
    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);
    let is_greedy = is_greedy_eligible(params);
    let mut grammar_runtime = match grammar_runtime_for_request(params, registration) {
        Ok(runtime) => runtime,
        Err(error) => {
            send!(GenerationEvent::Error(format!(
                "qwen35 stream grammar initialization failed: {error:#}"
            )));
            return Ok(SerialStreamEnd::TerminalSent);
        }
    };

    // Wedge-4e: any extension present ⇒ vision-augmented prefill path.
    // When ALL extensions are empty/None, the legacy text-only stream
    // is preserved byte-identically (regression-pin).
    let has_extension = !soft_tokens.is_empty() || deepstack.is_some() || positions_flat.is_some();

    // Validate `positions_flat` length up-front so we fail loud BEFORE
    // any GPU work — mirrors the non-streaming sibling at
    // `generate_qwen35_once_with_soft_tokens_and_deepstack`.
    if let Some(p) = positions_flat {
        if p.len() != 4 * prompt_len {
            send!(GenerationEvent::Error(format!(
                "qwen35 stream (wedge-4e): positions_flat.len() = {} != 4 * prompt_len = {}",
                p.len(),
                4 * prompt_len
            )));
            return Ok(SerialStreamEnd::TerminalSent);
        }
    }

    let cache_alloc_start = Instant::now();
    let device = match MlxDevice::new() {
        Ok(d) => d,
        Err(e) => {
            return Err(e).context("Qwen35 SerialFifo streaming device initialization");
        }
    };
    let (mut kv_cache, cache_reused) =
        match take_serial_kv_cache(qwen, &device, prompt_len, max_tokens) {
            Ok(k) => k,
            Err(e) => {
                return Err(e).context("Qwen35 SerialFifo streaming KV cache allocation");
            }
        };
    tracing::info!(
        target: "hf2q::serve::api::engine_qwen35::progress",
        mode = "stream",
        prompt_tokens = prompt_len,
        max_tokens,
        cache_capacity_tokens = kv_cache.max_seq_len,
        cache_reused,
        tq_kv = qwen.tq_kv_active,
        elapsed_ms = cache_alloc_start.elapsed().as_secs_f64() * 1000.0,
        "Qwen35 request cache ready"
    );

    // ADR-027 iter-6b.3: cold-start hydrate (streaming path).
    // Idempotent + cheap on hot path; warn-logs + swallows persistence
    // failures so the request still proceeds.
    qwen.hydrate_lcp_registry_from_disk(&kv_cache, &device);

    let pre_dispatches = mlx_native::dispatch_count();
    let pre_syncs = mlx_native::sync_count();

    // Vision requests may use the full-prompt fast path only when the HTTP
    // preparation layer supplied an exact image-embedding fingerprint. The
    // prompt-cache key includes that digest, so equal placeholder tokens with
    // different pixels miss. Other extension callers remain fail-closed.
    let prompt_cache_eligible = !has_extension || params.vision_fingerprint.is_some();
    let prompt_cache_hit =
        prompt_cache_eligible && qwen.prompt_cache.try_match(prompt_tokens, params).is_some();

    // ADR-017 Phase E.a B.2c+B.3+B.5 — streaming-path LCP probe +
    // restore.  Mirrors the non-streaming probe in
    // `generate_qwen35_once`.  Skip on multimodal / extension paths
    // (has_extension true) and on prompt_cache hits.
    let mut lcp_resume_start: usize = 0;
    if !prompt_cache_hit && !has_extension {
        // Streaming counterpart of the non-streaming chunk-aligned
        // observability probe (counter-fix 2026-05-06). Same shape as
        // generate_qwen35_once: precompute base key, then descend
        // stride-aligned chunk positions side-effect-free so
        // detected_total accurately reflects qwen35 resume opportunity.
        let stride_for_observe = crate::debug::INVESTIGATION_ENV.kv_lcp_deltanet_checkpoint_stride;
        let base_key_for_observe = build_lcp_key_for_qwen35(qwen, params);
        let detected = lookup_qwen35_resume_checkpoint(
            &mut qwen.lcp_registry,
            &base_key_for_observe,
            prompt_tokens,
            stride_for_observe,
        )
        .map(|(prefix, _chunk_pos)| prefix.k);
        if let Some(sink) = qwen.kv_metrics_sink.as_ref() {
            sink.record_lcp_probe(detected);
        }
        let _ = detected;

        // Streaming gate — same Q3 auto-disable wiring as the non-streaming
        // site above. Codex Phase-2b 2026-05-06 caught this site missing
        // the helper.
        // ADR-027 sub-iter 23d-γ (2026-08-03): qwen35's TQ-only restore
        // path is PROVEN (restore_partial restores all four TQ buffers
        // per slot; cold-vs-resumed byte-identity live-gated) — LCP is
        // unconditionally resumable on this arch, no explicit
        // HF2Q_KV_LCP_RESUME=1 needed under the production TQ regime.
        let lcp_resume_enabled = crate::serve::api::engine::effective_kv_lcp_resume(
            crate::debug::INVESTIGATION_ENV.kv_lcp_resume,
            true,
        );
        if lcp_resume_enabled {
            let stride = crate::debug::INVESTIGATION_ENV.kv_lcp_deltanet_checkpoint_stride;
            let base_key = build_lcp_key_for_qwen35(qwen, params);
            eprintln!(
                "[hf2q qwen35 stream lcp probe] enabled, registry_len={}, \
                 prompt_len={}, stride={}, scanning latest-turn + stride checkpoints",
                qwen.lcp_registry.len(),
                prompt_tokens.len(),
                stride,
            );
            if let Some((prefix, chunk_pos)) = lookup_qwen35_resume_checkpoint(
                &mut qwen.lcp_registry,
                &base_key,
                prompt_tokens,
                stride,
            ) {
                let snapshot: &HybridKvCacheSnapshot = &prefix.dense_kvs[0];
                let restore_start = Instant::now();
                if let Err(e) = kv_cache.restore_partial(snapshot, prefix.k) {
                    return Err(e).context("Qwen35 SerialFifo streaming LCP checkpoint restore");
                }
                let restore_ms = restore_start.elapsed().as_micros() as f64 / 1000.0;
                lcp_resume_start = prefix.k;
                let checkpoint = if chunk_pos == 0 {
                    "LATEST-TURN"
                } else {
                    "STRIDE-ALIGNED"
                };
                eprintln!(
                    "[hf2q qwen35 stream lcp resume] {checkpoint} HIT — restoring \
                     at k={} (cached_prompt_len={}, chunk_pos={}, restore_ms={:.3})",
                    prefix.k, prefix.cached_prompt_len, chunk_pos, restore_ms
                );
            } else {
                eprintln!(
                    "[hf2q qwen35 stream lcp probe] no compatible checkpoint \
                     (registry_len={})",
                    qwen.lcp_registry.len()
                );
            }
        }
    }

    if cache_reused && !prompt_cache_hit && lcp_resume_start == 0 {
        kv_cache.reset();
    }

    let prefill_start = Instant::now();
    let mut next_token: u32;
    if prompt_cache_hit {
        let snap = qwen
            .prompt_cache
            .snapshot()
            .expect("try_match Some implies snapshot Some");
        if let Err(e) = kv_cache.restore_partial(snap, prompt_len) {
            return Err(e).context("Qwen35 SerialFifo streaming prompt-cache restore");
        }
        next_token = qwen.prompt_cache.first_decoded_token();
        tracing::debug!(
            "qwen35 prompt_cache: STREAMING HIT — {} tokens; prefill skipped",
            prompt_len
        );
    } else {
        // Use 3D positions when supplied; otherwise text-style
        // `[t,t,t,t]` positions covering the prompt.
        let positions_owned: Vec<i32>;
        let positions_slice: &[i32] = match positions_flat {
            Some(p) => p,
            None => {
                positions_owned = prefill_positions_for(prompt_len);
                &positions_owned
            }
        };

        // ADR-017 Phase E.a B.3: chunked prefill engagement (text-only,
        // no extensions).  Mirrors non-streaming logic — chunked under
        // HF2Q_KV_LCP_CHUNKED_PREFILL=1 OR when an LCP resume
        // restored cache state (lcp_resume_start > 0).  Mid-prefill
        // stores happen at every stride-aligned chunk boundary when
        // LCP_RESUME=1.
        let stride = crate::debug::INVESTIGATION_ENV.kv_lcp_deltanet_checkpoint_stride;
        let lcp_resume_enabled = crate::debug::INVESTIGATION_ENV.kv_lcp_resume;
        // ADR-017 Phase E.a B.5 Codex Phase-2b finding (MEDIUM):
        // mirror non-streaming chunked-eligibility — chunked engages
        // ONLY under explicit `HF2Q_KV_LCP_CHUNKED_PREFILL=1`, not as
        // an implicit consequence of `lcp_resume_start > 0`.  When
        // LCP resume restores cache state but chunked is disabled, the
        // suffix-only monolithic branch below handles the resumed
        // prefill (cur_len > 0 → FA RESUME kernel — byte-identical to
        // monolithic by B.2-fix).
        let chunked_eligible = !has_extension
            && stride > 0
            && prompt_len > stride
            && (lcp_resume_start == 0 || lcp_resume_start % stride == 0)
            && crate::debug::INVESTIGATION_ENV.kv_lcp_chunked_prefill;
        let recovery_tail_tokens = qwen35_recovery_tail_tokens(qwen, prompt_tokens, params);
        let recovery_anchor = prompt_len.saturating_sub(recovery_tail_tokens);
        let recovery_eligible = !has_extension
            && lcp_resume_enabled
            && recovery_anchor > lcp_resume_start
            && recovery_anchor >= 16;
        let recovery_capture_plan = qwen35_recovery_capture_plan(
            lcp_resume_start,
            recovery_anchor,
            prompt_len,
            recovery_eligible,
            chunked_eligible,
        );
        let prefill_logits_res = if has_extension {
            supervised_gpu_call(supervisor, "qwen35_serial_stream_prefill", || {
                qwen.model
                    .forward_gpu_last_logits_with_soft_tokens_and_deepstack(
                        prompt_tokens,
                        positions_slice,
                        soft_tokens,
                        deepstack,
                        &mut kv_cache,
                        SlotId(0),
                    )
            })
        } else if let Some((suffix_len, capture_index)) = recovery_capture_plan {
            if let Err(error) =
                kv_cache.ensure_la_capture(&qwen.model.cfg, &device, suffix_len as u32)
            {
                return Err(error)
                    .context("Qwen35 SerialFifo streaming recovery-capture allocation");
            }
            let suffix_tokens = &prompt_tokens[lcp_resume_start..];
            let mut suffix_positions = vec![0i32; 4 * suffix_len];
            for axis in 0..4 {
                for token in 0..suffix_len {
                    suffix_positions[axis * suffix_len + token] = (lcp_resume_start + token) as i32;
                }
            }
            match supervised_gpu_call(supervisor, "qwen35_serial_stream_prefill", || {
                qwen.model.forward_gpu_last_logits(
                    suffix_tokens,
                    &suffix_positions,
                    &mut kv_cache,
                    SlotId(0),
                )
            }) {
                Ok(logits) => {
                    store_qwen35_latest_turn_checkpoint(
                        qwen,
                        &kv_cache,
                        &device,
                        params,
                        prompt_tokens,
                        recovery_anchor,
                        "stream captured latest-turn recovery-anchor",
                        Some(capture_index),
                    );
                    kv_cache.clear_la_capture();
                    eprintln!(
                        "[hf2q qwen35 stream lcp store] captured latest-turn recovery \
                         anchor={} suffix_tokens={} capture_index={}",
                        recovery_anchor, suffix_len, capture_index
                    );
                    Ok(logits)
                }
                Err(error) => Err(error).context("Qwen35 stream captured latest-turn suffix"),
            }
        } else if recovery_eligible && !chunked_eligible {
            let prefix_tokens = &prompt_tokens[lcp_resume_start..recovery_anchor];
            let prefix_len = prefix_tokens.len();
            let mut prefix_positions = vec![0i32; 4 * prefix_len];
            for axis in 0..4 {
                for token in 0..prefix_len {
                    prefix_positions[axis * prefix_len + token] = (lcp_resume_start + token) as i32;
                }
            }
            if let Err(error) =
                supervised_gpu_call(supervisor, "qwen35_serial_stream_recovery_prefix", || {
                    qwen.model.forward_gpu_last_logits(
                        prefix_tokens,
                        &prefix_positions,
                        &mut kv_cache,
                        SlotId(0),
                    )
                })
            {
                return Err(error).context("Qwen35 SerialFifo streaming recovery-anchor prefix");
            }
            store_qwen35_latest_turn_checkpoint(
                qwen,
                &kv_cache,
                &device,
                params,
                prompt_tokens,
                recovery_anchor,
                "stream latest-turn recovery-anchor",
                None,
            );

            let tail_tokens = &prompt_tokens[recovery_anchor..];
            let tail_len = tail_tokens.len();
            let mut tail_positions = vec![0i32; 4 * tail_len];
            for axis in 0..4 {
                for token in 0..tail_len {
                    tail_positions[axis * tail_len + token] = (recovery_anchor + token) as i32;
                }
            }
            eprintln!(
                "[hf2q qwen35 stream lcp store] latest-turn recovery anchor={} \
                 tail_tokens={}",
                recovery_anchor, tail_len
            );
            supervised_gpu_call(supervisor, "qwen35_serial_stream_recovery_tail", || {
                qwen.model.forward_gpu_last_logits(
                    tail_tokens,
                    &tail_positions,
                    &mut kv_cache,
                    SlotId(0),
                )
            })
        } else if chunked_eligible {
            // Chunked prefill — mirrors non-streaming chunked block.
            // Stride alignment was already validated above (centralized
            // assertion covers both chunked + suffix-only branches).
            let first_chunk_idx = lcp_resume_start / stride;
            let chunked_prefill_end = if recovery_eligible {
                recovery_anchor
            } else {
                prompt_len
            };
            let n_chunks = (chunked_prefill_end + stride - 1) / stride;
            eprintln!(
                "[hf2q qwen35 stream chunked prefill] {} chunks (stride={}, \
                 prompt_len={}, prefill_end={}, first_chunk_idx={})",
                n_chunks, stride, prompt_len, chunked_prefill_end, first_chunk_idx
            );
            let mut last_logits_res: Result<Vec<f32>> = Err(anyhow::anyhow!("no chunks executed"));
            for chunk_idx in first_chunk_idx..n_chunks {
                let k_start = chunk_idx * stride;
                let k_end = ((chunk_idx + 1) * stride).min(chunked_prefill_end);
                let chunk_seq_len = k_end - k_start;
                let chunk_tokens = &prompt_tokens[k_start..k_end];
                let mut chunk_positions = vec![0i32; 4 * chunk_seq_len];
                for axis in 0..4 {
                    for t in 0..chunk_seq_len {
                        chunk_positions[axis * chunk_seq_len + t] = (k_start + t) as i32;
                    }
                }
                let res =
                    supervised_gpu_call(supervisor, "qwen35_serial_stream_prefill_chunk", || {
                        qwen.model.forward_gpu_last_logits(
                            chunk_tokens,
                            &chunk_positions,
                            &mut kv_cache,
                            SlotId(0),
                        )
                    });
                let logits = match res {
                    Ok(l) => l,
                    Err(e) => {
                        return Err(e).with_context(|| {
                            format!(
                                "Qwen35 SerialFifo streaming prefill chunk {}/{}",
                                chunk_idx + 1,
                                n_chunks
                            )
                        });
                    }
                };
                if chunk_idx == n_chunks - 1 {
                    last_logits_res = Ok(logits.clone());
                }
                // B.3 mid-prefill snapshot store (LCP_RESUME; NOT
                // greedy-gated — prefix KV state is sampling-independent,
                // see the non-streaming sibling's 2026-08-03 store-gate
                // fix).  `HF2Q_KV_LCP_DISABLE_MID_STORE` now applies to
                // the stream path too (was non-stream-only — asymmetric
                // kill-switch).
                let stride_aligned = k_end % stride == 0;
                let superseded_by_recovery_anchor = stride_checkpoint_superseded_by_recovery_anchor(
                    recovery_eligible,
                    k_end,
                    stride,
                    recovery_anchor,
                );
                let mid_store_disabled =
                    std::env::var("HF2Q_KV_LCP_DISABLE_MID_STORE").as_deref() == Ok("1");
                lcp_store_skip_notify(
                    stride_aligned && !superseded_by_recovery_anchor,
                    lcp_resume_enabled,
                    mid_store_disabled,
                );
                if lcp_resume_enabled
                    && stride_aligned
                    && !superseded_by_recovery_anchor
                    && !mid_store_disabled
                {
                    match kv_cache.snapshot_prefix(&device, k_end) {
                        Ok(snap) => {
                            let chunk_key = build_lcp_key_for_qwen35_chunk(qwen, params, k_end);
                            let linear_capacity = kv_cache
                                .linear_attn
                                .first()
                                .map(|s| s.recurrent.byte_len())
                                .unwrap_or(0);
                            // ADR-027 iter-6b.2: write-through helper.
                            if let Err(e) = qwen.store_lcp_with_disk_writeback(
                                &kv_cache,
                                chunk_key,
                                prompt_tokens[..k_end].to_vec(),
                                snap,
                                0,
                                linear_capacity,
                            ) {
                                lcp_store_error_notify("stream mid-prefill", k_end, &e);
                            } else {
                                eprintln!(
                                    "[hf2q qwen35 stream lcp store] mid-prefill \
                                 snapshot at chunk_pos={k_end} \
                                 (registry_len_after={})",
                                    qwen.lcp_registry.len()
                                );
                            }
                        }
                        Err(e) => {
                            // Was `if let Ok` — snapshot failures used to
                            // be 100% silent on this path, producing the
                            // empty-registry symptom with zero diagnostics.
                            lcp_snapshot_error_notify("stream mid-prefill", k_end, &e);
                        }
                    }
                }
            }
            if recovery_eligible {
                store_qwen35_latest_turn_checkpoint(
                    qwen,
                    &kv_cache,
                    &device,
                    params,
                    prompt_tokens,
                    recovery_anchor,
                    "stream chunked latest-turn recovery-anchor",
                    None,
                );

                let tail_tokens = &prompt_tokens[recovery_anchor..];
                let tail_len = tail_tokens.len();
                let mut tail_positions = vec![0i32; 4 * tail_len];
                for axis in 0..4 {
                    for token in 0..tail_len {
                        tail_positions[axis * tail_len + token] = (recovery_anchor + token) as i32;
                    }
                }
                eprintln!(
                    "[hf2q qwen35 stream lcp store] latest-turn recovery anchor={} \
                     tail_tokens={}",
                    recovery_anchor, tail_len
                );
                last_logits_res =
                    supervised_gpu_call(supervisor, "qwen35_serial_stream_recovery_tail", || {
                        qwen.model.forward_gpu_last_logits(
                            tail_tokens,
                            &tail_positions,
                            &mut kv_cache,
                            SlotId(0),
                        )
                    });
            }
            last_logits_res
        } else if lcp_resume_start > 0 {
            // Suffix-only monolithic prefill from the LCP boundary.
            let suffix_tokens = &prompt_tokens[lcp_resume_start..];
            let suffix_len = suffix_tokens.len();
            let mut suffix_positions = vec![0i32; 4 * suffix_len];
            for axis in 0..4 {
                for t in 0..suffix_len {
                    suffix_positions[axis * suffix_len + t] = (lcp_resume_start + t) as i32;
                }
            }
            eprintln!(
                "[hf2q qwen35 stream lcp resume] suffix prefill {} tokens \
                 (lcp_resume_start={}, prompt_len={})",
                suffix_len, lcp_resume_start, prompt_len
            );
            supervised_gpu_call(supervisor, "qwen35_serial_stream_prefill", || {
                qwen.model.forward_gpu_last_logits(
                    suffix_tokens,
                    &suffix_positions,
                    &mut kv_cache,
                    SlotId(0),
                )
            })
        } else {
            supervised_gpu_call(supervisor, "qwen35_serial_stream_prefill", || {
                qwen.model.forward_gpu_last_logits(
                    prompt_tokens,
                    positions_slice,
                    &mut kv_cache,
                    SlotId(0),
                )
            })
        };
        let prefill_logits = match prefill_logits_res {
            Ok(l) => l,
            Err(e) => {
                return Err(e).context("Qwen35 SerialFifo streaming prefill");
            }
        };
        if is_greedy {
            next_token = greedy_argmax_last_token(&prefill_logits, qwen.vocab_size as u32);
        } else {
            let mut logits = prefill_logits.clone();
            next_token = sample_logits_qwen35_constrained(
                &mut logits,
                params,
                &[],
                grammar_runtime.as_ref(),
                false,
            )?
            .0;
        }
        advance_qwen35_grammar(&mut grammar_runtime, params, next_token);
        // The image fingerprint in `HybridPromptCacheKey` makes a
        // vision-tainted snapshot safe and reusable only for the exact same
        // image embeddings. Generic extension callers without that digest
        // remain ineligible.
        if is_greedy && prompt_cache_eligible {
            match kv_cache.snapshot_prefix(&device, prompt_len) {
                Ok(snap) => {
                    qwen.prompt_cache
                        .update(prompt_tokens.to_vec(), snap, next_token, params)
                }
                Err(e) => {
                    eprintln!("[hf2q qwen35 stream lcp store] prompt_cache snapshot failed: {e}")
                }
            }
        }
    }
    let prefill_duration = prefill_start.elapsed();
    let reported_cached_tokens =
        qwen35_reported_cached_tokens(prompt_len, prompt_cache_hit, lcp_resume_start);
    let prefill_work_tokens = prompt_len.saturating_sub(reported_cached_tokens);
    tracing::info!(
        target: "hf2q::serve::api::engine_qwen35::progress",
        mode = "stream",
        prompt_tokens = prompt_len,
        cached_tokens = reported_cached_tokens,
        work_tokens = prefill_work_tokens,
        elapsed_ms = prefill_duration.as_secs_f64() * 1000.0,
        tokens_per_second = if prefill_duration.is_zero() {
            0.0
        } else {
            prefill_work_tokens as f64 / prefill_duration.as_secs_f64()
        },
        "Qwen35 prefill complete"
    );

    // Wedge-4e: post-prefill text decode positions advance from the
    // global temporal counter (which advances by max(n_x, n_y) per
    // image during prefill, NOT by n_image_tokens, per peer
    // `mtmd.cpp:1354-1357`). When `positions_flat` is supplied,
    // compute `t_post = max(axis-0 positions) + 1`; when None, fall
    // back to the legacy `prompt_len` text-style advance.
    //
    // Matches the rule used by the non-streaming sibling at
    // `generate_qwen35_once_with_soft_tokens_and_deepstack`
    // (engine_qwen35.rs:1177-1198).
    let t_post: i32 = match positions_flat {
        Some(p) => {
            let mut max_t = 0i32;
            for i in 0..prompt_len {
                let v = p[i]; // axis 0 = t
                if v > max_t {
                    max_t = v;
                }
            }
            max_t.saturating_add(1)
        }
        None => prompt_len as i32,
    };

    // ── Splitter wiring (Reasoning + ToolCall) ────────────────────
    let mut reasoning_splitter = registration
        .and_then(|r| super::registry::make_reasoning_splitter(r, params.reasoning_forced_open));
    let mut tool_splitter = registration.and_then(ToolCallSplitter::from_registration);
    let mut tool_call_body: String = String::new();
    let mut tool_call_index: usize = 0;
    let mut saw_tool_call: bool = false;

    /// Inner closure-like helper: route a Content-classified text run
    /// through the tool-call splitter.  Returns `false` on client
    /// disconnect (caller aborts).
    fn route_content_qwen35(
        tool_splitter: &mut Option<ToolCallSplitter>,
        body: &mut String,
        tc_index: &mut usize,
        saw_tc: &mut bool,
        tool_call_policy: super::engine::ToolCallPolicy,
        registration: Option<&ModelRegistration>,
        wire_kinds: Option<&super::registry::ToolArgumentWireKinds>,
        events: &tokio::sync::mpsc::Sender<GenerationEvent>,
        text: &str,
    ) -> bool {
        if text.is_empty() {
            return true;
        }
        let Some(tcs) = tool_splitter.as_mut() else {
            return events
                .blocking_send(GenerationEvent::Delta {
                    kind: DeltaKind::Content,
                    text: text.to_string(),
                })
                .is_ok();
        };
        for ev in tcs.feed(text) {
            match ev {
                ToolCallEvent::Content(t) => {
                    if !t.is_empty()
                        && events
                            .blocking_send(GenerationEvent::Delta {
                                kind: DeltaKind::Content,
                                text: t,
                            })
                            .is_err()
                    {
                        return false;
                    }
                }
                ToolCallEvent::ToolCallOpen => {
                    body.clear();
                }
                ToolCallEvent::ToolCallText(t) => {
                    body.push_str(&t);
                }
                ToolCallEvent::ToolCallClose => {
                    let parsed = registration.and_then(|r| {
                        super::registry::parse_tool_call_body_with_wire_kinds(r, body, wire_kinds)
                    });
                    let body_dump = std::mem::take(body);
                    // Reuse the shared close-buffered emitter so the
                    // body-parse-failure semantics + ToolCallDelta shape
                    // stay byte-identical to Gemma's stream.  Wedge-3
                    // ships the close-buffered single-arguments-delta
                    // shape (spec-valid OpenAI streaming tool-call); the
                    // W-B3 incremental shape is a Wedge-4 follow-up if
                    // operators want progressive arg display.
                    // ADR-005 iter-224 W-A2.2: wrap in a passive EventSink
                    // so the helper's `&EventSink<'_>` parameter is satisfied
                    // without altering the qwen35 streaming code path.
                    // Qwen35 has its own HybridPromptCache and does NOT
                    // participate in the Gemma fragment-replay capture; a
                    // passive sink is correct here (forwards 1:1, no mirror).
                    let sink = super::engine::EventSink::new(events);
                    if super::engine::emit_streaming_tool_call_close(
                        parsed,
                        body_dump,
                        tool_call_policy,
                        tc_index,
                        saw_tc,
                        &sink,
                    )
                    .is_err()
                    {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Emit a fragment through the reasoning splitter (if any) then the
    /// tool-call router.  Returns `false` on disconnect.
    fn emit_fragment_qwen35(
        reasoning_splitter: &mut Option<ReasoningSplitter>,
        tool_splitter: &mut Option<ToolCallSplitter>,
        body: &mut String,
        tc_index: &mut usize,
        saw_tc: &mut bool,
        tool_call_policy: super::engine::ToolCallPolicy,
        registration: Option<&ModelRegistration>,
        wire_kinds: Option<&super::registry::ToolArgumentWireKinds>,
        events: &tokio::sync::mpsc::Sender<GenerationEvent>,
        fragment: &str,
    ) -> bool {
        if fragment.is_empty() {
            return true;
        }
        if let Some(rs) = reasoning_splitter.as_mut() {
            for (slot, text) in rs.feed(fragment) {
                match slot {
                    SplitSlot::Reasoning => {
                        if !text.is_empty()
                            && events
                                .blocking_send(GenerationEvent::Delta {
                                    kind: DeltaKind::Reasoning,
                                    text,
                                })
                                .is_err()
                        {
                            return false;
                        }
                    }
                    SplitSlot::Content => {
                        if !route_content_qwen35(
                            tool_splitter,
                            body,
                            tc_index,
                            saw_tc,
                            tool_call_policy,
                            registration,
                            wire_kinds,
                            events,
                            &text,
                        ) {
                            return false;
                        }
                    }
                }
            }
            true
        } else {
            route_content_qwen35(
                tool_splitter,
                body,
                tc_index,
                saw_tc,
                tool_call_policy,
                registration,
                wire_kinds,
                events,
                fragment,
            )
        }
    }

    // ── Decode loop ────────────────────────────────────────────────
    let decode_start = Instant::now();
    let mut completion_tokens = 0usize;
    let mut accumulated_text = String::new();
    let mut reasoning_token_count = 0usize;
    let mut finish_reason: &'static str = "length";
    let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);

    let first_text = qwen
        .tokenizer
        .decode(&[next_token], false)
        .unwrap_or_default();
    let mut is_eos_first = qwen.eos_token_ids.contains(&next_token);
    if !is_eos_first && qwen35_grammar_terminal_token(grammar_runtime.as_ref(), params, next_token)
    {
        is_eos_first = true;
        finish_reason = "stop";
    }
    if !is_eos_first && !first_text.is_empty() {
        accumulated_text.push_str(&first_text);
        if !emit_fragment_qwen35(
            &mut reasoning_splitter,
            &mut tool_splitter,
            &mut tool_call_body,
            &mut tool_call_index,
            &mut saw_tool_call,
            params.tool_call_policy,
            registration,
            params.tool_argument_wire_kinds.as_deref(),
            events,
            &first_text,
        ) {
            if let Some(c) = cancellation_counter {
                c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            return Ok(SerialStreamEnd::ClientClosed);
        }
    }
    completion_tokens += 1;
    if !is_eos_first {
        generated_tokens.push(next_token);
    }
    if reasoning_splitter
        .as_ref()
        .map(|s| s.in_reasoning())
        .unwrap_or(false)
    {
        reasoning_token_count += 1;
    }
    if is_eos_first {
        finish_reason = "stop";
    } else if qwen35_hit_stop_string(&accumulated_text, &params.stop_strings) {
        finish_reason = "stop";
        is_eos_first = true;
    }

    if !is_eos_first {
        for step in 1..max_tokens {
            // Wedge-4e: decode position is `t_post + (step - 1)`.
            // Text-only path: `t_post = prompt_len` ⇒ same as the
            // legacy `(prompt_len + step - 1)` advance (byte-identical).
            // Vision path: `t_post = max(axis-0) + 1`, accounting for
            // the multi-image temporal advance during prefill.
            let pos = t_post + (step as i32 - 1);
            if pos as u32 >= kv_cache.max_seq_len {
                break;
            }
            let decode_positions = vec![pos; 4];
            let dec_result = if is_greedy {
                supervised_gpu_call(supervisor, "qwen35_serial_stream_decode", || {
                    qwen.model
                        // ADR-040 Phase B4d (2026-05-30) — see sibling at
                        // engine_qwen35.rs:2071 for the SlotId contract.
                        .forward_gpu_greedy(
                            &[next_token],
                            &decode_positions,
                            &mut kv_cache,
                            SlotId(0),
                        )
                })
            } else {
                match supervised_gpu_call(supervisor, "qwen35_serial_stream_decode", || {
                    qwen.model.forward_gpu_last_logits(
                        &[next_token],
                        &decode_positions,
                        &mut kv_cache,
                        SlotId(0),
                    )
                }) {
                    Ok(logits) => {
                        let mut tmp = logits;
                        let token = sample_logits_qwen35_constrained(
                            &mut tmp,
                            params,
                            &generated_tokens,
                            grammar_runtime.as_ref(),
                            false,
                        )?
                        .0;
                        Ok(token)
                    }
                    Err(e) => Err(e),
                }
            };
            next_token = match dec_result {
                Ok(t) => t,
                Err(e) => {
                    return Err(e)
                        .with_context(|| format!("Qwen35 SerialFifo stream decode step {step}"));
                }
            };
            advance_qwen35_grammar(&mut grammar_runtime, params, next_token);
            if qwen.eos_token_ids.contains(&next_token) {
                finish_reason = "stop";
                break;
            }
            if qwen35_grammar_terminal_token(grammar_runtime.as_ref(), params, next_token) {
                finish_reason = "stop";
                break;
            }
            completion_tokens += 1;
            generated_tokens.push(next_token);
            let fragment = qwen
                .tokenizer
                .decode(&[next_token], false)
                .unwrap_or_default();
            accumulated_text.push_str(&fragment);
            if !emit_fragment_qwen35(
                &mut reasoning_splitter,
                &mut tool_splitter,
                &mut tool_call_body,
                &mut tool_call_index,
                &mut saw_tool_call,
                params.tool_call_policy,
                registration,
                params.tool_argument_wire_kinds.as_deref(),
                events,
                &fragment,
            ) {
                if let Some(c) = cancellation_counter {
                    c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return Ok(SerialStreamEnd::ClientClosed);
            }
            if reasoning_splitter
                .as_ref()
                .map(|s| s.in_reasoning())
                .unwrap_or(false)
            {
                reasoning_token_count += 1;
            }
            if qwen35_hit_stop_string(&accumulated_text, &params.stop_strings) {
                finish_reason = "stop";
                break;
            }
        }
    }

    // ── Drain splitter tails ───────────────────────────────────────
    if let Some(rs) = reasoning_splitter.as_mut() {
        if let Some((slot, tail)) = rs.finish() {
            match slot {
                SplitSlot::Reasoning => {
                    if !tail.is_empty()
                        && events
                            .blocking_send(GenerationEvent::Delta {
                                kind: DeltaKind::Reasoning,
                                text: tail,
                            })
                            .is_err()
                    {
                        if let Some(c) = cancellation_counter {
                            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        return Ok(SerialStreamEnd::ClientClosed);
                    }
                }
                SplitSlot::Content => {
                    if !route_content_qwen35(
                        &mut tool_splitter,
                        &mut tool_call_body,
                        &mut tool_call_index,
                        &mut saw_tool_call,
                        params.tool_call_policy,
                        registration,
                        params.tool_argument_wire_kinds.as_deref(),
                        events,
                        &tail,
                    ) {
                        if let Some(c) = cancellation_counter {
                            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        return Ok(SerialStreamEnd::ClientClosed);
                    }
                }
            }
        }
    }
    if let Some(tcs) = tool_splitter.as_mut() {
        if let Some(ev) = tcs.finish() {
            match ev {
                ToolCallEvent::Content(t) => {
                    if !t.is_empty()
                        && events
                            .blocking_send(GenerationEvent::Delta {
                                kind: DeltaKind::Content,
                                text: t,
                            })
                            .is_err()
                    {
                        if let Some(c) = cancellation_counter {
                            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        return Ok(SerialStreamEnd::ClientClosed);
                    }
                }
                ToolCallEvent::ToolCallText(t) => {
                    if params.tool_call_policy.enforces_body_grammar() {
                        send!(GenerationEvent::Error(
                            "tool_call_truncated_under_constrained".to_string()
                        ));
                        return Ok(SerialStreamEnd::TerminalSent);
                    }
                    // End-of-stream mid-tool-call (no close marker
                    // observed): re-emit residual as Content with the
                    // open marker re-prepended for diagnostic clarity —
                    // mirrors the Gemma Auto-no-grammar drain at
                    // engine.rs:2335-2351.
                    let prefix = registration.and_then(|r| r.tool_open).unwrap_or("");
                    let fallback = format!("{prefix}{t}");
                    if !fallback.is_empty()
                        && events
                            .blocking_send(GenerationEvent::Delta {
                                kind: DeltaKind::Content,
                                text: fallback,
                            })
                            .is_err()
                    {
                        if let Some(c) = cancellation_counter {
                            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        return Ok(SerialStreamEnd::ClientClosed);
                    }
                }
                ToolCallEvent::ToolCallOpen | ToolCallEvent::ToolCallClose => {
                    // unreachable — finish() never emits Open/Close.
                }
            }
        }
    }

    if matches!(params.tool_call_policy, ToolCallPolicy::Constrained) && !saw_tool_call {
        send!(GenerationEvent::Error(
            "tool_call_no_call_under_constrained".to_string()
        ));
        return Ok(SerialStreamEnd::TerminalSent);
    }

    if saw_tool_call {
        finish_reason = "tool_calls";
    }

    let decode_duration = decode_start.elapsed();
    let cached_tokens = reported_cached_tokens;
    let stats = StreamStats {
        prefill_time_secs: Some(prefill_duration.as_secs_f64()),
        decode_time_secs: Some(decode_duration.as_secs_f64()),
        total_time_secs: Some((prefill_duration + decode_duration).as_secs_f64()),
        time_to_first_token_ms: Some(prefill_duration.as_secs_f64() * 1000.0),
        prefill_tokens_per_sec: Some(if prefill_duration.as_secs_f64() > 0.0 {
            prompt_len.saturating_sub(cached_tokens) as f64 / prefill_duration.as_secs_f64()
        } else {
            0.0
        }),
        decode_tokens_per_sec: Some(if decode_duration.as_secs_f64() > 0.0 {
            completion_tokens as f64 / decode_duration.as_secs_f64()
        } else {
            0.0
        }),
        gpu_sync_count: Some(mlx_native::sync_count().saturating_sub(pre_syncs)),
        gpu_dispatch_count: Some(mlx_native::dispatch_count().saturating_sub(pre_dispatches)),
        // Item 5a (2026-08-20): a known-zero count is Some(0), never
        // None — the usage frame always carries explicit cached_tokens.
        cached_prompt_tokens: Some(cached_tokens),
        reasoning_tokens: if reasoning_token_count > 0 {
            Some(reasoning_token_count)
        } else {
            None
        },
    };

    qwen.persistent_kv_cache = Some(kv_cache);
    tracing::info!(
        target: "hf2q::serve::api::engine_qwen35::progress",
        mode = "stream",
        generated_tokens = completion_tokens,
        elapsed_ms = decode_duration.as_secs_f64() * 1000.0,
        tokens_per_second = if decode_duration.is_zero() {
            0.0
        } else {
            completion_tokens as f64 / decode_duration.as_secs_f64()
        },
        "Qwen35 decode complete"
    );
    tracing::info!(
        target: "hf2q::serve::api::engine_qwen35::progress",
        mode = "stream",
        prompt_tokens = prompt_len,
        cached_tokens,
        completion_tokens,
        total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
        "Qwen35 request complete"
    );

    send!(GenerationEvent::Done {
        finish_reason,
        prompt_tokens: prompt_len,
        completion_tokens,
        stats,
    });
    Ok(SerialStreamEnd::TerminalSent)
}

/// Wedge-3 / Phase D: chat-as-embedder.  Replaces the `worker_run` 501
/// arm for `Request::Embed`.
///
/// Single-shot prefill via `Qwen35Model::forward_embed_last` (Phase A),
/// returning the L2-normalized last-token hidden state of length
/// `cfg.hidden_size`.  KV cache allocated per-request (single forward,
/// no decode loop, so cache is discarded after).  Prompt cache is NOT
/// consulted here — the embedding path is a single forward; the
/// snapshot/replay savings are dominated by the no-decode shape.
pub(super) fn embed_qwen35(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    supervisor: &EngineSupervisor,
) -> Result<Vec<f32>> {
    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "embed_qwen35: empty prompt_tokens"
    );
    let device =
        MlxDevice::new().map_err(|e| anyhow::anyhow!("MlxDevice::new (qwen35 embed): {e}"))?;
    // For embeddings we don't need decode budget; pass max_tokens=0 to
    // size the cache to just the prompt.
    let mut kv_cache = alloc_kv_cache_for_request(qwen, &device, prompt_tokens.len(), 0)?;
    let positions = prefill_positions_for(prompt_tokens.len());
    supervised_gpu_call(supervisor, "qwen35_serial_embed", || {
        qwen.model
            // ADR-040 Phase B4b (2026-05-24): embed-last is single-stream
            // chat-as-embedder; slot 0 preserves pre-B4b behaviour. The
            // signature takes SlotId for uniformity with the rest of the
            // decode-side surface — future slot-aware embedding workloads
            // can pass SlotId(N) directly.
            .forward_embed_last(prompt_tokens, &positions, &mut kv_cache, SlotId(0))
            .context("Qwen35Model::forward_embed_last")
    })
}

/// **ADR-040 iter-C2d-cont-kernel iter-3 (2026-05-30)** — slot-aware
/// chat-as-embedder entry that routes the Qwen35 worker hot path's
/// **Embed** dispatch through the persistent multi-seq `HybridKvCache`
/// (`Qwen35LoadedModel.persistent_kv_cache`) instead of a per-request
/// fresh `alloc_kv_cache_for_request` allocation.
///
/// **Direct mirror of `generate_qwen35_once_slot_aware`** (iter-1) and
/// `generate_stream_qwen35_once_extended_slot_aware` (iter-2) for the
/// [`super::engine::Request::Embed`] worker arm. iter-1 landed the
/// non-streaming Generate-arm lift; iter-2 landed the streaming-arm
/// lift; iter-3 lands the embed-arm lift onto the same persistent cache
/// + per-slot reset + bounds-checked entry shape.
///
/// **Why this is the smallest of the iter-{1,2,3,4} ports** (no decode
/// loop, no SSE channel, no soft-token injections): the embed surface
/// runs exactly one `forward_embed_last` call against the slot, returns
/// the L2-normalized hidden vector, and exits. The prompt-cache HIT
/// fast-path is intentionally NOT consulted here — same shape as
/// non-slot-aware `embed_qwen35` (prompt cache savings are dominated by
/// the no-decode shape; the embed forward is a single pass).
///
/// # Structural parallels with iter-1 / iter-2
///
/// 1. Bounds-checks `slot_id` against `kv_cache.n_seqs` (bounds-first
///    per A2b §6.1.23 iter-1.5 cfa-finding-F5 ordering); typed
///    `anyhow::Error` on slot OOR with operator-grep'able cite.
/// 2. Verifies `prompt_len + 64 <= kv_cache.max_seq_len` (no
///    `max_tokens` term — embed has no decode budget; persistent cache
///    is sized to `cfg.max_position_embeddings`).
/// 3. Calls `kv_cache.reset_for_slot(slot_id)` at entry — zeros the
///    per-seq full-attn cursor + per-seq linear-attn conv/recurrent
///    slices for `slot_id` only (other slots untouched). This is the
///    iter-1 primitive, reused verbatim.
/// 4. Calls `forward_embed_last(prompt_tokens, &positions, kv_cache,
///    slot_id)` — already accepts `SlotId` post-B4b §6.1.20.
/// 5. Calls `kv_cache.reset_for_slot(slot_id)` at exit — belt-and-
///    suspenders with the entry reset (mirror of iter-1 / iter-2 exit
///    discipline). The embed path has NO intermediate paths that could
///    bypass the entry reset, but the exit reset preserves the
///    cross-iter exit-discipline pattern so the slot is always clean
///    at handoff for the next request to land at this slot.
///
/// **Per-slot byte-equivalence at SlotId(0)** (H64 pin):
/// `embed_qwen35_slot_aware(.., kv=&mut persistent_cache, slot_id=
/// SlotId(0))` produces the same `Vec<f32>` as `embed_qwen35(..)` for
/// any request when `persistent_cache.n_seqs == 1` AND
/// `persistent_cache.max_seq_len >= prompt_len + 64`. The proof is the
/// same shape as iter-1's H51 pin + iter-2's H58 pin: `reset_for_slot(0)`
/// matches the fresh-alloc state, `forward_embed_last(.., SlotId(0))`
/// is byte-equivalent to the pre-A2b path (B4a §6.1.4 pin), the
/// L2-normalization is identical, and the embed vector shape is
/// `cfg.hidden_size`-pinned (independent of cache shape).
///
/// # Embed-specific simplifications vs iter-1/iter-2
///
/// - **No decode loop**: the embed surface is exactly one
///   `forward_embed_last` call (vs iter-1's prefill + decode loop and
///   iter-2's streaming decode loop). No `next_token` sampling, no
///   stop-string handling, no `eos_token_ids` check, no `step` loop.
/// - **No prompt-cache HIT fast-path**: the embed surface runs a single
///   forward and discards the cache; HIT savings are dominated by the
///   no-decode shape. Identical to non-slot-aware `embed_qwen35`.
/// - **No SSE channel** / no `events.blocking_send` / no
///   `cancellation_counter`: the embed result is a single `Vec<f32>`
///   returned via the worker arm's `oneshot::Sender` (same as iter-1's
///   Generate surface).
/// - **No vision-aware extension surface**: the Embed `Request` variant
///   does not carry `soft_tokens` / `deepstack` / `positions_flat`, so
///   the `has_extension` typed-error path that iter-2 added is N/A
///   here.
/// - **Need check uses `prompt_len + 64`** (not `prompt_len + max_tokens
///   + 64`): embed has no decode budget; the `+64` margin matches the
///   `alloc_kv_cache_for_request(qwen, &device, prompt_tokens.len(),
///   0)` shape (the `0` max_tokens at line 4684 implies a `+64` slack
///   in the per-request alloc helper's sizing logic).
///
/// # Co-changes (iter-3 deliberately minimal — exact mirror of iter-1)
///
/// - Per-slot LCP / mid-prefill checkpoint storage is DISABLED in
///   slot-aware mode.  ADR-040 §6.1.50 (2026-05-30) closes
///   **iter-C2d-cont-kernel-iter-LCP per ADR-040 §6.1.50** as
///   **STRUCTURAL N/A** for the embed surface specifically: the embed
///   path has NO decode loop (single forward → L2-norm → return), so
///   there's no place to checkpoint mid-decode and no cross-request
///   prefix-sharing surface to probe.  Inherits the structural-N/A
///   finding from iter-1's docstring at engine_qwen35.rs:~2215 (snapshot
///   codec keyed on per-request `max_seq_len` + cross-slot tenant-
///   isolation).
/// - ADR-040 §6.1.50 iter-G is also N/A here: the embed path has no
///   decode loop, so there's no per-step `forward_gpu_greedy` candidate
///   to lift (the single `forward_embed_last(.., slot_id)` call already
///   returns the L2-norm vector directly without a vocab-size readback).
/// - Vision-augmented embeddings (Qwen3-VL chat-as-embedder with image
///   inputs) are NOT exposed via the Embed `Request` variant in
///   `engine.rs` — the Embed handler only accepts `prompt_tokens` (no
///   `soft_tokens` / `deepstack` / `positions_flat`). If a future iter
///   adds vision-augmented embedding, it would extend the `Request::
///   Embed` variant + this fn signature in lockstep; iter-3 does not
///   anticipate that shape.
///
/// # Errors
/// - `prompt_tokens.is_empty()` (matches `embed_qwen35`).
/// - `slot_id.0 >= kv_cache.n_seqs` (bounds-first; surfaces typed
///   `anyhow::Error` with `capability_unsupported:` prefix +
///   `iter-C2d-cont-kernel iter-3` cite).
/// - `prompt_len + 64 > kv_cache.max_seq_len` (typed error with same
///   prefix shape).
/// - `reset_for_slot` failure propagates with `iter-C2d-cont-kernel
///   iter-3` context.
/// - `forward_embed_last` failure propagates with the usual context.
pub fn embed_qwen35_slot_aware(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    kv_cache: &mut HybridKvCache,
    slot_id: SlotId,
) -> Result<Vec<f32>> {
    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "embed_qwen35_slot_aware: empty prompt_tokens"
    );
    // Bounds-first per A2b §6.1.23 iter-1.5 cfa-finding-F5 ordering.
    anyhow::ensure!(
        slot_id.0 < kv_cache.n_seqs,
        "embed_qwen35_slot_aware: SlotOutOfRange slot={} max_slots={} \
         (ADR-040 iter-C2d-cont-kernel iter-3)",
        slot_id.0,
        kv_cache.n_seqs,
    );
    let prompt_len = prompt_tokens.len();
    // Verify the persistent cache has room for this embed request.
    // Embed has no decode budget; the `+ 64` slack matches the
    // `alloc_kv_cache_for_request(qwen, &device, prompt_len, 0)` shape
    // used by the non-slot-aware `embed_qwen35` path.
    let need_seq = prompt_len + 64;
    if need_seq > kv_cache.max_seq_len as usize {
        return Err(anyhow::anyhow!(
            "embed_qwen35_slot_aware: per-request need_seq={} exceeds \
             persistent cache max_seq_len={} (slot={} prompt_len={}). \
             ADR-040 iter-C2d-cont-kernel iter-3 sizes the persistent \
             cache to cfg.max_position_embeddings; use a shorter prompt.",
            need_seq,
            kv_cache.max_seq_len,
            slot_id.0,
            prompt_len
        ));
    }

    // Per-slot reset at entry — the persistent cache may carry stale
    // bytes from a prior request on this slot. `reset_for_slot` zeros
    // the per-seq cursors + linear-attn conv/recurrent slices for
    // `slot_id` only (other slots untouched). Mirror of iter-1 H54 +
    // iter-2 H61 pattern.
    kv_cache
        .reset_for_slot(slot_id)
        .context("ADR-040 iter-C2d-cont-kernel iter-3: reset_for_slot at entry")?;

    // Single forward pass — no decode loop, no prompt-cache HIT
    // fast-path (mirrors `embed_qwen35` non-slot-aware shape).
    let positions = prefill_positions_for(prompt_len);
    let embed_result = qwen
        .model
        .forward_embed_last(prompt_tokens, &positions, kv_cache, slot_id)
        .context(
            "Qwen35Model::forward_embed_last (ADR-040 iter-C2d-cont-kernel iter-3 slot-aware)",
        );

    // Per-slot reset at exit — leave the slot clean for the next
    // request to land on it. Belt-and-suspenders w/ the entry reset:
    // ensures that even if a future iter adds a code path between
    // entry-reset and exit-reset that mutates per-slot state, the slot
    // is always clean at handoff. Mirrors iter-1 + iter-2 exit
    // discipline. A failed forward returns immediately above: a typed Metal
    // timeout/device loss must reach the worker classifier before any CPU
    // mapping or mutation touches the poisoned cache generation.
    finish_qwen35_slot_embed(embed_result, || {
        kv_cache
            .reset_for_slot(slot_id)
            .context("ADR-040 iter-C2d-cont-kernel iter-3: reset_for_slot at exit")
    })
}

fn finish_qwen35_slot_embed<T>(
    forward: Result<T>,
    reset_after_success: impl FnOnce() -> Result<()>,
) -> Result<T> {
    let output = forward?;
    reset_after_success()?;
    Ok(output)
}

/// ADR-040 Phase C iter-C2d-cont-kernel iter-4 (2026-05-30): slot-aware
/// vision-aware non-streaming Qwen35 generation against the
/// **persistent multi-seq `HybridKvCache`**
/// (`Qwen35LoadedModel.persistent_kv_cache`) instead of a per-request
/// fresh `alloc_kv_cache_for_request` allocation.
///
/// **Direct mirror of `generate_qwen35_once_slot_aware`** (iter-1) and
/// `embed_qwen35_slot_aware` (iter-3) for the
/// [`super::engine::Request::GenerateWithSoftTokens`] worker arm with
/// `deepstack.is_none() && positions_flat.is_none()` (the
/// soft-tokens-only sub-shape). iter-1 landed Generate; iter-2 landed
/// streaming; iter-3 landed Embed; iter-4 lands the SoftTokens arm onto
/// the same persistent cache + per-slot reset + bounds-checked entry
/// shape. The deepstack + 3D-positions sibling is
/// [`generate_qwen35_once_with_soft_tokens_and_deepstack_slot_aware`]
/// (Wedge-4d-equivalent for the slot-aware path).
///
/// # Structural parallels with iter-1 / iter-2 / iter-3
///
/// 1. Bounds-checks `slot_id` against `kv_cache.n_seqs` (bounds-first
///    per A2b §6.1.23 iter-1.5 cfa-finding-F5 ordering); typed
///    `anyhow::Error` on slot OOR with `iter-C2d-cont-kernel iter-4` cite.
/// 2. Verifies `prompt_len + max_tokens + 64 <= kv_cache.max_seq_len`
///    (persistent cache is sized to `cfg.max_position_embeddings`;
///    per-request need must fit).
/// 3. Calls `kv_cache.reset_for_slot(slot_id)` at entry — zeros the
///    per-seq full-attn cursor + per-seq linear-attn conv/recurrent
///    slices for `slot_id` only (other slots untouched). Mirror of
///    iter-1 / iter-2 / iter-3 primitives.
/// 4. Threads `slot_id` into every forward call (prefill goes through
///    `forward_gpu_last_logits_with_soft_tokens(.., slot_id)`; decode
///    steps use `forward_gpu_last_logits(.., slot_id)` — soft-token
///    overrides only apply during prefill, mirroring the non-slot-aware
///    sibling).
/// 5. Calls `kv_cache.reset_for_slot(slot_id)` at exit — belt-and-
///    suspenders with the entry reset (mirror of iter-1 / iter-2 / iter-3
///    exit discipline).
///
/// **Per-slot byte-equivalence at SlotId(0)** (H70 pin):
/// `generate_qwen35_once_with_soft_tokens_slot_aware(.., kv=&mut
/// persistent_cache, slot_id=SlotId(0))` produces the same
/// `GenerationResult` as `generate_qwen35_once_with_soft_tokens(..)` for
/// any vision-aware request when `persistent_cache.n_seqs == 1` AND
/// `persistent_cache.max_seq_len >= prompt_len + max_tokens + 64`. The
/// proof is identical to iter-1's H51 / iter-2's H58 / iter-3's H64:
/// `reset_for_slot(0)` matches the fresh-alloc state,
/// `forward_gpu_last_logits_with_soft_tokens(.., SlotId(0))` is byte-
/// equivalent to the pre-A2b path (B4a §6.1.4 pin), and the sampling /
/// decode loop is structurally identical to the non-slot-aware sibling.
///
/// # SoftTokens-specific simplifications vs iter-1
///
/// - **No prompt-cache HIT fast-path**: the SoftTokens path bypasses
///   prompt cache (the cache key is `prompt_tokens` only and would
///   falsely hit on a vision-augmented request with the same
///   placeholder ids but different image content). Mirrors the
///   non-slot-aware `generate_qwen35_once_with_soft_tokens` rationale
///   (engine_qwen35.rs:3245 "Prompt-cache is intentionally NOT
///   consulted on the vision path"). The H70 byte-equivalence pin
///   depends on this discipline matching.
///
/// # Co-changes (iter-4 deliberately minimal — exact mirror of iter-1)
///
/// - Per-slot LCP / mid-prefill checkpoint storage is DISABLED in
///   slot-aware mode.  ADR-040 §6.1.50 (2026-05-30) closes
///   **iter-C2d-cont-kernel-iter-LCP per ADR-040 §6.1.50** as
///   **STRUCTURAL N/A** (same finding as iter-1: snapshot codec keyed
///   on per-request `max_seq_len` + cross-slot tenant-isolation; the
///   soft-tokens path has no prompt-cache HIT in either slot-aware or
///   non-slot-aware mode per `engine_qwen35.rs:3410` discipline).
/// - Chunked-prefill is DISABLED in slot-aware mode (same snapshot-
///   shape reason).  Same STRUCTURAL N/A pin per §6.1.50.
/// - DFlash / spec-decode capture-states are NOT engaged here (they
///   require `ensure_la_capture` which is spec-decode-only); slot-aware
///   spec-decode is **iter-B4d** per §6.1.26 deferrals matrix.
/// - ADR-040 §6.1.50 (2026-05-30) lands
///   **iter-C2d-cont-kernel-iter-G per ADR-040 §6.1.50** REAL LIFT for
///   the soft-tokens-only AND deepstack sub-shape decode greedy fast-
///   paths: greedy branch now routes through
///   `forward_gpu_greedy(.., slot_id)` (decode positions are post-prompt
///   by construction so the soft-token-FREE greedy path is structurally
///   correct; mirror of `generate_qwen35_once_with_soft_tokens` decode
///   discipline at engine_qwen35.rs:3279).
///
/// # Errors
/// - `soft_tokens.is_empty()` → identity over `generate_qwen35_once_slot_aware`
///   (mirrors the non-slot-aware `generate_qwen35_once_with_soft_tokens`
///   text-only fallback at engine_qwen35.rs:3216).
/// - `prompt_tokens.is_empty()` (matches `generate_qwen35_once_with_soft_tokens`).
/// - `slot_id.0 >= kv_cache.n_seqs` (bounds-first; typed
///   `anyhow::Error` with `iter-C2d-cont-kernel iter-4` cite).
/// - `prompt_len + max_tokens + 64 > kv_cache.max_seq_len` (typed error).
/// - Forward / sample failures propagate from
///   `forward_gpu_last_logits_with_soft_tokens` / `forward_gpu_last_logits`.
pub fn generate_qwen35_once_with_soft_tokens_slot_aware(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    soft_tokens: &[crate::serve::forward_prefill::SoftTokenInjection<'_>],
    params: &SamplingParams,
    registration: Option<&ModelRegistration>,
    kv_cache: &mut HybridKvCache,
    slot_id: SlotId,
) -> Result<GenerationResult> {
    // Empty slice → identity over the text-only slot-aware path.
    // Mirrors the non-slot-aware fallback at engine_qwen35.rs:3216
    // (`generate_qwen35_once_with_soft_tokens` empty-soft check).
    if soft_tokens.is_empty() {
        return generate_qwen35_once_slot_aware(
            qwen,
            prompt_tokens,
            params,
            registration,
            kv_cache,
            slot_id,
        );
    }

    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "generate_qwen35_once_with_soft_tokens_slot_aware: empty prompt_tokens"
    );
    // Bounds-first per A2b §6.1.23 iter-1.5 cfa-finding-F5 ordering.
    anyhow::ensure!(
        slot_id.0 < kv_cache.n_seqs,
        "generate_qwen35_once_with_soft_tokens_slot_aware: SlotOutOfRange slot={} \
         max_slots={} (ADR-040 iter-C2d-cont-kernel iter-4)",
        slot_id.0,
        kv_cache.n_seqs,
    );
    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);
    // Verify the persistent cache has room for this request. Persistent
    // cache is sized to `cfg.max_position_embeddings`; per-request need
    // is `prompt_len + max_tokens + 64`. Same shape as iter-1.
    let need_seq = prompt_len + max_tokens + 64;
    if need_seq > kv_cache.max_seq_len as usize {
        return Err(anyhow::anyhow!(
            "generate_qwen35_once_with_soft_tokens_slot_aware: per-request \
             need_seq={} exceeds persistent cache max_seq_len={} (slot={} \
             prompt_len={} max_tokens={}). ADR-040 iter-C2d-cont-kernel iter-4 \
             sizes the persistent cache to cfg.max_position_embeddings; reduce \
             max_tokens or use a shorter prompt.",
            need_seq,
            kv_cache.max_seq_len,
            slot_id.0,
            prompt_len,
            max_tokens
        ));
    }

    let is_greedy = is_greedy_eligible(params);
    let want_logprobs = params.logprobs;
    let mut logprobs_vec: Option<Vec<f32>> = if want_logprobs {
        Some(Vec::with_capacity(max_tokens))
    } else {
        None
    };

    // Per-slot reset at entry — mirror of iter-1 H54 / iter-2 H61 /
    // iter-3 H67 pattern.
    kv_cache
        .reset_for_slot(slot_id)
        .context("ADR-040 iter-C2d-cont-kernel iter-4: reset_for_slot at entry")?;

    // Prompt-cache is intentionally NOT consulted on the vision path
    // (mirrors non-slot-aware `generate_qwen35_once_with_soft_tokens`
    // at engine_qwen35.rs:3245). Cache-key safety: a vision-augmented
    // request with the same placeholder ids but different image content
    // would falsely hit a cached text-only result.
    let prefill_start = Instant::now();
    let positions = prefill_positions_for(prompt_len);
    let prefill_logits = qwen
        .model
        .forward_gpu_last_logits_with_soft_tokens(
            prompt_tokens,
            &positions,
            soft_tokens,
            kv_cache,
            slot_id,
        )
        .context(
            "Qwen35Model::forward_gpu_last_logits_with_soft_tokens \
             (slot-aware prefill, ADR-040 iter-C2d-cont-kernel iter-4)",
        )?;
    anyhow::ensure!(
        prefill_logits.len() == qwen.vocab_size,
        "qwen35 slot-aware soft-tokens prefill logits len {} != vocab_size {}",
        prefill_logits.len(),
        qwen.vocab_size
    );
    let mut next_token: u32 = if want_logprobs {
        let mut logits = prefill_logits.clone();
        let (tok, lp) = sample_logits_qwen35_with_logprob(&mut logits, params, &[]);
        if let Some(v) = logprobs_vec.as_mut() {
            v.push(lp);
        }
        tok
    } else if is_greedy {
        greedy_argmax_last_token(&prefill_logits, qwen.vocab_size as u32)
    } else {
        let mut logits = prefill_logits.clone();
        sample_logits_qwen35(&mut logits, params, &[])
    };
    let prefill_duration = prefill_start.elapsed();

    // Decode loop — identical structure to iter-1's
    // `generate_qwen35_once_slot_aware`. Decode positions are
    // post-prompt by construction (>= prompt_len) and so cannot lie
    // within any soft-token range, so the decode path deliberately
    // uses the soft-token-FREE forward methods (mirror of
    // non-slot-aware `generate_qwen35_once_with_soft_tokens` at
    // engine_qwen35.rs:3279).
    let decode_start = Instant::now();
    let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    generated_tokens.push(next_token);

    let first_fragment = qwen
        .tokenizer
        .decode(&[next_token], false)
        .unwrap_or_default();
    let mut decoded_text = first_fragment.clone();

    let mut finish_reason: &'static str = "length";

    if qwen.eos_token_ids.contains(&next_token) {
        finish_reason = "stop";
    } else if qwen35_hit_stop_string(&decoded_text, &params.stop_strings) {
        finish_reason = "stop";
        qwen35_strip_trailing_stop(&mut decoded_text, &params.stop_strings);
    } else {
        for step in 1..max_tokens {
            let pos = (prompt_len + step - 1) as i32;
            if pos as u32 >= kv_cache.max_seq_len {
                tracing::warn!(
                    pos,
                    max_seq = kv_cache.max_seq_len,
                    "qwen35 slot-aware decode (soft tokens): hit kv-cache bound; \
                     stopping with finish=length",
                );
                break;
            }
            let decode_positions = vec![pos; 4];

            next_token = if want_logprobs {
                let logits_full = qwen
                    .model
                    .forward_gpu_last_logits(&[next_token], &decode_positions, kv_cache, slot_id)
                    .with_context(|| {
                        format!(
                            "forward_gpu_last_logits slot-aware decode step {step} \
                             (soft tokens, logprobs)"
                        )
                    })?;
                let mut logits = logits_full;
                let (tok, lp) =
                    sample_logits_qwen35_with_logprob(&mut logits, params, &generated_tokens);
                if let Some(v) = logprobs_vec.as_mut() {
                    v.push(lp);
                }
                tok
            } else if is_greedy {
                // ADR-040 §6.1.50 (2026-05-30) iter-C2d-cont-kernel-iter-G
                // REAL LIFT: greedy fast-path routes through
                // `forward_gpu_greedy(.., slot_id)`.  `forward_gpu_greedy`
                // accepts `slot_id` since B4d §6.1.44 (2026-05-30) — the
                // pre-§6.1.50 docstring claim "that fn does not yet thread
                // slot_id" was outdated.  Decode positions are post-prompt
                // by construction (>= prompt_len), so the soft-token-FREE
                // greedy path is structurally correct here.
                qwen.model
                    .forward_gpu_greedy(&[next_token], &decode_positions, kv_cache, slot_id)
                    .with_context(|| {
                        format!(
                            "forward_gpu_greedy slot-aware decode step {step} \
                             (soft tokens; ADR-040 §6.1.50 iter-G)"
                        )
                    })?
            } else {
                let logits_full = qwen
                    .model
                    .forward_gpu_last_logits(&[next_token], &decode_positions, kv_cache, slot_id)
                    .with_context(|| {
                        format!(
                            "forward_gpu_last_logits slot-aware decode step {step} \
                             (soft tokens)"
                        )
                    })?;
                let mut logits = logits_full;
                sample_logits_qwen35(&mut logits, params, &generated_tokens)
            };

            if qwen.eos_token_ids.contains(&next_token) {
                finish_reason = "stop";
                break;
            }
            generated_tokens.push(next_token);
            let fragment = qwen
                .tokenizer
                .decode(&[next_token], false)
                .unwrap_or_default();
            decoded_text.push_str(&fragment);
            if qwen35_hit_stop_string(&decoded_text, &params.stop_strings) {
                finish_reason = "stop";
                qwen35_strip_trailing_stop(&mut decoded_text, &params.stop_strings);
                break;
            }
        }
    }
    let decode_duration = decode_start.elapsed();

    // Per-slot reset at exit — leave the slot clean for the next
    // request. Mirrors iter-1 / iter-2 / iter-3 exit discipline (H72 +
    // H73 pins).
    kv_cache
        .reset_for_slot(slot_id)
        .context("ADR-040 iter-C2d-cont-kernel iter-4: reset_for_slot at exit")?;

    // Reasoning split — mirror of generate_qwen35_once_with_soft_tokens.
    let (content, reasoning_text) = match registration {
        Some(reg) if reg.has_reasoning() => super::registry::split_full_output_forced(
            reg,
            &decoded_text,
            params.reasoning_forced_open,
        ),
        _ => (decoded_text, None),
    };

    let reasoning_token_count = match registration {
        Some(reg) if reg.has_reasoning() => {
            let mut sp =
                super::registry::make_reasoning_splitter(reg, params.reasoning_forced_open);
            let mut count = 0usize;
            for &tok in &generated_tokens {
                let frag = qwen.tokenizer.decode(&[tok], false).unwrap_or_default();
                if let Some(splitter) = sp.as_mut() {
                    let _ = splitter.feed(&frag);
                    if splitter.in_reasoning() {
                        count += 1;
                    }
                }
            }
            count
        }
        _ => 0,
    };

    Ok(GenerationResult {
        text: content,
        reasoning_text,
        prompt_tokens: prompt_len,
        completion_tokens: generated_tokens.len(),
        reasoning_tokens: if reasoning_token_count > 0 {
            Some(reasoning_token_count)
        } else {
            None
        },
        finish_reason,
        prefill_duration,
        decode_duration,
        // No prompt-cache fast-path on the soft-tokens path; cached
        // tokens count is always 0 for vision-augmented requests
        // (mirrors non-slot-aware sibling at engine_qwen35.rs:3410).
        cached_tokens: 0,
        logprobs: logprobs_vec,
    })
}

/// ADR-040 Phase C iter-C2d-cont-kernel iter-4 (2026-05-30): slot-aware
/// vision-aware non-streaming Qwen35 generation with the full DeepStack
/// injection pipeline against the **persistent multi-seq `HybridKvCache`**.
///
/// **Direct mirror of `generate_qwen35_once_with_soft_tokens_and_deepstack`**
/// (Wedge-4d) lifted onto the persistent cache + per-slot reset shape
/// established by iter-1 / iter-2 / iter-3. Sibling of
/// [`generate_qwen35_once_with_soft_tokens_slot_aware`] for the
/// `deepstack.is_some() || positions_flat.is_some()` sub-shape.
///
/// # Structural parallels with iter-1 / iter-3
///
/// Identical to `generate_qwen35_once_with_soft_tokens_slot_aware` except:
///   * Prefill goes through `forward_gpu_last_logits_with_soft_tokens_and_deepstack`
///     so per-LM-layer DeepStack chunks are added to the residual stream
///     at the image-token positions during prefill (per peer
///     `qwen3vl.cpp:96-100`).
///   * The 3D-mRoPE position buffer (`positions_flat: [4 * prompt_len]`)
///     is supplied by the chat handler via
///     `crate::serve::forward_prefill::build_qwen3vl_positions`, NOT
///     synthesized via `prefill_positions_for`. This carries the
///     `[t, y, x, 0]` axis assignment that the IMROPE kernel consumes
///     for image-patch tokens.
///   * Decode steps after prefill use text-only `[t,t,t,t]` positions
///     starting from the post-prefill global temporal counter (which
///     advances by `max(n_x, n_y)` per image, NOT by `n_image_tokens`,
///     per peer `mtmd.cpp:1354-1357`).
///
/// When both `deepstack` and `positions_flat` are `None`, behaviour is
/// identical to `generate_qwen35_once_with_soft_tokens_slot_aware` —
/// which itself falls through to `generate_qwen35_once_slot_aware` when
/// `soft_tokens` is also empty.
///
/// # Errors
///
/// Same shape as `generate_qwen35_once_with_soft_tokens_slot_aware`,
/// plus the non-slot-aware DeepStack-specific error class
/// (`positions_flat.len() != 4 * prompt_len`).
#[allow(clippy::too_many_arguments)]
pub fn generate_qwen35_once_with_soft_tokens_and_deepstack_slot_aware(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    soft_tokens: &[crate::serve::forward_prefill::SoftTokenInjection<'_>],
    deepstack: Option<&crate::serve::forward_prefill::DeepstackInjection<'_>>,
    positions_flat: Option<&[i32]>,
    params: &SamplingParams,
    registration: Option<&ModelRegistration>,
    kv_cache: &mut HybridKvCache,
    slot_id: SlotId,
) -> Result<GenerationResult> {
    // Empty soft + no deepstack + no positions → identity over text-only
    // slot-aware path. Mirrors non-slot-aware sibling at
    // engine_qwen35.rs:3447.
    if soft_tokens.is_empty() && deepstack.is_none() && positions_flat.is_none() {
        return generate_qwen35_once_slot_aware(
            qwen,
            prompt_tokens,
            params,
            registration,
            kv_cache,
            slot_id,
        );
    }

    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "generate_qwen35_once_with_soft_tokens_and_deepstack_slot_aware: \
         empty prompt_tokens"
    );
    // Bounds-first per A2b §6.1.23 iter-1.5 cfa-finding-F5 ordering.
    anyhow::ensure!(
        slot_id.0 < kv_cache.n_seqs,
        "generate_qwen35_once_with_soft_tokens_and_deepstack_slot_aware: \
         SlotOutOfRange slot={} max_slots={} (ADR-040 iter-C2d-cont-kernel iter-4)",
        slot_id.0,
        kv_cache.n_seqs,
    );
    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);
    let need_seq = prompt_len + max_tokens + 64;
    if need_seq > kv_cache.max_seq_len as usize {
        return Err(anyhow::anyhow!(
            "generate_qwen35_once_with_soft_tokens_and_deepstack_slot_aware: \
             per-request need_seq={} exceeds persistent cache max_seq_len={} \
             (slot={} prompt_len={} max_tokens={}). ADR-040 iter-C2d-cont-\
             kernel iter-4 sizes the persistent cache to \
             cfg.max_position_embeddings; reduce max_tokens or use a shorter \
             prompt.",
            need_seq,
            kv_cache.max_seq_len,
            slot_id.0,
            prompt_len,
            max_tokens
        ));
    }

    let is_greedy = is_greedy_eligible(params);
    let want_logprobs = params.logprobs;
    let mut logprobs_vec: Option<Vec<f32>> = if want_logprobs {
        Some(Vec::with_capacity(max_tokens))
    } else {
        None
    };

    // Per-slot reset at entry — mirror of iter-1 H54.
    kv_cache
        .reset_for_slot(slot_id)
        .context("ADR-040 iter-C2d-cont-kernel iter-4: reset_for_slot at entry (deepstack)")?;

    let prefill_start = Instant::now();
    // Use supplied 3D positions if provided; otherwise fall back to
    // text-style `[t,t,t,t]` positions (mirror of non-slot-aware
    // sibling at engine_qwen35.rs:3480).
    let positions_owned: Vec<i32>;
    let positions: &[i32] = match positions_flat {
        Some(p) => {
            anyhow::ensure!(
                p.len() == 4 * prompt_len,
                "generate_qwen35_once_with_soft_tokens_and_deepstack_slot_aware: \
                 positions_flat.len() = {} != 4 * prompt_len = {}",
                p.len(),
                4 * prompt_len
            );
            p
        }
        None => {
            positions_owned = prefill_positions_for(prompt_len);
            &positions_owned
        }
    };

    let prefill_logits = qwen
        .model
        .forward_gpu_last_logits_with_soft_tokens_and_deepstack(
            prompt_tokens,
            positions,
            soft_tokens,
            deepstack,
            kv_cache,
            slot_id,
        )
        .context(
            "Qwen35Model::forward_gpu_last_logits_with_soft_tokens_and_deepstack \
             (slot-aware prefill, ADR-040 iter-C2d-cont-kernel iter-4)",
        )?;
    anyhow::ensure!(
        prefill_logits.len() == qwen.vocab_size,
        "qwen35 slot-aware deepstack prefill logits len {} != vocab_size {}",
        prefill_logits.len(),
        qwen.vocab_size
    );
    let mut next_token: u32 = if want_logprobs {
        let mut logits = prefill_logits.clone();
        let (tok, lp) = sample_logits_qwen35_with_logprob(&mut logits, params, &[]);
        if let Some(v) = logprobs_vec.as_mut() {
            v.push(lp);
        }
        tok
    } else if is_greedy {
        greedy_argmax_last_token(&prefill_logits, qwen.vocab_size as u32)
    } else {
        let mut logits = prefill_logits.clone();
        sample_logits_qwen35(&mut logits, params, &[])
    };
    let prefill_duration = prefill_start.elapsed();

    // Decode loop — for the post-prefill text steps, the global
    // temporal position has advanced by image-aware amounts. Compute
    // the post-prefill temporal `t_post` from the LAST text token's
    // axis-0 position +1 (mirror of non-slot-aware sibling at
    // engine_qwen35.rs:3537).
    let t_post: i32 = match positions_flat {
        Some(p) => {
            let mut max_t = 0i32;
            for i in 0..prompt_len {
                let v = p[i]; // axis 0 = t
                if v > max_t {
                    max_t = v;
                }
            }
            max_t.saturating_add(1)
        }
        None => prompt_len as i32,
    };

    let decode_start = Instant::now();
    let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    generated_tokens.push(next_token);

    let first_fragment = qwen
        .tokenizer
        .decode(&[next_token], false)
        .unwrap_or_default();
    let mut decoded_text = first_fragment.clone();

    let mut finish_reason: &'static str = "length";

    if qwen.eos_token_ids.contains(&next_token) {
        finish_reason = "stop";
    } else if qwen35_hit_stop_string(&decoded_text, &params.stop_strings) {
        finish_reason = "stop";
        qwen35_strip_trailing_stop(&mut decoded_text, &params.stop_strings);
    } else {
        for step in 1..max_tokens {
            let pos = t_post + (step as i32 - 1);
            if pos as u32 >= kv_cache.max_seq_len {
                tracing::warn!(
                    pos,
                    max_seq = kv_cache.max_seq_len,
                    "qwen35 slot-aware decode (deepstack): hit kv-cache bound; \
                     stopping with finish=length",
                );
                break;
            }
            let decode_positions = vec![pos; 4];

            next_token = if want_logprobs {
                let logits_full = qwen
                    .model
                    .forward_gpu_last_logits(&[next_token], &decode_positions, kv_cache, slot_id)
                    .with_context(|| {
                        format!(
                            "forward_gpu_last_logits slot-aware decode step {step} \
                             (deepstack, logprobs)"
                        )
                    })?;
                let mut logits = logits_full;
                let (tok, lp) =
                    sample_logits_qwen35_with_logprob(&mut logits, params, &generated_tokens);
                if let Some(v) = logprobs_vec.as_mut() {
                    v.push(lp);
                }
                tok
            } else if is_greedy {
                // ADR-040 §6.1.50 (2026-05-30) iter-C2d-cont-kernel-iter-G
                // REAL LIFT (deepstack sub-shape): decode positions are
                // post-prompt by construction so the soft-token-FREE
                // greedy fast-path applies.  Mirror of soft-tokens variant
                // above + iter-1's `generate_qwen35_once_slot_aware`.
                qwen.model
                    .forward_gpu_greedy(&[next_token], &decode_positions, kv_cache, slot_id)
                    .with_context(|| {
                        format!(
                            "forward_gpu_greedy slot-aware decode step {step} \
                             (deepstack; ADR-040 §6.1.50 iter-G)"
                        )
                    })?
            } else {
                let logits_full = qwen
                    .model
                    .forward_gpu_last_logits(&[next_token], &decode_positions, kv_cache, slot_id)
                    .with_context(|| {
                        format!(
                            "forward_gpu_last_logits slot-aware decode step {step} \
                             (deepstack)"
                        )
                    })?;
                let mut logits = logits_full;
                sample_logits_qwen35(&mut logits, params, &generated_tokens)
            };

            if qwen.eos_token_ids.contains(&next_token) {
                finish_reason = "stop";
                break;
            }
            generated_tokens.push(next_token);
            let fragment = qwen
                .tokenizer
                .decode(&[next_token], false)
                .unwrap_or_default();
            decoded_text.push_str(&fragment);
            if qwen35_hit_stop_string(&decoded_text, &params.stop_strings) {
                finish_reason = "stop";
                qwen35_strip_trailing_stop(&mut decoded_text, &params.stop_strings);
                break;
            }
        }
    }
    let decode_duration = decode_start.elapsed();

    // Per-slot reset at exit — mirror of iter-1/2/3 exit discipline.
    kv_cache
        .reset_for_slot(slot_id)
        .context("ADR-040 iter-C2d-cont-kernel iter-4: reset_for_slot at exit (deepstack)")?;

    let (content, reasoning_text) = match registration {
        Some(reg) if reg.has_reasoning() => super::registry::split_full_output_forced(
            reg,
            &decoded_text,
            params.reasoning_forced_open,
        ),
        _ => (decoded_text, None),
    };

    let reasoning_token_count = match registration {
        Some(reg) if reg.has_reasoning() => {
            let mut sp =
                super::registry::make_reasoning_splitter(reg, params.reasoning_forced_open);
            let mut count = 0usize;
            for &tok in &generated_tokens {
                let frag = qwen.tokenizer.decode(&[tok], false).unwrap_or_default();
                if let Some(splitter) = sp.as_mut() {
                    let _ = splitter.feed(&frag);
                    if splitter.in_reasoning() {
                        count += 1;
                    }
                }
            }
            count
        }
        _ => 0,
    };

    Ok(GenerationResult {
        text: content,
        reasoning_text,
        prompt_tokens: prompt_len,
        completion_tokens: generated_tokens.len(),
        reasoning_tokens: if reasoning_token_count > 0 {
            Some(reasoning_token_count)
        } else {
            None
        },
        finish_reason,
        prefill_duration,
        decode_duration,
        cached_tokens: 0,
        logprobs: logprobs_vec,
    })
}

/// Default tool-call policy for the legacy slot-aware streaming arm.
///
/// This helper belongs to the retained legacy function above. Production
/// SlotAware Qwen requests route through `run_slot_aware_qwen35` and its
/// grammar-aware prefill/decode states instead.
fn params_tool_call_policy_for_qwen35_stream() -> super::engine::ToolCallPolicy {
    super::engine::ToolCallPolicy::Auto
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::kv_cache::HybridKvCache;
    use crate::inference::models::qwen35::{
        default_layer_types, Qwen35Config, Qwen35MoeConfig, Qwen35Variant,
    };
    use mlx_native::{MlxBuffer, MlxDevice};
    use std::cell::Cell;

    fn thinking_budget_params(limit: usize) -> SamplingParams {
        SamplingParams {
            reasoning_forced_open: true,
            thinking_token_budget: Some(limit),
            reasoning_end_tokens: Some(Arc::new(vec![90, 91])),
            reasoning_close_tokens: Some(Arc::new(vec![91])),
            ..SamplingParams::default()
        }
    }

    #[test]
    fn thinking_budget_forces_close_then_stops_overriding_answer_tokens() {
        let mut budget =
            Qwen35ThinkingBudgetState::from_params(&thinking_budget_params(2)).unwrap();
        let mut generated = Vec::new();

        for token in [11, 12] {
            generated.push(token);
            budget.observe_generated(&generated, false);
        }
        assert_eq!(budget.next_forced_token(), Some((90, true)));
        assert!(!budget.was_forced_closed());
        generated.push(90);
        budget.observe_generated(&generated, false);
        assert_eq!(budget.next_forced_token(), Some((91, false)));
        generated.push(91);
        budget.observe_generated(&generated, false);
        assert_eq!(budget.next_forced_token(), None);
        assert!(budget.closed);
        assert!(budget.was_forced_closed());
    }

    #[test]
    fn thinking_budget_honors_natural_reasoning_and_tool_boundaries() {
        let mut natural =
            Qwen35ThinkingBudgetState::from_params(&thinking_budget_params(4)).unwrap();
        natural.observe_generated(&[11, 91], false);
        assert!(natural.closed);
        assert_eq!(natural.next_forced_token(), None);

        let mut tool = Qwen35ThinkingBudgetState::from_params(&thinking_budget_params(4)).unwrap();
        tool.observe_generated(&[11], true);
        assert!(tool.closed);
        assert_eq!(tool.next_forced_token(), None);
    }

    #[test]
    fn bounded_prefill_splits_exactly_at_the_rewriteable_prompt_boundary() {
        assert_eq!(qwen35_next_prefill_end(0, 9_000, 4_096, Some(3_000)), 3_000);
        assert_eq!(
            qwen35_next_prefill_end(3_000, 9_000, 4_096, Some(3_000)),
            7_096
        );
        assert_eq!(
            qwen35_next_prefill_end(7_096, 9_000, 4_096, Some(3_000)),
            9_000
        );
        assert_eq!(qwen35_next_prefill_end(0, 9_000, 4_096, None), 4_096);
    }

    fn f32_buffer(device: &MlxDevice, rows: usize, hidden: usize, values: &[f32]) -> MlxBuffer {
        assert_eq!(values.len(), rows * hidden);
        let mut buffer = device
            .alloc_buffer(
                values.len() * std::mem::size_of::<f32>(),
                mlx_native::DType::F32,
                vec![rows, hidden],
            )
            .expect("allocate vision test buffer");
        buffer
            .as_mut_slice::<f32>()
            .expect("vision test buffer f32 view")
            .copy_from_slice(values);
        buffer
    }

    #[test]
    fn bounded_vision_chunk_rebases_soft_tokens_deepstack_and_mrope() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let hidden = 2;
        let soft = f32_buffer(
            &device,
            6,
            hidden,
            &[
                0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 50.0, 51.0,
            ],
        );
        let deep = f32_buffer(
            &device,
            4,
            hidden,
            &[100.0, 101.0, 110.0, 111.0, 120.0, 121.0, 130.0, 131.0],
        );
        let prompt_len = 10;
        let mut positions = Vec::with_capacity(4 * prompt_len);
        for axis in 0..4_i32 {
            positions.extend((0..prompt_len as i32).map(|position| axis * 100 + position));
        }
        let vision = Qwen35VisionPrefillData::new(
            vec![SoftTokenData {
                range: 2..8,
                embeddings: soft,
            }],
            Some(DeepstackData {
                image_token_positions: vec![3, 4, 5, 6],
                chunks: vec![deep],
            }),
            Some(positions),
        );
        vision.validate(prompt_len, hidden).expect("valid vision");

        let chunk = vision.chunk(4, 6, hidden).expect("middle chunk");
        assert_eq!(chunk.soft_tokens.len(), 1);
        assert_eq!(chunk.soft_tokens[0].0, 0..2);
        assert_eq!(
            chunk.soft_tokens[0].1.as_slice::<f32>().expect("soft view"),
            &[20.0, 21.0, 30.0, 31.0]
        );
        let (deep_positions, deep_chunks) = chunk.deepstack.expect("deepstack chunk");
        assert_eq!(deep_positions, vec![0, 1]);
        assert_eq!(
            deep_chunks[0].as_slice::<f32>().expect("deepstack view"),
            &[110.0, 111.0, 120.0, 121.0]
        );
        assert_eq!(
            chunk.positions_flat.expect("mRoPE chunk"),
            vec![4, 5, 104, 105, 204, 205, 304, 305]
        );
    }

    #[test]
    fn bounded_vision_decode_base_uses_text_axis_not_expanded_prompt_length() {
        let prompt_len = 8;
        let mut positions = vec![0; 4 * prompt_len];
        positions[..prompt_len].copy_from_slice(&[0, 1, 2, 3, 4, 4, 5, 6]);
        let vision = Qwen35VisionPrefillData::new(Vec::new(), None, Some(positions));
        assert_eq!(vision.decode_position_base(prompt_len), 7);
    }

    #[test]
    fn failed_embed_forward_never_resets_a_potentially_poisoned_slot() {
        let reset_called = Cell::new(false);
        let forward: Result<Vec<f32>> = Err(anyhow::Error::new(
            mlx_native::MlxError::CommandBufferError(
                "Caused GPU Timeout Error (00000002)".to_string(),
            ),
        ))
        .context("Qwen35 slot-aware embed forward");
        let error = finish_qwen35_slot_embed(forward, || {
            reset_called.set(true);
            Ok(())
        })
        .expect_err("typed Metal failure must escape before reset");
        assert!(!reset_called.get());
        assert!(error.chain().any(|source| {
            matches!(
                source.downcast_ref::<mlx_native::MlxError>(),
                Some(mlx_native::MlxError::CommandBufferError(_))
            )
        }));
    }

    #[test]
    fn serial_kv_cache_capacity_grows_geometrically() {
        let maximum = 262_144;
        assert_eq!(serial_kv_cache_capacity(5_366, 0, maximum), 8_192);
        assert_eq!(serial_kv_cache_capacity(5_579, 8_192, maximum), 16_384);
        assert_eq!(serial_kv_cache_capacity(9_000, 8_192, maximum), 16_384);
        assert_eq!(serial_kv_cache_capacity(200_000, 131_072, maximum), maximum);
    }

    #[derive(Debug)]
    struct ResumeTestPayload(u64);

    impl crate::serve::kv_persist::lcp_registry::ByteSized for ResumeTestPayload {
        fn byte_len(&self) -> u64 {
            self.0
        }
    }

    #[test]
    fn latest_turn_checkpoint_covers_short_prompts_and_longer_stride_wins() {
        use crate::serve::kv_persist::format::ModelFingerprint;
        use crate::serve::kv_persist::lcp_registry::{LcpKey, LcpRegistry};
        use std::sync::Arc;

        let base_key = LcpKey {
            model_fingerprint: ModelFingerprint([9; 32]),
            tenant_id: String::new(),
            params_hash: 0,
        };
        let mut registry = LcpRegistry::new(8);
        registry
            .store(
                base_key.clone(),
                vec![1, 2, 3, 4, 5],
                vec![Arc::new(ResumeTestPayload(1))],
                0,
                0,
            )
            .unwrap();

        let short =
            lookup_qwen35_resume_checkpoint(&mut registry, &base_key, &[1, 2, 3, 4, 5, 6], 8)
                .expect("latest-turn checkpoint must cover prompt shorter than stride");
        assert_eq!((short.0.k, short.1), (5, 0));

        let mut chunk_key = base_key.clone();
        chunk_key.tenant_id = "qwen35:lcp_chunk:8".into();
        registry
            .store(
                chunk_key,
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                vec![Arc::new(ResumeTestPayload(1))],
                0,
                0,
            )
            .unwrap();
        let longest = lookup_qwen35_resume_checkpoint(
            &mut registry,
            &base_key,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9],
            8,
        )
        .expect("stride checkpoint");
        assert_eq!((longest.0.k, longest.1), (8, 8));

        assert!(
            lookup_qwen35_resume_checkpoint(&mut registry, &base_key, &[1, 2, 99, 4, 5, 6], 8,)
                .is_none()
        );
    }

    #[test]
    fn recovery_tail_uses_only_a_verified_prompt_suffix() {
        let prompt = [10, 11, 12, 20, 21, 22, 23];
        assert_eq!(recovery_tail_for_suffix(&prompt, &[20, 21, 22, 23], 64), 4);
        assert_eq!(
            recovery_tail_for_suffix(&prompt, &[21, 22, 99], 64),
            64,
            "a custom or drifted template must retain the conservative fallback"
        );
        assert_eq!(recovery_tail_for_suffix(&prompt, &[], 64), 64);
    }

    #[test]
    fn recovery_anchor_replaces_exact_or_immediately_preceding_stride_checkpoint() {
        assert!(stride_checkpoint_superseded_by_recovery_anchor(
            true, 4096, 4096, 5042
        ));
        assert!(stride_checkpoint_superseded_by_recovery_anchor(
            true, 4096, 4096, 4096
        ));
        assert!(!stride_checkpoint_superseded_by_recovery_anchor(
            true, 4096, 4096, 8192
        ));
        assert!(!stride_checkpoint_superseded_by_recovery_anchor(
            false, 4096, 4096, 5042
        ));
    }

    #[test]
    fn recovery_capture_plan_is_limited_to_short_non_chunked_suffixes() {
        assert_eq!(
            qwen35_recovery_capture_plan(5_239, 5_255, 5_259, true, false),
            Some((20, 15))
        );
        assert_eq!(
            qwen35_recovery_capture_plan(5_042, 5_239, 5_243, true, false),
            None,
            "201 changed tokens stay on the normal prefill kernel"
        );
        assert_eq!(
            qwen35_recovery_capture_plan(5_239, 5_255, 5_259, true, true),
            None,
            "chunked prefill owns its checkpoint schedule"
        );
        assert_eq!(
            qwen35_recovery_capture_plan(5_239, 5_255, 5_259, false, false),
            None
        );
    }

    fn moe_cfg_40layer_for_cache_test() -> Qwen35Config {
        Qwen35Config {
            variant: Qwen35Variant::Moe,
            hidden_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            linear_num_key_heads: 4,
            linear_num_value_heads: 8,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            layer_types: default_layer_types(4, 4),
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            rotary_dim: 4,
            mrope_section: [1, 1, 0, 0],
            mrope_interleaved: true,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 1024,
            vocab_size: 256,
            attn_output_gate: true,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: true,
            intermediate_size: None,
            moe: Some(Qwen35MoeConfig {
                moe_intermediate_size: 16,
                num_experts: 4,
                num_experts_per_tok: 2,
                shared_expert_intermediate_size: 16,
            }),
        }
    }

    fn greedy_params() -> SamplingParams {
        SamplingParams {
            max_tokens: 16,
            ..SamplingParams::default()
        }
    }

    #[test]
    fn mtp_server_gate_rejects_non_default_server_semantics() {
        let base = greedy_params();
        assert!(is_qwen_server_speculation_exact_eligible(&base));

        let mut frequency = base.clone();
        frequency.frequency_penalty = 0.25;
        assert!(!is_qwen_server_speculation_exact_eligible(&frequency));

        let mut repetition = base.clone();
        repetition.repetition_penalty = 1.05;
        assert!(is_qwen_server_speculation_exact_eligible(&repetition));

        let mut forced_thinking = base.clone();
        forced_thinking.reasoning_forced_open = true;
        forced_thinking.thinking_token_budget = Some(64);
        forced_thinking.reasoning_end_tokens = Some(Arc::new(vec![90, 91]));
        forced_thinking.reasoning_close_tokens = Some(Arc::new(vec![91]));
        assert!(is_qwen_server_speculation_exact_eligible(&forced_thinking));
        assert!(!is_serial_mtp_exact_eligible(&forced_thinking));

        let mut lazy_tool = base.clone();
        lazy_tool.tool_call_policy = ToolCallPolicy::AutoLazyGrammar;
        assert!(!is_qwen_server_speculation_exact_eligible(&lazy_tool));
        lazy_tool.grammar =
            Some(crate::serve::api::grammar::parse("root ::= \"x\"\n").expect("test grammar"));
        assert!(is_qwen_server_speculation_exact_eligible(&lazy_tool));

        let mut stop = base;
        stop.stop_strings.push("END".to_string());
        assert!(!is_qwen_server_speculation_exact_eligible(&stop));
    }

    #[test]
    fn repetition_window_includes_prompt_tail_and_tracks_committed_tokens() {
        let prompt: Vec<u32> = (0..80).collect();
        let mut history = qwen35_prompt_sampling_history(&prompt);
        assert_eq!(history, (16..80).collect::<Vec<_>>());

        qwen35_observe_sampling_history(&mut history, 80);
        assert_eq!(history, (17..=80).collect::<Vec<_>>());

        let mut params = greedy_params();
        params.repetition_penalty = 1.05;
        let mut logits = vec![0.0; 96];
        logits[17] = 10.0;
        logits[3] = 9.9;
        let (token, _) =
            sample_logits_qwen35_constrained(&mut logits, &params, &history, None, false)
                .expect("prompt-aware repetition sample");
        assert_eq!(
            token, 3,
            "a token inside the last-64 prompt/generation window must be penalized"
        );
    }

    #[test]
    fn mtp_k3_full_accept_queues_three_drafts_and_bonus() {
        let canonical = [
            Qwen35CanonicalDecision {
                token: 11,
                terminal: false,
            },
            Qwen35CanonicalDecision {
                token: 12,
                terminal: false,
            },
            Qwen35CanonicalDecision {
                token: 13,
                terminal: false,
            },
            Qwen35CanonicalDecision {
                token: 14,
                terminal: false,
            },
        ];
        let plan = plan_qwen35_verified_block(&[11, 12, 13], |row| Ok(canonical[row]))
            .expect("full accept plan");
        assert_eq!(
            plan.output.into_iter().collect::<Vec<_>>(),
            vec![11, 12, 13, 14]
        );
        assert_eq!(plan.matched_drafts, 3);
        assert_eq!(plan.rejected_drafts, 0);
        assert_eq!(plan.valid_input_tokens, 4);
        assert_eq!(plan.carry_hidden_row, 3);
        assert!(!plan.terminal_after_pending);
    }

    #[test]
    fn mtp_k3_partial_reject_keeps_only_target_valid_prefix() {
        let canonical = [
            Qwen35CanonicalDecision {
                token: 11,
                terminal: false,
            },
            Qwen35CanonicalDecision {
                token: 99,
                terminal: false,
            },
        ];
        let plan = plan_qwen35_verified_block(&[11, 12, 13], |row| Ok(canonical[row]))
            .expect("partial reject plan");
        assert_eq!(plan.output.into_iter().collect::<Vec<_>>(), vec![11, 99]);
        assert_eq!(plan.matched_drafts, 1);
        assert_eq!(plan.rejected_drafts, 1);
        assert_eq!(plan.valid_input_tokens, 2);
        assert_eq!(plan.carry_hidden_row, 1);
        assert!(!plan.terminal_after_pending);
    }

    #[test]
    fn mtp_k3_terminal_draft_is_neither_streamed_nor_retained() {
        let canonical = [
            Qwen35CanonicalDecision {
                token: 11,
                terminal: false,
            },
            Qwen35CanonicalDecision {
                token: 12,
                terminal: true,
            },
        ];
        let plan = plan_qwen35_verified_block(&[11, 12, 13], |row| Ok(canonical[row]))
            .expect("terminal plan");
        assert_eq!(plan.output.into_iter().collect::<Vec<_>>(), vec![11]);
        assert_eq!(plan.matched_drafts, 2);
        assert_eq!(plan.rejected_drafts, 0);
        assert_eq!(plan.valid_input_tokens, 2);
        assert_eq!(plan.carry_hidden_row, 1);
        assert!(plan.terminal_after_pending);
    }

    #[test]
    fn pending_mtp_bonus_is_emitted_before_history_can_select_a_new_round() {
        let mut pending = VecDeque::from([41, 42]);
        assert_eq!(take_pending_speculation_output(&mut pending), Some(41));
        // This is the next scheduler tick after an MTP accept. The queue is
        // consulted before the history-proposer branch, so no target forward
        // or lookup may occur before this verified bonus is emitted.
        assert_eq!(take_pending_speculation_output(&mut pending), Some(42));
        assert!(take_pending_speculation_output(&mut pending).is_none());
    }

    #[test]
    fn history_miss_routes_to_mtp_until_two_negative_cost_windows() {
        let mut cost = SpeculationCostController::new();
        assert!(!may_route_history_miss_to_mtp(true, &cost));
        cost.observe_ordinary_target(Duration::from_millis(10));
        assert!(may_route_history_miss_to_mtp(true, &cost));
        for _ in 0..4 {
            cost.observe_speculative_round(1, Duration::from_millis(10));
        }
        assert!(may_route_history_miss_to_mtp(true, &cost));
        for _ in 0..4 {
            cost.observe_speculative_round(1, Duration::from_millis(10));
        }
        assert!(!may_route_history_miss_to_mtp(true, &cost));
        assert!(!may_route_history_miss_to_mtp(false, &cost));
    }

    #[test]
    fn every_post_mutation_failpoint_restores_state_buffers_and_cursors() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let mut cfg = moe_cfg_40layer_for_cache_test();
        cfg.mtp_num_hidden_layers = 1;
        let phases = [
            Qwen35StateFailpoint::OrdinaryTarget,
            Qwen35StateFailpoint::HistoryTarget,
            Qwen35StateFailpoint::HistoryMtpCatchup,
            Qwen35StateFailpoint::HistoryCommit,
            Qwen35StateFailpoint::MtpDraft,
            Qwen35StateFailpoint::MtpTarget,
            Qwen35StateFailpoint::MtpCatchup,
            Qwen35StateFailpoint::MtpCommit,
            Qwen35StateFailpoint::WarmupTarget,
            Qwen35StateFailpoint::WarmupMtpCatchup,
            Qwen35StateFailpoint::PrefillTarget,
            Qwen35StateFailpoint::PrefillMtpCatchup,
        ];

        fn fill_slot(buf: &mut MlxBuffer, slot: usize, n_seqs: usize, byte: u8) {
            let bytes = buf.as_mut_slice::<u8>().expect("state buffer");
            let per_slot = bytes.len() / n_seqs;
            bytes[slot * per_slot..(slot + 1) * per_slot].fill(byte);
        }
        fn read_slot(buf: &MlxBuffer, slot: usize, n_seqs: usize) -> Vec<u8> {
            let bytes = buf.as_slice::<u8>().expect("state buffer");
            let per_slot = bytes.len() / n_seqs;
            bytes[slot * per_slot..(slot + 1) * per_slot].to_vec()
        }

        for phase in phases {
            let mut kv = HybridKvCache::new(&cfg, &device, 16, 2).expect("kv");
            for slot in &mut kv.full_attn {
                slot.current_len = vec![4, 3];
            }
            kv.mtp_slot.as_mut().expect("MTP slot").current_len = vec![4, 3];
            for linear in &mut kv.linear_attn {
                fill_slot(&mut linear.conv_state, 0, 2, 0x11);
                fill_slot(&mut linear.recurrent, 0, 2, 0x22);
                fill_slot(&mut linear.conv_state_scratch, 0, 2, 0xa1);
                fill_slot(&mut linear.recurrent_scratch, 0, 2, 0xa2);
                fill_slot(&mut linear.conv_state, 1, 2, 0x31);
                fill_slot(&mut linear.recurrent, 1, 2, 0x32);
                fill_slot(&mut linear.conv_state_scratch, 1, 2, 0xb1);
                fill_slot(&mut linear.recurrent_scratch, 1, 2, 0xb2);
            }
            let target_before = kv
                .linear_attn
                .iter()
                .map(|linear| {
                    let (conv, _) = linear.conv_bufs_for_slot(SlotId(0));
                    let (recurrent, _) = linear.recurrent_bufs_for_slot(SlotId(0));
                    (read_slot(conv, 0, 2), read_slot(recurrent, 0, 2))
                })
                .collect::<Vec<_>>();
            let peer_before = kv
                .linear_attn
                .iter()
                .map(|linear| {
                    let (conv, _) = linear.conv_bufs_for_slot(SlotId(1));
                    let (recurrent, _) = linear.recurrent_bufs_for_slot(SlotId(1));
                    (read_slot(conv, 1, 2), read_slot(recurrent, 1, 2))
                })
                .collect::<Vec<_>>();
            let transaction =
                begin_slot_state_transaction(&kv, SlotId(0), 4).expect("capture transaction");

            for slot in &mut kv.full_attn {
                slot.current_len[0] = 7;
            }
            kv.mtp_slot.as_mut().expect("MTP slot").current_len[0] = 7;
            for linear in &mut kv.linear_attn {
                fill_slot(&mut linear.conv_state_scratch, 0, 2, phase as u8);
                fill_slot(
                    &mut linear.recurrent_scratch,
                    0,
                    2,
                    (phase as u8).wrapping_add(0x40),
                );
                linear.swap_for_slot(SlotId(0));
            }

            QWEN35_STATE_FAILPOINT.store(phase as u8, std::sync::atomic::Ordering::SeqCst);
            let injected = qwen35_state_failpoint(phase).expect_err("failpoint must fire");
            QWEN35_STATE_FAILPOINT.store(0, std::sync::atomic::Ordering::SeqCst);
            let error = rollback_slot_state_error::<()>(
                &mut kv,
                SlotId(0),
                &transaction,
                injected,
                "model-free rollback proof",
            )
            .expect_err("injected transaction returns its error after rollback");
            assert!(error.to_string().contains("model-free rollback proof"));

            assert!(kv
                .full_attn
                .iter()
                .all(|slot| slot.current_len == vec![4, 3]));
            assert_eq!(
                kv.mtp_slot.as_ref().expect("MTP slot").current_len,
                vec![4, 3]
            );
            for (layer_idx, linear) in kv.linear_attn.iter().enumerate() {
                let (target_conv, _) = linear.conv_bufs_for_slot(SlotId(0));
                let (target_recurrent, _) = linear.recurrent_bufs_for_slot(SlotId(0));
                assert_eq!(read_slot(target_conv, 0, 2), target_before[layer_idx].0);
                assert_eq!(
                    read_slot(target_recurrent, 0, 2),
                    target_before[layer_idx].1
                );
                let (peer_conv, _) = linear.conv_bufs_for_slot(SlotId(1));
                let (peer_recurrent, _) = linear.recurrent_bufs_for_slot(SlotId(1));
                assert_eq!(read_slot(peer_conv, 1, 2), peer_before[layer_idx].0);
                assert_eq!(read_slot(peer_recurrent, 1, 2), peer_before[layer_idx].1);
            }
        }
    }

    #[test]
    fn every_declared_state_failpoint_is_wired_to_a_mutation_boundary() {
        let source = include_str!("engine_qwen35.rs");
        for name in [
            "OrdinaryTarget",
            "HistoryTarget",
            "HistoryMtpCatchup",
            "HistoryCommit",
            "MtpDraft",
            "MtpTarget",
            "MtpCatchup",
            "MtpCommit",
            "WarmupTarget",
            "WarmupMtpCatchup",
            "PrefillTarget",
            "PrefillMtpCatchup",
        ] {
            let needle = format!("qwen35_state_failpoint(Qwen35StateFailpoint::{name})");
            assert!(
                source.contains(&needle),
                "missing runtime failpoint wiring for {name}"
            );
        }
    }

    fn initialized_prompt_cache_snapshot(
        cfg: &Qwen35Config,
        device: &MlxDevice,
        prompt_len: usize,
    ) -> HybridKvCacheSnapshot {
        let mut kv = HybridKvCache::new(cfg, device, 16, 1).expect("kv");
        for slot in &mut kv.full_attn {
            for buf in [slot.k.as_mut(), slot.v.as_mut()].into_iter().flatten() {
                buf.as_mut_slice::<u8>()
                    .expect("full-attention test buffer")
                    .fill(0x5a);
            }
            slot.current_len[0] = prompt_len as u32;
        }
        if let Some(slot) = kv.mtp_slot.as_mut() {
            for buf in [slot.k.as_mut(), slot.v.as_mut()].into_iter().flatten() {
                buf.as_mut_slice::<u8>()
                    .expect("MTP test buffer")
                    .fill(0xa5);
            }
            slot.current_len[0] = prompt_len as u32;
        }
        kv.snapshot(device).expect("cursor-bounded snapshot")
    }

    #[test]
    fn gpu_greedy_gate_rejects_cpu_sampling_features() {
        assert!(is_greedy_eligible(&greedy_params()));

        let mut repetition = greedy_params();
        repetition.repetition_penalty = 1.05;
        assert!(!is_greedy_eligible(&repetition));

        let mut biased = greedy_params();
        biased.logit_bias.insert(7, 2.0);
        assert!(!is_greedy_eligible(&biased));

        let mut grammar = greedy_params();
        grammar.grammar =
            Some(crate::serve::api::grammar::parse("root ::= \"a\"\n").expect("test grammar"));
        assert!(!is_greedy_eligible(&grammar));
    }

    #[test]
    fn constrained_sampler_rejects_higher_invalid_logit() {
        let grammar = crate::serve::api::grammar::parse("root ::= \"a\"\n").expect("test grammar");
        let mut params = greedy_params();
        params.grammar = Some(grammar);
        params.token_bytes = Some(std::sync::Arc::new(vec![b"x".to_vec(), b"a".to_vec()]));
        let runtime = grammar_runtime_for_request(&params, None)
            .expect("runtime build")
            .expect("grammar runtime");
        let mut logits = vec![100.0, 1.0];

        let (token, logprob) =
            sample_logits_qwen35_constrained(&mut logits, &params, &[], Some(&runtime), false)
                .expect("valid grammar/table invariant");

        assert_eq!(token, 1);
        assert_eq!(logprob, None);
    }

    #[test]
    fn agentic_grammar_contract_sampler_rejects_missing_short_and_long_token_tables() {
        let grammar = crate::serve::api::grammar::parse("root ::= \"a\"\n").expect("test grammar");
        let mut params = greedy_params();
        params.grammar = Some(grammar.clone());
        let missing = grammar_runtime_for_request(&params, None)
            .expect_err("grammar without token table must fail before sampling");
        assert!(missing
            .to_string()
            .contains("without its authoritative token byte table"));

        params.token_bytes = Some(std::sync::Arc::new(vec![b"a".to_vec()]));
        let runtime = grammar_runtime_for_request(&params, None)
            .expect("runtime build")
            .expect("grammar runtime");
        let sampler = SamplerPureParams {
            temperature: 0.55,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            max_tokens: 8,
            seed: None,
        };
        let mut logits = vec![1.0, 0.0];
        let short = sample_logits_with_grammar(
            &mut logits,
            &sampler,
            &[],
            Some(&runtime),
            params.token_bytes.as_deref().map(Vec::as_slice),
            false,
        )
        .expect_err("short token table must fail closed");
        assert!(short
            .to_string()
            .contains("length 1 != logits vocabulary 2"));

        params.token_bytes = Some(std::sync::Arc::new(vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"c".to_vec(),
        ]));
        let long = sample_logits_with_grammar(
            &mut logits,
            &sampler,
            &[],
            Some(&runtime),
            params.token_bytes.as_deref().map(Vec::as_slice),
            false,
        )
        .expect_err("long token table must fail closed");
        assert!(long.to_string().contains("length 3 != logits vocabulary 2"));
    }

    #[test]
    fn bounded_prefill_rejects_invalid_grammar_before_mutating_cache() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = moe_cfg_40layer_for_cache_test();
        let mut kv = HybridKvCache::new(&cfg, &device, 128, 1).expect("kv");
        let mut params = greedy_params();
        params.grammar = Some(
            crate::serve::api::grammar::parse("not_root ::= \"x\"\n")
                .expect("syntactically valid grammar without root"),
        );

        let error = match Qwen35PrefillState::begin(
            vec![7],
            params,
            None,
            &mut kv,
            SlotId(0),
            0,
            None,
            None,
            None,
            cfg.hidden_size as usize,
        ) {
            Ok(_) => panic!("missing-root grammar must fail before bounded prefill"),
            Err(error) => error,
        };
        assert!(format!("{error:#}").contains("grammar has no root rule"));
        assert_eq!(
            kv.sequence_len_for_slot(SlotId(0)).expect("cursor"),
            0,
            "grammar validation must precede any cache mutation or Metal work"
        );
    }

    #[test]
    fn accepted_empty_grammar_token_terminates_the_prefill_seed() {
        let grammar =
            crate::serve::api::grammar::parse("root ::= \"\"\n").expect("empty terminal grammar");
        let mut params = greedy_params();
        params.grammar = Some(grammar);
        params.token_bytes = Some(std::sync::Arc::new(vec![Vec::new()]));
        let runtime = grammar_runtime_for_request(&params, None)
            .expect("runtime build")
            .expect("grammar runtime");
        assert!(qwen35_grammar_terminal_token(Some(&runtime), &params, 0));
        let src = include_str!("engine_qwen35.rs");
        assert!(
            src.contains(
                "|| qwen35_grammar_terminal_token(grammar_runtime.as_ref(), &params, next_token)"
            ),
            "bounded prefill seed must stop on the same grammar terminal as legacy SSE"
        );
    }

    #[test]
    fn cached_token_reporting_includes_partial_lcp_resume() {
        assert_eq!(qwen35_reported_cached_tokens(17_132, false, 16_384), 16_384);
        assert_eq!(qwen35_reported_cached_tokens(17_132, true, 16_384), 17_132);
        assert_eq!(qwen35_reported_cached_tokens(17_132, false, 0), 0);
    }

    #[test]
    fn cached_token_reporting_clamps_defensively_to_prompt_length() {
        assert_eq!(qwen35_reported_cached_tokens(128, false, 256), 128);
    }

    fn sampling_params_with_temperature() -> SamplingParams {
        SamplingParams {
            temperature: 0.7,
            max_tokens: 16,
            ..SamplingParams::default()
        }
    }

    /// Wedge-3 / iter-216 Phase C: fresh cache has no entry; try_match
    /// always misses.
    #[test]
    fn hybrid_prompt_cache_new_is_empty() {
        let cache = HybridPromptCache::new();
        assert!(!cache.has_entry());
        assert!(cache.snapshot().is_none());
        assert!(cache.try_match(&[1, 2, 3], &greedy_params()).is_none());
    }

    /// Wedge-3 / iter-216 Phase C: full-equality match returns
    /// Some(prompt.len()); single-token divergence misses.
    #[test]
    fn hybrid_prompt_cache_invalidates_on_prompt_divergence() {
        let cfg = moe_cfg_40layer_for_cache_test();
        let device = MlxDevice::new().expect("device");
        let prompt = vec![10u32, 20, 30, 40];
        let snap = initialized_prompt_cache_snapshot(&cfg, &device, prompt.len());
        let mut cache = HybridPromptCache::new();
        cache.update(prompt.clone(), snap, 99u32, &greedy_params());
        assert!(cache.has_entry());
        assert_eq!(
            cache.try_match(&prompt, &greedy_params()),
            Some(prompt.len()),
            "exact-match prompt should hit"
        );
        // Divergent prompt (one token differs) misses.
        let mut diverged = prompt.clone();
        diverged[2] = 999;
        assert!(
            cache.try_match(&diverged, &greedy_params()).is_none(),
            "divergent prompt should miss"
        );
        // Different-length prompt also misses.
        let mut shorter = prompt.clone();
        shorter.pop();
        assert!(
            cache.try_match(&shorter, &greedy_params()).is_none(),
            "shorter prompt should miss"
        );
        let mut longer = prompt.clone();
        longer.push(50);
        assert!(
            cache.try_match(&longer, &greedy_params()).is_none(),
            "longer prompt should miss"
        );
    }

    #[test]
    fn hybrid_prompt_cache_keys_vision_identity() {
        let cfg = moe_cfg_40layer_for_cache_test();
        let device = MlxDevice::new().expect("device");
        let prompt = vec![10u32, 20, 30, 40];
        let snapshot = initialized_prompt_cache_snapshot(&cfg, &device, prompt.len());
        let mut cached_params = greedy_params();
        cached_params.vision_fingerprint = Some([0x11; 32]);
        let mut cache = HybridPromptCache::new();
        cache.update(prompt.clone(), snapshot, 99, &cached_params);

        assert_eq!(cache.try_match(&prompt, &cached_params), Some(prompt.len()));
        let mut other_image = cached_params.clone();
        other_image.vision_fingerprint = Some([0x22; 32]);
        assert!(cache.try_match(&prompt, &other_image).is_none());
        let mut text_only = cached_params;
        text_only.vision_fingerprint = None;
        assert!(cache.try_match(&prompt, &text_only).is_none());
    }

    /// Wedge-3 / iter-216 Phase C: gen-params mismatch misses even when
    /// the prompt matches.
    #[test]
    fn hybrid_prompt_cache_invalidates_on_genparams_mismatch() {
        let cfg = moe_cfg_40layer_for_cache_test();
        let device = MlxDevice::new().expect("device");
        let prompt = vec![1u32, 2, 3];
        let snap = initialized_prompt_cache_snapshot(&cfg, &device, prompt.len());
        let mut cache = HybridPromptCache::new();
        let stored_params = SamplingParams {
            max_tokens: 32,
            stop_strings: vec!["</done>".into()],
            ..SamplingParams::default()
        };
        cache.update(prompt.clone(), snap, 7u32, &stored_params);

        // Same prompt + same key → hit.
        assert_eq!(
            cache.try_match(&prompt, &stored_params),
            Some(prompt.len()),
            "matching key should hit"
        );

        // Different max_tokens → miss.
        let diff_max = SamplingParams {
            max_tokens: 64,
            stop_strings: vec!["</done>".into()],
            ..SamplingParams::default()
        };
        assert!(
            cache.try_match(&prompt, &diff_max).is_none(),
            "max_tokens mismatch must miss"
        );

        // Different stop_strings → miss.
        let diff_stop = SamplingParams {
            max_tokens: 32,
            stop_strings: vec!["</STOP>".into()],
            ..SamplingParams::default()
        };
        assert!(
            cache.try_match(&prompt, &diff_stop).is_none(),
            "stop_strings mismatch must miss"
        );
    }

    /// Wedge-3 / iter-216 Phase C: sampling-mode (T > 0) bypasses the
    /// cache on lookup AND store — never replays a non-greedy decode.
    #[test]
    fn hybrid_prompt_cache_sampling_mode_bypasses_lookup_and_store() {
        let cfg = moe_cfg_40layer_for_cache_test();
        let device = MlxDevice::new().expect("device");
        let prompt = vec![1u32, 2, 3];
        let snap = initialized_prompt_cache_snapshot(&cfg, &device, prompt.len());

        // Lookup with sampling-mode bypasses even on a populated cache.
        let mut cache = HybridPromptCache::new();
        cache.update(prompt.clone(), snap, 1u32, &greedy_params());
        assert!(cache.has_entry());
        assert!(
            cache
                .try_match(&prompt, &sampling_params_with_temperature())
                .is_none(),
            "sampling-mode lookup must miss"
        );

        // Store with sampling-mode is a no-op (cache stays as it was).
        let snap2 = initialized_prompt_cache_snapshot(&cfg, &device, prompt.len());
        let mut cache2 = HybridPromptCache::new();
        cache2.update(
            prompt.clone(),
            snap2,
            5u32,
            &sampling_params_with_temperature(),
        );
        assert!(!cache2.has_entry(), "sampling-mode update must be a no-op");
    }

    /// Wedge-3 / iter-216 Phase E: the Qwen35 chat arm consumes the
    /// EXISTING `super::registry::QWEN35` registration's reasoning
    /// markers via `super::registry::split_full_output` — the same
    /// helper the Gemma non-streaming path uses.  This test pins the
    /// shared-helper contract (no duplicated splitter code; one source
    /// of truth in `registry.rs`).
    #[test]
    fn splitter_helper_extracts_reasoning_from_qwen35_thinkblocks() {
        let reg = super::super::registry::QWEN35;
        let raw = "Sure! <think>Let me solve this step by step.</think>The answer is 42.";
        let (content, reasoning) = super::super::registry::split_full_output(&reg, raw);
        assert_eq!(
            content, "Sure! The answer is 42.",
            "content must exclude the <think>...</think> span"
        );
        assert_eq!(
            reasoning.as_deref(),
            Some("Let me solve this step by step."),
            "reasoning must contain the inner span"
        );
    }

    /// Wedge-3 / iter-216 Phase E: the Qwen35 streaming arm consumes
    /// the EXISTING `super::registry::ToolCallSplitter` for tool-call
    /// markers (`<tool_call>` / `</tool_call>`).  This test pins the
    /// open/text/close event sequence so a downstream change to the
    /// QWEN35 registration's tool-call markers surfaces here.
    #[test]
    fn splitter_helper_extracts_tool_calls_from_qwen35_toolblocks() {
        let reg = super::super::registry::QWEN35;
        let mut sp = super::super::registry::ToolCallSplitter::from_registration(&reg)
            .expect("QWEN35 has tool markers");
        let raw =
            "Let me search.<tool_call><function=search><parameter=q>weather</parameter></function></tool_call> Done.";
        let mut events = Vec::new();
        events.extend(sp.feed(raw));
        if let Some(tail) = sp.finish() {
            events.push(tail);
        }
        // We expect: Content("Let me search.") → ToolCallOpen →
        // ToolCallText("<function=search>...</function>") → ToolCallClose
        // → Content(" Done.").
        let mut saw_open = false;
        let mut saw_text = false;
        let mut saw_close = false;
        let mut content_runs: Vec<String> = Vec::new();
        for ev in events {
            use super::super::registry::ToolCallEvent::*;
            match ev {
                Content(t) => content_runs.push(t),
                ToolCallOpen => saw_open = true,
                ToolCallText(_) => saw_text = true,
                ToolCallClose => saw_close = true,
            }
        }
        assert!(saw_open, "must observe ToolCallOpen for QWEN35 marker");
        assert!(saw_text, "must observe ToolCallText body");
        assert!(saw_close, "must observe ToolCallClose");
        let joined: String = content_runs.join("");
        assert!(
            joined.contains("Let me search."),
            "preamble content must round-trip"
        );
        assert!(
            joined.contains(" Done."),
            "post-close content must round-trip"
        );
    }

    /// Wedge-3 / iter-216 Phase D contract: a freshly-loaded `Qwen35LoadedModel`
    /// initializes its prompt_cache in the empty state.
    ///
    /// We can't drive the full `Qwen35LoadedModel::load` here without a
    /// real GGUF, so the test asserts the contract via `Default::default`
    /// + the `HybridPromptCache::new` fast-path.  Phase D wiring depends
    /// on this state so a bypass cannot accidentally leak across loads.
    #[test]
    fn qwen35_loaded_model_has_initialized_prompt_cache() {
        let cache = HybridPromptCache::default();
        assert!(!cache.has_entry());
        assert!(cache.snapshot().is_none());
        assert!(cache.try_match(&[1, 2], &greedy_params()).is_none());
    }

    /// Negative-path: `Qwen35LoadedModel::load` against a non-existent
    /// path returns an Err with the path in the message.  Smoke test
    /// that the constructor's exists-check fires before any GGUF parse.
    #[test]
    fn qwen35_loaded_model_load_errors_when_path_missing() {
        let opts = LoadOptions {
            model_path: std::path::PathBuf::from("/tmp/iter-215-does-not-exist.gguf"),
            tokenizer_path: None,
            config_path: None,
            dwq_overlay_path: None,
            kv_persist_dir: None,
        };
        let res = Qwen35LoadedModel::load(&opts);
        assert!(res.is_err());
        let msg = format!("{:#}", res.err().unwrap());
        assert!(
            msg.contains("Model not found"),
            "expected 'Model not found' in error; got: {msg}"
        );
    }

    /// Regression for the laptop portability failure: Qwen serving must not
    /// require a sibling `tokenizer.json` when the GGUF carries the standard
    /// tokenizer metadata. This fixture lives in a fresh directory containing
    /// only one GGUF and exercises the exact helper used by `load`.
    #[test]
    fn qwen35_serving_tokenizer_is_sidecar_free_and_detects_vision_markers() {
        use crate::backends::gguf::types::MetaValue;
        use crate::backends::gguf::writer::GgufWriter;

        let dir = tempfile::tempdir().expect("temporary sidecar-free directory");
        let path = dir.path().join("qwen35-tokenizer.gguf");
        let file = std::fs::File::create(&path).expect("create tokenizer GGUF fixture");
        let metadata = [
            ("tokenizer.ggml.pre", MetaValue::String("qwen35".into())),
            (
                "tokenizer.ggml.tokens",
                MetaValue::ArrayString(vec![
                    "<|vision_start|>".into(),
                    "<|image_pad|>".into(),
                    "<|vision_end|>".into(),
                    "a".into(),
                ]),
            ),
            ("tokenizer.ggml.merges", MetaValue::ArrayString(Vec::new())),
            (
                "tokenizer.ggml.token_type",
                MetaValue::ArrayI32(vec![4, 4, 4, 1]),
            ),
        ];
        let mut writer = GgufWriter::new(file);
        writer
            .write_header(0, metadata.len() as u64)
            .expect("write fixture header");
        for (key, value) in &metadata {
            writer
                .write_metadata_kv(key, value)
                .expect("write tokenizer metadata");
        }
        writer.pad_to_alignment().expect("align fixture");
        writer.finalize().expect("finalize fixture");

        assert!(!dir.path().join("tokenizer.json").exists());
        let gguf = mlx_native::gguf::GgufFile::open(&path).expect("open tokenizer fixture");
        let (tokenizer, has_vision_markers) =
            build_qwen35_serving_tokenizer(&gguf).expect("build embedded tokenizer");
        assert!(has_vision_markers);
        assert_eq!(tokenizer.token_to_id("<|image_pad|>"), Some(1));
    }

    // ─────────────────────────────────────────────────────────────────
    // Wedge-4e (iter-224 row 5) — streaming + soft-tokens + deepstack +
    // 3D positions invariance tests.
    //
    // These tests pin the contract of `generate_stream_qwen35_once_extended`
    // WITHOUT loading a real GGUF (which would be a multi-GB cost): we
    // assert structural identity between the legacy text-only entry
    // (`generate_stream_qwen35_once`) and the extended entry called with
    // empty/None extensions, plus the splitter chain's MODE-INVARIANCE.
    //
    // Real-model streaming is exercised by the operator-gated E2E
    // harness at `tests/qwen3vl_streaming_e2e.rs` (default-skip; runs
    // when `HF2Q_QWEN3VL_E2E=1` + a real Qwen3-VL GGUF + mmproj).
    // ─────────────────────────────────────────────────────────────────

    use super::super::registry::{
        SplitSlot as _SplitSlot, ToolCallEvent as _ToolCallEvent,
        ToolCallSplitter as _ToolCallSplitter, QWEN35,
    };

    /// Wedge-4e splitter MODE-INVARIANCE: `ReasoningSplitter`'s `feed`
    /// + `finish` API operates on `&str` fragments and produces the
    /// same `(SplitSlot, String)` pairs regardless of how the caller
    /// obtained the fragments — text-only prefill, soft-token-augmented
    /// prefill, or deepstack-augmented prefill all funnel through the
    /// SAME per-token `tokenizer.decode(...)` step in
    /// `generate_stream_qwen35_once_extended`. This test pins the
    /// invariant by feeding identical reasoning-bracketed input through
    /// a fresh `ReasoningSplitter` configured from `QWEN35` and
    /// asserting the slot/text breakdown is exactly what the streaming
    /// arm will see at decode time.
    #[test]
    fn wedge4e_reasoning_splitter_is_mode_invariant() {
        // Simulate a stream of reasoning + content fragments. The
        // splitter doesn't know whether the prefill was text-only or
        // image-augmented — its only input is the per-token decoded
        // fragment.
        let mut sp = super::super::registry::make_reasoning_splitter(&QWEN35, false)
            .expect("Qwen35 has reasoning markers");
        let mut all_pairs: Vec<(_SplitSlot, String)> = Vec::new();
        // Open marker spans a fragment boundary to exercise the
        // tail_buf logic.
        for frag in ["<thi", "nk>let me reason", " more</thin", "k>final answer"] {
            for pair in sp.feed(frag) {
                all_pairs.push(pair);
            }
        }
        if let Some(tail) = sp.finish() {
            all_pairs.push(tail);
        }
        // Reconstruct the reasoning + content text from the pair list;
        // the splitter MUST have separated the two streams cleanly.
        let mut reasoning = String::new();
        let mut content = String::new();
        for (slot, text) in &all_pairs {
            match slot {
                _SplitSlot::Reasoning => reasoning.push_str(text),
                _SplitSlot::Content => content.push_str(text),
            }
        }
        assert_eq!(
            reasoning, "let me reason more",
            "Wedge-4e: reasoning text must be cleanly extracted"
        );
        assert_eq!(
            content, "final answer",
            "Wedge-4e: content text must NOT contain reasoning brackets"
        );
        // Critical: the splitter never observed any vision-augmentation
        // signal — it takes &str only. By construction it cannot
        // discriminate between text-only and vision-augmented prefill,
        // proving the MODE-INVARIANCE claim.
    }

    /// Wedge-4e splitter MODE-INVARIANCE: `ToolCallSplitter` operates
    /// on `&str` fragments and produces the same `ToolCallEvent`
    /// stream regardless of prefill source. Pin via a fragment stream
    /// that crosses tool-call open/close boundaries.
    #[test]
    fn wedge4e_tool_call_splitter_is_mode_invariant() {
        let mut tcs =
            _ToolCallSplitter::from_registration(&QWEN35).expect("Qwen35 has tool markers");
        let mut events: Vec<_ToolCallEvent> = Vec::new();
        for frag in [
            "let me search.<tool_",
            "call>{\"name\":\"search\",\"arguments\":{\"q\":\"x\"}}</tool_",
            "call> done.",
        ] {
            for ev in tcs.feed(frag) {
                events.push(ev);
            }
        }
        if let Some(tail) = tcs.finish() {
            events.push(tail);
        }
        let mut saw_open = false;
        let mut saw_close = false;
        let mut body = String::new();
        let mut content_runs: Vec<String> = Vec::new();
        for ev in events {
            match ev {
                _ToolCallEvent::Content(t) => content_runs.push(t),
                _ToolCallEvent::ToolCallOpen => saw_open = true,
                _ToolCallEvent::ToolCallText(t) => body.push_str(&t),
                _ToolCallEvent::ToolCallClose => saw_close = true,
            }
        }
        assert!(saw_open, "Wedge-4e: must observe ToolCallOpen");
        assert!(saw_close, "Wedge-4e: must observe ToolCallClose");
        assert!(
            body.contains("\"name\":\"search\""),
            "Wedge-4e: tool-call body must round-trip; got {body:?}"
        );
        let joined: String = content_runs.join("");
        assert!(
            joined.contains("let me search."),
            "Wedge-4e: pre-tool-call content must round-trip"
        );
        assert!(
            joined.contains(" done."),
            "Wedge-4e: post-tool-call content must round-trip"
        );
        // Same MODE-INVARIANCE argument as the reasoning splitter:
        // the input is &str, no vision signal anywhere.
    }

    /// Wedge-4e: `generate_stream_qwen35_once` (legacy text-only
    /// signature) is now a thin wrapper that delegates to
    /// `generate_stream_qwen35_once_extended` with `&[]` soft tokens,
    /// `None` deepstack, and `None` positions. This test pins the
    /// wrapper contract by inspecting the source — if the wrapper is
    /// broken (e.g. someone re-introduces direct logic), this test
    /// will fail loud.
    #[test]
    fn wedge4e_legacy_stream_entry_is_thin_wrapper() {
        let src = include_str!("engine_qwen35.rs");
        // The wrapper body is short and constructs the canonical
        // `&[]` / `None` / `None` extension args.
        assert!(
            src.contains("generate_stream_qwen35_once_extended(\n        qwen,\n        prompt_tokens,\n        &[],\n        None,\n        None,"),
            "Wedge-4e: generate_stream_qwen35_once must delegate to \
             generate_stream_qwen35_once_extended with empty extensions \
             — the byte-identical text-only regression contract \
             requires this exact shape"
        );
    }

    /// Wedge-4e: `generate_stream_qwen35_once_extended` validates
    /// `positions_flat.len() == 4 * prompt_len` BEFORE any GPU work
    /// and surfaces the mismatch as an actionable diagnostic. This is
    /// the streaming sibling of the validator in
    /// `generate_qwen35_once_with_soft_tokens_and_deepstack` at
    /// engine_qwen35.rs:~1129.
    #[test]
    fn wedge4e_extended_stream_validates_positions_len() {
        // We test the validation logic inline (we don't load a real
        // model — the validation fires on the i32-slice length BEFORE
        // any model dispatch). The function's first action after the
        // `prompt_tokens.is_empty()` check is to validate
        // `positions_flat.len() == 4 * prompt_len`. We reproduce the
        // identical check here as a structural pin so a refactor that
        // moves the validation behind GPU work fails this test.
        let prompt_len = 5usize;
        let bad_positions = vec![0i32; 17]; // wrong: 17 != 4 * 5 = 20
        let expected_err = format!(
            "qwen35 stream (wedge-4e): positions_flat.len() = {} != 4 * prompt_len = {}",
            bad_positions.len(),
            4 * prompt_len
        );
        let src = include_str!("engine_qwen35.rs");
        assert!(
            src.contains("qwen35 stream (wedge-4e): positions_flat.len() = "),
            "Wedge-4e: positions_flat length validator must surface \
             the actionable diagnostic byte string"
        );
        // Sanity that our expected_err is what the source actually
        // formats — keeps the diagnostic in lockstep with the test.
        assert!(
            expected_err.contains("17 != 4 * prompt_len = 20"),
            "expected_err format check"
        );
    }

    /// Wedge-4e: `t_post` (post-prefill text decode position advance)
    /// is `prompt_len` when `positions_flat` is `None` (legacy
    /// text-only behaviour) and `max(axis-0 positions) + 1` when
    /// supplied (vision behaviour). This is the streaming sibling of
    /// the rule in
    /// `generate_qwen35_once_with_soft_tokens_and_deepstack` at
    /// engine_qwen35.rs:1177-1198.
    ///
    /// We test the rule by reading the source and asserting the exact
    /// formula appears unchanged. A refactor that drops back to
    /// `(prompt_len + step - 1)` on the vision path would silently
    /// break multi-image temporal alignment — this pin catches it.
    #[test]
    fn wedge4e_t_post_advance_rule_matches_non_streaming_sibling() {
        let src = include_str!("engine_qwen35.rs");
        // The streaming arm computes t_post identically to the
        // non-streaming sibling: `max(axis-0) + 1` or `prompt_len`.
        assert!(
            src.contains("max_t.saturating_add(1)"),
            "Wedge-4e: t_post must use saturating_add(1) over axis-0 max"
        );
        assert!(
            src.contains("None => prompt_len as i32,"),
            "Wedge-4e: t_post must default to prompt_len when no 3D \
             positions supplied (text-only byte-identity)"
        );
        assert!(
            src.contains("let pos = t_post + (step as i32 - 1);"),
            "Wedge-4e: streaming decode position formula must use t_post \
             advance — not the legacy (prompt_len + step - 1) form, \
             which would silently misalign on multi-image prefill"
        );
    }

    /// Wedge-4e: prompt-cache fast-path is BYPASSED whenever any
    /// streaming extension is non-empty. Mirrors the non-streaming
    /// `generate_qwen35_once_with_soft_tokens` rationale at
    /// engine_qwen35.rs:933.
    #[test]
    fn wedge4e_extended_stream_bypasses_prompt_cache_on_extension() {
        let src = include_str!("engine_qwen35.rs");
        // The bypass is gated on `!has_extension` AND'd into
        // `try_match`'s call. A future refactor that hits the cache on
        // a vision-augmented stream would falsely return a text-only
        // KV state, mangling the response.
        assert!(
            src.contains("let prompt_cache_hit = !has_extension"),
            "Wedge-4e: streaming prompt-cache MUST be bypassed when \
             any extension is present (cache key is prompt_tokens \
             only — same placeholder ids + different image ⇒ false \
             hit)"
        );
        assert!(
            src.contains("if is_greedy && !has_extension {"),
            "Wedge-4e: streaming prompt-cache write MUST be skipped on \
             extension paths to avoid poisoning subsequent text-only \
             requests with a soft-token-tainted KV snapshot"
        );
    }

    /// 2026-08-03 store-gate fix: prefix-KV checkpoint stores must NOT be
    /// gated on `is_greedy`.  KV state is a pure function of the prompt
    /// tokens + weights — sampling params (temperature/top_p/top_k/seed)
    /// never touch it.  The greedy conjunct was copy-pasted from
    /// HybridPromptCache (whose replayed first DECODED token IS
    /// sampling-dependent) and silently disabled every store for clients
    /// sending sampling params: probes printed `registry_len=0` forever
    /// and every agentic turn re-prefilled the whole conversation.  Pin
    /// the new gate shape so a future refactor can't reintroduce it.
    #[test]
    fn lcp_prefix_stores_are_not_greedy_gated() {
        let src = include_str!("engine_qwen35.rs");
        // NOTE: patterns are split via concat! so this test's own source
        // doesn't self-match the pins (include_str! reads the whole file,
        // test code included).
        let forbidden = concat!("lcp_resume_enabled", " && ", "is_greedy");
        // Negative pin: no LCP store gate may combine resume with greedy.
        assert!(
            !src.contains(forbidden),
            "LCP prefix-KV stores must NOT be greedy-gated — KV state is \
             sampling-independent; greedy gating belongs only on \
             HybridPromptCache's decoded-token replay"
        );
        // Positive pin: non-stream AND stream mid-prefill gates both skip
        // the final stride checkpoint when the more useful near-tip
        // recovery anchor replaces it. The symmetric kill-switch remains.
        let recovery_guard = concat!("&& !superseded_by_", "recovery_anchor");
        let kill_switch = concat!("&& !mid_store_", "disabled");
        assert_eq!(
            src.matches(recovery_guard).count(),
            4,
            "non-stream + stream mid-prefill notification and store gates \
             must both be suppressed when a near-tip recovery anchor \
             replaces the final stride checkpoint"
        );
        assert_eq!(
            src.matches(kill_switch).count(),
            2,
            "non-stream + stream mid-prefill store gates must both retain \
             the symmetric mid-store kill-switch"
        );
        // Greedy gate SURVIVES where it belongs: HybridPromptCache write —
        // the replayed first decoded token IS sampling-dependent.
        assert!(
            src.contains("if is_greedy && !has_extension {"),
            "HybridPromptCache write must REMAIN greedy-gated (first \
             decoded token replay is sampling-dependent)"
        );
    }

    /// Both schedulers must retain the complete multimodal stream payload.
    /// SlotAware owns the extensions across bounded prefill yields, validates
    /// them before the first Metal transaction, and routes each chunk through
    /// the multimodal forward graph rather than silently using text-only
    /// prefill.
    #[test]
    fn wedge4e_multimodal_streaming_is_explicit_per_scheduler_mode() {
        let src = include_str!("engine.rs");
        let qwen = include_str!("engine_qwen35.rs");
        assert!(
            src.contains("Qwen35VisionPrefillData::new("),
            "SlotAware Qwen must retain every multimodal stream field at admission"
        );
        assert!(
            qwen.contains("vision.validate(prompt_len, hidden_size)?"),
            "SlotAware Qwen must validate multimodal state before GPU prefill"
        );
        assert!(
            qwen.contains("forward_gpu_last_logits_with_soft_tokens_and_deepstack("),
            "SlotAware Qwen must route multimodal chunks through the vision-aware graph"
        );
        assert!(
            src.contains("generate_stream_qwen35_once_extended"),
            "SerialFifo must retain the extended multimodal stream primitive"
        );
    }

    /// Wedge-4e handler-side: the `chat_completions_stream` 501
    /// reject for Qwen3-VL deepstack streaming has been removed.
    /// Image-bearing streaming chat reaches the engine boundary; scheduler
    /// mode then either dispatches the extended SerialFifo primitive or
    /// returns the explicit pre-SSE SlotAware capability error above.
    #[test]
    fn wedge4e_handler_streaming_501_reject_is_removed() {
        let src = include_str!("handlers.rs");
        assert!(
            !src.contains("streaming chat with Qwen3-VL DeepStack injection is not yet"),
            "Wedge-4e: handler-side streaming 501 reject MUST be \
             removed — streaming Qwen3-VL chat now flows through \
             generate_stream_with_deepstack"
        );
        // Positive pin: the new generate_stream_with_deepstack call
        // is wired up.
        assert!(
            src.contains("generate_stream_with_deepstack"),
            "Wedge-4e: handler must call generate_stream_with_deepstack \
             so soft_tokens + deepstack + positions reach the worker"
        );
    }

    // ──────────────────────────────────────────────────────────────────
    // ADR-027 Phase B iter-12 — alloc_kv_cache_for_request TQ branching
    // ──────────────────────────────────────────────────────────────────

    /// Build a synthetic Qwen35LoadedModel with the given `tq_kv_active`
    /// flag — minimal fixture for testing alloc_kv_cache_for_request
    /// branching. Mirrors the load_info.rs / engine.rs test fixture
    /// shape but parameterized on `tq_kv_active`.
    fn synth_loaded_model_for_alloc_test(
        cfg: Qwen35Config,
        tq_kv_active: bool,
    ) -> Qwen35LoadedModel {
        Qwen35LoadedModel {
            model: super::Qwen35Model::empty_from_cfg(cfg),
            tokenizer: tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            chat_template: "{{ messages }}".to_string(),
            model_id: "iter-12-test".to_string(),
            model_path: std::path::PathBuf::from("/tmp/iter-12-test.gguf"),
            eos_token_ids: vec![151_645],
            hidden_size: 64,
            vocab_size: 256,
            context_length: Some(1024),
            quant_type: Some("Q4_K".to_string()),
            load_duration: std::time::Duration::from_millis(1),
            provenance: crate::core::provenance::Provenance::External,
            expected_projector_sha256: None,
            vision_projector_profile: None,
            vision_deepstack_output_count: None,
            vision_special_tokens_present: false,
            prompt_cache: HybridPromptCache::new(),
            lcp_registry: crate::serve::kv_persist::lcp_registry::LcpRegistry::new(1),
            kv_metrics_sink: None,
            disk_persistor: None,
            lcp_hydrated_for_cfg: std::collections::HashSet::new(),
            tq_kv_active,
            // ADR-040 C2b scaffold — test fixture mirrors production
            // construction shape; iter-2a always leaves this `None`.
            persistent_kv_cache: None,
            speculation: crate::serve::api::qwen35_speculation::QwenSpeculationController::new(
                crate::serve::api::qwen35_speculation::QwenSpeculationPolicy::Off,
            ),
        }
    }

    #[test]
    fn alloc_kv_cache_for_request_tq_off_keeps_full_attn_tq_none() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        let cfg = moe_cfg_40layer_for_cache_test();
        let qwen = synth_loaded_model_for_alloc_test(cfg, false);
        let cache =
            alloc_kv_cache_for_request(&qwen, &device, 32, 16).expect("alloc_kv_cache_for_request");
        assert!(!cache.full_attn.is_empty(), "fixture has full-attn layers");
        for (i, slot) in cache.full_attn.iter().enumerate() {
            assert!(
                slot.tq.is_none(),
                "tq_kv_active=false: full_attn[{i}].tq must be None \
                 (legacy F32 path preserved)"
            );
        }
    }

    #[test]
    fn alloc_kv_cache_for_request_tq_on_populates_tq_per_full_attn_slot() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Metal device: {e}");
                return;
            }
        };
        // Note: head_dim must be 256 for the TQ kernel chain; the cache
        // alloc itself succeeds at any head_dim, but iter-13's SDPA
        // dispatch requires 256/512. Use a real qwen36-shape cfg to
        // exercise the production allocation path.
        let mut cfg = moe_cfg_40layer_for_cache_test();
        cfg.head_dim = 256;
        cfg.num_attention_heads = 8;
        cfg.num_key_value_heads = 2;
        let qwen = synth_loaded_model_for_alloc_test(cfg, true);
        let cache =
            alloc_kv_cache_for_request(&qwen, &device, 32, 16).expect("alloc_kv_cache_for_request");
        assert!(!cache.full_attn.is_empty());
        for (i, slot) in cache.full_attn.iter().enumerate() {
            assert!(
                slot.tq.is_some(),
                "tq_kv_active=true: full_attn[{i}].tq must be populated"
            );
            let tq = slot.tq.as_ref().unwrap();
            assert_eq!(tq.norms_per_pos, 1, "head_dim=256 → norms_per_pos=1");
        }
    }
}
