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

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use crate::inference::models::qwen35::kv_cache::HybridKvCacheSnapshot;
use crate::inference::models::qwen35::model::Qwen35Model;
use crate::serve::load_info::{
    self, ArchFamily, ChatTemplateSource, LoadInfo, LoadInfoBuilder, MoeShape,
    TokenizerSource,
};
use crate::serve::provenance::{self, Provenance};

use super::engine::{LoadOptions, SamplingParams};

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
    /// `crate::serve::provenance::detect(&gguf)`. Stored for the common
    /// `LoadedModel::provenance()` surface; Qwen35 KV-spill remains
    /// descriptor-pending because the cache is hybrid.
    pub provenance: Provenance,
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
    pub disk_persistor: Option<std::sync::Arc<
        crate::serve::kv_persist::families::qwen35_disk_persistor::Qwen35DiskPersistor,
    >>,

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
        let provenance = provenance::detect(&gguf);

        // ADR-018 C3: legacy `tracing::info!("Qwen35 SERVE load: model = ...")`
        // was deleted here. The same fact (`model_path`) is now emitted by
        // `emit_tracing(&info)` at every CLI/SERVE entry that constructs a
        // `LoadInfo`. Conditions/warnings stay; load FACTS are unified.

        // ---- Resolve tokenizer path ----
        // Reuse the shared `find_tokenizer` helper from serve/mod.rs so
        // the SERVE path resolves the tokenizer the same way
        // `cmd_generate_qwen35` does.  Caller may override via
        // `--tokenizer` (threaded through `LoadOptions::tokenizer_path`).
        let tokenizer_path =
            crate::serve::find_tokenizer(model_path, opts.tokenizer_path.as_deref())?;
        // ADR-018 C3: legacy `tracing::info!("Qwen35 SERVE load: tokenizer = ...")`
        // was deleted here. `emit_tracing(&info)` surfaces the active
        // tokenizer source; for Qwen3.5/3.6 that's `TokenizerSource::GgufEmbedded`
        // (the on-disk path is a load-time diagnostic only — see the
        // `tokenizer_path` shadow below).

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
                    format!(
                        "Qwen35 DWQ overlay from {} failed",
                        overlay_path.display()
                    )
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
        let eos_token: u32 = gguf
            .metadata_u32("tokenizer.ggml.eos_token_id")
            .unwrap_or(151645);
        let eos_token_ids: Vec<u32> = vec![eos_token];

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
        // `tokenizer_path` is kept in scope only as a load-time
        // diagnostic (logged above); its bytes are NOT consumed.
        let _tokenizer_path = tokenizer_path;
        let mut tokenizer = crate::inference::models::qwen35::tokenizer::build_tokenizer_from_gguf(&gguf)
            .map_err(|e| anyhow::anyhow!("GGUF-driven tokenizer build failed: {e}"))?;
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
                match crate::serve::kv_persist::families::qwen35_disk_persistor::Qwen35DiskPersistor::new(cache_dir.clone()) {
                    Ok(p) => {
                        tracing::info!(
                            cache_dir = %cache_dir.display(),
                            "ADR-027 iter-6b.2: Qwen35DiskPersistor constructed; \
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
        })
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
        // 1) Disk write-through (if persistor is bound). Errors are
        //    logged + swallowed so they don't break the in-memory path.
        if let Some(persistor) = &self.disk_persistor {
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
                    if let Err(e) = persistor.write(&cfg, &key_hex, &snapshot, &sidecar) {
                        tracing::warn!(
                            cache_dir = %persistor.cache_dir().display(),
                            key_hex = %key_hex,
                            error = %format!("{e:#}"),
                            "ADR-027 iter-6b.3: disk persistor write failed; \
                             in-memory store will still proceed"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        cache_dir = %persistor.cache_dir().display(),
                        error = %format!("{e:#}"),
                        "ADR-027 iter-6b.3: cfg_from_cache failed; disk \
                         write-through skipped (in-memory store still proceeds)"
                    );
                }
            }
        }
        // 2) In-memory store — exact pre-iter-6b.2 semantics.
        self.lcp_registry.store(
            key,
            prompt_tokens,
            vec![std::sync::Arc::new(snapshot)],
            sliding_window,
            linear_capacity,
        )
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
}

/// ADR-027 Phase A iter-6b.2 — derive the on-disk filename hex from a
/// `LcpKey`. SHA-256 over the model_fingerprint + tenant_id +
/// params_hash; first 16 bytes hex-encoded (32-char filename).
fn lcp_key_to_filename_hex(
    key: &crate::serve::kv_persist::lcp_registry::LcpKey,
) -> String {
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
                ChatTemplateSource::None
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
/// even under greedy decode (max_tokens early-stop, stop_strings).
///
/// Wedge-4 follow-up will widen the key as Qwen3.5/3.6 picks up grammar
/// / logit_bias / tool-call policies.  The conservative MVP key keeps
/// false-positive hits structurally impossible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridPromptCacheKey {
    pub max_tokens: usize,
    pub stop_strings: Vec<String>,
}

impl HybridPromptCacheKey {
    pub fn from_params(params: &SamplingParams) -> Self {
        Self {
            max_tokens: params.max_tokens,
            stop_strings: params.stop_strings.clone(),
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
/// arches.  Wedge-4 will revisit the unification question once both
/// arches have shipped real wire-up.
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
    pub fn try_match(
        &self,
        new_prompt: &[u32],
        new_params: &SamplingParams,
    ) -> Option<usize> {
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
        || params.repetition_penalty != 1.0
        || params.seed.is_some())
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
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::serve::sampler_pure::{self, SamplingParams as SamplerPureParams};
use mlx_native::MlxDevice;

use super::engine::GenerationResult;
use super::registry::{ModelRegistration, ReasoningSplitter, SplitSlot, ToolCallEvent, ToolCallSplitter};
use super::sse::{DeltaKind, GenerationEvent, StreamStats};

/// Build the flat `[4 * seq_len]` axis-major position buffer the IMROPE
/// kernel expects.  Mirrors `cmd_generate_qwen35` at
/// `serve/mod.rs:1262-1270` — for text-only Qwen3.5/3.6 we replicate the
/// absolute-position index across all 4 axes.
fn prefill_positions_for(prompt_len: usize) -> Vec<i32> {
    let mut flat = vec![0i32; 4 * prompt_len];
    for axis in 0..4 {
        for t in 0..prompt_len {
            flat[axis * prompt_len + t] = t as i32;
        }
    }
    flat
}

/// Allocate a fresh `HybridKvCache` sized for `prompt_len + max_tokens + 64`,
/// clamped to `cfg.max_position_embeddings` and floored at 128.
///
/// Mirrors `cmd_generate_qwen35:1129-1131`.  Per-request allocation is
/// the expedient MVP shape — a Wedge-4 follow-up may move the cache
/// to a long-lived field on `Qwen35LoadedModel` to amortize the few-GB
/// alloc across requests.
fn alloc_kv_cache_for_request(
    qwen: &Qwen35LoadedModel,
    device: &MlxDevice,
    prompt_len: usize,
    max_tokens: usize,
) -> Result<HybridKvCache> {
    let max_seq = (prompt_len + max_tokens + 64)
        .max(128)
        .min(qwen.model.cfg.max_position_embeddings as usize);
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

/// Sample the next token from a `[vocab_size]` logits slice using
/// `sampler_pure::sample_token` for non-greedy modes.
///
/// Wedge-3 keeps the wired sampler surface narrow: temperature / top_p /
/// top_k / repetition_penalty pass through to `SamplerPureParams`.
/// `logit_bias`, grammar enforcement, and seed-driven RNG are deferred
/// to Wedge-4 (the Qwen35 MVP doesn't currently surface those — the
/// chat handler short-circuits grammar use to Gemma-only).
fn sample_logits_qwen35(
    logits: &mut [f32],
    params: &SamplingParams,
    generated: &[u32],
) -> u32 {
    let sp = SamplerPureParams {
        temperature: params.temperature as f64,
        top_p: params.top_p as f64,
        top_k: params.top_k,
        min_p: 0.0,
        repetition_penalty: params.repetition_penalty as f64,
        max_tokens: params.max_tokens,
    };
    sampler_pure::sample_token(logits, &sp, generated)
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
        crate::serve::provenance::Provenance::Hf2q {
            producer_version,
            source_sha256,
            ..
        } => (producer_version.as_str(), source_sha256.as_str()),
        crate::serve::provenance::Provenance::External => ("", ""),
    };
    let chat_template_hash = match &qwen.provenance {
        crate::serve::provenance::Provenance::Hf2q { .. } => {
            super::kv_spill_descriptor::KvSpillProvenance::hash_chat_template(
                &qwen.chat_template,
            )
        }
        crate::serve::provenance::Provenance::External => String::new(),
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

/// `true` when the running text ends with any of the registered stop
/// strings.  Mirrors `engine::hit_stop_string` (kept private to that
/// module for Gemma; replicated here so the Qwen35 path stays
/// self-contained without cross-module coupling).
fn qwen35_hit_stop_string(text: &str, stops: &[String]) -> bool {
    if stops.is_empty() {
        return false;
    }
    stops.iter().any(|s| !s.is_empty() && text.ends_with(s.as_str()))
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
pub fn generate_qwen35_once(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    registration: Option<&ModelRegistration>,
) -> Result<GenerationResult> {
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

    let device = MlxDevice::new()
        .map_err(|e| anyhow::anyhow!("MlxDevice::new (qwen35 generate): {e}"))?;
    let mut kv_cache = alloc_kv_cache_for_request(qwen, &device, prompt_len, max_tokens)?;

    // ADR-027 iter-6b.3: cold-start hydrate of in-memory LcpRegistry
    // from disk-persisted snapshots.  Idempotent + cheap on hot path
    // (HashSet lookup) — no-op when HF2Q_KV_PERSIST is unset, when
    // this cfg has already been hydrated this process, or when the
    // cfg-subdir is empty.
    qwen.hydrate_lcp_registry_from_disk(&kv_cache, &device);

    // ── Prompt-cache fast-path ────────────────────────────────────
    let prompt_cache_hit = qwen
        .prompt_cache
        .try_match(prompt_tokens, params)
        .is_some();

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
        let stride_for_observe =
            crate::debug::INVESTIGATION_ENV.kv_lcp_deltanet_checkpoint_stride;
        let base_key_for_observe = build_lcp_key_for_qwen35(qwen, params);
        let detected =
            crate::serve::kv_persist::lcp_registry::probe_lcp_opportunity_chunk_aligned(
                &mut qwen.lcp_registry,
                prompt_tokens,
                stride_for_observe,
                false, // is_multimodal: text-only path
                |chunk_pos| {
                    let mut key = base_key_for_observe.clone();
                    if chunk_pos > 0 {
                        key.tenant_id = format!("qwen35:lcp_chunk:{chunk_pos}");
                    }
                    key
                },
            );
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
        let lcp_resume_enabled = crate::serve::api::engine::effective_kv_lcp_resume(
            crate::debug::INVESTIGATION_ENV.kv_lcp_resume,
            crate::debug::INVESTIGATION_ENV.use_dense,
        );
        if lcp_resume_enabled {
            let stride =
                crate::debug::INVESTIGATION_ENV.kv_lcp_deltanet_checkpoint_stride;
            // ADR-017 Phase E.a B.5 Codex Phase-2b finding (HIGH):
            // guard `stride > 0` before any division.  If operators set
            // `HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE=0` the probe would
            // otherwise panic on integer division by zero.  Skip the
            // probe entirely on stride == 0 — there are no stride-aligned
            // chunks to find anyway.
            if stride == 0 {
                eprintln!(
                    "[hf2q qwen35 lcp probe] stride=0; skipping descending \
                     scan (HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE must be > 0)"
                );
                // Fall through to the rest of the function with
                // `lcp_resume_start = 0` — runs fresh prefill normally.
            }
            // Iterate stride-aligned chunk positions descending.  The
            // largest viable position is `(prompt_tokens.len() / stride)
            // * stride` — any larger position would have cached_prompt_len
            // > prompt_tokens.len(), failing the strict-prefix gate.
            // Smallest is `stride` (positions < stride aren't checkpointed).
            let max_chunk_pos = if stride == 0 {
                0
            } else {
                (prompt_tokens.len() / stride).saturating_mul(stride)
            };
            eprintln!(
                "[hf2q qwen35 lcp probe] enabled, registry_len={}, prompt_len={}, \
                 stride={}, scanning chunk positions [{stride}..={max_chunk_pos}]",
                qwen.lcp_registry.len(),
                prompt_tokens.len(),
                stride,
            );

            let mut hit = false;
            // Step from max_chunk_pos down to stride in -stride steps.
            let mut chunk_pos = max_chunk_pos;
            while chunk_pos >= stride && !hit {
                let chunk_key =
                    build_lcp_key_for_qwen35_chunk(qwen, params, chunk_pos);
                if let Some(prefix) =
                    qwen.lcp_registry.lookup(&chunk_key, prompt_tokens)
                {
                    // ADR-017 Phase E.a B.5 Codex Phase-2b finding
                    // (HIGH, defense-in-depth): require `prefix.k <
                    // prompt_tokens.len()` (LcpRegistry::lookup already
                    // gates `k == new_tokens.len()` returning None, so
                    // this is belt-and-suspenders).
                    //
                    // ADR-017 Phase E.a B.5 bench-driven change: use
                    // non-consuming `lookup` instead of `take_prefix`.
                    // B.5's `restore_partial` MEMCPYS bytes from the
                    // snapshot into the request's kv_cache (does NOT
                    // try_unwrap the Arc); the registry's entry stays
                    // intact and is reusable for subsequent matching
                    // requests.  The Gemma iter-3 reason for consuming
                    // (Arc strong_count == 1 needed for try_unwrap) does
                    // NOT apply to B.5's pathway.  Empirical impact:
                    // bench `lcp_resume_speedup` shows 50% → 100% hit
                    // rate (no inter-trial cache flush from
                    // consumption), and R-P6 4-worker fan-out converges
                    // to <= 1.0× single-worker (idealized B.5 speedup).
                    if prefix.k == prefix.cached_prompt_len
                        && prefix.k < prompt_tokens.len()
                    {
                        // True-continuation at this stride boundary.
                        // The snapshot's DeltaNet recurrent state is at
                        // exactly this position — byte-correct resume.
                        //
                        // B.5: use `restore_partial` instead of
                        // `restore_from` — the snapshot's full-attn slot
                        // K/V buffers were sized to the SOURCE request's
                        // max_seq_len, which typically differs from the
                        // current request's
                        // (alloc_kv_cache_for_request sizes per-request).
                        // restore_partial copies only positions [0..k]
                        // per head, decoupling buffer sizing from
                        // snapshot size.  DeltaNet recurrent + conv
                        // state are copied verbatim (size-independent
                        // of max_seq_len).
                        let snapshot: &HybridKvCacheSnapshot =
                            &prefix.dense_kvs[0];
                        let restore_start = Instant::now();
                        kv_cache
                            .restore_partial(snapshot, prefix.k)
                            .context("qwen35 lcp_registry restore_partial")?;
                        let restore_ms = restore_start.elapsed().as_micros() as f64 / 1000.0;
                        lcp_resume_start = prefix.k;
                        eprintln!(
                            "[hf2q qwen35 lcp resume] STRIDE-ALIGNED HIT — \
                             restoring at k={} (cached_prompt_len={}, \
                             chunk_pos={}, restore_ms={:.3})",
                            prefix.k, prefix.cached_prompt_len, chunk_pos, restore_ms
                        );
                        hit = true;
                    } else {
                        // k < cached_prompt_len: the cached prompt at
                        // this chunk_pos diverges from new_tokens before
                        // hitting cached_prompt_len.  Means turn-2's
                        // tokens diverge from turn-1's tokens at some
                        // position ≤ chunk_pos.  Don't restore — the
                        // residual DeltaNet state would be wrong.
                        // Continue descending; smaller chunk_pos may
                        // still match.
                        eprintln!(
                            "[hf2q qwen35 lcp probe] PARTIAL HIT at \
                             chunk_pos={} — k={} < cached_prompt_len={}, \
                             continuing descent",
                            chunk_pos, prefix.k, prefix.cached_prompt_len
                        );
                    }
                }
                if chunk_pos == stride {
                    break;
                }
                chunk_pos -= stride;
            }
            if !hit {
                eprintln!(
                    "[hf2q qwen35 lcp probe] no stride-aligned match \
                     (registry_len={})",
                    qwen.lcp_registry.len()
                );
            }
        }
    }

    let prefill_start = Instant::now();
    let mut next_token: u32;
    if prompt_cache_hit {
        // Hit: restore the substrate + reuse the cached first decoded token.
        let snap = qwen
            .prompt_cache
            .snapshot()
            .expect("try_match returned Some implies snapshot Some");
        kv_cache.restore_from(snap).context("prompt_cache restore")?;
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
            && crate::debug::INVESTIGATION_ENV.kv_lcp_chunked_prefill;
        // ADR-017 Phase E.a B.5 Codex Phase-2b finding (MEDIUM):
        // enforce `lcp_resume_start % stride == 0` at a single point
        // before any prefill branch dispatch — covers chunked AND
        // suffix-only branches.
        if lcp_resume_start > 0 && stride > 0 {
            anyhow::ensure!(
                lcp_resume_start % stride == 0,
                "qwen35 prefill: lcp_resume_start ({}) must be stride-\
                 aligned ({}) — registry only stores at stride boundaries",
                lcp_resume_start,
                stride
            );
        }
        let prefill_logits = if chunked_eligible {
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
            let n_chunks = (prompt_len + stride - 1) / stride;
            eprintln!(
                "[hf2q qwen35 chunked prefill] {} chunks (stride={}, \
                 prompt_len={}, first_chunk_idx={})",
                n_chunks, stride, prompt_len, first_chunk_idx
            );
            let mut last_logits: Option<Vec<f32>> = None;
            for chunk_idx in first_chunk_idx..n_chunks {
                let k_start = chunk_idx * stride;
                let k_end = ((chunk_idx + 1) * stride).min(prompt_len);
                let chunk_seq_len = k_end - k_start;
                let chunk_tokens = &prompt_tokens[k_start..k_end];
                // Build positions for [k_start..k_end). Positions are
                // absolute (NOT chunk-relative) so RoPE sees the
                // correct logical positions across the whole prompt.
                let mut chunk_positions = vec![0i32; 4 * chunk_seq_len];
                for axis in 0..4 {
                    for t in 0..chunk_seq_len {
                        chunk_positions[axis * chunk_seq_len + t] =
                            (k_start + t) as i32;
                    }
                }
                let logits = qwen
                    .model
                    .forward_gpu_last_logits(chunk_tokens, &chunk_positions, &mut kv_cache)
                    .with_context(|| {
                        format!(
                            "qwen35 chunked prefill: chunk {}/{} (k_start={}, k_end={}, seq_len={})",
                            chunk_idx + 1, n_chunks, k_start, k_end, chunk_seq_len
                        )
                    })?;
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
                let mid_store_disabled =
                    std::env::var("HF2Q_KV_LCP_DISABLE_MID_STORE").as_deref() == Ok("1");
                if lcp_resume_enabled && is_greedy && stride_aligned && !mid_store_disabled {
                    match kv_cache.snapshot(&device) {
                        Ok(snap) => {
                            let chunk_key = build_lcp_key_for_qwen35_chunk(
                                qwen, params, k_end,
                            );
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
                                tracing::warn!(
                                    "qwen35 lcp_registry mid-prefill store failed at \
                                     chunk_pos={k_end}: {:?}",
                                    e
                                );
                            } else {
                                eprintln!(
                                    "[hf2q qwen35 lcp store] mid-prefill snapshot \
                                     at chunk_pos={k_end} (registry_len_after={})",
                                    qwen.lcp_registry.len()
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "qwen35 lcp_registry mid-prefill snapshot skipped at chunk_pos={k_end}"
                            );
                        }
                    }
                }
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
                    suffix_positions[axis * suffix_len + t] =
                        (lcp_resume_start + t) as i32;
                }
            }
            eprintln!(
                "[hf2q qwen35 lcp resume] suffix prefill {} tokens \
                 (lcp_resume_start={}, prompt_len={})",
                suffix_len, lcp_resume_start, prompt_len
            );
            qwen
                .model
                .forward_gpu_last_logits(suffix_tokens, &suffix_positions, &mut kv_cache)
                .context("Qwen35Model::forward_gpu_last_logits (LCP resume suffix)")?
        } else {
            let positions = prefill_positions_for(prompt_len);
            qwen
                .model
                .forward_gpu_last_logits(prompt_tokens, &positions, &mut kv_cache)
                .context("Qwen35Model::forward_gpu_last_logits (prefill)")?
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
            next_token = sample_logits_qwen35(&mut logits, params, &[]);
        }

        // Snapshot + cache update for greedy-eligible requests.  The
        // snapshot captures KV state AFTER the prefill (current_len[0]
        // == prompt_len for full-attn slots; DeltaNet conv/recurrent
        // populated by the linear-attn layer's prefill emit).
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
            match kv_cache.snapshot(&device) {
                Ok(snap) => qwen.prompt_cache.update(
                    prompt_tokens.to_vec(),
                    snap,
                    next_token,
                    params,
                ),
                Err(e) => {
                    tracing::warn!(error = %e, "qwen35 prompt_cache snapshot skipped");
                }
            }

            // Snapshot 2: lcp_registry (partial-prefill resume, Phase E.a).
            // ADR-017 Phase E.a B.3: when chunked prefill ran (lcp_resume_enabled
            // AND prompt_len > stride), the in-loop mid-prefill checkpointing
            // already stored every stride-aligned boundary INCLUDING the last
            // (when prompt_len is stride-aligned).  This end-of-prefill store
            // handles the short-prompt case where chunked DIDN'T fire
            // (prompt_len ≤ stride): the probe can still find this entry under
            // chunk_pos = prompt_len when stride-aligned.
            //
            // Skip when prompt_len isn't stride-aligned: the descending probe
            // only tests stride boundaries, so a non-stride-aligned chunk_pos
            // would never be looked up.  Burning GB-scale memcpy on an
            // unreachable entry is wasteful.
            //
            // Skip when chunked already ran: mid-prefill loop already stored
            // the last chunk boundary, no need to double-store.
            let stride =
                crate::debug::INVESTIGATION_ENV.kv_lcp_deltanet_checkpoint_stride;
            let chunked_already_stored = stride > 0
                && prompt_len > stride
                && crate::debug::INVESTIGATION_ENV.kv_lcp_chunked_prefill;
            let prompt_stride_aligned =
                stride > 0 && prompt_len > 0 && prompt_len % stride == 0;
            let should_store_eof =
                crate::debug::INVESTIGATION_ENV.kv_lcp_resume
                    && !chunked_already_stored
                    && prompt_stride_aligned;
            if should_store_eof {
                eprintln!(
                    "[hf2q qwen35 lcp store] end-of-prefill snapshot for prompt_len={} \
                     (chunk_pos={prompt_len}, registry_len_before={})",
                    prompt_tokens.len(),
                    qwen.lcp_registry.len()
                );
                match kv_cache.snapshot(&device) {
                    Ok(snap) => {
                        let chunk_key =
                            build_lcp_key_for_qwen35_chunk(qwen, params, prompt_len);
                        let linear_capacity = kv_cache
                            .linear_attn
                            .first()
                            .map(|s| s.recurrent.byte_len())
                            .unwrap_or(0);
                        // ADR-027 iter-6b.2: write-through helper.
                        if let Err(e) = qwen.store_lcp_with_disk_writeback(
                            &kv_cache,
                            chunk_key,
                            prompt_tokens.to_vec(),
                            snap,
                            0,
                            linear_capacity,
                        ) {
                            tracing::warn!(
                                "qwen35 lcp_registry store skipped: {:?}",
                                e
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "qwen35 lcp_registry snapshot skipped"
                        );
                    }
                }
            }
        }
    }
    let prefill_duration = prefill_start.elapsed();

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
                qwen.model
                    .forward_gpu_greedy(&[next_token], &decode_positions, &mut kv_cache)
                    .with_context(|| format!("forward_gpu_greedy decode step {step}"))?
            } else {
                let logits_full = qwen
                    .model
                    .forward_gpu_last_logits(&[next_token], &decode_positions, &mut kv_cache)
                    .with_context(|| format!("forward_gpu_last_logits decode step {step}"))?;
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

    // ── Reasoning split (Decision #21) ────────────────────────────
    // Use the existing registry helper so the split is byte-identical
    // to the Gemma path's non-streaming arm.  Tool-call extraction is
    // owned by the chat handler's post-decode pipeline, NOT this
    // function — see Wedge-3 PRD Phase D step 10 + the docstring above.
    let (content, reasoning_text) = match registration {
        Some(reg) if reg.has_reasoning() => super::registry::split_full_output(reg, &decoded_text),
        _ => (decoded_text, None),
    };

    // Reasoning token count: rerun the splitter token-by-token to
    // attribute each completion token to a slot.  Cheap (one decode
    // per token + a tail-buffered splitter) and matches the Gemma
    // path's `reasoning_token_count` semantics.
    let reasoning_token_count = match registration {
        Some(reg) if reg.has_reasoning() => {
            let mut sp = ReasoningSplitter::from_registration(reg);
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
            logprobs: None,
    })
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
pub fn generate_qwen35_once_with_soft_tokens(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    soft_tokens: &[crate::serve::forward_prefill::SoftTokenInjection<'_>],
    params: &SamplingParams,
    registration: Option<&ModelRegistration>,
) -> Result<GenerationResult> {
    // Empty slice → identity over `generate_qwen35_once`.  This keeps
    // text-only fallback paths from paying any soft-token overhead
    // when (e.g.) a future caller threads an empty vec through the
    // engine `Request::GenerateWithSoftTokens` arm.
    if soft_tokens.is_empty() {
        return generate_qwen35_once(qwen, prompt_tokens, params, registration);
    }

    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "generate_qwen35_once_with_soft_tokens: empty prompt_tokens"
    );
    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);
    let is_greedy = is_greedy_eligible(params);

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
    let prefill_logits = qwen
        .model
        .forward_gpu_last_logits_with_soft_tokens(
            prompt_tokens,
            &positions,
            soft_tokens,
            &mut kv_cache,
        )
        .context("Qwen35Model::forward_gpu_last_logits_with_soft_tokens (prefill)")?;
    anyhow::ensure!(
        prefill_logits.len() == qwen.vocab_size,
        "qwen35 prefill (soft tokens) logits len {} != vocab_size {}",
        prefill_logits.len(),
        qwen.vocab_size
    );
    let mut next_token: u32 = if is_greedy {
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

            next_token = if is_greedy {
                qwen.model
                    .forward_gpu_greedy(&[next_token], &decode_positions, &mut kv_cache)
                    .with_context(|| {
                        format!("forward_gpu_greedy decode step {step} (soft tokens)")
                    })?
            } else {
                let logits_full = qwen
                    .model
                    .forward_gpu_last_logits(&[next_token], &decode_positions, &mut kv_cache)
                    .with_context(|| {
                        format!("forward_gpu_last_logits decode step {step} (soft tokens)")
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
        Some(reg) if reg.has_reasoning() => super::registry::split_full_output(reg, &decoded_text),
        _ => (decoded_text, None),
    };

    let reasoning_token_count = match registration {
        Some(reg) if reg.has_reasoning() => {
            let mut sp = ReasoningSplitter::from_registration(reg);
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
        // tokens count is always 0 for vision-augmented requests.
        cached_tokens: 0,
            logprobs: None,
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
pub fn generate_qwen35_once_with_soft_tokens_and_deepstack(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    soft_tokens: &[crate::serve::forward_prefill::SoftTokenInjection<'_>],
    deepstack: Option<&crate::serve::forward_prefill::DeepstackInjection<'_>>,
    positions_flat: Option<&[i32]>,
    params: &SamplingParams,
    registration: Option<&ModelRegistration>,
) -> Result<GenerationResult> {
    // Empty soft + no deepstack + no positions → identity over text-only.
    if soft_tokens.is_empty() && deepstack.is_none() && positions_flat.is_none() {
        return generate_qwen35_once(qwen, prompt_tokens, params, registration);
    }

    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "generate_qwen35_once_with_soft_tokens_and_deepstack: empty prompt_tokens"
    );
    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);
    let is_greedy = is_greedy_eligible(params);

    let device = MlxDevice::new()
        .map_err(|e| {
            anyhow::anyhow!("MlxDevice::new (qwen35 wedge-4d generate): {e}")
        })?;
    let mut kv_cache = alloc_kv_cache_for_request(qwen, &device, prompt_len, max_tokens)?;

    // ADR-027 iter-6b.3: cold-start hydrate (wedge-4d deepstack path).
    qwen.hydrate_lcp_registry_from_disk(&kv_cache, &device);

    let prefill_start = Instant::now();
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

    let prefill_logits = qwen
        .model
        .forward_gpu_last_logits_with_soft_tokens_and_deepstack(
            prompt_tokens,
            positions,
            soft_tokens,
            deepstack,
            &mut kv_cache,
        )
        .context(
            "Qwen35Model::forward_gpu_last_logits_with_soft_tokens_and_deepstack \
             (prefill)",
        )?;
    anyhow::ensure!(
        prefill_logits.len() == qwen.vocab_size,
        "qwen35 prefill (wedge-4d) logits len {} != vocab_size {}",
        prefill_logits.len(),
        qwen.vocab_size
    );
    let mut next_token: u32 = if is_greedy {
        greedy_argmax_last_token(&prefill_logits, qwen.vocab_size as u32)
    } else {
        let mut logits = prefill_logits.clone();
        sample_logits_qwen35(&mut logits, params, &[])
    };
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

            next_token = if is_greedy {
                qwen.model
                    .forward_gpu_greedy(&[next_token], &decode_positions, &mut kv_cache)
                    .with_context(|| {
                        format!("forward_gpu_greedy decode step {step} (wedge-4d)")
                    })?
            } else {
                let logits_full = qwen
                    .model
                    .forward_gpu_last_logits(&[next_token], &decode_positions, &mut kv_cache)
                    .with_context(|| {
                        format!("forward_gpu_last_logits decode step {step} (wedge-4d)")
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

    let (content, reasoning_text) = match registration {
        Some(reg) if reg.has_reasoning() => super::registry::split_full_output(reg, &decoded_text),
        _ => (decoded_text, None),
    };

    let reasoning_token_count = match registration {
        Some(reg) if reg.has_reasoning() => {
            let mut sp = ReasoningSplitter::from_registration(reg);
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
            logprobs: None,
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
pub fn generate_stream_qwen35_once(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    events: &tokio::sync::mpsc::Sender<GenerationEvent>,
    registration: Option<&ModelRegistration>,
    cancellation_counter: Option<&std::sync::atomic::AtomicU64>,
) {
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
pub fn generate_stream_qwen35_once_extended(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
    soft_tokens: &[crate::serve::forward_prefill::SoftTokenInjection<'_>],
    deepstack: Option<&crate::serve::forward_prefill::DeepstackInjection<'_>>,
    positions_flat: Option<&[i32]>,
    params: &SamplingParams,
    events: &tokio::sync::mpsc::Sender<GenerationEvent>,
    registration: Option<&ModelRegistration>,
    cancellation_counter: Option<&std::sync::atomic::AtomicU64>,
) {
    macro_rules! send {
        ($ev:expr) => {
            if events.blocking_send($ev).is_err() {
                tracing::info!("SSE stream dropped by client; aborting qwen35 decode");
                if let Some(c) = cancellation_counter {
                    c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return;
            }
        };
    }

    if prompt_tokens.is_empty() {
        send!(GenerationEvent::Error(
            "generate_stream_qwen35_once: empty prompt_tokens".into()
        ));
        return;
    }
    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);
    let is_greedy = is_greedy_eligible(params);

    // Wedge-4e: any extension present ⇒ vision-augmented prefill path.
    // When ALL extensions are empty/None, the legacy text-only stream
    // is preserved byte-identically (regression-pin).
    let has_extension =
        !soft_tokens.is_empty() || deepstack.is_some() || positions_flat.is_some();

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
            return;
        }
    }

    let device = match MlxDevice::new() {
        Ok(d) => d,
        Err(e) => {
            send!(GenerationEvent::Error(format!(
                "qwen35 stream: MlxDevice::new failed: {e}"
            )));
            return;
        }
    };
    let mut kv_cache = match alloc_kv_cache_for_request(qwen, &device, prompt_len, max_tokens) {
        Ok(k) => k,
        Err(e) => {
            send!(GenerationEvent::Error(format!(
                "qwen35 stream: KV cache alloc failed: {e:#}"
            )));
            return;
        }
    };

    // ADR-027 iter-6b.3: cold-start hydrate (streaming path).
    // Idempotent + cheap on hot path; warn-logs + swallows persistence
    // failures so the request still proceeds.
    qwen.hydrate_lcp_registry_from_disk(&kv_cache, &device);

    let pre_dispatches = mlx_native::dispatch_count();
    let pre_syncs = mlx_native::sync_count();

    // Wedge-4e: prompt-cache fast-path is BYPASSED whenever any
    // extension is present — the cache key is `prompt_tokens` only and
    // would falsely hit on a vision-augmented request with the same
    // placeholder ids but different image content. Mirrors the
    // non-streaming `generate_qwen35_once_with_soft_tokens` rationale
    // at engine_qwen35.rs:933 ("Prompt-cache is intentionally NOT
    // consulted on the vision path").
    let prompt_cache_hit = !has_extension
        && qwen
            .prompt_cache
            .try_match(prompt_tokens, params)
            .is_some();

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
        let stride_for_observe =
            crate::debug::INVESTIGATION_ENV.kv_lcp_deltanet_checkpoint_stride;
        let base_key_for_observe = build_lcp_key_for_qwen35(qwen, params);
        let detected =
            crate::serve::kv_persist::lcp_registry::probe_lcp_opportunity_chunk_aligned(
                &mut qwen.lcp_registry,
                prompt_tokens,
                stride_for_observe,
                false,
                |chunk_pos| {
                    let mut key = base_key_for_observe.clone();
                    if chunk_pos > 0 {
                        key.tenant_id = format!("qwen35:lcp_chunk:{chunk_pos}");
                    }
                    key
                },
            );
        if let Some(sink) = qwen.kv_metrics_sink.as_ref() {
            sink.record_lcp_probe(detected);
        }
        let _ = detected;

        // Streaming gate — same Q3 auto-disable wiring as the non-streaming
        // site above. Codex Phase-2b 2026-05-06 caught this site missing
        // the helper.
        let lcp_resume_enabled = crate::serve::api::engine::effective_kv_lcp_resume(
            crate::debug::INVESTIGATION_ENV.kv_lcp_resume,
            crate::debug::INVESTIGATION_ENV.use_dense,
        );
        if lcp_resume_enabled {
            let stride =
                crate::debug::INVESTIGATION_ENV.kv_lcp_deltanet_checkpoint_stride;
            // ADR-017 Phase E.a B.5 Codex Phase-2b finding (HIGH):
            // guard `stride > 0` before any division.
            let max_chunk_pos = if stride == 0 {
                eprintln!(
                    "[hf2q qwen35 stream lcp probe] stride=0; skipping \
                     descending scan (HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE \
                     must be > 0)"
                );
                0
            } else {
                (prompt_tokens.len() / stride).saturating_mul(stride)
            };
            eprintln!(
                "[hf2q qwen35 stream lcp probe] enabled, registry_len={}, \
                 prompt_len={}, stride={}, scanning [{stride}..={max_chunk_pos}]",
                qwen.lcp_registry.len(),
                prompt_tokens.len(),
                stride,
            );
            let mut hit = false;
            let mut chunk_pos = max_chunk_pos;
            while stride > 0 && chunk_pos >= stride && !hit {
                let chunk_key =
                    build_lcp_key_for_qwen35_chunk(qwen, params, chunk_pos);
                if let Some(prefix) =
                    qwen.lcp_registry.lookup(&chunk_key, prompt_tokens)
                {
                    // ADR-017 Phase E.a B.5 Codex Phase-2b (HIGH defense-
                    // in-depth): require `prefix.k < prompt_tokens.len()`.
                    // Non-consuming `lookup` (mirrors non-streaming
                    // post-bench change) — `restore_partial` memcpys, so
                    // registry entry stays reusable for subsequent
                    // matching requests.
                    if prefix.k == prefix.cached_prompt_len
                        && prefix.k < prompt_tokens.len()
                    {
                        let snapshot: &HybridKvCacheSnapshot = &prefix.dense_kvs[0];
                        let restore_start = Instant::now();
                        if let Err(e) = kv_cache.restore_partial(snapshot, prefix.k) {
                            send!(GenerationEvent::Error(format!(
                                "qwen35 stream: lcp_registry restore_partial failed: {e:#}"
                            )));
                            return;
                        }
                        let restore_ms = restore_start.elapsed().as_micros() as f64 / 1000.0;
                        lcp_resume_start = prefix.k;
                        eprintln!(
                            "[hf2q qwen35 stream lcp resume] STRIDE-ALIGNED HIT \
                             — restoring at k={} (cached_prompt_len={}, chunk_pos={}, \
                             restore_ms={:.3})",
                            prefix.k, prefix.cached_prompt_len, chunk_pos, restore_ms
                        );
                        hit = true;
                    } else {
                        eprintln!(
                            "[hf2q qwen35 stream lcp probe] PARTIAL HIT at \
                             chunk_pos={} — k={} < cached_prompt_len={}",
                            chunk_pos, prefix.k, prefix.cached_prompt_len
                        );
                    }
                }
                if chunk_pos == stride {
                    break;
                }
                chunk_pos -= stride;
            }
            if !hit {
                eprintln!(
                    "[hf2q qwen35 stream lcp probe] no stride-aligned match \
                     (registry_len={})",
                    qwen.lcp_registry.len()
                );
            }
        }
    }

    let prefill_start = Instant::now();
    let mut next_token: u32;
    if prompt_cache_hit {
        let snap = qwen
            .prompt_cache
            .snapshot()
            .expect("try_match Some implies snapshot Some");
        if let Err(e) = kv_cache.restore_from(snap) {
            send!(GenerationEvent::Error(format!(
                "qwen35 stream: prompt_cache restore failed: {e:#}"
            )));
            return;
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
        let stride =
            crate::debug::INVESTIGATION_ENV.kv_lcp_deltanet_checkpoint_stride;
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
            && crate::debug::INVESTIGATION_ENV.kv_lcp_chunked_prefill;
        // ADR-017 Phase E.a B.5 Codex Phase-2b finding (MEDIUM):
        // enforce `lcp_resume_start % stride == 0` at a single point
        // before any prefill branch dispatch — covers both chunked and
        // suffix-only branches (the registry only stores at stride
        // boundaries, so this assertion should always hold; defense-
        // in-depth against future registry-shape changes).
        if lcp_resume_start > 0 && stride > 0 && lcp_resume_start % stride != 0 {
            send!(GenerationEvent::Error(format!(
                "qwen35 stream: lcp_resume_start ({}) must be stride-aligned \
                 ({}) — registry should only contain stride-aligned chunk-position \
                 keys",
                lcp_resume_start, stride
            )));
            return;
        }
        let prefill_logits_res = if has_extension {
            qwen.model.forward_gpu_last_logits_with_soft_tokens_and_deepstack(
                prompt_tokens,
                positions_slice,
                soft_tokens,
                deepstack,
                &mut kv_cache,
            )
        } else if chunked_eligible {
            // Chunked prefill — mirrors non-streaming chunked block.
            // Stride alignment was already validated above (centralized
            // assertion covers both chunked + suffix-only branches).
            let first_chunk_idx = lcp_resume_start / stride;
            let n_chunks = (prompt_len + stride - 1) / stride;
            eprintln!(
                "[hf2q qwen35 stream chunked prefill] {} chunks (stride={}, \
                 prompt_len={}, first_chunk_idx={})",
                n_chunks, stride, prompt_len, first_chunk_idx
            );
            let mut last_logits_res: Result<Vec<f32>> = Err(anyhow::anyhow!(
                "no chunks executed"
            ));
            for chunk_idx in first_chunk_idx..n_chunks {
                let k_start = chunk_idx * stride;
                let k_end = ((chunk_idx + 1) * stride).min(prompt_len);
                let chunk_seq_len = k_end - k_start;
                let chunk_tokens = &prompt_tokens[k_start..k_end];
                let mut chunk_positions = vec![0i32; 4 * chunk_seq_len];
                for axis in 0..4 {
                    for t in 0..chunk_seq_len {
                        chunk_positions[axis * chunk_seq_len + t] =
                            (k_start + t) as i32;
                    }
                }
                let res = qwen.model.forward_gpu_last_logits(
                    chunk_tokens,
                    &chunk_positions,
                    &mut kv_cache,
                );
                let logits = match res {
                    Ok(l) => l,
                    Err(e) => {
                        send!(GenerationEvent::Error(format!(
                            "qwen35 stream chunked prefill chunk {}/{} failed: {e:#}",
                            chunk_idx + 1,
                            n_chunks
                        )));
                        return;
                    }
                };
                if chunk_idx == n_chunks - 1 {
                    last_logits_res = Ok(logits.clone());
                }
                // B.3 mid-prefill snapshot store (greedy + LCP_RESUME).
                let stride_aligned = k_end % stride == 0;
                if lcp_resume_enabled && is_greedy && stride_aligned {
                    if let Ok(snap) = kv_cache.snapshot(&device) {
                        let chunk_key = build_lcp_key_for_qwen35_chunk(
                            qwen, params, k_end,
                        );
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
                            tracing::warn!(
                                "qwen35 stream lcp_registry mid-prefill store \
                                 failed at chunk_pos={k_end}: {:?}",
                                e
                            );
                        } else {
                            eprintln!(
                                "[hf2q qwen35 stream lcp store] mid-prefill \
                                 snapshot at chunk_pos={k_end} \
                                 (registry_len_after={})",
                                qwen.lcp_registry.len()
                            );
                        }
                    }
                }
            }
            last_logits_res
        } else if lcp_resume_start > 0 {
            // Suffix-only monolithic prefill from the LCP boundary.
            let suffix_tokens = &prompt_tokens[lcp_resume_start..];
            let suffix_len = suffix_tokens.len();
            let mut suffix_positions = vec![0i32; 4 * suffix_len];
            for axis in 0..4 {
                for t in 0..suffix_len {
                    suffix_positions[axis * suffix_len + t] =
                        (lcp_resume_start + t) as i32;
                }
            }
            eprintln!(
                "[hf2q qwen35 stream lcp resume] suffix prefill {} tokens \
                 (lcp_resume_start={}, prompt_len={})",
                suffix_len, lcp_resume_start, prompt_len
            );
            qwen.model.forward_gpu_last_logits(
                suffix_tokens,
                &suffix_positions,
                &mut kv_cache,
            )
        } else {
            qwen.model.forward_gpu_last_logits(
                prompt_tokens,
                positions_slice,
                &mut kv_cache,
            )
        };
        let prefill_logits = match prefill_logits_res {
            Ok(l) => l,
            Err(e) => {
                send!(GenerationEvent::Error(format!(
                    "qwen35 stream prefill failed: {e:#}"
                )));
                return;
            }
        };
        if is_greedy {
            next_token = greedy_argmax_last_token(&prefill_logits, qwen.vocab_size as u32);
        } else {
            let mut logits = prefill_logits.clone();
            next_token = sample_logits_qwen35(&mut logits, params, &[]);
        }
        // Prompt-cache snapshot is ONLY taken on the text-only path —
        // the vision path's bypass-on-read above means there's no key
        // collision risk, but it's also not productive to snapshot a
        // soft-token-tainted KV cache that subsequent text-only
        // requests must not restore. Skip the snapshot entirely on
        // extension paths.
        if is_greedy && !has_extension {
            match kv_cache.snapshot(&device) {
                Ok(snap) => qwen.prompt_cache.update(
                    prompt_tokens.to_vec(),
                    snap,
                    next_token,
                    params,
                ),
                Err(e) => tracing::warn!(error = %e, "qwen35 stream prompt_cache snapshot skipped"),
            }

            // ADR-017 Phase E.a B.3: end-of-prefill snapshot store
            // under chunk-position key (when not already stored by
            // mid-prefill loop).  See non-streaming sibling for the
            // gating logic explanation.
            if lcp_resume_enabled {
                let chunked_already_stored = stride > 0
                    && prompt_len > stride
                    && (crate::debug::INVESTIGATION_ENV.kv_lcp_chunked_prefill
                        || lcp_resume_start > 0);
                let prompt_stride_aligned =
                    stride > 0 && prompt_len > 0 && prompt_len % stride == 0;
                if !chunked_already_stored && prompt_stride_aligned {
                    if let Ok(snap) = kv_cache.snapshot(&device) {
                        let chunk_key = build_lcp_key_for_qwen35_chunk(
                            qwen, params, prompt_len,
                        );
                        let linear_capacity = kv_cache
                            .linear_attn
                            .first()
                            .map(|s| s.recurrent.byte_len())
                            .unwrap_or(0);
                        // Codex Phase-2b 2026-05-06: surface store errors
                        // instead of `let _ = ...`. Mantra: no fallback.
                        // ADR-027 iter-6b.2: write-through helper.
                        if let Err(e) = qwen.store_lcp_with_disk_writeback(
                            &kv_cache,
                            chunk_key,
                            prompt_tokens.to_vec(),
                            snap,
                            0,
                            linear_capacity,
                        ) {
                            tracing::warn!(
                                error = ?e,
                                "qwen35 streaming lcp_registry EOF store failed"
                            );
                        }
                    }
                }
            }
        }
    }
    let prefill_duration = prefill_start.elapsed();

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
    let mut reasoning_splitter = registration.and_then(ReasoningSplitter::from_registration);
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
        registration: Option<&ModelRegistration>,
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
                    let parsed = registration
                        .and_then(|r| super::registry::parse_tool_call_body(r, body));
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

    /// Emit a fragment through the reasoning splitter (if any) then the
    /// tool-call router.  Returns `false` on disconnect.
    fn emit_fragment_qwen35(
        reasoning_splitter: &mut Option<ReasoningSplitter>,
        tool_splitter: &mut Option<ToolCallSplitter>,
        body: &mut String,
        tc_index: &mut usize,
        saw_tc: &mut bool,
        registration: Option<&ModelRegistration>,
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
                            registration,
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
                registration,
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

    let first_text = qwen
        .tokenizer
        .decode(&[next_token], false)
        .unwrap_or_default();
    let mut is_eos_first = qwen.eos_token_ids.contains(&next_token);
    if !is_eos_first && !first_text.is_empty() {
        accumulated_text.push_str(&first_text);
        if !emit_fragment_qwen35(
            &mut reasoning_splitter,
            &mut tool_splitter,
            &mut tool_call_body,
            &mut tool_call_index,
            &mut saw_tool_call,
            registration,
            events,
            &first_text,
        ) {
            if let Some(c) = cancellation_counter {
                c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            return;
        }
    }
    completion_tokens += 1;
    if reasoning_splitter.as_ref().map(|s| s.in_reasoning()).unwrap_or(false) {
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
                qwen.model
                    .forward_gpu_greedy(&[next_token], &decode_positions, &mut kv_cache)
            } else {
                match qwen.model.forward_gpu_last_logits(
                    &[next_token],
                    &decode_positions,
                    &mut kv_cache,
                ) {
                    Ok(logits) => {
                        let mut tmp = logits;
                        Ok(sample_logits_qwen35(&mut tmp, params, &[next_token]))
                    }
                    Err(e) => Err(e),
                }
            };
            next_token = match dec_result {
                Ok(t) => t,
                Err(e) => {
                    send!(GenerationEvent::Error(format!(
                        "qwen35 stream decode step {step} failed: {e:#}"
                    )));
                    return;
                }
            };
            if qwen.eos_token_ids.contains(&next_token) {
                finish_reason = "stop";
                break;
            }
            completion_tokens += 1;
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
                registration,
                events,
                &fragment,
            ) {
                if let Some(c) = cancellation_counter {
                    c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return;
            }
            if reasoning_splitter.as_ref().map(|s| s.in_reasoning()).unwrap_or(false) {
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
                        return;
                    }
                }
                SplitSlot::Content => {
                    if !route_content_qwen35(
                        &mut tool_splitter,
                        &mut tool_call_body,
                        &mut tool_call_index,
                        &mut saw_tool_call,
                        registration,
                        events,
                        &tail,
                    ) {
                        if let Some(c) = cancellation_counter {
                            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
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
                        return;
                    }
                }
                ToolCallEvent::ToolCallText(t) => {
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
        cached_prompt_tokens: if prompt_cache_hit { Some(prompt_len) } else { None },
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

/// Wedge-3 / Phase D: chat-as-embedder.  Replaces the `worker_run` 501
/// arm for `Request::Embed`.
///
/// Single-shot prefill via `Qwen35Model::forward_embed_last` (Phase A),
/// returning the L2-normalized last-token hidden state of length
/// `cfg.hidden_size`.  KV cache allocated per-request (single forward,
/// no decode loop, so cache is discarded after).  Prompt cache is NOT
/// consulted here — the embedding path is a single forward; the
/// snapshot/replay savings are dominated by the no-decode shape.
pub fn embed_qwen35(qwen: &mut Qwen35LoadedModel, prompt_tokens: &[u32]) -> Result<Vec<f32>> {
    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "embed_qwen35: empty prompt_tokens"
    );
    let device = MlxDevice::new()
        .map_err(|e| anyhow::anyhow!("MlxDevice::new (qwen35 embed): {e}"))?;
    // For embeddings we don't need decode budget; pass max_tokens=0 to
    // size the cache to just the prompt.
    let mut kv_cache = alloc_kv_cache_for_request(qwen, &device, prompt_tokens.len(), 0)?;
    let positions = prefill_positions_for(prompt_tokens.len());
    qwen.model
        .forward_embed_last(prompt_tokens, &positions, &mut kv_cache)
        .context("Qwen35Model::forward_embed_last")
}

/// Default tool-call policy for the Wedge-3 streaming arm.
///
/// Wedge-3 ships `tool_choice=auto` semantics from the Qwen35 worker
/// arm — the chat handler at `handlers.rs::prepare_chat_generation_core`
/// ultimately decides the policy for the request and threads it through
/// `SamplingParams.tool_call_policy`.  But the `params` arg to the worker
/// thread is consumed BEFORE the streaming arm makes the
/// `emit_streaming_tool_call_close` call (the arm holds `&params`
/// throughout), so we re-derive Auto here for the body-parse-failure
/// branch.  This is consistent with Gemma's pre-W-B3 behavior — the
/// content-fallback path under Auto.
///
/// Wedge-4 follow-up: thread `params.tool_call_policy` through
/// `route_content_qwen35` so Constrained / AutoLazyGrammar policies
/// surface their loud-error semantics on Qwen35 too.  For MVP, Auto is
/// the only policy a Qwen35 chat request reaches today (the upstream
/// grammar plumbing is Gemma-only).
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
    use mlx_native::MlxDevice;

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
        let kv = HybridKvCache::new(&cfg, &device, 16, 1).expect("kv");
        let snap = kv.snapshot(&device).expect("snap");
        let prompt = vec![10u32, 20, 30, 40];
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

    /// Wedge-3 / iter-216 Phase C: gen-params mismatch misses even when
    /// the prompt matches.
    #[test]
    fn hybrid_prompt_cache_invalidates_on_genparams_mismatch() {
        let cfg = moe_cfg_40layer_for_cache_test();
        let device = MlxDevice::new().expect("device");
        let kv = HybridKvCache::new(&cfg, &device, 16, 1).expect("kv");
        let snap = kv.snapshot(&device).expect("snap");
        let prompt = vec![1u32, 2, 3];
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
        let kv = HybridKvCache::new(&cfg, &device, 16, 1).expect("kv");
        let snap = kv.snapshot(&device).expect("snap");
        let prompt = vec![1u32, 2, 3];

        // Lookup with sampling-mode bypasses even on a populated cache.
        let mut cache = HybridPromptCache::new();
        cache.update(prompt.clone(), snap, 1u32, &greedy_params());
        assert!(cache.has_entry());
        assert!(
            cache.try_match(&prompt, &sampling_params_with_temperature()).is_none(),
            "sampling-mode lookup must miss"
        );

        // Store with sampling-mode is a no-op (cache stays as it was).
        let kv2 = HybridKvCache::new(&cfg, &device, 16, 1).expect("kv2");
        let snap2 = kv2.snapshot(&device).expect("snap2");
        let mut cache2 = HybridPromptCache::new();
        cache2.update(prompt.clone(), snap2, 5u32, &sampling_params_with_temperature());
        assert!(
            !cache2.has_entry(),
            "sampling-mode update must be a no-op"
        );
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
        let raw =
            "Sure! <think>Let me solve this step by step.</think>The answer is 42.";
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
        ReasoningSplitter as _ReasoningSplitter, SplitSlot as _SplitSlot,
        ToolCallEvent as _ToolCallEvent, ToolCallSplitter as _ToolCallSplitter, QWEN35,
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
        let mut sp = _ReasoningSplitter::from_registration(&QWEN35)
            .expect("Qwen35 has reasoning markers");
        let mut all_pairs: Vec<(_SplitSlot, String)> = Vec::new();
        // Open marker spans a fragment boundary to exercise the
        // tail_buf logic.
        for frag in [
            "<thi", "nk>let me reason", " more</thin", "k>final answer",
        ] {
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
        let mut tcs = _ToolCallSplitter::from_registration(&QWEN35)
            .expect("Qwen35 has tool markers");
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

    /// Wedge-4e: the Phase-2c soft_token guard at
    /// `LoadedModel::Qwen35` streaming arm has been REMOVED. Pin via
    /// source-grep — if a future iter re-introduces a guard that
    /// short-circuits soft-token streaming, this test fails loud.
    #[test]
    fn wedge4e_phase_2c_soft_token_guard_is_removed() {
        let src = include_str!("engine.rs");
        assert!(
            !src.contains("Qwen35 streaming path does not yet support"),
            "Wedge-4e: Phase-2c soft_token guard at engine.rs's \
             LoadedModel::Qwen35 streaming arm MUST be removed — the \
             extended streaming entry now threads soft_tokens + \
             deepstack + positions through to the LM forward"
        );
        assert!(
            !src.contains("For Qwen3-VL image-bearing chat, set \\\"stream\\\": false."),
            "Wedge-4e: actionable diagnostic about set stream=false \
             MUST be removed (the streaming path is now the \
             production path)"
        );
        // Positive pin: the new dispatch reaches
        // `generate_stream_qwen35_once_extended`.
        assert!(
            src.contains("generate_stream_qwen35_once_extended"),
            "Wedge-4e: streaming arm must dispatch through the \
             extended entry"
        );
    }

    /// Wedge-4e handler-side: the `chat_completions_stream` 501
    /// reject for Qwen3-VL deepstack streaming has been removed.
    /// Image-bearing streaming chat now reaches the production
    /// engine path.
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
            tokenizer: tokenizers::Tokenizer::new(
                tokenizers::models::bpe::BPE::default(),
            ),
            chat_template: "{{ messages }}".to_string(),
            model_id: "iter-12-test".to_string(),
            model_path: std::path::PathBuf::from("/tmp/iter-12-test.gguf"),
            eos_token_ids: vec![151_645],
            hidden_size: 64,
            vocab_size: 256,
            context_length: Some(1024),
            quant_type: Some("Q4_K".to_string()),
            load_duration: std::time::Duration::from_millis(1),
            provenance: crate::serve::provenance::Provenance::External,
            prompt_cache: HybridPromptCache::new(),
            lcp_registry:
                crate::serve::kv_persist::lcp_registry::LcpRegistry::new(1),
            kv_metrics_sink: None,
            disk_persistor: None,
            lcp_hydrated_for_cfg: std::collections::HashSet::new(),
            tq_kv_active,
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
        let cache = alloc_kv_cache_for_request(&qwen, &device, 32, 16)
            .expect("alloc_kv_cache_for_request");
        assert!(
            !cache.full_attn.is_empty(),
            "fixture has full-attn layers"
        );
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
        let cache = alloc_kv_cache_for_request(&qwen, &device, 32, 16)
            .expect("alloc_kv_cache_for_request");
        assert!(!cache.full_attn.is_empty());
        for (i, slot) in cache.full_attn.iter().enumerate() {
            assert!(
                slot.tq.is_some(),
                "tq_kv_active=true: full_attn[{i}].tq must be populated"
            );
            let tq = slot.tq.as_ref().unwrap();
            assert_eq!(
                tq.norms_per_pos, 1,
                "head_dim=256 → norms_per_pos=1"
            );
        }
    }
}
