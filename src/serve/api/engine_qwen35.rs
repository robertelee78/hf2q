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
        let model = Qwen35Model::load_from_gguf(&gguf, &mut progress)
            .context("Qwen35Model::load_from_gguf")?;
        // ADR-018 C3: legacy `tracing::info!("Qwen35 SERVE load: weights loaded ({} layers, variant={:?})", ...)`
        // was deleted here. `emit_tracing(&info)` surfaces both `n_layers`
        // and architecture facts as structured fields.

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
        })
    }
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
    HybridKvCache::new(&qwen.model.cfg, device, max_seq as u32, 1)
        .context("HybridKvCache::new")
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
        repetition_penalty: params.repetition_penalty as f64,
        max_tokens: params.max_tokens,
    };
    sampler_pure::sample_token(logits, &sp, generated)
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

    // ── Prompt-cache fast-path ────────────────────────────────────
    let prompt_cache_hit = qwen
        .prompt_cache
        .try_match(prompt_tokens, params)
        .is_some();

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
        let positions = prefill_positions_for(prompt_len);
        let prefill_logits = qwen
            .model
            .forward_gpu_last_logits(prompt_tokens, &positions, &mut kv_cache)
            .context("Qwen35Model::forward_gpu_last_logits (prefill)")?;
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
        if is_greedy {
            match kv_cache.snapshot(&device) {
                Ok(snap) => qwen.prompt_cache.update(
                    prompt_tokens.to_vec(),
                    snap,
                    next_token,
                    params,
                ),
                Err(e) => {
                    // Snapshot failure is non-fatal — we just don't cache
                    // this prefill.  Log and continue.
                    tracing::warn!(error = %e, "qwen35 prompt_cache snapshot skipped");
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
pub fn generate_stream_qwen35_once(
    qwen: &mut Qwen35LoadedModel,
    prompt_tokens: &[u32],
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

    let pre_dispatches = mlx_native::dispatch_count();
    let pre_syncs = mlx_native::sync_count();

    let prompt_cache_hit = qwen
        .prompt_cache
        .try_match(prompt_tokens, params)
        .is_some();

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
        let positions = prefill_positions_for(prompt_len);
        let prefill_logits = match qwen
            .model
            .forward_gpu_last_logits(prompt_tokens, &positions, &mut kv_cache)
        {
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
        if is_greedy {
            match kv_cache.snapshot(&device) {
                Ok(snap) => qwen.prompt_cache.update(
                    prompt_tokens.to_vec(),
                    snap,
                    next_token,
                    params,
                ),
                Err(e) => tracing::warn!(error = %e, "qwen35 stream prompt_cache snapshot skipped"),
            }
        }
    }
    let prefill_duration = prefill_start.elapsed();

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
                    if super::engine::emit_streaming_tool_call_close(
                        parsed,
                        body_dump,
                        params_tool_call_policy_for_qwen35_stream(),
                        tc_index,
                        saw_tc,
                        events,
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
            let pos = (prompt_len + step - 1) as i32;
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
