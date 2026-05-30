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

use anyhow::{anyhow, Context, Result};
use tokenizers::Tokenizer;

use crate::inference::models::qwen3vl_text::forward::forward_text_prefill_logits_last;
use crate::inference::models::qwen3vl_text::Qwen3VlTextModel;
use crate::serve::forward_prefill::{DeepstackInjection, SoftTokenInjection};
use crate::serve::load_info::{
    self, ArchFamily, ChatTemplateSource, LoadInfo, LoadInfoBuilder, TokenizerSource,
};
use crate::core::provenance::{self, Provenance};
use crate::serve::sampler_pure::{self, SamplingParams as SamplerPureParams};

use super::engine::{GenerationResult, LoadOptions, SamplingParams};
use super::registry::ModelRegistration;

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
    /// **ADR-040 Phase C iter-C2e (2026-05-30)** — structural witness
    /// for `EngineMode::SlotAware { max_slots: N }` spawn-time
    /// provisioning on the Qwen3-VL text-LM family.
    ///
    /// Provisioned at `spawn_with_mode(SlotAware { max_slots: N })`
    /// time via
    /// [`Self::provision_multi_seq_kv_for_slot_aware`] (called from
    /// `Engine::spawn_with_mode` per §6.1.52).  `None` for
    /// `EngineMode::SerialFifo` (byte-equivalence pin H222 holds —
    /// SerialFifo dispatch never touches this field); `Some(max_slots)`
    /// after SlotAware spawn.
    ///
    /// **Why a witness scalar, not a KV cache scaffold like Gemma 4 /
    /// Qwen35**: Qwen3-VL text-LM today runs the iter-9b *naive O(N²)
    /// re-prefill loop* in
    /// [`generate_qwen3vl_text_once`] — no persistent KV cache is
    /// allocated, every step re-prefills from position 0. The real
    /// per-step KV cache is upstream-blocked on **iter-228a** (the 501
    /// sentinel today; see ADR-005 Wedge-4 + this file's module
    /// docstring). Until iter-228a lands a real forward path past the
    /// 501 sentinel, the spawn-time provisioning surface is a
    /// structural witness only — the scalar carries `max_slots` so
    /// future iters + operator log greps + the H219 test bank can
    /// pin the provisioning call without any device alloc.
    ///
    /// **iter-C2e-cont (post iter-228a)** — replaces this scalar with
    /// a real `Option<Qwen3VlTextKvCache>` (or per-layer
    /// `Vec<MultiSeqQwen3VlKvBuffers>` depending on the iter-228a
    /// KV shape) and lifts the four worker arms onto the persistent
    /// cache.  The four worker arms TODAY surface typed
    /// `MultiSeqError::CapabilityUnsupported` at `SlotId(N>0)` per
    /// `iter-C2e-cont per ADR-040 §6.1.52 — gated on iter-228a`
    /// label, mirror of C2c+C2d-cont's worker-arm clamp shape.
    pub slot_aware_max_slots: Option<u32>,
}

impl Qwen3VlTextLoadedModel {
    /// **ADR-040 Phase C iter-C2e (2026-05-30)** — provision the
    /// structural witness for
    /// [`crate::serve::api::engine::EngineMode::SlotAware`] spawn-time
    /// activation on Qwen3-VL.
    ///
    /// Called by
    /// [`crate::serve::api::engine::Engine::spawn_with_mode`] when
    /// `EngineMode::SlotAware { max_slots: N }` is selected for a
    /// Qwen3-VL engine. Sets [`Self::slot_aware_max_slots`] to
    /// `Some(max_slots)` — a witness scalar, NOT a KV cache scaffold
    /// (cf. Gemma 4 `provision_multi_seq_kv_for_slot_aware` which
    /// allocates per-layer `MultiSeqHbKvBuffers` via the A3a
    /// allocator + Qwen35 `provision_multi_seq_kv_for_slot_aware`
    /// which allocates `HybridKvCache::new_with_options(.., n_seqs =
    /// max_slots, ..)`).
    ///
    /// **Why a witness, not a real KV cache** (load-bearing for the
    /// C2e iter shape): Qwen3-VL text-LM today runs the iter-9b
    /// naive O(N²) re-prefill loop with no persistent KV cache; every
    /// generated token re-runs the full prefill from position 0
    /// (see [`generate_qwen3vl_text_once`]). The real per-step KV
    /// cache is upstream-blocked on **iter-228a** (the 501 sentinel —
    /// see `Qwen3VlTextLoadedModel`'s module docstring + ADR-005
    /// Wedge-4 + ADR-040 §6.1.22's C2e cite). The C2e iter's role is
    /// to flip the spawn arm to `Ok(Engine)` so the
    /// `InflightBatchedScheduler` can be wired end-to-end at the API
    /// surface; the worker hot path on the persistent cache lands at
    /// iter-C2e-cont (post iter-228a).
    ///
    /// **SerialFifo byte-equivalence pin (H222)**: this method is
    /// NEVER called when `EngineMode::SerialFifo` is selected. The
    /// `EngineMode::SerialFifo` arm in
    /// `Engine::spawn_with_mode` does not invoke any
    /// `provision_multi_seq_kv_for_slot_aware` method on any family;
    /// the SerialFifo dispatch goes directly through the legacy
    /// `Engine::spawn` body with `slot_aware_max_slots: None`
    /// preserved verbatim from `Qwen3VlTextLoadedModel::load`.
    ///
    /// # Errors
    ///
    /// Returns an `anyhow::Error` if `max_slots == 0`. The spawn arm
    /// has a parallel pre-check that returns
    /// [`crate::serve::api::engine::EngineSpawnError::ModeNotYetWired`]
    /// at the API boundary BEFORE this method runs, so this is
    /// defense-in-depth.
    pub fn provision_multi_seq_kv_for_slot_aware(
        &mut self,
        max_slots: u32,
    ) -> Result<()> {
        if max_slots == 0 {
            anyhow::bail!(
                "ADR-040 C2e: provision_multi_seq_kv_for_slot_aware called with \
                 max_slots == 0; spawn_with_mode invariant is max_slots >= 1 \
                 (EngineMode::SlotAware variant enforces this at the API \
                 boundary — caller violated)"
            );
        }
        // iter-C2e: witness-only provisioning (no device alloc).  See
        // method docstring + ADR §6.1.52 for the "why a witness, not
        // a real KV cache" rationale.  iter-C2e-cont (post iter-228a)
        // replaces this with a real persistent-cache scaffold +
        // worker-hot-path lift.
        self.slot_aware_max_slots = Some(max_slots);
        Ok(())
    }

    /// **ADR-040 §6.1.55 iter-C2e-cont (2026-05-30)** — structural
    /// worker hot path lift for the four SlotAware worker arms.
    ///
    /// Replaces the iter-C2e inline `anyhow::anyhow!("capability_
    /// unsupported: ...")` synthesis at the four worker-arm clamps
    /// with a real helper that:
    ///
    /// 1. **Take/restores the witness scalar** —
    ///    `slot_aware_max_slots.take()` then writes the value back into
    ///    the field via `.replace(...)` after the sentinel call.  This
    ///    pins the C2e provisioning surface in the multi-seq lift call
    ///    graph (operator-grep at the C2e-cont site sees both the
    ///    provisioning-time witness write AND the worker-time
    ///    take/restore).  Today the take/restore is a no-op
    ///    structurally because the scalar is a `u32` not a real
    ///    cache; once iter-228a lands a real persistent KV cache the
    ///    take/restore becomes the real cache get-then-put discipline
    ///    that Qwen35 + Gemma 4 worker arms already use
    ///    (`q.persistent_kv_cache = Some(persistent)`).
    /// 2. **Delegates to the iter-228a 501 sentinel verbatim** —
    ///    [`crate::inference::models::qwen3vl_text::forward::
    ///    qwen3vl_text_forward_pending_err`] is the upstream 501
    ///    contract; the helper wraps it with the
    ///    `capability_unsupported:` prefix + the arm-specific sublabel
    ///    + the `SlotId(N)` cite that operator log greps depend on.
    ///    Sentinel propagates verbatim through `?` at the caller (H240
    ///    pins this).
    ///
    /// # Why the structural lift is "shippable now"
    ///
    /// Today the helper carries the witness take/restore over a
    /// `Option<u32>` scalar — purely structural.  When iter-228a lands
    /// past the 501 sentinel, the field flips to
    /// `Option<Qwen3VlTextKvCache>` (or a per-layer
    /// `Vec<MultiSeqQwen3VlKvBuffers>` depending on iter-228a's KV
    /// shape; see `slot_aware_max_slots` field docstring + ADR-040
    /// §6.1.52); the take/restore discipline established here is the
    /// production worker-arm shape Qwen35 + Gemma 4 already use.
    ///
    /// # Cross-references
    ///
    /// - ADR-040 §6.1.52 — iter-C2e spawn-time activation closure.
    /// - ADR-040 §6.1.55 — iter-C2e-cont structural lift closure
    ///   (this method's home).
    /// - `Qwen35Model::persistent_kv_cache` get-then-put pattern at
    ///   `engine.rs` worker arms — the production take/restore shape
    ///   this helper mirrors.
    pub fn handle_qwen3vl_slot_aware_n_gt_0_sentinel<T>(
        &mut self,
        slot_id: crate::serve::multi_seq_kv::SlotId,
        arm_sublabel: &str,
    ) -> Result<T> {
        // Witness take/restore — structural mirror of Qwen35 /
        // Gemma 4 `persistent_kv_cache = Some(persistent);` discipline.
        // Today the scalar is `Option<u32>`; iter-228a flips it to a
        // real cache type.  Pre-A4 byte-equivalence pin H239 holds
        // because SerialFifo NEVER calls this helper (the worker-arm
        // clamp predicate `handle.slot_id != SlotId(0)` short-circuits
        // at `SlotId(0)`).
        let witness = self.slot_aware_max_slots.take();
        // Delegate to the iter-228a 501 sentinel verbatim — the
        // forward path returns the typed-deferred error that propagates
        // through the worker arm + chat handler all the way to the
        // HTTP 501 response.  This is the load-bearing sentinel-
        // propagation step (H240).
        let sentinel_result: Result<T> =
            crate::inference::models::qwen3vl_text::forward::qwen3vl_text_forward_pending_err();
        // Restore the witness regardless of sentinel outcome —
        // keeps the spawn-time invariant intact for the next request.
        // Mirrors the Qwen35 `q.persistent_kv_cache = Some(persistent);`
        // unconditional post-pattern at engine.rs:5434+.
        self.slot_aware_max_slots = witness;
        // Wrap the sentinel error with the arm-specific
        // capability_unsupported label.  Pre-existing iter-C2e label
        // shape preserved verbatim per H220's operator-grep contract.
        sentinel_result.map_err(|sentinel_err| {
            anyhow!(
                "capability_unsupported: ADR-040 \
                 {arm_sublabel} (iter-C2e-cont per ADR-040 §6.1.55 — \
                 structural worker hot path lift gated on iter-228a \
                 Qwen3-VL forward path landing past the 501 sentinel; \
                 worker hot path lift onto the persistent multi-seq \
                 cache cannot land until the persistent cache itself \
                 exists). SlotId({}) — sentinel propagated verbatim: \
                 {sentinel_err}",
                slot_id.0,
            )
        })
    }
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
            // ADR-040 Phase C iter-C2e (2026-05-30) — None until
            // `spawn_with_mode(SlotAware { .. })` calls
            // `provision_multi_seq_kv_for_slot_aware` per §6.1.52.
            // SerialFifo dispatch leaves this `None` (H222
            // byte-equivalence pin).
            slot_aware_max_slots: None,
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
            // ADR-027 Phase B iter-17: Qwen3-VL has no TQ wiring yet
            // (text path forward landed iter-228b without TQ). Always
            // false until a future iter ports ADR-027 to qwen3vl_text.
            tq_kv_active: false,
            // ADR-040 §3.5 iter-A5c (cfa-A5b CRITICAL #1) — Qwen3-VL
            // text LM is plain dense GQA with a homogeneous
            // `(num_key_value_heads, head_dim)` shape across all
            // layers; the flattened scalar formula is EXACT for this
            // arch. `None` means "no override".
            kv_bytes_per_token_override: None,
        }
    }
}

// ===========================================================================
// iter-9b: Generate path
// ===========================================================================
//
// Replaces the iter-228a `qwen3vl_text_forward_pending` 501 sentinel
// in `engine.rs::worker_run`'s `LoadedModel::Qwen3VlText` arms with a
// real generation loop driven by [`forward_text_prefill_logits_last`].
//
// **Naive O(N²) re-prefill loop** — each generated token re-runs the
// full prefill from position 0. This is correct but wastes work; an
// iter-9c+ KV-cache-incremental path can avoid that. iter-9b's
// deliverable is correctness against AC-3 ("coherent generated text"),
// not performance — the live AC-3 prompts are all <500 tokens with
// small `max_tokens` budgets, so even O(N²) at 28 layers × 3 sessions
// per layer per token is acceptable for the closure window.

/// Build the flat `[4 * seq_len]` axis-major position buffer for the
/// IMROPE kernel. For text-only prompts (the iter-9b text Generate
/// path), all 4 axes carry the absolute token index — same shape as
/// `engine_qwen35::prefill_positions_for`. Image-bearing chats
/// (GenerateWithSoftTokens / GenerateStream) use
/// [`crate::serve::forward_prefill::build_qwen3vl_positions`] to lay
/// down distinct (t, y, x) values for image-token positions.
fn text_only_positions_for(prompt_len: usize) -> Vec<i32> {
    let mut flat = vec![0i32; 4 * prompt_len];
    for axis in 0..4 {
        for t in 0..prompt_len {
            flat[axis * prompt_len + t] = t as i32;
        }
    }
    flat
}

/// `true` when sampling parameters reduce to deterministic argmax.
/// Mirrors `engine_qwen35::is_greedy_eligible`. Used to short-circuit
/// the sampler dispatch — `argmax` is faster and avoids a frivolous
/// RNG seed when none was requested.
fn is_greedy_eligible_qwen3vl(params: &SamplingParams) -> bool {
    !(params.temperature > 0.0
        || params.top_k > 0
        || params.top_p < 1.0
        || params.repetition_penalty != 1.0
        || params.seed.is_some())
}

/// Argmax over a `[vocab]` logits slice. Returns the token id (u32).
/// Ties broken by lower index (the natural `iter::position_max`
/// ordering). Used for greedy decode in
/// [`generate_qwen3vl_text_once`].
fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best_idx: usize = 0;
    let mut best_val: f32 = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Sample one token from `logits` per `params`, with the standard
/// repetition-penalty + temperature + top-p / top-k / min-p chain
/// (`sampler_pure::sample_token`).
fn sample_logits_qwen3vl(
    logits: &mut [f32],
    params: &SamplingParams,
    generated: &[u32],
) -> u32 {
    let sp = SamplerPureParams {
        temperature: params.temperature as f64,
        top_p: params.top_p as f64,
        top_k: params.top_k,
        min_p: params.min_p as f64,
        repetition_penalty: params.repetition_penalty as f64,
        max_tokens: params.max_tokens,
    };
    sampler_pure::sample_token(logits, &sp, generated)
}

/// Decode tokens to text via the loaded HF tokenizer. The
/// `skip_special_tokens` flag is `false` to match the qwen35 path
/// (special tokens like `<|im_end|>` ARE meaningful and should not be
/// stripped — the EOS check happens before this point so any visible
/// special tokens are mid-stream artifacts the chat template chose to
/// expose).
fn decode_to_text(
    tokenizer: &Tokenizer,
    decoded_tokens: &[u32],
) -> Result<String> {
    tokenizer
        .decode(decoded_tokens, /* skip_special_tokens */ false)
        .map_err(|e| anyhow!("Qwen3-VL tokenizer decode: {e}"))
}

/// Run a non-streaming text-only generation request against a loaded
/// Qwen3-VL text-LM.
///
/// **iter-9b text-only path** — does NOT splice soft tokens or
/// deepstack residuals; the chat handler dispatches image-bearing
/// requests to [`generate_qwen3vl_text_with_soft_tokens_once`]
/// instead.
///
/// # Loop shape
///
/// 1. Tokenize → call `forward_text_prefill_logits_last` over the
///    growing prompt_tokens slice.
/// 2. Argmax (greedy) or `sampler_pure::sample_token` (sampling).
/// 3. EOS check → break with `finish_reason = "stop"`; otherwise push
///    the token onto both `decoded_tokens` (for tokenizer.decode) and
///    `tokens_so_far` (for next iteration's prefill).
/// 4. After `max_tokens`, exit with `finish_reason = "length"`.
///
/// Stop-string matching is **not yet implemented** in iter-9b — the
/// MVP exits on EOS or max_tokens. iter-9c can add the
/// `params.stop_strings` scan against the running decoded text.
/// Reasoning splitter + tool-call splitter are similarly deferred
/// (Qwen3-VL's chat template doesn't emit reasoning markers in the
/// canonical 2B/4B configurations).
///
/// # Errors
///
/// * Empty `prompt_tokens`.
/// * Any forward-path failure (propagated with the iteration step
///   number in context).
/// * Tokenizer decode failure on the final text.
pub fn generate_qwen3vl_text_once(
    qwen: &mut Qwen3VlTextLoadedModel,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    _registration: Option<&ModelRegistration>,
) -> Result<GenerationResult> {
    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "generate_qwen3vl_text_once: empty prompt_tokens"
    );
    let prompt_len = prompt_tokens.len();
    let max_tokens = params.max_tokens.max(1);
    let is_greedy = is_greedy_eligible_qwen3vl(params);

    // The "prefill" duration in the iter-9b naive loop covers the
    // FIRST forward + sample (which carries the full prompt cost).
    // Subsequent iterations are the "decode" cost. The split mirrors
    // what the OpenAI usage timing convention surfaces; downstream
    // metrics treat `prefill_duration` as the warm-cold latency
    // proxy.
    let mut tokens_so_far: Vec<u32> = prompt_tokens.to_vec();
    let mut decoded_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut finish_reason: &'static str = "length";

    let prefill_start = Instant::now();
    let mut prefill_duration: Duration = Duration::ZERO;
    let decode_start_outer = Instant::now();
    for step in 0..max_tokens {
        // Build text-only positions for the current sequence length.
        let positions = text_only_positions_for(tokens_so_far.len());

        // Run the full forward pass, returning the `[vocab]` logits at
        // the LAST position. iter-9b's naive loop re-runs this at
        // every step (no KV cache); iter-9c can add an incremental
        // path.
        let mut logits = forward_text_prefill_logits_last(
            &mut qwen.model,
            &tokens_so_far,
            &positions,
            None, // text-only: no DeepStack chunks
            &[],  // text-only: no soft-token injections
        )
        .with_context(|| format!("forward step {step}"))?;

        // After the first forward, log the prefill wall-clock so the
        // GenerationResult split is meaningful. (`step == 0` measures
        // the FULL cold prefill including all 28 × 3 sessions.)
        if step == 0 {
            prefill_duration = prefill_start.elapsed();
        }

        let next_token: u32 = if is_greedy {
            argmax_u32(&logits)
        } else {
            sample_logits_qwen3vl(&mut logits, params, &decoded_tokens)
        };

        // EOS: terminate before pushing — peer convention is that the
        // EOS token is the SIGNAL, not part of the visible output. The
        // chat handler doesn't render it in `message.content`.
        if qwen.eos_token_ids.contains(&next_token) {
            finish_reason = "stop";
            break;
        }

        decoded_tokens.push(next_token);
        tokens_so_far.push(next_token);
    }
    let decode_duration = decode_start_outer
        .elapsed()
        .saturating_sub(prefill_duration);

    let text = decode_to_text(&qwen.tokenizer, &decoded_tokens)?;

    Ok(GenerationResult {
        text,
        // iter-9b doesn't run the reasoning splitter — Qwen3-VL's chat
        // template doesn't register `<think>` markers in the canonical
        // 2B/4B GGUF, so `reasoning_text` is structurally absent.
        // iter-9c can add the splitter once a reasoning-marker
        // registration exists for this family.
        reasoning_text: None,
        prompt_tokens: prompt_len,
        completion_tokens: decoded_tokens.len(),
        reasoning_tokens: None,
        finish_reason,
        prefill_duration,
        decode_duration,
        // iter-9b has no prompt cache wired; cached_tokens is always 0.
        cached_tokens: 0,
            logprobs: None,
    })
}

/// Run a non-streaming generation request against a loaded Qwen3-VL
/// text-LM with **soft-token splicing** + **DeepStack injection** —
/// the image-bearing variant.
///
/// # Soft tokens
///
/// `soft_tokens` is a list of [`super::engine::SoftTokenInjection`]
/// ranges; for each range `[start, end)` and corresponding
/// `embeddings: &MlxBuffer` of shape `[end-start, hidden]`, the prefill
/// embedding stream's row `start..end` is OVERWRITTEN by the projected
/// vision tokens. Placeholder token ids (`<|image_pad|>`) are ignored
/// at those positions — the override fully replaces the embed-table
/// lookup. Mirrors qwen35's
/// `embed_tokens_gpu_with_soft_tokens` contract
/// (`forward_gpu.rs::embed_tokens_gpu_with_soft_tokens`).
///
/// # DeepStack
///
/// `deepstack` carries the per-LM-layer residual chunks built by the
/// ViT projector (Wedge-4c). The forward path dispatches
/// `image_token_residual_add_gpu` at every LM layer
/// `il < cfg.n_deepstack_layers` to scatter-add the chunk into image-
/// token positions. See `forward_text_prefill_logits_last`'s Phase D.
///
/// # Positions
///
/// `positions_flat` is the 4-axis IMROPE position buffer built by
/// [`crate::serve::forward_prefill::build_qwen3vl_positions`] from the
/// chat handler's image-grid metadata. iter-9b passes it through
/// verbatim for the prefill; for decoded tokens (positions
/// `prompt_len..tokens_so_far.len()`) we extend with monotone t-axis
/// values past the last image's `temporal_advance()`.
///
/// # Soft-token splicing implementation note
///
/// iter-9b's forward path doesn't yet take soft tokens directly —
/// `forward_text_prefill_logits_last` always uses the embed-table
/// lookup. To splice, we need to PRE-MUTATE the embedding stream
/// before the forward consumes it. The cleanest way is to extend
/// `forward_text_prefill_logits_last` with an
/// `Option<&[SoftTokenInjection<'_>]>` parameter; iter-9b takes that
/// approach below.
///
/// # Errors
///
/// As [`generate_qwen3vl_text_once`] plus:
/// * `positions_flat.len() != 4 * prompt_tokens.len()`.
/// * Any soft-token range out of bounds.
/// * Any DeepStack invariant violation (forwarded from the forward
///   function).
pub fn generate_qwen3vl_text_with_soft_tokens_once(
    qwen: &mut Qwen3VlTextLoadedModel,
    prompt_tokens: &[u32],
    soft_tokens: &[SoftTokenInjection<'_>],
    deepstack: Option<&DeepstackInjection<'_>>,
    positions_flat: &[i32],
    params: &SamplingParams,
    _registration: Option<&ModelRegistration>,
) -> Result<GenerationResult> {
    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "generate_qwen3vl_text_with_soft_tokens_once: empty prompt_tokens"
    );
    let prompt_len = prompt_tokens.len();
    if positions_flat.len() != 4 * prompt_len {
        return Err(anyhow!(
            "generate_qwen3vl_text_with_soft_tokens_once: positions_flat.len()={} != \
             4 * prompt_len ({})",
            positions_flat.len(),
            4 * prompt_len
        ));
    }
    let max_tokens = params.max_tokens.max(1);
    let is_greedy = is_greedy_eligible_qwen3vl(params);

    let mut tokens_so_far: Vec<u32> = prompt_tokens.to_vec();
    let mut decoded_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut finish_reason: &'static str = "length";

    // The position buffer grows by 1 per generated token. For the
    // text-only suffix (decoded tokens following the multimodal
    // prompt), we extend by replicating the next sequential index
    // across all 4 axes — same convention as
    // `text_only_positions_for`. Image-grid-spanning positions stay
    // verbatim from the original buffer.
    let mut current_positions: Vec<i32> = Vec::with_capacity(4 * (prompt_len + max_tokens));
    // Layout: `positions_flat` is axis-major over prompt_len; we
    // expand to `[4 * (prompt_len + max_tokens)]` as we go. For
    // step==0 we consume positions_flat verbatim; for step>0 we add
    // one column per step.

    let prefill_start = Instant::now();
    let mut prefill_duration: Duration = Duration::ZERO;
    let decode_start_outer = Instant::now();
    for step in 0..max_tokens {
        // Build positions for step `step`: prompt + decoded_tokens.
        // Layout still axis-major over total length (= prompt_len + step).
        let total_len = prompt_len + step;
        current_positions.resize(4 * total_len, 0i32);

        // Refill the prompt-side positions from positions_flat (axis-major).
        for axis in 0..4 {
            for t in 0..prompt_len {
                current_positions[axis * total_len + t] =
                    positions_flat[axis * prompt_len + t];
            }
        }
        // Append the decoded-token positions: monotone after the last
        // prompt-side t-axis value. Per peer
        // (`build_qwen3vl_positions::TEXT_AFTER_IMAGE` branch, used in
        // `forward_prefill.rs:262-268`), text tokens after an image
        // get t = t_global with t_global incrementing past the
        // image's temporal_advance(). For decoded tokens, we treat
        // each as a fresh text token and increment by 1 per step from
        // the last prompt-side t value.
        if step > 0 {
            // Find the last prompt-side t-axis position (axis 0,
            // index prompt_len-1).
            let last_prompt_t = positions_flat[prompt_len - 1];
            for s in 0..step {
                let p = prompt_len + s;
                let t_val = last_prompt_t + 1 + s as i32;
                for axis in 0..4 {
                    current_positions[axis * total_len + p] = t_val;
                }
            }
        }

        // iter-10a: soft-token splicing is implemented inside
        // `forward_text_prefill_logits_last` (CPU-side overwrite of
        // `<|image_pad|>` rows with ViT-projected embeddings before
        // the GPU upload). The forward re-runs from scratch each
        // step (no KV cache yet), so the embed table is re-read
        // every step — the override pass MUST run every step too,
        // otherwise step > 0 would forget the image embeddings and
        // the prompt would degrade to placeholder-token semantics.
        // The validate cost is small relative to the 28-layer
        // forward, so we pay it every step rather than caching.
        let mut logits = forward_text_prefill_logits_last(
            &mut qwen.model,
            &tokens_so_far,
            &current_positions,
            deepstack,
            soft_tokens,
        )
        .with_context(|| format!("forward step {step} (multimodal)"))?;

        if step == 0 {
            prefill_duration = prefill_start.elapsed();
        }

        let next_token: u32 = if is_greedy {
            argmax_u32(&logits)
        } else {
            sample_logits_qwen3vl(&mut logits, params, &decoded_tokens)
        };

        if qwen.eos_token_ids.contains(&next_token) {
            finish_reason = "stop";
            break;
        }

        decoded_tokens.push(next_token);
        tokens_so_far.push(next_token);
    }
    let decode_duration = decode_start_outer
        .elapsed()
        .saturating_sub(prefill_duration);

    let text = decode_to_text(&qwen.tokenizer, &decoded_tokens)?;

    Ok(GenerationResult {
        text,
        reasoning_text: None,
        prompt_tokens: prompt_len,
        completion_tokens: decoded_tokens.len(),
        reasoning_tokens: None,
        finish_reason,
        prefill_duration,
        decode_duration,
        cached_tokens: 0,
            logprobs: None,
    })
}
