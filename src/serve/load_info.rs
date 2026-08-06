//! Unified per-load metadata snapshot — the type-and-helper foundation for
//! the load-UX uniformity migration described in
//! `docs/research/model_load_ux_uniformity_2026-05-01.md` and accepted by
//! ADR-018.
//!
//! # Scope (commit C1)
//!
//! C1 of the migration introduces the type definitions and the two
//! GGUF-derivation helpers (`infer_quant_label`, `compute_bpw`) and
//! relocates the previously-duplicated `infer_quant_type_from_gguf`
//! body out of `serve::api::engine` and `serve::api::engine_qwen35` into
//! the single home here.
//!
//! C1 deliberately stops short of:
//!   * the `LoadInfoBuilder` trait + per-variant impls (C2),
//!   * `print_banner` / `emit_tracing` (C2),
//!   * any wiring of the new types into call sites (C3 / C4),
//!   * `/v1/models` propagation (C5).
//!
//! Visible behaviour delta in C1: zero.  This file is library code that no
//! production call site invokes yet — `infer_quant_label` IS already wired
//! (it replaces the two byte-identical legacy bodies), but its behaviour is
//! provably identical to what shipped before, so user-visible output is
//! unchanged.
//!
//! # Why a snapshot type, not borrowed state
//!
//! `LoadInfo` is constructed by each `*LoadedModel::load` *after* the
//! underlying load succeeds.  It owns no live model state (no buffers, no
//! tokenizer, no weights) — it is a snapshot of the relevant facts at
//! load-completion time.  This makes it cheap to clone into tracing
//! fields, the SERVE-mode banner, and `/v1/models` without leaking model
//! lifetimes into request handlers.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::core::provenance::Provenance;

const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

// ---------------------------------------------------------------------------
// Origin enums — uniform across arches
// ---------------------------------------------------------------------------

/// Origin of the chat template string actually in effect for a load.
///
/// Live origins plus an explicit `None` for pre-Wedge-3 GGUFs that lack the
/// key and haven't been routed through a fallback yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplateSource {
    /// Lifted verbatim from the GGUF metadata key
    /// `tokenizer.chat_template`.
    GgufEmbedded,
    /// Operator override via `--chat-template` (CLI generate flag).
    CliOverride,
    /// Hard-coded fallback (e.g. the Gemma4 API path's
    /// `FALLBACK_GEMMA4_API_CHAT_TEMPLATE`).  The named string identifies
    /// *which* fallback is in effect so the banner can name it without
    /// reaching back into `crate::serve::*` constants.
    HardcodedFallback {
        /// Stable identifier for the fallback in effect.  `&'static str`
        /// because every fallback is a compile-time constant.
        name: &'static str,
    },
    /// Architecture-owned encoder that implements the upstream prompt
    /// protocol directly instead of interpreting a Jinja template.
    NativeEncoding {
        /// Stable protocol/encoder identifier shown in the load banner.
        name: &'static str,
    },
    /// Empty / not yet rendered.  Pre-Wedge-3 Qwen35 GGUFs lacked
    /// `tokenizer.chat_template`; the engine emits a `tracing::warn!` at
    /// load and substitutes a hard-coded fallback.  `None` documents the
    /// raw absence; once the fallback is selected the variant flips to
    /// `HardcodedFallback`.
    None,
}

/// Source of the active tokenizer.
///
/// `HfTokenizerJson` is the path Gemma takes today (a `tokenizer.json`
/// loaded via `tokenizers::Tokenizer::from_file`), and `GgufEmbedded` is
/// the path Qwen3.5/3.6 takes (`build_tokenizer_from_gguf` mirroring
/// `llama-vocab.cpp:2197-2253` to avoid the apex-GGUF OOB-token bug
/// documented at engine_qwen35.rs:148-178).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerSource {
    /// `tokenizers::Tokenizer::from_file(<path>)`.
    HfTokenizerJson {
        /// Filesystem path to the `tokenizer.json` actually loaded.
        path: PathBuf,
    },
    /// `build_tokenizer_from_gguf` — reads `tokenizer.ggml.tokens`,
    /// `tokenizer.ggml.merges`, and per-token type metadata directly from
    /// the GGUF KV section.
    GgufEmbedded,
}

// ---------------------------------------------------------------------------
// Architecture facts
// ---------------------------------------------------------------------------

/// Mixture-of-Experts shape, when applicable.  `None` (the wrapping
/// `Option<MoeShape>` on `LoadInfo`) for dense models.
///
/// `n_experts_per_tok` is the routed-experts count actually firing per
/// token — for MoE+shared-expert architectures this is `top_k`, NOT
/// `top_k + 1`.  The shared expert is implicit in this banner field;
/// surfacing it explicitly is a future enhancement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeShape {
    /// Total expert count in the layer (`{arch}.expert_count`).
    pub n_experts: u32,
    /// Routed experts per token (`{arch}.expert_used_count`).
    pub n_experts_per_tok: u32,
}

/// Vision-projector pairing — `None` if no mmproj is loaded.
///
/// The path is canonical (already-resolved); the SHA-256 is provenance
/// data sourced from the projector GGUF's own
/// `hf2q.mmproj_sha256` key (when written by the hf2q writer) and may
/// be `None` for externally-produced mmprojs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionProjector {
    /// Filesystem path to the projector GGUF.
    pub mmproj_path: PathBuf,
    /// Optional lowercase-hex SHA-256 of the projector GGUF, lifted from
    /// its provenance metadata.
    pub mmproj_sha256: Option<String>,
}

/// One-of-many fact: which forward-pass family this model dispatches to.
///
/// This is *not* the GGUF `general.architecture` string — that is a
/// finer-grained name (e.g. `qwen35moe`).  `ArchFamily` is the dispatch
/// bucket, used by both the banner (display) and any future code path
/// that needs to dispatch on family without re-parsing the arch string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchFamily {
    /// Gemma4 / gemma4-shaped (sliding-window hybrid + global; MoE
    /// optional).
    Gemma4,
    /// Qwen3.5 / Qwen3.6 (DeltaNet linear-attn + periodic full-attn,
    /// dense or MoE).
    Qwen35,
    /// Qwen3-VL **text** LM (plain dense GQA + per-head Q/K RMSNorm +
    /// 3D-mRoPE + DeepStack-residual injection). The vision side is a
    /// separate ViT (Wedge-4c) that produces image-token embeddings the
    /// text LM consumes. ADR-005 Wedge-4 / iter-228a.
    Qwen3VlText,
    /// DeepSeek-V4-Flash verifier with compressed/indexed sparse attention.
    Deepseek4,
    /// Reserved — Llama4 (placeholder; the dispatcher errors at the
    /// `LoadedModel::load` arch peek today).
    Llama4Reserved,
}

impl ArchFamily {
    /// Stable lowercase string form, suitable for banner output and
    /// `tracing::info!` field values.
    pub fn as_str(&self) -> &'static str {
        match self {
            ArchFamily::Gemma4 => "gemma4",
            ArchFamily::Qwen35 => "qwen35",
            ArchFamily::Qwen3VlText => "qwen3vl-text",
            ArchFamily::Deepseek4 => "deepseek4",
            ArchFamily::Llama4Reserved => "llama4",
        }
    }

    /// `true` iff this architecture supports a separate mmproj GGUF for
    /// vision (so the banner should distinguish "mmproj-required (none
    /// loaded)" from "n/a (text-only arch)").  Gemma4 ships as
    /// text+mmproj pairs; Qwen3-VL text LM has a companion ViT.  Pure
    /// text architectures (Qwen35) never carry vision.
    pub fn supports_mmproj(&self) -> bool {
        matches!(self, ArchFamily::Gemma4 | ArchFamily::Qwen3VlText)
    }
}

// ---------------------------------------------------------------------------
// LoadInfo — the snapshot
// ---------------------------------------------------------------------------

/// Unified per-load metadata snapshot.
///
/// All facts about a successful load that the unified banner, structured
/// tracing, and (eventually) the OpenAI-compatible `/v1/models` response
/// need.
///
/// Fields are grouped (and ordered) the same way the on-screen banner
/// renders them — Identity, Hardware, Architecture, Quantization,
/// Tokenizer, Provenance, Vision, Wall-clock — so a `Debug`-formatted
/// dump is readable as-is.
///
/// Field count and shape match the design doc spec at
/// `docs/research/model_load_ux_uniformity_2026-05-01.md` §5.1.
#[derive(Debug, Clone)]
pub struct LoadInfo {
    // ---------- Identity ----------
    /// `general.name` if present, else the file stem of `model_path`.
    /// Same shape as today's `LoadedModel::model_id()`.
    pub model_id: String,
    /// Raw GGUF `general.architecture` value (e.g. `"gemma4"`,
    /// `"qwen35"`, `"qwen35moe"`).
    pub arch_str: String,
    /// Coarse dispatch bucket — see [`ArchFamily`].
    pub arch_family: ArchFamily,
    /// Filesystem path to the GGUF actually opened.
    pub model_path: PathBuf,
    /// On-disk GGUF size in bytes, as reported by
    /// `std::fs::metadata(...).len()`.
    pub on_disk_bytes: u64,

    // ---------- Hardware / backend ----------
    /// `gpu.gpu_name()` — e.g. `"Apple M5 Max"` *before* `short_chip_label`
    /// strips the vendor prefix.
    pub backend_chip: String,
    /// Backend label — `"mlx-native"` today.  `&'static str` because the
    /// only legal value is compile-time fixed per ADR-008.
    pub backend: &'static str,

    // ---------- Architecture facts ----------
    /// Number of transformer layers actually resident.
    pub n_layers: u32,
    /// Hidden / model dimension.
    pub hidden_size: u32,
    /// Tokenizer vocabulary size.
    pub vocab_size: u32,
    /// Multi-head attention head count.
    pub n_attention_heads: u32,
    /// KV-head count — equals `n_attention_heads` for vanilla MHA, less
    /// for GQA.
    pub n_key_value_heads: u32,
    /// Per-head dimension.  Qwen3.5 stores it explicitly; Gemma4 derives
    /// it from `hidden_size / n_attention_heads`.
    pub head_dim: u32,
    /// Sliding-window size in tokens — `None` for full-attention-only
    /// models.  Gemma4 sets this; Qwen35 leaves it `None` and surfaces
    /// `full_attention_interval` instead.
    pub sliding_window: Option<u32>,
    /// Qwen35-specific: full-attention layer period (one full-attn layer
    /// every `n` layers).  `None` for arches without this concept.
    pub full_attention_interval: Option<u32>,
    /// Maximum context length declared by the GGUF
    /// (`{arch}.context_length`).
    pub max_context_length: Option<u32>,
    /// MoE shape if the model is MoE; `None` for dense.
    pub moe: Option<MoeShape>,

    // ---------- Quantization ----------
    /// Dominant non-fp tensor type (e.g. `"Q4_K"`, `"Q6_K"`) — produced
    /// by [`infer_quant_label`].  `None` for pure-fp models.
    pub quant_label: Option<String>,
    /// Bits-per-weight, parameter-weighted average across non-fp tensors
    /// — produced by [`compute_bpw`].  `None` if there are no non-fp
    /// tensors or if computation was skipped.
    pub quant_bpw: Option<f32>,

    // ---------- Tokenizer / chat template ----------
    /// Where the active tokenizer came from.
    pub tokenizer_source: TokenizerSource,
    /// Every token id treated as end-of-sequence by the engine.  Always
    /// non-empty in practice (at least `tokenizer.ggml.eos_token_id`).
    pub eos_token_ids: Vec<u32>,
    /// BOS token if the GGUF declared one
    /// (`tokenizer.ggml.bos_token_id`); `None` for tokenizers that don't.
    pub bos_token_id: Option<u32>,
    /// Where the chat template in effect came from.
    pub chat_template_source: ChatTemplateSource,

    // ---------- Provenance (ADR-017 §F4) ----------
    /// Mirrors `LoadedModel::provenance()`.  Populated for both arches
    /// in commit C2 below (today only Gemma populates a non-`External`
    /// value).
    pub provenance: Provenance,

    // ---------- Vision / multimodal ----------
    /// `Some` only if the operator passed `--mmproj` *and* the load path
    /// supports it for this arch.  Today: Gemma yes, Qwen35 no (vision
    /// path returns 501 — engine.rs:2192+).
    pub vision_projector: Option<VisionProjector>,

    // ---------- Wall-clock / memory ----------
    /// Wall-clock time spent in `*LoadedModel::load`, GPU-init through
    /// weights-resident.
    pub load_wall_clock: Duration,
    /// Best-effort post-load resident bytes for weights (ADR-017's
    /// `resident_bytes_weights`).  `None` if not measured (Qwen35 today).
    pub resident_weight_bytes: Option<u64>,
    /// KV-cache memory budget for the engine, derived from
    /// `EngineConfig` in SERVE mode and from `args.max_tokens` in CLI
    /// mode.  `None` for the CLI Gemma path (which lazily allocates
    /// per-prefill — current behaviour).
    pub kv_cache_budget_bytes: Option<u64>,

    // ---------- KV-persist / spill (ADR-017) ----------
    /// `true` iff the engine will bind a KV-spill hook for this load
    /// (i.e. `--kv-persist=PATH` is set AND a per-family factory matched).
    /// Always `false` for Qwen35 today — see ADR-017 Phase B-hybrid
    /// fence.
    pub kv_spill_active: bool,

    /// ADR-027 Phase B iter-17 — `true` iff this load will allocate
    /// TQ-active full-attn KV buffers (sourced from `HF2Q_TQ_KV` env
    /// at engine load via `is_tq_active_mode()`). When `true`, every
    /// per-prefill cache allocates ONLY the TQ-encoded buffers
    /// (`k_packed`/`k_norms`/`v_packed`/`v_norms` per full-attn slot);
    /// the F32 K/V backing is DROPPED at alloc time per iter-34
    /// (sub-iter 23c-β.5) for the **3.94× per-slot KV memory savings**
    /// vs the F32-only baseline. The decode SDPA dispatch routes
    /// through `flash_attn_vec_tq_hb`; the prefill resume SDPA dequants
    /// slot.tq → temp F32 (unrotated) and dispatches the same dense
    /// resume kernel via `apply_flash_attn_prefill_seq_major_resume_via_tq_cache`
    /// (iter-33). iter-16 validated coherence on real qwen36
    /// 35B-A3B-APEX-Q5_K_M (output BYTE-IDENTICAL to F32 baseline at
    /// ~128 tok/s); iter-43 extended byte-identity to 32K context.
    ///
    /// Currently surfaced only for `Qwen35` family; Gemma's TQ-on
    /// state is reflected in `kv_spill_active` via the
    /// `tq_packed_descriptor` engine binding (different mechanism;
    /// future iter may unify).
    pub tq_kv_active: bool,

    /// **ADR-040 §3.5 iter-A5c (cfa-A5b CRITICAL #1)** — exact per-token KV
    /// byte cost when the architecture is heterogeneous in shape across
    /// layers (e.g. Gemma 4: sliding layers use `num_key_value_heads ×
    /// head_dim`, full-attention layers use `num_global_key_value_heads
    /// × global_head_dim`).
    ///
    /// When `Some(n)`, [`Self::kv_bytes_per_token`] returns `n` verbatim
    /// instead of the flattened `n_layers × n_kv_heads × head_dim × 4 × 2`
    /// scalar formula. When `None`, the flattened formula is used (the
    /// homogeneous-architecture path: Qwen3.5/3.6 keep one
    /// `(n_kv_heads, head_dim)` shape across all layers).
    ///
    /// Builders populate this with `sum_over_layers(n_kv_heads_l ×
    /// head_dim_l × dtype_bytes × 2)`, computed against the architecture
    /// config's per-layer accessors (e.g. for Gemma 4 the configured
    /// `config.layer_types` + `config.num_kv_heads_for_layer(i)` +
    /// `config.head_dim_for_layer(i)` at `src/serve/config.rs:346-353`).
    ///
    /// Pre-iter-A5c, Gemma 4 flattened to the SLIDING shape (8×256) —
    /// for canonical 30-layer Gemma 4 27B with 5 full-attn layers
    /// (2×512) this OVER-counts by ~9% (61_440 vs exact 56_320 elements
    /// per token) — safe upper bound (false-rejects borderline
    /// requests) but not the operator-honest exact value the ADR
    /// promises. iter-A5c closes this gap.
    pub kv_bytes_per_token_override: Option<u64>,
}

impl LoadInfo {
    /// **ADR-040 §3.5 iter-A5b** — operator-honest upper bound on
    /// per-token KV bytes for this loaded model.
    ///
    /// The formula mirrors [`estimate_kv_tokens`] (used for the load
    /// banner), generalised to the production
    /// "K + V, F32-equivalent worst case" shape so the engine seam can
    /// compute `kv_bytes_needed = (prompt_tokens + max_tokens) *
    /// kv_bytes_per_token()` at admit time without needing per-arch
    /// model state.
    ///
    /// Per-token cost (bytes):
    ///
    /// ```text
    /// n_layers × n_kv_heads × head_dim × dtype_bytes × 2  (K and V)
    /// ```
    ///
    /// where `dtype_bytes = 4` (F32) is the conservative upper bound
    /// that matches the legacy [`estimate_kv_tokens`] formula. Real
    /// production KV may live in F16 (`HF2Q_F16_KV`), packed U8 (TQ,
    /// ADR-007), or hybrid Full/Sliding allocations (Gemma 4 sliding
    /// layers cap at `sliding_window`, not `max_seq_len`) — every one
    /// of those is at-most-equal to this F32-flat estimate. Using the
    /// upper bound means [`AdmitError::SlotBudgetExceeded`] rejects
    /// requests whose maximum possible KV would exceed the per-slot
    /// budget — operator-honest false-reject of borderline cases,
    /// never a false-accept that would surface as mid-decode OOM.
    ///
    /// Returns `0` when any of `n_layers`, `n_key_value_heads`, or
    /// `head_dim` is zero (the synthetic-fixture / test-loader path);
    /// callers MUST treat `0` as "do not enforce" (mirrors
    /// [`crate::serve::scheduler::AdmitRequest::kv_bytes_needed`]'s
    /// `0`-means-disabled semantics).
    ///
    /// # Why F32 not the actual KV dtype
    ///
    /// `LoadInfo` does not currently carry the per-layer KV dtype
    /// (TQ + hybrid arches have heterogeneous dtypes per layer — see
    /// `inference/models/gemma4/kv_cache.rs` `MlxKvCache` (TQ U8 +
    /// F32 norms) vs `DenseKvBuffers` (F16 or F32 per `HF2Q_F16_KV`)).
    /// Threading the per-layer dtype vector through the engine seam
    /// would re-litigate the bytes-vs-tokens decision documented at
    /// `serve/scheduler.rs` iter-A5 header. F32 is the largest dtype
    /// reachable in production today; using it as the upper bound is
    /// the simplest operator-honest choice that does not require
    /// per-arch math at the engine seam.
    ///
    /// # iter-A5c (cfa-A5b CRITICAL #1) — exact per-layer math
    ///
    /// For heterogeneous architectures (Gemma 4 sliding vs full layers
    /// have DIFFERENT `(n_kv_heads, head_dim)` shapes), the loader
    /// populates [`Self::kv_bytes_per_token_override`] with the EXACT
    /// per-layer sum and that value short-circuits the scalar formula
    /// below. Without the override we would over-count canonical Gemma
    /// 4 27B by ~9% (flattened 30×8×256 vs exact 25×8×256 + 5×2×512)
    /// — safe upper bound but not operator-honest exact.
    pub fn kv_bytes_per_token(&self) -> u64 {
        if let Some(exact) = self.kv_bytes_per_token_override {
            return exact;
        }
        let n_layers = u64::from(self.n_layers);
        let n_kv = u64::from(self.n_key_value_heads);
        let hd = u64::from(self.head_dim);
        if n_layers == 0 || n_kv == 0 || hd == 0 {
            return 0;
        }
        // F32 (4 bytes) × 2 (K and V) — conservative upper bound.
        n_layers
            .saturating_mul(n_kv)
            .saturating_mul(hd)
            .saturating_mul(4)
            .saturating_mul(2)
    }

    /// **ADR-040 §3.5 iter-A5b** — compute the KV byte cost a request
    /// of `(prompt_tokens + max_tokens)` tokens would put on a single
    /// physical KV slot. Returns `0` when [`Self::kv_bytes_per_token`]
    /// is `0` (synthetic loader / missing arch facts) — caller treats
    /// `0` as "do not enforce" per the scheduler's
    /// `kv_bytes_needed: 0` opt-out contract.
    ///
    /// Saturating-arithmetic throughout — a u32×u32 multiply that
    /// overflows surfaces as `u64::MAX`, which the per-slot budget
    /// check trivially rejects (the operator-actionable failure mode
    /// — caller is asking for absurd seq_len).
    pub fn kv_bytes_for_request(&self, prompt_tokens: u32, max_tokens: u32) -> u64 {
        let per_token = self.kv_bytes_per_token();
        if per_token == 0 {
            return 0;
        }
        let total_tokens = u64::from(prompt_tokens).saturating_add(u64::from(max_tokens));
        total_tokens.saturating_mul(per_token)
    }
}

/// **ADR-040 §3.5 iter-A5c (cfa-A5b CRITICAL #1)** — exact per-token KV byte
/// cost for a Gemma 4 config, summed across the heterogeneous per-layer
/// `(num_kv_heads, head_dim)` shape.
///
/// Gemma 4 carries two distinct KV shapes:
/// - **Sliding** layers: `num_key_value_heads × head_dim` (canonical
///   27B: 8 × 256).
/// - **Full**-attention layers: `num_global_key_value_heads ×
///   global_head_dim` (canonical 27B: 2 × 512).
///
/// The flattened scalar formula at [`LoadInfo::kv_bytes_per_token`] uses
/// only the sliding shape (the values stored on `LoadInfo` today via
/// `GemmaLoadedModel::build_load_info`), over-counting canonical 30-layer
/// Gemma 4 27B by ~9% (61_440 vs exact 56_320 elements per token). The
/// over-count is a SAFE upper bound (false-rejects borderline requests,
/// never under-counts → no false-accept that would surface as mid-decode
/// OOM), but ADR-040 §3.5 promises the EXACT per-token cost so the engine
/// seam admit-time check matches the actual KV allocation shape — that's
/// what this helper computes from
/// [`crate::serve::config::Gemma4Config::layer_types`] +
/// `num_kv_heads_for_layer` + `head_dim_for_layer`.
///
/// The dtype used is F32 (4 bytes) for K and V — same conservative dtype
/// upper bound as the flattened formula, mirroring the rationale at
/// [`LoadInfo::kv_bytes_per_token`] (`LoadInfo` does not carry per-layer
/// KV dtype; TQ-packed slots actually use U8 + F32 norms but the F32-
/// equivalent worst case keeps the engine seam dtype-agnostic and never
/// false-accepts on a TQ-on load that later drops to F32 for a
/// non-TQ-eligible layer).
///
/// Saturating-arithmetic throughout — see
/// [`LoadInfo::kv_bytes_for_request`] for the symmetric overflow story.
pub fn gemma4_exact_kv_bytes_per_token(cfg: &crate::serve::config::Gemma4Config) -> u64 {
    use crate::serve::config::LayerType;
    let mut total: u64 = 0;
    for (i, layer_type) in cfg.layer_types.iter().enumerate() {
        // Defense-in-depth: the per-layer accessors derive from
        // `layer_types[i]`, but pinning the branch here keeps the
        // intent local to this helper.
        let (nkv, hd) = match layer_type {
            LayerType::Sliding => (cfg.num_key_value_heads as u64, cfg.head_dim as u64),
            LayerType::Full => (
                cfg.num_global_key_value_heads as u64,
                cfg.global_head_dim as u64,
            ),
        };
        // Cross-check against the public per-layer accessors — if the
        // per-layer accessor ever diverges from the (LayerType ⇒ shape)
        // mapping above, the debug_assert! catches the drift before it
        // can corrupt the budget calculation. In release builds the
        // helper still uses the local mapping (the public accessor IS
        // the same logic per `src/serve/config.rs:346-353`).
        debug_assert_eq!(
            cfg.num_kv_heads_for_layer(i) as u64,
            nkv,
            "L{i}: num_kv_heads_for_layer drift vs LayerType mapping"
        );
        debug_assert_eq!(
            cfg.head_dim_for_layer(i) as u64,
            hd,
            "L{i}: head_dim_for_layer drift vs LayerType mapping"
        );
        // Per-layer per-token bytes: nkv × hd × 4 (F32) × 2 (K + V).
        total = total.saturating_add(nkv.saturating_mul(hd).saturating_mul(4).saturating_mul(2));
    }
    total
}

/// Implemented by each loaded-model variant to produce the shared load
/// snapshot from owned model state plus the still-open GGUF metadata.
pub trait LoadInfoBuilder {
    fn build_load_info(
        &self,
        gguf: &mlx_native::gguf::GgufFile,
        load_wall_clock: std::time::Duration,
        kv_cache_budget_bytes: Option<u64>,
        kv_spill_active: bool,
    ) -> LoadInfo;
}

pub(crate) fn model_id_from_gguf(gguf: &mlx_native::gguf::GgufFile, model_path: &Path) -> String {
    gguf.metadata_string("general.name")
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            model_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_string())
        })
}

pub(crate) fn arch_str_from_gguf(gguf: &mlx_native::gguf::GgufFile) -> String {
    gguf.metadata_string("general.architecture")
        .unwrap_or("unknown")
        .to_string()
}

pub(crate) fn on_disk_bytes(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

pub(crate) fn chat_template_source(
    gguf: &mlx_native::gguf::GgufFile,
    fallback_name: Option<&'static str>,
) -> ChatTemplateSource {
    if gguf.metadata_string("tokenizer.chat_template").is_some() {
        ChatTemplateSource::GgufEmbedded
    } else if let Some(name) = fallback_name {
        ChatTemplateSource::HardcodedFallback { name }
    } else {
        ChatTemplateSource::None
    }
}

/// Estimate training-time dense KV token capacity for the banner.  This is
/// only a display hint; callers still pass the authoritative byte budget.
fn estimate_kv_tokens(info: &LoadInfo) -> Option<u64> {
    let budget = info.kv_cache_budget_bytes?;
    let per_token = u64::from(info.n_layers)
        .checked_mul(u64::from(info.n_key_value_heads))?
        .checked_mul(u64::from(info.head_dim))?
        .checked_mul(4)?;
    if per_token == 0 {
        return None;
    }
    Some(budget / per_token)
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

fn fmt_gib(bytes: u64) -> String {
    format!("{:.2} GiB", gib(bytes))
}

fn fmt_opt_u32(v: Option<u32>) -> String {
    v.map(|v| v.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn fmt_opt_bytes(v: Option<u64>) -> String {
    v.map(fmt_gib).unwrap_or_else(|| "none".to_string())
}

fn fmt_tokenizer_source(source: &TokenizerSource) -> String {
    match source {
        TokenizerSource::HfTokenizerJson { path } => {
            format!("hf-tokenizer-json ({})", path.display())
        }
        TokenizerSource::GgufEmbedded => "gguf-embedded (<= mirrors llama-vocab.cpp)".to_string(),
    }
}

fn fmt_chat_template_source(source: &ChatTemplateSource) -> String {
    match source {
        ChatTemplateSource::GgufEmbedded => "gguf-embedded".to_string(),
        ChatTemplateSource::CliOverride => "cli-override".to_string(),
        ChatTemplateSource::HardcodedFallback { name } => format!("hardcoded-fallback ({name})"),
        ChatTemplateSource::NativeEncoding { name } => format!("native-encoding ({name})"),
        ChatTemplateSource::None => "none".to_string(),
    }
}

fn fmt_moe(moe: Option<MoeShape>) -> String {
    moe.map(|m| format!("{} experts/{} active", m.n_experts, m.n_experts_per_tok))
        .unwrap_or_else(|| "none".to_string())
}

fn fmt_quant(info: &LoadInfo) -> String {
    let label = info.quant_label.as_deref().unwrap_or("none");
    let bpw = info
        .quant_bpw
        .map(|v| format!("~{v:.2} bpw"))
        .unwrap_or_else(|| "none".to_string());
    let resident = fmt_opt_bytes(info.resident_weight_bytes);
    if label == "none" {
        format!("none dominant, {bpw}, mlx-native resident {resident}")
    } else {
        format!("{label} dominant, {bpw}, mlx-native resident {resident}")
    }
}

fn fmt_provenance(provenance: &Provenance) -> String {
    match provenance {
        Provenance::External => "external".to_string(),
        Provenance::Hf2q {
            producer_version,
            source_sha256,
            ..
        } => {
            let prefix: String = source_sha256.chars().take(4).collect();
            format!("hf2q (producer {producer_version}, source_sha {prefix}…)")
        }
    }
}

fn fmt_vision(arch: ArchFamily, vision: &Option<VisionProjector>) -> String {
    match vision {
        Some(v) => {
            let sha = v.mmproj_sha256.as_deref().unwrap_or("none");
            format!("{} (sha256 {sha})", v.mmproj_path.display())
        }
        None if arch.supports_mmproj() => {
            "mmproj-required (no mmproj loaded; pass --mmproj)".to_string()
        }
        None => "n/a (text-only arch)".to_string(),
    }
}

fn fmt_kv_budget(info: &LoadInfo) -> String {
    match info.kv_cache_budget_bytes {
        Some(bytes) => match estimate_kv_tokens(info) {
            Some(tokens) => format!("{} (~{} tokens)", fmt_gib(bytes), tokens),
            None => fmt_gib(bytes),
        },
        None => "none".to_string(),
    }
}

/// Print the unified 13-line load banner.
pub fn print_banner<W: std::io::Write>(
    info: &LoadInfo,
    w: &mut W,
    tty: bool,
) -> std::io::Result<()> {
    let (d, r) = if tty { (DIM, RESET) } else { ("", "") };
    writeln!(
        w,
        "{d}hf2q load: backend = {} ({}){r}",
        info.backend,
        crate::serve::header::short_chip_label(&info.backend_chip)
    )?;
    writeln!(
        w,
        "{d}hf2q load: model = {} (arch = {}, family = {}){r}",
        info.model_id,
        info.arch_str,
        info.arch_family.as_str()
    )?;
    writeln!(
        w,
        "{d}hf2q load: source = {} ({} on disk){r}",
        info.model_path.display(),
        fmt_gib(info.on_disk_bytes)
    )?;
    writeln!(
        w,
        "{d}hf2q load: layout = {} layers, {} heads ({} kv), head_dim={}, hidden={}, vocab={}{r}",
        info.n_layers,
        info.n_attention_heads,
        info.n_key_value_heads,
        info.head_dim,
        info.hidden_size,
        info.vocab_size
    )?;
    writeln!(
        w,
        "{d}hf2q load: features = sliding_window={}, full_attn_every={}, moe={}{r}",
        fmt_opt_u32(info.sliding_window),
        fmt_opt_u32(info.full_attention_interval),
        fmt_moe(info.moe)
    )?;
    writeln!(w, "{d}hf2q load: quant = {}{r}", fmt_quant(info))?;
    writeln!(
        w,
        "{d}hf2q load: max_ctx_train = {}, kv_budget = {}{r}",
        fmt_opt_u32(info.max_context_length),
        fmt_kv_budget(info)
    )?;
    writeln!(
        w,
        "{d}hf2q load: tokenizer = {}{r}",
        fmt_tokenizer_source(&info.tokenizer_source)
    )?;
    writeln!(
        w,
        "{d}hf2q load: chat_template = {}{r}",
        fmt_chat_template_source(&info.chat_template_source)
    )?;
    writeln!(
        w,
        "{d}hf2q load: provenance = {}{r}",
        fmt_provenance(&info.provenance)
    )?;
    writeln!(
        w,
        "{d}hf2q load: vision = {}{r}",
        fmt_vision(info.arch_family, &info.vision_projector)
    )?;
    writeln!(
        w,
        "{d}hf2q load: kv_spill = {}{r}",
        if info.kv_spill_active {
            "active"
        } else {
            "inactive"
        }
    )?;
    // ADR-027 Phase B iter-17 — surface TQ-on state to operators so
    // HF2Q_TQ_KV=1 loads are visibly distinct from the F32 default at
    // load time. Format mirrors `kv_spill = {active|inactive}` for
    // grep symmetry. iter-48 (sub-iter 23e+1) extends the active
    // string with the iter-34 memory-savings ratio so operators see
    // the realized 3.94× savings at load time without having to
    // consult the ADR.
    //
    // 2026-05-23: extended to also surface Gemma 4's TQ-on state
    // (ADR-007 Path C close). On Gemma 4, TQ is the production
    // default (8-bit Lloyd-Max + D1 SRHT via `flash_attn_vec_tq_hb`
    // / `flash_attn_vec_tq`); inactive only when the operator opts
    // out via `HF2Q_USE_DENSE=1` or `HF2Q_LAYER_POLICY=dense_all`.
    // The active-string text branches on `arch_family` so the
    // Qwen35-specific "ADR-027 Phase B; 3.94×" callout doesn't leak
    // onto the Gemma 4 banner.
    let tq_kv_text: &str = if info.tq_kv_active {
        match info.arch_family {
            ArchFamily::Qwen35 => "active (8-bit Lloyd-Max + D1 SRHT, ADR-027 Phase B; F32 K/V dropped at alloc — 3.94× per-slot KV savings)",
            ArchFamily::Gemma4 => "active (8-bit Lloyd-Max + D1 SRHT, ADR-007 Path C; production default; HF2Q_USE_DENSE=1 to opt out)",
            _ => "active (8-bit Lloyd-Max + D1 SRHT)",
        }
    } else {
        "inactive"
    };
    writeln!(w, "{d}hf2q load: tq_kv = {}{r}", tq_kv_text)?;
    writeln!(
        w,
        "{d}hf2q load: ready in {:.2} s{r}",
        info.load_wall_clock.as_secs_f64()
    )?;
    w.flush()
}

/// Emit the banner facts as structured tracing events. Field names match
/// `LoadInfo` exactly so JSON logs remain grep- and diff-friendly.
pub fn emit_tracing(info: &LoadInfo) {
    tracing::info!(model_id = %info.model_id);
    tracing::info!(arch_str = %info.arch_str);
    tracing::info!(arch_family = %info.arch_family.as_str());
    tracing::info!(model_path = %info.model_path.display());
    tracing::info!(on_disk_bytes = info.on_disk_bytes);
    tracing::info!(backend_chip = %info.backend_chip, backend = info.backend);
    tracing::info!(
        n_layers = info.n_layers,
        hidden_size = info.hidden_size,
        vocab_size = info.vocab_size,
        n_attention_heads = info.n_attention_heads,
        n_key_value_heads = info.n_key_value_heads,
        head_dim = info.head_dim
    );
    tracing::info!(
        sliding_window = ?info.sliding_window,
        full_attention_interval = ?info.full_attention_interval,
        max_context_length = ?info.max_context_length,
        moe = ?info.moe
    );
    tracing::info!(quant_label = ?info.quant_label, quant_bpw = ?info.quant_bpw);
    tracing::info!(
        tokenizer_source = ?info.tokenizer_source,
        eos_token_ids = ?info.eos_token_ids,
        bos_token_id = ?info.bos_token_id,
        chat_template_source = ?info.chat_template_source
    );
    tracing::info!(provenance = ?info.provenance);
    tracing::info!(vision_projector = ?info.vision_projector);
    tracing::info!(
        load_wall_clock = ?info.load_wall_clock,
        resident_weight_bytes = ?info.resident_weight_bytes,
        kv_cache_budget_bytes = ?info.kv_cache_budget_bytes,
        kv_spill_active = info.kv_spill_active
    );
}

// ---------------------------------------------------------------------------
// Helpers — derivation from an open GGUF
// ---------------------------------------------------------------------------

/// Dominant non-fp tensor-type label.
///
/// Builds a histogram of `GgmlType` variants over every tensor in the
/// open GGUF (skipping `F32` / `F16`) and returns the label with the
/// largest count, ties broken by `HashMap` iteration order (the legacy
/// behaviour preserved verbatim from
/// `engine.rs::infer_quant_type_from_gguf` — kept as-is so the C1
/// migration introduces no observable behaviour delta).
///
/// Returns `None` for pure-fp GGUFs (every tensor is `F32` or `F16`) and
/// for empty GGUFs.
///
/// # Why this lives here
///
/// Two byte-identical copies of this 27-LOC body shipped in
/// `engine.rs:3148-3174` and `engine_qwen35.rs:246-272` prior to C1.  The
/// design doc §5.7 promotes both call sites to this single home so that
/// future arches (Llama4 reserved) inherit it for free, and so the
/// histogram algorithm can grow atomically (e.g. to surface BPW alongside
/// the label) without re-syncing two private copies.
pub fn infer_quant_label(gguf: &mlx_native::gguf::GgufFile) -> Option<String> {
    use mlx_native::GgmlType;
    use std::collections::HashMap;

    let mut histogram: HashMap<&'static str, usize> = HashMap::new();
    for name in gguf.tensor_names() {
        let Some(info) = gguf.tensor_info(name) else {
            continue;
        };
        if matches!(info.ggml_type, GgmlType::F32 | GgmlType::F16) {
            continue;
        }
        let label = match info.ggml_type {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::Q4_0 => "Q4_0",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q2_K => "Q2_K",
            GgmlType::Q4_K => "Q4_K",
            GgmlType::Q5_K => "Q5_K",
            GgmlType::Q6_K => "Q6_K",
            GgmlType::I16 => "I16",
            GgmlType::I32 => "I32",
            GgmlType::Q5_1 => "Q5_1",
            GgmlType::IQ4_NL => "IQ4_NL",
            GgmlType::IQ4_XS => "IQ4_XS",
        };
        *histogram.entry(label).or_insert(0) += 1;
    }
    histogram
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(k, _)| k.to_string())
}

/// Parameter-weighted bits-per-weight, averaged across all non-fp
/// tensors in the open GGUF.
///
/// # Algorithm
///
/// For each tensor whose `ggml_type` is *not* `F32` or `F16`:
///   1. `n_elements = shape.iter().product::<usize>()` — total elements
///      stored.
///   2. `block_count = n_elements / block_values` — must be exact (GGUF
///      enforces shape-block alignment at parse time via
///      `compute_byte_len`; we re-verify here defensively, returning
///      `None` rather than panicking on a malformed file).
///   3. `tensor_bytes = block_count * block_bytes`.
///   4. Accumulate into `total_elements` and `total_bytes`.
///
/// Final BPW = `(total_bytes * 8) / total_elements`.  Returns `None` if
/// no non-fp tensors were seen (pure-fp GGUF) or if any tensor's element
/// count was not block-aligned.
///
/// # Why parameter-weighted, not type-weighted
///
/// The naive approach — average BPW across distinct types — would
/// overweight a single small int-token-type-id tensor.  We weight by
/// element count so the headline number reflects the dominant
/// contributor to the file's quantized footprint, the same way
/// llama.cpp's `llm_load_print_meta` reports a single BPW for the load.
///
/// # Closed-form sanity
///
/// For a single-type GGUF, the result equals the type's intrinsic
/// `block_bytes * 8 / block_values`:
///   * Q4_K → 144 × 8 / 256 = 4.5 bpw exactly
///   * Q6_K → 210 × 8 / 256 = 6.5625 bpw exactly
///   * Q4_0 → 18 × 8 / 32 = 4.5 bpw
///   * Q8_0 → 34 × 8 / 32 = 8.5 bpw
pub fn compute_bpw(gguf: &mlx_native::gguf::GgufFile) -> Option<f32> {
    use mlx_native::GgmlType;

    let mut total_elements: u128 = 0;
    let mut total_bytes: u128 = 0;

    for name in gguf.tensor_names() {
        let Some(info) = gguf.tensor_info(name) else {
            continue;
        };
        if matches!(info.ggml_type, GgmlType::F32 | GgmlType::F16) {
            continue;
        }

        let n_elements: usize = info.shape.iter().product();
        if n_elements == 0 {
            continue;
        }

        let block_values = info.ggml_type.block_values() as usize;
        let block_bytes = info.ggml_type.block_bytes() as usize;
        if block_values == 0 {
            // Defensive: every legal GgmlType has block_values >= 1.
            return None;
        }
        if n_elements % block_values != 0 {
            // GGUF parse already validated shape divisibility (see
            // `mlx_native::gguf::compute_byte_len`).  Reaching this branch
            // would mean a TensorInfo invariant was violated — bail out
            // rather than emit a misleading BPW.
            return None;
        }
        let block_count = n_elements / block_values;
        let tensor_bytes = block_count.checked_mul(block_bytes)?;

        total_elements = total_elements.checked_add(n_elements as u128)?;
        total_bytes = total_bytes.checked_add(tensor_bytes as u128)?;
    }

    if total_elements == 0 {
        return None;
    }

    Some((total_bytes as f64 * 8.0 / total_elements as f64) as f32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Synthetic-GGUF unit tests.  We build minimal GGUF v3 byte streams
    //! in a tempdir, open them via `mlx_native::gguf::GgufFile::open`,
    //! and run the helpers against the parsed file.
    //!
    //! Why not mock the GGUF interface directly: `GgufFile` has private
    //! fields and no public constructor for unit tests, so a real
    //! parse-the-bytes round-trip is the cheapest path.  The synthetic
    //! GGUF builder below mirrors the proven pattern in
    //! `mlx-native/tests/test_gguf_load_tensor_into_pool.rs::write_minimal_f32_gguf`.

    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    // GGML type IDs (must match `mlx_native::gguf::GGML_TYPE_*`).
    const GGML_TYPE_F32: u32 = 0;
    const GGML_TYPE_F16: u32 = 1;
    const GGML_TYPE_Q4_K: u32 = 12;
    const GGML_TYPE_Q6_K: u32 = 14;
    const GGML_TYPE_Q8_0: u32 = 8;

    // Block constants mirroring `GgmlType::block_values()` /
    // `block_bytes()`.
    const BLOCK_VALUES_Q4_K: usize = 256;
    const BLOCK_BYTES_Q4_K: usize = 144;
    const BLOCK_VALUES_Q6_K: usize = 256;
    const BLOCK_BYTES_Q6_K: usize = 210;
    const BLOCK_VALUES_Q8_0: usize = 32;
    const BLOCK_BYTES_Q8_0: usize = 34;

    /// Description of a single tensor for `write_synthetic_gguf`.
    struct TensorSpec {
        name: &'static str,
        /// Shape (innermost dimension first, GGUF storage order).
        shape: Vec<usize>,
        ggml_type_id: u32,
        /// Total bytes the tensor occupies in the data section.  Caller
        /// pre-computes (n_elements / block_values) * block_bytes.
        byte_len: usize,
    }

    enum KvSpec {
        String(&'static str, &'static str),
        U32(&'static str, u32),
    }

    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Write a minimal GGUF v3 file with zero metadata KV pairs and the
    /// given tensors.  Tensor data is filled with zeros — every helper
    /// under test inspects metadata only, not values.
    fn write_synthetic_gguf(path: &Path, tensors: &[TensorSpec]) {
        write_synthetic_gguf_with_metadata(path, &[], tensors);
    }

    fn write_synthetic_gguf_with_metadata(path: &Path, kvs: &[KvSpec], tensors: &[TensorSpec]) {
        let mut buf: Vec<u8> = Vec::new();

        // Header: magic, version, n_tensors, n_kv.
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());

        for kv in kvs {
            match kv {
                KvSpec::String(key, value) => {
                    write_gguf_string(&mut buf, key);
                    buf.extend_from_slice(&8u32.to_le_bytes());
                    write_gguf_string(&mut buf, value);
                }
                KvSpec::U32(key, value) => {
                    write_gguf_string(&mut buf, key);
                    buf.extend_from_slice(&4u32.to_le_bytes());
                    buf.extend_from_slice(&value.to_le_bytes());
                }
            }
        }

        // Tensor info entries.  Offsets are relative to the tensor-data
        // base; we lay tensors out back-to-back with no inter-tensor
        // padding (the loader doesn't require any per-tensor alignment
        // beyond the data-section base).
        let mut data_offset: u64 = 0;
        for t in tensors {
            write_gguf_string(&mut buf, t.name);

            buf.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
            for &d in &t.shape {
                buf.extend_from_slice(&(d as u64).to_le_bytes());
            }
            buf.extend_from_slice(&t.ggml_type_id.to_le_bytes());
            buf.extend_from_slice(&data_offset.to_le_bytes());

            data_offset += t.byte_len as u64;
        }

        // Pad to the GGUF default alignment (32 bytes).
        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        // Tensor data — all zeros.
        let total_data: usize = tensors.iter().map(|t| t.byte_len).sum();
        buf.extend(std::iter::repeat(0u8).take(total_data));

        let mut f = File::create(path).expect("create synthetic gguf");
        f.write_all(&buf).expect("write synthetic gguf");
        f.flush().expect("flush synthetic gguf");
    }

    /// Helper: temp path unique per test name + pid so parallel tests
    /// don't collide.
    fn tmp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "load_info_test_{}_{}_{}.gguf",
            label,
            std::process::id(),
            // Nanosecond suffix: under cargo's parallel-test runner two
            // tests can share `test_name + pid` in the rare case of a
            // re-run with cached binaries.
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ))
    }

    // ── arch_family.as_str ─────────────────────────────────────────────

    #[test]
    fn arch_family_as_str_is_stable() {
        assert_eq!(ArchFamily::Gemma4.as_str(), "gemma4");
        assert_eq!(ArchFamily::Qwen35.as_str(), "qwen35");
        assert_eq!(ArchFamily::Llama4Reserved.as_str(), "llama4");
    }

    // ── infer_quant_label ──────────────────────────────────────────────

    #[test]
    fn infer_quant_label_q4k_dominant() {
        // 3 Q4_K tensors + 1 Q6_K tensor → Q4_K wins by count.
        let path = tmp_path("q4k_dominant");
        let tensors = vec![
            TensorSpec {
                name: "blk.0.attn_q.weight",
                shape: vec![BLOCK_VALUES_Q4_K, 4],
                ggml_type_id: GGML_TYPE_Q4_K,
                byte_len: 4 * BLOCK_BYTES_Q4_K,
            },
            TensorSpec {
                name: "blk.0.attn_k.weight",
                shape: vec![BLOCK_VALUES_Q4_K, 4],
                ggml_type_id: GGML_TYPE_Q4_K,
                byte_len: 4 * BLOCK_BYTES_Q4_K,
            },
            TensorSpec {
                name: "blk.0.attn_v.weight",
                shape: vec![BLOCK_VALUES_Q4_K, 4],
                ggml_type_id: GGML_TYPE_Q4_K,
                byte_len: 4 * BLOCK_BYTES_Q4_K,
            },
            TensorSpec {
                name: "output.weight",
                shape: vec![BLOCK_VALUES_Q6_K, 4],
                ggml_type_id: GGML_TYPE_Q6_K,
                byte_len: 4 * BLOCK_BYTES_Q6_K,
            },
        ];
        write_synthetic_gguf(&path, &tensors);

        let gguf = mlx_native::gguf::GgufFile::open(&path).expect("open synthetic gguf");
        assert_eq!(infer_quant_label(&gguf), Some("Q4_K".to_string()));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn infer_quant_label_returns_none_for_pure_fp() {
        let path = tmp_path("pure_fp");
        let tensors = vec![
            TensorSpec {
                name: "norm.weight",
                shape: vec![64],
                ggml_type_id: GGML_TYPE_F32,
                byte_len: 64 * 4,
            },
            TensorSpec {
                name: "output_norm.weight",
                shape: vec![64],
                ggml_type_id: GGML_TYPE_F16,
                byte_len: 64 * 2,
            },
        ];
        write_synthetic_gguf(&path, &tensors);

        let gguf = mlx_native::gguf::GgufFile::open(&path).expect("open synthetic gguf");
        assert_eq!(infer_quant_label(&gguf), None);

        let _ = std::fs::remove_file(&path);
    }

    /// Golden test: assert byte-identity with the legacy private body
    /// from `engine_qwen35.rs:246-272`.  We re-implement the legacy fn
    /// inline and run both against the same synthetic GGUF, asserting
    /// equality.  This pins H1: the C1 relocation introduces zero
    /// behaviour delta.
    #[test]
    fn infer_quant_label_matches_legacy_body() {
        // Inline copy of the legacy fn body (verbatim from
        // engine_qwen35.rs:246-272 prior to C1).  Kept here for the
        // duration of the C1 migration; once C2 lands, downstream code
        // either trusts `load_info::infer_quant_label` directly or
        // re-pins through this golden.
        fn legacy(gguf: &mlx_native::gguf::GgufFile) -> Option<String> {
            use mlx_native::GgmlType;
            use std::collections::HashMap;

            let mut histogram: HashMap<&'static str, usize> = HashMap::new();
            for name in gguf.tensor_names() {
                let Some(info) = gguf.tensor_info(name) else {
                    continue;
                };
                if matches!(info.ggml_type, GgmlType::F32 | GgmlType::F16) {
                    continue;
                }
                let label = match info.ggml_type {
                    GgmlType::F32 => "F32",
                    GgmlType::F16 => "F16",
                    GgmlType::Q4_0 => "Q4_0",
                    GgmlType::Q8_0 => "Q8_0",
                    GgmlType::Q2_K => "Q2_K",
                    GgmlType::Q4_K => "Q4_K",
                    GgmlType::Q5_K => "Q5_K",
                    GgmlType::Q6_K => "Q6_K",
                    GgmlType::I16 => "I16",
                    GgmlType::I32 => "I32",
                    GgmlType::Q5_1 => "Q5_1",
                    GgmlType::IQ4_NL => "IQ4_NL",
                    GgmlType::IQ4_XS => "IQ4_XS",
                };
                *histogram.entry(label).or_insert(0) += 1;
            }
            histogram
                .into_iter()
                .max_by_key(|(_, n)| *n)
                .map(|(k, _)| k.to_string())
        }

        // Mixed-type fixture: every variant the legacy match arm enumerates
        // (excluding F32/F16 which are skipped, and Q5_K which is a valid
        // GGML type id we want exercised).  Use sizes that satisfy the
        // GGUF block-alignment invariant.
        let path = tmp_path("legacy_match");
        let tensors = vec![
            TensorSpec {
                name: "a",
                shape: vec![BLOCK_VALUES_Q4_K, 2],
                ggml_type_id: GGML_TYPE_Q4_K,
                byte_len: 2 * BLOCK_BYTES_Q4_K,
            },
            TensorSpec {
                name: "b",
                shape: vec![BLOCK_VALUES_Q4_K],
                ggml_type_id: GGML_TYPE_Q4_K,
                byte_len: BLOCK_BYTES_Q4_K,
            },
            TensorSpec {
                name: "c",
                shape: vec![BLOCK_VALUES_Q6_K],
                ggml_type_id: GGML_TYPE_Q6_K,
                byte_len: BLOCK_BYTES_Q6_K,
            },
            TensorSpec {
                name: "d",
                shape: vec![BLOCK_VALUES_Q8_0, 4],
                ggml_type_id: GGML_TYPE_Q8_0,
                byte_len: 4 * BLOCK_BYTES_Q8_0,
            },
            TensorSpec {
                name: "norm",
                shape: vec![32],
                ggml_type_id: GGML_TYPE_F32,
                byte_len: 32 * 4,
            },
        ];
        write_synthetic_gguf(&path, &tensors);

        let gguf = mlx_native::gguf::GgufFile::open(&path).expect("open synthetic gguf");
        assert_eq!(infer_quant_label(&gguf), legacy(&gguf));
        // Sanity check: the dominant type by count is Q4_K (2 vs 1).
        assert_eq!(infer_quant_label(&gguf), Some("Q4_K".to_string()));

        let _ = std::fs::remove_file(&path);
    }

    // ── compute_bpw ────────────────────────────────────────────────────

    #[test]
    fn compute_bpw_pure_q4k() {
        // Single Q4_K tensor of 256 elements → BPW must equal exactly
        // 4.5 (144 × 8 / 256).
        let path = tmp_path("pure_q4k");
        let tensors = vec![TensorSpec {
            name: "w",
            shape: vec![BLOCK_VALUES_Q4_K],
            ggml_type_id: GGML_TYPE_Q4_K,
            byte_len: BLOCK_BYTES_Q4_K,
        }];
        write_synthetic_gguf(&path, &tensors);

        let gguf = mlx_native::gguf::GgufFile::open(&path).expect("open synthetic gguf");
        let bpw = compute_bpw(&gguf).expect("non-empty quant set");
        assert!(
            (bpw - 4.5).abs() < 0.01,
            "expected ~4.5 bpw for pure Q4_K, got {bpw}"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn compute_bpw_pure_q6k() {
        // Single Q6_K tensor of 256 elements → BPW must equal exactly
        // 6.5625 (210 × 8 / 256).
        let path = tmp_path("pure_q6k");
        let tensors = vec![TensorSpec {
            name: "w",
            shape: vec![BLOCK_VALUES_Q6_K],
            ggml_type_id: GGML_TYPE_Q6_K,
            byte_len: BLOCK_BYTES_Q6_K,
        }];
        write_synthetic_gguf(&path, &tensors);

        let gguf = mlx_native::gguf::GgufFile::open(&path).expect("open synthetic gguf");
        let bpw = compute_bpw(&gguf).expect("non-empty quant set");
        assert!(
            (bpw - 6.5625).abs() < 0.01,
            "expected ~6.5625 bpw for pure Q6_K, got {bpw}"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn compute_bpw_mixed_types() {
        // 1 Q4_K (256 elts, 144 B) + 1 Q8_0 (32 elts, 34 B)
        //   total_bytes  = 144 + 34 = 178
        //   total_elts   = 256 + 32 = 288
        //   expected bpw = 178 × 8 / 288 ≈ 4.9444…
        let path = tmp_path("mixed");
        let tensors = vec![
            TensorSpec {
                name: "q4k",
                shape: vec![BLOCK_VALUES_Q4_K],
                ggml_type_id: GGML_TYPE_Q4_K,
                byte_len: BLOCK_BYTES_Q4_K,
            },
            TensorSpec {
                name: "q8_0",
                shape: vec![BLOCK_VALUES_Q8_0],
                ggml_type_id: GGML_TYPE_Q8_0,
                byte_len: BLOCK_BYTES_Q8_0,
            },
        ];
        write_synthetic_gguf(&path, &tensors);

        let gguf = mlx_native::gguf::GgufFile::open(&path).expect("open synthetic gguf");
        let bpw = compute_bpw(&gguf).expect("non-empty quant set");
        let expected = (178.0 * 8.0) / 288.0; // ≈ 4.94444...
                                              // ±5% tolerance per design-doc spec line 754; bpw is exact-by-
                                              // construction here so the test would pass at ±0.001 too.
        assert!(
            (bpw - expected).abs() / expected < 0.05,
            "expected ~{expected:.4} bpw, got {bpw}"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn compute_bpw_returns_none_for_no_quant_tensors() {
        // Pure-fp GGUF: every helper must classify it as "no quantized
        // weight" and return None.
        let path = tmp_path("no_quant");
        let tensors = vec![
            TensorSpec {
                name: "norm.weight",
                shape: vec![64],
                ggml_type_id: GGML_TYPE_F32,
                byte_len: 64 * 4,
            },
            TensorSpec {
                name: "embd.weight",
                shape: vec![32, 4],
                ggml_type_id: GGML_TYPE_F16,
                byte_len: 128 * 2,
            },
        ];
        write_synthetic_gguf(&path, &tensors);

        let gguf = mlx_native::gguf::GgufFile::open(&path).expect("open synthetic gguf");
        assert_eq!(compute_bpw(&gguf), None);

        let _ = std::fs::remove_file(&path);
    }

    // ── Smoke: types compile + `Debug + Clone` derive holds ───────────

    /// Constructibility smoke: build a `LoadInfo` from minimal facts and
    /// round-trip through `Clone` + `Debug`.  This catches accidental
    /// breakage of the `derive(Debug, Clone)` contract that downstream
    /// tracing relies on.
    #[test]
    fn load_info_struct_compiles_and_clones() {
        let info = LoadInfo {
            model_id: "test-model".to_string(),
            arch_str: "qwen35".to_string(),
            arch_family: ArchFamily::Qwen35,
            model_path: PathBuf::from("/tmp/test.gguf"),
            on_disk_bytes: 1024,
            backend_chip: "Apple M5 Max".to_string(),
            backend: "mlx-native",
            n_layers: 64,
            hidden_size: 4096,
            vocab_size: 151_936,
            n_attention_heads: 16,
            n_key_value_heads: 4,
            head_dim: 128,
            sliding_window: None,
            full_attention_interval: Some(4),
            max_context_length: Some(262_144),
            moe: Some(MoeShape {
                n_experts: 128,
                n_experts_per_tok: 8,
            }),
            quant_label: Some("Q4_K".to_string()),
            quant_bpw: Some(4.55),
            tokenizer_source: TokenizerSource::GgufEmbedded,
            eos_token_ids: vec![151_643, 151_645],
            bos_token_id: None,
            chat_template_source: ChatTemplateSource::GgufEmbedded,
            provenance: Provenance::External,
            vision_projector: None,
            load_wall_clock: Duration::from_secs_f64(6.84),
            resident_weight_bytes: Some(16 * 1024 * 1024 * 1024),
            kv_cache_budget_bytes: Some(4 * 1024 * 1024 * 1024),
            kv_spill_active: false,
            tq_kv_active: false,
            kv_bytes_per_token_override: None,
        };

        let cloned = info.clone();
        assert_eq!(cloned.model_id, "test-model");
        assert_eq!(cloned.arch_family.as_str(), "qwen35");
        // Debug must not panic.
        let dbg = format!("{cloned:?}");
        assert!(dbg.contains("test-model"));
    }

    fn golden_qwen35moe_info() -> LoadInfo {
        LoadInfo {
            model_id: "<model_id>".to_string(),
            arch_str: "qwen35moe".to_string(),
            arch_family: ArchFamily::Qwen35,
            model_path: PathBuf::from("<on-disk path>"),
            on_disk_bytes: (29.83_f64 * 1024.0 * 1024.0 * 1024.0).round() as u64,
            backend_chip: "Apple M5 Max".to_string(),
            backend: "mlx-native",
            n_layers: 64,
            hidden_size: 4096,
            vocab_size: 151_936,
            n_attention_heads: 16,
            n_key_value_heads: 4,
            head_dim: 128,
            sliding_window: None,
            full_attention_interval: Some(4),
            max_context_length: Some(262_144),
            moe: Some(MoeShape {
                n_experts: 128,
                n_experts_per_tok: 8,
            }),
            quant_label: Some("Q4_K".to_string()),
            quant_bpw: Some(4.55),
            tokenizer_source: TokenizerSource::GgufEmbedded,
            eos_token_ids: vec![151_645],
            bos_token_id: None,
            chat_template_source: ChatTemplateSource::GgufEmbedded,
            provenance: Provenance::Hf2q {
                producer_version: "hf2q 0.1.0".to_string(),
                source_sha256: "7f3abc".to_string(),
                mmproj_sha256: None,
            },
            vision_projector: None,
            load_wall_clock: Duration::from_secs_f64(6.84),
            resident_weight_bytes: Some((16.42_f64 * 1024.0 * 1024.0 * 1024.0).round() as u64),
            kv_cache_budget_bytes: Some(4 * 1024 * 1024 * 1024),
            kv_spill_active: false,
            tq_kv_active: false,
            kv_bytes_per_token_override: None,
        }
    }

    #[test]
    fn print_banner_golden_qwen35moe() {
        let info = golden_qwen35moe_info();
        let mut buf = Vec::new();
        print_banner(&info, &mut buf, false).expect("print banner");
        let got = String::from_utf8(buf).expect("utf8");
        assert_eq!(
            got,
            "hf2q load: backend = mlx-native (M5 Max)\n\
             hf2q load: model = <model_id> (arch = qwen35moe, family = qwen35)\n\
             hf2q load: source = <on-disk path> (29.83 GiB on disk)\n\
             hf2q load: layout = 64 layers, 16 heads (4 kv), head_dim=128, hidden=4096, vocab=151936\n\
             hf2q load: features = sliding_window=none, full_attn_every=4, moe=128 experts/8 active\n\
             hf2q load: quant = Q4_K dominant, ~4.55 bpw, mlx-native resident 16.42 GiB\n\
             hf2q load: max_ctx_train = 262144, kv_budget = 4.00 GiB (~32768 tokens)\n\
             hf2q load: tokenizer = gguf-embedded (<= mirrors llama-vocab.cpp)\n\
             hf2q load: chat_template = gguf-embedded\n\
             hf2q load: provenance = hf2q (producer hf2q 0.1.0, source_sha 7f3a…)\n\
             hf2q load: vision = n/a (text-only arch)\n\
             hf2q load: kv_spill = inactive\n\
             hf2q load: tq_kv = inactive\n\
             hf2q load: ready in 6.84 s\n"
        );
    }

    #[test]
    fn print_banner_handles_absent_optional_fields() {
        let mut info = golden_qwen35moe_info();
        info.sliding_window = None;
        info.full_attention_interval = None;
        info.max_context_length = None;
        info.moe = None;
        info.quant_label = None;
        info.quant_bpw = None;
        info.resident_weight_bytes = None;
        info.kv_cache_budget_bytes = None;
        info.chat_template_source = ChatTemplateSource::None;
        info.provenance = Provenance::External;
        let mut buf = Vec::new();
        print_banner(&info, &mut buf, false).expect("print banner");
        let got = String::from_utf8(buf).expect("utf8");
        assert!(got.contains("sliding_window=none, full_attn_every=none, moe=none"));
        assert!(got.contains("quant = none dominant, none, mlx-native resident none"));
        assert!(got.contains("max_ctx_train = none, kv_budget = none"));
        assert!(got.contains("chat_template = none"));
        assert!(got.contains("provenance = external"));
        assert!(got.contains("vision = n/a (text-only arch)"));
    }

    #[test]
    fn load_info_builder_qwen35_smoke() {
        use crate::inference::models::qwen35::model::Qwen35Model;
        use crate::inference::models::qwen35::{
            default_layer_types, Qwen35Config, Qwen35MoeConfig, Qwen35Variant,
        };
        use crate::serve::api::engine_qwen35::{HybridPromptCache, Qwen35LoadedModel};

        let path = tmp_path("qwen35_builder");
        write_synthetic_gguf_with_metadata(
            &path,
            &[
                KvSpec::String("general.architecture", "qwen35moe"),
                KvSpec::U32("tokenizer.ggml.bos_token_id", 151_643),
                KvSpec::String("tokenizer.chat_template", "{{ messages }}"),
            ],
            &[TensorSpec {
                name: "blk.0.ffn_gate_exps.weight",
                shape: vec![BLOCK_VALUES_Q4_K],
                ggml_type_id: GGML_TYPE_Q4_K,
                byte_len: BLOCK_BYTES_Q4_K,
            }],
        );
        let gguf = mlx_native::gguf::GgufFile::open(&path).expect("open synthetic gguf");
        let cfg = Qwen35Config {
            variant: Qwen35Variant::Moe,
            hidden_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 16,
            linear_num_key_heads: 2,
            linear_num_value_heads: 2,
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
        };
        let loaded = Qwen35LoadedModel {
            model: Qwen35Model::empty_from_cfg(cfg),
            tokenizer: tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            chat_template: "{{ messages }}".to_string(),
            model_id: "qwen-id".to_string(),
            model_path: path.clone(),
            eos_token_ids: vec![151_645],
            hidden_size: 64,
            vocab_size: 256,
            context_length: Some(1024),
            quant_type: Some("Q4_K".to_string()),
            load_duration: Duration::from_millis(7),
            provenance: Provenance::External,
            prompt_cache: HybridPromptCache::new(),
            lcp_registry: crate::serve::kv_persist::lcp_registry::LcpRegistry::new(1),
            kv_metrics_sink: None,
            disk_persistor: None,
            lcp_hydrated_for_cfg: std::collections::HashSet::new(),
            tq_kv_active: false,
            // ADR-040 Phase C iter-2a (C2b) — scaffold field on
            // `Qwen35LoadedModel`; test fixtures must populate it (here
            // as `None`) to keep struct-literal construction
            // compatible. Production iter-2a path also stores `None`.
            persistent_kv_cache: None,
        };
        let info = loaded.build_load_info(
            &gguf,
            Duration::from_millis(7),
            Some(4 * 1024 * 1024),
            false,
        );
        assert_eq!(info.model_id, "qwen-id");
        assert_eq!(info.arch_str, "qwen35moe");
        assert_eq!(info.arch_family, ArchFamily::Qwen35);
        assert_eq!(info.model_path, path);
        assert!(info.on_disk_bytes > 0);
        assert_eq!(info.n_layers, 4);
        assert_eq!(info.hidden_size, 64);
        assert_eq!(info.vocab_size, 256);
        assert_eq!(info.n_attention_heads, 8);
        assert_eq!(info.n_key_value_heads, 2);
        assert_eq!(info.head_dim, 16);
        assert_eq!(info.sliding_window, None);
        assert_eq!(info.full_attention_interval, Some(4));
        assert_eq!(info.max_context_length, Some(1024));
        assert_eq!(
            info.moe,
            Some(MoeShape {
                n_experts: 4,
                n_experts_per_tok: 2
            })
        );
        assert_eq!(info.quant_label, Some("Q4_K".to_string()));
        assert_eq!(info.quant_bpw, Some(4.5));
        assert_eq!(info.tokenizer_source, TokenizerSource::GgufEmbedded);
        assert_eq!(info.eos_token_ids, vec![151_645]);
        assert_eq!(info.bos_token_id, Some(151_643));
        assert_eq!(info.chat_template_source, ChatTemplateSource::GgufEmbedded);
        assert_eq!(info.provenance, Provenance::External);
        assert_eq!(info.vision_projector, None);
        assert_eq!(info.load_wall_clock, Duration::from_millis(7));
        assert_eq!(info.resident_weight_bytes, None);
        assert_eq!(info.kv_cache_budget_bytes, Some(4 * 1024 * 1024));
        assert!(!info.kv_spill_active);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_info_builder_gemma_smoke() {
        let path = tmp_path("gemma_builder");
        write_synthetic_gguf_with_metadata(
            &path,
            &[
                KvSpec::String("general.architecture", "gemma4"),
                KvSpec::U32("tokenizer.ggml.bos_token_id", 2),
            ],
            &[TensorSpec {
                name: "blk.0.attn_q.weight",
                shape: vec![BLOCK_VALUES_Q6_K],
                ggml_type_id: GGML_TYPE_Q6_K,
                byte_len: BLOCK_BYTES_Q6_K,
            }],
        );
        let gguf = mlx_native::gguf::GgufFile::open(&path).expect("open synthetic gguf");
        assert_eq!(arch_str_from_gguf(&gguf), "gemma4");
        assert_eq!(compute_bpw(&gguf), Some(6.5625));
        assert_eq!(gguf.metadata_u32("tokenizer.ggml.bos_token_id"), Some(2));
        let info = LoadInfo {
            model_id: "gemma-id".to_string(),
            arch_str: arch_str_from_gguf(&gguf),
            arch_family: ArchFamily::Gemma4,
            model_path: PathBuf::from("gemma-id"),
            on_disk_bytes: 0,
            backend_chip: "Apple M5 Max".to_string(),
            backend: "mlx-native",
            n_layers: 2,
            hidden_size: 32,
            vocab_size: 128,
            n_attention_heads: 4,
            n_key_value_heads: 2,
            head_dim: 8,
            sliding_window: Some(1024),
            full_attention_interval: None,
            max_context_length: Some(4096),
            moe: Some(MoeShape {
                n_experts: 8,
                n_experts_per_tok: 2,
            }),
            quant_label: infer_quant_label(&gguf),
            quant_bpw: compute_bpw(&gguf),
            tokenizer_source: TokenizerSource::HfTokenizerJson {
                path: PathBuf::from("tokenizer.json"),
            },
            eos_token_ids: vec![1, 106],
            bos_token_id: gguf.metadata_u32("tokenizer.ggml.bos_token_id"),
            chat_template_source: chat_template_source(
                &gguf,
                Some("FALLBACK_GEMMA4_API_CHAT_TEMPLATE"),
            ),
            provenance: Provenance::External,
            vision_projector: None,
            load_wall_clock: Duration::from_millis(12),
            resident_weight_bytes: None,
            kv_cache_budget_bytes: None,
            kv_spill_active: false,
            tq_kv_active: false,
            kv_bytes_per_token_override: None,
        };
        assert_eq!(info.arch_family, ArchFamily::Gemma4);
        assert_eq!(info.quant_label, Some("Q6_K".to_string()));
        assert!(matches!(
            info.chat_template_source,
            ChatTemplateSource::HardcodedFallback { name }
                if name == "FALLBACK_GEMMA4_API_CHAT_TEMPLATE"
        ));

        let _ = std::fs::remove_file(&path);
    }

    #[derive(Clone, Default)]
    struct RecordingLayer {
        events: std::sync::Arc<std::sync::Mutex<Vec<Vec<String>>>>,
    }

    struct FieldNameVisitor<'a> {
        names: &'a mut Vec<String>,
    }

    impl tracing::field::Visit for FieldNameVisitor<'_> {
        fn record_debug(&mut self, field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {
            self.names.push(field.name().to_string());
        }
    }

    impl<S> tracing_subscriber::Layer<S> for RecordingLayer
    where
        S: tracing::Subscriber,
    {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut names = Vec::new();
            event.record(&mut FieldNameVisitor { names: &mut names });
            self.events.lock().expect("events lock").push(names);
        }
    }

    fn capture_emit_tracing(info: &LoadInfo) -> Vec<Vec<String>> {
        use tracing_subscriber::prelude::*;

        let layer = RecordingLayer::default();
        let events = layer.events.clone();
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || emit_tracing(info));
        let captured = events.lock().expect("events lock").clone();
        captured
    }

    #[test]
    fn emit_tracing_emits_at_least_10_events() {
        let info = golden_qwen35moe_info();
        let events = capture_emit_tracing(&info);
        assert!(
            events.len() >= 10,
            "expected at least 10 tracing events, got {}",
            events.len()
        );
    }

    #[test]
    fn emit_tracing_field_names_match_load_info() {
        let info = golden_qwen35moe_info();
        let events = capture_emit_tracing(&info);
        let names: std::collections::BTreeSet<String> = events.into_iter().flatten().collect();
        let expected = [
            "model_id",
            "arch_str",
            "arch_family",
            "model_path",
            "on_disk_bytes",
            "backend_chip",
            "backend",
            "n_layers",
            "hidden_size",
            "vocab_size",
            "n_attention_heads",
            "n_key_value_heads",
            "head_dim",
            "sliding_window",
            "full_attention_interval",
            "max_context_length",
            "moe",
            "quant_label",
            "quant_bpw",
            "tokenizer_source",
            "eos_token_ids",
            "bos_token_id",
            "chat_template_source",
            "provenance",
            "vision_projector",
            "load_wall_clock",
            "resident_weight_bytes",
            "kv_cache_budget_bytes",
            "kv_spill_active",
        ];
        for field in expected {
            assert!(names.contains(field), "missing tracing field {field}");
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // ADR-040 §3.5 iter-A5b — kv_bytes_per_token / kv_bytes_for_request
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn kv_bytes_per_token_qwen35moe_golden_matches_f32_kv_formula() {
        let info = golden_qwen35moe_info();
        // 64 layers × 4 kv heads × 128 head_dim × 4 (F32) × 2 (K+V)
        // = 262_144 bytes per token = 256 KiB.
        let expected: u64 = 64 * 4 * 128 * 4 * 2;
        assert_eq!(
            info.kv_bytes_per_token(),
            expected,
            "qwen35moe golden fixture: kv_bytes_per_token MUST be 256 KiB"
        );
        // Spot-check the for_request formula against a 4096-token total.
        // (prompt=2048 + max_tokens=2048) × 256 KiB = 1 GiB.
        let total_tokens: u64 = 2048 + 2048;
        assert_eq!(
            info.kv_bytes_for_request(2048, 2048),
            total_tokens * expected,
        );
    }

    #[test]
    fn kv_bytes_per_token_zero_when_arch_facts_missing() {
        let mut info = golden_qwen35moe_info();
        info.n_layers = 0;
        assert_eq!(
            info.kv_bytes_per_token(),
            0,
            "zero n_layers ⇒ 0 (synthetic loader / test fixture)"
        );
        let mut info = golden_qwen35moe_info();
        info.n_key_value_heads = 0;
        assert_eq!(info.kv_bytes_per_token(), 0, "zero n_key_value_heads ⇒ 0");
        let mut info = golden_qwen35moe_info();
        info.head_dim = 0;
        assert_eq!(info.kv_bytes_per_token(), 0, "zero head_dim ⇒ 0");
    }

    #[test]
    fn kv_bytes_for_request_returns_zero_when_per_token_is_zero() {
        let mut info = golden_qwen35moe_info();
        info.n_layers = 0;
        // Even with huge token counts the result is 0 — caller treats
        // as "do not enforce" per the scheduler opt-out contract.
        assert_eq!(info.kv_bytes_for_request(u32::MAX, u32::MAX), 0);
    }

    #[test]
    fn kv_bytes_for_request_overflow_saturates_at_u64_max() {
        // n_layers=64 × kv=4 × hd=128 × 4 × 2 = 262_144 bytes/token.
        // u64::MAX / 262_144 ≈ 7.04e13 tokens — well above u32::MAX
        // total (~4.3e9). So u32::MAX prompt + u32::MAX max_tokens =
        // ~8.6e9 tokens × 262_144 ≈ 2.25e15 bytes, which fits in u64.
        // Manufacture an overflow by inflating n_layers to a value
        // that makes the multiply saturate.
        let mut info = golden_qwen35moe_info();
        info.n_layers = u32::MAX;
        info.n_key_value_heads = u32::MAX;
        info.head_dim = u32::MAX;
        // Per-token bytes saturates to u64::MAX under any reasonable
        // token count.
        let needed = info.kv_bytes_for_request(1, 1);
        assert_eq!(
            needed,
            u64::MAX,
            "saturating-mul must surface u64::MAX (caller's scheduler \
             check rejects as SlotBudgetExceeded)"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // ADR-040 §3.5 iter-A5c (cfa-A5b CRITICAL #1) — Gemma 4 exact per-layer
    // KV byte cost: closes the over-count gap codex flagged in iter-A5b.
    // ─────────────────────────────────────────────────────────────────────

    /// Build a canonical Gemma 4 27B-shape `Gemma4Config`:
    /// - 30 layers; `(i+1) % 6 == 0` → Full (5 of 30 layers: indices 5, 11,
    ///   17, 23, 29). Pattern matches `from_config_json` default at
    ///   `src/serve/config.rs:84-91`.
    /// - Sliding: 8 KV heads × 256 head_dim.
    /// - Full: 2 KV heads × 512 head_dim.
    ///
    /// These shapes mirror the canonical google/gemma-4-27b-it config —
    /// per the comment at `src/serve/config.rs:130-145` referencing the
    /// gguf-dump output of the bundled APEX GGUF.
    fn canonical_gemma4_27b_config() -> crate::serve::config::Gemma4Config {
        use crate::serve::config::{Gemma4Config, LayerType};
        let layer_types: Vec<LayerType> = (0..30)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    LayerType::Full
                } else {
                    LayerType::Sliding
                }
            })
            .collect();
        Gemma4Config {
            vocab_size: 262_144,
            hidden_size: 5376,
            intermediate_size: 21_504,
            moe_intermediate_size: 0,
            num_hidden_layers: 30,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            num_global_key_value_heads: 2,
            head_dim: 256,
            global_head_dim: 512,
            rms_norm_eps: 1e-6,
            rope_theta_sliding: 10_000.0,
            rope_theta_global: 1_000_000.0,
            sliding_window: 1024,
            max_position_embeddings: 131_072,
            final_logit_softcapping: None,
            attention_bias: false,
            attention_k_eq_v: true,
            tie_word_embeddings: true,
            num_experts: 0,
            top_k_experts: 0,
            layer_types,
        }
    }

    /// **CRITICAL #1 GOLDEN** — exact per-layer sum for canonical Gemma 4
    /// 27B differs from the flattened scalar.
    ///
    /// Per-token elements (F32, K+V):
    /// - 25 sliding layers × 8 KV heads × 256 head_dim × 4 (F32) × 2
    ///   (K+V) = 409_600 bytes
    /// - 5 full layers × 2 KV heads × 512 head_dim × 4 × 2 = 40_960 bytes
    /// - Sum: 450_560 bytes per token.
    ///
    /// The flattened scalar (30 × 8 × 256 × 4 × 2 = 491_520 bytes/token)
    /// OVER-counts by 40_960 bytes (~9%). The over-count is a safe upper
    /// bound — it false-rejects borderline requests, never false-accepts —
    /// but iter-A5c ships the exact value so the admit-time check matches
    /// the actual KV allocation shape at `gemma4/model.rs:1247-1257`.
    #[test]
    fn a5c_gemma4_exact_kv_bytes_per_token_matches_per_layer_sum() {
        let cfg = canonical_gemma4_27b_config();
        let exact = super::gemma4_exact_kv_bytes_per_token(&cfg);
        let expected_sliding: u64 = 25 * 8 * 256 * 4 * 2;
        let expected_full: u64 = 5 * 2 * 512 * 4 * 2;
        let expected: u64 = expected_sliding + expected_full;
        assert_eq!(
            exact, expected,
            "Gemma 4 27B exact: 25 sliding (8×256) + 5 full (2×512) × 4 × 2 = {expected}",
        );
        assert_eq!(
            exact, 450_560,
            "Gemma 4 27B canonical per-token bytes = 450 KiB-minus-1"
        );
    }

    /// **CRITICAL #1 GOLDEN** — when `LoadInfo` carries the override, the
    /// per-token getter returns the exact value, NOT the over-counted flat
    /// scalar formula.
    #[test]
    fn a5c_load_info_kv_bytes_per_token_uses_override_when_present() {
        let cfg = canonical_gemma4_27b_config();
        let exact = super::gemma4_exact_kv_bytes_per_token(&cfg);
        // Construct a LoadInfo mimicking `GemmaLoadedModel::build_load_info`
        // shape: stores the SLIDING (n_kv_heads, head_dim) but the
        // override holds the exact per-layer sum.
        let info = LoadInfo {
            model_id: "gemma-4-27b-it-canonical".to_string(),
            arch_str: "gemma4".to_string(),
            arch_family: ArchFamily::Gemma4,
            model_path: PathBuf::from("/canonical/gemma4-27b.gguf"),
            on_disk_bytes: 0,
            backend_chip: "test-gpu".to_string(),
            backend: "mlx-native",
            n_layers: cfg.num_hidden_layers as u32,
            hidden_size: cfg.hidden_size as u32,
            vocab_size: cfg.vocab_size as u32,
            n_attention_heads: cfg.num_attention_heads as u32,
            n_key_value_heads: cfg.num_key_value_heads as u32,
            head_dim: cfg.head_dim as u32,
            sliding_window: Some(cfg.sliding_window as u32),
            full_attention_interval: cfg.full_attention_interval(),
            max_context_length: Some(cfg.max_position_embeddings as u32),
            moe: None,
            quant_label: None,
            quant_bpw: None,
            tokenizer_source: TokenizerSource::HfTokenizerJson {
                path: PathBuf::from("/canonical/tokenizer.json"),
            },
            eos_token_ids: vec![1, 106],
            bos_token_id: Some(2),
            chat_template_source: ChatTemplateSource::GgufEmbedded,
            provenance: Provenance::External,
            vision_projector: None,
            load_wall_clock: Duration::ZERO,
            resident_weight_bytes: None,
            kv_cache_budget_bytes: None,
            kv_spill_active: false,
            tq_kv_active: false,
            kv_bytes_per_token_override: Some(exact),
        };
        // Override wins over the flattened formula.
        assert_eq!(
            info.kv_bytes_per_token(),
            exact,
            "kv_bytes_per_token MUST honour the override when present"
        );
        // Flat formula sanity-check — if the implementation accidentally
        // ignored the override, this would be the answer it returned, and
        // it differs from `exact` by exactly the (full - sliding) shape
        // delta documented in `gemma4_exact_kv_bytes_per_token`.
        let flat: u64 = 30 * 8 * 256 * 4 * 2;
        assert_ne!(
            exact, flat,
            "exact and flat MUST differ — otherwise the test could pass even \
             if `kv_bytes_per_token` ignored the override"
        );
        assert_eq!(flat, 491_520, "canonical flat formula sanity");
        // Falsifier: if the override were ignored, the getter would
        // return the flat scalar, which is HIGHER than exact.
        assert!(
            info.kv_bytes_per_token() < flat,
            "exact override MUST be < flattened over-count (proves override \
             is being honoured; not just a coincidence of identical values)"
        );
    }

    /// **CRITICAL #1 GOLDEN** — falsifier for the flat path: when override
    /// is `None`, the homogeneous-arch formula still runs (Qwen3.5/3.6
    /// behaviour preserved verbatim).
    #[test]
    fn a5c_load_info_kv_bytes_per_token_falls_back_to_flat_when_no_override() {
        let info = golden_qwen35moe_info();
        assert!(
            info.kv_bytes_per_token_override.is_none(),
            "Qwen35 golden fixture MUST NOT carry an override — Qwen35 layers \
             are homogeneous and the flat formula is exact"
        );
        let flat: u64 = 64 * 4 * 128 * 4 * 2;
        assert_eq!(
            info.kv_bytes_per_token(),
            flat,
            "no override ⇒ flat formula path (homogeneous arch)"
        );
    }

    /// **CRITICAL #1** — symmetric falsifier: a Gemma 4 fixture with the
    /// override populated MUST return the exact value even when the flat
    /// formula would return a different number. This guards against a
    /// future regression where someone removes the override-honouring
    /// branch in `kv_bytes_per_token`.
    #[test]
    fn a5c_load_info_override_distinct_from_flat_falsifies_regression() {
        let cfg = canonical_gemma4_27b_config();
        let exact = super::gemma4_exact_kv_bytes_per_token(&cfg);
        // Synthesize an override-carrying info with a deliberately
        // distinct exact value to prove the getter doesn't accidentally
        // run the flat formula.
        let mut info = golden_qwen35moe_info();
        info.arch_family = ArchFamily::Gemma4;
        info.kv_bytes_per_token_override = Some(exact);
        assert_eq!(
            info.kv_bytes_per_token(),
            exact,
            "override MUST short-circuit the flat formula"
        );
    }
}
