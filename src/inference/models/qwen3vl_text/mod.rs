//! Qwen3-VL **text-side** LM module (ADR-005 Wedge-4 / iter-228a).
//!
//! # Scope
//!
//! Qwen3-VL is a vision-language model: a ViT + DeepStack image encoder
//! produces image-token embeddings (Wedge-4c LANDED in `vit_gpu_qwen3vl.rs`)
//! that are spliced into the LM's input embedding stream. **This module
//! owns only the dense LM transformer** that consumes the spliced token
//! stream and emits next-token logits. The ViT side is intentionally
//! orthogonal (and already shipped).
//!
//! # Module layout
//!
//! - `mod.rs` (this file): [`Qwen3VlTextConfig`] parser, arch detection
//!   helpers, [`Qwen3VlTextLayerKind`] enum, public re-exports.
//! - `weights.rs`: [`Qwen3VlTextWeights`] — per-layer + global tensor
//!   buffers loaded from the GGUF (q4_0 dense FFN, q4_0 attention,
//!   per-head Q/K RMSNorm, layer norms, embed, final norm).
//! - `model.rs`: [`Qwen3VlTextModel`] — the loaded-model bundle (config +
//!   weights + GPU context).
//! - `forward.rs`: forward-path scaffolding + the iter-228b sentinel
//!   surface. The actual transformer-forward implementation is
//!   iter-228b scope (see [`Qwen3VlTextModel::forward_not_yet_wired`]).
//!
//! # Why a new module instead of reusing `qwen35`?
//!
//! Qwen3-VL text is a **plain dense transformer with biased GQA
//! attention + per-head Q/K RMSNorm + 3D-mRoPE**. It has:
//!
//! - **Zero** SSM / DeltaNet structure (Qwen3.5 hybrid mandates it).
//! - **No** `ssm.{state_size,group_count,inner_size,conv_kernel}`
//!   metadata keys, **no** `full_attention_interval` key.
//! - Optional attn-q/k/v **biases** (peer `qwen3vl.cpp` references
//!   `wo_b`; in practice the q4_0 GGUFs hf2q's converter emits do NOT
//!   include biases).
//! - Per-head Q-norm + K-norm (one-per-head, shape [head_dim]).
//! - DeepStack residual injection on the first `n_deepstack_layers`
//!   layers (ground-truth at `qwen3vl.cpp` lines 96-100; hf2q's
//!   `image_token_residual_add_gpu` already implements the per-position
//!   add).
//!
//! Routing it through `Qwen35Model::load_from_gguf` is structurally
//! impossible — that loader requires SSM keys absent from Qwen3-VL
//! GGUFs. iter-227 LANDED an actionable-error dispatch shim; iter-228a
//! (this module) lands the **load** path so the GGUF actually opens
//! through hf2q's engine without falling back to the iter-227 bail.
//!
//! # iter-228a vs iter-228b split
//!
//! - **iter-228a (this iter)**: module skeleton, [`Qwen3VlTextConfig`]
//!   parser, [`weights::Qwen3VlTextWeights`] full loader, engine seam
//!   ([`crate::serve::api::engine::LoadedModel::Qwen3VlText`]),
//!   operator-gated load harness against the real
//!   `Qwen/Qwen3-VL-2B-Instruct` GGUF.
//! - **iter-228b**: dense transformer forward (per-layer attn + GQA
//!   flash-attn + 3D-mRoPE + SiLU FFN + DeepStack residual) wired through
//!   [`crate::serve::api::engine::worker_run`]'s Generate / GenerateStream /
//!   GenerateWithSoftTokens arms, replacing the [`forward::QWEN3VL_TEXT_FORWARD_PENDING_SENTINEL`]
//!   bail with the live forward chain. This mirrors the iter-215 →
//!   Wedge-3 split that landed Qwen3.5/3.6 (see
//!   `engine_qwen35.rs::QWEN35_NOT_IMPLEMENTED_SENTINEL`).
//!
//! Splitting the work this way means iter-228a unblocks operator-script
//! Step 5 (model load + readyz) without faking forward, and lets
//! iter-228b focus on the LM-forward surgery against a known-good load
//! pipeline (no compounding load-bug-vs-forward-bug ambiguity).

use anyhow::{anyhow, Context, Result};
use mlx_native::gguf::{GgufFile, MetadataValue};

pub mod forward;
pub mod model;
pub mod weights;

pub use model::Qwen3VlTextModel;
pub use weights::Qwen3VlTextWeights;

// ---------------------------------------------------------------------------
// Architecture detection
// ---------------------------------------------------------------------------
//
// The arch-detection predicates [`is_qwen3_vl_arch`] /
// [`is_qwen3_vl_moe_arch`] live in the sibling `qwen35` module
// (`crate::inference::models::qwen35::is_qwen3_vl_arch`) so iter-227's
// dispatch shim can call them without first deciding which family
// owns the dispatch. They're re-exported here for callers that already
// know they're on a Qwen3-VL load path.

pub use crate::inference::models::qwen35::{
    is_qwen3_vl_arch, is_qwen3_vl_moe_arch, ARCH_QWEN3VLMOE_UPSTREAM,
    ARCH_QWEN3VL_UPSTREAM, ARCH_QWEN3_VL,
};

// ---------------------------------------------------------------------------
// Layer-kind enum
// ---------------------------------------------------------------------------

/// Per-layer kind for the Qwen3-VL text transformer.
///
/// **Intentionally trivial** — every Qwen3-VL text-LM layer is a
/// [`Self::Dense`] block with the same shape (biased GQA + per-head
/// Q/K RMSNorm + SiLU-gated FFN). Unlike Qwen3.5's hybrid stack, there
/// are no DeltaNet / sliding-window / per-layer variants. The enum
/// exists for forward-compatibility (a future Qwen3-VL-MoE convert
/// pipeline could add a `Moe` variant) and to keep the shape parallel
/// to [`crate::inference::models::qwen35::Qwen35LayerKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3VlTextLayerKind {
    /// Dense GQA + SiLU-gated FFN (every Qwen3-VL-2B/4B text layer).
    Dense,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Default RoPE base frequency for Qwen3-VL text (matches HF
/// `text_config.rope_theta` for `Qwen/Qwen3-VL-2B-Instruct` /
/// `Qwen/Qwen3-VL-4B-Instruct`).
///
/// The real-model GGUF at
/// `/opt/hf2q/.cfa-archive/wedge4f-out/qwen3-vl-2b-q4_0.gguf` does NOT
/// emit `qwen3_vl.rope.freq_base`, so this default applies. iter-228b's
/// 3D-mRoPE wiring consumes this value.
pub const DEFAULT_ROPE_THETA: f32 = 5_000_000.0;

/// Default RMSNorm epsilon for Qwen3-VL text (matches HF
/// `text_config.rms_norm_eps`).
///
/// The real-model GGUF does NOT emit
/// `qwen3_vl.attention.layer_norm_rms_epsilon`, so this default applies.
/// iter-228b's per-layer norm dispatch consumes this value.
pub const DEFAULT_RMS_NORM_EPS: f32 = 1.0e-6;

/// Default IMROPE section sizes `[t, y, x, pad]` for Qwen3-VL text.
///
/// Matches HF `text_config.rope_scaling.mrope_section = [24, 20, 20]`
/// for `Qwen/Qwen3-VL-2B-Instruct` and `Qwen/Qwen3-VL-4B-Instruct`,
/// padded to 4 entries with a trailing `0` to match the canonical peer
/// `int sections[4]` layout (`src/models/qwen3vl.cpp:67`,
/// `convert_hf_to_gguf.py:11944`).
///
/// Sum = 24 + 20 + 20 = **64** = `head_dim/2` (head_dim=128). Under the
/// `mrope_interleaved=true` IMROPE convention only the first half of
/// each head is rotated; the upper half is identity. The 4 sections
/// allocate the rotary half across `[t, y, x, pad]` axes — `t` gets 24
/// of the 64 rotary slots; `y` and `x` each get 20; the trailing axis
/// is 0 (padding for the canonical 4-section format).
///
/// iter-228b's 3D-mRoPE wiring consumes this value to thread the
/// `(t, y, x, z)` axes built by `build_qwen3vl_positions`
/// (`src/serve/forward_prefill.rs:220`) into mlx-native's
/// `dispatch_rope_multi_cached` kernel under
/// [`QWEN3VL_ROPE_MODE`] = 40 (IMROPE).
///
/// **Bug-history note** (iter-8a): A pre-iter-8a default of
/// `[3, 3, 2]` (`[u32; 3]`) was a copy of HF's *vision-side* rope
/// sections from a misread, not the text-side. The text-side ground
/// truth is the 4-int `[24, 20, 20, 0]` that the peer reads via
/// `LLM_KV_ROPE_DIMENSION_SECTIONS` and that hf2q's converter emits
/// via `emit_qwen3vl_metadata` (`src/backends/gguf.rs:3848`).
pub const DEFAULT_MROPE_SECTION: [u32; 4] = [24, 20, 20, 0];

/// Default number of DeepStack residual injection layers for
/// Qwen3-VL-2B/4B text.
///
/// Matches HF `text_config.deepstack_visual_indexes` length = 3. The
/// peer source `qwen3vl.cpp:96-100` injects DeepStack residuals into
/// the first `n_deepstack_layers` LM layers; hf2q's
/// `image_token_residual_add_gpu` (Wedge-4c.5) implements the
/// per-position add, and the per-LM-layer dispatch is iter-228b
/// scope.
pub const DEFAULT_N_DEEPSTACK_LAYERS: usize = 3;

/// IMROPE rotary mode index for Qwen3-VL **text-LM** RoPE-multi.
///
/// Matches `GGML_ROPE_TYPE_IMROPE = 40` (`/opt/llama.cpp/ggml/include/ggml.h:254`)
/// — the same mode Qwen3.5 / Qwen3.6 use. The peer dispatch
/// (`/opt/llama.cpp/src/llama-model.cpp:2316-2320`) returns
/// `LLAMA_ROPE_TYPE_IMROPE` for `LLM_ARCH_QWEN3VL`, and the text-LM
/// graph (`/opt/llama.cpp/src/models/qwen3vl.cpp:95-108`) calls
/// `ggml_rope_multi(..., rope_type, ...)` with that value.
///
/// **Bug-history note** (iter-8a): A pre-iter-8a value of `24`
/// (`GGML_ROPE_TYPE_VISION`) was a misread of the **mtmd ViT** prelude
/// `ggml_rope_multi(..., MROPE, ...)` call at
/// `/opt/llama.cpp/tools/mtmd/models/qwen3vl.cpp:45-58` — that's the
/// vision encoder's own RoPE, NOT the text-LM's. The text-LM uses
/// IMROPE. Wiring `mode=24` would still call `dispatch_rope_multi_cached`
/// but with the wrong frequency-band layout (Vision uses non-interleaved
/// y/x; IMROPE uses `mrope_interleaved=true` t/y/x/pad), producing
/// silently-wrong attention.
pub const QWEN3VL_ROPE_MODE: u32 = 40;

/// Architecture config for a Qwen3-VL **text** LM, parsed from GGUF.
///
/// Source of truth is the GGUF metadata at the `qwen3_vl.*` prefix,
/// validated against the real `Qwen/Qwen3-VL-2B-Instruct` q4_0 dump:
///
/// ```text
///   qwen3_vl.block_count           = 28
///   qwen3_vl.embedding_length      = 2048
///   qwen3_vl.attention.head_count  = 16
///   qwen3_vl.attention.head_count_kv = 8
///   qwen3_vl.feed_forward_length   = 6144
///   qwen3_vl.context_length        = 262144
/// ```
///
/// Defaults filled by [`Qwen3VlTextConfig::from_gguf`] when keys are
/// absent: `rope_theta = 5e6`, `rms_norm_eps = 1e-6`, `head_dim`
/// derived from `hidden_size / num_attention_heads`,
/// `mrope_section = [24, 20, 20, 0]`, `n_deepstack_layers = 3`.
///
/// `vocab_size` and `tied_word_embeddings` are derived from the GGUF's
/// tensor table (the converter emits 310 tensors for tied; 311 for
/// untied — the presence/absence of `output.weight` is authoritative).
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3VlTextConfig {
    /// Number of transformer layers (28 for 2B, 36 for 4B).
    pub num_hidden_layers: u32,
    /// Hidden / model dimension (2048 for 2B, 2560 for 4B).
    pub hidden_size: u32,
    /// Multi-head attention head count (16 for 2B).
    pub num_attention_heads: u32,
    /// KV-head count (GQA: 8 for 2B → 2:1 GQA ratio).
    pub num_key_value_heads: u32,
    /// Per-head dimension (128 for 2B; derived as
    /// `hidden_size / num_attention_heads` when GGUF doesn't emit it).
    pub head_dim: u32,
    /// FFN intermediate (gate/up/down hidden) dim (6144 for 2B).
    pub intermediate_size: u32,
    /// Vocabulary size (151936 for Qwen3-VL family).
    pub vocab_size: u32,
    /// Maximum context length declared by the GGUF (262144 for 2B).
    pub max_position_embeddings: u32,
    /// RoPE base frequency. Defaults to [`DEFAULT_ROPE_THETA`] when
    /// `qwen3_vl.rope.freq_base` is absent.
    pub rope_theta: f32,
    /// RMSNorm epsilon. Defaults to [`DEFAULT_RMS_NORM_EPS`] when
    /// `qwen3_vl.attention.layer_norm_rms_epsilon` is absent.
    pub rms_norm_eps: f32,
    /// 4-int IMROPE section sizes `[t, y, x, pad]`. Read from GGUF key
    /// `{prefix}.rope.dimension_sections` when present (the converter
    /// emits it; peer requires it); falls back to
    /// [`DEFAULT_MROPE_SECTION`] = `[24, 20, 20, 0]`.
    pub mrope_section: [u32; 4],
    /// Number of LM layers that consume DeepStack residual injections
    /// (the first `n_deepstack_layers` layers; peer `qwen3vl.cpp:96`).
    pub n_deepstack_layers: usize,
    /// `true` iff `output.weight` is absent from the GGUF tensor table
    /// → the LM head reuses `token_embd.weight`. Derived at parse
    /// time from the tensor table; cannot be overridden.
    pub tied_word_embeddings: bool,
    /// Per-layer kind table. Authoritative — every entry is
    /// [`Qwen3VlTextLayerKind::Dense`] for the 2B/4B variants, but
    /// kept as a `Vec` so a future hybrid Qwen3-VL variant could
    /// override on a per-layer basis without breaking callers.
    pub layer_types: Vec<Qwen3VlTextLayerKind>,
}

impl Qwen3VlTextConfig {
    /// Parse the config from an open Qwen3-VL GGUF.
    ///
    /// Reads all `qwen3_vl.*` metadata keys; falls back to the defaults
    /// declared in this module's constants for keys the converter omits.
    /// Detects tied embeddings by checking for `output.weight` in the
    /// tensor table. `vocab_size` is read from the embedding tensor's
    /// shape (more reliable than the metadata key, which the converter
    /// may omit).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // The arch string is recognized as Qwen3-VL, so the metadata
        // prefix is one of the three known spellings — pick the one
        // that has tensor-side keys actually present. iter-228a's real
        // GGUF uses the underscored `qwen3_vl` form; the upstream
        // `qwen3vl` form is forward-compat for a future llama.cpp-aligned
        // converter.
        let arch = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| anyhow!("Qwen3VlTextConfig: missing general.architecture"))?;
        if !is_qwen3_vl_arch(arch) {
            return Err(anyhow!(
                "Qwen3VlTextConfig::from_gguf: arch {arch:?} is not a Qwen3-VL family arch \
                 (expected one of {:?}, {:?}, {:?})",
                ARCH_QWEN3_VL,
                ARCH_QWEN3VL_UPSTREAM,
                ARCH_QWEN3VLMOE_UPSTREAM,
            ));
        }
        if is_qwen3_vl_moe_arch(arch) {
            return Err(anyhow!(
                "Qwen3VlTextConfig::from_gguf: MoE Qwen3-VL ({arch:?}) is not yet supported \
                 by hf2q (no convert pipeline emits it; iter-228 closes only the dense path)"
            ));
        }
        // Try the underscored prefix first (what hf2q's converter
        // actually emits today; matches the real-model GGUF). Fall
        // back to the upstream prefix for forward-compat with a future
        // llama.cpp-aligned converter.
        let prefix = if gguf
            .metadata_u32(&format!("{}.block_count", ARCH_QWEN3_VL))
            .is_some()
        {
            ARCH_QWEN3_VL
        } else if gguf
            .metadata_u32(&format!("{}.block_count", ARCH_QWEN3VL_UPSTREAM))
            .is_some()
        {
            ARCH_QWEN3VL_UPSTREAM
        } else {
            return Err(anyhow!(
                "Qwen3VlTextConfig::from_gguf: neither {:?}.block_count nor {:?}.block_count \
                 is present in metadata — GGUF appears Qwen3-VL by arch but is missing core \
                 architecture facts.",
                ARCH_QWEN3_VL,
                ARCH_QWEN3VL_UPSTREAM,
            ));
        };
        let req_u32 = |key: &str| -> Result<u32> {
            gguf.metadata_u32(&format!("{prefix}.{key}"))
                .ok_or_else(|| anyhow!("Qwen3VlTextConfig: missing {prefix}.{key}"))
        };

        let num_hidden_layers = req_u32("block_count")
            .context("Qwen3VlTextConfig: block_count")?;
        let hidden_size = req_u32("embedding_length")
            .context("Qwen3VlTextConfig: embedding_length")?;
        let num_attention_heads = req_u32("attention.head_count")
            .context("Qwen3VlTextConfig: attention.head_count")?;
        let num_key_value_heads = req_u32("attention.head_count_kv")
            .context("Qwen3VlTextConfig: attention.head_count_kv")?;
        let intermediate_size = req_u32("feed_forward_length")
            .context("Qwen3VlTextConfig: feed_forward_length")?;
        let max_position_embeddings = gguf
            .metadata_u32(&format!("{prefix}.context_length"))
            .unwrap_or(0);

        // iter-228a Phase-2c (Codex review of 3811f5d, finding #1 medium):
        // explicit invariants on parsed dimensions BEFORE deriving
        // head_dim — guards against malformed GGUFs whose tensor
        // shapes happen to match Qwen3-VL but whose metadata is
        // internally inconsistent (would silently route to iter-228b
        // forward with a wrong attention layout).
        if num_attention_heads == 0 {
            return Err(anyhow!(
                "Qwen3VlTextConfig: num_attention_heads is zero — invalid"
            ));
        }
        if num_key_value_heads == 0 {
            return Err(anyhow!(
                "Qwen3VlTextConfig: num_key_value_heads is zero — invalid \
                 (also avoids modulo-by-zero in the GQA divisibility check)"
            ));
        }
        if hidden_size % num_attention_heads != 0 {
            return Err(anyhow!(
                "Qwen3VlTextConfig: hidden_size ({hidden_size}) must be \
                 divisible by num_attention_heads ({num_attention_heads}) \
                 for the derived head_dim to be exact"
            ));
        }

        // head_dim: prefer explicit `attention.key_length` (some
        // converters emit it); else derive from hidden / heads. For the
        // real-model GGUF the explicit key is absent → derived value
        // is 2048 / 16 = 128 ✓.
        let head_dim = gguf
            .metadata_u32(&format!("{prefix}.attention.key_length"))
            .unwrap_or(hidden_size / num_attention_heads);
        if head_dim == 0 {
            return Err(anyhow!(
                "Qwen3VlTextConfig: derived head_dim is zero (hidden={hidden_size}, \
                 heads={num_attention_heads}); refusing to load"
            ));
        }
        // Phase-2c invariant: when explicit key_length is supplied,
        // it MUST be consistent with hidden_size / num_attention_heads.
        // Catches a config where heads + head_dim disagree with hidden.
        if hidden_size != num_attention_heads * head_dim {
            return Err(anyhow!(
                "Qwen3VlTextConfig: hidden_size ({hidden_size}) != \
                 num_attention_heads ({num_attention_heads}) * head_dim \
                 ({head_dim}) — config is internally inconsistent"
            ));
        }
        if num_attention_heads % num_key_value_heads != 0 {
            return Err(anyhow!(
                "Qwen3VlTextConfig: invalid GQA shape — num_attention_heads={num_attention_heads} \
                 must be divisible by num_key_value_heads={num_key_value_heads}"
            ));
        }

        let rope_theta = gguf
            .metadata_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(DEFAULT_ROPE_THETA);
        let rms_norm_eps = gguf
            .metadata_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(DEFAULT_RMS_NORM_EPS);

        // 4-int IMROPE sections — peer reads via
        // `LLM_KV_ROPE_DIMENSION_SECTIONS` at `qwen3vl.cpp:5` with
        // `get_key_or_arr(... 4, true)` (mandatory; `true` = required).
        // hf2q's converter (`backends/gguf.rs::emit_qwen3vl_metadata`,
        // line 3848) emits exactly 4 i32 entries `[24, 20, 20, 0]`
        // per HF `text_config.rope_scaling.mrope_section` padded.
        // Pre-iter-228b GGUFs (mtime ≤ 2026-05-02) lack the key —
        // fall through to [`DEFAULT_MROPE_SECTION`] for those, with a
        // single warn-once on the first unwrap.
        let mrope_section: [u32; 4] = match gguf
            .metadata(&format!("{prefix}.rope.dimension_sections"))
        {
            Some(MetadataValue::Array(arr)) => {
                if arr.len() < 4 {
                    return Err(anyhow!(
                        "Qwen3VlTextConfig: {prefix}.rope.dimension_sections has length {}, \
                         must be ≥ 4 (peer canonical layout is exactly 4 ints; \
                         convert_hf_to_gguf.py:11944 pads with 0 when source has 3)",
                        arr.len()
                    ));
                }
                let mut out = [0u32; 4];
                for (i, slot) in out.iter_mut().enumerate() {
                    *slot = match &arr[i] {
                        MetadataValue::Int32(v) if *v >= 0 => *v as u32,
                        MetadataValue::Uint32(v) => *v,
                        other => {
                            return Err(anyhow!(
                                "Qwen3VlTextConfig: {prefix}.rope.dimension_sections[{i}] \
                                 has unexpected metadata type ({other:?}); expected Int32 \
                                 or Uint32"
                            ));
                        }
                    };
                }
                out
            }
            Some(other) => {
                return Err(anyhow!(
                    "Qwen3VlTextConfig: {prefix}.rope.dimension_sections has unexpected \
                     metadata kind ({other:?}); expected Array of Int32"
                ));
            }
            None => DEFAULT_MROPE_SECTION,
        };
        // Phase-2c invariant: rotary axis count must equal head_dim/2
        // under IMROPE (`mrope_interleaved=true`). Sum >= 0 by construction.
        // For Qwen3-VL-2B/4B: 24+20+20+0 = 64 = 128/2 ✓.
        let rotary_axis_total: u32 = mrope_section.iter().sum();
        let rotary_axis_target = head_dim / 2;
        if rotary_axis_total != rotary_axis_target {
            return Err(anyhow!(
                "Qwen3VlTextConfig: rope.dimension_sections sum ({rotary_axis_total}) != \
                 head_dim/2 ({rotary_axis_target}) — IMROPE rotary-axis budget violated; \
                 sections={mrope_section:?}, head_dim={head_dim}"
            ));
        }

        // n_deepstack_layers — peer reads via `LLM_KV_NUM_DEEPSTACK_LAYERS`
        // at `qwen3vl.cpp:4` with `get_key(..., false)` (optional;
        // defaults to 0). hf2q's converter
        // (`backends/gguf.rs::emit_qwen3vl_metadata`, line 3914) emits
        // `len(vision_config.deepstack_visual_indexes)` = 3 for the
        // 2B/4B Instruct variants. Default to
        // [`DEFAULT_N_DEEPSTACK_LAYERS`] = 3 when absent (matches HF
        // text_config + iter-228a's pre-existing default).
        let n_deepstack_layers = gguf
            .metadata_u32(&format!("{prefix}.n_deepstack_layers"))
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_N_DEEPSTACK_LAYERS);
        if n_deepstack_layers > num_hidden_layers as usize {
            return Err(anyhow!(
                "Qwen3VlTextConfig: n_deepstack_layers ({n_deepstack_layers}) exceeds \
                 num_hidden_layers ({num_hidden_layers}) — peer's per-layer dispatch \
                 (`il < n_deepstack_layers`) would index past layers[]"
            ));
        }

        // vocab_size: derive from the embedding tensor's shape.
        //
        // GGUF stores dimensions innermost-first; the mlx-native loader
        // REVERSES them at parse time
        // (`mlx_native::gguf::mod::rs:858-861`) to match the
        // `[rows, cols]` convention. So `token_embd.weight` is shape
        // `[vocab_size, hidden_size]` after the reverse: gguf-dump
        // shows `2048, 151936` on disk → `[151936, 2048]` in memory →
        // `shape[0] = vocab_size`.
        let vocab_size = match gguf.tensor_info("token_embd.weight") {
            Some(info) if info.shape.len() == 2 => info.shape[0] as u32,
            Some(info) => {
                return Err(anyhow!(
                    "Qwen3VlTextConfig: token_embd.weight has unexpected rank {} (shape={:?})",
                    info.shape.len(),
                    info.shape
                ));
            }
            None => {
                return Err(anyhow!(
                    "Qwen3VlTextConfig: missing token_embd.weight tensor (cannot derive vocab_size)"
                ));
            }
        };

        // Tied embeddings: presence of `output.weight` in the tensor
        // table is the authoritative signal. The real-model GGUF emits
        // 310 tensors total → tied; an untied variant would emit 311
        // (one extra `output.weight`). HF text_config.tie_word_embeddings
        // is already baked into this signal at convert time.
        let tied_word_embeddings = gguf.tensor_info("output.weight").is_none();

        let layer_types =
            vec![Qwen3VlTextLayerKind::Dense; num_hidden_layers as usize];

        Ok(Self {
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            mrope_section,
            n_deepstack_layers,
            tied_word_embeddings,
            layer_types,
        })
    }

    /// GQA group ratio (`num_attention_heads / num_key_value_heads`).
    /// Always divisible — invariant checked at parse time.
    pub fn gqa_group_ratio(&self) -> u32 {
        self.num_attention_heads / self.num_key_value_heads
    }
}

// ---------------------------------------------------------------------------
// Tests — config parser
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

/// Synthetic-GGUF helpers used by [`tests`] below + by integration tests
/// in `tests/qwen3vl_text_lm_forward.rs`. These produce a header-only +
/// metadata-only + tensor-info-only GGUF (no tensor data payload), which
/// is sufficient for callers that consume metadata + tensor-shape
/// information but never invoke `load_tensor` / `load_tensor_f32`.
///
/// Public-but-cfg(test)-or-feature-gated would be cleaner; this is
/// `pub(crate)` so the integration test crate (which is a separate
/// compilation unit) can't reach it. The integration test re-implements
/// the helper inline against the GGUF binary format — that's intentional:
/// the integration test against the real 1.32 GB GGUF needs no synthetic
/// fixture, and the synthetic-fixture surface is unit-test-only.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use std::io::Write;
    use std::path::PathBuf;

    /// Minimal GGUF metadata key descriptor for [`write_minimal_qwen3vl_gguf`].
    /// `Str` is currently unused by in-tree tests but kept for forward-
    /// compat with future tests that need string-typed metadata keys.
    pub(crate) enum Kv {
        #[allow(dead_code)]
        Str(String, String),
        U32(String, u32),
        /// Float32 metadata value (rope.freq_base + layer_norm_rms_epsilon).
        #[allow(dead_code)]
        F32(String, f32),
        /// Array of i32 metadata value (rope.dimension_sections — the
        /// canonical 4-int IMROPE sections array). The on-wire encoding
        /// matches GGUF's `Array(Int32)` variant: type=GGUF_TYPE_ARRAY (9),
        /// then inner_type=GGUF_TYPE_INT32 (5), then u64 length, then
        /// length × i32 LE.
        ArrayI32(String, Vec<i32>),
    }

    /// Tensor descriptor: name + shape (innermost-first per GGUF
    /// convention). All synthetic tensors are emitted as F32 with byte_len
    /// = product(shape) * 4, but the tensor data section is left
    /// uninitialized — callers must NOT call `load_tensor*` on these
    /// tensors. [`Qwen3VlTextConfig::from_gguf`] only consumes
    /// [`mlx_native::gguf::GgufFile::tensor_info`] which is satisfied by
    /// the on-disk tensor-info section.
    pub(crate) struct TensorDesc<'a> {
        pub name: &'a str,
        pub shape: &'a [usize],
    }

    /// Write a minimal valid GGUF (header + metadata + N tensor infos +
    /// zero-padded tensor-data section) and return its on-disk path.
    /// Caller is responsible for `std::fs::remove_file` (the file lives
    /// in the OS tmp dir; cleanup is best-effort).
    pub(crate) fn write_minimal_qwen3vl_gguf(
        arch_str: &str,
        extra_kvs: &[Kv],
        tensors: &[TensorDesc<'_>],
    ) -> PathBuf {
        const GGUF_TYPE_UINT32: u32 = 4;
        const GGUF_TYPE_INT32: u32 = 5;
        const GGUF_TYPE_FLOAT32: u32 = 6;
        const GGUF_TYPE_STRING: u32 = 8;
        const GGUF_TYPE_ARRAY: u32 = 9;
        const GGML_TYPE_F32: u32 = 0;
        const ALIGNMENT: u64 = 32;

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // tensor_count
        // metadata_kv_count = 1 (general.architecture) + extras
        let kv_count = 1 + extra_kvs.len() as u64;
        buf.extend_from_slice(&kv_count.to_le_bytes());
        // KV: general.architecture
        let key = b"general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        let val = arch_str.as_bytes();
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);
        // Extra KVs.
        for kv in extra_kvs {
            match kv {
                Kv::Str(k, v) => {
                    buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
                    buf.extend_from_slice(k.as_bytes());
                    buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
                    buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    buf.extend_from_slice(v.as_bytes());
                }
                Kv::U32(k, v) => {
                    buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
                    buf.extend_from_slice(k.as_bytes());
                    buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                Kv::F32(k, v) => {
                    buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
                    buf.extend_from_slice(k.as_bytes());
                    buf.extend_from_slice(&GGUF_TYPE_FLOAT32.to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                Kv::ArrayI32(k, vs) => {
                    buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
                    buf.extend_from_slice(k.as_bytes());
                    buf.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
                    // GGUF Array on-wire: inner_type (u32) + length (u64) + values.
                    buf.extend_from_slice(&GGUF_TYPE_INT32.to_le_bytes());
                    buf.extend_from_slice(&(vs.len() as u64).to_le_bytes());
                    for v in vs {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
        // Tensor info section.
        let mut data_offset: u64 = 0;
        for t in tensors {
            buf.extend_from_slice(&(t.name.len() as u64).to_le_bytes());
            buf.extend_from_slice(t.name.as_bytes());
            buf.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
            for d in t.shape {
                buf.extend_from_slice(&(*d as u64).to_le_bytes());
            }
            buf.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
            buf.extend_from_slice(&data_offset.to_le_bytes());
            let elem_count: usize = t.shape.iter().product();
            data_offset += (elem_count * 4) as u64;
        }
        // Pad to ALIGNMENT and add zero-tensor-data section.
        let header_len = buf.len() as u64;
        let pad = ((header_len + ALIGNMENT - 1) & !(ALIGNMENT - 1)) - header_len;
        buf.extend(std::iter::repeat(0u8).take(pad as usize));
        // Tensor data: zero-padded.
        buf.extend(std::iter::repeat(0u8).take(data_offset as usize));

        let path = std::env::temp_dir().join(format!(
            "hf2q-qwen3vl-test-{}.gguf",
            std::process::id() as u64 ^ rand_u64()
        ));
        let mut f = std::fs::File::create(&path).expect("create temp gguf");
        f.write_all(&buf).expect("write gguf");
        f.flush().expect("flush gguf");
        path
    }

    /// Cheap process-local random u64 to avoid temp-file name collisions
    /// when multiple test threads write fixtures concurrently. Uses
    /// nanos-of-day XOR'd with a thread-local counter; collision-free
    /// enough for a unit-test temp-path namespace.
    fn rand_u64() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(0);
        nanos ^ (COUNTER.fetch_add(1, Ordering::Relaxed)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }
}

// ---------------------------------------------------------------------------
// Tests — config parser
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::test_fixtures::{write_minimal_qwen3vl_gguf, Kv, TensorDesc};
    use super::*;

    fn standard_2b_kvs(prefix: &str) -> Vec<Kv> {
        vec![
            Kv::U32(format!("{prefix}.block_count"), 28),
            Kv::U32(format!("{prefix}.embedding_length"), 2048),
            Kv::U32(format!("{prefix}.attention.head_count"), 16),
            Kv::U32(format!("{prefix}.attention.head_count_kv"), 8),
            Kv::U32(format!("{prefix}.feed_forward_length"), 6144),
            Kv::U32(format!("{prefix}.context_length"), 262144),
        ]
    }

    #[test]
    fn from_gguf_parses_underscored_arch_with_real_2b_shape() {
        // The real-model GGUF arch.
        let kvs = standard_2b_kvs("qwen3_vl");
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("qwen3_vl", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let cfg = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect("parse Qwen3VlTextConfig from synthetic 2B GGUF");
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.intermediate_size, 6144);
        assert_eq!(cfg.head_dim, 128, "derived head_dim 2048/16");
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.max_position_embeddings, 262144);
        assert!(
            (cfg.rope_theta - DEFAULT_ROPE_THETA).abs() < 1.0,
            "rope_theta default {}",
            cfg.rope_theta
        );
        assert!(
            (cfg.rms_norm_eps - DEFAULT_RMS_NORM_EPS).abs() < 1e-12,
            "rms_norm_eps default {}",
            cfg.rms_norm_eps
        );
        assert_eq!(cfg.mrope_section, DEFAULT_MROPE_SECTION);
        assert_eq!(cfg.n_deepstack_layers, DEFAULT_N_DEEPSTACK_LAYERS);
        assert!(
            cfg.tied_word_embeddings,
            "tied detection — output.weight is NOT emitted in this fixture"
        );
        assert_eq!(cfg.layer_types.len(), 28);
        assert!(
            cfg.layer_types.iter().all(|k| *k == Qwen3VlTextLayerKind::Dense),
            "every layer is Dense for Qwen3-VL-2B/4B"
        );
        assert_eq!(cfg.gqa_group_ratio(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_parses_upstream_no_underscore_arch() {
        // Forward-compat: a future converter that aligns with llama.cpp
        // upstream emits `qwen3vl` (no underscore). Both prefix + arch
        // string forms must work.
        let kvs = standard_2b_kvs("qwen3vl");
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("qwen3vl", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let cfg = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect("parse Qwen3VlTextConfig from upstream-arch GGUF");
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.hidden_size, 2048);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_rejects_non_qwen3vl_arch() {
        let kvs = standard_2b_kvs("gemma4");
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("gemma4", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let err = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect_err("non-Qwen3-VL arch must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("not a Qwen3-VL family arch"),
            "expected 'not a Qwen3-VL family arch' in error message; got: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_rejects_moe_variant_with_specific_message() {
        // MoE Qwen3-VL is recognized at the dispatch level (so we route
        // to a clear error rather than a Gemma-path crash), but the
        // dense config parser must refuse it explicitly.
        let kvs = standard_2b_kvs("qwen3vlmoe");
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("qwen3vlmoe", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let err = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect_err("MoE Qwen3-VL must be rejected by the dense parser");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("MoE Qwen3-VL"),
            "expected MoE-specific message; got: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_detects_untied_embeddings_when_output_present() {
        // The real Qwen3-VL-2B converter emits 310 tensors (tied). A
        // future converter (or a hand-untied fork) would emit
        // `output.weight` → tied_word_embeddings = false.
        let kvs = standard_2b_kvs("qwen3_vl");
        let tensors = [
            TensorDesc {
                name: "token_embd.weight",
                shape: &[2048, 151936],
            },
            TensorDesc {
                name: "output.weight",
                shape: &[2048, 151936],
            },
        ];
        let path = write_minimal_qwen3vl_gguf("qwen3_vl", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let cfg = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect("parse Qwen3VlTextConfig with output.weight present");
        assert!(
            !cfg.tied_word_embeddings,
            "output.weight present → must NOT report tied"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_rejects_invalid_gqa_shape() {
        // num_attention_heads not divisible by num_key_value_heads —
        // structurally impossible for valid GQA.
        let kvs = vec![
            Kv::U32("qwen3_vl.block_count".to_string(), 28),
            Kv::U32("qwen3_vl.embedding_length".to_string(), 2048),
            Kv::U32("qwen3_vl.attention.head_count".to_string(), 16),
            Kv::U32("qwen3_vl.attention.head_count_kv".to_string(), 5),
            Kv::U32("qwen3_vl.feed_forward_length".to_string(), 6144),
            Kv::U32("qwen3_vl.context_length".to_string(), 262144),
        ];
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("qwen3_vl", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let err = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect_err("indivisible GQA shape must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("invalid GQA shape"),
            "expected 'invalid GQA shape' in error; got: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn arch_constants_match_qwen35_module_re_export() {
        // Sanity: the re-exported constants here are byte-equal to
        // the Qwen35 module's authoritative copies.
        assert_eq!(ARCH_QWEN3_VL, "qwen3_vl");
        assert_eq!(ARCH_QWEN3VL_UPSTREAM, "qwen3vl");
        assert_eq!(ARCH_QWEN3VLMOE_UPSTREAM, "qwen3vlmoe");
    }

    #[test]
    fn defaults_match_hf_text_config_for_qwen3vl_2b() {
        // Pin the defaults to the HF text_config values — if these
        // ever drift, this test fails loudly and forces a docstring
        // update.
        //
        // Pinned 2026-05-07 against
        // `/opt/hf2q/.cfa-archive/wedge4f-out/config.json`:
        //   text_config.rope_theta                  = 5_000_000
        //   text_config.rms_norm_eps                = 1.0e-6
        //   text_config.rope_scaling.mrope_section  = [24, 20, 20]
        //     (padded to 4: [24, 20, 20, 0])
        //   vision_config.deepstack_visual_indexes  = [5, 11, 17]
        //     → n_deepstack_layers = len(...) = 3
        //   LLAMA_ROPE_TYPE_IMROPE                  = 40
        //     (`/opt/llama.cpp/src/llama-model.cpp:2316-2320` returns
        //      IMROPE for LLM_ARCH_QWEN3VL)
        assert!((DEFAULT_ROPE_THETA - 5_000_000.0).abs() < 1.0);
        assert!((DEFAULT_RMS_NORM_EPS - 1.0e-6).abs() < 1e-12);
        assert_eq!(DEFAULT_MROPE_SECTION, [24, 20, 20, 0]);
        assert_eq!(
            DEFAULT_MROPE_SECTION.iter().sum::<u32>(),
            64,
            "IMROPE rotary-axis sum must equal head_dim/2 = 64 for Qwen3-VL-2B"
        );
        assert_eq!(DEFAULT_N_DEEPSTACK_LAYERS, 3);
        assert_eq!(
            QWEN3VL_ROPE_MODE, 40,
            "Qwen3-VL text-LM is IMROPE=40 (Qwen3.5/3.6 share the mode); \
             pre-iter-8a value of 24 (VISION) was a misread of the mtmd \
             ViT prelude's RoPE-multi mode"
        );
    }

    #[test]
    fn from_gguf_reads_explicit_rope_dimension_sections() {
        // When the GGUF carries `qwen3_vl.rope.dimension_sections`
        // (the canonical 4-int IMROPE sections array hf2q's converter
        // emits at `gguf.rs::emit_qwen3vl_metadata` line 3848), the
        // parser MUST read it instead of falling back to the default.
        // Use sections that sum to head_dim/2 = 64 but DIFFER from
        // the default [24, 20, 20, 0] so the test would fail if the
        // fallback path silently fired.
        let mut kvs = standard_2b_kvs("qwen3_vl");
        kvs.push(Kv::ArrayI32(
            "qwen3_vl.rope.dimension_sections".to_string(),
            vec![20, 22, 22, 0],
        ));
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("qwen3_vl", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let cfg = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect("parse with explicit rope.dimension_sections");
        assert_eq!(
            cfg.mrope_section,
            [20, 22, 22, 0],
            "explicit GGUF sections must override the default"
        );
        assert_eq!(cfg.mrope_section.iter().sum::<u32>(), 64);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_rejects_rope_sections_with_wrong_sum() {
        // Sections that don't sum to head_dim/2 violate the IMROPE
        // rotary-axis budget; refuse the load so the operator catches
        // the misconfig before silent attention drift.
        let mut kvs = standard_2b_kvs("qwen3_vl");
        kvs.push(Kv::ArrayI32(
            "qwen3_vl.rope.dimension_sections".to_string(),
            vec![10, 10, 10, 0], // sums to 30, not 64
        ));
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("qwen3_vl", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let err = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect_err("wrong-sum rope sections must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("rotary-axis budget"),
            "expected 'rotary-axis budget' in error; got: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_rejects_rope_sections_array_too_short() {
        // Peer canonical layout is exactly 4 entries (Qwen3-VL-2B's
        // 3-int HF mrope_section gets padded with 0); the converter
        // emits 4. A 3-entry array is malformed.
        let mut kvs = standard_2b_kvs("qwen3_vl");
        kvs.push(Kv::ArrayI32(
            "qwen3_vl.rope.dimension_sections".to_string(),
            vec![24, 20, 20], // only 3 entries
        ));
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("qwen3_vl", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let err = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect_err("3-entry rope sections must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("must be ≥ 4"),
            "expected '≥ 4' length error; got: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_reads_explicit_n_deepstack_layers() {
        // When the GGUF carries `qwen3_vl.n_deepstack_layers` (hf2q's
        // converter emits it from
        // `vision_config.deepstack_visual_indexes` length), the parser
        // MUST honor it. Use a non-default value to prove the override
        // path fires.
        let mut kvs = standard_2b_kvs("qwen3_vl");
        kvs.push(Kv::U32(
            "qwen3_vl.n_deepstack_layers".to_string(),
            5,
        ));
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("qwen3_vl", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let cfg = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect("parse with explicit n_deepstack_layers");
        assert_eq!(cfg.n_deepstack_layers, 5);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_rejects_n_deepstack_exceeding_n_layers() {
        // n_deepstack_layers > num_hidden_layers would let the
        // per-LM-layer dispatch (`il < n_deepstack_layers`) index past
        // layers[]. Refuse at parse time.
        let mut kvs = standard_2b_kvs("qwen3_vl");
        kvs.push(Kv::U32(
            "qwen3_vl.n_deepstack_layers".to_string(),
            100, // > 28 layers
        ));
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("qwen3_vl", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let err = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect_err("n_deepstack_layers > num_hidden_layers must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("exceeds"),
            "expected 'exceeds' in error; got: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Real-model load gate: when
    /// `HF2Q_QWEN3VL_LM_LOAD=1` and the canonical fixture
    /// `/opt/hf2q/.cfa-archive/wedge4f-out/qwen3-vl-2b-q4_0.gguf` is
    /// present, parse the config and pin against text_config.
    #[test]
    fn from_gguf_parses_real_qwen3vl_2b_when_operator_gated() {
        if std::env::var("HF2Q_QWEN3VL_LM_LOAD").ok().as_deref() != Some("1") {
            eprintln!("skip: HF2Q_QWEN3VL_LM_LOAD!=1");
            return;
        }
        let p = std::path::PathBuf::from(
            "/opt/hf2q/.cfa-archive/wedge4f-out/qwen3-vl-2b-q4_0.gguf",
        );
        if !p.exists() {
            eprintln!("skip: real GGUF fixture not present at {}", p.display());
            return;
        }
        let gguf = GgufFile::open(&p).expect("open real Qwen3-VL-2B GGUF");
        let cfg = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect("parse Qwen3VlTextConfig from real Qwen3-VL-2B GGUF");
        // Pin against HF text_config.
        assert_eq!(cfg.num_hidden_layers, 28, "Qwen3-VL-2B layers");
        assert_eq!(cfg.hidden_size, 2048, "Qwen3-VL-2B hidden");
        assert_eq!(cfg.num_attention_heads, 16, "Qwen3-VL-2B heads");
        assert_eq!(cfg.num_key_value_heads, 8, "Qwen3-VL-2B kv heads");
        assert_eq!(cfg.intermediate_size, 6144, "Qwen3-VL-2B ffn");
        assert_eq!(cfg.head_dim, 128, "Qwen3-VL-2B head_dim");
        assert_eq!(cfg.vocab_size, 151936, "Qwen3-VL family vocab");
        assert_eq!(cfg.max_position_embeddings, 262144, "Qwen3-VL-2B ctx");
        assert!(cfg.tied_word_embeddings, "Qwen3-VL-2B tied embeddings");
    }
}
