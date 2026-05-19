//! Gemma 4 (real release) HF→GGUF tensor-name + metadata mapper.
//!
//! Port of `/opt/llama.cpp/conversion/gemma.py::Gemma4Model`
//! (`@ModelBase.register("Gemma4ForConditionalGeneration")`,
//! `model_arch = gguf.MODEL_ARCH.GEMMA4`) and its inherited
//! `Gemma3Model` / `TextModel` `set_gguf_parameters` chain, plus the
//! Gemma 4-specific overlay at `gemma.py:617-765`. Cross-validated
//! against the operator's `/opt/hf2q/models/google-gemma-4-26b-a4b-it`
//! checkpoint and llama.cpp's `gguf-py/gguf/tensor_mapping.py` /
//! `constants.py::MODEL_ARCH.GEMMA4` (`:2450-2483`).
//!
//! Per ADR-033 §P0 + the 2026-05-18 real-model finding
//! (`docs/adr-033-real-model-findings/2026-05-18-gemma4-arch-mismatch.md`):
//! the prior mapper at this path was written against ADR-033's
//! "gemma4 = gemma3-shape" assumption, which is **wrong** for the
//! actual google-gemma-4-26b-a4b-it release. Real Gemma 4 is a
//! multimodal-wrapped, MoE+dense-hybrid architecture with FOUR norm
//! pairs per block, fused gate+up experts, a router sub-block, and a
//! per-layer `layer_scalar`. This file is the **full** port; the prior
//! gemma3-shape arms are deleted per [[feedback-no-backwards-compat-2026-05-18]].
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]]: every unrecognized
//! tensor returns `None` (caller errors out); the explicit-drop variant
//! is reserved for vision / audio / multimodal-projector tensors the
//! caller knows are off-path.
//!
//! # Distinctive Gemma 4 quirks (vs Llama-3 / Qwen3MoE / Gemma 3)
//!
//! 1. **Multimodal wrapper.** The text decoder is nested under
//!    `model.language_model.` (the outer `model.` houses
//!    `vision_tower.*`, `embed_vision.*`, `audio_tower.*`). The mapper
//!    transparently strips `language_model.` so the rest of the table
//!    can be written against the same `model.layers.<N>.<rest>` shape
//!    used by Llama-3 / Qwen3MoE. Mirrors the canonical Python strip at
//!    `conversion/base.py:540-541` (`if "language_model." in name: name
//!    = name.replace("language_model.", "")`).
//!
//! 2. **Fused gate+up experts.** Unlike Qwen3MoE (which has separate
//!    per-expert `experts.<E>.gate_proj` / `experts.<E>.up_proj` /
//!    `experts.<E>.down_proj` and needs MoE-stacking by the
//!    orchestrator), Gemma 4 ships **pre-fused** 3-D tensors:
//!      - `experts.gate_up_proj` shape `[n_experts, 2*moe_ffn, hidden]`
//!        (gate and up concatenated along the inner axis)
//!      - `experts.down_proj` shape `[n_experts, hidden, moe_ffn]`
//!    These map to the single GGUF tensor `blk.<N>.ffn_gate_up_exps.weight`
//!    (per `MODEL_TENSOR.FFN_GATE_UP_EXP` /
//!    `gguf-py/gguf/constants.py:1086`, `gguf-py/gguf/tensor_mapping.py:593`)
//!    and `blk.<N>.ffn_down_exps.weight` — NO stacking needed.
//!    The mapper returns `Direct` for both; the orchestrator's
//!    shape-reverse (`[E, 2*ffn, h] → [h, 2*ffn, E]` and `[E, h, ffn] →
//!    [ffn, h, E]`) lands the GGUF layout llama.cpp's gemma4 loader
//!    expects.
//!
//! 3. **Four norm pairs per block (Gemma 4 hybrid attention + MoE).**
//!    HF                                                  GGUF
//!    `input_layernorm`                                   `attn_norm`
//!    `post_attention_layernorm`                          `post_attention_norm`
//!    `pre_feedforward_layernorm`                         `ffn_norm`        (FFN_PRE_NORM)
//!    `pre_feedforward_layernorm_2`                       `pre_ffw_norm_2`  (FFN_PRE_NORM_2; gemma4)
//!    `post_feedforward_layernorm`                        `post_ffw_norm`   (FFN_POST_NORM)
//!    `post_feedforward_layernorm_1`                      `post_ffw_norm_1` (FFN_POST_NORM_1; gemma4)
//!    `post_feedforward_layernorm_2`                      `post_ffw_norm_2` (FFN_POST_NORM_2; gemma4)
//!    The trailing-digit variants are unique to Gemma 4 and surface at
//!    llama.cpp's loader as `MODEL_TENSOR.FFN_{PRE,POST}_NORM_{1,2}`
//!    (`gguf-py/gguf/constants.py:552-555`, `:1069-1071`).
//!
//! 4. **Parallel dense FFN _and_ routed experts.** Every block carries
//!    BOTH `mlp.{gate,up,down}_proj.weight` (dense SwiGLU FFN with dim
//!    `intermediate_size`) AND the routed `experts.{gate_up,down}_proj`
//!    pair (dim `moe_intermediate_size`, per-expert). llama.cpp's
//!    `MODEL_ARCH.GEMMA4` includes both `FFN_{GATE,DOWN,UP}` and
//!    `FFN_{GATE_UP,DOWN}_EXP{,S}` (`constants.py:2461-2467`); the
//!    runtime sums the two branches.
//!
//! 5. **Router sub-block.** Each block has a `router.proj.weight`
//!    (`[n_experts, hidden]`) that selects per-token experts. Maps to
//!    `blk.<N>.ffn_gate_inp.weight` (the canonical
//!    `MODEL_TENSOR.FFN_GATE_INP` slot; `tensor_mapping.py:451` has the
//!    `# gemma4` comment on this exact entry). The optional scalar
//!    `router.scale` (per-channel) and `router.per_expert_scale`
//!    (per-expert) become sub-name extensions of `ffn_gate_inp` and
//!    `ffn_down_exps` respectively per `gemma.py::Gemma4Model.modify_tensors:754-765`:
//!      `router.scale`            → `blk.<N>.ffn_gate_inp.scale`
//!      `router.per_expert_scale` → `blk.<N>.ffn_down_exps.scale`
//!    (the per-expert scale folds onto the expert down-projection at
//!    runtime — see the `_generate_nvfp4_tensors` comment at
//!    `gemma.py:719-741` for the algebra).
//!
//! 6. **`layer_scalar`.** A 1-D scalar tensor (shape `[1]`) per block
//!    that gates the layer's output contribution. Maps to
//!    `MODEL_TENSOR.LAYER_OUT_SCALE` →
//!    `blk.<N>.layer_output_scale.weight`
//!    (`constants.py:1091`, `tensor_mapping.py:718`). Per
//!    `gemma.py::Gemma4Model.filter_tensors:747-748` the HF on-disk name
//!    has NO `.weight` suffix; the Python converter appends one to make
//!    it match the canonical GGUF name. We do the same here (the bare
//!    HF name without `.weight` is the canonical match key).
//!
//! 7. **Tied embeddings.** Gemma 4 ties `lm_head` to `embed_tokens`;
//!    production checkpoints OMIT `lm_head.weight`. If a fork ships
//!    it the mapper returns `None` (the caller drops the duplicate or
//!    surfaces it as `UnmappedTensor`).
//!
//! 8. **GGUF architecture string is `"gemma4"`, NOT `"gemma3"`.**
//!    `LLM_ARCH_GEMMA4` is a distinct entry at
//!    `/opt/llama.cpp/src/llama-arch.cpp:59` and `MODEL_ARCH.GEMMA4`
//!    has its own tensor list at `constants.py:2450`. The prior mapper
//!    at this path emitted `"gemma3"` per ADR-033's outdated spec — that
//!    is fixed here. hf2q's runtime already keys on `"gemma4"` (see
//!    `src/backends/gguf.rs:353,785,1999,2300,2972`,
//!    `src/quantize/ggml_quants/tensor_ref.rs:68`,
//!    `src/quantize/ggml_quants/apex/arches.rs:76`).

use crate::backends::gguf::types::MetaValue;
use crate::convert::source_reader::HfTensor;
use crate::quantize::ggml_quants::SourceDtype;

// ===========================================================================
// MappedTensor — Gemma 4 mapper output
// ===========================================================================

/// What the convert orchestrator should do with one HF tensor name once
/// the Gemma 4 mapper has classified it.
///
/// All on-disk Gemma 4 weight kinds collapse to one of these variants;
/// there is no MoE expert-stacking accumulator needed (the experts are
/// pre-fused on disk — see module-level comment #2). The `Drop` variant
/// fires for vision / audio / embed_vision tensors that the per-arch
/// mapper signs off as off-path for the text-decoder convert.
///
/// Per [[feedback-no-loop-suppression-2026-05-17]]: unrecognized names
/// return `None` from [`map_tensor_name`], NOT `Drop` — the caller MUST
/// surface unmapped tensors as a typed error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MappedTensor {
    /// 1:1 HF → GGUF rename. Carries the canonical GGUF tensor name
    /// (e.g. `blk.7.attn_q.weight`, `token_embd.weight`,
    /// `blk.0.ffn_gate_up_exps.weight`).
    Direct(String),

    /// Explicitly-discardable HF tensor. Used ONLY for tensors that
    /// belong to a sidecar / off-path modality (vision tower, embed
    /// vision, audio tower, std_bias / std_scale, etc.). The caller
    /// drops these silently because the mmproj sidecar / multimodal
    /// pipeline owns them — they are NOT part of the text-decoder
    /// GGUF output. Mirrors `Gemma4Model.filter_tensors` rejecting
    /// vision / audio paths in the text-only convert.
    Drop,
}

// ===========================================================================
// map_tensor_name — canonical entry point
// ===========================================================================

/// Translate one HuggingFace tensor name to its [`MappedTensor`] outcome
/// for Gemma 4. Returns `None` if `hf_name` is not one of the Gemma 4
/// text-decoder weight kinds.
///
/// Full Gemma 4 inventory (cross-checked against
/// `/opt/hf2q/models/google-gemma-4-26b-a4b-it/model.safetensors.index.json`
/// 2026-05-18 — 30-layer, 128-expert text decoder + vision/audio
/// sidecars):
///
/// | HF name (after `model.language_model.` strip)                       | Outcome                                            |
/// |---------------------------------------------------------------------|----------------------------------------------------|
/// | `model.embed_tokens.weight`                                         | `Direct("token_embd.weight")`                      |
/// | `model.norm.weight`                                                 | `Direct("output_norm.weight")`                     |
/// | `lm_head.weight` *(usually absent — tied embeddings)*               | `Direct("output.weight")`                          |
/// | `model.layers.<N>.input_layernorm.weight`                           | `Direct("blk.<N>.attn_norm.weight")`               |
/// | `model.layers.<N>.post_attention_layernorm.weight`                  | `Direct("blk.<N>.post_attention_norm.weight")`     |
/// | `model.layers.<N>.pre_feedforward_layernorm.weight`                 | `Direct("blk.<N>.ffn_norm.weight")`                |
/// | `model.layers.<N>.pre_feedforward_layernorm_2.weight`               | `Direct("blk.<N>.pre_ffw_norm_2.weight")`          |
/// | `model.layers.<N>.post_feedforward_layernorm.weight`                | `Direct("blk.<N>.post_ffw_norm.weight")`           |
/// | `model.layers.<N>.post_feedforward_layernorm_1.weight`              | `Direct("blk.<N>.post_ffw_norm_1.weight")`         |
/// | `model.layers.<N>.post_feedforward_layernorm_2.weight`              | `Direct("blk.<N>.post_ffw_norm_2.weight")`         |
/// | `model.layers.<N>.self_attn.q_proj.weight`                          | `Direct("blk.<N>.attn_q.weight")`                  |
/// | `model.layers.<N>.self_attn.k_proj.weight`                          | `Direct("blk.<N>.attn_k.weight")`                  |
/// | `model.layers.<N>.self_attn.v_proj.weight` *(KV-shared layers omit)*| `Direct("blk.<N>.attn_v.weight")`                  |
/// | `model.layers.<N>.self_attn.o_proj.weight`                          | `Direct("blk.<N>.attn_output.weight")`             |
/// | `model.layers.<N>.self_attn.q_norm.weight`                          | `Direct("blk.<N>.attn_q_norm.weight")`             |
/// | `model.layers.<N>.self_attn.k_norm.weight`                          | `Direct("blk.<N>.attn_k_norm.weight")`             |
/// | `model.layers.<N>.mlp.gate_proj.weight`                             | `Direct("blk.<N>.ffn_gate.weight")`                |
/// | `model.layers.<N>.mlp.up_proj.weight`                               | `Direct("blk.<N>.ffn_up.weight")`                  |
/// | `model.layers.<N>.mlp.down_proj.weight`                             | `Direct("blk.<N>.ffn_down.weight")`                |
/// | `model.layers.<N>.experts.gate_up_proj` *(no .weight on disk)*      | `Direct("blk.<N>.ffn_gate_up_exps.weight")`        |
/// | `model.layers.<N>.experts.down_proj` *(no .weight on disk)*         | `Direct("blk.<N>.ffn_down_exps.weight")`           |
/// | `model.layers.<N>.router.proj.weight`                               | `Direct("blk.<N>.ffn_gate_inp.weight")`            |
/// | `model.layers.<N>.router.scale` *(no .weight on disk)*              | `Direct("blk.<N>.ffn_gate_inp.scale")`             |
/// | `model.layers.<N>.router.per_expert_scale` *(no .weight on disk)*   | `Direct("blk.<N>.ffn_down_exps.scale")`            |
/// | `model.layers.<N>.layer_scalar` *(no .weight on disk)*              | `Direct("blk.<N>.layer_output_scale.weight")`      |
///
/// Off-path (returns `Drop` — caller silently discards; these belong to
/// the mmproj / audio sidecar pipelines, not the text-decoder GGUF):
///   - `model.vision_tower.<*>` — SigLIP vision encoder
///   - `model.embed_vision.<*>` — multimodal projection
///   - `model.audio_tower.<*>` — audio encoder (when present)
///   - `model.embed_audio.<*>` — audio projection (when present)
///   - `vision_model.<*>` / `audio_model.<*>` — alternate top-level
///     packaging seen on some Gemma 4 forks
///
/// Names returning `None` (caller errors out per
/// [[feedback-no-loop-suppression-2026-05-17]]):
///   - Linear biases (Gemma 4 has no biases).
///   - Per-block names with malformed integer indices (leading zeros,
///     signs).
///   - Any per-block suffix not in the table above.
pub fn map_tensor_name(hf_name: &str) -> Option<MappedTensor> {
    // ---- 1. Off-path vision / audio early-drop -----------------------------
    //
    // Vision tower / embed_vision / audio tower are handled by the
    // mmproj / audio sidecar pipelines — NOT the text-decoder GGUF this
    // mapper drives. Surface them as Drop so the caller's
    // `UnmappedTensor` error path doesn't trip on them; the
    // `is_vision_tensor_pattern` predicate in the inference path
    // already routes runtime loads correctly.
    if is_offpath_modality_tensor(hf_name) {
        return Some(MappedTensor::Drop);
    }

    // ---- 2. Multimodal wrapper strip ---------------------------------------
    //
    // Real Gemma 4 nests the text decoder under `model.language_model.`.
    // We strip that prefix so the rest of the match table is shape-shared
    // with Llama-3 / Qwen3MoE conventions (`model.<rest>`). Mirrors the
    // canonical Python strip at `conversion/base.py:540-541`.
    //
    // After strip, `model.language_model.foo` becomes `model.foo`, which
    // the per-block parser below recognizes. If the name didn't carry
    // the wrapper (older Gemma 4 forks pre-multimodal-bundle, or pure
    // language-model checkpoints), it falls through unchanged.
    let working_cow = strip_language_model_prefix(hf_name);
    let working: &str = working_cow.as_ref();

    // ---- 3. Globals -------------------------------------------------------
    match working {
        "model.embed_tokens.weight" => {
            return Some(MappedTensor::Direct("token_embd.weight".to_string()));
        }
        "model.norm.weight" => {
            return Some(MappedTensor::Direct("output_norm.weight".to_string()));
        }
        // Tied-embedding default omits lm_head; if a checkpoint ships it
        // anyway, map cleanly so the caller can decide whether to drop
        // the duplicate at the safetensors-load layer.
        "lm_head.weight" => {
            return Some(MappedTensor::Direct("output.weight".to_string()));
        }
        // Synthesized `rope_freqs.weight` — appended by
        // `build_synthesized_tensors` and fed back through this mapper
        // along the same `MapOutcome::Direct` path as on-disk tensors.
        // The 1:1 identity match here is intentional: it isolates the
        // synthesized name from any future HF-side rename collisions and
        // documents the convert-side ownership of this tensor's name.
        GEMMA4_ROPE_FREQS_TENSOR_NAME => {
            return Some(MappedTensor::Direct(
                GEMMA4_ROPE_FREQS_TENSOR_NAME.to_string(),
            ));
        }
        _ => {}
    }

    // ---- 4. Per-block: `model.layers.<N>.<rest>` --------------------------
    let stripped = working.strip_prefix("model.layers.")?;
    let dot = stripped.find('.')?;
    let (layer_str, rest_with_dot) = stripped.split_at(dot);
    // Strict layer-index parse (no leading zeros / signs) — matches
    // Llama-3 / Qwen3MoE / MiniMax-M2 policy.
    let layer: usize = layer_str.parse().ok()?;
    if layer.to_string() != layer_str {
        return None;
    }
    let rest = &rest_with_dot[1..]; // skip the dot

    let gguf_suffix: &str = match rest {
        // ---- 4a. Norm pairs (Gemma 4 quirk #3) -----------------------------
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "post_attention_norm.weight",

        // FFN_PRE_NORM + FFN_PRE_NORM_2 (the latter is gemma4-only).
        "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
        "pre_feedforward_layernorm_2.weight" => "pre_ffw_norm_2.weight",

        // FFN_POST_NORM + FFN_POST_NORM_1 + FFN_POST_NORM_2 (latter two
        // gemma4-only). Order matters: longer suffixes must be matched
        // before shorter (`_2` and `_1` before bare). Rust's match on
        // string slices is exact, so the longer arms appear first
        // explicitly — no priority bug possible.
        "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
        "post_feedforward_layernorm_1.weight" => "post_ffw_norm_1.weight",
        "post_feedforward_layernorm_2.weight" => "post_ffw_norm_2.weight",

        // ---- 4b. Attention projections (same as Llama-3) -------------------
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",

        // Per-head Q/K norms (Gemma 4 quirk #3 part b — same as Qwen3).
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",

        // ---- 4c. Parallel dense FFN (Gemma 4 quirk #4) --------------------
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",

        // ---- 4d. Fused MoE experts (Gemma 4 quirk #2) ---------------------
        //
        // `gate_up_proj` is pre-fused gate+up `[n_experts, 2*moe_ffn,
        // hidden]` and lands on the canonical `MODEL_TENSOR.FFN_GATE_UP_EXP`
        // GGUF tensor (`constants.py:1086 — "blk.{bid}.ffn_gate_up_exps"`).
        // NO weight suffix on disk; we synthesize `.weight` at the GGUF
        // side to match the canonical name convention.
        //
        // `down_proj` is the standard FFN_DOWN_EXP (`constants.py:1085 —
        // "blk.{bid}.ffn_down_exps"`). Also NO weight suffix on disk.
        "experts.gate_up_proj" => "ffn_gate_up_exps.weight",
        "experts.down_proj" => "ffn_down_exps.weight",

        // ---- 4e. Router sub-block (Gemma 4 quirk #5) ----------------------
        //
        // `router.proj.weight` is the per-block expert-selection
        // projection — the standard `MODEL_TENSOR.FFN_GATE_INP`
        // (`tensor_mapping.py:451 — "model.layers.{bid}.router.proj" #
        // gemma4`).
        "router.proj.weight" => "ffn_gate_inp.weight",

        // `router.scale` is a per-channel (hidden-dim) scalar gate.
        // `gemma.py::Gemma4Model.modify_tensors:755-757` re-targets it as
        // a `.scale` sub-name of FFN_GATE_INP (the format_tensor_name
        // call passes `suffix=".scale"`). We mirror exactly.
        "router.scale" => "ffn_gate_inp.scale",

        // `router.per_expert_scale` is a `[n_experts]` vector that folds
        // onto each expert's down-projection at runtime. Per
        // `gemma.py::Gemma4Model.modify_tensors:759-762` it's a `.scale`
        // sub-name of FFN_DOWN_EXP.
        "router.per_expert_scale" => "ffn_down_exps.scale",

        // ---- 4f. Layer scalar (Gemma 4 quirk #6) --------------------------
        //
        // 1-D scalar per layer; HF stores without `.weight` suffix and
        // Python's `filter_tensors:747-748` appends one. The GGUF name
        // is `MODEL_TENSOR.LAYER_OUT_SCALE` →
        // `blk.{bid}.layer_output_scale.weight` (`constants.py:1091`).
        "layer_scalar" => "layer_output_scale.weight",

        _ => return None,
    };

    Some(MappedTensor::Direct(format!("blk.{layer}.{gguf_suffix}")))
}

/// Strip the multimodal-wrapper `language_model.` prefix when present,
/// returning the inner text-decoder name. Mirrors
/// `conversion/base.py:540-541`:
///
/// ```python
/// if "language_model." in name:
///     name = name.replace("language_model.", "")
/// ```
///
/// Concretely: `model.language_model.embed_tokens.weight` →
/// `model.embed_tokens.weight`. Names that DON'T contain
/// `language_model.` pass through unchanged. We use a `Cow`-less return
/// (always `&str`) by exploiting the fact that the inner slice is
/// already canonical when no strip is needed.
fn strip_language_model_prefix(hf_name: &str) -> std::borrow::Cow<'_, str> {
    if let Some(idx) = hf_name.find("language_model.") {
        // Concatenate the bytes before `language_model.` with the bytes
        // after, dropping the wrapper. Allocates only when needed.
        let head = &hf_name[..idx];
        let tail = &hf_name[idx + "language_model.".len()..];
        std::borrow::Cow::Owned(format!("{head}{tail}"))
    } else {
        std::borrow::Cow::Borrowed(hf_name)
    }
}

/// Predicate: should this HF tensor name be silently dropped by the
/// gemma4 text-decoder mapper because it belongs to a sidecar modality
/// (vision tower / audio tower / embed_vision / embed_audio)?
///
/// The convert-v2 pipeline emits the text-decoder GGUF only; sidecar
/// modalities are produced by separate mmproj / audio-tower convert
/// passes. Returning `Drop` (rather than `None`) signals to the driver
/// that the discard is sanctioned by the per-arch mapper, not an
/// unmapped-tensor bug.
///
/// Matches:
///   - `model.vision_tower.<*>` — SigLIP encoder + std_bias/std_scale
///   - `model.embed_vision.<*>` — vision→text projection
///   - `model.audio_tower.<*>` — Conformer audio encoder
///   - `model.embed_audio.<*>` — audio→text projection
///   - `vision_model.<*>` — alternate naming on some forks
///   - `audio_model.<*>` — alternate naming on some forks
fn is_offpath_modality_tensor(hf_name: &str) -> bool {
    hf_name.contains("model.vision_tower.")
        || hf_name.contains("model.embed_vision.")
        || hf_name.contains("model.audio_tower.")
        || hf_name.contains("model.embed_audio.")
        || hf_name.starts_with("vision_model.")
        || hf_name.starts_with("audio_model.")
}

// ===========================================================================
// build_metadata — GGUF KV pairs for Gemma 4
// ===========================================================================

/// Build the GGUF metadata KV pairs for a Gemma 4 model from its HF
/// `config.json`. Port of `conversion/gemma.py::Gemma4Model::set_gguf_parameters`
/// (`:655-700`) + the inherited `Gemma3Model.set_gguf_parameters` +
/// `TextModel.set_gguf_parameters` chain.
///
/// `general.architecture = "gemma4"`. KV prefix is `gemma4.*`. Mirrors
/// `MODEL_ARCH_NAMES[MODEL_ARCH.GEMMA4]` at `constants.py:947`.
///
/// **Caller flatten contract.** `config` is expected to be the **inner**
/// text-decoder config (post-`text_config` flatten). The driver at
/// `src/convert/cli_driver.rs::effective_config` performs this flatten
/// before calling `build_metadata_for_arch`, so callers that route
/// through the driver do not need to handle it here. If a raw outer
/// multimodal-wrapper config is passed, the `[]`-indexed required keys
/// will panic from missing fields — that's the canonical missing-config
/// signal per the per-arch mapper convention.
///
/// Required HF keys (mandatory; missing → panic from `[]` indexing):
///   - `hidden_size`
///   - `num_hidden_layers`
///   - `intermediate_size`         (dense FFN dim, parallel to MoE)
///   - `num_attention_heads`
///   - `max_position_embeddings`
///   - `rms_norm_eps`
///   - `head_dim`                  (sliding-window head dim — Gemma quirk)
///   - `num_kv_shared_layers`      (Gemma 4 KV-sharing count;
///                                  `gemma.py:659`)
///   - `layer_types`               (per-layer SWA/full classification;
///                                  `gemma.py:665-666`)
///
/// Optional keys (defaulted, mirroring `Gemma4Model.set_gguf_parameters`):
///   - `num_key_value_heads`         — defaults to `num_attention_heads`
///   - `global_head_dim`             — defaults to `head_dim` (Gemma 4-specific
///                                      override for global-attention layers)
///   - `moe_intermediate_size`       — per-expert FFN dim; absent on dense
///                                      Gemma 4 forks
///   - `num_experts`                 — MoE-only; absent on dense forks
///   - `top_k_experts`               — per-token activated experts; absent
///                                      on dense forks
///   - `sliding_window`              — defaults to 4096
///   - `rope_theta`                  — defaults to 10000.0. Gemma 4 hides
///                                      this inside `rope_parameters`; we
///                                      look there too.
///   - `_name_or_path`               — defaults to `"model"`
///   - `hidden_size_per_layer_input` — defaults to 0 (`gemma.py:663` — the
///                                      `or 0` fallback when the field is
///                                      absent on dense forks).
///   - `num_global_key_value_heads`  — when present alongside
///                                      `num_key_value_heads` AND differing,
///                                      `head_count_kv` becomes an array
///                                      (`gemma.py:687-691`).
///   - `use_double_wide_mlp`         — defaults to `false`; when `true`,
///                                      `feed_forward_length` becomes a
///                                      per-layer array (`gemma.py:680-684`).
///   - `rope_parameters.full_attention.partial_rotary_factor` — defaults
///                                      to 1.0 for the swa rope dim (the
///                                      gemma.py path uses
///                                      `hparams.get("partial_rotary_factor", 1.0)`
///                                      — that's a top-level fallback; we
///                                      mirror it for compatibility with
///                                      older forks).
///
/// `file_type` is the chosen `LlamaFtype` as a `u32`.
pub fn build_metadata(config: &serde_json::Value, file_type: u32) -> Vec<(String, MetaValue)> {
    let name = config
        .get("_name_or_path")
        .and_then(|v| v.as_str())
        .unwrap_or("model")
        .to_string();

    let hidden_size = config["hidden_size"]
        .as_u64()
        .expect("config.json missing required key `hidden_size`") as u32;
    let n_layers = config["num_hidden_layers"]
        .as_u64()
        .expect("config.json missing required key `num_hidden_layers`") as u32;
    let ffn_len = config["intermediate_size"]
        .as_u64()
        .expect("config.json missing required key `intermediate_size`") as u32;
    let n_head = config["num_attention_heads"]
        .as_u64()
        .expect("config.json missing required key `num_attention_heads`") as u32;
    let ctx_len = config["max_position_embeddings"]
        .as_u64()
        .expect("config.json missing required key `max_position_embeddings`") as u32;
    let rms_eps = config["rms_norm_eps"]
        .as_f64()
        .expect("config.json missing required key `rms_norm_eps`") as f32;
    let head_dim = config["head_dim"]
        .as_u64()
        .expect("config.json missing required key `head_dim`") as u32;

    let n_head_kv = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(n_head);
    let global_head_dim = config
        .get("global_head_dim")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(head_dim);
    let sliding_window = config
        .get("sliding_window")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(4096);
    let rope_theta = resolve_rope_theta(config);

    // ---- Gemma 4-specific required hparams --------------------------------
    //
    // `num_kv_shared_layers` is REQUIRED by `Gemma4Model.set_gguf_parameters`
    // (`gemma.py:658-660`): the model loader uses the count to know how many
    // tail layers reuse the prior layer's K/V instead of allocating their
    // own. Per [[feedback-no-loop-suppression-2026-05-17]] we surface
    // missing-required via the `.expect()` panic convention used elsewhere
    // in this file (matches the "Caller flatten contract" doc above).
    let num_kv_shared_layers = config["num_kv_shared_layers"]
        .as_u64()
        .expect("config.json missing required key `num_kv_shared_layers`")
        as u32;

    // `layer_types` is REQUIRED by `Gemma4Model.set_gguf_parameters`
    // (`gemma.py:665-666`): an array of strings, one per block, each either
    // `"sliding_attention"` or `"full_attention"`. We classify each entry as
    // a boolean (`is_swa = (t == "sliding_attention")`) — the runtime needs
    // the per-layer SWA mask to dispatch the right attention kernel.
    let layer_types_raw = config["layer_types"]
        .as_array()
        .expect("config.json missing required key `layer_types` (array)");
    if layer_types_raw.len() as u32 != n_layers {
        panic!(
            "config.json `layer_types` array length {} does not match \
             `num_hidden_layers` {}",
            layer_types_raw.len(),
            n_layers,
        );
    }
    let sliding_window_pattern: Vec<bool> = layer_types_raw
        .iter()
        .map(|v| {
            v.as_str()
                .expect("`layer_types[i]` must be a string")
                == "sliding_attention"
        })
        .collect();

    // ---- Gemma 4-specific optional hparams --------------------------------
    //
    // `hidden_size_per_layer_input` is OPTIONAL with a documented `or 0`
    // default at `gemma.py:663` (`self.hparams.get("hidden_size_per_layer_input") or 0`).
    let hidden_size_per_layer_input = config
        .get("hidden_size_per_layer_input")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(0);

    // RoPE dims for global vs SWA layers (`gemma.py:695-700`):
    //   - n_rot_full = int(global_head_dim)   — proportional rope; the
    //     unrotated dims are zeroed via the synthesized `rope_freqs.weight`
    //     tensor (see `build_synthesized_tensors`).
    //   - n_rot_swa  = int(head_dim * partial_rotary_factor_swa)
    // `partial_rotary_factor` defaults to 1.0 per `gemma.py:696`
    // (`self.hparams.get("partial_rotary_factor", 1.0)`).
    let partial_rotary_factor_swa = config
        .get("partial_rotary_factor")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let n_rot_full = global_head_dim;
    let n_rot_swa = (head_dim as f64 * partial_rotary_factor_swa) as u32;

    let mut kv: Vec<(String, MetaValue)> = vec![
        (
            "general.architecture".into(),
            MetaValue::String("gemma4".into()),
        ),
        ("general.name".into(), MetaValue::String(name)),
        ("gemma4.context_length".into(), MetaValue::U32(ctx_len)),
        (
            "gemma4.embedding_length".into(),
            MetaValue::U32(hidden_size),
        ),
        ("gemma4.block_count".into(), MetaValue::U32(n_layers)),
        (
            "gemma4.attention.head_count".into(),
            MetaValue::U32(n_head),
        ),
        (
            "gemma4.attention.key_length".into(),
            MetaValue::U32(global_head_dim),
        ),
        (
            "gemma4.attention.value_length".into(),
            MetaValue::U32(global_head_dim),
        ),
        (
            "gemma4.attention.key_length_swa".into(),
            MetaValue::U32(head_dim),
        ),
        (
            "gemma4.attention.value_length_swa".into(),
            MetaValue::U32(head_dim),
        ),
        (
            "gemma4.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F32(rms_eps),
        ),
        ("gemma4.rope.freq_base".into(), MetaValue::F32(rope_theta)),
        (
            "gemma4.attention.sliding_window".into(),
            MetaValue::U32(sliding_window),
        ),
        // ---- Gemma 4-specific KV (gemma.py:659-700) -----------------------
        //
        // `shared_kv_layers` count — runtime needs this to know how many
        // tail layers reuse the prior layer's K/V cache. Per
        // `gemma.py:660` + `llama-arch.cpp:245` canonical key
        // `%s.attention.shared_kv_layers`.
        (
            "gemma4.attention.shared_kv_layers".into(),
            MetaValue::U32(num_kv_shared_layers),
        ),
        // Per-layer extra-embedding width. `gemma.py:663` gates this on
        // `hparams.get(...) or 0` — present even when zero (`gemma4` real
        // checkpoint has `hidden_size_per_layer_input: 0`). Canonical key
        // `%s.embedding_length_per_layer_input` per `llama-arch.cpp:170`.
        (
            "gemma4.embedding_length_per_layer_input".into(),
            MetaValue::U32(hidden_size_per_layer_input),
        ),
        // Per-layer SWA mask as array-of-bool. `gemma.py:665-666` —
        // `[t == "sliding_attention" for t in self.hparams["layer_types"]]`.
        // Canonical key `%s.attention.sliding_window_pattern` per
        // `llama-arch.cpp:232`. Array length = `n_layers`.
        (
            "gemma4.attention.sliding_window_pattern".into(),
            MetaValue::ArrayBool(sliding_window_pattern.clone()),
        ),
        // RoPE dimension count for global-attention layers
        // (`gemma.py:697,699` — `int(head_dim_full)` since the
        // proportional-rope unrotated dims are masked via the synthesized
        // `rope_freqs.weight` tensor, not by reducing the dim).
        // Canonical key `%s.rope.dimension_count`.
        (
            "gemma4.rope.dimension_count".into(),
            MetaValue::U32(n_rot_full),
        ),
        // RoPE dimension count for sliding-window layers
        // (`gemma.py:696,698,700` — `int(head_dim_swa * partial_rotary_factor_swa)`).
        // Canonical key `%s.rope.dimension_count_swa` per
        // `llama-arch.cpp:248`.
        (
            "gemma4.rope.dimension_count_swa".into(),
            MetaValue::U32(n_rot_swa),
        ),
    ];

    // ---- feed_forward_length — scalar vs array (gemma.py:680-684) --------
    //
    // When `use_double_wide_mlp` is `true` (Gemma 4-specific dwm tier),
    // the last `num_kv_shared_layers` block entries are 2*n_ff and the
    // rest are n_ff. Default `false` emits the canonical scalar.
    let use_double_wide_mlp = config
        .get("use_double_wide_mlp")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if use_double_wide_mlp {
        let first_kv_shared_layer_idx = n_layers.saturating_sub(num_kv_shared_layers);
        let n_ff_arr: Vec<u32> = (0..n_layers)
            .map(|il| {
                if il < first_kv_shared_layer_idx {
                    ffn_len
                } else {
                    ffn_len * 2
                }
            })
            .collect();
        kv.push((
            "gemma4.feed_forward_length".into(),
            MetaValue::ArrayU32(n_ff_arr),
        ));
    } else {
        kv.push((
            "gemma4.feed_forward_length".into(),
            MetaValue::U32(ffn_len),
        ));
    }

    // ---- head_count_kv — scalar vs array (gemma.py:687-691) --------------
    //
    // Per-layer KV head count differs between global and sliding-window
    // layers when `num_global_key_value_heads` is present AND differs from
    // `num_key_value_heads`. Array entries are
    // `num_key_value_heads if is_swa else num_global_key_value_heads`.
    let num_global_kv = config
        .get("num_global_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32);
    if let Some(num_global_kv) = num_global_kv {
        if num_global_kv != n_head_kv {
            let head_count_kv_arr: Vec<u32> = sliding_window_pattern
                .iter()
                .map(|&is_swa| if is_swa { n_head_kv } else { num_global_kv })
                .collect();
            kv.push((
                "gemma4.attention.head_count_kv".into(),
                MetaValue::ArrayU32(head_count_kv_arr),
            ));
        } else {
            // Present but equal — scalar form is byte-equivalent.
            kv.push((
                "gemma4.attention.head_count_kv".into(),
                MetaValue::U32(n_head_kv),
            ));
        }
    } else {
        kv.push((
            "gemma4.attention.head_count_kv".into(),
            MetaValue::U32(n_head_kv),
        ));
    }

    // ---- MoE-only KV (skipped on dense Gemma 4 forks) --------------------
    //
    // `num_experts` is the canonical Gemma 4 HF key (NOT `num_local_experts`
    // — that's Qwen / Mixtral). `top_k_experts` is the per-token activated
    // count (Gemma 4-specific name; llama.cpp normalizes to `expert_used_count`
    // metadata regardless of HF key).
    if let Some(n_experts) = config.get("num_experts").and_then(|v| v.as_u64()) {
        kv.push((
            "gemma4.expert_count".into(),
            MetaValue::U32(n_experts as u32),
        ));
    }
    if let Some(top_k) = config
        .get("top_k_experts")
        .or_else(|| config.get("num_experts_per_tok"))
        .and_then(|v| v.as_u64())
    {
        kv.push((
            "gemma4.expert_used_count".into(),
            MetaValue::U32(top_k as u32),
        ));
    }
    if let Some(moe_ffn) = config
        .get("moe_intermediate_size")
        .or_else(|| config.get("expert_intermediate_size"))
        .and_then(|v| v.as_u64())
    {
        kv.push((
            "gemma4.expert_feed_forward_length".into(),
            MetaValue::U32(moe_ffn as u32),
        ));
    }

    kv.push(("general.file_type".into(), MetaValue::U32(file_type)));

    kv
}

// ===========================================================================
// build_synthesized_tensors — extra tensors NOT in safetensors
// ===========================================================================

/// Canonical GGUF tensor name for the Gemma 4 ROPE_FREQS table.
/// Matches `gguf-py/gguf/constants.py:1047` (`"rope_freqs"`) plus the
/// canonical `.weight` suffix appended by `format_tensor_name`.
pub const GEMMA4_ROPE_FREQS_TENSOR_NAME: &str = "rope_freqs.weight";

/// Build the list of synthesized tensors that Gemma 4 needs but are NOT
/// present in the safetensors. Port of
/// `conversion/gemma.py::Gemma4Model::generate_extra_tensors`
/// (`:702-718`).
///
/// **`rope_freqs.weight`** (REQUIRED, F32, shape `[global_head_dim / 2]`):
/// Gemma 4 full-attention layers use `rope_type = "proportional"` with a
/// `partial_rotary_factor < 1.0`. The expected ordering of cos/sin/zero
/// dims is `cc000000ss000000` but ggml's neox RoPE kernel only supports
/// `ccss000000000000` and we can't rearrange the head without breaking
/// `use_alternative_attention`. The fix is a per-dim freq_factors table:
/// the first `n_rot_full = int(global_head_dim * partial_rotary_factor_full / 2)`
/// entries are `1.0` (rotate normally) and the remaining
/// `int(global_head_dim / 2) - n_rot_full` entries are `1e30` (collapse
/// to zero rotation — these are the unrotated dims).
///
/// Per `gemma.py:715-717`:
/// ```python
/// values = [1.0] * n_rot_full + [1e30] * n_unrot_full
/// rope_freqs_full = torch.tensor(values, dtype=torch.float32)
/// yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), rope_freqs_full)
/// ```
///
/// Required HF keys (mandatory; missing → panic):
///   - `global_head_dim` *(optional in `build_metadata` — defaults to
///     `head_dim` there — but here we require it because the
///     proportional-rope path is only meaningful when there is an
///     explicit global head dim)*. We mirror `build_metadata`'s default
///     so synthesis succeeds on Gemma 3-shape forks without the override.
///   - `rope_parameters.full_attention.partial_rotary_factor` *(REQUIRED;
///     `gemma.py:712` `assert rope_params_full["rope_type"] == "proportional"`
///     and `partial_rotary_factor_full = rope_params_full["partial_rotary_factor"]`)*.
///
/// The returned tensor is in PyTorch-shape order (matches `HfTensor`
/// convention; the orchestrator reverses to GGUF order at the boundary).
/// Source dtype is `F32` because the synthesized values are exact —
/// there's no on-disk source to dequantize from.
pub fn build_synthesized_tensors(config: &serde_json::Value) -> Vec<HfTensor> {
    let head_dim = config["head_dim"]
        .as_u64()
        .expect("config.json missing required key `head_dim`")
        as u32;
    let global_head_dim = config
        .get("global_head_dim")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(head_dim);

    // `partial_rotary_factor_full` from `rope_parameters.full_attention`.
    // `gemma.py:705` requires `rope_type == "proportional"`; if the
    // checkpoint lacks the nested dict we treat that as a malformed Gemma
    // 4 config (per [[feedback-no-loop-suppression-2026-05-17]] — no
    // silent default to 1.0 which would emit a no-op table).
    let rope_params_full = config
        .get("rope_parameters")
        .and_then(|v| v.get("full_attention"))
        .expect(
            "config.json missing required path \
             `rope_parameters.full_attention` (required for \
             ROPE_FREQS synthesis — see gemma.py:704-717)",
        );
    // gemma.py:705 — `assert rope_params_full["rope_type"] == "proportional"`.
    // The synthesized [1.0, …, 1e30, …] mask is mathematically only valid
    // for the proportional-rope variant; any other rope_type means the
    // synthesized table would be wrong, so this MUST hard-error rather
    // than silently emit it.
    let rope_type = rope_params_full
        .get("rope_type")
        .and_then(|v| v.as_str())
        .expect(
            "config.json missing required key \
             `rope_parameters.full_attention.rope_type` \
             (see gemma.py:705)",
        );
    assert_eq!(
        rope_type, "proportional",
        "ROPE_FREQS synthesis only valid for rope_type=proportional, got `{rope_type}` (gemma.py:705)"
    );
    let partial_rotary_factor_full = rope_params_full
        .get("partial_rotary_factor")
        .and_then(|v| v.as_f64())
        .expect(
            "config.json missing required key \
             `rope_parameters.full_attention.partial_rotary_factor`",
        );

    // `gemma.py:713-715` — exact formula from the reference.
    // `n_rot_full + n_unrot_full == table_len` by construction (the reference
    // computes `n_unrot_full = head_dim_full/2 - n_rot_full` as a direct
    // subtraction; any underflow would be a malformed config, not a runtime
    // recoverable state). Asserting matches gemma.py's implicit precondition
    // — `saturating_sub` would silently truncate a bad partial_rotary_factor.
    let table_len = (global_head_dim / 2) as usize;
    let n_rot_full = (global_head_dim as f64 * partial_rotary_factor_full / 2.0) as usize;
    assert!(
        n_rot_full <= table_len,
        "ROPE_FREQS synthesis: n_rot_full ({n_rot_full}) > table_len ({table_len}) — \
         invalid combination of global_head_dim={global_head_dim} and \
         partial_rotary_factor={partial_rotary_factor_full}; see gemma.py:713-715"
    );
    let n_unrot_full = table_len - n_rot_full;

    let mut values = Vec::with_capacity(table_len);
    values.extend(std::iter::repeat(1.0_f32).take(n_rot_full));
    values.extend(std::iter::repeat(1.0e30_f32).take(n_unrot_full));

    vec![HfTensor {
        name: GEMMA4_ROPE_FREQS_TENSOR_NAME.to_string(),
        shape: vec![table_len],
        source_dtype: SourceDtype::F32,
        data: values,
    }]
}

/// Resolve `rope_theta` from the HF config. Gemma 4 hides this inside
/// `rope_parameters.full_attention.rope_theta` (the nested dict that
/// also carries `partial_rotary_factor` for the proportional-rope
/// path). Older Gemma 4 forks may use the flat top-level `rope_theta`;
/// we accept either, defaulting to `10000.0` if neither is present.
fn resolve_rope_theta(config: &serde_json::Value) -> f32 {
    if let Some(rt) = config.get("rope_theta").and_then(|v| v.as_f64()) {
        return rt as f32;
    }
    if let Some(rp) = config.get("rope_parameters") {
        // Try `full_attention.rope_theta` first (canonical Gemma 4 nest).
        if let Some(rt) = rp
            .get("full_attention")
            .and_then(|v| v.get("rope_theta"))
            .and_then(|v| v.as_f64())
        {
            return rt as f32;
        }
        // Fallback: flat `rope_parameters.rope_theta` (Gemma 3-shape).
        if let Some(rt) = rp.get("rope_theta").and_then(|v| v.as_f64()) {
            return rt as f32;
        }
    }
    10000.0
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // map_tensor_name — prefix strip
    // -----------------------------------------------------------------------

    /// The mapper strips `model.language_model.` transparently so the
    /// SAME HF name (with OR without the multimodal wrapper) lands on the
    /// SAME GGUF target. Surfaced 2026-05-18 as the root mismatch the
    /// prior gemma3-shape mapper fell on.
    #[test]
    fn gemma4_strips_language_model_prefix() {
        // Globals: both forms must land on `token_embd.weight`.
        for hf in [
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        ] {
            assert_eq!(
                map_tensor_name(hf),
                Some(MappedTensor::Direct("token_embd.weight".into())),
                "{hf:?} must map to token_embd.weight"
            );
        }

        // Per-block: both forms must land on `blk.7.attn_q.weight`.
        for hf in [
            "model.layers.7.self_attn.q_proj.weight",
            "model.language_model.layers.7.self_attn.q_proj.weight",
        ] {
            assert_eq!(
                map_tensor_name(hf),
                Some(MappedTensor::Direct("blk.7.attn_q.weight".into())),
                "{hf:?} must map to blk.7.attn_q.weight"
            );
        }

        // `model.norm.weight` → `output_norm.weight` (both wrappings).
        for hf in [
            "model.norm.weight",
            "model.language_model.norm.weight",
        ] {
            assert_eq!(
                map_tensor_name(hf),
                Some(MappedTensor::Direct("output_norm.weight".into())),
            );
        }
    }

    // -----------------------------------------------------------------------
    // map_tensor_name — fused experts (Gemma 4 quirk #2)
    // -----------------------------------------------------------------------

    /// `experts.gate_up_proj` is the FUSED gate+up expert tensor;
    /// llama.cpp's MODEL_TENSOR.FFN_GATE_UP_EXP keeps it as one GGUF
    /// tensor (NO split into separate gate+up). The mapper returns one
    /// `Direct` outcome that the orchestrator emits as a single
    /// `blk.<N>.ffn_gate_up_exps.weight` of shape
    /// `[hidden, 2*moe_ffn, n_experts]` (GGUF order).
    #[test]
    fn gemma4_experts_gate_up_proj_returns_fused_direct() {
        // Both layer indices (single + multi digit) and both wrappings.
        let cases = [
            (
                "model.language_model.layers.0.experts.gate_up_proj",
                "blk.0.ffn_gate_up_exps.weight",
            ),
            (
                "model.layers.0.experts.gate_up_proj",
                "blk.0.ffn_gate_up_exps.weight",
            ),
            (
                "model.language_model.layers.15.experts.gate_up_proj",
                "blk.15.ffn_gate_up_exps.weight",
            ),
            (
                "model.language_model.layers.29.experts.gate_up_proj",
                "blk.29.ffn_gate_up_exps.weight",
            ),
        ];
        for (hf, gguf) in cases {
            assert_eq!(
                map_tensor_name(hf),
                Some(MappedTensor::Direct(gguf.into())),
                "{hf:?} → expected {gguf:?}"
            );
        }
    }

    /// `experts.down_proj` is the standard MoE down-projection;
    /// llama.cpp's MODEL_TENSOR.FFN_DOWN_EXP maps to
    /// `blk.<N>.ffn_down_exps.weight`. The HF tensor lacks the
    /// `.weight` suffix on disk (per the operator's
    /// google-gemma-4-26b-a4b-it inventory); we recognize the bare form.
    #[test]
    fn gemma4_experts_down_proj_maps_to_ffn_down_exps() {
        for hf in [
            "model.language_model.layers.0.experts.down_proj",
            "model.layers.0.experts.down_proj",
        ] {
            assert_eq!(
                map_tensor_name(hf),
                Some(MappedTensor::Direct("blk.0.ffn_down_exps.weight".into())),
                "{hf:?} → expected blk.0.ffn_down_exps.weight"
            );
        }
    }

    // -----------------------------------------------------------------------
    // map_tensor_name — four norm pairs (Gemma 4 quirk #3)
    // -----------------------------------------------------------------------

    /// `pre_feedforward_layernorm` (FFN_PRE_NORM) vs
    /// `pre_feedforward_layernorm_2` (FFN_PRE_NORM_2; gemma4-only) MUST
    /// map to DISTINCT GGUF target names. Similarly for the three
    /// post-FFN norm variants.
    #[test]
    fn gemma4_dual_norms_disambiguated() {
        let pairs: &[(&str, &str)] = &[
            (
                "model.language_model.layers.0.pre_feedforward_layernorm.weight",
                "blk.0.ffn_norm.weight",
            ),
            (
                "model.language_model.layers.0.pre_feedforward_layernorm_2.weight",
                "blk.0.pre_ffw_norm_2.weight",
            ),
            (
                "model.language_model.layers.0.post_feedforward_layernorm.weight",
                "blk.0.post_ffw_norm.weight",
            ),
            (
                "model.language_model.layers.0.post_feedforward_layernorm_1.weight",
                "blk.0.post_ffw_norm_1.weight",
            ),
            (
                "model.language_model.layers.0.post_feedforward_layernorm_2.weight",
                "blk.0.post_ffw_norm_2.weight",
            ),
        ];

        let mut seen_gguf: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (hf, gguf) in pairs {
            let got = map_tensor_name(hf);
            assert_eq!(
                got,
                Some(MappedTensor::Direct((*gguf).into())),
                "{hf:?} → expected {gguf:?}, got {got:?}"
            );
            // No two HF norm variants may share a GGUF target.
            assert!(
                seen_gguf.insert((*gguf).into()),
                "norm GGUF target {gguf:?} appears twice (load-bearing for correctness)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // map_tensor_name — router sub-block (Gemma 4 quirk #5)
    // -----------------------------------------------------------------------

    /// `router.proj.weight` → `blk.<N>.ffn_gate_inp.weight`. This is the
    /// canonical MODEL_TENSOR.FFN_GATE_INP slot (`tensor_mapping.py:451`
    /// has the `# gemma4` comment on this exact entry).
    #[test]
    fn gemma4_router_proj() {
        for (hf, gguf) in [
            (
                "model.language_model.layers.0.router.proj.weight",
                "blk.0.ffn_gate_inp.weight",
            ),
            (
                "model.layers.13.router.proj.weight",
                "blk.13.ffn_gate_inp.weight",
            ),
        ] {
            assert_eq!(
                map_tensor_name(hf),
                Some(MappedTensor::Direct(gguf.into())),
                "{hf:?} → expected {gguf:?}"
            );
        }
    }

    /// `router.scale` (per-channel hidden-dim scalar) →
    /// `blk.<N>.ffn_gate_inp.scale`, per
    /// `gemma.py::Gemma4Model.modify_tensors:755-757`.
    /// `router.per_expert_scale` ([n_experts] vector) →
    /// `blk.<N>.ffn_down_exps.scale`, per `gemma.py:759-762`. Both have
    /// NO `.weight` suffix on disk (HF stores the bare name).
    #[test]
    fn gemma4_router_scale_subnames() {
        assert_eq!(
            map_tensor_name("model.language_model.layers.0.router.scale"),
            Some(MappedTensor::Direct("blk.0.ffn_gate_inp.scale".into()))
        );
        assert_eq!(
            map_tensor_name("model.language_model.layers.7.router.per_expert_scale"),
            Some(MappedTensor::Direct("blk.7.ffn_down_exps.scale".into()))
        );
    }

    // -----------------------------------------------------------------------
    // map_tensor_name — layer_scalar (Gemma 4 quirk #6)
    // -----------------------------------------------------------------------

    /// `layer_scalar` (1-D `[1]` tensor, no `.weight` on disk) maps to
    /// `MODEL_TENSOR.LAYER_OUT_SCALE` →
    /// `blk.<N>.layer_output_scale.weight` (`constants.py:1091`,
    /// `tensor_mapping.py:718`). Python's
    /// `Gemma4Model.filter_tensors:747-748` appends the `.weight`
    /// suffix; we do the equivalent at the GGUF side.
    #[test]
    fn gemma4_layer_scalar() {
        for layer in [0usize, 15, 29] {
            let hf = format!("model.language_model.layers.{layer}.layer_scalar");
            let want = format!("blk.{layer}.layer_output_scale.weight");
            assert_eq!(
                map_tensor_name(&hf),
                Some(MappedTensor::Direct(want.clone())),
                "{hf:?} → expected {want:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // map_tensor_name — parallel dense FFN (Gemma 4 quirk #4)
    // -----------------------------------------------------------------------

    /// Gemma 4 has BOTH MoE experts AND a parallel dense FFN. The dense
    /// path lives at `mlp.{gate,up,down}_proj.weight` and maps to the
    /// canonical `blk.<N>.ffn_{gate,up,down}.weight` GGUF names —
    /// distinct from the routed-expert tensors above.
    #[test]
    fn gemma4_parallel_dense_ffn() {
        let cases = [
            (
                "model.language_model.layers.3.mlp.gate_proj.weight",
                "blk.3.ffn_gate.weight",
            ),
            (
                "model.language_model.layers.3.mlp.up_proj.weight",
                "blk.3.ffn_up.weight",
            ),
            (
                "model.language_model.layers.3.mlp.down_proj.weight",
                "blk.3.ffn_down.weight",
            ),
        ];
        for (hf, gguf) in cases {
            assert_eq!(
                map_tensor_name(hf),
                Some(MappedTensor::Direct(gguf.into())),
                "{hf:?} → expected {gguf:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // map_tensor_name — attention block (matches Qwen3 / Gemma 4 shared shape)
    // -----------------------------------------------------------------------

    /// Q/K/V/O projections + Q/K head-dim norms map cleanly. v_proj may
    /// be absent on KV-shared layers (the operator's
    /// google-gemma-4-26b-a4b-it has 25/30 v_proj — 5 layers share KV)
    /// but the mapper itself doesn't know about that; absence simply
    /// means the safetensors omitted the tensor and the mapper is never
    /// asked about it.
    #[test]
    fn gemma4_attention_block() {
        let cases = [
            (
                "model.language_model.layers.5.self_attn.q_proj.weight",
                "blk.5.attn_q.weight",
            ),
            (
                "model.language_model.layers.5.self_attn.k_proj.weight",
                "blk.5.attn_k.weight",
            ),
            (
                "model.language_model.layers.5.self_attn.v_proj.weight",
                "blk.5.attn_v.weight",
            ),
            (
                "model.language_model.layers.5.self_attn.o_proj.weight",
                "blk.5.attn_output.weight",
            ),
            (
                "model.language_model.layers.5.self_attn.q_norm.weight",
                "blk.5.attn_q_norm.weight",
            ),
            (
                "model.language_model.layers.5.self_attn.k_norm.weight",
                "blk.5.attn_k_norm.weight",
            ),
        ];
        for (hf, gguf) in cases {
            assert_eq!(
                map_tensor_name(hf),
                Some(MappedTensor::Direct(gguf.into())),
                "{hf:?} → expected {gguf:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // map_tensor_name — off-path modality drop
    // -----------------------------------------------------------------------

    /// Vision/audio sidecar tensors return `Drop` — the
    /// text-decoder convert silently discards them (they're produced by
    /// the mmproj / audio-tower convert sidecars).
    #[test]
    fn gemma4_vision_tower_returns_drop() {
        let cases = [
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
            "model.vision_tower.patch_embedder.input_proj.weight",
            "model.vision_tower.patch_embedder.position_embedding_table",
            "model.vision_tower.std_bias",
            "model.vision_tower.std_scale",
            "model.embed_vision.embedding_projection.weight",
            "model.audio_tower.encoder.layers.0.self_attn.q_proj.weight",
            "model.embed_audio.linear.weight",
            "vision_model.encoder.layers.0.attn.q_proj.weight",
            "audio_model.encoder.layers.0.attn.q_proj.weight",
        ];
        for hf in cases {
            assert_eq!(
                map_tensor_name(hf),
                Some(MappedTensor::Drop),
                "{hf:?} → expected Drop (off-path modality)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // map_tensor_name — rejection cases
    // -----------------------------------------------------------------------

    /// Unknown / out-of-scope names surface as `None`. Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: caller errors out,
    /// never silently skip.
    #[test]
    fn gemma4_tensor_name_rejects_unknown_kinds() {
        let cases = [
            // Unknown global.
            "model.unknown.weight",
            // Wrong prefix.
            "transformer.layers.0.attn.weight",
            // Gemma 4 has no biases on linear projections.
            "model.language_model.layers.0.self_attn.q_proj.bias",
            // Malformed layer index (leading zero).
            "model.language_model.layers.01.self_attn.q_proj.weight",
            // Empty layer index.
            "model.language_model.layers..self_attn.q_proj.weight",
            // No layer prefix at all.
            "model.language_model.layers.self_attn.q_proj.weight",
            // Unknown per-block suffix.
            "model.language_model.layers.0.unknown.weight",
            // Qwen2MoE-style shared-expert tensor (not present in Gemma 4).
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            // Qwen3MoE-style per-expert tensor (Gemma 4 fuses gate+up
            // — these names belong to a different arch's MoE shape).
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
        ];
        for hf in cases {
            assert_eq!(
                map_tensor_name(hf),
                None,
                "{hf:?} → expected None (unmapped)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // build_metadata
    // -----------------------------------------------------------------------

    /// Metadata round-trip with a synthetic config matching the
    /// operator's google-gemma-4-26b-a4b-it shape (post-`text_config`
    /// flatten — see the build_metadata doc-comment "Caller flatten
    /// contract"). All required KV pairs present with the right types
    /// + values.
    #[test]
    fn gemma4_metadata_from_real_config() {
        // Shape from /opt/hf2q/models/google-gemma-4-26b-a4b-it/config.json
        // (text_config sub-object). All 30 layers carry the canonical
        // SWA/full mask from the operator's checkpoint (layer indices
        // 5, 11, 17, 23, 29 are `full_attention`; rest are
        // `sliding_attention`).
        let layer_types = (0..30u32)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect::<Vec<_>>();
        let cfg = json!({
            "_name_or_path": "google/gemma-4-26b-a4b-it",
            "model_type": "gemma4_text",
            "hidden_size": 2816,
            "num_hidden_layers": 30,
            "intermediate_size": 2112,
            "moe_intermediate_size": 704,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 2,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "layer_types": layer_types,
            "use_double_wide_mlp": false,
            "head_dim": 256,
            "global_head_dim": 512,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1.0e-6,
            "sliding_window": 1024,
            "num_experts": 128,
            "top_k_experts": 4,
            "rope_parameters": {
                "full_attention": {
                    "rope_theta": 1_000_000.0,
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                }
            },
        });

        let kv = build_metadata(&cfg, 17 /* MostlyQ5_K_M */);
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        // Architecture must be gemma4 — NOT gemma3 (the prior mapper got
        // this wrong; see module-level quirk #8).
        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("gemma4".into()),
            "Gemma 4 emits general.architecture=gemma4 (LLM_ARCH_GEMMA4)"
        );
        assert_eq!(
            by_key["general.name"],
            MetaValue::String("google/gemma-4-26b-a4b-it".into())
        );
        assert_eq!(by_key["gemma4.context_length"], MetaValue::U32(262144));
        assert_eq!(by_key["gemma4.embedding_length"], MetaValue::U32(2816));
        assert_eq!(by_key["gemma4.block_count"], MetaValue::U32(30));
        assert_eq!(by_key["gemma4.attention.head_count"], MetaValue::U32(16));
        // `feed_forward_length` and `head_count_kv` shape is asserted
        // below (scalar vs array depends on use_double_wide_mlp and
        // num_global_key_value_heads).
        //
        // Global key/value length uses `global_head_dim` per
        // `Gemma4Model.set_gguf_parameters:671-672`; SWA uses `head_dim`.
        assert_eq!(by_key["gemma4.attention.key_length"], MetaValue::U32(512));
        assert_eq!(by_key["gemma4.attention.value_length"], MetaValue::U32(512));
        assert_eq!(by_key["gemma4.attention.key_length_swa"], MetaValue::U32(256));
        assert_eq!(by_key["gemma4.attention.value_length_swa"], MetaValue::U32(256));
        assert_eq!(
            by_key["gemma4.attention.layer_norm_rms_epsilon"],
            MetaValue::F32(1.0e-6)
        );
        assert_eq!(
            by_key["gemma4.rope.freq_base"],
            MetaValue::F32(1_000_000.0),
            "rope_theta must come from rope_parameters.full_attention.rope_theta"
        );
        assert_eq!(by_key["gemma4.attention.sliding_window"], MetaValue::U32(1024));
        // MoE KV
        assert_eq!(by_key["gemma4.expert_count"], MetaValue::U32(128));
        assert_eq!(by_key["gemma4.expert_used_count"], MetaValue::U32(4));
        assert_eq!(
            by_key["gemma4.expert_feed_forward_length"],
            MetaValue::U32(704)
        );
        assert_eq!(by_key["general.file_type"], MetaValue::U32(17));

        // ---- Gemma 4-specific KV (gemma.py:659-700) -----------------------
        assert_eq!(
            by_key["gemma4.attention.shared_kv_layers"],
            MetaValue::U32(0),
            "shared_kv_layers from num_kv_shared_layers"
        );
        assert_eq!(
            by_key["gemma4.embedding_length_per_layer_input"],
            MetaValue::U32(0),
            "embedding_length_per_layer_input from hidden_size_per_layer_input"
        );
        // sliding_window_pattern: array-of-bool, len=30.
        let swa_pattern = match &by_key["gemma4.attention.sliding_window_pattern"] {
            MetaValue::ArrayBool(v) => v.clone(),
            other => panic!("expected ArrayBool, got {other:?}"),
        };
        assert_eq!(swa_pattern.len(), 30, "swa pattern must have one entry per layer");
        // Layer indices 5, 11, 17, 23, 29 are `full_attention` → false; rest true.
        for (i, &is_swa) in swa_pattern.iter().enumerate() {
            let expected_swa = !matches!(i, 5 | 11 | 17 | 23 | 29);
            assert_eq!(is_swa, expected_swa, "layer {i} swa-flag mismatch");
        }
        // RoPE dims: n_rot_full=512, n_rot_swa=256*0.25=64 (but the swa
        // `partial_rotary_factor` is the top-level fallback default 1.0 in
        // build_metadata — Gemma 4 real config DOES NOT carry a top-level
        // `partial_rotary_factor`, so swa rope-dim = head_dim*1.0 = 256).
        assert_eq!(
            by_key["gemma4.rope.dimension_count"],
            MetaValue::U32(512),
            "rope.dimension_count = int(global_head_dim)"
        );
        assert_eq!(
            by_key["gemma4.rope.dimension_count_swa"],
            MetaValue::U32(256),
            "rope.dimension_count_swa = int(head_dim * partial_rotary_factor=1.0)"
        );

        // head_count_kv: array-of-u32 because num_global_key_value_heads=2 ≠
        // num_key_value_heads=8. Entries: 2 for global (full_attention)
        // layers, 8 for swa layers.
        let hck = match &by_key["gemma4.attention.head_count_kv"] {
            MetaValue::ArrayU32(v) => v.clone(),
            other => panic!("expected ArrayU32 for head_count_kv (global=2 ≠ swa=8), got {other:?}"),
        };
        assert_eq!(hck.len(), 30, "head_count_kv array length = block_count");
        for (i, &hck_i) in hck.iter().enumerate() {
            let expected = if matches!(i, 5 | 11 | 17 | 23 | 29) { 2 } else { 8 };
            assert_eq!(hck_i, expected, "layer {i} head_count_kv mismatch");
        }

        // feed_forward_length: scalar because use_double_wide_mlp=false.
        assert_eq!(
            by_key["gemma4.feed_forward_length"],
            MetaValue::U32(2112),
            "feed_forward_length scalar when use_double_wide_mlp=false"
        );
    }

    /// Optional-key defaults trigger when the HF config omits
    /// `num_key_value_heads`, `global_head_dim`, `sliding_window`,
    /// `rope_theta`, `_name_or_path`, and all MoE keys. Mirrors
    /// `Gemma4Model.set_gguf_parameters`'s `.get(...)` defaults plus the
    /// dense-Gemma 4 (no-MoE-KV) fork. `num_kv_shared_layers` +
    /// `layer_types` are REQUIRED — included with minimal values.
    #[test]
    fn gemma4_metadata_optional_key_defaults() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "head_dim": 16,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-6,
            "num_kv_shared_layers": 0,
            "layer_types": ["sliding_attention"],
            // num_key_value_heads omitted → defaults to num_attention_heads
            // global_head_dim omitted → defaults to head_dim
            // sliding_window omitted → defaults to 4096
            // rope_theta omitted → defaults to 10000.0
            // _name_or_path omitted → defaults to "model"
            // num_experts / top_k_experts omitted → dense fork, no MoE KV
            // hidden_size_per_layer_input omitted → defaults to 0
            // num_global_key_value_heads omitted → head_count_kv stays scalar
            // use_double_wide_mlp omitted → feed_forward_length stays scalar
        });
        let kv = build_metadata(&cfg, 0);
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        assert_eq!(
            by_key["general.name"],
            MetaValue::String("model".into()),
            "name defaults to 'model' when _name_or_path absent"
        );
        assert_eq!(
            by_key["gemma4.attention.head_count_kv"],
            MetaValue::U32(4),
            "num_key_value_heads defaults to num_attention_heads"
        );
        assert_eq!(
            by_key["gemma4.attention.key_length"],
            MetaValue::U32(16),
            "global_head_dim defaults to head_dim"
        );
        assert_eq!(
            by_key["gemma4.rope.freq_base"],
            MetaValue::F32(10000.0),
            "rope_theta defaults to 10000.0"
        );
        assert_eq!(
            by_key["gemma4.attention.sliding_window"],
            MetaValue::U32(4096),
            "sliding_window defaults to 4096"
        );
        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("gemma4".into()),
        );
        // MoE KV pairs absent on dense fork.
        assert!(
            !by_key.contains_key("gemma4.expert_count"),
            "expert_count must be absent when num_experts is absent"
        );
        assert!(
            !by_key.contains_key("gemma4.expert_used_count"),
            "expert_used_count must be absent when top_k_experts is absent"
        );
        assert!(
            !by_key.contains_key("gemma4.expert_feed_forward_length"),
            "expert_feed_forward_length must be absent when moe_intermediate_size is absent"
        );
        assert_eq!(by_key["general.file_type"], MetaValue::U32(0));

        // ---- New Gemma 4-specific KV defaults (dense fork) ----------------
        // shared_kv_layers always present (REQUIRED). Value 0 (no shared KV).
        assert_eq!(
            by_key["gemma4.attention.shared_kv_layers"],
            MetaValue::U32(0),
        );
        // embedding_length_per_layer_input defaults to 0 when absent.
        assert_eq!(
            by_key["gemma4.embedding_length_per_layer_input"],
            MetaValue::U32(0),
        );
        // sliding_window_pattern: 1-element array (n_layers=1, single
        // sliding layer in the fixture).
        assert_eq!(
            by_key["gemma4.attention.sliding_window_pattern"],
            MetaValue::ArrayBool(vec![true]),
        );
        // RoPE dims: global = global_head_dim = head_dim = 16; swa =
        // head_dim * 1.0 = 16 (partial_rotary_factor defaults to 1.0).
        assert_eq!(by_key["gemma4.rope.dimension_count"], MetaValue::U32(16));
        assert_eq!(by_key["gemma4.rope.dimension_count_swa"], MetaValue::U32(16));
        // head_count_kv stays scalar (num_global_key_value_heads absent).
        assert_eq!(
            by_key["gemma4.attention.head_count_kv"],
            MetaValue::U32(4),
            "head_count_kv stays scalar when num_global_key_value_heads is absent"
        );
        // feed_forward_length stays scalar (use_double_wide_mlp default false).
        assert_eq!(
            by_key["gemma4.feed_forward_length"],
            MetaValue::U32(64),
            "feed_forward_length stays scalar when use_double_wide_mlp is false"
        );
    }

    /// `rope_theta` resolution: flat top-level wins, then
    /// `rope_parameters.full_attention.rope_theta` (Gemma 4 canonical),
    /// then `rope_parameters.rope_theta` (Gemma 3 fallback), then
    /// default 10000.0.
    #[test]
    fn gemma4_resolve_rope_theta_three_paths() {
        // Path 1 — flat top-level (older Gemma 4 forks).
        let cfg = json!({ "rope_theta": 500_000.0 });
        assert_eq!(resolve_rope_theta(&cfg), 500_000.0);

        // Path 2 — canonical Gemma 4 nested.
        let cfg = json!({
            "rope_parameters": {
                "full_attention": {
                    "rope_theta": 1_000_000.0,
                }
            }
        });
        assert_eq!(resolve_rope_theta(&cfg), 1_000_000.0);

        // Path 3 — Gemma 3-shape fallback.
        let cfg = json!({
            "rope_parameters": { "rope_theta": 250_000.0 }
        });
        assert_eq!(resolve_rope_theta(&cfg), 250_000.0);

        // Default when nothing matches.
        let cfg = json!({});
        assert_eq!(resolve_rope_theta(&cfg), 10000.0);
    }

    /// `file_type` round-trips as a plain u32 in `general.file_type`
    /// (matches `gguf_writer.add_file_type(self.ftype)` at base.py).
    #[test]
    fn gemma4_metadata_ftype_round_trips() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "head_dim": 16,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-6,
            "num_kv_shared_layers": 0,
            "layer_types": ["sliding_attention"],
        });
        for &ftype in &[0u32, 1, 7, 15, 17, 23] {
            let kv = build_metadata(&cfg, ftype);
            let by_key: std::collections::HashMap<_, _> =
                kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
            assert_eq!(
                by_key["general.file_type"],
                MetaValue::U32(ftype),
                "file_type {ftype} must round-trip as MetaValue::U32"
            );
        }
    }

    // -----------------------------------------------------------------------
    // strip_language_model_prefix unit-level
    // -----------------------------------------------------------------------

    /// Concrete fixture for the prefix-strip helper. With the wrapper:
    /// drop the substring. Without: pass through unchanged.
    #[test]
    fn gemma4_strip_language_model_helper() {
        assert_eq!(
            strip_language_model_prefix("model.language_model.embed_tokens.weight").as_ref(),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            strip_language_model_prefix("model.embed_tokens.weight").as_ref(),
            "model.embed_tokens.weight"
        );
        // Strip happens anywhere in the path (mirrors Python's
        // `name.replace`); we exercise that too.
        assert_eq!(
            strip_language_model_prefix("outer.language_model.inner").as_ref(),
            "outer.inner"
        );
    }

    // -----------------------------------------------------------------------
    // build_metadata — REQUIRED-key panic paths
    // -----------------------------------------------------------------------

    /// `num_kv_shared_layers` is REQUIRED per `gemma.py:659`. Per
    /// [[feedback-no-loop-suppression-2026-05-17]] we panic via
    /// `.expect(...)` rather than silently defaulting to zero (which
    /// would emit an incorrect on-disk file and silently bend runtime
    /// KV-sharing behavior).
    #[test]
    #[should_panic(expected = "num_kv_shared_layers")]
    fn gemma4_metadata_panics_on_missing_num_kv_shared_layers() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "head_dim": 16,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-6,
            "layer_types": ["sliding_attention"],
            // num_kv_shared_layers DELIBERATELY OMITTED
        });
        let _ = build_metadata(&cfg, 0);
    }

    /// `layer_types` is REQUIRED per `gemma.py:665-666`. Without it we
    /// cannot emit `sliding_window_pattern`, which the runtime keys on
    /// to dispatch the right attention kernel per layer.
    #[test]
    #[should_panic(expected = "layer_types")]
    fn gemma4_metadata_panics_on_missing_layer_types() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "head_dim": 16,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-6,
            "num_kv_shared_layers": 0,
            // layer_types DELIBERATELY OMITTED
        });
        let _ = build_metadata(&cfg, 0);
    }

    /// `layer_types` length must match `num_hidden_layers`. Per the
    /// no-loop-suppression rule we surface mismatch via panic instead of
    /// padding / truncating silently (which would emit a wrong
    /// `sliding_window_pattern` array).
    #[test]
    #[should_panic(expected = "layer_types")]
    fn gemma4_metadata_panics_on_layer_types_length_mismatch() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 3,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "head_dim": 16,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-6,
            "num_kv_shared_layers": 0,
            "layer_types": ["sliding_attention", "full_attention"],  // len=2, but n_layers=3
        });
        let _ = build_metadata(&cfg, 0);
    }

    // -----------------------------------------------------------------------
    // build_metadata — array-shaped metadata
    // -----------------------------------------------------------------------

    /// When `use_double_wide_mlp` is true, `feed_forward_length` becomes
    /// an array per `gemma.py:680-684`: the last `num_kv_shared_layers`
    /// block entries are `2 * n_ff`, the rest are `n_ff`.
    #[test]
    fn gemma4_metadata_feed_forward_length_array_on_dwm() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 6,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "head_dim": 16,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-6,
            "num_kv_shared_layers": 2,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
            ],
            "use_double_wide_mlp": true,
        });
        let kv = build_metadata(&cfg, 0);
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        // first_kv_shared_layer_idx = 6 - 2 = 4. Layers [0..4) → 64;
        // layers [4..6) → 128.
        assert_eq!(
            by_key["gemma4.feed_forward_length"],
            MetaValue::ArrayU32(vec![64, 64, 64, 64, 128, 128]),
            "dwm array: last num_kv_shared_layers entries are 2*n_ff"
        );
    }

    /// When `num_global_key_value_heads` is present and differs from
    /// `num_key_value_heads`, `head_count_kv` becomes a per-layer array
    /// (`gemma.py:687-691`). Equal values keep the scalar form.
    #[test]
    fn gemma4_metadata_head_count_kv_array_path() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "intermediate_size": 64,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_global_key_value_heads": 2,  // differs from 4 → array
            "head_dim": 16,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-6,
            "num_kv_shared_layers": 0,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "full_attention",
            ],
        });
        let kv = build_metadata(&cfg, 0);
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        // swa layers (0,1) → num_key_value_heads=4
        // full layers (2,3) → num_global_key_value_heads=2
        assert_eq!(
            by_key["gemma4.attention.head_count_kv"],
            MetaValue::ArrayU32(vec![4, 4, 2, 2]),
        );
    }

    /// When `num_global_key_value_heads` equals `num_key_value_heads`,
    /// the array form is byte-equivalent to scalar — emit scalar.
    #[test]
    fn gemma4_metadata_head_count_kv_scalar_when_equal() {
        let cfg = json!({
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_global_key_value_heads": 4,
            "head_dim": 16,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1.0e-6,
            "num_kv_shared_layers": 0,
            "layer_types": ["sliding_attention", "full_attention"],
        });
        let kv = build_metadata(&cfg, 0);
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        assert_eq!(
            by_key["gemma4.attention.head_count_kv"],
            MetaValue::U32(4),
            "equal global/swa kv heads → scalar form"
        );
    }

    // -----------------------------------------------------------------------
    // build_synthesized_tensors — ROPE_FREQS table (gemma.py:702-718)
    // -----------------------------------------------------------------------

    /// Exact-fixture round-trip for the synthesized ROPE_FREQS table.
    /// `global_head_dim=512`, `partial_rotary_factor=0.25` →
    ///   - `table_len = global_head_dim / 2 = 256`
    ///   - `n_rot_full = int(512 * 0.25 / 2) = 64`
    ///   - `n_unrot_full = 256 - 64 = 192`
    /// First 64 entries `1.0`, remaining 192 entries `1e30`.
    #[test]
    fn gemma4_synthesized_rope_freqs_real_config() {
        let cfg = json!({
            "head_dim": 256,
            "global_head_dim": 512,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_type": "proportional",
                }
            },
        });
        let tensors = build_synthesized_tensors(&cfg);
        assert_eq!(tensors.len(), 1, "exactly one synthesized tensor for Gemma 4");
        let t = &tensors[0];
        assert_eq!(t.name, "rope_freqs.weight");
        assert_eq!(t.shape, vec![256]);
        assert_eq!(t.data.len(), 256);
        // First 64 entries: 1.0; remaining: 1e30.
        for (i, &v) in t.data.iter().enumerate() {
            if i < 64 {
                assert_eq!(v, 1.0, "entry {i} (rotated dim) must be 1.0");
            } else {
                assert_eq!(v, 1.0e30, "entry {i} (unrotated dim) must be 1e30");
            }
        }
    }

    /// Synthesis panics cleanly when the required
    /// `rope_parameters.full_attention` block is absent — per
    /// [[feedback-no-loop-suppression-2026-05-17]] we don't silently
    /// emit a no-op table that would silently break the model.
    #[test]
    #[should_panic(expected = "rope_parameters.full_attention")]
    fn gemma4_synthesized_rope_freqs_panics_on_missing_rope_params() {
        let cfg = json!({
            "head_dim": 256,
            "global_head_dim": 512,
            // rope_parameters.full_attention DELIBERATELY OMITTED
        });
        let _ = build_synthesized_tensors(&cfg);
    }

    /// Synthesis panics when `partial_rotary_factor` is absent inside an
    /// otherwise present `rope_parameters.full_attention`.
    #[test]
    #[should_panic(expected = "partial_rotary_factor")]
    fn gemma4_synthesized_rope_freqs_panics_on_missing_partial_rotary_factor() {
        let cfg = json!({
            "head_dim": 256,
            "global_head_dim": 512,
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "proportional",
                    // partial_rotary_factor DELIBERATELY OMITTED
                }
            },
        });
        let _ = build_synthesized_tensors(&cfg);
    }

    /// Synthesis panics when `rope_type` is absent — gemma.py:705 asserts
    /// it equals `"proportional"`, so a missing key cannot be defaulted
    /// without silently changing semantics.
    #[test]
    #[should_panic(expected = "rope_parameters.full_attention.rope_type")]
    fn gemma4_synthesized_rope_freqs_panics_on_missing_rope_type() {
        let cfg = json!({
            "head_dim": 256,
            "global_head_dim": 512,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    // rope_type DELIBERATELY OMITTED
                }
            },
        });
        let _ = build_synthesized_tensors(&cfg);
    }

    /// Synthesis panics when `rope_type != "proportional"` — the
    /// `[1.0, …, 1e30, …]` collapse mask is mathematically only valid
    /// for the proportional-RoPE variant.
    #[test]
    #[should_panic(expected = "rope_type=proportional")]
    fn gemma4_synthesized_rope_freqs_panics_on_wrong_rope_type() {
        let cfg = json!({
            "head_dim": 256,
            "global_head_dim": 512,
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "linear",
                    "partial_rotary_factor": 0.25,
                }
            },
        });
        let _ = build_synthesized_tensors(&cfg);
    }

    /// Synthesis panics when `partial_rotary_factor * global_head_dim / 2`
    /// exceeds `global_head_dim / 2` (i.e. `partial_rotary_factor > 1.0`)
    /// — the reference arithmetic in `gemma.py:713-715` is a direct
    /// subtraction; underflow would mean a malformed config that the
    /// previous `saturating_sub` was hiding.
    #[test]
    #[should_panic(expected = "n_rot_full")]
    fn gemma4_synthesized_rope_freqs_panics_on_invalid_partial_factor() {
        let cfg = json!({
            "head_dim": 256,
            "global_head_dim": 512,
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 1.5,  // > 1.0 — invalid
                }
            },
        });
        let _ = build_synthesized_tensors(&cfg);
    }

    // -----------------------------------------------------------------------
    // map_tensor_name — synthesized name passthrough
    // -----------------------------------------------------------------------

    /// `rope_freqs.weight` (the canonical GGUF name we synthesize) must
    /// round-trip through `map_tensor_name` as a `Direct` identity so the
    /// driver's standard map+stage path emits it.
    #[test]
    fn gemma4_maps_synthesized_rope_freqs() {
        assert_eq!(
            map_tensor_name("rope_freqs.weight"),
            Some(MappedTensor::Direct("rope_freqs.weight".into()))
        );
    }
}
