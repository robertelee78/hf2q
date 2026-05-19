//! Gemma-4 mmproj HF→GGUF tensor-name + metadata mapper.
//!
//! Port of `/opt/llama.cpp/conversion/gemma.py::Gemma3VisionModel` plus the
//! `MmprojModel` base it `super()`s into
//! (`/opt/llama.cpp/conversion/base.py:2152-2305`). The Gemma-4 multimodal
//! family reuses the Gemma-3 SigLIP vision tower + 2-layer MLP projector
//! verbatim (HF `Gemma3ForConditionalGeneration` /
//! `Gemma4ForConditionalGeneration` checkpoints expose the same submodule
//! tree: `model.vision_tower.vision_model.` + `model.multi_modal_projector.`).
//!
//! The mmproj sidecar is written as a SEPARATE GGUF file (suffixed
//! `-mmproj.gguf`) consumed by `llama-mtmd-cli`. Its on-disk schema lives in
//! `/opt/llama.cpp/tools/mtmd/clip-impl.h`:
//!
//! - `general.architecture = "clip"` (per
//!   `gguf-py/gguf/constants.py::MODEL_ARCH_NAMES[MMPROJ] = "clip"`).
//! - Vision-encoder weights are written under the `v.` prefix
//!   (`TN_PATCH_EMBD = "v.patch_embd.weight"`,
//!   `TN_POS_EMBD = "%s.position_embd.weight"`, `TN_LN_1 = "%s.blk.%d.ln1.%s"`,
//!   `TN_ATTN_OUTPUT = "%s.blk.%d.attn_out.%s"`, etc.).
//! - Projector weights are written under the `mm.` prefix
//!   (`TN_MM_INP_PROJ = "mm.input_projection.weight"`,
//!   `TN_MM_SOFT_EMB_N = "mm.soft_emb_norm.weight"`).
//! - The projector-type KV is `clip.projector_type = "gemma3"` (per
//!   `VISION_PROJECTOR_TYPE_NAMES[GEMMA3] = "gemma3"`). There is NO
//!   `clip.has_text_encoder` and NO `clip.has_gemma3_projector` KV in the
//!   canonical schema — discrimination happens entirely through
//!   `clip.projector_type` + `clip.has_vision_encoder`.
//!
//! Per ADR-033 §P0 "Per-arch convert-side mapping at
//! `src/convert/arch/<arch>.rs`". Per
//! [[feedback-no-backwards-compat-2026-05-18]]: no fallback / no aliasing —
//! every HF name we recognize maps to exactly one GGUF name, and every other
//! name returns `None`. Per [[feedback-no-loop-suppression-2026-05-17]]:
//! callers MUST NOT silently skip a `None` — propagate as a typed error.

use crate::backends::gguf::types::MetaValue;

/// Translate one HuggingFace tensor name (as seen in
/// `model.safetensors` for an `Gemma{3,4}ForConditionalGeneration`
/// checkpoint) to its canonical mmproj GGUF tensor name. Returns `None` if
/// `hf_name` is not one of the recognized SigLIP vision-tower or
/// Gemma3-projector weights.
///
/// SigLIP vision encoder (per-block, `<N>` = block index):
///
/// | HF name                                                                                              | GGUF name                          |
/// |------------------------------------------------------------------------------------------------------|------------------------------------|
/// | `model.vision_tower.vision_model.embeddings.patch_embedding.weight`                                  | `v.patch_embd.weight`              |
/// | `model.vision_tower.vision_model.embeddings.patch_embedding.bias`                                    | `v.patch_embd.bias`                |
/// | `model.vision_tower.vision_model.embeddings.position_embedding.weight`                               | `v.position_embd.weight`           |
/// | `model.vision_tower.vision_model.encoder.layers.<N>.layer_norm1.{weight,bias}`                       | `v.blk.<N>.ln1.{weight,bias}`      |
/// | `model.vision_tower.vision_model.encoder.layers.<N>.layer_norm2.{weight,bias}`                       | `v.blk.<N>.ln2.{weight,bias}`      |
/// | `model.vision_tower.vision_model.encoder.layers.<N>.self_attn.q_proj.{weight,bias}`                  | `v.blk.<N>.attn_q.{weight,bias}`   |
/// | `model.vision_tower.vision_model.encoder.layers.<N>.self_attn.k_proj.{weight,bias}`                  | `v.blk.<N>.attn_k.{weight,bias}`   |
/// | `model.vision_tower.vision_model.encoder.layers.<N>.self_attn.v_proj.{weight,bias}`                  | `v.blk.<N>.attn_v.{weight,bias}`   |
/// | `model.vision_tower.vision_model.encoder.layers.<N>.self_attn.out_proj.{weight,bias}`                | `v.blk.<N>.attn_out.{weight,bias}` |
/// | `model.vision_tower.vision_model.encoder.layers.<N>.mlp.fc1.{weight,bias}`                           | `v.blk.<N>.ffn_up.{weight,bias}`   |
/// | `model.vision_tower.vision_model.encoder.layers.<N>.mlp.fc2.{weight,bias}`                           | `v.blk.<N>.ffn_down.{weight,bias}` |
/// | `model.vision_tower.vision_model.post_layernorm.{weight,bias}`                                       | `v.post_ln.{weight,bias}`          |
///
/// Gemma-3 MLP projector (no per-block index; the projector is a single
/// 2-layer MLP with a soft-embedding norm):
///
/// | HF name                                                  | GGUF name                       |
/// |----------------------------------------------------------|---------------------------------|
/// | `model.multi_modal_projector.mm_input_projection_weight` | `mm.input_projection.weight`    |
/// | `model.multi_modal_projector.mm_soft_emb_norm.weight`    | `mm.soft_emb_norm.weight`       |
///
/// Note: the projector's `mm_input_projection_weight` is intentionally
/// snake-case `_weight` (not `.weight`) in the upstream HF checkpoint
/// (it's a registered `nn.Parameter`, not an `nn.Linear`); this mapper
/// accepts both the raw `_weight` form and the dotted `.weight` form that
/// gemma.py's `filter_tensors` rewrites it to (see
/// `/opt/llama.cpp/conversion/gemma.py:289`).
pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    // ---- Projector (no per-block index) ---------------------------------
    // Upstream HF stores `mm_input_projection_weight` as a single
    // `nn.Parameter`; gemma.py normalizes `_weight` → `.weight` before
    // hitting `tensor_mapping.py`. Accept both forms.
    match hf_name {
        "model.multi_modal_projector.mm_input_projection_weight"
        | "model.multi_modal_projector.mm_input_projection.weight" => {
            return Some("mm.input_projection.weight".to_string());
        }
        "model.multi_modal_projector.mm_soft_emb_norm.weight" => {
            return Some("mm.soft_emb_norm.weight".to_string());
        }
        _ => {}
    }

    // ---- Vision-tower globals -------------------------------------------
    let v_prefix = "model.vision_tower.vision_model.";
    let v_rest = hf_name.strip_prefix(v_prefix)?;
    match v_rest {
        "embeddings.patch_embedding.weight" => {
            return Some("v.patch_embd.weight".to_string());
        }
        "embeddings.patch_embedding.bias" => {
            return Some("v.patch_embd.bias".to_string());
        }
        "embeddings.position_embedding.weight" => {
            return Some("v.position_embd.weight".to_string());
        }
        "post_layernorm.weight" => return Some("v.post_ln.weight".to_string()),
        "post_layernorm.bias" => return Some("v.post_ln.bias".to_string()),
        _ => {}
    }

    // ---- Vision-tower per-block: `encoder.layers.<N>.<rest>` ------------
    let after_layers = v_rest.strip_prefix("encoder.layers.")?;
    let dot = after_layers.find('.')?;
    let (layer_str, rest_with_dot) = after_layers.split_at(dot);
    // Strict integer parse — reject leading zeros / signs (matches llama3.rs).
    let layer: usize = layer_str.parse().ok()?;
    if layer.to_string() != layer_str {
        return None;
    }
    let rest = &rest_with_dot[1..]; // skip the dot

    // Split `<sub>.<param>` where <param> ∈ {weight, bias} — SigLIP layers
    // have biases on every linear AND every layer-norm (unlike Llama-3
    // which is bias-less).
    let (sub, param) = split_weight_or_bias(rest)?;

    let new_sub = match sub {
        "layer_norm1" => "ln1",
        "layer_norm2" => "ln2",
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.out_proj" => "attn_out",
        "mlp.fc1" => "ffn_up",
        "mlp.fc2" => "ffn_down",
        _ => return None,
    };

    Some(format!("v.blk.{layer}.{new_sub}.{param}"))
}

/// Strip a trailing `.weight` or `.bias` from `s` and return the head + the
/// param kind. Returns `None` if `s` ends with neither.
fn split_weight_or_bias(s: &str) -> Option<(&str, &str)> {
    if let Some(head) = s.strip_suffix(".weight") {
        Some((head, "weight"))
    } else if let Some(head) = s.strip_suffix(".bias") {
        Some((head, "bias"))
    } else {
        None
    }
}

/// Build the mmproj GGUF metadata KV pairs for a Gemma-4 / Gemma-3 vision
/// model. Port of `Gemma3VisionModel::set_gguf_parameters` +
/// `MmprojModel::set_gguf_parameters` (base.py:2270-2304).
///
/// `vision_config` is the `vision_config` sub-object of the HF top-level
/// `config.json` (a `Gemma{3,4}ForConditionalGeneration` checkpoint nests
/// vision hparams there, e.g.
/// `config["vision_config"]["hidden_size"]`).
///
/// Required `vision_config` keys (mandatory; missing key → caller-side
/// panic from the `[]` indexing):
///   - `image_size`
///   - `patch_size`
///   - `hidden_size`
///   - `intermediate_size`
///   - `num_hidden_layers`
///   - `num_attention_heads`
///
/// Optional `vision_config` keys (defaulted to HF defaults):
///   - `layer_norm_eps` — defaults to `1e-6` per
///     `gemma.py::Gemma3VisionModel::set_gguf_parameters:257`
///     (`hparams.get("layer_norm_eps", 1e-6)`).
///
/// `file_type` is the chosen ftype as a `u32` (matches `add_file_type` at
/// base.py:2271).
///
/// Emitted KV keys (per `gguf-py/gguf/constants.py::Keys.{Clip,ClipVision}`):
///   - `general.architecture` = `"clip"`
///   - `general.file_type` = `file_type`
///   - `clip.has_vision_encoder` = `true`
///   - `clip.has_audio_encoder` = `false` (Gemma-3 vision-only mmproj)
///   - `clip.projector_type` = `"gemma3"`
///   - `clip.use_gelu` = `true` (SigLIP activation; matches
///     `add_vision_use_gelu(True)` at gemma.py:258)
///   - `clip.vision.image_size` = vision_config["image_size"]
///   - `clip.vision.patch_size` = vision_config["patch_size"]
///   - `clip.vision.embedding_length` = vision_config["hidden_size"]
///   - `clip.vision.feed_forward_length` = vision_config["intermediate_size"]
///   - `clip.vision.block_count` = vision_config["num_hidden_layers"]
///   - `clip.vision.attention.head_count` = vision_config["num_attention_heads"]
///   - `clip.vision.attention.layer_norm_epsilon` = vision_config["layer_norm_eps"]
///
/// Notes:
///   - The canonical mmproj schema has NO `clip.has_text_encoder` and NO
///     `clip.has_gemma3_projector` KVs (the user-supplied spec lists those,
///     but they do not exist in `gguf-py/gguf/constants.py::Keys.Clip` —
///     discrimination is entirely via `clip.projector_type` +
///     `clip.has_vision_encoder`). We follow the canonical schema, not the
///     spec.
///   - `clip.vision.projection_dim` (the LLM text-side embedding size) is
///     written by the text-side mapper, not here, since the mmproj file
///     does not own that hparam. v1 omits it; downstream consumers must
///     provide it via `--text-config` or by reading the paired text-model
///     GGUF.
pub fn build_metadata(vision_config: &serde_json::Value, file_type: u32) -> Vec<(String, MetaValue)> {
    let image_size = vision_config["image_size"]
        .as_u64()
        .expect("vision_config missing required key `image_size`") as u32;
    let patch_size = vision_config["patch_size"]
        .as_u64()
        .expect("vision_config missing required key `patch_size`") as u32;
    let hidden_size = vision_config["hidden_size"]
        .as_u64()
        .expect("vision_config missing required key `hidden_size`") as u32;
    let ffn_len = vision_config["intermediate_size"]
        .as_u64()
        .expect("vision_config missing required key `intermediate_size`") as u32;
    let n_layers = vision_config["num_hidden_layers"]
        .as_u64()
        .expect("vision_config missing required key `num_hidden_layers`") as u32;
    let n_head = vision_config["num_attention_heads"]
        .as_u64()
        .expect("vision_config missing required key `num_attention_heads`") as u32;

    // Optional with HF default (matches gemma.py's `.get("layer_norm_eps", 1e-6)`).
    let ln_eps = vision_config
        .get("layer_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0e-6) as f32;

    vec![
        (
            "general.architecture".into(),
            MetaValue::String("clip".into()),
        ),
        ("general.file_type".into(), MetaValue::U32(file_type)),
        (
            "clip.has_vision_encoder".into(),
            MetaValue::Bool(true),
        ),
        (
            "clip.has_audio_encoder".into(),
            MetaValue::Bool(false),
        ),
        (
            "clip.projector_type".into(),
            MetaValue::String("gemma3".into()),
        ),
        ("clip.use_gelu".into(), MetaValue::Bool(true)),
        ("clip.vision.image_size".into(), MetaValue::U32(image_size)),
        ("clip.vision.patch_size".into(), MetaValue::U32(patch_size)),
        (
            "clip.vision.embedding_length".into(),
            MetaValue::U32(hidden_size),
        ),
        (
            "clip.vision.feed_forward_length".into(),
            MetaValue::U32(ffn_len),
        ),
        ("clip.vision.block_count".into(), MetaValue::U32(n_layers)),
        (
            "clip.vision.attention.head_count".into(),
            MetaValue::U32(n_head),
        ),
        (
            "clip.vision.attention.layer_norm_epsilon".into(),
            MetaValue::F32(ln_eps),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Acceptance test 1 — round-trip the canonical table for every HF
    /// name kind. Asserts `map_tensor_name(hf) → Some(gguf)` with the exact
    /// pair from the ADR-033 §P0 mmproj table.
    #[test]
    fn gemma4_mmproj_tensor_name_round_trip() {
        let cases: &[(&str, &str)] = &[
            // Vision-tower globals.
            (
                "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
                "v.patch_embd.weight",
            ),
            (
                "model.vision_tower.vision_model.embeddings.patch_embedding.bias",
                "v.patch_embd.bias",
            ),
            (
                "model.vision_tower.vision_model.embeddings.position_embedding.weight",
                "v.position_embd.weight",
            ),
            (
                "model.vision_tower.vision_model.post_layernorm.weight",
                "v.post_ln.weight",
            ),
            (
                "model.vision_tower.vision_model.post_layernorm.bias",
                "v.post_ln.bias",
            ),
            // Per-block — sample L=0, L=13, L=26 (SigLIP-So400m has 27
            // layers; pick edge / mid / depth).
            (
                "model.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight",
                "v.blk.0.ln1.weight",
            ),
            (
                "model.vision_tower.vision_model.encoder.layers.0.layer_norm1.bias",
                "v.blk.0.ln1.bias",
            ),
            (
                "model.vision_tower.vision_model.encoder.layers.13.layer_norm2.weight",
                "v.blk.13.ln2.weight",
            ),
            (
                "model.vision_tower.vision_model.encoder.layers.13.layer_norm2.bias",
                "v.blk.13.ln2.bias",
            ),
            (
                "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
                "v.blk.0.attn_q.weight",
            ),
            (
                "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.bias",
                "v.blk.0.attn_q.bias",
            ),
            (
                "model.vision_tower.vision_model.encoder.layers.7.self_attn.k_proj.weight",
                "v.blk.7.attn_k.weight",
            ),
            (
                "model.vision_tower.vision_model.encoder.layers.7.self_attn.v_proj.weight",
                "v.blk.7.attn_v.weight",
            ),
            (
                "model.vision_tower.vision_model.encoder.layers.7.self_attn.out_proj.weight",
                "v.blk.7.attn_out.weight",
            ),
            (
                "model.vision_tower.vision_model.encoder.layers.7.self_attn.out_proj.bias",
                "v.blk.7.attn_out.bias",
            ),
            (
                "model.vision_tower.vision_model.encoder.layers.26.mlp.fc1.weight",
                "v.blk.26.ffn_up.weight",
            ),
            (
                "model.vision_tower.vision_model.encoder.layers.26.mlp.fc1.bias",
                "v.blk.26.ffn_up.bias",
            ),
            (
                "model.vision_tower.vision_model.encoder.layers.26.mlp.fc2.weight",
                "v.blk.26.ffn_down.weight",
            ),
            (
                "model.vision_tower.vision_model.encoder.layers.26.mlp.fc2.bias",
                "v.blk.26.ffn_down.bias",
            ),
            // Projector — both the raw `_weight` parameter form and the
            // `.weight` form gemma.py rewrites it to.
            (
                "model.multi_modal_projector.mm_input_projection_weight",
                "mm.input_projection.weight",
            ),
            (
                "model.multi_modal_projector.mm_input_projection.weight",
                "mm.input_projection.weight",
            ),
            (
                "model.multi_modal_projector.mm_soft_emb_norm.weight",
                "mm.soft_emb_norm.weight",
            ),
        ];

        for &(hf, expected_gguf) in cases {
            let got = map_tensor_name(hf);
            assert_eq!(
                got.as_deref(),
                Some(expected_gguf),
                "map_tensor_name({hf:?}) = {got:?}, want Some({expected_gguf:?})"
            );
        }
    }

    /// Sibling — unknown names must surface as `None`. Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: the caller is expected
    /// to error on this, never silently skip.
    #[test]
    fn gemma4_mmproj_tensor_name_rejects_unknown_kinds() {
        // Wrong top-level prefix (text-side weight leaked into mmproj path).
        assert_eq!(map_tensor_name("model.embed_tokens.weight"), None);
        assert_eq!(map_tensor_name("lm_head.weight"), None);
        // Missing `model.` prefix (some checkpoints expose `vision_tower.`
        // directly — those go through a separate code path; this mapper
        // is strict about the `model.` wrap as canonical Gemma3/4 HF).
        assert_eq!(
            map_tensor_name("vision_tower.vision_model.embeddings.patch_embedding.weight"),
            None
        );
        // SigLIP has no `pre_layernorm` (only `post_layernorm`).
        assert_eq!(
            map_tensor_name("model.vision_tower.vision_model.pre_layernorm.weight"),
            None
        );
        // Unknown per-block sub.
        assert_eq!(
            map_tensor_name(
                "model.vision_tower.vision_model.encoder.layers.0.layer_norm3.weight"
            ),
            None
        );
        // Malformed layer index (leading zero — matches llama3.rs strictness).
        assert_eq!(
            map_tensor_name(
                "model.vision_tower.vision_model.encoder.layers.01.layer_norm1.weight"
            ),
            None
        );
        // Empty layer index.
        assert_eq!(
            map_tensor_name(
                "model.vision_tower.vision_model.encoder.layers..layer_norm1.weight"
            ),
            None
        );
        // No layer prefix at all.
        assert_eq!(
            map_tensor_name(
                "model.vision_tower.vision_model.encoder.layers.layer_norm1.weight"
            ),
            None
        );
        // Unknown projector tensor.
        assert_eq!(
            map_tensor_name("model.multi_modal_projector.unknown.weight"),
            None
        );
        // Non-weight / non-bias suffix on a known sub.
        assert_eq!(
            map_tensor_name(
                "model.vision_tower.vision_model.encoder.layers.0.layer_norm1.running_mean"
            ),
            None
        );
    }

    /// Acceptance test 2 — feed a minimal hand-written vision_config and
    /// verify all 13 KV pairs come back with the right types + values.
    /// Values mirror the SigLIP-So400m hparams that Gemma-3-27B-it actually
    /// ships (image_size=896, patch_size=14, hidden_size=1152, etc.).
    #[test]
    fn gemma4_mmproj_metadata_built_from_vision_config() {
        let vcfg = json!({
            "image_size": 896,
            "patch_size": 14,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "layer_norm_eps": 1.0e-6,
        });

        let kv = build_metadata(&vcfg, 17 /* MostlyQ5_K_M */);

        assert_eq!(kv.len(), 13, "Gemma4 mmproj emits 13 KV pairs at v1");
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("clip".into())
        );
        assert_eq!(by_key["general.file_type"], MetaValue::U32(17));
        assert_eq!(by_key["clip.has_vision_encoder"], MetaValue::Bool(true));
        assert_eq!(by_key["clip.has_audio_encoder"], MetaValue::Bool(false));
        assert_eq!(
            by_key["clip.projector_type"],
            MetaValue::String("gemma3".into())
        );
        assert_eq!(by_key["clip.use_gelu"], MetaValue::Bool(true));
        assert_eq!(by_key["clip.vision.image_size"], MetaValue::U32(896));
        assert_eq!(by_key["clip.vision.patch_size"], MetaValue::U32(14));
        assert_eq!(by_key["clip.vision.embedding_length"], MetaValue::U32(1152));
        assert_eq!(
            by_key["clip.vision.feed_forward_length"],
            MetaValue::U32(4304)
        );
        assert_eq!(by_key["clip.vision.block_count"], MetaValue::U32(27));
        assert_eq!(
            by_key["clip.vision.attention.head_count"],
            MetaValue::U32(16)
        );
        assert_eq!(
            by_key["clip.vision.attention.layer_norm_epsilon"],
            MetaValue::F32(1.0e-6)
        );
    }

    /// Sibling — verify the `layer_norm_eps` default kicks in when absent
    /// (matches `gemma.py:257`: `hparams.get("layer_norm_eps", 1e-6)`).
    #[test]
    fn gemma4_mmproj_metadata_layer_norm_eps_default() {
        let vcfg = json!({
            "image_size": 224,
            "patch_size": 14,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            // layer_norm_eps omitted → defaults to 1e-6
        });

        let kv = build_metadata(&vcfg, 0);
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        assert_eq!(
            by_key["clip.vision.attention.layer_norm_epsilon"],
            MetaValue::F32(1.0e-6),
            "layer_norm_eps defaults to 1e-6 per gemma.py:257"
        );
        // Spot-check that required keys still flow through with the
        // smaller-fixture values (catches a class of metadata-builder
        // regression where the eps default short-circuits other keys).
        assert_eq!(by_key["clip.vision.image_size"], MetaValue::U32(224));
        assert_eq!(by_key["clip.vision.block_count"], MetaValue::U32(12));
    }
}
