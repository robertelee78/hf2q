//! Gemma 4 Vision (transformer-style) mmproj HF→GGUF tensor-name + metadata mapper.
//!
//! Port of `/opt/llama.cpp/conversion/gemma.py::Gemma4VisionAudioModel`
//! (lines 769-840) for the **vision-only** Gemma 4 26B-A4B-IT release. The
//! Gemma 4 vision tower is structurally distinct from the Gemma 3 SigLIP
//! tower handled by `gemma4_mmproj.rs`:
//!
//! - **Tensor naming**: `model.vision_tower.encoder.layers.<N>.*` (plural
//!   `layers`, no `vision_model` infix), with a `.linear.` infix on MLP
//!   weights (`mlp.{gate,up,down}_proj.linear.weight`).
//! - **Block structure**: transformer-style with SwiGLU FFN
//!   (`gate_proj + up_proj` → silu → `down_proj`), pre + post layernorms
//!   around BOTH attention and FFN (input_layernorm /
//!   post_attention_layernorm, pre_feedforward_layernorm /
//!   post_feedforward_layernorm), and per-head `q_norm`/`k_norm` RMS norms.
//! - **Projector**: 'gemma4v' (vs Gemma 3's 'gemma3'). Single linear
//!   projection from vision hidden (1152) to text decoder hidden (2816)
//!   via `model.embed_vision.embedding_projection.weight`.
//! - **Patch embedder**: 2-D `(out, p*p*3)` HF weight reshaped to
//!   `(out, 3, p, p)` GGUF for the standard conv2d convention. Via
//!   `BakeOp::PatchEmbedderReshape`.
//!
//! Per ADR-033 §10 task #73. Per [[feedback-no-backwards-compat-2026-05-18]]:
//! every HF name we recognize maps to exactly one GGUF name; everything
//! else returns `None` and the caller surfaces a typed error.

use crate::backends::gguf::types::MetaValue;

/// Translate one HuggingFace tensor name to its canonical GGUF mmproj
/// tensor name for a Gemma 4 vision tower. Returns `None` for unrecognized
/// names — callers MUST surface as a typed error per the no-loop-suppression
/// rule.
///
/// Tensor inventory (verified against canonical mmproj dump for
/// google-gemma-4-26b-a4b-it — 356 tensors total = 5 globals + 27 blocks × 13):
///
/// **Globals:**
///
/// | HF name                                                              | GGUF name                          |
/// |----------------------------------------------------------------------|------------------------------------|
/// | `model.embed_vision.embedding_projection.weight`                     | `mm.input_projection.weight`       |
/// | `model.vision_tower.patch_embedder.input_proj.weight` *(needs bake)* | `v.patch_embd.weight`              |
/// | `model.vision_tower.patch_embedder.position_embedding_table`         | `v.position_embd.weight`           |
/// | `model.vision_tower.std_bias`                                        | `v.std_bias`                       |
/// | `model.vision_tower.std_scale`                                       | `v.std_scale`                      |
///
/// **Per-block** (HF prefix `model.vision_tower.encoder.layers.<N>.`,
/// GGUF prefix `v.blk.<N>.`):
///
/// | HF suffix                                       | GGUF suffix                  |
/// |-------------------------------------------------|------------------------------|
/// | `input_layernorm.weight`                        | `ln1.weight`                 |
/// | `self_attn.q_proj.linear.weight`                | `attn_q.weight`              |
/// | `self_attn.k_proj.linear.weight`                | `attn_k.weight`              |
/// | `self_attn.v_proj.linear.weight`                | `attn_v.weight`              |
/// | `self_attn.o_proj.linear.weight`                | `attn_out.weight`            |
/// | `self_attn.q_norm.weight`                       | `attn_q_norm.weight`         |
/// | `self_attn.k_norm.weight`                       | `attn_k_norm.weight`         |
/// | `post_attention_layernorm.weight`               | `attn_post_norm.weight`      |
/// | `pre_feedforward_layernorm.weight`              | `ln2.weight`                 |
/// | `post_feedforward_layernorm.weight`             | `ffn_post_norm.weight`       |
/// | `mlp.gate_proj.linear.weight`                   | `ffn_gate.weight`            |
/// | `mlp.up_proj.linear.weight`                     | `ffn_up.weight`              |
/// | `mlp.down_proj.linear.weight`                   | `ffn_down.weight`            |
pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    // -- Globals ----------------------------------------------------------
    if hf_name == "model.embed_vision.embedding_projection.weight" {
        return Some("mm.input_projection.weight".to_string());
    }
    if hf_name == "model.vision_tower.patch_embedder.input_proj.weight" {
        return Some("v.patch_embd.weight".to_string());
    }
    if hf_name == "model.vision_tower.patch_embedder.position_embedding_table" {
        return Some("v.position_embd.weight".to_string());
    }
    if hf_name == "model.vision_tower.std_bias" {
        return Some("v.std_bias".to_string());
    }
    if hf_name == "model.vision_tower.std_scale" {
        return Some("v.std_scale".to_string());
    }

    // -- Per-block --------------------------------------------------------
    let rest = hf_name.strip_prefix("model.vision_tower.encoder.layers.")?;
    let (layer_str, suffix) = rest.split_once('.')?;
    let layer: u32 = layer_str.parse().ok()?;
    let blk = format!("v.blk.{}", layer);

    let mapped_suffix = match suffix {
        "input_layernorm.weight" => "ln1.weight",
        "self_attn.q_proj.linear.weight" => "attn_q.weight",
        "self_attn.k_proj.linear.weight" => "attn_k.weight",
        "self_attn.v_proj.linear.weight" => "attn_v.weight",
        "self_attn.o_proj.linear.weight" => "attn_out.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "post_attention_layernorm.weight" => "attn_post_norm.weight",
        "pre_feedforward_layernorm.weight" => "ln2.weight",
        "post_feedforward_layernorm.weight" => "ffn_post_norm.weight",
        "mlp.gate_proj.linear.weight" => "ffn_gate.weight",
        "mlp.up_proj.linear.weight" => "ffn_up.weight",
        "mlp.down_proj.linear.weight" => "ffn_down.weight",
        _ => return None,
    };
    Some(format!("{}.{}", blk, mapped_suffix))
}

/// Build the GGUF metadata KV pairs for a Gemma 4 vision mmproj sidecar.
///
/// Verified against canonical mmproj dump for google-gemma-4-26b-a4b-it
/// (23 KV pairs in canonical insertion order):
///   architecture('clip'), type('mmproj'), sampling.*, name, finetune,
///   basename, size_label, file_type, has_vision_encoder, projection_dim,
///   image_size, patch_size, embedding_length, feed_forward_length,
///   block_count, attention.head_count, image_mean, image_std,
///   projector_type('gemma4v'), attention.layer_norm_epsilon,
///   quantization_version
///
/// `vision_config` is the `text_config`-sibling vision sub-object at
/// `config.json::vision_config`. `text_hidden_size` is the text decoder's
/// hidden_size (used as `projection_dim` — the output dim of the
/// `mm.input_projection.weight` tensor).
pub fn build_metadata(
    vision_config: &serde_json::Value,
    text_hidden_size: u32,
    file_type: u32,
    model_card: Option<&crate::convert::model_card::ModelCard>,
    sampling: Option<&crate::convert::model_card::SamplingConfig>,
    model_dir_basename: Option<&str>,
) -> Vec<(String, MetaValue)> {
    use crate::convert::model_card::get_model_id_components;

    let raw_name = model_dir_basename
        .map(|s| s.to_string())
        .unwrap_or_else(|| "model".to_string());
    let mut id_components = get_model_id_components(&raw_name);
    // mmproj ID-component override: canonical's Metadata.get_model_id_components
    // at /opt/llama.cpp/gguf-py/gguf/metadata.py:295-309 reclassifies a
    // numeric size_label part as `finetune` when its implied parameter
    // count is far from the model's total_params. For mmproj sidecars,
    // total_params is the vision tower (~440M for Gemma 4 26B-A4B-IT),
    // while the directory name carries `26b-a4b-it` — `26b` (= 26e9
    // params) is FAR from 440M, so canonical moves it from size_label
    // to finetune. Result: size_label='a4B', finetune='26b-it'.
    //
    // We replicate the post-classification step: when our size_label
    // ends with `B-a<digit>B` (the Gemma 4 26B-a4B form), keep only
    // the `a<digit>B` portion as size_label and prepend the leading
    // <num>b to finetune. Lowercase the `b` (canonical does this via
    // `part = part[:-1] + part[-1].lower()` at metadata.py:309 when
    // reclassifying to context-length/finetune).
    if let Some(sl) = id_components.size_label.clone() {
        // Match `<digits>B-a<digit>B` shape.
        if let Some(dash_idx) = sl.find("-a") {
            let (prefix, suffix) = sl.split_at(dash_idx);
            // prefix should be like "26B" and suffix like "-a4B"
            if prefix.ends_with('B')
                && prefix.len() >= 2
                && prefix[..prefix.len() - 1]
                    .chars()
                    .all(|c| c.is_ascii_digit())
                && suffix.starts_with("-a")
            {
                // Reclassify prefix → finetune (lowercased `b` per canonical).
                let prefix_lower = format!(
                    "{}{}",
                    &prefix[..prefix.len() - 1],
                    prefix[prefix.len() - 1..].to_ascii_lowercase()
                );
                // Strip the leading dash from suffix → new size_label.
                let new_size_label = suffix.strip_prefix('-').unwrap_or(suffix).to_string();
                let finetune_prefixed = match &id_components.finetune {
                    Some(existing) => format!("{}-{}", prefix_lower, existing),
                    None => prefix_lower,
                };
                id_components.size_label = Some(new_size_label);
                id_components.finetune = Some(finetune_prefixed);
            }
        }
    }
    let display_name = id_components
        .name
        .clone()
        .unwrap_or_else(|| raw_name.clone());

    // canonical's vision_config keys (per
    // /opt/hf2q/models/google-gemma-4-26b-a4b-it/config.json):
    //   hidden_size = 1152
    //   intermediate_size = 4304
    //   depth = 27 (block count)
    //   num_attention_heads = 16
    //   image_size = 224 (unused/synthetic; set in canonical)
    //   patch_size = 16 (from canonical convert; not always in config)
    //   layer_norm_eps = 1e-6
    let embedding_length = vision_config["hidden_size"]
        .as_u64()
        .expect("vision_config missing required key `hidden_size`")
        as u32;
    let feed_forward_length = vision_config["intermediate_size"]
        .as_u64()
        .expect("vision_config missing required key `intermediate_size`")
        as u32;
    // Block count: HF config uses `num_hidden_layers`, canonical
    // gemma.py reads via find_hparam(["num_hidden_layers", "depth"]).
    let block_count = vision_config
        .get("num_hidden_layers")
        .or_else(|| vision_config.get("depth"))
        .and_then(|v| v.as_u64())
        .expect("vision_config missing required key `num_hidden_layers` or `depth`")
        as u32;
    let head_count = vision_config["num_attention_heads"]
        .as_u64()
        .expect("vision_config missing required key `num_attention_heads`")
        as u32;
    let layer_norm_eps = vision_config
        .get("layer_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0e-6) as f32;
    let image_size = vision_config
        .get("image_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(224) as u32;
    let patch_size = vision_config
        .get("patch_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as u32;

    // mmproj prelude — DISTINCT from text-decoder prelude:
    //   architecture = 'clip', type = 'mmproj', sampling.*, name,
    //   finetune, basename, size_label, file_type
    //   then clip.* keys, then general.quantization_version postlude.
    // Canonical mmproj does NOT emit general.license / general.tags /
    // general.languages (verified against /tmp/gemma_canon_mmproj_f16.gguf
    // dump). The model_card param is accepted for API parity with
    // text-decoder build_metadata but most fields are dropped here.
    let _ = model_card; // mmproj omits license/tags/languages

    let mut kv: Vec<(String, MetaValue)> = Vec::with_capacity(24);
    kv.push((
        "general.architecture".into(),
        MetaValue::String("clip".into()),
    ));
    kv.push(("general.type".into(), MetaValue::String("mmproj".into())));
    if let Some(s) = sampling {
        if let Some(v) = s.top_k {
            kv.push(("general.sampling.top_k".into(), MetaValue::I32(v)));
        }
        if let Some(v) = s.top_p {
            kv.push(("general.sampling.top_p".into(), MetaValue::F32(v)));
        }
        if let Some(v) = s.temperature {
            kv.push(("general.sampling.temp".into(), MetaValue::F32(v)));
        }
    }
    kv.push(("general.name".into(), MetaValue::String(display_name)));
    if let Some(f) = &id_components.finetune {
        kv.push(("general.finetune".into(), MetaValue::String(f.clone())));
    }
    if let Some(b) = &id_components.basename {
        kv.push(("general.basename".into(), MetaValue::String(b.clone())));
    }
    if let Some(sl) = &id_components.size_label {
        kv.push(("general.size_label".into(), MetaValue::String(sl.clone())));
    }
    kv.push(("general.file_type".into(), MetaValue::U32(file_type)));
    kv.push(("clip.has_vision_encoder".into(), MetaValue::Bool(true)));
    kv.push((
        "clip.vision.projection_dim".into(),
        MetaValue::U32(text_hidden_size),
    ));
    kv.push(("clip.vision.image_size".into(), MetaValue::U32(image_size)));
    kv.push(("clip.vision.patch_size".into(), MetaValue::U32(patch_size)));
    kv.push((
        "clip.vision.embedding_length".into(),
        MetaValue::U32(embedding_length),
    ));
    kv.push((
        "clip.vision.feed_forward_length".into(),
        MetaValue::U32(feed_forward_length),
    ));
    kv.push((
        "clip.vision.block_count".into(),
        MetaValue::U32(block_count),
    ));
    kv.push((
        "clip.vision.attention.head_count".into(),
        MetaValue::U32(head_count),
    ));
    // Canonical Gemma 4 mmproj uses identity image normalization
    // (image_mean=[0,0,0], image_std=[1,1,1]) — the actual std/mean
    // get baked into the v.std_bias / v.std_scale tensors.
    kv.push((
        "clip.vision.image_mean".into(),
        MetaValue::ArrayF32(vec![0.0, 0.0, 0.0]),
    ));
    kv.push((
        "clip.vision.image_std".into(),
        MetaValue::ArrayF32(vec![1.0, 1.0, 1.0]),
    ));
    kv.push((
        "clip.vision.projector_type".into(),
        MetaValue::String("gemma4v".into()),
    ));
    kv.push((
        "clip.vision.attention.layer_norm_epsilon".into(),
        MetaValue::F32(layer_norm_eps),
    ));
    kv.push(("general.quantization_version".into(), MetaValue::U32(2)));

    kv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_globals() {
        assert_eq!(
            map_tensor_name("model.embed_vision.embedding_projection.weight"),
            Some("mm.input_projection.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("model.vision_tower.patch_embedder.input_proj.weight"),
            Some("v.patch_embd.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("model.vision_tower.patch_embedder.position_embedding_table"),
            Some("v.position_embd.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("model.vision_tower.std_bias"),
            Some("v.std_bias".to_string())
        );
        assert_eq!(
            map_tensor_name("model.vision_tower.std_scale"),
            Some("v.std_scale".to_string())
        );
    }

    #[test]
    fn map_per_block() {
        let cases: &[(&str, &str)] = &[
            (
                "model.vision_tower.encoder.layers.0.input_layernorm.weight",
                "v.blk.0.ln1.weight",
            ),
            (
                "model.vision_tower.encoder.layers.5.self_attn.q_proj.linear.weight",
                "v.blk.5.attn_q.weight",
            ),
            (
                "model.vision_tower.encoder.layers.10.self_attn.k_proj.linear.weight",
                "v.blk.10.attn_k.weight",
            ),
            (
                "model.vision_tower.encoder.layers.15.self_attn.v_proj.linear.weight",
                "v.blk.15.attn_v.weight",
            ),
            (
                "model.vision_tower.encoder.layers.20.self_attn.o_proj.linear.weight",
                "v.blk.20.attn_out.weight",
            ),
            (
                "model.vision_tower.encoder.layers.0.self_attn.q_norm.weight",
                "v.blk.0.attn_q_norm.weight",
            ),
            (
                "model.vision_tower.encoder.layers.0.self_attn.k_norm.weight",
                "v.blk.0.attn_k_norm.weight",
            ),
            (
                "model.vision_tower.encoder.layers.0.post_attention_layernorm.weight",
                "v.blk.0.attn_post_norm.weight",
            ),
            (
                "model.vision_tower.encoder.layers.0.pre_feedforward_layernorm.weight",
                "v.blk.0.ln2.weight",
            ),
            (
                "model.vision_tower.encoder.layers.0.post_feedforward_layernorm.weight",
                "v.blk.0.ffn_post_norm.weight",
            ),
            (
                "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight",
                "v.blk.0.ffn_gate.weight",
            ),
            (
                "model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight",
                "v.blk.0.ffn_up.weight",
            ),
            (
                "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight",
                "v.blk.0.ffn_down.weight",
            ),
            (
                "model.vision_tower.encoder.layers.26.mlp.down_proj.linear.weight",
                "v.blk.26.ffn_down.weight",
            ),
        ];
        for (hf, gguf) in cases {
            assert_eq!(
                map_tensor_name(hf),
                Some(gguf.to_string()),
                "{} -> {} mismatch",
                hf,
                gguf
            );
        }
    }

    #[test]
    fn map_unknown_returns_none() {
        // Wrong prefix (Gemma 3 SigLIP convention should not match here).
        assert_eq!(
            map_tensor_name("model.vision_tower.vision_model.embeddings.patch_embedding.weight"),
            None
        );
        // Wrong suffix.
        assert_eq!(
            map_tensor_name("model.vision_tower.encoder.layers.0.unknown.weight"),
            None
        );
        // Wrong layer prefix.
        assert_eq!(
            map_tensor_name("model.vision_tower.encoder.layers.abc.input_layernorm.weight"),
            None
        );
    }
}
