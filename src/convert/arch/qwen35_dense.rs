//! Dense Qwen3.5-family converter used by Qwen3.8-27B.
//!
//! The official Qwen3.8 checkpoint uses the `Qwen3_5ForConditionalGeneration`
//! architecture and wraps the text decoder below `model.language_model.*`.
//! This module emits the native `qwen35` GGUF tensor and metadata contract
//! consumed by `src/inference/models/qwen35`; vision tensors are explicitly
//! separated from the text artifact.

use crate::backends::gguf::types::MetaValue;
use crate::convert::arch::bake::BakeOp;
use crate::convert::arch::qwen35moe_full::{
    map_linear_attn, MappedTensor, Qwen35LinearAttentionCtx,
};

/// HF dimensions required by the dense mapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DenseCtx {
    pub num_hidden_layers: usize,
    pub linear: Qwen35LinearAttentionCtx,
    pub multimodal_wrapping: bool,
}

/// Map one official HF tensor into the native qwen35 text GGUF.
///
/// Unknown text tensors return `None` and therefore fail conversion. Vision
/// tensors return `Drop` because they belong to the separately emitted
/// projector artifact.
pub fn map_tensor_name(
    hf_name: &str,
    hf_shape: &[usize],
    ctx: &Qwen35DenseCtx,
) -> Option<MappedTensor> {
    let canonical = if ctx.multimodal_wrapping {
        if let Some(stripped) = hf_name.strip_prefix("model.language_model.") {
            Some(format!("model.{stripped}"))
        } else if hf_name.starts_with("model.visual.") {
            return Some(MappedTensor::Drop);
        } else if hf_name == "lm_head.weight" || hf_name.starts_with("mtp.") {
            None
        } else {
            return None;
        }
    } else {
        None
    };
    let name = canonical.as_deref().unwrap_or(hf_name);

    if let Some(rest) = name.strip_prefix("mtp.") {
        return map_mtp(rest, hf_shape, ctx);
    }

    match name {
        "model.embed_tokens.weight" => {
            return Some(MappedTensor::Direct("token_embd.weight".into()));
        }
        "model.norm.weight" => {
            return Some(MappedTensor::DirectWithBake {
                gguf_name: "output_norm.weight".into(),
                bake: BakeOp::AddOne,
            });
        }
        "lm_head.weight" => return Some(MappedTensor::Direct("output.weight".into())),
        _ => {}
    }

    let rest = name.strip_prefix("model.layers.")?;
    let dot = rest.find('.')?;
    let (layer_text, suffix_with_dot) = rest.split_at(dot);
    let layer: usize = layer_text.parse().ok()?;
    if layer.to_string() != layer_text || layer >= ctx.num_hidden_layers {
        return None;
    }
    map_per_block(layer, &suffix_with_dot[1..], hf_shape, ctx)
}

fn map_per_block(
    layer: usize,
    rest: &str,
    hf_shape: &[usize],
    ctx: &Qwen35DenseCtx,
) -> Option<MappedTensor> {
    let blk = |suffix: &str| format!("blk.{layer}.{suffix}");

    match rest {
        "input_layernorm.weight" => {
            return Some(MappedTensor::DirectWithBake {
                gguf_name: blk("attn_norm.weight"),
                bake: BakeOp::AddOne,
            });
        }
        "post_attention_layernorm.weight" => {
            return Some(MappedTensor::DirectWithBake {
                gguf_name: blk("post_attention_norm.weight"),
                bake: BakeOp::AddOne,
            });
        }
        "self_attn.q_proj.weight" => {
            return Some(MappedTensor::Direct(blk("attn_q.weight")));
        }
        "self_attn.k_proj.weight" => {
            return Some(MappedTensor::Direct(blk("attn_k.weight")));
        }
        "self_attn.v_proj.weight" => {
            return Some(MappedTensor::Direct(blk("attn_v.weight")));
        }
        "self_attn.o_proj.weight" => {
            return Some(MappedTensor::Direct(blk("attn_output.weight")));
        }
        "self_attn.q_norm.weight" => {
            return Some(MappedTensor::DirectWithBake {
                gguf_name: blk("attn_q_norm.weight"),
                bake: BakeOp::AddOne,
            });
        }
        "self_attn.k_norm.weight" => {
            return Some(MappedTensor::DirectWithBake {
                gguf_name: blk("attn_k_norm.weight"),
                bake: BakeOp::AddOne,
            });
        }
        "mlp.gate_proj.weight" => {
            return Some(MappedTensor::Direct(blk("ffn_gate.weight")));
        }
        "mlp.up_proj.weight" => return Some(MappedTensor::Direct(blk("ffn_up.weight"))),
        "mlp.down_proj.weight" => {
            return Some(MappedTensor::Direct(blk("ffn_down.weight")));
        }
        _ => {}
    }

    let linear_rest = rest.strip_prefix("linear_attn.")?;
    map_linear_attn(layer, linear_rest, hf_shape, &ctx.linear)
}

fn map_mtp(rest: &str, hf_shape: &[usize], ctx: &Qwen35DenseCtx) -> Option<MappedTensor> {
    let layer = ctx.num_hidden_layers;
    let nextn = |suffix: &str| format!("blk.{layer}.nextn.{suffix}");

    match rest {
        "fc.weight" => return Some(MappedTensor::Direct(nextn("eh_proj.weight"))),
        "pre_fc_norm_embedding.weight" => {
            return Some(MappedTensor::DirectWithBake {
                gguf_name: nextn("enorm.weight"),
                bake: BakeOp::AddOne,
            });
        }
        "pre_fc_norm_hidden.weight" => {
            return Some(MappedTensor::DirectWithBake {
                gguf_name: nextn("hnorm.weight"),
                bake: BakeOp::AddOne,
            });
        }
        "norm.weight" => {
            return Some(MappedTensor::DirectWithBake {
                gguf_name: nextn("shared_head_norm.weight"),
                bake: BakeOp::AddOne,
            });
        }
        _ => {}
    }

    let layers = rest.strip_prefix("layers.")?;
    let dot = layers.find('.')?;
    let (index_text, suffix_with_dot) = layers.split_at(dot);
    let index: usize = index_text.parse().ok()?;
    if index.to_string() != index_text || index != 0 {
        return None;
    }
    map_per_block(layer, &suffix_with_dot[1..], hf_shape, ctx)
}

fn effective_text_config(config: &serde_json::Value) -> &serde_json::Value {
    config.get("text_config").unwrap_or(config)
}

/// Build the metadata consumed by the native dense qwen35 loader.
pub fn build_metadata(
    config: &serde_json::Value,
    file_type: u32,
    model_card: Option<&crate::convert::model_card::ModelCard>,
    sampling: Option<&crate::convert::model_card::SamplingConfig>,
    model_dir_basename: Option<&str>,
    size_label_override: Option<&str>,
) -> Vec<(String, MetaValue)> {
    use crate::convert::model_card::{
        emit_general_postlude, emit_general_prelude, get_model_id_components,
    };

    let text = effective_text_config(config);
    let u32_required = |key: &str| {
        text.get(key)
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!("config missing {key}")) as u32
    };
    let raw_name = model_dir_basename
        .map(ToOwned::to_owned)
        .or_else(|| {
            config
                .get("_name_or_path")
                .and_then(|v| v.as_str())
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| "model".into());
    let id = get_model_id_components(&raw_name);
    let display_name = id.name.clone().unwrap_or_else(|| raw_name.clone());

    let hidden = u32_required("hidden_size");
    let base_layers = u32_required("num_hidden_layers");
    let mtp_layers = text
        .get("mtp_num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let heads = u32_required("num_attention_heads");
    let kv_heads = text
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(heads as u64) as u32;
    let head_dim = text
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or((hidden / heads) as u64) as u32;
    let partial_rotary = text
        .get("partial_rotary_factor")
        .and_then(|v| v.as_f64())
        .or_else(|| {
            text.get("rope_parameters")
                .and_then(|v| v.get("partial_rotary_factor"))
                .and_then(|v| v.as_f64())
        })
        .unwrap_or(0.25);
    let rope_theta = text
        .get("rope_parameters")
        .and_then(|v| v.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .or_else(|| text.get("rope_theta").and_then(|v| v.as_f64()))
        .unwrap_or(10_000.0) as f32;
    let mut sections: Vec<i32> = text
        .get("rope_parameters")
        .and_then(|v| v.get("mrope_section"))
        .or_else(|| text.get("mrope_section"))
        .and_then(|v| v.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|v| v.as_i64().map(|n| n as i32))
                .collect()
        })
        .unwrap_or_else(|| vec![11, 11, 10]);
    while sections.len() < 4 {
        sections.push(0);
    }
    sections.truncate(4);

    let mut kv = emit_general_prelude(
        "qwen35",
        display_name,
        &id,
        size_label_override,
        model_card,
        sampling,
    );
    kv.extend([
        (
            "qwen35.block_count".into(),
            MetaValue::U32(base_layers + mtp_layers),
        ),
        (
            "qwen35.context_length".into(),
            MetaValue::U32(u32_required("max_position_embeddings")),
        ),
        ("qwen35.embedding_length".into(), MetaValue::U32(hidden)),
        (
            "qwen35.feed_forward_length".into(),
            MetaValue::U32(u32_required("intermediate_size")),
        ),
        ("qwen35.attention.head_count".into(), MetaValue::U32(heads)),
        (
            "qwen35.attention.head_count_kv".into(),
            MetaValue::U32(kv_heads),
        ),
        (
            "qwen35.rope.dimension_sections".into(),
            MetaValue::ArrayI32(sections),
        ),
        ("qwen35.rope.freq_base".into(), MetaValue::F32(rope_theta)),
        (
            "qwen35.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F32(
                text.get("rms_norm_eps")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1e-6) as f32,
            ),
        ),
        (
            "qwen35.attention.key_length".into(),
            MetaValue::U32(head_dim),
        ),
        (
            "qwen35.attention.value_length".into(),
            MetaValue::U32(head_dim),
        ),
        (
            "qwen35.ssm.conv_kernel".into(),
            MetaValue::U32(u32_required("linear_conv_kernel_dim")),
        ),
        (
            "qwen35.ssm.state_size".into(),
            MetaValue::U32(u32_required("linear_key_head_dim")),
        ),
        (
            "qwen35.ssm.group_count".into(),
            MetaValue::U32(u32_required("linear_num_key_heads")),
        ),
        (
            "qwen35.ssm.time_step_rank".into(),
            MetaValue::U32(u32_required("linear_num_value_heads")),
        ),
        (
            "qwen35.ssm.inner_size".into(),
            MetaValue::U32(
                u32_required("linear_num_value_heads") * u32_required("linear_value_head_dim"),
            ),
        ),
        (
            "qwen35.full_attention_interval".into(),
            MetaValue::U32(
                text.get("full_attention_interval")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4) as u32,
            ),
        ),
        (
            "qwen35.rope.dimension_count".into(),
            MetaValue::U32((head_dim as f64 * partial_rotary) as u32),
        ),
    ]);
    let is_multimodal = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .is_some_and(|architectures| {
            architectures.iter().any(|value| {
                matches!(
                    value.as_str(),
                    Some("Qwen3_5ForConditionalGeneration")
                        | Some("Qwen3_5MoeForConditionalGeneration")
                )
            })
        });
    if is_multimodal {
        let deepstack_layers = config
            .get("vision_config")
            .and_then(|value| value.get("deepstack_visual_indexes"))
            .and_then(|value| value.as_array())
            .map_or(0, |values| values.len() as u32);
        kv.push((
            "hf2q.vision.projector_profile".into(),
            MetaValue::String("qwen3vl_siglip".into()),
        ));
        kv.push((
            "hf2q.vision.deepstack_output_count".into(),
            MetaValue::U32(deepstack_layers),
        ));
    }
    if mtp_layers > 0 {
        kv.push((
            "qwen35.nextn_predict_layers".into(),
            MetaValue::U32(mtp_layers),
        ));
        kv.push((
            "qwen35.nextn.use_dedicated_embeddings".into(),
            MetaValue::Bool(
                text.get("mtp_use_dedicated_embeddings")
                    .and_then(|v| v.as_bool())
                    // The official Qwen3.8 configuration explicitly shares
                    // the main token embedding with the appended MTP block.
                    // Preserve that native-family default for older configs;
                    // an explicit checkpoint value always wins.
                    .unwrap_or(false),
            ),
        ));
    }
    kv.extend(emit_general_postlude(file_type));
    kv
}

#[cfg(test)]
#[path = "qwen35_dense_tests.rs"]
mod tests;
