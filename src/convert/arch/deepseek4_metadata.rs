//! Canonical GGUF metadata for DeepSeek-V4 text checkpoints.

use crate::backends::gguf::types::MetaValue;

fn u32_req(config: &serde_json::Value, key: &'static str) -> u32 {
    config[key]
        .as_u64()
        .unwrap_or_else(|| panic!("config.json missing required u32 key `{key}`")) as u32
}

fn f32_req(config: &serde_json::Value, key: &'static str) -> f32 {
    config[key]
        .as_f64()
        .unwrap_or_else(|| panic!("config.json missing required f32 key `{key}`")) as f32
}

/// Build the architecture block emitted by llama.cpp's
/// `DeepseekV4Model.set_gguf_parameters`, plus the common text-model
/// keys it inherits.
pub fn build_metadata(
    config: &serde_json::Value,
    file_type: u32,
    model_card: Option<&crate::convert::model_card::ModelCard>,
    sampling: Option<&crate::convert::model_card::SamplingConfig>,
    model_dir_basename: Option<&str>,
) -> Vec<(String, MetaValue)> {
    use crate::convert::model_card::{
        emit_general_postlude, emit_general_prelude, get_model_id_components,
    };

    let raw_name = config
        .get("_name_or_path")
        .and_then(|v| v.as_str())
        .or(model_dir_basename)
        .unwrap_or("DeepSeek-V4")
        .to_string();
    let ids = get_model_id_components(&raw_name);
    let display_name = ids.name.clone().unwrap_or(raw_name);
    let mut kv = emit_general_prelude("deepseek4", display_name, &ids, None, model_card, sampling);

    let layers = u32_req(config, "num_hidden_layers");
    let hidden = u32_req(config, "hidden_size");
    let heads = u32_req(config, "num_attention_heads");
    let kv_heads = u32_req(config, "num_key_value_heads");
    let context = u32_req(config, "max_position_embeddings");
    let rms_eps = f32_req(config, "rms_norm_eps");
    let experts = u32_req(config, "n_routed_experts");
    let used = u32_req(config, "num_experts_per_tok");
    let shared = u32_req(config, "n_shared_experts");
    let swiglu = f32_req(config, "swiglu_limit");
    assert_eq!(
        config["scoring_func"].as_str(),
        Some("sqrtsoftplus"),
        "DeepSeek-V4 requires scoring_func=sqrtsoftplus"
    );

    kv.extend([
        ("deepseek4.block_count".into(), MetaValue::U32(layers)),
        ("deepseek4.context_length".into(), MetaValue::U32(context)),
        ("deepseek4.embedding_length".into(), MetaValue::U32(hidden)),
        (
            "deepseek4.embedding_length_out".into(),
            MetaValue::U32(hidden * u32_req(config, "hc_mult")),
        ),
        (
            "deepseek4.attention.head_count".into(),
            MetaValue::U32(heads),
        ),
        (
            "deepseek4.attention.head_count_kv".into(),
            MetaValue::U32(kv_heads),
        ),
        (
            "deepseek4.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F32(rms_eps),
        ),
        (
            "deepseek4.vocab_size".into(),
            MetaValue::U32(u32_req(config, "vocab_size")),
        ),
        ("deepseek4.expert_count".into(), MetaValue::U32(experts)),
        ("deepseek4.expert_used_count".into(), MetaValue::U32(used)),
        (
            "deepseek4.expert_shared_count".into(),
            MetaValue::U32(shared),
        ),
        (
            "deepseek4.expert_feed_forward_length".into(),
            MetaValue::U32(u32_req(config, "moe_intermediate_size")),
        ),
        (
            "deepseek4.expert_weights_scale".into(),
            MetaValue::F32(f32_req(config, "routed_scaling_factor")),
        ),
        (
            "deepseek4.expert_weights_norm".into(),
            MetaValue::Bool(
                config["norm_topk_prob"]
                    .as_bool()
                    .expect("config.json missing bool `norm_topk_prob`"),
            ),
        ),
        // `sqrtsoftplus` is GGUF ExpertGatingFuncType value 4.
        ("deepseek4.expert_gating_func".into(), MetaValue::U32(4)),
        (
            "deepseek4.swiglu_clamp_exp".into(),
            MetaValue::ArrayF32(vec![swiglu; layers as usize]),
        ),
        (
            "deepseek4.swiglu_clamp_shexp".into(),
            MetaValue::ArrayF32(vec![swiglu; layers as usize]),
        ),
        (
            "deepseek4.rope.dimension_count".into(),
            MetaValue::U32(u32_req(config, "qk_rope_head_dim")),
        ),
        (
            "deepseek4.attention.q_lora_rank".into(),
            MetaValue::U32(u32_req(config, "q_lora_rank")),
        ),
        (
            "deepseek4.attention.sliding_window".into(),
            MetaValue::U32(u32_req(config, "sliding_window")),
        ),
        (
            "deepseek4.attention.indexer.head_count".into(),
            MetaValue::U32(u32_req(config, "index_n_heads")),
        ),
        (
            "deepseek4.attention.indexer.key_length".into(),
            MetaValue::U32(u32_req(config, "index_head_dim")),
        ),
        (
            "deepseek4.attention.indexer.top_k".into(),
            MetaValue::U32(u32_req(config, "index_topk")),
        ),
        (
            "deepseek4.attention.output_group_count".into(),
            MetaValue::U32(u32_req(config, "o_groups")),
        ),
        (
            "deepseek4.attention.output_lora_rank".into(),
            MetaValue::U32(u32_req(config, "o_lora_rank")),
        ),
        (
            "deepseek4.attention.compress_rope_freq_base".into(),
            MetaValue::F32(f32_req(config, "compress_rope_theta")),
        ),
        (
            "deepseek4.hyper_connection.count".into(),
            MetaValue::U32(u32_req(config, "hc_mult")),
        ),
        (
            "deepseek4.hyper_connection.sinkhorn_iterations".into(),
            MetaValue::U32(u32_req(config, "hc_sinkhorn_iters")),
        ),
        (
            "deepseek4.hyper_connection.epsilon".into(),
            MetaValue::F32(f32_req(config, "hc_eps")),
        ),
        (
            "deepseek4.hash_layer_count".into(),
            MetaValue::U32(u32_req(config, "num_hash_layers")),
        ),
    ]);

    let ratios = config["compress_ratios"]
        .as_array()
        .expect("config.json missing array `compress_ratios`")
        .iter()
        .map(|v| v.as_u64().expect("compress ratio must be u32") as u32)
        .collect::<Vec<_>>();
    assert_eq!(ratios.len(), layers as usize, "one compress ratio per layer");
    assert!(
        ratios.iter().all(|v| matches!(v, 0 | 4 | 128)),
        "DeepSeek-V4 compress ratios must be 0, 4, or 128"
    );
    kv.push((
        "deepseek4.attention.compress_ratios".into(),
        MetaValue::ArrayU32(ratios),
    ));

    if let Some(rope) = config.get("rope_scaling").and_then(|v| v.as_object()) {
        let rope_type = rope
            .get("rope_type")
            .or_else(|| rope.get("type"))
            .and_then(|v| v.as_str());
        if rope_type == Some("yarn") {
            kv.push((
                "deepseek4.rope.scaling.type".into(),
                MetaValue::String("yarn".into()),
            ));
            if let Some(v) = rope.get("factor").and_then(|v| v.as_f64()) {
                kv.push((
                    "deepseek4.rope.scaling.factor".into(),
                    MetaValue::F32(v as f32),
                ));
            }
            if let Some(v) = rope
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
            {
                kv.push((
                    "deepseek4.rope.scaling.original_context_length".into(),
                    MetaValue::U32(v as u32),
                ));
            }
        }
    }
    if let Some(v) = config.get("rope_theta").and_then(|v| v.as_f64()) {
        kv.push(("deepseek4.rope.freq_base".into(), MetaValue::F32(v as f32)));
    }

    kv.extend(emit_general_postlude(file_type));
    kv
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn emits_deepseek4_specific_contract() {
        let cfg = json!({
            "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 1, "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6, "vocab_size": 32, "n_routed_experts": 4,
            "num_experts_per_tok": 2, "n_shared_experts": 1,
            "moe_intermediate_size": 32, "routed_scaling_factor": 1.5,
            "norm_topk_prob": true, "scoring_func":"sqrtsoftplus",
            "swiglu_limit": 10.0, "qk_rope_head_dim": 8,
            "q_lora_rank": 8, "sliding_window": 128, "index_n_heads": 4,
            "index_head_dim": 8, "index_topk": 16, "o_groups": 2,
            "o_lora_rank": 8, "compress_ratios": [0, 4], "compress_rope_theta": 160000.0,
            "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
            "num_hash_layers": 1, "rope_scaling": {"type":"yarn", "factor":16.0,
              "original_max_position_embeddings":65536}
        });
        let by_key: std::collections::HashMap<_, _> =
            build_metadata(&cfg, 21, None, None, Some("DeepSeek-V4-Flash-0731"))
                .into_iter()
                .collect();
        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("deepseek4".into())
        );
        assert_eq!(by_key["deepseek4.embedding_length_out"], MetaValue::U32(64));
        assert_eq!(by_key["deepseek4.expert_gating_func"], MetaValue::U32(4));
        assert_eq!(
            by_key["deepseek4.attention.compress_ratios"],
            MetaValue::ArrayU32(vec![0, 4])
        );
    }
}
