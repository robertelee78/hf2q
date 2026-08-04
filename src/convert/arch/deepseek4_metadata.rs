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

fn base_compress_ratios(config: &serde_json::Value, layers: u32) -> Vec<u32> {
    let ratios = config["compress_ratios"]
        .as_array()
        .expect("config.json missing array `compress_ratios`")
        .iter()
        .map(|v| v.as_u64().expect("compress ratio must be u32") as u32)
        .collect::<Vec<_>>();
    let layer_count = layers as usize;
    assert!(
        ratios.len() >= layer_count,
        "DeepSeek-V4 compress ratios must cover every base layer"
    );
    assert!(
        ratios[..layer_count]
            .iter()
            .all(|v| matches!(v, 0 | 4 | 128)),
        "DeepSeek-V4 base compress ratios must be 0, 4, or 128"
    );

    // The official 0731 config appends three zero entries for checkpoint
    // MTP/DSpark stages. Base conversion excludes every `mtp.*` tensor, so
    // those entries must not leak into the 43-element base-model metadata.
    let excluded_tail = &ratios[layer_count..];
    assert!(
        excluded_tail.is_empty()
            || (excluded_tail.len() == 3 && excluded_tail.iter().all(|&ratio| ratio == 0)),
        "DeepSeek-V4 excluded MTP/DSpark compress-ratio tail must be exactly three zeros"
    );

    ratios[..layer_count].to_vec()
}

/// Build the base-artifact architecture block emitted by llama.cpp's
/// `DeepseekV4Model.set_gguf_parameters`, plus the common text-model
/// keys it inherits. Checkpoint-only MTP/DSpark array entries are excluded.
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
            "deepseek4.attention.key_length".into(),
            MetaValue::U32(u32_req(config, "head_dim")),
        ),
        (
            "deepseek4.attention.value_length".into(),
            MetaValue::U32(u32_req(config, "head_dim")),
        ),
        (
            "deepseek4.attention.layer_norm_rms_epsilon".into(),
            MetaValue::F32(rms_eps),
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

    let ratios = base_compress_ratios(config, layers);
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
            if let Some(v) = rope.get("extrapolation_factor").and_then(|v| v.as_f64()) {
                kv.push((
                    "deepseek4.rope.scaling.yarn_ext_factor".into(),
                    MetaValue::F32(v as f32),
                ));
            }
            if let Some(v) = rope
                .get("attention_factor")
                .or_else(|| rope.get("attn_factor"))
                .and_then(|v| v.as_f64())
            {
                kv.push((
                    "deepseek4.rope.scaling.yarn_attn_factor".into(),
                    MetaValue::F32(v as f32),
                ));
            }
            if let Some(v) = rope.get("beta_fast").and_then(|v| v.as_f64()) {
                kv.push((
                    "deepseek4.rope.scaling.yarn_beta_fast".into(),
                    MetaValue::F32(v as f32),
                ));
            }
            if let Some(v) = rope.get("beta_slow").and_then(|v| v.as_f64()) {
                kv.push((
                    "deepseek4.rope.scaling.yarn_beta_slow".into(),
                    MetaValue::F32(v as f32),
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

    fn fixture() -> serde_json::Value {
        json!({
            "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 1, "head_dim": 512, "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6, "vocab_size": 32, "n_routed_experts": 4,
            "num_experts_per_tok": 2, "n_shared_experts": 1,
            "moe_intermediate_size": 32, "routed_scaling_factor": 1.5,
            "norm_topk_prob": true, "scoring_func":"sqrtsoftplus",
            "swiglu_limit": 10.0, "qk_rope_head_dim": 8,
            "q_lora_rank": 8, "sliding_window": 128, "index_n_heads": 4,
            "index_head_dim": 8, "index_topk": 16, "o_groups": 2,
            "o_lora_rank": 8, "compress_ratios": [0, 4], "compress_rope_theta": 160000.0,
            "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
            "num_hash_layers": 1, "rope_theta":10000.0,
            "rope_scaling": {"type":"yarn", "factor":16.0,
              "original_max_position_embeddings":65536, "extrapolation_factor":1.25,
              "attention_factor":1.125, "beta_fast":32.0, "beta_slow":1.0}
        })
    }

    #[test]
    fn emits_deepseek4_specific_contract() {
        let cfg = fixture();
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
        assert!(!by_key.contains_key("deepseek4.vocab_size"));
        assert_eq!(
            by_key["deepseek4.attention.key_length"],
            MetaValue::U32(512)
        );
        assert_eq!(
            by_key["deepseek4.attention.value_length"],
            MetaValue::U32(512)
        );
        assert_eq!(
            by_key["deepseek4.attention.compress_ratios"],
            MetaValue::ArrayU32(vec![0, 4])
        );
        assert_eq!(
            by_key["deepseek4.rope.scaling.yarn_ext_factor"],
            MetaValue::F32(1.25)
        );
        assert_eq!(
            by_key["deepseek4.rope.scaling.yarn_attn_factor"],
            MetaValue::F32(1.125)
        );
        assert_eq!(
            by_key["deepseek4.rope.scaling.yarn_beta_fast"],
            MetaValue::F32(32.0)
        );
        assert_eq!(
            by_key["deepseek4.rope.scaling.yarn_beta_slow"],
            MetaValue::F32(1.0)
        );

        let actual_keys = by_key
            .keys()
            .filter(|key| key.starts_with("deepseek4."))
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let expected_keys = [
            "deepseek4.block_count",
            "deepseek4.context_length",
            "deepseek4.embedding_length",
            "deepseek4.embedding_length_out",
            "deepseek4.attention.head_count",
            "deepseek4.attention.head_count_kv",
            "deepseek4.attention.key_length",
            "deepseek4.attention.value_length",
            "deepseek4.attention.layer_norm_rms_epsilon",
            "deepseek4.expert_count",
            "deepseek4.expert_used_count",
            "deepseek4.expert_shared_count",
            "deepseek4.expert_feed_forward_length",
            "deepseek4.expert_weights_scale",
            "deepseek4.expert_weights_norm",
            "deepseek4.expert_gating_func",
            "deepseek4.swiglu_clamp_exp",
            "deepseek4.swiglu_clamp_shexp",
            "deepseek4.rope.dimension_count",
            "deepseek4.attention.q_lora_rank",
            "deepseek4.attention.sliding_window",
            "deepseek4.attention.indexer.head_count",
            "deepseek4.attention.indexer.key_length",
            "deepseek4.attention.indexer.top_k",
            "deepseek4.attention.output_group_count",
            "deepseek4.attention.output_lora_rank",
            "deepseek4.attention.compress_rope_freq_base",
            "deepseek4.hyper_connection.count",
            "deepseek4.hyper_connection.sinkhorn_iterations",
            "deepseek4.hyper_connection.epsilon",
            "deepseek4.hash_layer_count",
            "deepseek4.attention.compress_ratios",
            "deepseek4.rope.scaling.type",
            "deepseek4.rope.scaling.factor",
            "deepseek4.rope.scaling.original_context_length",
            "deepseek4.rope.scaling.yarn_ext_factor",
            "deepseek4.rope.scaling.yarn_attn_factor",
            "deepseek4.rope.scaling.yarn_beta_fast",
            "deepseek4.rope.scaling.yarn_beta_slow",
            "deepseek4.rope.freq_base",
        ]
        .into_iter()
        .collect();
        assert_eq!(actual_keys, expected_keys);
    }

    #[test]
    #[should_panic(expected = "head_dim")]
    fn head_dim_is_required_like_current_llama_converter() {
        let mut cfg = fixture();
        cfg.as_object_mut().unwrap().remove("head_dim");
        let _ = build_metadata(&cfg, 21, None, None, Some("DeepSeek-V4-Flash-0731"));
    }

    #[test]
    fn official_43_layer_config_excludes_three_zero_mtp_dspark_ratios() {
        let mut cfg = fixture();
        cfg["num_hidden_layers"] = json!(43);
        let mut ratios = vec![0, 0];
        ratios.extend((2..43).map(|layer| if layer % 2 == 0 { 4 } else { 128 }));
        ratios.extend([0, 0, 0]);
        cfg["compress_ratios"] = json!(ratios);

        let by_key: std::collections::HashMap<_, _> =
            build_metadata(&cfg, 21, None, None, Some("DeepSeek-V4-Flash-0731"))
                .into_iter()
                .collect();
        let MetaValue::ArrayU32(emitted) = &by_key["deepseek4.attention.compress_ratios"] else {
            panic!("compress ratios must be a u32 array");
        };
        assert_eq!(emitted.len(), 43);
        assert_eq!(emitted, &ratios[..43]);
    }

    #[test]
    #[should_panic(expected = "exactly three zeros")]
    fn malformed_excluded_compress_ratio_tail_fails_closed() {
        let mut cfg = fixture();
        cfg["compress_ratios"] = json!([0, 4, 0, 4, 0]);
        let _ = build_metadata(&cfg, 21, None, None, Some("DeepSeek-V4-Flash-0731"));
    }
}
