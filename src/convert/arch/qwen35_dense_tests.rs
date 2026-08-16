use super::*;
use crate::convert::arch::qwen35moe_full::MappedTensor;

fn ctx() -> Qwen35DenseCtx {
    Qwen35DenseCtx {
        num_hidden_layers: 64,
        linear: Qwen35LinearAttentionCtx {
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
        },
        multimodal_wrapping: true,
    }
}

#[test]
fn qwen38_maps_dense_text_and_drops_vision() {
    assert!(matches!(
        map_tensor_name(
            "model.language_model.layers.0.mlp.gate_proj.weight",
            &[17408, 5120],
            &ctx()
        ),
        Some(MappedTensor::Direct(name)) if name == "blk.0.ffn_gate.weight"
    ));
    assert!(matches!(
        map_tensor_name("model.visual.pos_embed.weight", &[2304, 1152], &ctx()),
        Some(MappedTensor::Drop)
    ));
}

#[test]
fn qwen38_maps_all_fifteen_mtp_tensors() {
    let names = [
        "mtp.fc.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ];
    for name in names {
        assert!(
            map_tensor_name(name, &[1], &ctx()).is_some(),
            "unmapped official Qwen3.8 MTP tensor {name}"
        );
    }
}

#[test]
fn qwen38_official_text_inventory_maps_to_866_unique_tensors() {
    let ctx = ctx();
    let mut source: Vec<(String, Vec<usize>)> = vec![
        (
            "model.language_model.embed_tokens.weight".into(),
            vec![248320, 5120],
        ),
        ("model.language_model.norm.weight".into(), vec![5120]),
        ("lm_head.weight".into(), vec![248320, 5120]),
    ];
    for layer in 0..64 {
        let p = format!("model.language_model.layers.{layer}");
        source.extend([
            (format!("{p}.input_layernorm.weight"), vec![5120]),
            (format!("{p}.post_attention_layernorm.weight"), vec![5120]),
            (format!("{p}.mlp.gate_proj.weight"), vec![17408, 5120]),
            (format!("{p}.mlp.up_proj.weight"), vec![17408, 5120]),
            (format!("{p}.mlp.down_proj.weight"), vec![5120, 17408]),
        ]);
        if (layer + 1) % 4 == 0 {
            source.extend([
                // Qwen3.5-family full attention fuses the sigmoid output
                // gate above the query rows: 2 * (24 * 256) = 12,288.
                (format!("{p}.self_attn.q_proj.weight"), vec![12288, 5120]),
                (format!("{p}.self_attn.k_proj.weight"), vec![1024, 5120]),
                (format!("{p}.self_attn.v_proj.weight"), vec![1024, 5120]),
                (format!("{p}.self_attn.o_proj.weight"), vec![5120, 6144]),
                (format!("{p}.self_attn.q_norm.weight"), vec![256]),
                (format!("{p}.self_attn.k_norm.weight"), vec![256]),
            ]);
        } else {
            source.extend([
                (format!("{p}.linear_attn.A_log"), vec![48]),
                (format!("{p}.linear_attn.conv1d.weight"), vec![10240, 1, 4]),
                (format!("{p}.linear_attn.dt_bias"), vec![48]),
                (format!("{p}.linear_attn.in_proj_a.weight"), vec![48, 5120]),
                (format!("{p}.linear_attn.in_proj_b.weight"), vec![48, 5120]),
                (
                    format!("{p}.linear_attn.in_proj_qkv.weight"),
                    vec![10240, 5120],
                ),
                (
                    format!("{p}.linear_attn.in_proj_z.weight"),
                    vec![6144, 5120],
                ),
                (format!("{p}.linear_attn.norm.weight"), vec![128]),
                (format!("{p}.linear_attn.out_proj.weight"), vec![5120, 6144]),
            ]);
        }
    }
    source.extend([
        ("mtp.fc.weight".into(), vec![5120, 10240]),
        ("mtp.layers.0.input_layernorm.weight".into(), vec![5120]),
        (
            "mtp.layers.0.mlp.down_proj.weight".into(),
            vec![5120, 17408],
        ),
        (
            "mtp.layers.0.mlp.gate_proj.weight".into(),
            vec![17408, 5120],
        ),
        ("mtp.layers.0.mlp.up_proj.weight".into(), vec![17408, 5120]),
        (
            "mtp.layers.0.post_attention_layernorm.weight".into(),
            vec![5120],
        ),
        ("mtp.layers.0.self_attn.k_norm.weight".into(), vec![256]),
        (
            "mtp.layers.0.self_attn.k_proj.weight".into(),
            vec![1024, 5120],
        ),
        (
            "mtp.layers.0.self_attn.o_proj.weight".into(),
            vec![5120, 6144],
        ),
        ("mtp.layers.0.self_attn.q_norm.weight".into(), vec![256]),
        (
            "mtp.layers.0.self_attn.q_proj.weight".into(),
            vec![12288, 5120],
        ),
        (
            "mtp.layers.0.self_attn.v_proj.weight".into(),
            vec![1024, 5120],
        ),
        ("mtp.norm.weight".into(), vec![5120]),
        ("mtp.pre_fc_norm_embedding.weight".into(), vec![5120]),
        ("mtp.pre_fc_norm_hidden.weight".into(), vec![5120]),
    ]);

    assert_eq!(source.len(), 866);
    let mut destinations = std::collections::HashSet::new();
    for (name, shape) in source {
        let mapped = map_tensor_name(&name, &shape, &ctx)
            .unwrap_or_else(|| panic!("unmapped official Qwen3.8 tensor {name}"));
        let destination = match mapped {
            MappedTensor::Direct(name) => name,
            MappedTensor::DirectWithBake { gguf_name, .. } => gguf_name,
            other => panic!("unexpected map outcome for {name}: {other:?}"),
        };
        assert!(
            destinations.insert(destination.clone()),
            "duplicate {destination}"
        );
    }
    assert_eq!(destinations.len(), 866);
}

#[test]
fn qwen38_metadata_matches_native_loader_contract() {
    let config: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/qwen38/config.json"))
            .expect("fixture");
    let kv = build_metadata(&config, 15, None, None, Some("Qwen3.8-27B"), None);
    let map: std::collections::HashMap<_, _> = kv.into_iter().collect();
    assert_eq!(
        map["general.architecture"],
        MetaValue::String("qwen35".into())
    );
    assert_eq!(map["qwen35.block_count"], MetaValue::U32(65));
    assert_eq!(map["qwen35.feed_forward_length"], MetaValue::U32(17408));
    assert_eq!(map["qwen35.context_length"], MetaValue::U32(262144));
    assert_eq!(map["qwen35.nextn_predict_layers"], MetaValue::U32(1));
    assert_eq!(
        map["qwen35.nextn.use_dedicated_embeddings"],
        MetaValue::Bool(false)
    );

    let mut explicit = config;
    explicit["text_config"]["mtp_use_dedicated_embeddings"] = serde_json::json!(true);
    let explicit_map: std::collections::HashMap<_, _> =
        build_metadata(&explicit, 15, None, None, Some("future-dense"), None)
            .into_iter()
            .collect();
    assert_eq!(
        explicit_map["qwen35.nextn.use_dedicated_embeddings"],
        MetaValue::Bool(true)
    );
}
