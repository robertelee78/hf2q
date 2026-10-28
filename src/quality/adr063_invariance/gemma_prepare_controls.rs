//! Actual metadata derivation/recipe/shape guards, no model or device.
use super::*;
use crate::serve::config::{Gemma4Config,LayerType};
fn cfg() -> Gemma4Config {
    // Exact scalar mapping from frozen GGUF bf65b7e1...: block_count30,
    // head_count_kv[8,8,8,8,8,2] repeated; swa/global key_length256/512,
    // window1024/context262144. No tensor/tokenizer payload fixture.
    Gemma4Config {
        vocab_size: 262144,
        hidden_size: 2816,
        intermediate_size: 2112,
        moe_intermediate_size: 704,
        num_hidden_layers: 30,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        num_global_key_value_heads: 2,
        head_dim: 256,
        global_head_dim: 512,
        rms_norm_eps: 1e-6,
        rope_theta_sliding: 10000.,
        rope_theta_global: 1000000.,
        sliding_window: 1024,
        max_position_embeddings: 262144,
        final_logit_softcapping: Some(30.),
        attention_bias: false,
        attention_k_eq_v: true,
        tie_word_embeddings: true,
        num_experts: 128,
        top_k_experts: 8,
        layer_types: (0..30)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    LayerType::Full
                } else {
                    LayerType::Sliding
                }
            })
            .collect(),
    }
}
#[test]
fn actual_retained_input_slice_is_fixed_distinct144_not_generation() {
    let mut vectors:[Vec<u32>;4]=std::array::from_fn(|i|vec![i as u32;253952]);
    let work=derive(&vectors,262144).unwrap();
    assert_eq!(work.lanes.iter().map(Vec::len).collect::<Vec<_>>(),vec![144;4]);
    assert_eq!(work.lanes[2],vectors[2][..144]);
    vectors[3][0]=262144;assert!(derive(&vectors,262144).is_err());vectors[3][0]=3;
    vectors[3]=vectors[2].clone();assert!(derive(&vectors,262144).is_err());
    vectors[3]=vec![3;253951];assert!(derive(&vectors,262144).is_err());
    assert_ne!(le_hash(&[1,256]),le_hash(&[256,1]));
}
#[test]
fn metadata_recipe_and_capture_plan_use_actual_endpoint_functions() {
    let cfg=cfg();let r=super::super::gemma_actual::recipe(&cfg);
    assert_eq!(r.actual_model_config_sha256,hash(format!("{cfg:?}").as_bytes()));
    assert_eq!(r.capacity,262144);assert_eq!(r.slots,4);assert_eq!(r.output_allowance,8192);
    let plan=planned(&cfg).unwrap();assert_eq!(plan["full144_logits"],603979776u64);
    assert_eq!(plan["one_live_serial_anchor_payload"],158105600u64);
    assert!(plan["max_endpoint_payload"].as_u64().unwrap()<2*1024*1024*1024);
    let mut bad=cfg;bad.sliding_window=64;assert!(planned(&bad).is_err());
}
#[test]
fn config_rejects_unknown_fields_and_output_cannot_target_checkout() {
    let bound=serde_json::json!({"path":"/private/tmp/x","sha256":"a".repeat(64)});
    let mut v=serde_json::json!({"model":{"path":"/private/tmp/model","retained_full_sha256":MODEL_SHA,
        "file_identity":{"device":1,"inode":2,"bytes":3,"mtime":4,"mtime_ns":5,"ctime":6,"ctime_ns":7}},
        "full_content_receipt":bound,"tokenizer":bound,"template":bound,"producer_metadata":bound,
        "producer_build_receipt":bound,"producer_export_receipt":bound,"producer_config":bound,
        "producer_vectors":[bound,bound,bound,bound],"expected_exporter_source_commit":"b".repeat(40),
        "expected_exporter_binary_sha256":"c".repeat(64),"output":"/private/tmp/new-q1-gemma"});
    assert!(serde_json::from_value::<Config>(v.clone()).is_ok());
    v["model_fallback"]=serde_json::json!(true);
    assert!(serde_json::from_value::<Config>(v).err().unwrap().to_string().contains("unknown field"));
    assert!(output_preflight(&std::env::current_dir().unwrap()).is_err());
}

#[test]
fn header_boundary_refuses_family_fallback_or_template_drift() {
    assert!(require_header(Some("gemma4"),Some("exact"),"exact").is_ok());
    for architecture in [None,Some("qwen35"),Some("deepseek4")] {
        assert!(require_header(architecture,Some("exact"),"exact").is_err());
    }
    assert!(require_header(Some("gemma4"),Some("drift"),"exact").is_err());
    assert!(require_header(Some("gemma4"),None,"exact").is_err());
}
