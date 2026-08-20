use super::*;
use crate::inference::models::qwen35::{default_layer_types, Qwen35Variant};

fn config_with_authenticated_mtp() -> Qwen35Config {
    Qwen35Config {
        variant: Qwen35Variant::Dense,
        hidden_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 1,
        head_dim: 64,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 2,
        layer_types: default_layer_types(2, 2),
        partial_rotary_factor: 0.25,
        rope_theta: 10_000.0,
        rotary_dim: 16,
        mrope_section: [4, 4, 4, 4],
        mrope_interleaved: true,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 64,
        vocab_size: 32,
        attn_output_gate: true,
        mtp_num_hidden_layers: 1,
        mtp_use_dedicated_embeddings: false,
        intermediate_size: Some(512),
        moe: None,
    }
}

fn official_dense_cache_config() -> Qwen35Config {
    let mut config = config_with_authenticated_mtp();
    config.hidden_size = 5_120;
    config.num_hidden_layers = 64;
    config.num_attention_heads = 24;
    config.num_key_value_heads = 4;
    config.head_dim = 256;
    config.linear_num_key_heads = 16;
    config.linear_num_value_heads = 48;
    config.linear_key_head_dim = 128;
    config.linear_value_head_dim = 128;
    config.full_attention_interval = 4;
    config.layer_types = default_layer_types(64, 4);
    config.rotary_dim = 64;
    config.mrope_section = [11, 11, 10, 0];
    config.max_position_embeddings = 262_144;
    config.vocab_size = 248_320;
    config.intermediate_size = Some(17_408);
    config
}

#[test]
fn checked_plan_excludes_mtp_and_rejects_invalid_geometry() {
    let config = config_with_authenticated_mtp();
    let plan = plan_qwen35_base_text_cache(&config, 16).unwrap();
    assert_eq!(plan.full_attention_slots, 1);
    assert_eq!(plan.linear_attention_slots, 1);
    assert_eq!(plan.buffer_records.len(), 6);
    assert_eq!(plan.base_full_attention_cache_bytes, 2 * 16 * 64 * 4);
    assert_eq!(
        plan.base_linear_attention_state_bytes,
        2 * (512 * 3 * 4) + 2 * (128 * 128 * 2 * 4)
    );
    assert_eq!(
        plan.total_payload_bytes,
        plan.base_full_attention_cache_bytes + plan.base_linear_attention_state_bytes
    );
    assert!(!plan.mtp_slot_allocated);
    assert!(!plan.tq_kv_active);
    assert!(!plan.linear_capture_allocated);
    assert_eq!(plan.layout_sha256.len(), 64);

    let mut malformed = config.clone();
    malformed.layer_types.pop();
    assert!(plan_qwen35_base_text_cache(&malformed, 16).is_err());
    let mut zero = config.clone();
    zero.linear_num_key_heads = 0;
    assert!(plan_qwen35_base_text_cache(&zero, 16).is_err());
    let mut moe = config.clone();
    moe.variant = Qwen35Variant::Moe;
    moe.intermediate_size = None;
    assert!(plan_qwen35_base_text_cache(&moe, 16).is_err());
    let mut too_many_layers = config.clone();
    too_many_layers.num_hidden_layers = 257;
    too_many_layers.layer_types = default_layer_types(257, 2);
    assert!(plan_qwen35_base_text_cache(&too_many_layers, 16).is_err());
    assert!(plan_qwen35_base_text_cache(&config, 0).is_err());
    assert!(plan_qwen35_base_text_cache(&config, 65).is_err());
    let official = official_dense_cache_config();
    assert!(plan_qwen35_base_text_cache(&official, 4_097).is_err());
}

#[test]
fn official_dense_4k_cache_payload_matches_the_b3a_oracle() {
    let plan = plan_qwen35_base_text_cache(&official_dense_cache_config(), 4_096).unwrap();
    assert_eq!(plan.full_attention_slots, 16);
    assert_eq!(plan.linear_attention_slots, 48);
    assert_eq!(plan.buffer_records.len(), 224);
    assert_eq!(plan.base_full_attention_cache_bytes, 536_870_912);
    assert_eq!(plan.base_linear_attention_state_bytes, 313_786_368);
    assert_eq!(plan.total_payload_bytes, 850_657_280);
}

#[test]
fn metal_cache_is_fresh_base_text_only_and_receipt_bound() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let config = config_with_authenticated_mtp();
    let mut prepared = prepare_qwen35_base_text_cache(&config, &device, 16).unwrap();

    validate_fresh_cache(&prepared.cache, &config, &device, &prepared.receipt).unwrap();
    assert!(prepared.cache.mtp_slot.is_none());
    assert!(!prepared.cache.tq_kv_active);
    assert_eq!(prepared.cache.n_seqs, 1);
    assert_eq!(prepared.receipt.receipt_sha256.len(), 64);
    assert_eq!(
        prepared.receipt.actual_payload_bytes,
        prepared.receipt.plan.total_payload_bytes
    );
    for slot in &prepared.cache.linear_attn {
        assert!(slot.capture_states.is_none());
        assert!(slot.conv_capture_states.is_none());
        assert_eq!(slot.pp_flipped, vec![false]);
        assert!(slot
            .conv_state
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0));
        assert!(slot
            .recurrent
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0));
        assert!(slot
            .conv_state_scratch
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0));
        assert!(slot
            .recurrent_scratch
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0));
    }

    prepared.cache.full_attn[0].current_len[0] = 1;
    assert!(validate_fresh_cache(&prepared.cache, &config, &device, &prepared.receipt).is_err());
}
