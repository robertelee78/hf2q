use super::*;

#[test]
fn gemma_activation_argmax_executes_native_u32_parameter_contract() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = crate::serve::gpu::GpuContext::new().expect("Metal test requires a GPU");
    let device = ctx.device().clone();
    let config = Gemma4Config {
        vocab_size: 8,
        hidden_size: 4,
        intermediate_size: 8,
        moe_intermediate_size: 4,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        num_global_key_value_heads: 1,
        head_dim: 4,
        global_head_dim: 4,
        rms_norm_eps: 1e-6,
        rope_theta_sliding: 10000.0,
        rope_theta_global: 10000.0,
        sliding_window: 8,
        max_position_embeddings: 8,
        final_logit_softcapping: None,
        attention_bias: false,
        attention_k_eq_v: false,
        tie_word_embeddings: true,
        num_experts: 2,
        top_k_experts: 1,
        layer_types: vec![crate::serve::config::LayerType::Sliding],
    };
    let mut buffers = alloc_activation_buffers(&device, &config).unwrap();
    buffers.argmax_params.as_mut_slice::<u32>().unwrap()[0] = config.vocab_size as u32;
    for (logits, expected_index, expected_value) in [
        ([-8.0, -3.0, -9.0, -4.0, -5.0, -6.0, -7.0, -1.0], 7, -1.0),
        ([0.0, 3.0, 3.0, 1.0, -5.0, -6.0, -7.0, -8.0], 1, 3.0),
    ] {
        buffers
            .logits
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&logits);
        let (executor, registry) = ctx.split();
        let mut session = executor.begin().unwrap();
        mlx_native::ops::argmax::dispatch_argmax_f32(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            &buffers.logits,
            &buffers.argmax_index,
            &buffers.argmax_value,
            &buffers.argmax_params,
            config.vocab_size as u32,
        )
        .expect("production Gemma buffers must satisfy native argmax ABI");
        session.finish().unwrap();
        assert_eq!(
            buffers.argmax_index.as_slice::<u32>().unwrap()[0],
            expected_index
        );
        assert_eq!(
            buffers.argmax_value.as_slice::<f32>().unwrap()[0],
            expected_value
        );
    }
}
