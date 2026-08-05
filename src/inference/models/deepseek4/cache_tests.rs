use mlx_native::{DType, MlxDevice};

use super::cache::{CacheError, CacheKind, Deepseek4Cache, Deepseek4CachePlan};
use super::Deepseek4Config;

fn config(ratios: Vec<u32>) -> Deepseek4Config {
    let layers = ratios.len() as u32;
    Deepseek4Config {
        num_hidden_layers: layers,
        hidden_size: 4096,
        hidden_size_out: 16384,
        max_position_embeddings: 1_048_576,
        vocab_size: 129280,
        num_attention_heads: 64,
        num_key_value_heads: 1,
        head_dim: 512,
        rope_head_dim: 64,
        rope_theta: 10000.0,
        rope_factor: 16.0,
        original_context_length: 65536,
        yarn_beta_fast: 32.0,
        yarn_beta_slow: 1.0,
        q_lora_rank: 1024,
        o_lora_rank: 1024,
        output_groups: 8,
        sliding_window: 128,
        compress_ratios: ratios,
        compress_rope_theta: 160000.0,
        index_num_heads: 64,
        index_head_dim: 128,
        index_top_k: 512,
        rms_norm_eps: 1e-6,
        num_experts: 256,
        num_experts_per_tok: 6,
        num_shared_experts: 1,
        expert_intermediate_size: 2048,
        route_scale: 1.5,
        normalize_topk: true,
        swiglu_clamp_experts: vec![10.0; layers as usize],
        swiglu_clamp_shared: vec![10.0; layers as usize],
        hyper_connection_count: 4,
        hyper_connection_sinkhorn_iterations: 20,
        hyper_connection_epsilon: 1e-6,
        hash_layer_count: layers.min(3),
    }
}

fn official_config() -> Deepseek4Config {
    config(
        (0..43)
            .map(|layer| {
                if layer < 2 {
                    0
                } else if layer % 2 == 0 {
                    4
                } else {
                    128
                }
            })
            .collect(),
    )
}

#[test]
fn official_one_million_context_plan_has_exact_shapes_and_bytes() {
    let plan = Deepseek4CachePlan::for_context(&official_config(), 1_048_576).unwrap();
    assert_eq!(plan.layers.len(), 43);
    assert_eq!(plan.resident_bytes, 7_232_045_056);

    assert_eq!(plan.layers[0].attention_kv.shape, vec![128, 512]);
    assert_eq!(plan.layers[0].window_kv.shape, vec![128, 512]);
    assert!(plan.layers[0].compressed_kv.is_none());
    assert!(plan.layers[0].indexer_kv.is_none());

    let ratio_four = &plan.layers[2];
    assert_eq!(ratio_four.compress_ratio, 4);
    assert_eq!(
        ratio_four.compressed_kv.as_ref().unwrap().shape,
        vec![262_144, 512]
    );
    assert_eq!(ratio_four.attention_kv.shape, vec![262_272, 512]);
    assert_eq!(
        ratio_four.indexer_kv.as_ref().unwrap().shape,
        vec![262_144, 128]
    );
    assert_eq!(
        ratio_four.main_kv_state.as_ref().unwrap().shape,
        vec![1, 8, 1024]
    );
    assert_eq!(
        ratio_four.main_score_state.as_ref().unwrap().shape,
        vec![1, 8, 1024]
    );
    assert_eq!(
        ratio_four.indexer_kv_state.as_ref().unwrap().shape,
        vec![1, 8, 256]
    );
    assert_eq!(
        ratio_four.indexer_score_state.as_ref().unwrap().shape,
        vec![1, 8, 256]
    );
    assert_eq!(ratio_four.main_kv_state.as_ref().unwrap().dtype, DType::F32);

    let ratio_128 = &plan.layers[3];
    assert_eq!(ratio_128.compress_ratio, 128);
    assert_eq!(
        ratio_128.compressed_kv.as_ref().unwrap().shape,
        vec![8192, 512]
    );
    assert_eq!(ratio_128.attention_kv.shape, vec![8320, 512]);
    assert_eq!(
        ratio_128.main_kv_state.as_ref().unwrap().shape,
        vec![1, 128, 512]
    );
    assert!(ratio_128.indexer_kv.is_none());
    assert_eq!(
        plan.layers
            .iter()
            .map(|layer| layer.resident_bytes)
            .sum::<u64>(),
        plan.resident_bytes
    );
}

#[test]
fn malformed_context_schedule_and_overflow_fail_closed() {
    let cfg = official_config();
    assert!(matches!(
        Deepseek4CachePlan::for_context(&cfg, 0),
        Err(CacheError::EmptyContext)
    ));
    assert!(matches!(
        Deepseek4CachePlan::for_context(&cfg, 1_048_577),
        Err(CacheError::ContextBound { .. })
    ));

    let mut malformed = cfg.clone();
    malformed.sliding_window = 127;
    assert!(matches!(
        Deepseek4CachePlan::for_context(&malformed, 1024),
        Err(CacheError::SlidingWindow { actual: 127 })
    ));
    malformed = cfg.clone();
    malformed.compress_ratios.pop();
    assert!(matches!(
        Deepseek4CachePlan::for_context(&malformed, 1024),
        Err(CacheError::LayerCount { .. })
    ));
    malformed = cfg.clone();
    malformed.compress_ratios[0] = 8;
    assert!(matches!(
        Deepseek4CachePlan::for_context(&malformed, 1024),
        Err(CacheError::CompressionRatio { layer: 0, ratio: 8 })
    ));

    let mut overflowing = config(vec![4, 4]);
    overflowing.max_position_embeddings = u32::MAX;
    overflowing.head_dim = u32::MAX;
    assert!(matches!(
        Deepseek4CachePlan::for_context(&overflowing, u32::MAX as usize),
        Err(CacheError::ByteOverflow {
            layer: 1,
            kind: CacheKind::AttentionKv
        })
    ));
}

#[test]
fn allocator_materializes_the_plan_as_zeroed_bf16_buffers() {
    let mut cfg = config(vec![0, 4, 128]);
    cfg.max_position_embeddings = 128;
    cfg.head_dim = 32;
    cfg.index_head_dim = 16;
    let plan = Deepseek4CachePlan::for_context(&cfg, 128).unwrap();
    assert_eq!(plan.resident_bytes, 66_624);

    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let cache = Deepseek4Cache::allocate(&plan, MlxDevice::new().unwrap()).unwrap();
    assert_eq!(cache.resident_bytes(), plan.resident_bytes);
    assert_eq!(cache.layers().len(), 3);
    assert_eq!(cache.layers()[1].attention_kv.dtype(), DType::BF16);
    assert_eq!(cache.layers()[1].attention_kv.shape(), &[160, 32]);
    assert_eq!(cache.layers()[1].window_kv.dtype(), DType::BF16);
    assert_eq!(cache.layers()[1].window_kv.shape(), &[128, 32]);
    assert_eq!(
        cache.layers()[1].compressed_kv.as_ref().unwrap().shape(),
        &[32, 32]
    );
    assert_eq!(
        cache.layers()[1].indexer_kv.as_ref().unwrap().shape(),
        &[32, 16]
    );
    assert_eq!(
        cache.layers()[1]
            .compressed_kv
            .as_ref()
            .unwrap()
            .byte_offset(),
        (128 * 32 * DType::BF16.size_of()) as u64
    );
    assert_eq!(
        cache.layers()[1].main_kv_state.as_ref().unwrap().shape(),
        &[1, 8, 64]
    );
    assert_eq!(
        cache.layers()[1]
            .indexer_score_state
            .as_ref()
            .unwrap()
            .shape(),
        &[1, 8, 32]
    );
    assert!(cache.layers()[1]
        .indexer_kv
        .as_ref()
        .unwrap()
        .as_slice::<u16>()
        .unwrap()
        .iter()
        .all(|value| *value == 0));
    assert!(cache.layers()[1]
        .main_kv_state
        .as_ref()
        .unwrap()
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .all(|value| *value == 0.0));
    assert!(cache.layers()[1]
        .main_score_state
        .as_ref()
        .unwrap()
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .all(|value| *value == f32::NEG_INFINITY));
}

#[test]
fn cache_steps_publish_only_complete_groups_and_commit_transactionally() {
    let cfg = config(vec![4, 128]);
    let plan = Deepseek4CachePlan::for_context(&cfg, 128).unwrap();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut cache = Deepseek4Cache::allocate(&plan, MlxDevice::new().unwrap()).unwrap();

    let first = cache.plan_next_step().unwrap();
    assert_eq!(first.position, 0);
    assert_eq!(first.layers[0].window_write_slot, 0);
    assert_eq!(first.layers[0].compressed_write_slot, None);
    assert_eq!(first.layers[0].compressed_valid_after, 0);
    assert!(matches!(
        cache.commit_step(1),
        Err(CacheError::StepOutOfOrder {
            expected: 0,
            actual: 1
        })
    ));
    cache.commit_step(first.position).unwrap();

    for expected in 1..4 {
        let step = cache.plan_next_step().unwrap();
        assert_eq!(step.position, expected);
        cache.commit_step(step.position).unwrap();
    }
    let after_boundary = cache.plan_next_step().unwrap();
    assert_eq!(after_boundary.position, 4);
    assert_eq!(after_boundary.layers[0].compressed_write_slot, None);
    assert_eq!(after_boundary.layers[0].compressed_valid_after, 1);

    cache.reset().unwrap();
    for expected in 0..128 {
        let step = cache.plan_next_step().unwrap();
        assert_eq!(step.position, expected);
        if expected == 3 {
            assert_eq!(step.layers[0].compressed_write_slot, Some(0));
            assert_eq!(step.layers[0].indexer_write_slot, Some(0));
        }
        if expected == 127 {
            assert_eq!(step.layers[0].compressed_write_slot, Some(31));
            assert_eq!(step.layers[1].compressed_write_slot, Some(0));
            assert_eq!(step.layers[1].indexer_write_slot, None);
        }
        cache.commit_step(step.position).unwrap();
    }
    assert!(matches!(
        cache.plan_next_step(),
        Err(CacheError::ContextExhausted { maximum: 128 })
    ));
}

#[test]
fn partial_token_poison_requires_reset_before_replay() {
    let cfg = config(vec![4, 128]);
    let plan = Deepseek4CachePlan::for_context(&cfg, 128).unwrap();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut cache = Deepseek4Cache::allocate(&plan, MlxDevice::new().unwrap()).unwrap();
    cache.poison();
    assert!(cache.is_poisoned());
    assert!(matches!(cache.plan_next_step(), Err(CacheError::Poisoned)));
    assert!(matches!(cache.commit_step(0), Err(CacheError::Poisoned)));
    cache.reset().unwrap();
    assert!(!cache.is_poisoned());
    assert_eq!(cache.plan_next_step().unwrap().position, 0);
}

#[test]
fn start_zero_prefill_span_counts_complete_groups_and_publishes_once() {
    let cfg = config(vec![4, 128]);
    let plan = Deepseek4CachePlan::for_context(&cfg, 128).unwrap();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut cache = Deepseek4Cache::allocate(&plan, MlxDevice::new().unwrap()).unwrap();

    for (tokens, ratio4, ratio128) in [(1, 0, 0), (3, 0, 0), (4, 1, 0), (127, 31, 0), (128, 32, 1)]
    {
        let span = cache.plan_prefill_start0(tokens).unwrap();
        assert_eq!(span.start_position, 0);
        assert_eq!(span.token_count, tokens);
        assert_eq!(span.layers[0].window_valid_after, tokens);
        assert_eq!(span.layers[0].compressed_count, ratio4);
        assert_eq!(span.layers[0].compressed_valid_after, ratio4);
        assert_eq!(span.layers[0].indexer_count, ratio4);
        assert_eq!(span.layers[0].indexer_valid_after, ratio4);
        assert_eq!(span.layers[1].compressed_count, ratio128);
        assert_eq!(span.layers[1].compressed_valid_after, ratio128);
        assert_eq!(span.layers[1].indexer_count, 0);
    }

    assert!(matches!(
        cache.plan_prefill_start0(0),
        Err(CacheError::EmptyPrefill)
    ));
    assert!(matches!(
        cache.plan_prefill_start0(129),
        Err(CacheError::ContextBound {
            requested: 129,
            maximum: 128
        })
    ));
    assert!(matches!(
        cache.commit_prefill(1, 4),
        Err(CacheError::StepOutOfOrder {
            expected: 0,
            actual: 1
        })
    ));
    cache.commit_prefill(0, 4).unwrap();
    assert_eq!(cache.position(), 4);
    assert!(matches!(
        cache.plan_prefill_start0(4),
        Err(CacheError::PrefillNotEmpty { position: 4 })
    ));
    cache.poison();
    assert!(matches!(
        cache.commit_prefill(4, 1),
        Err(CacheError::Poisoned)
    ));
    cache.reset().unwrap();
    assert_eq!(cache.position(), 0);
}

#[test]
fn bounded_prefill_chunks_append_across_window_and_compression_boundaries() {
    let cfg = config(vec![4, 128]);
    let plan = Deepseek4CachePlan::for_context(&cfg, 512).unwrap();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut cache = Deepseek4Cache::allocate(&plan, MlxDevice::new().unwrap()).unwrap();

    let first = cache.plan_prefill(127).unwrap();
    assert_eq!(first.start_position, 0);
    assert_eq!(first.layers[0].compressed_write_start, 0);
    assert_eq!(first.layers[0].compressed_count, 31);
    assert_eq!(first.layers[0].compressed_valid_after, 31);
    cache.commit_prefill(0, 127).unwrap();

    let boundary = cache.plan_prefill(2).unwrap();
    assert_eq!(boundary.start_position, 127);
    assert_eq!(boundary.layers[0].window_write_start, 127);
    assert_eq!(boundary.layers[0].window_valid_after, 128);
    assert_eq!(boundary.layers[0].compressed_write_start, 31);
    assert_eq!(boundary.layers[0].compressed_count, 1);
    assert_eq!(boundary.layers[0].compressed_valid_after, 32);
    assert_eq!(boundary.layers[1].compressed_write_start, 0);
    assert_eq!(boundary.layers[1].compressed_count, 1);
    assert_eq!(boundary.layers[1].compressed_valid_after, 1);
    cache.commit_prefill(127, 2).unwrap();

    let next = cache.plan_prefill(128).unwrap();
    assert_eq!(next.start_position, 129);
    assert_eq!(next.layers[0].window_write_start, 1);
    assert_eq!(next.layers[0].compressed_write_start, 32);
    assert_eq!(next.layers[0].compressed_count, 32);
    assert_eq!(next.layers[0].compressed_valid_after, 64);
    assert_eq!(next.layers[1].compressed_count, 1);
    assert_eq!(next.layers[1].compressed_valid_after, 2);

    assert!(matches!(
        cache.plan_prefill(129),
        Err(CacheError::PrefillWindow {
            requested: 129,
            maximum: 128
        })
    ));
}

#[test]
fn snapshot_restore_recovers_kv_recurrent_state_and_position_without_aliasing() {
    let mut cfg = config(vec![0, 4, 128]);
    cfg.max_position_embeddings = 128;
    cfg.head_dim = 8;
    cfg.index_head_dim = 4;
    let plan = Deepseek4CachePlan::for_context(&cfg, 128).unwrap();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut cache = Deepseek4Cache::allocate(&plan, MlxDevice::new().unwrap()).unwrap();

    cache.layers_mut()[1]
        .attention_kv
        .as_mut_slice::<u16>()
        .unwrap()[3] = 0x1234;
    cache.layers_mut()[1]
        .indexer_kv
        .as_mut()
        .unwrap()
        .as_mut_slice::<u16>()
        .unwrap()[2] = 0x5678;
    cache.layers_mut()[1]
        .main_kv_state
        .as_mut()
        .unwrap()
        .as_mut_slice::<f32>()
        .unwrap()[1] = 9.25;
    for expected in 0..7 {
        let step = cache.plan_next_step().unwrap();
        assert_eq!(step.position, expected);
        cache.commit_step(step.position).unwrap();
    }

    let snapshot = cache.snapshot().unwrap();
    assert_eq!(snapshot.position(), 7);
    assert_eq!(snapshot.resident_bytes(), cache.resident_bytes());

    cache.layers_mut()[1]
        .attention_kv
        .as_mut_slice::<u16>()
        .unwrap()[3] = 0;
    cache.layers_mut()[1]
        .indexer_kv
        .as_mut()
        .unwrap()
        .as_mut_slice::<u16>()
        .unwrap()[2] = 0;
    cache.layers_mut()[1]
        .main_kv_state
        .as_mut()
        .unwrap()
        .as_mut_slice::<f32>()
        .unwrap()[1] = -1.0;
    cache.commit_step(7).unwrap();

    cache.restore(&snapshot).unwrap();
    assert_eq!(cache.position(), 7);
    assert_eq!(
        cache.layers()[1]
            .attention_kv
            .as_slice::<u16>()
            .unwrap()[3],
        0x1234
    );
    assert_eq!(
        cache.layers()[1]
            .indexer_kv
            .as_ref()
            .unwrap()
            .as_slice::<u16>()
            .unwrap()[2],
        0x5678
    );
    assert_eq!(
        cache.layers()[1]
            .main_kv_state
            .as_ref()
            .unwrap()
            .as_slice::<f32>()
            .unwrap()[1],
        9.25
    );

    // Mutating the live cache after restore must not mutate the snapshot.
    cache.layers_mut()[1]
        .attention_kv
        .as_mut_slice::<u16>()
        .unwrap()[3] = 0xabcd;
    cache.restore(&snapshot).unwrap();
    assert_eq!(
        cache.layers()[1]
            .attention_kv
            .as_slice::<u16>()
            .unwrap()[3],
        0x1234
    );
}

#[test]
fn restore_rejects_a_snapshot_from_a_different_cache_plan() {
    let mut cfg = config(vec![4]);
    cfg.max_position_embeddings = 128;
    cfg.head_dim = 8;
    cfg.index_head_dim = 4;
    let short = Deepseek4CachePlan::for_context(&cfg, 64).unwrap();
    let long = Deepseek4CachePlan::for_context(&cfg, 128).unwrap();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let short_cache = Deepseek4Cache::allocate(&short, MlxDevice::new().unwrap()).unwrap();
    let snapshot = short_cache.snapshot().unwrap();
    let mut long_cache = Deepseek4Cache::allocate(&long, MlxDevice::new().unwrap()).unwrap();

    assert!(matches!(
        long_cache.restore(&snapshot),
        Err(CacheError::SnapshotPlanMismatch)
    ));
}
