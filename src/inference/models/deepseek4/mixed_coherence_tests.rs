//! ADR-049 B.0 model-free proof for cooperative prefill in a Mixed step.

use mlx_native::{MlxBuffer, MlxDevice};

use super::cache::{Deepseek4Cache, Deepseek4CachePlan};
use super::decode_cohort::publish_verifier_cohort_after_gate;
use super::verifier_forward::publish_prefill_cohort_after_gate;
use super::Deepseek4Config;

#[derive(Clone, Debug, Eq, PartialEq)]
struct CacheByteIdentity {
    position: usize,
    poisoned: bool,
    layers: Vec<LayerByteIdentity>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LayerByteIdentity {
    attention_kv: Vec<u16>,
    indexer_kv: Option<Vec<u16>>,
    main_kv_state: Option<Vec<u32>>,
    main_score_state: Option<Vec<u32>>,
    indexer_kv_state: Option<Vec<u32>>,
    indexer_score_state: Option<Vec<u32>>,
}

fn config() -> Deepseek4Config {
    Deepseek4Config {
        num_hidden_layers: 2,
        hidden_size: 4096,
        hidden_size_out: 16384,
        max_position_embeddings: 256,
        vocab_size: 129280,
        num_attention_heads: 64,
        num_key_value_heads: 1,
        head_dim: 8,
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
        compress_ratios: vec![4, 128],
        compress_rope_theta: 160000.0,
        index_num_heads: 64,
        index_head_dim: 4,
        index_top_k: 512,
        rms_norm_eps: 1e-6,
        num_experts: 256,
        num_experts_per_tok: 6,
        num_shared_experts: 1,
        expert_intermediate_size: 2048,
        route_scale: 1.5,
        normalize_topk: true,
        swiglu_clamp_experts: vec![10.0; 2],
        swiglu_clamp_shared: vec![10.0; 2],
        hyper_connection_count: 4,
        hyper_connection_sinkhorn_iterations: 20,
        hyper_connection_epsilon: 1e-6,
        hash_layer_count: 2,
    }
}

fn f32_buffer_bits(buffer: Option<&MlxBuffer>) -> Option<Vec<u32>> {
    buffer.map(|buffer| {
        buffer
            .as_slice::<f32>()
            .expect("read DeepSeek-V4 F32 cache state")
            .iter()
            .map(|value| value.to_bits())
            .collect()
    })
}

fn cache_byte_identity(cache: &Deepseek4Cache) -> CacheByteIdentity {
    CacheByteIdentity {
        position: cache.position(),
        poisoned: cache.is_poisoned(),
        layers: cache
            .layers()
            .iter()
            .map(|layer| LayerByteIdentity {
                attention_kv: layer
                    .attention_kv
                    .as_slice::<u16>()
                    .expect("read DeepSeek-V4 attention/cache bytes")
                    .to_vec(),
                indexer_kv: layer.indexer_kv.as_ref().map(|buffer| {
                    buffer
                        .as_slice::<u16>()
                        .expect("read DeepSeek-V4 indexer cache bytes")
                        .to_vec()
                }),
                main_kv_state: f32_buffer_bits(layer.main_kv_state.as_ref()),
                main_score_state: f32_buffer_bits(layer.main_score_state.as_ref()),
                indexer_kv_state: f32_buffer_bits(layer.indexer_kv_state.as_ref()),
                indexer_score_state: f32_buffer_bits(layer.indexer_score_state.as_ref()),
            })
            .collect(),
    }
}

fn seed_cache_bytes(cache: &mut Deepseek4Cache, seed: u16) {
    for (layer_index, layer) in cache.layers_mut().iter_mut().enumerate() {
        for (index, value) in layer
            .attention_kv
            .as_mut_slice::<u16>()
            .expect("seed DeepSeek-V4 attention/cache bytes")
            .iter_mut()
            .enumerate()
        {
            *value = seed
                .wrapping_add((layer_index as u16).wrapping_mul(257))
                .wrapping_add(index as u16);
        }
        if let Some(buffer) = layer.indexer_kv.as_mut() {
            for (index, value) in buffer
                .as_mut_slice::<u16>()
                .expect("seed DeepSeek-V4 indexer cache bytes")
                .iter_mut()
                .enumerate()
            {
                *value = seed
                    .wrapping_add(0x4000)
                    .wrapping_add((layer_index as u16).wrapping_mul(257))
                    .wrapping_add(index as u16);
            }
        }
        for (state_index, buffer) in [
            layer.main_kv_state.as_mut(),
            layer.main_score_state.as_mut(),
            layer.indexer_kv_state.as_mut(),
            layer.indexer_score_state.as_mut(),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(buffer) = buffer {
                for (index, value) in buffer
                    .as_mut_slice::<f32>()
                    .expect("seed DeepSeek-V4 compressor accumulator")
                    .iter_mut()
                    .enumerate()
                {
                    *value = f32::from(
                        seed.wrapping_add((layer_index as u16).wrapping_mul(31))
                            .wrapping_add((state_index as u16).wrapping_mul(7))
                            .wrapping_add(index as u16),
                    ) / 16.0;
                }
            }
        }
    }
}

fn mixed_publication_caches() -> Vec<Deepseek4Cache> {
    let plan = Deepseek4CachePlan::for_context(&config(), 256).unwrap();
    let device = MlxDevice::new().unwrap();
    (0..6)
        .map(|lane| {
            let mut cache = Deepseek4Cache::allocate(&plan, device.clone()).unwrap();
            cache.commit_prefill(0, 128).unwrap();
            seed_cache_bytes(&mut cache, 0x0100_u16.wrapping_mul(lane + 1));
            cache
        })
        .collect()
}

fn publish_decode_step(caches: &mut [Deepseek4Cache]) {
    let [_, _, decode0, decode1, decode2, decode3] = caches else {
        unreachable!("six-lane Mixed publication fixture")
    };
    let mut decode = [decode0, decode1, decode2, decode3];
    publish_verifier_cohort_after_gate(&mut decode, [128; 4], || Ok(())).unwrap();
}

#[test]
fn mixed_prefill_commit_preserves_decode_lane_cursor_and_cache_bytes() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    // Lanes 0-1 are an aligned cooperative prefill; lanes 2-5 are the
    // concurrently runnable B=4 decoder. Mirror the Mixed-step publication
    // order: decode first, then cooperative prefill.
    let mut caches = mixed_publication_caches();
    publish_decode_step(&mut caches);
    let after_decode = caches[2..]
        .iter()
        .map(cache_byte_identity)
        .collect::<Vec<_>>();
    assert!(after_decode
        .iter()
        .all(|identity| identity.position == 129 && !identity.poisoned));

    let spans = caches[..2]
        .iter()
        .map(|cache| cache.plan_prefill(128).unwrap())
        .collect::<Vec<_>>();
    let [prefill0, prefill1, _, _, _, _] = caches.as_mut_slice() else {
        unreachable!("six-lane Mixed publication fixture")
    };
    publish_prefill_cohort_after_gate(&mut [prefill0, prefill1], &spans, || Ok(())).unwrap();

    assert!(caches[..2]
        .iter()
        .all(|cache| cache.position() == 256 && !cache.is_poisoned()));
    assert_eq!(
        caches[2..]
            .iter()
            .map(cache_byte_identity)
            .collect::<Vec<_>>(),
        after_decode,
        "cooperative prefill commit changed a decode peer's cursor, KV bytes, or compressor accumulators"
    );
}

#[test]
fn mixed_prefill_poison_preserves_decode_lane_cursor_and_cache_bytes() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut caches = mixed_publication_caches();
    publish_decode_step(&mut caches);
    let after_decode = caches[2..]
        .iter()
        .map(cache_byte_identity)
        .collect::<Vec<_>>();

    let spans = caches[..2]
        .iter()
        .map(|cache| cache.plan_prefill(128).unwrap())
        .collect::<Vec<_>>();
    let [prefill0, prefill1, _, _, _, _] = caches.as_mut_slice() else {
        unreachable!("six-lane Mixed publication fixture")
    };
    let error = publish_prefill_cohort_after_gate(&mut [prefill0, prefill1], &spans, || {
        anyhow::bail!("synthetic Mixed cooperative-prefill supervisor rejection")
    })
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("rejected before cache publication"));

    assert!(caches[..2]
        .iter()
        .all(|cache| cache.position() == 128 && cache.is_poisoned()));
    assert_eq!(
        caches[2..]
            .iter()
            .map(cache_byte_identity)
            .collect::<Vec<_>>(),
        after_decode,
        "cooperative prefill poison changed a decode peer's cursor, KV bytes, or compressor accumulators"
    );
}
