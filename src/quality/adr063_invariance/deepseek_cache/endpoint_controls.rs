//! Source-plan active extent controls, no device/model construction.
use super::*;
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

#[test]
fn ds_complete_cache_plan_distinguishes_ratio4_ratio128_and_ring_wrap() {
    let cfg=config(vec![0,4,128]);
    let e=expected(&cfg,262144,&[(64,false),(64,true),(128,false),(144,false)]).unwrap();
    assert_eq!(e.len(),184);
    let one=|layer,stage,pos|e.iter().find(|t|t.at.agent=="agent-0" &&t.at.layer==layer
        &&t.at.stage==stage &&t.at.logical_step==pos &&t.at.segment==0).unwrap();
    assert_eq!(one(1,Stage::CacheCompressed,64).shape,vec![16,512]);
    assert_eq!(one(1,Stage::CacheIndexer,144).shape,vec![36,128]);
    assert_eq!(one(1,Stage::CacheMainKv,64).shape,vec![1,8,1024]);
    assert_eq!(one(1,Stage::CacheIndexerScore,64).shape,vec![1,8,256]);
    assert_eq!(one(2,Stage::CacheMainScore,64).shape,vec![1,128,512]);
    assert_eq!(one(2,Stage::CacheCompressed,128).shape,vec![1,512]);
    assert_eq!(one(2,Stage::CacheWindow,144).shape,vec![128,512]);
    assert!(!e.iter().any(|t|t.at.layer==2 &&t.at.stage==Stage::CacheCompressed &&t.at.logical_step==64));
    assert!(!e.iter().any(|t|t.at.layer==0 &&t.at.stage!=Stage::CacheWindow));
    assert!(expected(&cfg,262144,&[(0,false)]).is_err());
    assert!(expected(&cfg,262144,&[(128,true)]).is_err());
    assert!(expected(&config(vec![8]),262144,&[(64,false)]).is_err());
}
#[test]
fn ds_exact64_requires_full_bytes_not_only_cursor() {
    let source=Tensor {at:at(0,64,1,Stage::CacheMainScore,false),scalar:Scalar::F32,
        shape:vec![1,8,1024],logical_bytes:32768,parent_identity:1,parent_bytes:32768,parent_offset:0,
        submission_id:None,checked_drain:"HOST_CONTROL".into(),file:"a.bin".into(),sha256:"a".repeat(64),pages:vec![]};
    let mut restored=source.clone();restored.at.segment=1;restored.file="b.bin".into();
    assert_eq!(require_restore64_exact(&[source.clone(),restored.clone()]).unwrap(),32768);
    restored.sha256="b".repeat(64);assert!(require_restore64_exact(&[source.clone(),restored]).is_err());
    assert!(require_restore64_exact(&[source]).is_err());assert!(require_restore64_exact(&[]).is_err());
}
