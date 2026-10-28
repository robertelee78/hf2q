//! Actual endpoint plan/refusal controls. No model/device/Metal construction.
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
fn ordinary_axis_refusal_is_before_model_acquisition() {
    for t in Trajectory::ALL {
        let admitted=matches!(t,Trajectory::T0ColdSerial|Trajectory::T1Resume64|
            Trajectory::T3Chunk64|Trajectory::T4ReverseOrder|Trajectory::T7RepeatCold|Trajectory::T8RepeatCold);
        assert_eq!(require_ordinary(t).is_ok(),admitted);
    }
}
#[test]
fn complete_active_cache_plan_has_every_head_and_distinct_actual_restore64() {
    let cfg=cfg();
    let t0=gemma_cache::expected(&cfg,&[0,1,2,3],&positions(Trajectory::T0ColdSerial)).unwrap();
    // 25 sliding*8 heads +5 global*2 heads; three actual cache roles.
    assert_eq!(t0.len(),4*2*(25*8+5*2)*3);
    let t1=gemma_cache::expected(&cfg,&[0,1,2,3],&positions(Trajectory::T1Resume64)).unwrap();
    assert_eq!(t1.len(),t0.len()*2);
    let coordinates=t1.iter().map(|e|e.at.clone()).collect::<std::collections::BTreeSet<_>>();
    assert_eq!(coordinates.len(),t1.len());
    assert!(t0.iter().all(|e| [128,144].contains(&e.at.logical_step)));
    assert_eq!(t1.iter().filter(|e|e.at.logical_step==64 && e.at.segment>=1000).count(),4*210*3);
    let mut bad=cfg.clone();bad.sliding_window=128;
    assert!(gemma_cache::expected(&bad,&[0],&[(144,false)]).is_err());
    assert!(gemma_cache::expected(&cfg,&[4],&[(128,false)]).is_err());
    assert!(gemma_cache::expected(&cfg,&[0],&[(128,true)]).is_err());
}
#[test]
fn endpoint_scope_cannot_relabel_packed_graph_or_omit_rows() {
    use std::sync::atomic::{AtomicU64,Ordering};
    static NEXT:AtomicU64=AtomicU64::new(0);
    assert!(endpoint::head_rows(0,64,None).unwrap().is_none());
    let root=std::env::temp_dir().canonicalize().unwrap().join(format!("q1-gemma-head-{}-{}",
        std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
    let identity=Identity {family:Family::Gemma4,source_commit:"a".repeat(40),binary_sha256:"b".repeat(64),
        native_version:env!("HF2Q_MLX_NATIVE_VERSION").into(),native_checksum:"c".repeat(64),
        model_sha256:"d".repeat(64),tokenizer_sha256:"e".repeat(64),template_sha256:"f".repeat(64),
        config_sha256:"1".repeat(64),token_vectors_sha256:"2".repeat(64)};
    let expected=vec![ExpectedTensor {at:gemma_cache::at(0,64,0,Stage::CacheKey,0,false),
        scalar:Scalar::F16,shape:vec![1,64,256]}];
    let store=Store::new(&root,identity,"HOST_ONLY_CONTROL".into(),expected,32768,1).unwrap();
    let scope=endpoint::install(store).unwrap();
    endpoint::begin_call(0,0,64).unwrap();
    assert_eq!(endpoint::head_rows(0,64,None).unwrap().unwrap(),(0..64).collect::<Vec<_>>());
    assert!(endpoint::head_rows(0,64,Some(vec![(0,0,128,0),(1,0,128,128),(2,0,128,256),(3,0,128,384)])).is_err());
    assert!(endpoint::head_rows(64,64,None).is_err());
    assert!(endpoint::begin_call(1,0,64).is_err());
    assert!(endpoint::end_call().is_err());
    drop(scope);
    assert!(endpoint::head_rows(0,64,None).unwrap().is_none());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn packed_head_owner_mapping_uses_actual_offsets_not_width_as_membership() {
    let layout=vec![(0,0,128,0),(1,0,128,128),(2,0,128,256),(3,0,128,384)];
    let owners=endpoint::packed_owners(&layout).unwrap();
    assert_eq!(owners.len(),512);
    assert_eq!(owners[127],(0,127));assert_eq!(owners[128],(1,0));
    assert_eq!(owners[511],(3,127));
    for (entry,value) in [(1,(1,0,128,127)),(2,(1,0,128,256)),(0,(0,1,128,0)),(3,(3,0,127,384))] {
        let mut bad=layout.clone();bad[entry]=value;
        assert!(endpoint::packed_owners(&bad).is_err());
    }
    // One temporal512-row request is not four independent128-row inputs.
    assert!(endpoint::packed_owners(&[(0,0,512,0)]).is_err());
}
