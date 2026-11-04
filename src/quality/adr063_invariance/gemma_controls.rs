//! Actual endpoint plan/refusal controls. No model/device/Metal construction.
use super::*;
use crate::serve::config::{Gemma4Config,LayerType};
pub(crate) fn cfg() -> Gemma4Config {
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
/// A complete approved packet as JSON (identity/workload valid), optionally a
/// cohort localizer packet with stores over layer 1 keys only.
fn packet(trajectory:&str,localizer:bool)->serde_json::Value {
    let hex=|c:char,n:usize|c.to_string().repeat(n);
    let lanes=(0..4u32).map(|lane|(0..144).map(|i|lane*1000+i).collect::<Vec<_>>()).collect::<Vec<_>>();
    let mut v=serde_json::json!({"execution_approved":true,
        "acquisition":{"composition":{"path":"/x/c.json","sha256":hex('a',64)},"independent_review":{"path":"/x/r.json","sha256":hex('b',64)}},
        "identity":{"family":"Gemma4","source_commit":hex('0',40),"binary_sha256":hex('1',64),"native_version":env!("HF2Q_MLX_NATIVE_VERSION"),
            "native_checksum":hex('2',64),"model_sha256":hex('3',64),"tokenizer_sha256":hex('4',64),"template_sha256":hex('5',64),
            "config_sha256":hex('6',64),"token_vectors_sha256":hex('7',64)},
        "workload":{"family":"Gemma4","vocab_size":262144,"lanes":lanes},"trajectory":trajectory,
        "model":"/x/model.gguf","output":"/x/out","configured_context":262144,"configured_output_allowance":8192,
        "host_payload_limit":2u64*1024*1024*1024,
        "routing_witness":{"payload_bytes":super::super::gemma_routing_witness::PAYLOAD_BYTES,
            "metadata_bytes":super::super::gemma_routing_witness::METADATA_BYTES,
            "cpu_ledger_bytes":super::super::gemma_routing_witness::CPU_LEDGER_BYTES}});
    if localizer {
        v["stage_localizer"]=serde_json::json!({"rerun_id":"cohort-x","layers":[1],"prefill_stages":["Key"],
            "decode_stages":[],"host_payload_limit":4194304});
        v["lanes"]=serde_json::json!([0,1,2,3]);
        v["expected_stores"]=serde_json::json!({"endpoint":[],"stages":[]});
    }
    v
}
#[test]
fn q1_gemma_cohort_localizer_packets_run_only_under_their_own_filters() {
    let parse=|v:serde_json::Value|serde_json::from_value::<Packet>(v).unwrap();
    for t in ["T0ColdSerial","T2ActualCohort"] {
        assert!(preflight(&parse(packet(t,false)),false).is_ok());
        assert!(preflight(&parse(packet(t,true)),true).is_ok());
        // An endpoint filter refuses a localizer packet and vice versa.
        assert!(preflight(&parse(packet(t,true)),false).is_err());
        assert!(preflight(&parse(packet(t,false)),true).is_err());
    }
    // Only the All128 reference and the packed route are localizer routes.
    for t in ["T1Resume64","T3Chunk64","T4ReverseOrder"] {
        assert!(preflight(&parse(packet(t,true)),true).is_err());
    }
    let mut v=packet("T2ActualCohort",true);v["lanes"]=serde_json::json!([0,1,2]);
    assert!(preflight(&parse(v),true).is_err());
    let mut v=packet("T2ActualCohort",true);v["expected_stores"]["extra"]=serde_json::json!([]);
    assert!(preflight(&parse(v),true).is_err());
    let mut v=packet("T2ActualCohort",true);v["routing_witness"]=serde_json::Value::Null;
    assert!(preflight(&parse(v),true).is_err(),"observer control mirrors the witness-ON b3 stores");
    let mut v=packet("T0ColdSerial",true);v.as_object_mut().unwrap().remove("lanes");
    assert!(preflight(&parse(v),true).is_err());
}
#[test]
fn q1_gemma_endpoint_expected_is_the_reviewed_t0_t2_extent() {
    let cfg=cfg();
    for t in [Trajectory::T0ColdSerial,Trajectory::T2ActualCohort] {
        let e=endpoint_expected(&cfg,t).unwrap();
        // The b3 T0/T2 endpoint stores: 5616 records, 788765696 planned bytes.
        assert_eq!(e.len(),5616);
        assert_eq!(payload(&e).unwrap(),788765696);
    }
}
#[test]
fn q1_gemma_warm_kernel_defaults_to_shipped_vec_hybrid_and_refuses_undeclared_tiled() {
    use crate::inference::models::gemma4::GemmaLiveKernel as K;
    assert_eq!(crate::serve::gemma_live_attention::default_warm_live_kernel(),K::VecHybrid);
    assert_eq!(WarmLiveKernel::default(),WarmLiveKernel::VecHybrid);
    assert_eq!(require_warm_kernel(WarmLiveKernel::VecHybrid,K::VecHybrid).unwrap(),WarmLiveKernel::VecHybrid);
    assert!(require_warm_kernel(WarmLiveKernel::VecHybrid,K::TiledF16).is_err());
    assert!(require_warm_kernel(WarmLiveKernel::TiledF16,K::VecHybrid).is_err());
    assert_eq!(require_warm_kernel(WarmLiveKernel::TiledF16,K::TiledF16).unwrap(),WarmLiveKernel::TiledF16);
    assert_eq!(serde_json::to_value(WarmLiveKernel::VecHybrid).unwrap(),serde_json::json!("VecHybrid"));
    assert!(serde_json::from_value::<WarmLiveKernel>(serde_json::json!("Tiled")).is_err());
}
#[test]
fn complete_active_cache_plan_has_every_head_and_distinct_actual_restore64() {
    let cfg=cfg();
    let t0=gemma_cache::expected(&cfg,&[0,1,2,3],&positions(Trajectory::T0ColdSerial)).unwrap();
    // 25 sliding*8 heads +5 global*2 heads; three actual cache roles.
    assert_eq!(t0.len(),4*2*(25*8+5*2)*3);
    assert_eq!(positions(Trajectory::T6CancelRestore),positions(Trajectory::T1Resume64));
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
    let pause=endpoint::pause_unpublished_tail().unwrap();
    assert!(endpoint::head_rows(64,64,None).unwrap().is_none());
    assert!(endpoint::begin_call(0,64,64).is_err());
    assert!(endpoint::pause_unpublished_tail().is_err());
    drop(pause);
    endpoint::begin_call(0,0,64).unwrap();
    assert_eq!(endpoint::head_rows(0,64,None).unwrap().unwrap(),(0..64).collect::<Vec<_>>());
    assert!(endpoint::head_rows(0,64,Some(vec![(0,0,128,0),(1,0,128,128),(2,0,128,256),(3,0,128,384)])).is_err());
    assert!(endpoint::head_rows(64,64,None).is_err());
    assert!(endpoint::begin_call(1,0,64).is_err());
    assert!(endpoint::end_call().is_err());
    assert!(endpoint::pause_unpublished_tail().is_err());
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

#[test]
fn actual_restore64_hash_gate_rejects_cursor_only_or_changed_bytes() {
    let source=Tensor {at:gemma_cache::at(0,64,0,Stage::CacheValuePacked,0,false),
        scalar:Scalar::U8,shape:vec![1,64,256],logical_bytes:16384,parent_identity:1,
        parent_bytes:16384,parent_offset:0,submission_id:None,checked_drain:"HOST_CONTROL".into(),
        file:"source.bin".into(),sha256:"a".repeat(64),pages:vec![]};
    let mut restored=source.clone();restored.at.segment+=1000;restored.file="restored.bin".into();
    assert_eq!(gemma_cache::require_restore64_exact(&[source.clone(),restored.clone()]).unwrap(),16384);
    restored.sha256="b".repeat(64);
    assert!(gemma_cache::require_restore64_exact(&[source.clone(),restored]).is_err());
    assert!(gemma_cache::require_restore64_exact(&[source]).is_err());
    assert!(gemma_cache::require_restore64_exact(&[]).is_err());
}
