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

#[test]
fn actual_runtime_recipe_selectors_export_old_and_routing_labels() {
    let cfg=cfg();let rows=routing_recipe_export(&cfg).unwrap();assert_eq!(rows.len(),4);
    for (i,(adapter,routing)) in [("ordinary",false),("T5",false),("ordinary",true),("T5",true)].into_iter().enumerate() {
        let actual=if adapter=="ordinary" {super::super::gemma_actual::selected_recipe(&cfg,routing)}
            else {super::super::gemma_ngram::configuration_recipe(&cfg,routing)};
        assert_eq!(rows[i]["adapter"],adapter);assert_eq!(rows[i]["routing_witness"],routing);
        assert_eq!(rows[i]["config_sha256"],acquisition::config_digest(&actual).unwrap());
        assert_eq!(rows[i]["recipe"],serde_json::to_value(actual).unwrap());
        assert_eq!(rows[i]["recipe"]["actual_model_config_sha256"],hash(format!("{cfg:?}").as_bytes()));
    }
    assert_eq!(rows[0]["recipe"],serde_json::to_value(super::super::gemma_actual::recipe(&cfg)).unwrap());
    assert_eq!(rows[0]["recipe"]["route"],"gemma-ordinary-full-row-endpoints-v1");
    assert_eq!(rows[1]["recipe"]["route"],"gemma-ngram-injected-one-real-round-then-ordinary-v1");
    assert_eq!(rows[2]["recipe"]["route"],"gemma-ordinary-full-row-endpoints-native-routing1511-v1");
    assert_eq!(rows[3]["recipe"]["route"],"gemma-ngram-injected-native-routing1511-v1");
    for (old,new) in [(0,2),(1,3)] {
        assert_ne!(rows[old]["config_sha256"],rows[new]["config_sha256"]);
        let mut a=rows[old]["recipe"].clone();let mut b=rows[new]["recipe"].clone();
        a.as_object_mut().unwrap().remove("route");b.as_object_mut().unwrap().remove("route");assert_eq!(a,b);
    }
}
#[test]
fn metadata_routing_optin_defaults_false_and_rejects_null_or_unknown() {
    let bound=serde_json::json!({"path":"/private/tmp/x","sha256":"a".repeat(64)});
    let mut value=serde_json::json!({"model":{"path":"/private/tmp/model","retained_full_sha256":MODEL_SHA,
        "file_identity":{"device":1,"inode":2,"bytes":3,"mtime":4,"mtime_ns":5,"ctime":6,"ctime_ns":7}},
        "full_content_receipt":bound,"tokenizer":bound,"template":bound,"producer_metadata":bound,
        "producer_build_receipt":bound,"producer_export_receipt":bound,"producer_config":bound,
        "producer_vectors":[bound,bound,bound,bound],"expected_exporter_source_commit":"b".repeat(40),
        "expected_exporter_binary_sha256":"c".repeat(64),"output":"/private/tmp/new-q1-gemma"});
    assert!(!serde_json::from_value::<Config>(value.clone()).unwrap().routing_recipes);
    value["routing_recipes"]=serde_json::json!(true);assert!(serde_json::from_value::<Config>(value.clone()).unwrap().routing_recipes);
    value["routing_recipes"]=serde_json::Value::Null;assert!(serde_json::from_value::<Config>(value.clone()).is_err());
    value["routing_recipes"]=serde_json::json!(false);value["routing_recipe_fallback"]=serde_json::json!(true);
    assert!(serde_json::from_value::<Config>(value).is_err());
}

fn localizer_config(localizer:Option<serde_json::Value>)->serde_json::Value {
    let bound=serde_json::json!({"path":"/private/tmp/x","sha256":"a".repeat(64)});
    let mut value=serde_json::json!({"model":{"path":"/private/tmp/model","retained_full_sha256":MODEL_SHA,
        "file_identity":{"device":1,"inode":2,"bytes":3,"mtime":4,"mtime_ns":5,"ctime":6,"ctime_ns":7}},
        "full_content_receipt":bound,"tokenizer":bound,"template":bound,"producer_metadata":bound,
        "producer_build_receipt":bound,"producer_export_receipt":bound,"producer_config":bound,
        "producer_vectors":[bound,bound,bound,bound],"expected_exporter_source_commit":"b".repeat(40),
        "expected_exporter_binary_sha256":"c".repeat(64),"output":"/private/tmp/new-q1-gemma"});
    if let Some(l)=localizer {value["localizer"]=l;}
    value
}
fn two_routes(normalize_rows:bool)->serde_json::Value {
    serde_json::json!({"agent":0,"layers":[1,6],"stages":super::super::gemma_localizer::PREFILL_STAGES,
        "normalize_rows":normalize_rows,"routes":[{"rerun_id":"loc-r128","prefill_chunks":[128]},
        {"rerun_id":"loc-r64","prefill_chunks":[64,64]}]})
}
#[test]
fn localizer_recipe_pins_todays_stage_localizer_literal() {
    let cfg=cfg();let r=super::super::gemma_localizer::localizer_recipe(&cfg,262144);
    // The literal both localizer config checks used before this cutover.
    assert_eq!(serde_json::to_value(&r).unwrap(),serde_json::json!({"family":"Gemma4","capacity":262144,"slots":4,
        "output_allowance":8192,"cache_mode":"GemmaHybrid","route":"stage-localizer",
        "actual_model_config_sha256":hash(format!("{:?}",&cfg).as_bytes())}));
    for routing in [false,true] {
        assert_ne!(acquisition::config_digest(&r).unwrap(),
            acquisition::config_digest(&super::super::gemma_actual::selected_recipe(&cfg,routing)).unwrap());
    }
    let mut other=cfg.clone();other.max_position_embeddings=128;
    assert_ne!(super::super::gemma_localizer::localizer_recipe(&other,262144).actual_model_config_sha256,
        r.actual_model_config_sha256);
}
#[test]
fn localizer_manifest_routes_freeze_identical_coordinates_within_cap() {
    use super::super::gemma_localizer::{localizer_manifest,HOST_PAYLOAD_CAP,PREFILL_STAGES};
    use super::super::receipt::Scalar;
    use std::collections::BTreeSet;
    let cfg=cfg();
    let (t128,e128,b128)=localizer_manifest(&cfg,0,&[128],&[1,6],&PREFILL_STAGES,true).unwrap();
    let (t64,e64,b64)=localizer_manifest(&cfg,0,&[64,64],&[6,1],&PREFILL_STAGES,true).unwrap();
    assert_eq!(e128,e64);assert_eq!(b128,b64);assert!(b128<=HOST_PAYLOAD_CAP);
    let set=|e:&[super::super::receipt::ExpectedTensor]|e.iter().map(|x|x.at.clone()).collect::<BTreeSet<_>>();
    assert_eq!(set(&e128[..]),set(&e64[..]));assert_eq!(set(&e128[..]).len(),e128.len());
    assert_eq!(e128.len(),2*14*128);
    // Two sliding layers: 3x[16,256] + 3x[8,256] + 5x[2816] F32, 128 logits, 2x top-8.
    assert_eq!(b128,2*128*(3*16*256*4+3*8*256*4+5*2816*4+128*4+8*4+8*4));
    assert_eq!(b128,33_439_744);
    assert_eq!(e128[0].at.layer,1);assert_eq!(e128[0].at.stage,Stage::Query);
    assert_eq!(e128[0].shape,vec![1,16,256]);assert_eq!(e128[0].scalar,Scalar::F32);
    assert_eq!(e128[1].at.logical_row_start,1);
    let ids=e128.iter().find(|e|e.at.stage==Stage::RouterIds).unwrap();
    assert_eq!((ids.scalar,ids.shape.clone()),(Scalar::U32,vec![1,8]));
    assert!(e128.iter().all(|e|e.at.agent=="agent-0" && e.at.logical_step==0 && e.at.segment==0));
    assert_eq!(t128.len(),28);assert_eq!(t64.len(),56);
    assert_eq!(t64.iter().filter(|t|t.source_step==64).count(),28);
    for t in t128.iter().chain(&t64) {
        assert_eq!(t.source_stage,t.target.at.stage);assert_eq!(t.source_layer,t.target.at.layer);
        assert_eq!(t.target.at.logical_row_start,t.source_step);assert_eq!(t.target.at.logical_step,0);
        assert!(t.target.normalize_rows);
    }
    // Global layer geometry (layer 5: 2 KV heads x 512).
    let (_,full,_)=localizer_manifest(&cfg,2,&[128],&[5],&[Stage::Key],true).unwrap();
    assert_eq!(full[0].shape,vec![1,2,512]);assert_eq!(full[0].at.agent,"agent-2");
    // Unnormalized captures are per chunk, so routes do not share coordinates.
    let (_,raw128,_)=localizer_manifest(&cfg,0,&[128],&[1],&[Stage::Key],false).unwrap();
    let (_,raw64,_)=localizer_manifest(&cfg,0,&[64,64],&[1],&[Stage::Key],false).unwrap();
    assert_eq!(raw128[0].shape,vec![128,8,256]);assert_eq!(raw64.len(),2);assert_ne!(raw128,raw64);
}
#[test]
fn localizer_manifest_refuses_unbound_scope() {
    use super::super::gemma_localizer::{localizer_manifest,PREFILL_STAGES};
    let cfg=cfg();
    assert!(localizer_manifest(&cfg,4,&[128],&[1],&PREFILL_STAGES,true).is_err());
    assert!(localizer_manifest(&cfg,0,&[64,63],&[1],&PREFILL_STAGES,true).is_err());
    assert!(localizer_manifest(&cfg,0,&[128],&[],&PREFILL_STAGES,true).is_err());
    assert!(localizer_manifest(&cfg,0,&[128],&[30],&PREFILL_STAGES,true).is_err());
    assert!(localizer_manifest(&cfg,0,&[128],&[1,1],&PREFILL_STAGES,true).is_err());
    assert!(localizer_manifest(&cfg,0,&[128],&[1],&[],true).is_err());
    assert!(localizer_manifest(&cfg,0,&[128],&[1],&[Stage::Norm],true).is_err());
    assert!(localizer_manifest(&cfg,0,&[128],&[1],&[Stage::Key,Stage::Key],true).is_err());
    // A 16 MiB F32 hidden row x 5 roles x 128 rows exceeds the 1 GiB host cap.
    let mut wide=cfg;wide.hidden_size=1<<22;
    assert!(localizer_manifest(&wide,0,&[128],&[1],&PREFILL_STAGES,true).is_err());
}
#[test]
fn localizer_prep_config_is_optional_closed_and_route_invariant() {
    assert!(serde_json::from_value::<Config>(localizer_config(None)).unwrap().localizer.is_none());
    let c:Config=serde_json::from_value(localizer_config(Some(two_routes(true)))).unwrap();
    let l=c.localizer.as_ref().unwrap();
    let routes=localizer_routes(&cfg(),l).unwrap();
    assert_eq!(routes.iter().map(|(r,_)|r.rerun_id.as_str()).collect::<Vec<_>>(),vec!["loc-r128","loc-r64"]);
    assert_eq!(routes[0].1.1,routes[1].1.1);assert_eq!(routes[0].1.2,routes[1].1.2);
    let mut unknown=two_routes(true);unknown["warm_live_kernel"]=serde_json::json!("VecHybrid");
    assert!(serde_json::from_value::<Config>(localizer_config(Some(unknown))).is_err());
    let raw:Config=serde_json::from_value(localizer_config(Some(two_routes(false)))).unwrap();
    assert!(localizer_routes(&cfg(),raw.localizer.as_ref().unwrap()).is_err());
    let mut duplicate=two_routes(true);duplicate["routes"][1]["rerun_id"]=serde_json::json!("loc-r128");
    let duplicate:Config=serde_json::from_value(localizer_config(Some(duplicate))).unwrap();
    assert!(localizer_routes(&cfg(),duplicate.localizer.as_ref().unwrap()).is_err());
    let mut unsafe_id=two_routes(true);unsafe_id["routes"][0]["rerun_id"]=serde_json::json!("../x");
    let unsafe_id:Config=serde_json::from_value(localizer_config(Some(unsafe_id))).unwrap();
    assert!(localizer_routes(&cfg(),unsafe_id.localizer.as_ref().unwrap()).is_err());
}
