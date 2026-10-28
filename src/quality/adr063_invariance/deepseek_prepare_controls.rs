//! Actual derivation and provenance guards; no model, tokenizer or device load.
use super::*;
#[test]
fn four_real_prompts_must_leave_four_distinct_first144_and_valid_vocab(){
    let mut ids:[Vec<u32>;4]=std::array::from_fn(|i|vec![i as u32;757]);
    let work=derive(&ids,129280).unwrap();assert_eq!(work.lanes[2],ids[2][..144]);
    ids[3][0]=129280;assert!(derive(&ids,129280).is_err());ids[3][0]=3;
    ids[3][..144].copy_from_slice(&vec![2;144]);assert!(derive(&ids,129280).is_err());
    ids[3]=vec![3;143];assert!(derive(&ids,129280).is_err());
}
#[test]
fn full_native_replay_must_match_before_slice(){
    let ids=vec![7u32;200];let render="nativeDS";let wire="a".repeat(64);
    let mut obs=json!({"turn":0,"request_sha256":wire,"prompt_tokens":200,
        "rendered_sha256":hash(render.as_bytes()),"prompt_token_ids_sha256":hash(&serde_json::to_vec(&ids).unwrap())});
    replay_matches(&ids,render,&wire,&obs).unwrap();
    let mut different=ids.clone();different[199]=8;assert!(replay_matches(&different,render,&wire,&obs).is_err());
    obs["rendered_sha256"]=json!("b".repeat(64));assert!(replay_matches(&ids,render,&wire,&obs).is_err());
    assert_ne!(le_hash(&[1,256]),le_hash(&[256,1]));
}
#[test]
fn pending_source_binary_and_unknown_fields_cannot_be_execution_approval(){
    let b=json!({"path":"/private/tmp/bound","sha256":"a".repeat(64)});
    let mut v=json!({"model":{"path":"/private/tmp/model","retained_full_sha256":MODEL_SHA,
        "file_identity":{"device":1,"inode":2,"bytes":3,"mtime":4,"mtime_ns":5,"ctime":6,"ctime_ns":7}},
        "full_content_receipt":b,"tokenizer":b,"template":b,"producer_execution":b,"producer_helper_binding":b,
        "producers":[{"config":b,"proof":b,"request":b},{"config":b,"proof":b,"request":b},
            {"config":b,"proof":b,"request":b},{"config":b,"proof":b,"request":b}],
        "expected_exporter_source_commit":null,"expected_exporter_binary_sha256":null,"output":"/private/tmp/new-ds"});
    let c:Config=serde_json::from_value(v.clone()).unwrap();assert!(c.expected_exporter_source_commit.is_none());
    v["fallback_family"]=json!("qwen");assert!(serde_json::from_value::<Config>(v).is_err());
    assert!(preflight(&std::env::current_dir().unwrap()).is_err());
}
#[test]
fn native_ds_renderer_refuses_wrong_family_or_template_before_tokenization(){
    let t=tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
    assert!(crate::serve::api::handlers::q1_deepseek_prepare::replay(b"{}",&t,"wrong").is_err());
    assert!(crate::serve::api::handlers::q1_deepseek_prepare::replay(b"{\"model\":\"Qwen\"}",&t,
        crate::core::chat_templates::DEEPSEEK_V4_FLASH_0731).is_err());
}
