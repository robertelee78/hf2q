//! Actual manifest/audit/projection controls, no model/device/header.
use super::*;
use super::super::store::Store;
use std::sync::atomic::{AtomicU64,Ordering};
static NEXT:AtomicU64=AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {fn new()->Self {Self(std::env::temp_dir().canonicalize().unwrap().join(format!(
    "q1-gemma-artifact-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed))))}}
impl Drop for Temp {fn drop(&mut self){let _=std::fs::remove_dir_all(&self.0);}}
fn id()->Identity {Identity {family:Family::Gemma4,source_commit:"a".repeat(40),binary_sha256:"b".repeat(64),
    native_version:env!("HF2Q_MLX_NATIVE_VERSION").into(),native_checksum:"c".repeat(64),
    model_sha256:"d".repeat(64),tokenizer_sha256:"e".repeat(64),template_sha256:"f".repeat(64),
    config_sha256:"1".repeat(64),token_vectors_sha256:"2".repeat(64)}}
fn fixture(root:&Path)->Trace {
    let required=(0..2).map(|i|ExpectedTensor {at:Coordinate {logical_step:if i==0 {128}else{64},
        layer:0,stage:Stage::CacheValueNorm,agent:"agent-0".into(),logical_row_start:0,segment:0},
        scalar:Scalar::F32,shape:vec![1]}).collect::<Vec<_>>();
    let mut store=Store::new(root,id(),"HOST_ONLY".into(),required.clone(),8,2).unwrap();
    for e in required {store.capture(e.at,Scalar::F32,e.shape,1,4,0,None,"HOST_CONTROL",&1f32.to_le_bytes()).unwrap();}
    store.finish().unwrap()
}
#[test]
fn actual_original_audit_refuses_extra_corruption_before_any_projection() {
    let l=Temp::new();let r=Temp::new();let left=fixture(&l.0);let right=fixture(&r.0);
    let common=vec![left.coverage.expected[0].clone()];
    let positive=audited_pair(&l.0,&left,&left.coverage.expected,&r.0,&right,&right.coverage.expected,&common).unwrap();
    assert_eq!(positive.2.coverage.captured.len(),1);assert_eq!(positive.3.coverage.captured.len(),1);
    // This original extra64 is excluded from common, but must still be audited.
    let t=&right.coverage.captured[1];let file=r.0.join(&t.file);
    std::fs::write(&file,2f32.to_le_bytes()).unwrap();
    assert!(audited_pair(&l.0,&left,&left.coverage.expected,&r.0,&right,&right.coverage.expected,&common).is_err());
    let mut bad=right.clone();let nan=f32::NAN.to_le_bytes();std::fs::write(&file,nan).unwrap();
    bad.coverage.captured[1].sha256=hash(&nan);bad.coverage.captured[1].pages[0].sha256=hash(&nan);
    assert!(audit::original(&r.0,&bad,&right.coverage.expected).err().unwrap().to_string().contains("nonfinite"));
    std::fs::write(&file,1f32.to_le_bytes()).unwrap();
    let mut missing=right.clone();missing.coverage.captured.pop();
    assert!(audit::original(&r.0,&missing,&right.coverage.expected).is_err());
    let mut duplicate=right.clone();duplicate.coverage.captured[1]=duplicate.coverage.captured[0].clone();
    assert!(audit::original(&r.0,&duplicate,&right.coverage.expected).is_err());
    let mut changed=right.clone();changed.coverage.expected.pop();
    assert!(audit::original(&r.0,&changed,&right.coverage.expected).is_err());
    assert_eq!(left.coverage.captured.len(),2);assert_eq!(right.coverage.captured.len(),2);
}
#[test]
fn selected_exact_config_and_all_original_manifests_are_source_bound() {
    let cfg=selected::cfg();let mut identity=id();
    let recipe=super::super::gemma_actual::recipe(&cfg);
    identity.config_sha256=acquisition::config_digest(&recipe).unwrap();
    let prepared=json!({"schema":"hf2q.q1.gemma-endpoint-preparation.v1",
        "actual_gemma_config_debug":format!("{cfg:?}"),"actual_gemma_config_debug_sha256":recipe.actual_model_config_sha256,
        "recipe":recipe,"config_sha256":identity.config_sha256,"exporter_source_commit":identity.source_commit,
        "exporter_binary_sha256":identity.binary_sha256,"native_version":identity.native_version,
        "model_full_sha256_retained":identity.model_sha256,"tokenizer_file_sha256":identity.tokenizer_sha256,
        "template_file_sha256":identity.template_sha256,"token_vectors_sha256":identity.token_vectors_sha256});
    assert!(selected_config(&prepared,&identity).is_ok());
    let mut wrong=prepared.clone();wrong["actual_gemma_config_debug"]=json!("approximate");
    assert!(selected_config(&wrong,&identity).is_err());
    wrong=prepared.clone();wrong["recipe"]["slots"]=json!(1);assert!(selected_config(&wrong,&identity).is_err());
    assert!(selected_config(&json!({"actual_gemma_config_debug":"approximately Gemma"}),&identity).is_err());
    let common=expected(&cfg,Trajectory::T0ColdSerial).unwrap();assert_eq!(common.len(),5616);
    for (trajectory,count) in [(Trajectory::T1Resume64,10656),(Trajectory::T2ActualCohort,5616),
        (Trajectory::T3Chunk64,8136),(Trajectory::T6CancelRestore,10656)] {
        let all=expected(&cfg,trajectory).unwrap();assert_eq!(all.len(),count);
        assert_eq!(all.iter().map(|t|t.at.clone()).collect::<BTreeSet<_>>().len(),count);
        assert!(common.iter().all(|c|all.contains(c)));
    }
    assert!(expected(&cfg,Trajectory::T5GenuineSpeculation).is_err());
}
#[test]
fn same_root_or_nonce_remains_invalid_under_actual_core() {
    let a=Temp::new();let b=Temp::new();let mut l=fixture(&a.0);let mut r=fixture(&b.0);
    l.acquisition=Some(acquisition::synthetic(&a.0,&l.identity));
    r.acquisition=Some(acquisition::synthetic(&b.0,&r.identity));
    let same=compare::compare(&a.0,&l,&a.0,&l,&BTreeMap::new());
    assert_eq!(same.disposition,Disposition::InvalidEvidence);
    r.acquisition.as_mut().unwrap().nonce=l.acquisition.as_ref().unwrap().nonce.clone();
    let reused=compare::compare(&a.0,&l,&b.0,&r,&BTreeMap::new());
    assert_eq!(reused.disposition,Disposition::InvalidEvidence);
    r.identity.source_commit="9".repeat(40);
    assert_eq!(compare::compare(&a.0,&l,&b.0,&r,&BTreeMap::new()).disposition,Disposition::IdentityMismatch);
}
