//! Actual planner/hash-gate controls, no model/device construction.
use super::*;
#[test]
fn q36_complete_checkpoint_plan_preserves_full144_and_preload_budget() {
    let cfg=qwen_cache::selected_header_fixture(false);
    let t0=expected_cache(&cfg,Family::Qwen36,Trajectory::T0ColdSerial,262144,[0,0]).unwrap();
    let t3=expected_cache(&cfg,Family::Qwen36,Trajectory::T3Chunk64,262144,[0,0]).unwrap();
    let t1=expected_cache(&cfg,Family::Qwen36,Trajectory::T1Resume64,262144,[0,0]).unwrap();
    let extra=t3.len()-t0.len();assert!(extra>0);assert_eq!(t1.len(),t0.len()+2*extra);
    for tensors in [&t0,&t3,&t1] {
        let coordinates=tensors.iter().map(|t|t.at.clone()).collect::<std::collections::BTreeSet<_>>();
        assert_eq!(coordinates.len(),tensors.len());
        let bytes=planned_payload(tensors,cfg.vocab_size as usize,2*1024*1024*1024).unwrap();
        assert!(bytes>576*cfg.vocab_size as u64*4);
        assert!(planned_payload(tensors,cfg.vocab_size as usize,bytes-1).is_err());
        assert!(serde_json::to_vec(tensors).unwrap().len()+65536<1048576);
    }
    let before=t1.iter().filter(|t|t.at.logical_step==64 && t.at.segment<1000).count();
    let after=t1.iter().filter(|t|t.at.logical_step==64 && t.at.segment>=1000).count();
    assert_eq!(before,extra);assert_eq!(after,extra);
    assert!(expected_cache(&cfg,Family::Qwen38,Trajectory::T1Resume64,262144,[0,0]).is_err());
    assert!(expected_cache(&cfg,Family::Qwen36,Trajectory::T1Resume64,262144,[64,64]).is_err());
    assert!(planned_payload(&t1,usize::MAX,2*1024*1024*1024).is_err());
    let q38=qwen_cache::selected_header_fixture(true);
    let old=expected_cache(&q38,Family::Qwen38,Trajectory::T1Resume64,262144,[0,0]).unwrap();
    assert!(old.iter().all(|t|matches!(t.at.logical_step,128|144)));
    assert!(!captures64(Family::Qwen36,Trajectory::T2ActualCohort));
    assert!(!captures64(Family::Qwen36,Trajectory::T6CancelRestore));
}
#[test]
fn q36_actual_fullhash_gate_refuses_missing_duplicate_and_rehashed_layout() {
    let source=Tensor {at:Coordinate {logical_step:64,layer:0,stage:Stage::CacheValuePacked,
        agent:"agent-0".into(),logical_row_start:0,segment:0},scalar:Scalar::U8,
        shape:vec![64,256],logical_bytes:16384,parent_identity:1,parent_bytes:16384,
        parent_offset:0,submission_id:None,checked_drain:"HOST_CONTROL".into(),
        file:"source.bin".into(),sha256:"a".repeat(64),pages:vec![]};
    let mut restored=source.clone();restored.at.segment=1000;restored.parent_identity=2;
    restored.file="restored.bin".into();
    assert_eq!(require_restore64_exact(&[source.clone(),restored.clone()]).unwrap(),16384);
    for change in 0..5 {
        let mut bad=restored.clone();
        match change {0=>bad.sha256="b".repeat(64),1=>bad.shape=vec![32,512],
            2=>bad.scalar=Scalar::F16,3=>bad.logical_bytes=0,_=>bad.at.agent="agent-1".into()}
        assert!(require_restore64_exact(&[source.clone(),bad]).is_err());
    }
    assert!(require_restore64_exact(&[source.clone()]).is_err());
    assert!(require_restore64_exact(&[source.clone(),restored.clone(),restored]).is_err());
    assert!(require_restore64_exact(&[]).is_err());
}
