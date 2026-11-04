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
    assert!(planned_payload(&t1,cfg.vocab_size as usize,ENDPOINT_PAYLOAD_LIMIT_MAX+1).is_err());
    for family in [Family::Qwen36,Family::Qwen38] {
        assert!(!captures64(family,Trajectory::T0ColdSerial));
        assert!(!captures64(family,Trajectory::T4ReverseOrder));
        assert!(!captures64(family,Trajectory::T7RepeatCold));
        assert!(!captures64(family,Trajectory::T2ActualCohort));
        assert!(!captures64(family,Trajectory::T6CancelRestore));
        assert!(captures64(family,Trajectory::T1Resume64));
        assert!(captures64(family,Trajectory::T3Chunk64));
    }
}
/// Qwen3.8 T1/T3 carry the same direct restore-point evidence as Qwen3.6:
/// T3 adds the source 64-row active cache, T1 adds it plus its segment+1000
/// restored twin, and T0/T4/T7 stay at the 128/144 boundaries only.
#[test]
fn q38_restore64_plan_adds_source_and_restored_active64_only_to_t1_t3() {
    let cfg=qwen_cache::selected_header_fixture(true);
    let plan=|t|expected_cache(&cfg,Family::Qwen38,t,262144,[0,0]).unwrap();
    let (t0,t1,t3)=(plan(Trajectory::T0ColdSerial),plan(Trajectory::T1Resume64),plan(Trajectory::T3Chunk64));
    for t in [Trajectory::T0ColdSerial,Trajectory::T4ReverseOrder,Trajectory::T7RepeatCold] {
        assert!(plan(t).iter().all(|e|matches!(e.at.logical_step,128|144)));
    }
    let extra=t3.len()-t0.len();assert!(extra>0);assert_eq!(t1.len(),t0.len()+2*extra);
    let source=t1.iter().filter(|e|e.at.logical_step==64 && e.at.segment<1000).collect::<Vec<_>>();
    let restored=t1.iter().filter(|e|e.at.logical_step==64 && e.at.segment>=1000).collect::<Vec<_>>();
    assert_eq!(source.len(),extra);assert_eq!(restored.len(),extra);
    for s in &source {
        let mut twin=s.at.clone();twin.segment+=1000;
        let r=restored.iter().filter(|r|r.at==twin).collect::<Vec<_>>();
        assert_eq!(r.len(),1);assert!(r[0].scalar==s.scalar && r[0].shape==s.shape);
    }
    // T3 extras are exactly T1's source (non-restored) active64 records.
    let t3_64=t3.iter().filter(|e|e.at.logical_step==64).map(|e|e.at.clone()).collect::<std::collections::BTreeSet<_>>();
    assert_eq!(t3_64,source.iter().map(|e|e.at.clone()).collect());
    for tensors in [&t0,&t3,&t1] {
        let coordinates=tensors.iter().map(|t|t.at.clone()).collect::<std::collections::BTreeSet<_>>();
        assert_eq!(coordinates.len(),tensors.len());
        planned_payload(tensors,cfg.vocab_size as usize,ENDPOINT_PAYLOAD_LIMIT_MAX).unwrap();
    }
    assert!(expected_cache(&cfg,Family::Qwen38,Trajectory::T1Resume64,262144,[64,64]).is_err());
    assert!(expected_cache(&cfg,Family::Qwen36,Trajectory::T1Resume64,262144,[0,0]).is_err());
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
