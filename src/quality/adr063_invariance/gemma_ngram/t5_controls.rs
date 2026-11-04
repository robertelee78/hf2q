//! Hosted Q1 Gemma T5 / verifier-route localizer controls. Header fixture
//! geometry only; synthetic values are never actual speculative evidence.
use super::*;
use super::{localizer,runner::t5_run::{self as run,Branch,Extra},t5::{self,Arm}};
use crate::adr063_invariance::{family::{self,OWNED_AGENT},gemma_cohort_localizer::{DECODE_STAGES,STAGE_PAYLOAD_CAP},
    gemma_localizer::PREFILL_STAGES,receipt::*};
fn cfg()->crate::serve::config::Gemma4Config {crate::adr063_invariance::gemma_actual::controls::cfg()}
fn identity()->serde_json::Value {
    json!({"family":"Gemma4","source_commit":"a".repeat(40),"binary_sha256":"b".repeat(64),"native_version":env!("HF2Q_MLX_NATIVE_VERSION"),
        "native_checksum":"c".repeat(64),"model_sha256":"d".repeat(64),"tokenizer_sha256":"e".repeat(64),"template_sha256":"f".repeat(64),
        "config_sha256":"1".repeat(64),"token_vectors_sha256":"2".repeat(64)})
}
fn binding(name:&str)->serde_json::Value {json!({"path":format!("/private/q1/{name}"),"sha256":"9".repeat(64)})}
/// A complete packet; the localizer keys exactly under `plan`.
fn packet(arm:Arm,plan:Option<&localizer::Plan>,branch:Branch,pass:u8)->serde_json::Value {
    use crate::adr063_invariance::gemma_routing_witness as w;
    let mut p=json!({"execution_approved":true,"identity":identity(),"acquisition":{"composition":binding("composition.json"),
        "independent_review":binding("review.json")},"model":"/private/q1/model.gguf","output":"/private/q1/out",
        "greedy_workload":binding("workload.json"),"independent_dataset_review":binding("dataset-review.json"),
        "configured_context":262144,"configured_output_allowance":8192,"logical_serial_owners":4,
        "proposal_origin":t5::INJECTED_ORIGIN,"ngram_k":3,"ngram_min":1,"ngram_max":3,"payload_limit":PAYLOAD,"metadata_limit":METADATA,
        "routing_witness":{"payload_bytes":w::PAYLOAD_BYTES,"metadata_bytes":w::METADATA_BYTES,"cpu_ledger_bytes":w::CPU_LEDGER_BYTES},
        "arm":arm,"lanes":[0,1,2,3],"host_payload_limit":t5::host_payload_limit(&cfg(),arm,plan.map_or(0,|p|p.host_payload_limit)).unwrap()});
    if let Some(plan)=plan {
        let o=p.as_object_mut().unwrap();
        o.insert("stage_localizer".into(),json!(plan));o.insert("branch".into(),json!(branch));o.insert("pass".into(),json!(pass));
        o.insert("t5_receipts".into(),json!({"ordinary":binding("t5ord/execution.json"),"ngram":binding("t5ngram/execution.json")}));
        o.insert("manifest".into(),binding("manifest.json"));
        if branch==Branch::P {o.insert("all128_precondition".into(),binding("all128-precondition.json"));}
    }
    p
}
fn extra(v:serde_json::Value)->Extra {
    run::extra_of(serde_json::from_value(v).unwrap())
}
fn pass(branch:Branch,pass:u8,accepted:Option<[u8;4]>,onsets:Option<[Option<u32>;4]>)->super::export::Pass {
    super::export::Pass {branch,pass,rerun_prefix:"gemma-t5loc".into(),accepted,onset_layers:onsets}
}
#[test]
fn q1_gemma_t5_manifest_counts_and_payload_on_selected_geometry(){
    let cfg=cfg();
    for lane in 0..4 {for pos in [128,144] {assert_eq!(t5::cache(&cfg,lane,pos).unwrap().len(),t5::CACHE_RECORDS);}}
    assert!(t5::cache(&cfg,4,128).is_err() && t5::cache(&cfg,0,64).is_err());
    let (prefork,committed,raw)=(t5::prefork(&cfg).unwrap(),t5::committed(&cfg).unwrap(),t5::raw(&cfg));
    assert_eq!((prefork.len(),committed.len(),raw.len()),(4204,4264,4));
    assert_eq!((t5::payload(&prefork).unwrap(),t5::payload(&committed).unwrap(),t5::payload(&raw).unwrap()),
        (321_839_104,424_459_264,16_777_216));
    // One cache row: 11 x 56,320 + 880 bytes; 630 Hybrid + 420 dense companion records.
    let one=t5::cache(&cfg,0,128).unwrap();assert_eq!(t5::payload(&one).unwrap(),128*(11*56320+880));
    assert_eq!(one.iter().filter(|e|e.at.segment>=t5::DENSE_SEGMENT).count(),420);
    assert!(one.iter().filter(|e|e.at.segment>=t5::DENSE_SEGMENT).all(|e|e.scalar==Scalar::F32 &&
        matches!(e.at.stage,Stage::CacheKey|Stage::CacheValue) && e.at.segment<t5::DENSE_SEGMENT+8));
    for set in [&prefork,&committed] {
        let unique:std::collections::BTreeSet<_>=set.iter().map(|e|e.at.clone()).collect();assert_eq!(unique.len(),set.len());
    }
    assert!(prefork.iter().filter(|e|e.at.stage==Stage::Logits).all(|e|e.at.logical_row_start==127 && e.at.layer==u32::MAX));
    assert_eq!(committed.iter().filter(|e|e.at.stage==Stage::Logits).count(),64);
    assert!(raw.iter().all(|e|e.shape==[4,262144] && e.at.logical_step==128 && e.at.stage==Stage::SpecVerifierLogits && e.at.layer==u32::MAX));
    assert_eq!(t5::stores(&cfg,Arm::OrdinaryGlobal).unwrap().iter().map(|s|s.0).collect::<Vec<_>>(),[t5::PREFORK,t5::COMMITTED]);
    assert_eq!(t5::stores(&cfg,Arm::NgramTransaction).unwrap().iter().map(|s|s.0).collect::<Vec<_>>(),[t5::PREFORK,t5::COMMITTED,t5::RAW]);
    let charge=t5::witness_charge();
    assert_eq!(t5::host_payload_limit(&cfg,Arm::OrdinaryGlobal,0).unwrap(),746_298_368+charge);
    assert_eq!(t5::host_payload_limit(&cfg,Arm::NgramTransaction,0).unwrap(),763_075_584+charge);
    assert!(t5::host_payload_limit(&cfg,Arm::NgramTransaction,PAYLOAD).is_err());
}
#[test]
fn q1_gemma_t5_journal_cache_roles_land_on_manifest_coordinates(){
    let cfg=cfg();
    let manifest:std::collections::BTreeSet<_>=t5::cache(&cfg,2,144).unwrap().into_iter().map(|e|e.at).collect();
    let mut captured=std::collections::BTreeSet::new();
    for layer in 0..30 {let heads=if (layer+1)%6==0 {2}else{8};
        for role in ["CacheKey","CacheValuePacked","CacheValueNorm","PrefillDenseKey","PrefillDenseValue"] {for head in 0..heads {
            let (stage,segment)=journal::store_role(role,head).unwrap();
            let mut at=crate::adr063_invariance::gemma_cache::at(2,144,layer,stage,head,false);at.segment=segment;assert!(captured.insert(at));
        }}}
    assert_eq!(captured,manifest);
    assert!(journal::store_role("CacheValueDense",0).is_err());
}
#[test]
fn q1_gemma_t5_packet_and_arm_are_closed(){
    for arm in [Arm::OrdinaryGlobal,Arm::NgramTransaction] {
        let x=extra(packet(arm,None,Branch::N,1));assert_eq!(x.arm,arm);run::pairing(&x,false).unwrap();
        // A plain T5 packet never runs under the localizer filter.
        assert!(run::pairing(&x,true).is_err());
    }
    let mut bad=packet(Arm::OrdinaryGlobal,None,Branch::N,1);bad["arm"]=json!("OrdinarySlotAware");
    assert!(serde_json::from_value::<run::T5Packet>(bad).is_err());
    let mut bad=packet(Arm::OrdinaryGlobal,None,Branch::N,1);bad["expected_stores"]=json!({});
    assert!(serde_json::from_value::<run::T5Packet>(bad).is_err());
    let mut bad=packet(Arm::OrdinaryGlobal,None,Branch::N,1);bad.as_object_mut().unwrap().remove("host_payload_limit");
    assert!(serde_json::from_value::<run::T5Packet>(bad).is_err());
    let mut bad=packet(Arm::OrdinaryGlobal,None,Branch::N,1);bad["lanes"]=json!([0,1,2]);
    assert!(run::pairing(&extra(bad),false).is_err());
    // Localizer keys only under the localizer filter, and only together.
    let p1=super::export::plan(&cfg(),&pass(Branch::N,1,Some([2,2,1,0]),None),Arm::NgramTransaction).unwrap();
    let good=packet(Arm::NgramTransaction,Some(&p1),Branch::N,1);
    run::pairing(&extra(good.clone()),true).unwrap();
    assert!(run::pairing(&extra(good.clone()),false).is_err());
    let mut partial=good.clone();partial.as_object_mut().unwrap().remove("manifest");
    assert!(run::pairing(&extra(partial),true).is_err());
    let mut wrong_pass=good.clone();wrong_pass["pass"]=json!(2);
    assert!(run::pairing(&extra(wrong_pass),true).is_err());
    // Branch P: ordinary arm only, accepted null.
    let pp=super::export::plan(&cfg(),&pass(Branch::P,1,None,None),Arm::OrdinaryGlobal).unwrap();
    run::pairing(&extra(packet(Arm::OrdinaryGlobal,Some(&pp),Branch::P,1)),true).unwrap();
    assert!(run::pairing(&extra(packet(Arm::NgramTransaction,Some(&pp),Branch::P,1)),true).is_err());
    assert!(run::pairing(&extra(packet(Arm::OrdinaryGlobal,Some(&p1),Branch::P,1)),true).is_err());
    assert!(run::pairing(&extra(packet(Arm::OrdinaryGlobal,Some(&pp),Branch::N,1)),true).is_err());
    assert!(super::export::plan(&cfg(),&pass(Branch::P,1,None,None),Arm::NgramTransaction).is_err());
    assert!(super::export::plan(&cfg(),&pass(Branch::N,1,None,None),Arm::NgramTransaction).is_err());
    let mut three=p1.clone();three.accepted=Some([3,0,0,0]);
    assert!(run::pairing(&extra(packet(Arm::NgramTransaction,Some(&three),Branch::N,1)),true).is_err());
    // The All128 precondition record is bound exactly by branch P.
    let mut no_pre=packet(Arm::OrdinaryGlobal,Some(&pp),Branch::P,1);no_pre.as_object_mut().unwrap().remove("all128_precondition");
    assert!(run::pairing(&extra(no_pre),true).is_err());
    let mut n_pre=good;n_pre["all128_precondition"]=binding("all128-precondition.json");
    assert!(run::pairing(&extra(n_pre),true).is_err());
    let mut t5_pre=packet(Arm::OrdinaryGlobal,None,Branch::N,1);t5_pre["all128_precondition"]=binding("all128-precondition.json");
    assert!(run::pairing(&extra(t5_pre),false).is_err());
}
fn precondition(holds:bool)->serde_json::Value {
    json!({"schema":run::PRECONDITION_SCHEMA,"endpoint_trace":binding("all128/trace.json"),"workload":binding("workload.json"),
        "lanes":(0..4u32).map(|l|json!({"lane":l,"argmax":1128+l,"frozen_128":1128+l,"equal":holds||l!=2})).collect::<Vec<_>>(),
        "holds":holds,"whole_q1_qualified":false})
}
#[test]
fn q1_gemma_t5_branch_p_requires_a_holding_all128_precondition(){
    let (sha,frozen)=("9".repeat(64),[1128,1129,1130,1131]);
    run::check_precondition(&precondition(true),&sha,frozen).unwrap();
    // A failed precondition is the build-drift path: branch P is not run.
    let e=run::check_precondition(&precondition(false),&sha,frozen).unwrap_err();
    assert!(format!("{e:#}").contains("DriftAll128"),"{e:#}");
    assert!(run::check_precondition(&precondition(true),&"8".repeat(64),frozen).is_err());
    assert!(run::check_precondition(&precondition(true),&sha,[1128,1129,1130,1]).is_err());
    for (key,value) in [("schema",json!("hf2q.q1.other.v1")),("whole_q1_qualified",json!(true)),("lanes",json!([]))] {
        let mut bad=precondition(true);bad[key]=value;assert!(run::check_precondition(&bad,&sha,frozen).is_err(),"{key}");
    }
    let mut bad=precondition(true);bad["lanes"][1]["equal"]=json!(false);
    assert!(run::check_precondition(&bad,&sha,frozen).is_err());
    let mut bad=precondition(true);bad["lanes"][3]["argmax"]=json!(7);
    assert!(run::check_precondition(&bad,&sha,frozen).is_err());
}
const HOOK_VOCAB:usize=8;
/// A real installed T5 diagnostic (journal + Q1 stores of a small synthetic
/// vocabulary) with lane 0's owner in place; no model or device.
fn hook_harness(tag:&str,arm:Arm,frozen_accepted:Option<[usize;4]>)->(std::path::PathBuf,Scope,Vec<u32>) {
    use crate::adr063_invariance::store::Store;
    let root=std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("hf2q-q1-t5hook-{tag}-{}",std::process::id()));
    let _=std::fs::remove_dir_all(&root);
    let scope=Scope::install(&root,json!({"schema":"HOST_ONLY_CONTROL"})).unwrap();
    let identity:Identity=serde_json::from_value(identity()).unwrap();
    let logits=|row:usize|ExpectedTensor {at:t5::logits_at(0,row),scalar:Scalar::F32,shape:vec![1,HOOK_VOCAB]};
    let store=|name:&str,expected:Vec<ExpectedTensor>|{let n=expected.len();
        Store::new(&root.join(name),identity.clone(),t5::trajectory(arm,name),expected,1<<20,n).unwrap()};
    let raw=(arm==Arm::NgramTransaction).then(||store(t5::RAW,vec![ExpectedTensor {at:Coordinate {logical_step:128,layer:u32::MAX,
        stage:Stage::SpecVerifierLogits,agent:"agent-0".into(),logical_row_start:0,segment:0},scalar:Scalar::F32,shape:vec![4,HOOK_VOCAB]}]));
    let state=t5::State {arm,vocab:HOOK_VOCAB,prefork:store(t5::PREFORK,vec![logits(127)]),committed:store(t5::COMMITTED,(128..144).map(logits).collect()),
        raw,lanes:Vec::new(),verifier:Vec::new(),frozen_accepted,pause:None,class_b:None,localizer:false};
    assert!(scope.set_t5(state).is_ok());
    let frozen:Vec<u32>=(0..144).map(|n|1000+n).collect();
    scope.owner(0,&frozen).unwrap();
    (root,scope,frozen)
}
/// The owner after a checked verifier: its four argmax rows and post-softcap bytes.
fn verified(rows:[u32;4]) {
    with(|a|{let o=a.owner.as_mut().unwrap();o.phase=Phase::Verified;o.rows=rows.to_vec();
        a.t5.as_mut().unwrap().verifier=(0..4*HOOK_VOCAB).flat_map(|i|(i as f32).to_le_bytes()).collect();Ok(())}).unwrap();
}
fn phase()->Phase {ACTIVE.with(|a|a.borrow().as_ref().unwrap().owner.as_ref().unwrap().phase)}
fn close(root:std::path::PathBuf,scope:Scope) {scope.finish("HOST_ONLY_CONTROL").unwrap();std::fs::remove_dir_all(root).unwrap();}
#[test]
fn q1_gemma_t5_decision_hook_refuses_a_frozen_accepted_mismatch(){
    let (root,scope,frozen)=hook_harness("frozen",Arm::NgramTransaction,Some([1,0,0,0]));
    let fixture=super::fixture(&frozen,262144).unwrap();let argmax=[fixture[0],fixture[1],7,9];
    verified(argmax);
    let e=decision(&fixture,&argmax,2,7).unwrap_err();
    assert!(format!("{e:#}").contains("frozen localizer plan"),"{e:#}");
    // Refused before any store write or transaction record; the owner never decided.
    assert_eq!(phase(),Phase::Verified);
    let s=scope.take_t5().unwrap();
    assert!(s.lanes[0].transaction.as_ref().unwrap().accepted.is_none());
    assert_eq!(s.raw.unwrap().finish().unwrap().coverage.captured.len(),0);
    assert_eq!(s.committed.finish().unwrap().coverage.captured.len(),0);
    close(root,scope);
}
#[test]
fn q1_gemma_t5_hooks_commit_verifier_rows_cursor_and_teacher_forced_inputs(){
    let (root,scope,frozen)=hook_harness("commit",Arm::NgramTransaction,Some([2,0,0,0]));
    let fixture=super::fixture(&frozen,262144).unwrap();let argmax=[fixture[0],fixture[1],7,9];
    verified(argmax);decision(&fixture,&argmax,2,7).unwrap();assert_eq!(phase(),Phase::Decided);
    // Rollback cursor: only the real committed 129+a is admitted.
    let mut output=frozen[..129].to_vec();output.extend([fixture[0],fixture[1],7]);
    for wrong in [130,132] {assert!(rollback_committed(&output,1,wrong).is_err());assert_eq!(phase(),Phase::Decided);}
    assert!(rollback_committed(&output[..131],1,131).is_err());
    assert!(rollback_committed(&output,1,131).unwrap());assert_eq!(phase(),Phase::Committed);
    // Teacher forcing: inputs are frozen[p]; the fallback is never an input.
    t5_teacher_forced(0,131,frozen[131],5).unwrap();
    assert!(t5_teacher_forced(0,132,7,5).is_err());
    assert!(t5_teacher_forced(0,133,frozen[133],5).is_err(),"rows in position order");
    assert!(t5_lane_inputs_checked(0).is_err(),"incomplete continuation");
    for p in 132..144 {t5_teacher_forced(0,p,frozen[p],5).unwrap();}
    t5_lane_inputs_checked(0).unwrap();
    let s=scope.take_t5().unwrap();
    let x=s.lanes[0].transaction.clone().unwrap();
    assert_eq!((x.accepted,x.rejected,x.fallback,x.rollback_cursor,x.verifier_argmax),(Some(2),Some(1),Some(7),Some(131),argmax.to_vec()));
    assert_eq!(s.lanes[0].decode_inputs,(131..144).map(|position|t5::Row {position,token:frozen[position]}).collect::<Vec<_>>());
    assert_eq!(s.raw.unwrap().finish().unwrap().coverage.captured.len(),1);
    let committed=s.committed.finish().unwrap();
    assert_eq!(committed.coverage.captured.iter().map(|t|t.at.logical_row_start).collect::<Vec<_>>(),[128,129,130]);
    close(root,scope);
}
#[test]
fn q1_gemma_t5_initial_hook_records_the_class_b_refusal(){
    let (root,scope,frozen)=hook_harness("classb",Arm::NgramTransaction,None);
    assert!(initial_decided(&frozen[1..129],frozen[128]).is_err(),"another prompt");
    assert_eq!(phase(),Phase::Ready);
    let e=initial_decided(&frozen[..128],999).unwrap_err();
    assert_eq!(format!("{e}"),t5::CLASS_B);assert_eq!(phase(),Phase::Initial);
    let s=scope.take_t5().unwrap();
    assert_eq!(s.class_b,Some(json!({"row":127,"reason":t5::CLASS_B,"lane":0})));
    assert_eq!((s.lanes[0].initial_argmax,s.lanes[0].initial_equals_frozen),(Some(999),Some(false)));
    close(root,scope);
    let (root,scope,frozen)=hook_harness("agree",Arm::NgramTransaction,None);
    initial_decided(&frozen[..128],frozen[128]).unwrap();
    let s=scope.take_t5().unwrap();assert_eq!(s.class_b,None);assert_eq!(s.lanes[0].initial_equals_frozen,Some(true));
    close(root,scope);
}
fn receipt(arm:Arm,status:&str,class_b:serde_json::Value,accepted:[usize;4])->serde_json::Value {
    json!({"schema":t5::SCHEMA,"arm":arm,"status":status,"error":if status=="CompleteScopedAcquisition" {json!(null)}else{json!("x")},
        "lanes":[0,1,2,3],"class_b_refusal":class_b,"whole_q1_qualified":false,
        "decisions":(0..4).map(|l|json!({"lane":l,"transaction":{"accepted":accepted[l]}})).collect::<Vec<_>>()})
}
#[test]
fn q1_gemma_t5_initial_decision_refusal_is_a_class_b_record(){
    assert_eq!(t5::initial_refusal(Arm::NgramTransaction,2,7,8),
        Some(json!({"row":127,"reason":"RouteSensitive: global-route initial decision differs from frozen at row 127","lane":2})));
    assert_eq!(t5::initial_refusal(Arm::NgramTransaction,2,8,8),None);
    // The ordinary arm records a cross-route observation, never a refusal.
    assert_eq!(t5::initial_refusal(Arm::OrdinaryGlobal,2,7,8),None);
    let ord=receipt(Arm::OrdinaryGlobal,"CompleteScopedAcquisition",json!(null),[0;4]);
    let refused=receipt(Arm::NgramTransaction,"FailedOrIncomplete",t5::initial_refusal(Arm::NgramTransaction,1,7,8).unwrap(),[0;4]);
    let done=receipt(Arm::NgramTransaction,"CompleteScopedAcquisition",json!(null),[2,2,1,0]);
    run::check_receipts(&ord,&refused,Branch::P,None).unwrap();
    assert!(run::check_receipts(&ord,&done,Branch::P,None).is_err());
    assert!(run::check_receipts(&ord,&refused,Branch::N,Some([0;4])).is_err());
    run::check_receipts(&ord,&done,Branch::N,Some([2,2,1,0])).unwrap();
    assert!(run::check_receipts(&ord,&done,Branch::N,Some([2,2,1,1])).is_err());
    assert!(run::check_receipts(&done,&ord,Branch::N,Some([2,2,1,0])).is_err());
}
#[test]
fn q1_gemma_t5_teacher_forcing_uses_frozen_inputs_only(){
    let frozen:Vec<u32>=(0..144).map(|n|1000+n).collect();
    assert_eq!(t5::decode_inputs(Arm::OrdinaryGlobal,&frozen,None).unwrap(),(128..144).map(|p|(p,1000+p as u32)).collect::<Vec<_>>());
    for a in 0..=2 {
        let rows=t5::decode_inputs(Arm::NgramTransaction,&frozen,Some(a)).unwrap();
        assert_eq!(rows.first().unwrap().0,129+a);assert_eq!(rows.len(),15-a);assert!(rows.iter().all(|&(p,t)|t==frozen[p]));
    }
    assert!(t5::decode_inputs(Arm::NgramTransaction,&frozen,Some(3)).is_err());
    assert!(t5::decode_inputs(Arm::NgramTransaction,&frozen,None).is_err());
    assert!(t5::decode_inputs(Arm::OrdinaryGlobal,&frozen,Some(0)).is_err());
    assert!(t5::decode_inputs(Arm::OrdinaryGlobal,&frozen[..143],None).is_err());
    // The offline receipt join refuses a fallback or model argmax used as an input.
    let lanes=vec![frozen.clone();4];
    let fixture=super::fixture(&frozen,262144).unwrap();
    let lane=|l:usize,inputs:Vec<(usize,u32)>|json!({"lane":l,"initial_argmax":1128,"frozen_128":1128,"initial_equals_frozen":true,
        "decode_inputs":inputs.iter().map(|&(p,t)|json!({"position":p,"token":t})).collect::<Vec<_>>(),
        "decode_argmax":inputs.iter().map(|&(p,_)|json!({"position":p,"token":5})).collect::<Vec<_>>(),
        "transaction":{"proposer_ran":true,"proposed":[],"injected":fixture,"injected_origin":t5::INJECTED_ORIGIN,
            "verifier_argmax":[1129,1130,1131,1132],"accepted":2,"rejected":1,"fallback":1131,"rollback_cursor":131}});
    let good=(0..4).map(|l|lane(l,t5::decode_inputs(Arm::NgramTransaction,&frozen,Some(2)).unwrap())).collect::<Vec<_>>();
    let v=|d:Vec<serde_json::Value>|json!({"schema":t5::SCHEMA,"arm":"NgramTransaction","status":"CompleteScopedAcquisition","error":null,
        "class_b_refusal":null,"whole_q1_qualified":false,"lanes":[0,1,2,3],"decisions":d});
    assert_eq!(super::compare::execution(&v(good.clone()),Arm::NgramTransaction,&lanes).unwrap(),vec![1128;4]);
    let mut forced=good.clone();forced[1]["decode_inputs"][0]["token"]=json!(7);
    assert!(super::compare::execution(&v(forced),Arm::NgramTransaction,&lanes).is_err());
    let mut fallback=good;fallback[3]["transaction"]["fallback"]=json!(7);
    assert!(super::compare::execution(&v(fallback),Arm::NgramTransaction,&lanes).is_err());
}
#[test]
fn q1_gemma_t5_rollback_cursor_other_than_129_plus_a_refuses(){
    for a in 0..=2 {
        t5::require_cursor(129+a,a).unwrap();
        for wrong in [128+a,130+a,132] {assert!(t5::require_cursor(wrong,a).is_err());}
        require_commit(129+a,130+a,3-a).unwrap();
    }
    assert!(t5::require_cursor(132,3).is_err());
    assert!(t5::require_accepted(3).is_err());
}
#[test]
fn q1_gemma_t5_inactive_hooks_return_original_values(){
    assert!(ACTIVE.with(|a|a.borrow().is_none()));
    assert_eq!(proposal(&[1,2],&[9,9,9],3).unwrap(),None);
    routing_before_initial().unwrap();before_verify(&[1,2,3],0).unwrap();decision(&[1],&[1,2],1,2).unwrap();
    assert!(!family::enabled(Family::Gemma4) && !family::paused() && !family::selected(Family::Gemma4,0,0,Stage::Query));
    assert!(family::pause().is_err(),"no diagnostic to pause");
}
#[test]
fn q1_gemma_ngram_localizer_manifest_counts_for_each_accepted_and_refusals(){
    let cfg=cfg();
    for a in 0..=2usize {
        let accepted=[a;4];
        let (targets,expected,_)=localizer::manifest(&cfg,&(0..30).collect::<Vec<_>>(),&localizer::PASS1_STAGES,&localizer::PASS1_STAGES,
            Arm::NgramTransaction,Some(accepted)).unwrap();
        assert_eq!(expected.len(),3*30*132*4+3*30*(15-a)*4);
        assert_eq!(targets.len(),3*30+(15-a)*3*30);
        // Per lane, execution order: verify rows 0..131 at segment 1, then decode 129+a..143 at segment 0.
        let lane0:Vec<_>=expected.iter().filter(|e|e.at.agent=="agent-0").collect();
        let split=lane0.iter().position(|e|e.at.segment==0).unwrap();
        assert!(lane0[..split].iter().all(|e|e.at.segment==localizer::VERIFY_SEGMENT && e.at.logical_row_start<132));
        assert!(lane0[split..].iter().all(|e|e.at.segment==0 && (129+a as u64..144).contains(&e.at.logical_row_start)));
        assert!(lane0[split..].windows(2).all(|w|w[0].at.logical_row_start<=w[1].at.logical_row_start));
        assert!(expected.iter().all(|e|e.at.logical_step==0 && e.shape[0]==1));
        let unique:std::collections::BTreeSet<_>=expected.iter().map(|e|e.at.clone()).collect();assert_eq!(unique.len(),expected.len());
    }
    // Mixed per-lane counts give per-lane decode extents.
    let (_,mixed,_)=localizer::manifest(&cfg,&[3],&[Stage::Attention],&[Stage::Attention],Arm::NgramTransaction,Some([0,1,2,2])).unwrap();
    for (lane,a) in [0,1,2,2].into_iter().enumerate() {
        assert_eq!(mixed.iter().filter(|e|e.at.agent==format!("agent-{lane}") && e.at.segment==0).count(),15-a);
    }
    let (_,ordinary,_)=localizer::manifest(&cfg,&[3],&[Stage::Attention],&[Stage::Attention],Arm::OrdinaryGlobal,None).unwrap();
    assert_eq!(ordinary.len(),4*(128+16));assert!(ordinary.iter().all(|e|e.at.segment==0));
    assert!(localizer::manifest(&cfg,&[3],&[Stage::Attention],&[],Arm::NgramTransaction,Some([3,0,0,0])).is_err());
    assert!(localizer::manifest(&cfg,&[3],&[Stage::Attention],&[],Arm::NgramTransaction,None).is_err());
    let mut plan=super::export::plan(&cfg,&pass(Branch::N,1,Some([1,1,1,1]),None),Arm::NgramTransaction).unwrap();
    plan.accepted=Some([3,1,1,1]);assert!(plan.accepted().is_err());
}
#[test]
fn q1_gemma_ngram_verify_owners_are_segment_one_rows_0_to_131(){
    assert_eq!(localizer::verify_owners(2,0,132).unwrap(),(0..132).map(|p|(2,p)).collect::<Vec<_>>());
    for (lane,start,rows) in [(2,0,133),(2,1,131),(2,128,4),(4,0,132),(0,0,0)] {assert!(localizer::verify_owners(lane,start,rows).is_err());}
    let (targets,_,_)=localizer::manifest(&cfg(),&[0],&[Stage::Key],&[Stage::Key],Arm::NgramTransaction,Some([0;4])).unwrap();
    let verify=&targets[&(0,0,Stage::Key)];
    assert!(verify.normalize_rows && verify.at.agent==OWNED_AGENT && verify.at.segment==localizer::VERIFY_SEGMENT);
    assert_eq!(targets[&(131,0,Stage::Key)].at.segment,0);
    // The serial-lane owner map of the verify call is exactly these owners.
    assert_eq!(crate::adr063_invariance::gemma_prefill::row_owners(None,Some(2),0,132).unwrap().unwrap(),localizer::verify_owners(2,0,132).unwrap());
}
#[test]
fn q1_gemma_family_pause_captures_nothing_and_refuses_nesting(){
    use crate::adr063_invariance::store::Store;
    let root=std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("hf2q-q1-pause-{}",std::process::id()));
    let _=std::fs::remove_dir_all(&root);
    let identity:Identity=serde_json::from_value(identity()).unwrap();
    let at=|agent:&str,row:u64|Coordinate {logical_step:0,layer:0,stage:Stage::RouterIds,agent:agent.into(),logical_row_start:row,segment:1};
    let expected=(0..2).map(|row|ExpectedTensor {at:at("agent-1",row),scalar:Scalar::U32,shape:vec![1,2]}).collect::<Vec<_>>();
    let store=Store::new(&root,identity,"HOST_ONLY_CONTROL".into(),expected,1024,2).unwrap();
    let mut targets=crate::adr063_invariance::gemma_cohort_localizer::Targets::new();
    targets.insert((0,0,Stage::RouterIds),CaptureTarget {at:at(OWNED_AGENT,0),normalize_rows:true});
    let _scope=family::install(Family::Gemma4,store,targets).unwrap();
    let _gpu=crate::inference::hf2q_gpu_test_lock();
    let device=mlx_native::MlxDevice::new().unwrap();
    let mut buf=device.alloc_buffer(16,mlx_native::DType::U32,vec![2,2]).unwrap();
    buf.as_mut_slice::<u32>().unwrap().copy_from_slice(&[0,1,2,3]);
    let owners=[(1,0),(1,1)];
    {
        let _pause=family::pause().unwrap();
        assert!(family::paused() && !family::enabled(Family::Gemma4) && !family::selected(Family::Gemma4,0,0,Stage::RouterIds));
        assert!(family::pause().is_err(),"nested pause refuses");
        family::drained_owned_rows(Family::Gemma4,0,0,Stage::RouterIds,&buf,&owners,"HOST").unwrap();
    }
    assert!(!family::paused() && family::enabled(Family::Gemma4));
    family::drained_owned_rows(Family::Gemma4,0,0,Stage::RouterIds,&buf,&owners,"HOST").unwrap();
    let trace=family::finish().unwrap();
    // Exactly the post-pause rows; nothing captured or refused under the pause.
    assert_eq!(trace.coverage.captured.len(),2);assert!(trace.coverage.errors.is_empty());
    assert!(trace.coverage.captured.iter().all(|t|t.at.segment==1 && t.at.agent=="agent-1"));
    std::fs::remove_dir_all(&root).unwrap();
}
#[test]
fn q1_gemma_t5_localizer_admits_only_the_bound_manifest_extent(){
    let cfg=cfg();
    let plan=super::export::plan(&cfg,&pass(Branch::N,2,Some([1,2,0,1]),Some([Some(3),Some(0),None,Some(4)])),Arm::NgramTransaction).unwrap();
    assert_eq!(plan.layers,[0,2,3,4]);
    let m=super::export::manifest(&cfg,Arm::NgramTransaction,Some(&plan)).unwrap();
    let x=extra(packet(Arm::NgramTransaction,Some(&plan),Branch::N,2));
    let (_,bytes,stages)=run::admit_manifest(&cfg,&x,&m).unwrap();
    assert_eq!(bytes,plan.host_payload_limit);assert_eq!(&stages,&m.expected_stores[t5::STAGES]);
    let tamper=|f:&dyn Fn(&mut run::Manifest)|{let mut m=super::export::manifest(&cfg,Arm::NgramTransaction,Some(&plan)).unwrap();f(&mut m);run::admit_manifest(&cfg,&x,&m).is_err()};
    assert!(tamper(&|m:&mut run::Manifest|m.expected_stores.get_mut(t5::STAGES).unwrap().swap(0,1)));
    assert!(tamper(&|m:&mut run::Manifest|{m.expected_stores.get_mut(t5::STAGES).unwrap().pop();}));
    assert!(tamper(&|m:&mut run::Manifest|{m.expected_stores.remove(t5::RAW);}));
    assert!(tamper(&|m:&mut run::Manifest|{m.expected_stores.get_mut(t5::COMMITTED).unwrap().pop();}));
    assert!(tamper(&|m:&mut run::Manifest|m.host_payload_limit+=1));
    assert!(tamper(&|m:&mut run::Manifest|m.lanes=vec![0,1,2]));
    // A Plan with another frozen accepted count or layer set is a different extent.
    let mut other=plan.clone();other.accepted=Some([1,2,0,0]);
    assert!(localizer::admit(&cfg,&other,Arm::NgramTransaction,&stages).is_err());
    let mut other=plan.clone();other.layers=vec![0,2,3];
    assert!(localizer::admit(&cfg,&other,Arm::NgramTransaction,&stages).is_err());
    let mut other=plan.clone();other.host_payload_limit-=1;
    assert!(localizer::admit(&cfg,&other,Arm::NgramTransaction,&stages).is_err());
    let mut other=plan;other.rerun_id="../x".into();
    assert!(localizer::admit(&cfg,&other,Arm::NgramTransaction,&stages).is_err());
}
#[test]
fn q1_gemma_t5_pass2_layer_union_rule_and_caps(){
    assert_eq!(localizer::pass2_layers(&[Some(0),None,None,None]).unwrap(),[0]);
    assert_eq!(localizer::pass2_layers(&[Some(5),Some(5),None,Some(12)]).unwrap(),[4,5,11,12]);
    assert_eq!(localizer::pass2_layers(&[Some(1),Some(0),Some(2),None]).unwrap(),[0,1,2]);
    assert!(localizer::pass2_layers(&[None;4]).is_err());
    assert!(localizer::pass2_layers(&[Some(30),None,None,None]).is_err());
    let worst=[Some(5),Some(11),Some(17),Some(23)];
    let layers=localizer::pass2_layers(&worst).unwrap();assert_eq!(layers,[4,5,10,11,16,17,22,23]);
    // Worst case (four global layers, ~0.69 GB) stays under every T2Loc quota.
    let cfg=cfg();
    for (arm,accepted,records) in [(Arm::NgramTransaction,Some([0;4]),64_896),(Arm::OrdinaryGlobal,None,63_488)] {
        let (targets,expected,bytes)=localizer::manifest(&cfg,&layers,&PREFILL_STAGES,&DECODE_STAGES,arm,accepted).unwrap();
        assert_eq!(expected.len(),records);assert!(expected.len()<=localizer::RECORD_CAP);
        assert!(targets.len()<=localizer::TARGET_CAP && bytes<=STAGE_PAYLOAD_CAP && bytes<700_000_000,"{bytes}");
    }
    assert!(localizer::manifest(&cfg,&(0..9).collect::<Vec<_>>(),&PREFILL_STAGES,&DECODE_STAGES,Arm::NgramTransaction,Some([0;4])).is_err());
    let p=super::export::plan(&cfg,&pass(Branch::N,2,Some([0;4]),Some(worst)),Arm::NgramTransaction).unwrap();
    localizer::require_pass(&p,2).unwrap();assert!(localizer::require_pass(&p,1).is_err());
    assert!(super::export::plan(&cfg,&pass(Branch::N,2,Some([0;4]),None),Arm::NgramTransaction).is_err());
    assert!(super::export::plan(&cfg,&pass(Branch::N,1,Some([0;4]),Some(worst)),Arm::NgramTransaction).is_err());
}
#[test]
fn q1_gemma_t5_largest_packet_and_manifest_fit_their_reads(){
    let cfg=cfg();
    let worst=pass(Branch::N,2,Some([0;4]),Some([Some(5),Some(11),Some(17),Some(23)]));
    let plan=super::export::plan(&cfg,&worst,Arm::NgramTransaction).unwrap();
    let packet_n=serde_json::to_vec_pretty(&packet(Arm::NgramTransaction,Some(&plan),Branch::N,2)).unwrap();
    assert!(packet_n.len()<=run::PACKET_BYTES,"{}",packet_n.len());
    // Branch P pass 2 additionally binds the All128 precondition record.
    let p_plan=super::export::plan(&cfg,&pass(Branch::P,2,None,Some([Some(5),Some(11),Some(17),Some(23)])),Arm::OrdinaryGlobal).unwrap();
    let packet_p=serde_json::to_vec_pretty(&packet(Arm::OrdinaryGlobal,Some(&p_plan),Branch::P,2)).unwrap();
    assert!(packet_p.len()<=run::PACKET_BYTES,"{}",packet_p.len());
    run::pairing(&extra(serde_json::from_slice(&packet_p).unwrap()),true).unwrap();
    let files=super::export::files(&cfg,Some(&worst)).unwrap();
    assert_eq!(files.iter().map(|f|f.0.as_str()).collect::<Vec<_>>(),["prepared.json","manifest-t5loc-ord-p2.json","manifest-t5loc-ngram-p2.json"]);
    let largest=files.iter().map(|f|f.1.len()).max().unwrap();
    assert!(largest as u64<=run::MANIFEST_BYTES,"{largest}");
    let m:run::Manifest=serde_json::from_slice(&files[2].1).unwrap();
    assert_eq!((m.expected_stores.len(),m.expected_stores[t5::STAGES].len()),(4,64_896));
    let t5_files=super::export::files(&cfg,None).unwrap();
    let ord:run::Manifest=serde_json::from_slice(&t5_files[1].1).unwrap();
    assert_eq!((ord.lanes.clone(),ord.expected_stores.len(),ord.host_payload_limit),
        (vec![0,1,2,3],2,t5::host_payload_limit(&cfg,Arm::OrdinaryGlobal,0).unwrap()));
    let p:serde_json::Value=serde_json::from_slice(&files[0].1).unwrap();
    assert_eq!(p["cases"][1]["packet_extent"]["stage_localizer"],json!(plan));
    assert_eq!(super::export::files(&cfg,Some(&pass(Branch::P,1,None,None))).unwrap().len(),2);
}
