//! Artifact-only Gemma endpoints. Full originals audited BEFORE any projection.
use super::{acquisition::{self,FileBinding},compare,gemma_cache,protocol::{Trajectory,Workload},receipt::*,store::hash};
use anyhow::{ensure,Context,Result};
use serde::Deserialize;
use serde_json::{json,Value};
use std::{collections::{BTreeMap,BTreeSet},fs::OpenOptions,io::{Read,Write},os::unix::fs::OpenOptionsExt,path::{Path,PathBuf}};
#[path="gemma_artifact_compare/selected.rs"] mod selected;
#[path="gemma_artifact_compare/audit.rs"] mod audit;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    trajectory:Trajectory,trace:FileBinding,execution:FileBinding,runtime_packet:FileBinding,
    composition:FileBinding,acquisition_review:FileBinding,supervisor_receipt:FileBinding,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    schema:String,producer_identity:Identity,analyzer_source:String,analyzer_binary_sha256:String,
    prepared:FileBinding,workload:FileBinding,left:Case,right:Case,output:PathBuf,
}
fn bytes(b:&FileBinding,max:u64)->Result<Vec<u8>> {
    ensure!(b.path.canonicalize()?==b.path,"canonical bound evidence");
    acquisition::checked_file(b,max)
}
fn value(b:&FileBinding)->Result<Value> {Ok(serde_json::from_slice(&bytes(b,2*1024*1024)?)?)}
fn load(b:&FileBinding)->Result<(PathBuf,Trace)> {
    let root=b.path.parent().context("trace root")?.to_path_buf();
    ensure!(root.canonicalize()?==root,"canonical original root");
    Ok((root,serde_json::from_slice(&bytes(b,128*1024*1024)?)?))
}
fn output_preflight(p:&Path)->Result<()> {
    ensure!(p.is_absolute() && p.parent().context("output parent")?.canonicalize()?==p.parent().unwrap()
        && !p.ancestors().any(|p|p.join(".git").exists()),"private output outside Git");
    match std::fs::symlink_metadata(p) {Err(e) if e.kind()==std::io::ErrorKind::NotFound=>Ok(()),
        _=>anyhow::bail!("create_new output only")}
}
fn expected(cfg:&crate::serve::config::Gemma4Config,t:Trajectory)->Result<Vec<ExpectedTensor>> {
    ensure!(matches!(t,Trajectory::T0ColdSerial|Trajectory::T1Resume64|Trajectory::T2ActualCohort|
        Trajectory::T3Chunk64|Trajectory::T6CancelRestore),"only explicit actual Gemma endpoint axes");
    let mut out=gemma_cache::expected(cfg,&[0,1,2,3],&super::gemma_actual::positions(t))
        .map_err(anyhow::Error::msg)?;
    for lane in 0..4 {for row in 0..144 {out.push(ExpectedTensor {at:Coordinate {
        logical_step:0,layer:u32::MAX,stage:Stage::Logits,agent:format!("agent-{lane}"),
        logical_row_start:row,segment:0},scalar:Scalar::F32,shape:vec![1,cfg.vocab_size]});}}
    Ok(out)
}
fn selected_config(p:&Value,identity:&Identity)->Result<crate::serve::config::Gemma4Config> {
    let cfg=selected::cfg();let recipe=super::gemma_actual::recipe(&cfg);
    ensure!(p["schema"]=="hf2q.q1.gemma-endpoint-preparation.v1"
        && p["actual_gemma_config_debug"]==format!("{cfg:?}")
        && p["actual_gemma_config_debug_sha256"]==recipe.actual_model_config_sha256
        && p["recipe"]==serde_json::to_value(&recipe)?
        && p["config_sha256"]==acquisition::config_digest(&recipe)?
        && p["config_sha256"]==identity.config_sha256
        && p["exporter_source_commit"]==identity.source_commit
        && p["exporter_binary_sha256"]==identity.binary_sha256
        && p["native_version"]==identity.native_version
        && p["model_full_sha256_retained"]==identity.model_sha256
        && p["tokenizer_file_sha256"]==identity.tokenizer_sha256
        && p["template_file_sha256"]==identity.template_sha256
        && p["token_vectors_sha256"]==identity.token_vectors_sha256,
        "full actual cfg Debug/recipe/producer identities must match BEFORE static manifest use");
    Ok(cfg)
}
fn runtime(case:&Case,root:&Path,trace:&Trace,id:&Identity,prep:&Value,work:&Workload)->Result<()> {
    ensure!(trace.identity==*id && trace.comparison_key==acquisition::ComparisonKey::from(id)
        && trace.coverage.trajectory==case.trajectory.label(),"original immutable identity/trajectory");
    let a=trace.acquisition.as_ref().context("runtime acquisition absent")?;
    ensure!(a.provenance=="IndependentRootReviewedComposition" && a.required_model_opens==3
        && a.observed_model_opens>=3 && a.actual_binary_sha256==id.binary_sha256 && a.output_root==root
        && serde_json::to_value(&a.opened_model)?==prep["model_file_identity"],"actual dispatcher3/FD/source/root");
    ensure!(case.execution.path==root.join("execution.json") && case.runtime_packet.path==root.parent().unwrap().join("packet.json")
        && case.supervisor_receipt.path==root.parent().unwrap().join("receipt.json"),"original runtime file/root joins");
    let ex=value(&case.execution)?;let packet=value(&case.runtime_packet)?;
    let composition=value(&case.composition)?;let review=value(&case.acquisition_review)?;let receipt=value(&case.supervisor_receipt)?;
    ensure!(a.packet_sha256==case.runtime_packet.sha256 && ex["packet_sha256"]==case.runtime_packet.sha256
        && a.composition_sha256==case.composition.sha256 && a.independent_review_sha256==case.acquisition_review.sha256
        && composition["schema"]=="hf2q.q1.acquisition-composition.v1"
        && composition["comparison_key"]==serde_json::to_value(acquisition::ComparisonKey::from(id))?
        && composition["model_file"]==prep["model_file_identity"]
        && composition["full_content_receipt"]["path"]==prep["full_content_receipt_path"]
        && composition["full_content_receipt"]["sha256"]==prep["full_content_receipt_sha256"]
        && review==json!({"schema":"hf2q.q1.acquisition-review.v1","verdict":"Accepted","composition_sha256":case.composition.sha256}),
        "original packet/composition/independent review binding");
    let full:FileBinding=serde_json::from_value(composition["full_content_receipt"].clone())?;
    acquisition::full_content_receipt_has_model(&bytes(&full,16*1024*1024)?,&id.model_sha256)?;
    ensure!(packet["execution_approved"]==true && packet["identity"]==serde_json::to_value(id)?
        && packet["workload"]==serde_json::to_value(work)? && packet["trajectory"]==serde_json::to_value(case.trajectory)?
        && packet["model"]==prep["model_path"]
        && packet["configured_context"]==262144 && packet["configured_output_allowance"]==8192
        && packet["output"].as_str()==root.to_str()
        && packet["acquisition"]["composition"]==json!({"path":case.composition.path,"sha256":case.composition.sha256})
        && packet["acquisition"]["independent_review"]==json!({"path":case.acquisition_review.path,"sha256":case.acquisition_review.sha256}),
        "actual runtime packet unchanged family/config/workload/acquisition");
    ensure!(receipt["case_complete"]==true && receipt["exit"]==0
        && receipt["cleanup"]["reaped"]==true && receipt["cleanup"]["group_empty"]==true
        && receipt["cleanup"]["errors"]==json!([]) && receipt.get("failure").is_none()
        && receipt.get("post_failure").is_none()
        && ex["trajectory"]==serde_json::to_value(case.trajectory)? && ex["logical_lanes"]==4
        && ex["logit_rows_per_lane"]==144 && ex["whole_q1_qualified"]==false
        && ex["capture_records"]==trace.coverage.captured.len()
        && ex["planned_tensor_payload"].as_u64()==trace.coverage.captured.iter()
            .try_fold(0u64,|n,t|n.checked_add(t.logical_bytes)),"actual completed original runtime/full144 scope");
    if matches!(case.trajectory,Trajectory::T1Resume64|Trajectory::T6CancelRestore) {
        ensure!(ex["active64_exact_copy"]["Ok"].as_u64().is_some_and(|v|v>0),"runtime exact64 gate was not successful");
    }
    if case.trajectory==Trajectory::T6CancelRestore {
        let all=ex["actual_cancellation"].as_array().context("four owning cancellations")?;
        ensure!(all.len()==4 && all.iter().filter_map(|v|v["slot"].as_u64()).collect::<BTreeSet<_>>()==BTreeSet::from([0,1,2,3])
            && all.iter().all(|v|v["real_receiver_closed_after_checked_advance"]==true
                && v["checked_prefix"]==64 && v["actual_unpublished_end"]==128 && v["recovered_cursor"]==64
                && v["retained_tokens"]==64 && v["scheduler_in_flight_after"]==0 && v["supervisor_healthy"]==true),
            "actual four-owner cancellation recovery");
    }
    Ok(())
}
fn project(trace:&Trace,common:&[ExpectedTensor])->Result<Trace> {
    let mut out=trace.clone();out.coverage.expected=common.to_vec();
    let map=trace.coverage.captured.iter().map(|t|(t.at.clone(),t)).collect::<BTreeMap<_,_>>();
    out.coverage.captured=common.iter().map(|e|map.get(&e.at).map(|t|(**t).clone())
        .context("common original coordinate missing")).collect::<Result<_>>()?;
    Ok(out)
}
fn audited_pair(lroot:&Path,l:&Trace,le:&[ExpectedTensor],rroot:&Path,r:&Trace,re:&[ExpectedTensor],
    common:&[ExpectedTensor])->Result<(Value,Value,Trace,Trace)> {
    let la=audit::original(lroot,l,le)?;let ra=audit::original(rroot,r,re)?;
    // Both originals, including all extras, MUST finish audit before either projection.
    Ok((la,ra,project(l,common)?,project(r,common)?))
}
fn run(p:Packet)->Result<()> {
    ensure!(p.schema=="hf2q.q1.gemma-artifact-comparison.v1","closed helper packet schema");output_preflight(&p.output)?;
    p.producer_identity.validate().map_err(anyhow::Error::msg)?;
    ensure!(p.producer_identity.family==Family::Gemma4
        && p.producer_identity.source_commit=="190abe5b7e5c790090fa48a2c3d0db82127e1e17"
        && p.producer_identity.binary_sha256=="bd2ed0f6ea79cc302b5df7065a44e7924d9a2f780dda30d2cbfecf91e4b2a204"
        && p.producer_identity.model_sha256=="bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53"
        && p.producer_identity.native_version=="0.15.8"
        && p.producer_identity.native_version==env!("HF2Q_MLX_NATIVE_VERSION")
        && p.producer_identity.native_checksum=="2ac4d45c9de8707f55cc0474e26f09720e3afa886b37c5617c41007353380fe7"
        && option_env!("HF2Q_BUILD_GIT_SHA")==Some(p.analyzer_source.as_str())
        && acquisition::executable_digest()?==p.analyzer_binary_sha256,"separate exact analyzer and native158 producer binding");
    let prepared=value(&p.prepared)?;let cfg=selected_config(&prepared,&p.producer_identity)?;
    let work:Workload=serde_json::from_slice(&bytes(&p.workload,1024*1024)?)?;work.validate().map_err(anyhow::Error::msg)?;
    ensure!(hash(&bytes(&p.workload,1024*1024)?)==prepared["workload_file_sha256"]
        && work.family==Family::Gemma4 && work.vocab_size==cfg.vocab_size as u32
        && acquisition::tokens_digest(&work)?==p.producer_identity.token_vectors_sha256,"one actual frozen144 workload");
    let (lr,left)=load(&p.left.trace)?;let (rr,right)=load(&p.right.trace)?;
    runtime(&p.left,&lr,&left,&p.producer_identity,&prepared,&work)?;
    runtime(&p.right,&rr,&right,&p.producer_identity,&prepared,&work)?;
    ensure!(lr!=rr && acquisition::validate_pair_for(&lr,&left,&rr,&right,false).map_err(anyhow::Error::msg)?,
        "distinct independently observed actual acquisitions; no same-root/nonce spoof");
    ensure!(left.acquisition.as_ref().unwrap().device_registry_id==right.acquisition.as_ref().unwrap().device_registry_id,
        "same actual physical device registry required; queues remain original independent observations");
    let common=expected(&cfg,Trajectory::T0ColdSerial)?;ensure!(common.len()==5616,"complete selected5616 common coords");
    let le=expected(&cfg,p.left.trajectory)?;let re=expected(&cfg,p.right.trajectory)?;
    let (la,ra,l,r)=audited_pair(&lr,&left,&le,&rr,&right,&re,&common)?;
    let restore=|case:&Case,root:&Path,t:&Trace|->Result<Value> {
        if matches!(case.trajectory,Trajectory::T1Resume64|Trajectory::T6CancelRestore) {
            let result=audit::restore64(root,t)?;
            ensure!(result["actual_bytes_equal"]==value(&case.execution)?["active64_exact_copy"]["Ok"],
                "independently checked exact64 byte count matches original runtime gate");Ok(result)
        }
        else {Ok(Value::Null)}
    };
    let lexact=restore(&p.left,&lr,&left)?;let rexact=restore(&p.right,&rr,&right)?;
    let core=compare::compare(&lr,&l,&rr,&r,&BTreeMap::new());
    let invalid=matches!(&core.disposition,Disposition::InvalidEvidence|Disposition::IdentityMismatch|Disposition::LocalizationIncomplete);
    let report=json!({"schema":"hf2q.q1.gemma-artifact-comparison-result.v1",
        "producer_identity":p.producer_identity,"analyzer_source":p.analyzer_source,
        "analyzer_binary_sha256":p.analyzer_binary_sha256,"analyzer_native_version":env!("HF2Q_MLX_NATIVE_VERSION"),
        "left_original_audit":la,"right_original_audit":ra,"left_exact64":lexact,"right_exact64":rexact,
        "declared_projection":{"kind":"in_memory_common128144_cache_plus_all144_logits",
            "common_coordinates":common.len(),"left_original_coordinates":le.len(),"right_original_coordinates":re.len(),
            "left_noncommon_audited_not_compared":le.len()-common.len(),"right_noncommon_audited_not_compared":re.len()-common.len(),
            "original_files_rewritten":false,"first_difference_is_manifest_order_not_causal":true},
        "core_disposition":core.disposition,"first_difference":core.first_difference,"compared_bytes":core.compared_bytes,
        "compared_elements":core.compared_elements,"max_abs_floating_only":core.max_abs,"rms_floating_only":core.rms,
        "relative_rms_floating_only":core.relative_rms,"cosine_floating_only":core.cosine,
        "nonfinite_elements":core.nonfinite_elements,"reason":core.reason,
        "runtime_acquisition_qualified":core.runtime_acquisition_qualified,"oracle_qualified":false,
        "captured_extent_qualified":false,"whole_q1_qualified":false,
        "packed_u8_semantics":"floating_codebook_storage: encoded differences retained, no decoded metric/policy",
        "qualification_scope":"diagnostic metrics/disposition preserved; no routing/full-matrix/tolerance qualification"});
    let data=serde_json::to_vec_pretty(&report)?;ensure!(data.len()<=2*1024*1024,"bounded report");
    output_preflight(&p.output)?;OpenOptions::new().create_new(true).write(true).mode(0o600).open(&p.output)?.write_all(&data)?;
    ensure!(!invalid,"invalid core disposition retained in report; no complete marker");
    println!("Q1_GEMMA_ARTIFACT_COMPARISON_COMPLETE common=5616 oracle_qualified=false");Ok(())
}
#[test]
#[ignore="artifact-only complete original validation; no model/device"]
fn q1_gemma_complete_endpoint_artifact_comparison() {
    let path=std::env::var_os("HF2Q_Q1_GEMMA_COMPARE_PACKET").expect("explicit comparison packet");
    let mut raw=Vec::new();std::fs::File::open(path).unwrap().take(1024*1024+1).read_to_end(&mut raw).unwrap();assert!(raw.len()<=1024*1024);
    run(serde_json::from_slice(&raw).unwrap()).unwrap();
}
#[path="gemma_artifact_compare/controls.rs"] mod controls;
