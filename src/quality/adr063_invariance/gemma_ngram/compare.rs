//! Offline Gemma T5 committed-store comparison (design §2.4, cache@144 row):
//! ordinary-global `committed` versus n-gram `committed`, element-wise with
//! `NumericPolicy::Unspecified` for every stage. No model or device. The
//! Class B decision of record for the 64 Logits rows is `envelope`.
use super::t5::{self,Arm};
use crate::adr063_invariance::{acquisition::{self,FileBinding},compare,protocol::Workload,receipt::*};
use anyhow::{ensure,Context,Result};
use serde::Deserialize;
use serde_json::{json,Value};
use std::{fs::OpenOptions,io::Write,os::unix::fs::OpenOptionsExt,path::{Path,PathBuf}};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    schema:String,identity:Identity,greedy_workload:FileBinding,
    ordinary_committed:FileBinding,ngram_committed:FileBinding,
    ordinary_execution:FileBinding,ngram_execution:FileBinding,output:PathBuf,
}
fn load(binding:&FileBinding)->Result<(PathBuf,Trace)> {
    let root=binding.path.parent().context("trace parent")?.to_path_buf();
    ensure!(root.canonicalize()?==root && root.file_name().and_then(|n|n.to_str())==Some(t5::COMMITTED),"canonical committed store root");
    Ok((root,serde_json::from_slice(&acquisition::checked_file(binding,128*1024*1024)?)?))
}
fn manifest(trace:&Trace,identity:&Identity,expected:&[ExpectedTensor],arm:Arm,root:&Path)->Result<()> {
    ensure!(trace.schema=="hf2q.adr063.invariance.v2" && trace.identity==*identity &&
        trace.coverage.trajectory==t5::trajectory(arm,t5::COMMITTED) && trace.coverage.expected==expected &&
        trace.coverage.captured.len()==expected.len() && trace.coverage.errors.is_empty(),"complete Gemma T5 committed manifest");
    let a=trace.acquisition.as_ref().context("actual acquisition missing")?;
    ensure!(a.required_model_opens==3 && a.observed_model_opens>=3 && a.actual_binary_sha256==identity.binary_sha256 &&
        a.output_root==root,"Gemma T5 three-open acquisition/root binding");
    Ok(())
}
/// Receipt join: completed phase of this arm, teacher-forced frozen inputs on
/// every decode row, and the n-gram transaction shape.
pub(super) fn execution(v:&Value,arm:Arm,frozen:&[Vec<u32>])->Result<Vec<u32>> {
    ensure!(v["schema"]==t5::SCHEMA && v["arm"]==arm.label() && v["status"]=="CompleteScopedAcquisition" && v["error"].is_null() &&
        v["class_b_refusal"].is_null() && v["whole_q1_qualified"]==false && v["lanes"]==json!([0,1,2,3]),"completed Gemma T5 phase receipt");
    let d=v["decisions"].as_array().context("decisions")?;ensure!(d.len()==4 && frozen.len()==4,"four lanes");
    let mut initial=Vec::new();
    for (lane,l) in d.iter().enumerate() {
        ensure!(l["lane"]==lane && l["frozen_128"]==frozen[lane][128],"lane/frozen join");
        let accepted=match arm {
            Arm::OrdinaryGlobal=>{ensure!(l["transaction"].is_null(),"ordinary arm has no transaction");None}
            Arm::NgramTransaction=>{let x=&l["transaction"];
                let a=x["accepted"].as_u64().context("accepted")? as usize;t5::require_accepted(a)?;
                let fixture=super::fixture(&frozen[lane],262144)?;
                ensure!(x["proposer_ran"]==true && x["injected"]==json!(fixture) && x["injected_origin"]==t5::INJECTED_ORIGIN &&
                    x["rejected"]==3-a && x["rollback_cursor"]==129+a && x["verifier_argmax"].as_array().is_some_and(|r|r.len()==4) &&
                    x["fallback"]==x["verifier_argmax"][a],"injected K3 transaction shape");
                Some(a)}
        };
        let inputs=t5::decode_inputs(arm,&frozen[lane],accepted)?;
        ensure!(l["decode_inputs"]==json!(inputs.iter().map(|&(position,token)|json!({"position":position,"token":token})).collect::<Vec<_>>()),
            "teacher-forced decode inputs are the frozen lineage");
        ensure!(l["decode_argmax"].as_array().is_some_and(|r|r.len()==inputs.len()),"one decision per decode row");
        initial.push(u32::try_from(l["initial_argmax"].as_u64().context("initial argmax")?)?);
    }
    Ok(initial)
}
fn output(path:&Path,bytes:&[u8])->Result<()> {
    let parent=path.parent().context("output parent")?;
    ensure!(path.is_absolute() && parent.canonicalize()?==parent && !path.ancestors().any(|p|p.join(".git").exists()),"private canonical output outside Git");
    ensure!(bytes.len()<=1024*1024,"bounded report");
    OpenOptions::new().write(true).create_new(true).mode(0o600).open(path)?.write_all(bytes)?;Ok(())
}
fn run(p:Packet)->Result<()> {
    ensure!(p.schema=="hf2q.q1.gemma-t5-artifact-comparison.v1" && p.identity.family==Family::Gemma4,"explicit Gemma T5 artifact packet");
    p.identity.validate().map_err(anyhow::Error::msg)?;
    // Header-fixture geometry only fixes the manifest extent; identity equality
    // of both stores with the packet (and compare's key check) binds the config.
    let cfg=crate::adr063_invariance::gemma_actual::controls::cfg();
    let work:Workload=serde_json::from_slice(&acquisition::checked_file(&p.greedy_workload,1024*1024)?)?;
    work.validate().map_err(anyhow::Error::msg)?;
    ensure!(work.family==Family::Gemma4 && acquisition::tokens_digest(&work)?==p.identity.token_vectors_sha256,"frozen greedy144 digest");
    let expected=t5::committed(&cfg).map_err(anyhow::Error::msg)?;
    let (or,o)=load(&p.ordinary_committed)?;let (nr,n)=load(&p.ngram_committed)?;
    manifest(&o,&p.identity,&expected,Arm::OrdinaryGlobal,&or)?;manifest(&n,&p.identity,&expected,Arm::NgramTransaction,&nr)?;
    let oe:Value=serde_json::from_slice(&acquisition::checked_file(&p.ordinary_execution,2*1024*1024)?)?;
    let ne:Value=serde_json::from_slice(&acquisition::checked_file(&p.ngram_execution,2*1024*1024)?)?;
    let oi=execution(&oe,Arm::OrdinaryGlobal,&work.lanes)?;let ni=execution(&ne,Arm::NgramTransaction,&work.lanes)?;
    ensure!(oe["packet_sha256"]==o.acquisition.as_ref().unwrap().packet_sha256 && ne["packet_sha256"]==n.acquisition.as_ref().unwrap().packet_sha256,
        "committed store / phase receipt acquisition join");
    let policies=expected.iter().map(|e|(e.at.stage,NumericPolicy::Unspecified)).collect();
    let committed=compare::compare(&or,&o,&nr,&n,&policies);
    ensure!(!matches!(committed.disposition,Disposition::InvalidEvidence|Disposition::IdentityMismatch|Disposition::LocalizationIncomplete),
        "invalid committed evidence: {committed:?}");
    output(&p.output,&serde_json::to_vec_pretty(&json!({"schema":"hf2q.q1.gemma-t5-artifact-report.v1","identity":p.identity,
        "committed":committed,"logical_lanes":4,"committed_logits_rows":[128,144],"cache_boundary":144,"records":expected.len(),
        "initial_argmax_ordinary":oi,"initial_argmax_ngram":ni,"initial_argmax_equal":oi==ni,
        "float_policy":"Unspecified","float_disposition":"ExplicitInconclusiveNumericPolicy",
        "route_class":"ClassB: OptionC 132-row re-prefill verifier vs M1 decode; Logits rows decided by envelope, not here",
        "whole_q1_qualified":false,"performance_qualified":false,"physical_admission":false}))?)?;
    println!("Q1_GEMMA_T5_ARTIFACT_COMPARE_COMPLETE lanes=4 committed_records={} float_policy=Unspecified whole_q1=false",expected.len());
    Ok(())
}
#[test]
#[ignore="explicit retained Gemma T5 artifact packet only; no model/device; root supervised"]
fn q1_gemma_t5_artifact_compare() {
    let p=PathBuf::from(std::env::var_os("HF2Q_Q1_GEMMA_T5_COMPARE_PACKET").expect("explicit packet"));
    let m=std::fs::symlink_metadata(&p).unwrap();
    assert!(p.is_absolute() && m.is_file() && !m.file_type().is_symlink() && m.len()<=1024*1024);
    run(serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()).unwrap();
}
