//! MODEL-ASSISTED prefix128 + sixteen actual ordinary GPU selections.
//! Preparation provenance is intentionally not a comparable Q1 acquisition.
use super::{acquisition::{self,FileBinding},gemma_endpoint as endpoint,protocol::Workload,
    receipt::*,store::{hash,Store}};
use anyhow::{ensure,Context,Result};
use serde::{Deserialize,Serialize};
use std::{fs::OpenOptions,io::{Read,Write},os::unix::fs::OpenOptionsExt,path::{Path,PathBuf}};
use crate::serve::{api::engine::{LoadOptions,LoadedModel},multi_seq_kv::SlotId};
const MODEL:&str="bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53";
const VOCAB:usize=262144;
const PAYLOAD:u64=4*17*VOCAB as u64*4;
const EOS:[u32;2]=[1,106]; // actual selected Gemma ordinary CLI stop IDs
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    execution_approved:bool,acquisition:acquisition::Binding,identity:Identity,
    original_workload:FileBinding,model:PathBuf,output:PathBuf,
    configured_context:u32,configured_output_allowance:u32,speculation:String,
    eos_ids:Vec<u32>,equal_max_policy:String,payload_limit:u64,
}
#[derive(Serialize)]
struct Decision {lane:usize,input_position:usize,gpu_selected:u32,unique_max:bool,
    eos:bool,row_sha256:String}
fn output_preflight(p:&Path)->Result<()> {
    ensure!(p.is_absolute() && !p.ancestors().any(|p|p.join(".git").exists()),"private output outsideGit");
    ensure!(p.parent().context("output parent")?.canonicalize()?==p.parent().unwrap(),"canonical parent");
    ensure!(matches!(std::fs::symlink_metadata(p),Err(e) if e.kind()==std::io::ErrorKind::NotFound),"exclusive new output");Ok(())
}
fn prefixes(work:&Workload)->Result<Vec<Vec<u32>>> {
    work.validate().map_err(anyhow::Error::msg)?;
    ensure!(work.family==Family::Gemma4 && work.vocab_size as usize==VOCAB,"selected original Gemma workload");
    Ok(work.lanes.iter().map(|v|v[..128].to_vec()).collect())
}
fn preflight(p:&Packet)->Result<()> {
    p.identity.validate().map_err(anyhow::Error::msg)?;output_preflight(&p.output)?;
    ensure!(p.execution_approved && p.identity.family==Family::Gemma4 && p.identity.model_sha256==MODEL,
        "approved selected Gemma preparation only");
    ensure!(p.configured_context==262144 && p.configured_output_allowance==8192 &&
        p.speculation=="off" && p.eos_ids==EOS && p.equal_max_policy=="ineligible_on_multiple_finite_maxima",
        "frozen ordinary profile/EOS/ambiguity policy");
    ensure!(p.payload_limit==PAYLOAD && PAYLOAD<=72*1024*1024,"exact68MiB row payload");
    for name in ["HF2Q_SPEC_EAGLE3","HF2Q_SPEC_NGRAM","HF2Q_SPEC_DFLASH"] {
        ensure!(std::env::var(name).ok().is_none_or(|v|v=="0"),"ambient speculative activation: {name}");
    }Ok(())
}
/// Checks the ACTUAL GPU choice; never supplies a CPU-selected input token.
fn inspect(values:&[f32],selected:u32)->Result<(bool,bool)> {
    ensure!(values.len()==VOCAB && (selected as usize)<values.len(),"full selected vocabulary/ID");
    ensure!(values.iter().all(|v|v.is_finite()),"nonfinite ordinary logits");
    let maximum=values.iter().copied().fold(f32::NEG_INFINITY,f32::max);
    ensure!(values[selected as usize]==maximum,"GPU selection disagrees with full-row maximum");
    let unique=values.iter().filter(|&&v|v==maximum).count()==1;
    Ok((unique,EOS.contains(&selected)))
}
fn require_canonical(decision:&Decision)->Result<()> {
    ensure!(decision.unique_max && !decision.eos,"IneligibleGreedyPreparation: ambiguous maximum or EOS among sixteen canonical IDs");Ok(())
}
fn require_chain(prompts:&[Vec<u32>],lanes:&[Vec<u32>],decisions:&[Decision],carry:&[u32])->Result<()> {
    ensure!(prompts.len()==4 && lanes.len()==4 && decisions.len()==68 && carry.len()==4,"complete four-lane chain");
    for lane in 0..4 {
        ensure!(prompts[lane].len()==128 && lanes[lane].len()==144 && lanes[lane][..128]==prompts[lane],"unchanged prefix128/full144");
        for j in 0..17 {
            let d=&decisions[lane*17+j];
            ensure!(d.lane==lane && d.input_position==127+j,"observed row chronology/ownership");
            if j<16 {require_canonical(d)?;ensure!(lanes[lane][128+j]==d.gpu_selected,"canonical token/logit edge mismatch");}
            else {ensure!(carry[lane]==d.gpu_selected,"final g16 carry mismatch");}
        }
    }Ok(())
}
fn save(path:&Path,value:&impl Serialize)->Result<()> {
    let bytes=serde_json::to_vec_pretty(value)?;ensure!(bytes.len()<=1024*1024,"separate1MiB metadata bound");
    let mut f=OpenOptions::new().write(true).create_new(true).mode(0o600).open(path)?;f.write_all(&bytes)?;Ok(())
}
fn decision(lane:usize,pos:usize,id:u32,values:&[f32])->Result<Decision> {
    let (unique_max,eos)=inspect(values,id)?;
    let bytes:Vec<u8>=values.iter().flat_map(|v|v.to_le_bytes()).collect();
    Ok(Decision {lane,input_position:pos,gpu_selected:id,unique_max,eos,row_sha256:hash(&bytes)})
}
#[test]
#[ignore="root-only MODEL-ASSISTED Gemma greedy preparation; separate reviewed operating packet"]
fn q1_gemma_model_assisted_greedy144_prepare(){run().unwrap();}
fn run()->Result<()> {
    let path=std::env::var_os("HF2Q_Q1_GEMMA_GREEDY_PACKET").context("explicit preparation packet")?;
    let mut raw=Vec::new();std::fs::File::open(path)?.take(1024*1024+1).read_to_end(&mut raw)?;
    ensure!(raw.len()<=1024*1024,"packet quota");let p:Packet=serde_json::from_slice(&raw)?;preflight(&p)?;
    let original=acquisition::checked_file(&p.original_workload,1024*1024)?;
    let original:Workload=serde_json::from_slice(&original)?;let prompts=prefixes(&original)?;
    let scope=acquisition::Scope::begin_preparation(&p.acquisition,&p.identity,&prompts,VOCAB as u32,&p.model,&p.output,&raw)?;
    let header=acquisition::open_model(&p.model)?;
    ensure!(header.metadata_string("general.architecture")==Some("gemma4"),"explicit Gemma header");
    let mut cfg=crate::serve::config::Gemma4Config::from_gguf(&header)?;cfg.max_position_embeddings=262144;
    ensure!(cfg.vocab_size==VOCAB,"selected vocabulary");scope.check_config(&p.identity,&super::gemma_actual::recipe(&cfg))?;
    drop(header);
    let expected=(0..4).flat_map(|lane|(127..144).map(move|pos|ExpectedTensor {
        at:Coordinate {logical_step:0,layer:u32::MAX,stage:Stage::Logits,agent:format!("agent-{lane}"),
            logical_row_start:pos,segment:0},scalar:Scalar::F32,shape:vec![1,VOCAB]})).collect();
    let mut store=Store::new(&p.output,p.identity.clone(),"ModelAssistedGreedyPreparationNotT5".into(),expected,PAYLOAD,68)
        .map_err(anyhow::Error::msg)?;
    let _gpu=crate::inference::hf2q_gpu_test_lock();
    let opts=LoadOptions {model_path:p.model.clone(),tokenizer_path:None,config_path:None,dwq_overlay_path:None,
        kv_persist_dir:None,kv_persist_budget_bytes:0,glp_path:None,glp_alpha:None,kv_graft_path:None};
    let LoadedModel::Gemma(mut model)=LoadedModel::load_with_context(&opts,Some(262144))?
        else {anyhow::bail!("selected Gemma load")};
    scope.check_config(&p.identity,&super::gemma_actual::recipe(&model.config))?;
    ensure!(model.eos_token_ids==p.eos_ids,"actual loaded EOS set differs from frozen preparation");
    let observed=scope.observed_preparation(model.ctx.executor.device(),&model as *const _ as usize as u64,3)?;
    store.bind_acquisition(observed).map_err(anyhow::Error::msg)?;
    model.provision_multi_seq_kv_for_slot_aware(4)?;
    ensure!(model.multi_seq_kv_hybrid.is_some() && model.multi_seq_kv_dense.is_none() && model.multi_seq_kv_mlx.is_none(),"selected Hybrid only");
    let _capture=endpoint::install_preparation(store).map_err(anyhow::Error::msg)?;
    let mut lanes=prompts.clone();let mut decisions=Vec::new();let mut carries=Vec::new();let mut forwards=0;
    let mut result=(||->Result<()> {
        for lane in 0..4 {
            endpoint::begin_call(lane,0,128).map_err(anyhow::Error::msg)?;
            let mut next=model.weights.forward_prefill_with_soft_tokens_slot_aware(&prompts[lane],&[],8192,&mut model.ctx,
                SlotId(lane as u32),model.multi_seq_kv.as_mut().context("slot scaffold")?,model.multi_seq_kv_hybrid.as_mut(),
                model.multi_seq_kv_dense.as_mut(),model.multi_seq_kv_mlx.as_mut())?;forwards+=1;
            crate::serve::api::engine::q1_gemma_cohort::commit_after_checked_return(&mut model,SlotId(lane as u32),0,128)?;
            endpoint::end_call().map_err(anyhow::Error::msg)?;
            decisions.push(decision(lane,127,next,model.weights.activations.logits.as_logical_slice::<f32>()?)?);
            for pos in 128..144 {
                require_canonical(decisions.last().unwrap())?;lanes[lane].push(next);
                endpoint::begin_call(lane,pos,1).map_err(anyhow::Error::msg)?;
                next=model.weights.forward_decode_slot_aware(next,pos,&mut model.ctx,&mut None,SlotId(lane as u32),
                    model.multi_seq_kv.as_mut().context("slot scaffold")?,model.multi_seq_kv_hybrid.as_mut(),
                    model.multi_seq_kv_dense.as_mut(),model.multi_seq_kv_mlx.as_mut())?;forwards+=1;
                crate::serve::api::engine::q1_gemma_cohort::commit_after_checked_return(&mut model,SlotId(lane as u32),pos,pos+1)?;
                endpoint::logits(0,&model.weights.activations.logits,VOCAB).map_err(anyhow::Error::msg)?;
                endpoint::end_call().map_err(anyhow::Error::msg)?;
                decisions.push(decision(lane,pos,next,model.weights.activations.logits.as_logical_slice::<f32>()?)?);
            }carries.push(next);
        }Ok(())
    })();
    if result.is_ok() {result=require_chain(&prompts,&lanes,&decisions,&carries);}
    if let Err(ref e)=result {endpoint::with_store(|s|{s.fail(e.to_string());Ok(())}).map_err(anyhow::Error::msg)?;}
    scope.check_unchanged()?;
    let trace=endpoint::finish().map_err(anyhow::Error::msg)?;
    ensure!(std::fs::metadata(p.output.join("trace.json"))?.len()<=1024*1024,"preparation trace metadata1MiB");
    let eligible=result.is_ok() && trace.coverage.errors.is_empty() && forwards==68 && decisions.len()==68;
    let workload=Workload {family:Family::Gemma4,vocab_size:VOCAB as u32,lanes};
    save(&p.output.join("preparation.json"),&serde_json::json!({"schema":"hf2q.q1.gemma-model-assisted-greedy144.v1",
        "model_assisted":true,"eligible_pending_independent_review":eligible,"final_dataset_approved":false,
        "source_workload":p.original_workload.sha256,"packet_sha256":hash(&raw),"decisions":decisions,
        "actual_checked_forward_count":forwards,
        "proposed_workload_token_vectors_sha256":if eligible {Some(acquisition::tokens_digest(&workload)?)}else{None},
        "proposed_workload_json_sha256":if eligible {Some(hash(&serde_json::to_vec_pretty(&workload)?))}else{None},"full_logit_payload_bytes":PAYLOAD,"carry_g16":carries,
        "failure":result.as_ref().err().map(ToString::to_string),"speculation":"off","eos_ids":EOS,
        "equal_max_policy":p.equal_max_policy,"physical_admission":false,"t5_qualified":false}))?;
    result?;ensure!(eligible,"incomplete greedy preparation");workload.validate().map_err(anyhow::Error::msg)?;
    save(&p.output.join("workload144-proposed.json"),&workload)?;
    println!("Q1_GEMMA_MODEL_ASSISTED_GREEDY144_COMPLETE lanes=4 forwards=68 rows=68 payload=71303168 final_dataset_approved=false");Ok(())
}
#[cfg(test)]
#[path="gemma_greedy_prepare_controls.rs"]
mod controls;
