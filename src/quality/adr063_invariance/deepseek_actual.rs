//! Root-run-only complete ordinary DS endpoints. No owning-cancellation or
//! internal-stage/task/F2 qualification is inferred from this producer.
use super::{capture,deepseek_cache::endpoint as cache_capture,deepseek_runner::{self,Observation},
    protocol::{Trajectory,Workload},receipt::*,store::{hash,Store}};
use crate::inference::models::deepseek4::{Deepseek4Config,cache::Deepseek4CachePlan};
use anyhow::{ensure,Context,Result};
use serde::Deserialize;
use std::{fs::OpenOptions,io::{Read,Write},os::unix::fs::OpenOptionsExt,path::PathBuf};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    execution_approved:bool,acquisition:super::acquisition::Binding,model:PathBuf,output:PathBuf,
    identity:Identity,workload:Workload,trajectory:Trajectory,configured_capacity:usize,
    configured_output_allowance:u64,requested_payload_limit:u64,
}
pub(super) fn require_ordinary(t:Trajectory)->Result<()> {
    ensure!(!matches!(t,Trajectory::T5GenuineSpeculation|Trajectory::T6CancelRestore),
        "LocalizationIncomplete: DS genuine T5 producer and owning-worker T6 need separate actual adapters");Ok(())
}
fn boundaries(t:Trajectory)->Vec<(usize,bool)> {
    let mut out=vec![(128,false),(144,false)];
    if matches!(t,Trajectory::T1Resume64|Trajectory::T3Chunk64){out.push((64,false));}
    if t==Trajectory::T1Resume64{out.push((64,true));}out
}
fn recipe(cfg:&Deepseek4Config)->super::acquisition::ConfigRecipe<'static> {
    super::acquisition::ConfigRecipe {family:Family::Deepseek4,capacity:262144,slots:4,output_allowance:8192,
        cache_mode:"DeepseekNative-BF16-ring/compressed-F32state",route:"deepseek-ordinary-fullcache-endpoints-v1",
        actual_model_config_sha256:hash(format!("{cfg:?}").as_bytes())}
}
fn expected(cfg:&Deepseek4Config,t:Trajectory)->Result<Vec<ExpectedTensor>> {
    let mut out=cache_capture::expected(cfg,262144,&boundaries(t)).map_err(anyhow::Error::msg)?;
    for position in 0..144 {for lane in 0..4 {for (stage,shape) in [
        (Stage::LayerOutput,vec![1,cfg.hyper_connection_count as usize,cfg.hidden_size as usize]),
        (Stage::Logits,vec![1,cfg.vocab_size as usize])] {
        out.push(ExpectedTensor {at:coordinate(lane,position,stage,cfg.num_hidden_layers),scalar:Scalar::F32,shape});
    }}}
    Ok(out)
}
fn payload(expected:&[ExpectedTensor])->Result<u64> {
    expected.iter().try_fold(0u64,|sum,e|e.shape.iter().try_fold(e.scalar.bytes() as u64,
        |n,&d|n.checked_mul(d as u64)).and_then(|n|sum.checked_add(n))).context("DS payload overflow")
}
#[test]
#[ignore="root-controlled actual DeepSeek ordinary endpoints; reviewed identity/resource packet"]
fn q1_deepseek_actual_endpoint_trace(){run().unwrap();}
fn run()->Result<()> {
    ensure!(cfg!(target_endian="little"),"raw diagnostic encoding requires little endian");
    let path=std::env::var_os("HF2Q_Q1_DS_PACKET").context("explicit Q1 DS packet")?;
    let mut raw=Vec::new();std::fs::File::open(path)?.take(1024*1024+1).read_to_end(&mut raw)?;
    ensure!(raw.len()<=1024*1024,"bounded packet");let packet:Packet=serde_json::from_slice(&raw)?;
    packet.identity.validate().map_err(anyhow::Error::msg)?;packet.workload.validate().map_err(anyhow::Error::msg)?;
    ensure!(packet.execution_approved && packet.identity.family==Family::Deepseek4
        &&packet.workload.family==Family::Deepseek4,"explicit approved DS family");
    require_ordinary(packet.trajectory)?;
    ensure!(packet.configured_capacity==262144 &&packet.configured_output_allowance==8192,
        "full selected context/output profile required");
    ensure!(packet.requested_payload_limit>0 &&packet.requested_payload_limit<=2*1024*1024*1024,
        "bounded capture payload");
    ensure!(!super::family::enabled(Family::Deepseek4),"ordinary endpoints cannot silently activate per-layer diagnostic splits");
    let scope=super::acquisition::Scope::begin(&packet.acquisition,&packet.identity,&packet.workload,
        &packet.model,&packet.output,&raw)?;
    let gguf=super::acquisition::open_model(&packet.model)?;
    let cfg=Deepseek4Config::from_gguf(&gguf)?;
    ensure!(cfg.vocab_size==packet.workload.vocab_size,"DS actual vocabulary mismatch");
    scope.check_config(&packet.identity,&recipe(&cfg))?;
    let expected=expected(&cfg,packet.trajectory)?;let bytes=payload(&expected)?;
    ensure!(bytes<=packet.requested_payload_limit,"config-derived full payload exceeds pre-load bound");
    let cache_plan=Deepseek4CachePlan::for_context(&cfg,262144)?;
    let four_caches=cache_plan.resident_bytes.checked_mul(4).context("DS fourcache overflow")?;
    let records=expected.len();let mut store=Store::new(&packet.output,packet.identity.clone(),
        packet.trajectory.label().into(),expected,packet.requested_payload_limit,records).map_err(anyhow::Error::msg)?;
    let _gpu=crate::inference::hf2q_gpu_test_lock();
    let mut model=crate::inference::models::deepseek4::Deepseek4Model::load_from_gguf(&gguf)?;
    scope.check_config(&packet.identity,&recipe(&model.cfg))?;
    store.bind_acquisition(scope.observed(model.ctx.executor.device(),&model as *const _ as usize as u64,1)?)
        .map_err(anyhow::Error::msg)?;
    let result=deepseek_runner::run(&mut model,&packet.workload,packet.trajectory,packet.configured_capacity,|event| {
        match event {
            Observation::Forward {lane,end,state,logits,cache}=>{
                ensure!(cache.position()==end,"DS actual callback cursor mismatch");
                let start=end.checked_sub(state.shape()[0]).context("DS row origin underflow")?;
                for (stage,b,drain) in [(Stage::LayerOutput,state,"actual DS target verifier checked terminal + cursor publication"),
                    (Stage::Logits,logits,"actual DS full-row native head checked finish")] {
                    capture::rows_after_checked_drain(&mut store,coordinate(lane,start,stage,cfg.num_hidden_layers),
                        b,None,drain).map_err(anyhow::Error::msg)?;
                }
                if boundaries(packet.trajectory).contains(&(end,false)) {
                    cache_capture::after_checked_terminal(&mut store,&cfg,cache,lane,end,false).map_err(anyhow::Error::msg)?;
                }
            }
            Observation::Restored64 {lane,cache}=>{
                ensure!(packet.trajectory==Trajectory::T1Resume64,"unexpected diagnostic restore observation");
                cache_capture::after_checked_terminal(&mut store,&cfg,cache,lane,64,true).map_err(anyhow::Error::msg)?;
            }
        }Ok(())
    });
    if let Err(ref e)=result {store.fail(format!("actual DS execution: {e:#}"));}
    scope.check_unchanged()?;let trace=store.finish().map_err(anyhow::Error::msg)?;let execution=result?;
    ensure!(trace.coverage.errors.is_empty(),"DS full endpoint/cache coverage incomplete");
    let exact64=if packet.trajectory==Trajectory::T1Resume64 {
        Some(cache_capture::require_restore64_exact(&trace.coverage.captured))
    }else{None};
    let output=serde_json::json!({"schema":"hf2q.adr063.q1.endpoint-acquisition.v2","packet_sha256":hash(&raw),
        "execution":execution,"trace_records":trace.coverage.captured.len(),"planned_capture_bytes":bytes,
        "selected_context":262144,"output_allowance":8192,"logical_lanes":4,"four_cache_requested_bytes":four_caches,
        "cache_boundaries":boundaries(packet.trajectory),"full_active64_exact_restore":exact64,
        "window_order":"physical128-row ring with absolute logical boundary; not chronological reordering",
        "physical_cohort_scope":"T2 packed FFN prefill only; decode independent row graphs; no whole-body/F2 B4 claim",
        "T1_owner_scope":"same native cache backing; append-only compressed/indexer prefix retained",
        "internal_stage_localization":"NotEstablishedByEndpointTrace","whole_q1_qualified":false,
        "physical_admission":false,"task_quality_qualified":false,"float_policy":"Unspecified"});
    OpenOptions::new().create_new(true).write(true).mode(0o600).open(packet.output.join("execution.json"))?
        .write_all(&serde_json::to_vec_pretty(&output)?)?;
    if let Some(exact)=exact64 {exact.map_err(anyhow::Error::msg)?;}
    println!("Q1_DS_ENDPOINT_ACQUISITION_COMPLETE lanes=4 logits=144 whole_q1=false");Ok(())
}
fn coordinate(lane:usize,row:usize,stage:Stage,layer:u32)->Coordinate {
    Coordinate {logical_step:0,layer,stage,agent:format!("agent-{lane}"),logical_row_start:row as u64,segment:0}
}
#[cfg(test)]
#[path="deepseek_endpoint_controls.rs"]
mod controls;
