//! Root-only actual four-logical-lane Gemma ordinary trajectories. No T2/T5/T6
//! substitution, no model/device construction in hosted controls.
use super::{acquisition, gemma_cache, gemma_endpoint as endpoint, protocol::*,receipt::*,store::{hash,Store}};
use anyhow::{ensure,Context,Result};
use serde::Deserialize;
use std::{fs::OpenOptions,io::{Read,Write},os::unix::fs::OpenOptionsExt,path::PathBuf};
use crate::serve::{api::engine::{GemmaLoadedModel,LoadOptions,LoadedModel},multi_seq_kv::SlotId};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    execution_approved:bool, acquisition:acquisition::Binding,identity:Identity,
    workload:Workload,trajectory:Trajectory,model:PathBuf,output:PathBuf,
    configured_context:u32,configured_output_allowance:usize,host_payload_limit:u64,
}
fn preflight(p:&Packet)->Result<()> {
    p.identity.validate().map_err(anyhow::Error::msg)?;p.workload.validate().map_err(anyhow::Error::msg)?;
    ensure!(p.execution_approved && p.identity.family==Family::Gemma4 && p.workload.family==Family::Gemma4,
        "explicit approved Gemma packet/family required");
    if !matches!(p.trajectory,Trajectory::T2ActualCohort|Trajectory::T6CancelRestore) { require_ordinary(p.trajectory)?; }
    else if p.trajectory==Trajectory::T2ActualCohort { ensure!(std::env::var("HF2Q_BATCHED_BODY").as_deref()!=Ok("0"),
        "Gemma T2 selected body opt-out remains respected"); }
    ensure!(p.configured_context==262144 && p.configured_output_allowance==8192,
        "selected Gemma context/output profile required");
    ensure!(p.host_payload_limit>0 && p.host_payload_limit<=2*1024*1024*1024,
        "bounded explicit Gemma host payload required");
    Ok(())
}
pub(super) fn require_ordinary(t:Trajectory)->Result<()> {
    ensure!(matches!(t,Trajectory::T0ColdSerial|Trajectory::T1Resume64|Trajectory::T3Chunk64|
        Trajectory::T4ReverseOrder|Trajectory::T7RepeatCold|Trajectory::T8RepeatCold),
        "LocalizationIncomplete: Gemma T2 physical graph, T5 producer, T6 owning cancellation require separate real adapters");
    Ok(())
}
fn recipe(cfg:&crate::serve::config::Gemma4Config)->acquisition::ConfigRecipe<'static> {
    acquisition::ConfigRecipe {family:Family::Gemma4,capacity:262144,slots:4,output_allowance:8192,
        cache_mode:"GemmaHybrid-F16K-U8V-no-xlen",route:"gemma-ordinary-full-row-endpoints-v1",
        actual_model_config_sha256:hash(format!("{cfg:?}").as_bytes())}
}
fn positions(t:Trajectory)->Vec<(usize,bool)> {
    let mut p=vec![(128,false),(144,false)];
    if matches!(t,Trajectory::T1Resume64|Trajectory::T3Chunk64|Trajectory::T6CancelRestore) {p.push((64,false));}
    if matches!(t,Trajectory::T1Resume64|Trajectory::T6CancelRestore) {p.push((64,true));}p
}
fn cache(model:&GemmaLoadedModel,lane:usize,pos:usize,restored:bool)->Result<()> {
    endpoint::with_store(|s|gemma_cache::after_checked_return(s,&model.config,
        model.multi_seq_kv_hybrid.as_ref().ok_or("Gemma Hybrid absent")?,lane,pos,restored))
        .map_err(anyhow::Error::msg)
}
fn prefill(model:&mut GemmaLoadedModel,tokens:&[u32],lane:usize,cached:usize)->Result<()> {
    endpoint::begin_call(lane,cached,tokens.len()-cached).map_err(anyhow::Error::msg)?;
    let hb=model.multi_seq_kv.as_mut().context("Gemma slot scaffold")?;
    if cached==0 {
        model.weights.forward_prefill_with_soft_tokens_slot_aware(tokens,&[],8192,&mut model.ctx,
            SlotId(lane as u32),hb,model.multi_seq_kv_hybrid.as_mut(),model.multi_seq_kv_dense.as_mut(),
            model.multi_seq_kv_mlx.as_mut())?;
    } else {
        model.weights.forward_prefill_with_soft_tokens_slot_aware_resume(tokens,&[],8192,&mut model.ctx,
            SlotId(lane as u32),hb,model.multi_seq_kv_hybrid.as_mut(),model.multi_seq_kv_dense.as_mut(),
            model.multi_seq_kv_mlx.as_mut(),cached)?;
    }
    crate::serve::api::engine::q1_gemma_cohort::commit_after_checked_return(
        model,SlotId(lane as u32),cached,tokens.len())?;
    endpoint::end_call().map_err(anyhow::Error::msg)
}
#[test]
#[ignore="root supervised actual Gemma model; reviewed acquisition/resource packet required"]
fn q1_gemma_actual_ordinary_endpoint_trace() {run(None).unwrap();}
#[test]
#[ignore="root supervised actual Gemma packed cohort; reviewed resource/identity packet"]
fn q1_gemma_actual_cohort_endpoint_trace() {run(Some(Trajectory::T2ActualCohort)).unwrap();}
#[test]
#[ignore="root supervised actual Gemma owning-worker cancellation and exact64 restore"]
fn q1_gemma_actual_cancel_endpoint_trace() {run(Some(Trajectory::T6CancelRestore)).unwrap();}
fn run(separate:Option<Trajectory>)->Result<()> {
    ensure!(cfg!(target_endian="little"),"raw tensor encoding requires little endian");
    let path=std::env::var_os("HF2Q_Q1_GEMMA_ENDPOINT_PACKET").context("explicit Gemma endpoint packet")?;
    let mut raw=Vec::new();std::fs::File::open(path)?.take(1024*1024+1).read_to_end(&mut raw)?;
    ensure!(raw.len()<=1024*1024,"Gemma packet quota");
    let p:Packet=serde_json::from_slice(&raw)?;preflight(&p)?;
    if let Some(t)=separate {ensure!(p.trajectory==t,"actual test filter/trajectory mismatch");}
    else {require_ordinary(p.trajectory)?;}
    let scope=acquisition::Scope::begin(&p.acquisition,&p.identity,&p.workload,&p.model,&p.output,&raw)?;
    let header=acquisition::open_model(&p.model)?;
    ensure!(header.metadata_string("general.architecture")==Some("gemma4"),"no family fallback");
    let mut cfg=crate::serve::config::Gemma4Config::from_gguf(&header)?;
    cfg.max_position_embeddings=p.configured_context as usize;
    ensure!(cfg.vocab_size==p.workload.vocab_size as usize,"Gemma actual vocabulary mismatch");
    scope.check_config(&p.identity,&recipe(&cfg))?;drop(header);
    let mut expected=gemma_cache::expected(&cfg,&[0,1,2,3],&positions(p.trajectory)).map_err(anyhow::Error::msg)?;
    for lane in 0..4 {for pos in 0..144 {expected.push(ExpectedTensor {at:Coordinate {
        logical_step:0,layer:u32::MAX,stage:Stage::Logits,agent:format!("agent-{lane}"),
        logical_row_start:pos,segment:0},scalar:Scalar::F32,shape:vec![1,cfg.vocab_size]});}}
    let bytes=expected.iter().try_fold(0u64,|sum,e| {
        let n=e.shape.iter().try_fold(e.scalar.bytes() as u64,|a,&b|a.checked_mul(b as u64))?;
        sum.checked_add(n)
    }).context("Gemma capture payload overflow")?;
    ensure!(bytes<=p.host_payload_limit,"Gemma payload exceeds pre-load bound");
    let records=expected.len();
    let mut store=Store::new(&p.output,p.identity.clone(),p.trajectory.label().into(),expected,
        p.host_payload_limit,records).map_err(anyhow::Error::msg)?;
    let _gpu=crate::inference::hf2q_gpu_test_lock();
    let options=LoadOptions {model_path:p.model.clone(),tokenizer_path:None,config_path:None,
        dwq_overlay_path:None,kv_persist_dir:None,kv_persist_budget_bytes:0,glp_path:None,glp_alpha:None,kv_graft_path:None};
    let LoadedModel::Gemma(mut model)=LoadedModel::load_with_context(&options,Some(p.configured_context))?
        else {anyhow::bail!("explicit Gemma model required")};
    scope.check_config(&p.identity,&recipe(&model.config))?;
    store.bind_acquisition(scope.observed(model.ctx.executor.device(),&model as *const _ as usize as u64,3)?)
        .map_err(anyhow::Error::msg)?;
    model.provision_multi_seq_kv_for_slot_aware(4)?;
    ensure!(model.multi_seq_kv_hybrid.is_some() && model.multi_seq_kv_dense.is_none()
        && model.multi_seq_kv_mlx.is_none(),"selected Hybrid only; other actual legs require explicit coverage");
    let _capture=endpoint::install(store).map_err(anyhow::Error::msg)?;
    let order=if p.trajectory==Trajectory::T4ReverseOrder {[3,2,1,0]}else{[0,1,2,3]};
    let mut cancellation=Vec::new();
    let result=(||->Result<()> {
        if p.trajectory==Trajectory::T2ActualCohort {
            use crate::serve::api::engine::q1_gemma_cohort;
            let layout=(0..4).map(|lane|(lane as u32,0,128,lane*128)).collect::<Vec<_>>();
            endpoint::begin_packed(&layout).map_err(anyhow::Error::msg)?;
            let seqs=(0..4).map(|lane|(p.workload.lanes[lane][..128].to_vec(),SlotId(lane as u32)))
                .collect::<Vec<_>>();
            q1_gemma_cohort::prefill(&mut model,&seqs)?;
            endpoint::end_call().map_err(anyhow::Error::msg)?;
            for lane in 0..4 {cache(&model,lane,128,false)?;}
            let slots=[SlotId(0),SlotId(1),SlotId(2),SlotId(3)];
            for position in 128..144 {
                let tokens=p.workload.lanes.iter().map(|v|v[position]).collect::<Vec<_>>();
                let logits=q1_gemma_cohort::decode(&mut model,&tokens,&slots,position)?;
                for (lane,row) in logits.chunks_exact(cfg.vocab_size).enumerate() {
                    endpoint::downloaded_logits(lane,position,row).map_err(anyhow::Error::msg)?;
                }
            }
            for lane in 0..4 {cache(&model,lane,144,false)?;}
            return Ok(());
        }
        for lane in order {
            let tokens=&p.workload.lanes[lane];
            if p.trajectory==Trajectory::T6CancelRestore {
                endpoint::begin_call(lane,0,64).map_err(anyhow::Error::msg)?;
                cancellation.push(crate::serve::api::engine::q1_gemma_cancel::run_loaded(
                    &mut model,lane,&tokens[..128],|restored,hybrid| {
                        endpoint::with_store(|store|gemma_cache::after_checked_return(
                            store,&cfg,hybrid,lane,64,restored)).map_err(anyhow::Error::msg)
                    })?);
                prefill(&mut model,&tokens[..128],lane,64)?;
            } else if matches!(p.trajectory,Trajectory::T1Resume64|Trajectory::T3Chunk64) {
                prefill(&mut model,&tokens[..64],lane,0)?;cache(&model,lane,64,false)?;
                if p.trajectory==Trajectory::T1Resume64 {
                    use crate::inference::models::gemma4::kv_cache::{snapshot_gemma_hybrid_slot_anchor,restore_gemma_hybrid_slot_anchor};
                    let anchor=snapshot_gemma_hybrid_slot_anchor(model.multi_seq_kv_hybrid.as_ref().unwrap(),SlotId(lane as u32),64)?;
                    restore_gemma_hybrid_slot_anchor(model.multi_seq_kv_hybrid.as_mut().unwrap(),SlotId(lane as u32),&anchor,64)?;
                    cache(&model,lane,64,true)?;
                }
                prefill(&mut model,&tokens[..128],lane,64)?;
            } else {prefill(&mut model,&tokens[..128],lane,0)?;}
            cache(&model,lane,128,false)?;
            for pos in 128..144 {
                endpoint::begin_call(lane,pos,1).map_err(anyhow::Error::msg)?;
                model.weights.forward_decode_slot_aware(tokens[pos],pos,&mut model.ctx,&mut None,SlotId(lane as u32),
                    model.multi_seq_kv.as_mut().context("Gemma slot scaffold")?,model.multi_seq_kv_hybrid.as_mut(),
                    model.multi_seq_kv_dense.as_mut(),model.multi_seq_kv_mlx.as_mut())?;
                crate::serve::api::engine::q1_gemma_cohort::commit_after_checked_return(
                    &mut model,SlotId(lane as u32),pos,pos+1)?;
                endpoint::logits(0,&model.weights.activations.logits,cfg.vocab_size).map_err(anyhow::Error::msg)?;
                endpoint::end_call().map_err(anyhow::Error::msg)?;
            }
            cache(&model,lane,144,false)?;
        }Ok(())
    })();
    if let Err(ref error)=result {let _=endpoint::with_store(|s|{s.fail(error.to_string());Ok(())});}
    scope.check_unchanged()?;
    let trace=endpoint::finish().map_err(anyhow::Error::msg)?;result?;
    ensure!(trace.coverage.errors.is_empty(),"Gemma endpoint extent incomplete");
    let exact64=if matches!(p.trajectory,Trajectory::T1Resume64|Trajectory::T6CancelRestore) {
        Some(gemma_cache::require_restore64_exact(&trace.coverage.captured))
    } else {None};
    let report=serde_json::json!({"schema":"hf2q.q1.gemma-ordinary-endpoint.v1","packet_sha256":hash(&raw),
        "trajectory":p.trajectory,"actual_cancellation":cancellation,"active64_exact_copy":exact64,"logical_lanes":4,"physical_graph_membership":if p.trajectory==Trajectory::T2ActualCohort {
            "actual packed prefill512 rows with four128-owner spans; actual decode input4rows; operator-specific source geometry, not whole-kernel/F2 membership"
        } else {"per-request ordinary; not T2"},
        "logit_rows_per_lane":144,"cache_boundaries":positions(p.trajectory),"capture_records":records,
        "planned_tensor_payload":bytes,"oracle_quality_qualified":false,"whole_q1_qualified":false,
        "float_policy":"Unspecified","diagnostic_head_scope":"same production scalar head applied to every actual prefill hidden row"});
    let mut file=OpenOptions::new().create_new(true).write(true).mode(0o600).open(p.output.join("execution.json"))?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    if let Some(exact)=exact64 {exact.map_err(anyhow::Error::msg)?;}
    println!("Q1_GEMMA_ORDINARY_ENDPOINT_COMPLETE lanes=4 logits=144 whole_q1=false");Ok(())
}
#[cfg(test)]
#[path="gemma_controls.rs"]
mod controls;
