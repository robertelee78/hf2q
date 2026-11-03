//! Root-only actual four-logical-lane Gemma ordinary trajectories. No T2/T5/T6
//! substitution, no model/device construction in hosted controls. The cohort
//! stage localizer (`gemma_cohort_localizer`) runs the T0/T2 bodies here, under
//! its own filters, with a nested stage store beside the unchanged endpoints.
use super::{acquisition, gemma_cache, gemma_cohort_localizer as cohort, gemma_endpoint as endpoint, protocol::*,receipt::*,store::{hash,Store}};
use anyhow::{ensure,Context,Result};
use serde::{Deserialize,Serialize};
use std::{collections::BTreeMap,fs::OpenOptions,io::{Read,Write},os::unix::fs::OpenOptionsExt,path::PathBuf};
use crate::serve::{api::engine::{GemmaLoadedModel,LoadOptions,LoadedModel},multi_seq_kv::SlotId};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Packet {
    execution_approved:bool, acquisition:acquisition::Binding,identity:Identity,
    workload:Workload,trajectory:Trajectory,model:PathBuf,output:PathBuf,
    configured_context:u32,configured_output_allowance:usize,host_payload_limit:u64,
    #[serde(default)] routing_witness:Option<super::gemma_routing_witness::Budget>,
    /// Warm live-resume kernel the loaded model must resolve; absent = VecHybrid.
    #[serde(default)] warm_live_kernel:WarmLiveKernel,
    /// Cohort stage localizer only: the stage plan, the lanes it covers and
    /// every store's frozen record list (`endpoint`, `stages`).
    #[serde(default)] stage_localizer:Option<cohort::Plan>,
    #[serde(default)] lanes:Option<Vec<usize>>,
    #[serde(default)] expected_stores:Option<BTreeMap<String,Vec<ExpectedTensor>>>,
}
/// Gemma warm live-resume attention kernel (T1/T3/T6 64-row resumes and the
/// localizer's `[64,64]` route). It is resolved once at model load and is not
/// in `config_sha256`. `VecHybrid` is the shipped default since #544 and the
/// kernel of every retained Gemma Q1 run; `TiledF16` must be declared.
#[derive(Clone,Copy,Debug,Default,Deserialize,Serialize,PartialEq,Eq)]
pub(super) enum WarmLiveKernel {#[default] VecHybrid,TiledF16}
/// Refuses a loaded model whose warm kernel is not the packet's declared one.
pub(super) fn require_warm_kernel(declared:WarmLiveKernel,resolved:crate::inference::models::gemma4::GemmaLiveKernel)->Result<WarmLiveKernel> {
    use crate::inference::models::gemma4::GemmaLiveKernel as K;
    let resolved=match resolved {K::VecHybrid=>WarmLiveKernel::VecHybrid,K::TiledF16=>WarmLiveKernel::TiledF16};
    ensure!(resolved==declared,"resolved Gemma warm live kernel {resolved:?} differs from the packet's {declared:?} (absent = shipped VecHybrid)");
    Ok(resolved)
}
const ENDPOINT_PACKET_BYTES:usize=1024*1024;
/// A localizer packet carries both frozen store lists (indent-2 JSON).
const LOCALIZER_PACKET_BYTES:usize=16*1024*1024;
pub(super) fn preflight(p:&Packet,localizer:bool)->Result<()> {
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
    super::gemma_routing_witness::validate_budget(p.routing_witness.as_ref())?;
    ensure!(p.stage_localizer.is_some()==localizer && p.lanes.is_some()==localizer && p.expected_stores.is_some()==localizer,
        "stage-localizer packets run only under their own filters, and only they carry stores");
    if localizer {
        cohort::route(p.trajectory).map_err(anyhow::Error::msg)?;
        ensure!(p.routing_witness.is_some(),"cohort localizer mirrors the witness-ON b3 endpoint stores");
        ensure!(p.lanes.as_deref()==Some(&[0,1,2,3][..]),"cohort localizer covers all four workload lanes");
        ensure!(p.expected_stores.as_ref().is_some_and(|s|s.len()==2 && s.contains_key(cohort::ENDPOINT_STORE)
            && s.contains_key(cohort::STAGE_STORE)),"exactly the endpoint and stage stores");
        ensure!(std::env::var_os("HF2Q_PARALLEL_ENCODE").is_none(),
            "single-row decode stage hooks run on the owning thread only");
    }
    Ok(())
}
pub(super) fn require_ordinary(t:Trajectory)->Result<()> {
    ensure!(matches!(t,Trajectory::T0ColdSerial|Trajectory::T1Resume64|Trajectory::T3Chunk64|
        Trajectory::T4ReverseOrder|Trajectory::T7RepeatCold|Trajectory::T8RepeatCold),
        "LocalizationIncomplete: Gemma T2 physical graph, T5 producer, T6 owning cancellation require separate real adapters");
    Ok(())
}
pub(super) fn recipe(cfg:&crate::serve::config::Gemma4Config)->acquisition::ConfigRecipe<'static> {
    acquisition::ConfigRecipe {family:Family::Gemma4,capacity:262144,slots:4,output_allowance:8192,
        cache_mode:"GemmaHybrid-F16K-U8V-no-xlen",route:"gemma-ordinary-full-row-endpoints-v1",
        actual_model_config_sha256:hash(format!("{cfg:?}").as_bytes())}
}
/// Same selected recipe used by actual acquisition checks and metadata export.
pub(super) fn selected_recipe(cfg:&crate::serve::config::Gemma4Config,routing:bool)->acquisition::ConfigRecipe<'static> {
    let mut r=recipe(cfg);
    if routing {r.route="gemma-ordinary-full-row-endpoints-native-routing1511-v1";}r
}
pub(super) fn positions(t:Trajectory)->Vec<(usize,bool)> {
    let mut p=vec![(128,false),(144,false)];
    if matches!(t,Trajectory::T1Resume64|Trajectory::T3Chunk64|Trajectory::T6CancelRestore) {p.push((64,false));}
    if matches!(t,Trajectory::T1Resume64|Trajectory::T6CancelRestore) {p.push((64,true));}p
}
/// Every endpoint record of one trajectory, in capture order: active cache at
/// its boundaries, then all four lanes' 144 logit rows.
pub(super) fn endpoint_expected(cfg:&crate::serve::config::Gemma4Config,t:Trajectory)->Result<Vec<ExpectedTensor>> {
    let mut expected=gemma_cache::expected(cfg,&[0,1,2,3],&positions(t)).map_err(anyhow::Error::msg)?;
    for lane in 0..4 {for pos in 0..144 {expected.push(ExpectedTensor {at:Coordinate {
        logical_step:0,layer:u32::MAX,stage:Stage::Logits,agent:format!("agent-{lane}"),
        logical_row_start:pos,segment:0},scalar:Scalar::F32,shape:vec![1,cfg.vocab_size]});}}
    Ok(expected)
}
fn payload(expected:&[ExpectedTensor])->Result<u64> {
    expected.iter().try_fold(0u64,|sum,e| {
        let n=e.shape.iter().try_fold(e.scalar.bytes() as u64,|a,&b|a.checked_mul(b as u64))?;
        sum.checked_add(n)
    }).context("Gemma capture payload overflow")
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
fn q1_gemma_actual_ordinary_endpoint_trace() {run(None,false).unwrap();}
#[test]
#[ignore="root supervised actual Gemma packed cohort; reviewed resource/identity packet"]
fn q1_gemma_actual_cohort_endpoint_trace() {run(Some(Trajectory::T2ActualCohort),false).unwrap();}
#[test]
#[ignore="root supervised actual Gemma owning-worker cancellation and exact64 restore"]
fn q1_gemma_actual_cancel_endpoint_trace() {run(Some(Trajectory::T6CancelRestore),false).unwrap();}
#[test]
#[ignore="root supervised actual Gemma four-lane All128 stage localizer; reviewed multistore packet"]
fn q1_gemma_actual_all128_stage_localizer() {run(Some(Trajectory::T0ColdSerial),true).unwrap();}
#[test]
#[ignore="root supervised actual Gemma packed T2 stage localizer; reviewed multistore packet"]
fn q1_gemma_actual_cohort_stage_localizer() {run(Some(Trajectory::T2ActualCohort),true).unwrap();}
fn run(separate:Option<Trajectory>,localizer:bool)->Result<()> {
    ensure!(cfg!(target_endian="little"),"raw tensor encoding requires little endian");
    let path=std::env::var_os("HF2Q_Q1_GEMMA_ENDPOINT_PACKET").context("explicit Gemma endpoint packet")?;
    let quota=if localizer {LOCALIZER_PACKET_BYTES}else{ENDPOINT_PACKET_BYTES};
    let mut raw=Vec::new();std::fs::File::open(path)?.take(quota as u64+1).read_to_end(&mut raw)?;
    ensure!(raw.len()<=quota,"Gemma packet quota");
    let p:Packet=serde_json::from_slice(&raw)?;preflight(&p,localizer)?;
    if let Some(t)=separate {ensure!(p.trajectory==t,"actual test filter/trajectory mismatch");}
    else {require_ordinary(p.trajectory)?;}
    let recipe_for=|cfg:&crate::serve::config::Gemma4Config| if localizer {cohort::recipe(cfg)}
        else {selected_recipe(cfg,p.routing_witness.is_some())};
    let scope=acquisition::Scope::begin(&p.acquisition,&p.identity,&p.workload,&p.model,&p.output,&raw)?;
    let header=acquisition::open_model(&p.model)?;
    ensure!(header.metadata_string("general.architecture")==Some("gemma4"),"no family fallback");
    let mut cfg=crate::serve::config::Gemma4Config::from_gguf(&header)?;
    cfg.max_position_embeddings=p.configured_context as usize;
    ensure!(cfg.vocab_size==p.workload.vocab_size as usize,"Gemma actual vocabulary mismatch");
    scope.check_config(&p.identity,&recipe_for(&cfg))?;drop(header);
    let expected=endpoint_expected(&cfg,p.trajectory)?;
    let bytes=payload(&expected)?;
    ensure!(bytes<=p.host_payload_limit,"Gemma payload exceeds pre-load bound");
    let witness=super::gemma_routing_witness::PAYLOAD_BYTES+super::gemma_routing_witness::METADATA_BYTES
        +super::gemma_routing_witness::CPU_LEDGER_BYTES;
    if p.routing_witness.is_some(){ensure!(bytes+witness<=p.host_payload_limit,
        "routing+metadata+CPU ledger must join pre-load host bound");}
    // Cohort localizer: both frozen store lists must be the source manifests,
    // and endpoint + routing + stage bytes join the one packet bound.
    let stage_root=p.output.join(cohort::STAGE_STORE);
    let (stage_plan,mut stage_targets,stage_bytes)=match (&p.stage_localizer,&p.expected_stores) {
        (Some(plan),Some(stores))=>{
            ensure!(stores[cohort::ENDPOINT_STORE]==expected,"packet endpoint store differs from the source endpoint manifest");
            let frozen=&stores[cohort::STAGE_STORE];
            let (targets,stage_bytes)=cohort::admit(&cfg,plan,frozen).map_err(anyhow::Error::msg)?;
            ensure!(bytes+witness+plan.host_payload_limit<=p.host_payload_limit,
                "endpoint+routing+stage payload must join the packet host bound");
            (Some((plan,frozen.clone())),Some(targets),stage_bytes)
        }
        _=>(None,None,0),
    };
    let records=expected.len();
    let mut store=Store::new(&p.output,p.identity.clone(),p.trajectory.label().into(),expected,
        p.host_payload_limit,records).map_err(anyhow::Error::msg)?;
    let mut stage_store=match stage_plan {
        Some((plan,frozen))=>{let n=frozen.len();Some(Store::new(&stage_root,p.identity.clone(),plan.rerun_id.clone(),
            frozen,plan.host_payload_limit,n).map_err(anyhow::Error::msg)?)}
        None=>None,
    };
    let _gpu=crate::inference::hf2q_gpu_test_lock();
    let options=LoadOptions {model_path:p.model.clone(),tokenizer_path:None,config_path:None,
        dwq_overlay_path:None,kv_persist_dir:None,kv_persist_budget_bytes:0,glp_path:None,glp_alpha:None,kv_graft_path:None};
    let LoadedModel::Gemma(mut model)=LoadedModel::load_with_context(&options,Some(p.configured_context))?
        else {anyhow::bail!("explicit Gemma model required")};
    scope.check_config(&p.identity,&recipe_for(&model.config))?;
    let warm_live_kernel=require_warm_kernel(p.warm_live_kernel,model.weights.warm_live_kernel)?;
    let observed=scope.observed(model.ctx.executor.device(),&model as *const _ as usize as u64,3)?;
    store.bind_acquisition(observed.clone()).map_err(anyhow::Error::msg)?;
    if let Some(stages)=stage_store.as_mut() {
        // Same acquisition; the nested store's own root.
        stages.bind_acquisition(acquisition::Acquisition {output_root:stage_root.clone(),..observed.clone()})
            .map_err(anyhow::Error::msg)?;
        if p.stage_localizer.as_ref().is_some_and(|plan|plan.prefill_stages.iter().chain(&plan.decode_stages)
            .any(|s|matches!(s,Stage::RouterLogits|Stage::RouterIds|Stage::RouterWeights))) {
            stages.missing_capability("Gemma actual native post-softmax/renormalized selection kth+1 GPU score is not a captured stage output; the routing witness carries the selection-domain margins");
        }
    }
    model.provision_multi_seq_kv_for_slot_aware(4)?;
    ensure!(model.multi_seq_kv_hybrid.is_some() && model.multi_seq_kv_dense.is_none()
        && model.multi_seq_kv_mlx.is_none(),"selected Hybrid only; other actual legs require explicit coverage");
    let routing_mode=if p.trajectory==Trajectory::T6CancelRestore {super::gemma_routing_witness::Mode::Cancelled}else{super::gemma_routing_witness::Mode::Ordinary};
    let _routing=super::gemma_routing_witness::install(model.ctx.executor.device(),&p.output,&p.identity,&observed,routing_mode,p.routing_witness.as_ref())?;
    let _capture=endpoint::install(store).map_err(anyhow::Error::msg)?;
    let _stages=match (stage_store,stage_targets.take()) {
        (Some(stages),Some(targets))=>Some(super::family::install(Family::Gemma4,stages,targets).map_err(anyhow::Error::msg)?),
        _=>None,
    };
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
                super::gemma_routing_witness::begin(super::gemma_routing_witness::Namespace::Canonical,
                    (0..4).map(|lane|super::gemma_routing_witness::Row{lane,position}).collect())?;
                let logits=q1_gemma_cohort::decode(&mut model,&tokens,&slots,position)?;
                super::gemma_routing_witness::end_checked()?;
                for (lane,row) in logits.chunks_exact(cfg.vocab_size).enumerate() {
                    endpoint::downloaded_logits(lane,position,row).map_err(anyhow::Error::msg)?;
                }
            }
            for lane in 0..4 {cache(&model,lane,144,false)?;}
            return Ok(());
        }
        for lane in order {
            // All128 reference localizer: every single-sequence call of this
            // lane owns its stage rows (packed T2 rows carry their own owners).
            let _lane=if localizer {Some(super::gemma_prefill::serial_lane(lane)?)}else{None};
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
    let stage_trace=localizer.then(|| super::family::finish().map_err(anyhow::Error::msg));
    let trace=endpoint::finish().map_err(anyhow::Error::msg)?;result?;
    let stage_trace=stage_trace.transpose()?;
    super::gemma_routing_witness::finish()?;
    ensure!(trace.coverage.errors.is_empty(),"Gemma endpoint extent incomplete");
    if let Some(stages)=&stage_trace {
        ensure!(stages.coverage.errors.is_empty(),"Gemma cohort stage coverage incomplete: {:?}",stages.coverage.errors);
    }
    let exact64=if matches!(p.trajectory,Trajectory::T1Resume64|Trajectory::T6CancelRestore) {
        Some(gemma_cache::require_restore64_exact(&trace.coverage.captured))
    } else {None};
    let mut report=serde_json::json!({"schema":"hf2q.q1.gemma-ordinary-endpoint.v1","packet_sha256":hash(&raw),
        "trajectory":p.trajectory,"actual_cancellation":cancellation,"active64_exact_copy":exact64,"logical_lanes":4,"physical_graph_membership":if p.trajectory==Trajectory::T2ActualCohort {
            "actual packed prefill512 rows with four128-owner spans; actual decode input4rows; operator-specific source geometry, not whole-kernel/F2 membership"
        } else {"per-request ordinary; not T2"},
        "logit_rows_per_lane":144,"cache_boundaries":positions(p.trajectory),"capture_records":records,"warm_live_kernel":warm_live_kernel,
        "planned_tensor_payload":bytes,"oracle_quality_qualified":false,"whole_q1_qualified":false,
        "float_policy":"Unspecified","diagnostic_head_scope":"same production scalar head applied to every actual prefill hidden row"});
    let mut stage_records=0;
    if let (Some(stages),Some(plan))=(&stage_trace,&p.stage_localizer) {
        stage_records=stages.coverage.captured.len();
        let extra=serde_json::json!({"schema":cohort::SCHEMA,"rerun_id":plan.rerun_id,
            "stage_route":cohort::route(p.trajectory).map_err(anyhow::Error::msg)?,"lanes":[0,1,2,3],
            "stage_store":cohort::STAGE_STORE,"stage_records":stage_records,"stage_planned_payload":stage_bytes,
            "stage_layers":plan.layers,"prefill_stages":plan.prefill_stages,"decode_stages":plan.decode_stages,
            "row_owners":if p.trajectory==Trajectory::T2ActualCohort {
                "packed prefill rows via gemma_endpoint::packed_owners (lane*128); batched decode row i = slot i = lane i"
            } else {"one serial-lane lease per lane around its single-sequence prefill and single-row decode"},
            "decode_hooks":if p.trajectory==Trajectory::T2ActualCohort {"batched_body.rs capture_batched"}else{"gpu_full_attn.rs single-row capture"},
            "observer_effect_control":"exact of this endpoint store against the b3 endpoint store of the same trajectory",
            "route_class":"ClassB packed-versus-All128; localization implies no fix"});
        report.as_object_mut().context("report object")?.extend(extra.as_object().context("stage receipt")?.clone());
    }
    let mut file=OpenOptions::new().create_new(true).write(true).mode(0o600).open(p.output.join("execution.json"))?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    if let Some(exact)=exact64 {exact.map_err(anyhow::Error::msg)?;}
    if localizer {
        println!("{} records={} whole_q1_qualified=false",cohort::MARKER,records+stage_records);
    } else {
        println!("Q1_GEMMA_ORDINARY_ENDPOINT_COMPLETE lanes=4 logits=144 whole_q1=false");
    }
    Ok(())
}
#[cfg(test)]
#[path="gemma_controls.rs"]
pub(super) mod controls;
