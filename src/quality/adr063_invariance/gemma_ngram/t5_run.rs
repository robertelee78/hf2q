//! Root-only Q1 Gemma T5 phase (design §2.3) and its verifier-route stage
//! localizer mode (§2.6). The packet carries no record list: every store's
//! extent comes from `t5`/`localizer` source manifests; the supervisor binds
//! the same extent from the exported manifest file.
use super::*;
use super::super::{localizer,t5::{self,Arm}};
use crate::adr063_invariance::{family,gemma_routing_witness as witness,receipt::*,store::{hash,Store}};
use serde::Serialize;
use std::{collections::BTreeMap,fs::OpenOptions,io::Write,os::unix::fs::OpenOptionsExt};
pub(in super::super) const PACKET_BYTES:usize=1024*1024;
pub(in super::super) const MANIFEST_BYTES:u64=64*1024*1024;
#[derive(Clone,Copy,Debug,PartialEq,Eq,Serialize,Deserialize)]
pub(in super::super) enum Branch {N,P}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]
pub(in super::super) struct Receipts {pub(in super::super) ordinary:FileBinding,pub(in super::super) ngram:FileBinding}
/// The existing closed n-gram profile plus the T5 arm/extent keys. The
/// localizer-only keys are present exactly under the localizer filter.
#[derive(Deserialize)]#[serde(deny_unknown_fields)]
pub(in super::super) struct T5Packet {
    execution_approved:bool,identity:Identity,acquisition:acquisition::Binding,
    model:PathBuf,output:PathBuf,greedy_workload:FileBinding,independent_dataset_review:FileBinding,
    configured_context:u32,configured_output_allowance:u32,logical_serial_owners:usize,
    proposal_origin:String,ngram_k:u32,ngram_min:u32,ngram_max:u32,payload_limit:u64,metadata_limit:u64,
    #[serde(default)] routing_witness:Option<witness::Budget>,
    arm:Arm,lanes:Vec<usize>,host_payload_limit:u64,
    #[serde(default)] stage_localizer:Option<localizer::Plan>,
    #[serde(default)] branch:Option<Branch>,
    #[serde(default)] pass:Option<u8>,
    #[serde(default)] t5_receipts:Option<Receipts>,
    #[serde(default)] manifest:Option<FileBinding>,
    /// Branch P only: the reviewed All128 row-127 precondition record.
    #[serde(default)] all128_precondition:Option<FileBinding>,
}
/// The supervisor manifest-form extent (exactly these keys).
#[derive(Serialize,Deserialize)]#[serde(deny_unknown_fields)]
pub(in super::super) struct Manifest {
    pub(in super::super) lanes:Vec<usize>,pub(in super::super) host_payload_limit:u64,
    pub(in super::super) expected_stores:BTreeMap<String,Vec<ExpectedTensor>>,
}
pub(in super::super) struct Extra {
    pub(in super::super) arm:Arm,pub(in super::super) lanes:Vec<usize>,pub(in super::super) host_payload_limit:u64,
    pub(in super::super) stage_localizer:Option<localizer::Plan>,pub(in super::super) branch:Option<Branch>,
    pub(in super::super) pass:Option<u8>,pub(in super::super) t5_receipts:Option<Receipts>,pub(in super::super) manifest:Option<FileBinding>,
    pub(in super::super) all128_precondition:Option<FileBinding>,
}
fn split(p:T5Packet)->(Packet,Extra) {
    let T5Packet {execution_approved,identity,acquisition,model,output,greedy_workload,independent_dataset_review,
        configured_context,configured_output_allowance,logical_serial_owners,proposal_origin,ngram_k,ngram_min,ngram_max,
        payload_limit,metadata_limit,routing_witness,arm,lanes,host_payload_limit,stage_localizer,branch,pass,t5_receipts,manifest,all128_precondition}=p;
    (Packet {execution_approved,identity,acquisition,model,output,greedy_workload,independent_dataset_review,configured_context,
        configured_output_allowance,logical_serial_owners,proposal_origin,ngram_k,ngram_min,ngram_max,payload_limit,metadata_limit,routing_witness},
     Extra {arm,lanes,host_payload_limit,stage_localizer,branch,pass,t5_receipts,manifest,all128_precondition})
}
/// The T5 keys of a parsed packet (hosted pairing controls).
pub(in super::super) fn extra_of(p:T5Packet)->Extra {split(p).1}
/// Explicit packet/test pairing and the frozen branch/pass rules.
pub(in super::super) fn pairing(x:&Extra,localizer:bool)->Result<()> {
    ensure!(x.lanes==[0,1,2,3],"T5 covers all four workload lanes");
    ensure!(x.stage_localizer.is_some()==localizer && x.branch.is_some()==localizer && x.pass.is_some()==localizer &&
        x.t5_receipts.is_some()==localizer && x.manifest.is_some()==localizer,
        "stage_localizer packets run only under the localizer filter, and only they bind T5 receipts and a manifest");
    ensure!(x.all128_precondition.is_some()==(x.branch==Some(Branch::P)),
        "the All128 precondition record is bound exactly by branch P packets");
    let (Some(plan),Some(branch),Some(pass))=(&x.stage_localizer,x.branch,x.pass) else {return Ok(())};
    localizer::require_pass(plan,pass).map_err(anyhow::Error::msg)?;
    plan.accepted().map_err(anyhow::Error::msg)?;
    match branch {
        Branch::P=>ensure!(x.arm==Arm::OrdinaryGlobal && plan.accepted.is_none(),
            "branch P runs only the ordinary-global arm, with no accepted count"),
        Branch::N=>ensure!(plan.accepted.is_some(),"branch N freezes the T5 n-gram receipt's accepted counts"),
    }
    Ok(())
}
/// The two bound T5 phase receipts against the branch rule.
pub(in super::super) fn check_receipts(ord:&Value,ng:&Value,branch:Branch,accepted:Option<[usize;4]>)->Result<()> {
    ensure!(ord["schema"]==t5::SCHEMA && ng["schema"]==t5::SCHEMA && ord["arm"]=="OrdinaryGlobal" && ng["arm"]=="NgramTransaction" &&
        ord["whole_q1_qualified"]==false && ng["whole_q1_qualified"]==false,"bound T5 ordinary/n-gram phase receipts");
    ensure!(ord["status"]=="CompleteScopedAcquisition" && ord["error"].is_null(),"ordinary-global T5 receipt complete");
    match branch {
        Branch::N=>{
            ensure!(ng["status"]=="CompleteScopedAcquisition" && ng["class_b_refusal"].is_null(),"branch N needs a complete T5 n-gram receipt");
            let a=accepted.context("branch N accepted counts")?;
            let d=ng["decisions"].as_array().context("n-gram decisions")?;ensure!(d.len()==4,"four n-gram lanes");
            for lane in 0..4 {ensure!(d[lane]["lane"]==lane && d[lane]["transaction"]["accepted"]==a[lane],
                "frozen accepted differs from the T5 n-gram receipt on lane {lane}");}
        }
        Branch::P=>ensure!(accepted.is_none() && ng["class_b_refusal"]["row"]==127 && ng["class_b_refusal"]["reason"]==t5::CLASS_B,
            "branch P needs the T5 n-gram row-127 initial-decision refusal"),
    }
    Ok(())
}
pub(in super::super) const PRECONDITION_SCHEMA:&str="hf2q.q1.gemma-all128-precondition.v1";
/// Branch P precondition (design §2.6, condition 7): the All128 reference's
/// own row-127 argmax equals frozen[128] on the same workload. When it does
/// not hold, branch P is not run; the record becomes a build-drift localization.
pub(in super::super) fn check_precondition(v:&Value,greedy_workload_sha256:&str,frozen_128:[u32;4])->Result<()> {
    ensure!(v["schema"]==PRECONDITION_SCHEMA && v["workload"]["sha256"]==greedy_workload_sha256 && v["whole_q1_qualified"]==false,
        "bound All128 precondition record of this greedy workload");
    ensure!(v["holds"]==true,"All128 precondition does not hold: branch P is not run (DriftAll128 build-drift localization instead)");
    let lanes=v["lanes"].as_array().context("precondition lanes")?;
    ensure!(lanes.len()==4 && lanes.iter().enumerate().all(|(l,r)|r["lane"]==l && r["frozen_128"]==frozen_128[l] &&
        r["argmax"]==frozen_128[l] && r["equal"]==true),"All128 row-127 argmax equals frozen[128] on every lane");
    Ok(())
}
/// Admission: the packet extent equals the bound manifest, whose T5 stores
/// are the source manifests and whose stage store is the Plan's extent.
pub(in super::super) fn admit_manifest(cfg:&crate::serve::config::Gemma4Config,x:&Extra,m:&Manifest)
    ->Result<(crate::adr063_invariance::gemma_cohort_localizer::Targets,u64,Vec<ExpectedTensor>)> {
    let plan=x.stage_localizer.as_ref().context("stage plan")?;
    ensure!(m.lanes==x.lanes && m.host_payload_limit==x.host_payload_limit,"packet lanes/host bound differ from the bound manifest");
    let stores=t5::stores(cfg,x.arm).map_err(anyhow::Error::msg)?;
    ensure!(m.expected_stores.len()==stores.len()+1,"bound manifest stores are exactly the T5 stores and stages");
    for (name,expected) in &stores {
        ensure!(m.expected_stores.get(*name)==Some(expected),"bound manifest {name} differs from the source manifest");
    }
    let bound=m.expected_stores.get(t5::STAGES).context("bound manifest stage extent")?;
    let (targets,bytes)=localizer::admit(cfg,plan,x.arm,bound).map_err(anyhow::Error::msg)?;
    Ok((targets,bytes,bound.clone()))
}
fn decode(m:&mut crate::serve::api::engine::GemmaLoadedModel,lane:usize,position:usize,input:u32,ns:witness::Namespace)->Result<()> {
    witness::serial(ns,lane,position,1)?;
    let next=m.weights.forward_decode(input,position,&mut m.ctx,&mut None)?;
    witness::end_checked()?;
    t5_decode_row(&m.weights,lane,position,input,next)
}
pub(super) fn run(localizer:bool)->Result<()> {
    use crate::serve::api::engine::{LoadOptions,LoadedModel};
    ensure!(cfg!(target_endian="little"),"LE artifact encoding");
    let packet=std::env::var_os("HF2Q_Q1_GEMMA_T5_PACKET").context("explicit Gemma T5 packet")?;
    let mut raw=Vec::new();File::open(packet)?.take(PACKET_BYTES as u64+1).read_to_end(&mut raw)?;ensure!(raw.len()<=PACKET_BYTES,"packet bound");
    let (p,x)=split(serde_json::from_slice(&raw)?);
    preflight(&p)?;pairing(&x,localizer)?;
    ensure!(p.routing_witness.is_some(),"T5 routing witness is ON in both arms with the exact admitted budget");
    if localizer {ensure!(std::env::var_os("HF2Q_PARALLEL_ENCODE").is_none(),
        "single-row decode stage hooks capture on the owning thread only; HF2Q_PARALLEL_ENCODE refused");}
    let workload=reviewed_workload(&p)?;
    let frozen_accepted=match &x.stage_localizer {Some(plan)=>plan.accepted().map_err(anyhow::Error::msg)?,None=>None};
    if let (Some(r),Some(branch))=(&x.t5_receipts,x.branch) {
        let ord:Value=serde_json::from_slice(&acquisition::checked_file(&r.ordinary,2*1024*1024)?)?;
        let ng:Value=serde_json::from_slice(&acquisition::checked_file(&r.ngram,2*1024*1024)?)?;
        check_receipts(&ord,&ng,branch,frozen_accepted)?;
    }
    let precondition=match &x.all128_precondition {
        Some(b)=>{let frozen:[u32;4]=std::array::from_fn(|l|workload.lanes[l][128]);
            check_precondition(&serde_json::from_slice(&acquisition::checked_file(b,1024*1024)?)?,&p.greedy_workload.sha256,frozen)?;
            Some(json!({"path":b.path,"sha256":b.sha256,"holds":true}))}
        None=>None,
    };
    let acquisition=acquisition::Scope::begin(&p.acquisition,&p.identity,&workload,&p.model,&p.output,&raw)?;
    let header=acquisition::open_model(&p.model)?;ensure!(header.metadata_string("general.architecture")==Some("gemma4"),"explicit Gemma");
    let mut cfg=crate::serve::config::Gemma4Config::from_gguf(&header)?;cfg.max_position_embeddings=262144;
    ensure!(cfg.num_hidden_layers==30 && cfg.hidden_size==2816 && cfg.num_experts==128 && cfg.top_k_experts==8 &&
        cfg.sliding_window>=148 && cfg.vocab_size==262144 && cfg.num_attention_heads==16 &&
        cfg.num_key_value_heads==8 && cfg.num_global_key_value_heads==2 && cfg.head_dim==256 && cfg.global_head_dim==512 &&
        (0..30).all(|l|cfg.is_full_attention(l)==((l+1)%6==0)),"selected MoE geometry before model");
    acquisition.check_config(&p.identity,&selected_recipe(&cfg,true))?;drop(header);
    let stores=t5::stores(&cfg,x.arm).map_err(anyhow::Error::msg)?;
    let stage=match &x.manifest {
        Some(binding)=>{let m:Manifest=serde_json::from_slice(&acquisition::checked_file(binding,MANIFEST_BYTES)?)?;Some(admit_manifest(&cfg,&x,&m)?)}
        None=>None,
    };
    let stage_limit=x.stage_localizer.as_ref().map_or(0,|plan|plan.host_payload_limit);
    let host=t5::host_payload_limit(&cfg,x.arm,stage_limit).map_err(anyhow::Error::msg)?;
    ensure!(x.host_payload_limit==host && host<=p.payload_limit,
        "packet host bound must be exactly the source extent: T5 stores + routing charge + stage plan");
    let _gpu=crate::inference::hf2q_gpu_test_lock();
    let opts=LoadOptions {model_path:p.model.clone(),tokenizer_path:None,config_path:None,dwq_overlay_path:None,
        kv_persist_dir:None,kv_persist_budget_bytes:0,glp_path:None,glp_alpha:None,kv_graft_path:None,lcp_resume_budget:crate::serve::kv_persist::lcp_registry::LcpResumeBudget::DEFAULT};
    let LoadedModel::Gemma(mut m)=LoadedModel::load_with_context(&opts,Some(262144))? else {anyhow::bail!("selected Gemma");};
    acquisition.check_config(&p.identity,&selected_recipe(&m.config,true))?;
    ensure!(m.eos_token_ids==[1,106],"frozen actual EOS");
    let observed=acquisition.observed(m.ctx.executor.device(),&m as *const _ as usize as u64,3)?;
    let scope=Scope::install(&p.output,json!({"schema":if localizer {t5::LOCALIZER_SCHEMA}else{t5::SCHEMA},"identity":p.identity,
        "acquisition":observed,"packet_sha256":hash(&raw),"recipe":selected_recipe(&cfg,true),"arm":x.arm,
        "greedy_workload":p.greedy_workload.sha256,"dataset_review":p.independent_dataset_review.sha256,
        "configured_context":262144,"configured_output_allowance":8192,"serial_owner_epochs":4,"journal_scope":"events only; tensors in the T5 stores",
        "not_mounted_slots":true,"physical_batch4_claim":false,"host_payload_limit":host}))?;
    // From the first store on, every error is a failure receipt: each created
    // store is failed and finished and execution.json is written before the
    // error returns. Stores live in exactly one of these holders at any time.
    let mut loose:Vec<(&'static str,Store)>=Vec::new();
    let mut held:Option<t5::State>=None;
    let mut stage_loose:Option<Store>=None;
    let mut stage_scope:Option<family::Scope>=None;
    let mut routing_scope:Option<witness::Scope>=None;
    let result=(||->Result<()> {
        for (name,expected) in stores {
            let (n,bytes,root)=(expected.len(),t5::payload(&expected).map_err(anyhow::Error::msg)?,p.output.join(name));
            let mut s=Store::new(&root,p.identity.clone(),t5::trajectory(x.arm,name),expected,bytes,n).map_err(anyhow::Error::msg)?;
            let bound=s.bind_acquisition(acquisition::Acquisition {output_root:root,..observed.clone()});
            loose.push((name,s));bound.map_err(anyhow::Error::msg)?;
        }
        ensure!(loose.len()==if x.arm==Arm::NgramTransaction {3}else{2},"T5 store set");
        let mut made=std::mem::take(&mut loose).into_iter().map(|(_,s)|s);
        let (Some(prefork),Some(committed))=(made.next(),made.next()) else {anyhow::bail!("T5 store set")};
        if let Err((state,e))=scope.set_t5(t5::State {arm:x.arm,vocab:cfg.vocab_size,prefork,committed,raw:made.next(),lanes:Vec::new(),
            verifier:Vec::new(),frozen_accepted,pause:None,class_b:None,localizer}) {held=Some(*state);return Err(e);}
        let mode=match x.arm {Arm::OrdinaryGlobal=>witness::Mode::Ordinary,Arm::NgramTransaction=>witness::Mode::Transaction};
        routing_scope=witness::install(m.ctx.executor.device(),&p.output,&p.identity,&observed,mode,p.routing_witness.as_ref())?;
        if let (Some(plan),Some((targets,_,frozen)))=(&x.stage_localizer,stage) {
            let root=p.output.join(t5::STAGES);let n=frozen.len();
            let mut s=Store::new(&root,p.identity.clone(),plan.rerun_id.clone(),frozen,plan.host_payload_limit,n).map_err(anyhow::Error::msg)?;
            if plan.prefill_stages.iter().chain(&plan.decode_stages).any(|s|matches!(s,Stage::RouterLogits|Stage::RouterIds|Stage::RouterWeights)) {
                s.missing_capability("Gemma actual native post-softmax/renormalized selection kth+1 GPU score is not a captured stage output; the routing witness carries the selection-domain margins");
            }
            if let Err(e)=s.bind_acquisition(acquisition::Acquisition {output_root:root,..observed.clone()}) {stage_loose=Some(s);anyhow::bail!(e);}
            match family::install_or_return(Family::Gemma4,s,targets) {
                Ok(installed)=>stage_scope=Some(installed),
                Err((e,s))=>{stage_loose=Some(*s);anyhow::bail!(e);}
            }
        }
        for lane in 0..4 {
            // Checked prior owner already returned. Clear global active banks, no physical-refund claim.
            m.weights.hybrid_kv=None;m.weights.dense_kvs=None;m.weights.leg_hb_encoded=None;
            m.weights.hybrid_kv_snapshot_for_lcp=None;m.weights.dense_kvs_snapshot_for_lcp=None;
            for c in &mut m.weights.kv_caches {c.write_pos=0;c.seq_len=0;}
            m.weights.decode_step=0;scope.owner(lane,&workload.lanes[lane])?;
            let tokens=workload.lanes[lane].clone();
            // Every prefill row and single-row decode row of this lane is owned (lane, position).
            let _lane=if localizer {Some(crate::adr063_invariance::gemma_prefill::serial_lane(lane)?)}else{None};
            match x.arm {
                Arm::OrdinaryGlobal=>{
                    witness::serial(witness::Namespace::Canonical,lane,0,128)?;
                    let first=m.weights.forward_prefill_batched(&tokens[..128],8192,0,&mut m.ctx)?;
                    witness::end_checked()?;
                    t5_ordinary_initial(&m.weights,lane,first)?;
                    for (position,input) in t5::decode_inputs(Arm::OrdinaryGlobal,&tokens,None)? {
                        decode(&mut m,lane,position,input,witness::Namespace::Canonical)?;
                    }
                }
                Arm::NgramTransaction=>{
                    // G5-A1's identical initial call is not captured again: the
                    // pause is released by the `initial` hook, keeping step 0 for the verify.
                    if localizer {with(|a|{a.t5.as_mut().context("T5 state")?.pause=Some(family::pause().map_err(anyhow::Error::msg)?);Ok(())})?;}
                    let output=crate::inference::spec_decode::ngram_orchestrator::dispatch_ngram_generate(
                        &mut m.weights,&tokens[..128],8192,3,1,3,&m.eos_token_ids,&mut m.ctx)?;
                    let a=t5_accepted(lane)?;
                    ensure!(output.len()==130+a && !family::paused(),"one committed real transaction; pause released at the initial hook");
                    for (position,input) in t5::decode_inputs(Arm::NgramTransaction,&tokens,Some(a))? {
                        decode(&mut m,lane,position,input,witness::Namespace::T5Continuation)?;
                    }
                }
            }
            t5_lane_end(&m.weights,lane)?;
        }Ok(())
    })();
    // Run, acquisition and routing-witness failures all mark every store failed.
    let failure=result.as_ref().err().map(|e|format!("{e:#}"));
    let unchanged=acquisition.check_unchanged();
    let witness_end=if failure.is_none() {witness::finish().err().map(|e|format!("routing witness: {e:#}"))} else {None};
    let abort=failure.clone().or(unchanged.as_ref().err().map(|e|format!("{e:#}"))).or(witness_end);
    let mut state=scope.take_t5().ok().or(held);
    if let Some(s)=state.as_mut() {drop(s.pause.take());}
    let stage_trace=match (stage_scope.is_some(),stage_loose) {
        (true,_)=>{if let Some(e)=&abort {family::fail(e.clone());}Some(family::finish())}
        (false,Some(mut s))=>{if let Some(e)=&abort {s.fail(e.clone());}Some(s.finish())}
        (false,None)=>None,
    };
    let (decisions,class_b,made)=match state {
        Some(t5::State {prefork,committed,raw,lanes,class_b,..})=>{
            let mut v=vec![(t5::PREFORK,prefork),(t5::COMMITTED,committed)];
            if let Some(r)=raw {v.push((t5::RAW,r));}(lanes,class_b,v)
        }
        None=>(Vec::new(),None,loose),
    };
    let mut counts=serde_json::Map::new();let mut incomplete=Vec::new();
    for (name,mut store) in made {
        if let Some(e)=&abort {store.fail(e.clone());}
        match store.finish() {
            Ok(trace)=>{counts.insert(name.into(),json!(trace.coverage.captured.len()));
                if !trace.coverage.errors.is_empty() {incomplete.push(format!("{name}: {:?}",trace.coverage.errors));}}
            Err(e)=>{counts.insert(name.into(),Value::Null);incomplete.push(format!("{name}: {e}"));}
        }
    }
    let mut stage_records=0;
    match &stage_trace {
        Some(Ok(trace))=>{stage_records=trace.coverage.captured.len();
            if !trace.coverage.errors.is_empty() {incomplete.push(format!("{}: {:?}",t5::STAGES,trace.coverage.errors));}}
        Some(Err(e))=>incomplete.push(format!("{}: {e}",t5::STAGES)),
        None=>{}
    }
    drop((stage_scope,routing_scope));
    let error=abort.or((!incomplete.is_empty()).then(||format!("incomplete stores: {}",incomplete.join("; "))));
    let status=if error.is_none() {"CompleteScopedAcquisition"}else{"FailedOrIncomplete"};
    let _=with(|a|a.journal.event("execution_result",json!({"failure":error,"class_b_refusal":class_b,"whole_q1_qualified":false})));
    let records=counts.values().filter_map(Value::as_u64).sum::<u64>()+stage_records as u64;
    let mut report=json!({"schema":if localizer {t5::LOCALIZER_SCHEMA}else{t5::SCHEMA},"packet_sha256":hash(&raw),"arm":x.arm,
        "status":status,"error":error,"lanes":[0,1,2,3],"decisions":decisions,"class_b_refusal":class_b,"stores":counts,
        "routing_witness":"routing-witness.jsonl","whole_q1_qualified":false});
    if localizer {
        let extra=json!({"branch":x.branch,"pass":x.pass,"plan":x.stage_localizer,"stage_records":stage_records,
            "all128_precondition":precondition});
        report.as_object_mut().context("report object")?.extend(extra.as_object().context("localizer receipt")?.clone());
    }
    OpenOptions::new().create_new(true).write(true).mode(0o600).open(p.output.join("execution.json"))?
        .write_all(&serde_json::to_vec_pretty(&report)?)?;
    scope.finish(status)?;
    result?;unchanged?;
    if let Some(e)=error {anyhow::bail!(e);}
    println!("{} records={records} whole_q1_qualified=false",if localizer {t5::LOCALIZER_MARKER}else{t5::MARKER});
    Ok(())
}
