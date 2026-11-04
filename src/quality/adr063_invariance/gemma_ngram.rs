//! Test-only observer over REAL Gemma n-gram proposal/verify/accept/rollback.
use anyhow::{ensure,Context,Result};
use mlx_native::MlxBuffer;
use serde_json::{json,Value};
use std::cell::RefCell;
use crate::inference::models::gemma4::MlxModelWeights;
#[path="gemma_ngram/journal.rs"]mod journal;
#[path="gemma_ngram/runner.rs"]mod runner;
#[path="gemma_ngram/t5.rs"]mod t5;
/// gemma_ngram_localizer: the §2.6 verifier-route localizer extent.
#[path="gemma_ngram/localizer.rs"]mod localizer;
#[path="gemma_ngram/compare.rs"]mod compare;
#[path="gemma_ngram/export.rs"]mod export;
#[cfg(test)]#[path="gemma_ngram/controls.rs"]mod controls;
#[cfg(test)]#[path="gemma_ngram/t5_controls.rs"]mod t5_controls;
pub use journal::{PAYLOAD,METADATA};
/// cfg(test)-only metadata visibility; no model/config/acquisition bypass.
pub(super) fn configuration_recipe(cfg:&crate::serve::config::Gemma4Config,routing:bool)->crate::adr063_invariance::acquisition::ConfigRecipe<'static> {
    runner::selected_recipe(cfg,routing)
}

fn require_selected_profile(hybrid:bool,dense:bool,allocation_bits:u32,effective_bits:u32,full_f16:bool)->Result<()> {
    ensure!(hybrid && !dense && allocation_bits==8 && effective_bits==8 && !full_f16,
        "T5 requires actual selected Hybrid/8bit/non-full-F16 before model load");Ok(())
}
fn full_f16_enabled(raw:Option<&str>)->bool {matches!(raw,Some("1"|"true"|"on"))}
fn selected_profile()->Result<()> {
    let env=&*crate::debug::INVESTIGATION_ENV;
    let full=std::env::var("HF2Q_FULL_F16_KV").ok();
    require_selected_profile(env.hybrid_kv,env.use_dense,env.tq_codebook_bits,
        crate::serve::api::tq_packed_descriptor::effective_tq_codebook_bits(),full_f16_enabled(full.as_deref()))
}
fn planned_payload_upper()->u64 {16u64*148*(11*56320+4*220)+84*262144*4}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]enum Phase {Ready,Initial,Proposed,Verifying,Verified,Decided,Committed}
struct Owner {lane:usize,canonical:Vec<u32>,fixture:Vec<u32>,phase:Phase,rows:Vec<u32>,output_before:Vec<u32>,accepted:usize,rejected:usize}
/// `t5` present = Q1 T5 mode: tensors go to the T5 stores, the journal keeps events only.
struct Active {journal:journal::Journal,owner:Option<Owner>,t5:Option<t5::State>}
thread_local!{static ACTIVE:RefCell<Option<Active>>=const{RefCell::new(None)};}
pub struct Scope;
impl Drop for Scope {fn drop(&mut self){ACTIVE.with(|a|{if let Some(mut a)=a.borrow_mut().take(){let _=a.journal.finish("AbandonedIncomplete");}});}}
fn with<T>(f:impl FnOnce(&mut Active)->Result<T>)->Result<Option<T>> {
    ACTIVE.with(|a|{let mut a=a.borrow_mut();a.as_mut().map(f).transpose()})
}
fn fixture(canonical:&[u32],vocab:u32)->Result<Vec<u32>> {
    ensure!(canonical.len()==144 && vocab>2 && canonical.iter().all(|&id|id<vocab),"frozen complete144 fixture");
    let mut wrong=(canonical[131]+1)%vocab;
    while [1,106].contains(&wrong){wrong=(wrong+1)%vocab;}
    ensure!(wrong!=canonical[131],"fixture mutation must differ");Ok(vec![canonical[129],canonical[130],wrong])
}
fn coverage(accepted:usize,rejected:usize)->(bool,bool) {(accepted>0,rejected>0)}
fn require_commit(pos:usize,output_len:usize,rejected:usize)->Result<()> {
    ensure!(output_len>=2 && rejected<=3 && pos==output_len-1 && pos==132-rejected,"actual rollback cursor, not invented commit");Ok(())
}
fn transition(actual:Phase,expected:Phase)->Result<()> {ensure!(actual==expected,"actual transaction order mismatch");Ok(())}
/// Tensor sink of the v3 journal or, in T5 mode, events only.
fn events_sink(t5:&Option<t5::State>)->journal::Sink<'static> {if t5.is_some() {journal::Sink::Events}else{journal::Sink::Journal}}
impl Scope {
    fn install(root:&std::path::Path,header:Value)->Result<Self> {
        ACTIVE.with(|a|{ensure!(a.borrow().is_none(),"nested Gemma ngram diagnostic");Ok::<_,anyhow::Error>(())})?;
        let mut journal=journal::Journal::new(root)?;journal.event("header",header)?;
        ACTIVE.with(|a|*a.borrow_mut()=Some(Active {journal,owner:None,t5:None}));Ok(Self)
    }
    /// Binds the T5 stores; on refusal the state comes back to the caller,
    /// which still owns (and must fail+finish) every created store.
    fn set_t5(&self,state:t5::State)->std::result::Result<(),(Box<t5::State>,anyhow::Error)> {
        ACTIVE.with(|a|match a.borrow_mut().as_mut() {
            Some(a) if a.t5.is_none() && a.owner.is_none()=>{a.t5=Some(state);Ok(())}
            _=>Err((Box::new(state),anyhow::anyhow!("T5 stores bind before the first owner of an installed diagnostic"))),
        })
    }
    fn take_t5(&self)->Result<t5::State> {with(|a|a.t5.take().context("T5 state missing"))?.context("diagnostic missing")}
    fn owner(&self,lane:usize,canonical:&[u32])->Result<()> {
        with(|a|{ensure!(lane<4 && a.owner.as_ref().is_none_or(|o|o.phase==Phase::Committed),"one serial owner/transaction");
            let injected=fixture(canonical,262144)?;
            if let Some(t)=a.t5.as_mut() {t.begin(lane,canonical)?;}
            a.owner=Some(Owner {lane,canonical:canonical.to_vec(),fixture:injected,phase:Phase::Ready,rows:Vec::new(),output_before:Vec::new(),accepted:0,rejected:0});Ok(())})?.context("diagnostic missing")?;Ok(())
    }
    fn observed_end(&self,m:&MlxModelWeights,output:&[u32],carry:u32)->Result<()> {
        with(|a|{let o=a.owner.as_ref().context("owner")?;transition(o.phase,Phase::Committed)?;
            ensure!(a.t5.is_none(),"v3 observed lineage is not a T5 teacher-forced continuation");
            ensure!(output.len()==144,"complete actual144 inputs, not forced dataset");
            let actual=a.journal.snapshot(m,o.lane,"ordinary_final144",journal::Sink::Journal)?;ensure!(actual==144,"actual final cursor");
            let mismatch=output.iter().zip(&o.canonical).position(|(x,y)|x!=y);
            a.journal.event("observed_lineage",json!({"owner_epoch":o.lane,"actual_inputs":output,"carry_g16":carry,
                "first_canonical_input_mismatch":mismatch,"incomparable_after_fork":mismatch.is_some(),
                "accepted_transition_observed":coverage(o.accepted,o.rejected).0,"rejected_transition_observed":coverage(o.accepted,o.rejected).1,
                "ordinary_ngram_proposal_quality_qualified":false,"final_dataset_approved":false}))?;Ok(())})?;Ok(())
    }
    fn finish(self,status:&str)->Result<()> {
        ACTIVE.with(|a|{let mut a=a.borrow_mut().take().context("missing journal")?;a.journal.finish(status)})
    }
}
/// T5 ordinary-global arm, after its own `forward_prefill_batched(lane[..128])`:
/// head row 127 and cache@128 into `prefork`. A differing initial argmax is a
/// cross-route Class B observation here, never a refusal.
fn t5_ordinary_initial(m:&MlxModelWeights,lane:usize,id:u32)->Result<()> {with(|a|{
    let o=a.owner.as_ref().context("owner")?;transition(o.phase,Phase::Ready)?;ensure!(o.lane==lane,"T5 owner lane");
    let t=a.t5.as_mut().context("T5 state")?;ensure!(t.arm==t5::Arm::OrdinaryGlobal,"ordinary initial in the ordinary arm only");
    t5::head_row(&mut t.prefork,m,t.vocab,lane,127)?;
    ensure!(a.journal.snapshot(m,lane,"ordinary_global_initial128",journal::Sink::Store(&mut t.prefork,t5::CACHE_RECORDS))?==128,"actual128 cursor");
    let frozen=o.canonical[128];let l=t.lane(lane)?;l.initial_argmax=Some(id);l.initial_equals_frozen=Some(id==frozen);
    a.journal.event("initial_gpu_selection",json!({"owner_epoch":lane,"actual_gpu_id":id,"canonical_id":frozen,
        "cross_route_class_b_observation":id!=frozen,"source":"actual ordinary-global returned GPU argmax"}))?;Ok(())
})?.context("T5 diagnostic missing")}
/// One teacher-forced `forward_decode(frozen[p], p)` row: committed `Logits`
/// row p; the input is the frozen ID, the returned argmax only a decision.
fn t5_decode_row(m:&MlxModelWeights,lane:usize,position:usize,input:u32,next:u32)->Result<()> {
    t5_teacher_forced(lane,position,input,next)?;
    with(|a|{let t=a.t5.as_mut().context("T5 state")?;t5::head_row(&mut t.committed,m,t.vocab,lane,position)})?.context("T5 diagnostic missing")
}
/// Model-free half of [`t5_decode_row`]: the input must be frozen[p]; the
/// returned argmax is recorded as a decision only.
fn t5_teacher_forced(lane:usize,position:usize,input:u32,next:u32)->Result<()> {with(|a|{
    let o=a.owner.as_ref().context("owner")?;let t=a.t5.as_mut().context("T5 state")?;
    transition(o.phase,if t.arm==t5::Arm::OrdinaryGlobal {Phase::Ready}else{Phase::Committed})?;
    ensure!(o.lane==lane && o.canonical.get(position)==Some(&input),"teacher-forced decode input must be frozen[p]");
    let l=t.lane(lane)?;ensure!(l.decode_inputs.last().is_none_or(|r|r.position+1==position),"teacher-forced rows in position order");
    l.decode_inputs.push(t5::Row {position,token:input});l.decode_argmax.push(t5::Row {position,token:next});
    a.journal.event("teacher_forced_decode",json!({"owner_epoch":lane,"position":position,"frozen_input":input,"actual_gpu_next_id":next}))?;Ok(())
})?.context("T5 diagnostic missing")}
/// Cache@144 into `committed`; closes the lane owner.
fn t5_lane_end(m:&MlxModelWeights,lane:usize)->Result<()> {
    t5_lane_inputs_checked(lane)?;
    with(|a|{let o=a.owner.as_mut().context("owner")?;let t=a.t5.as_mut().context("T5 state")?;
        ensure!(a.journal.snapshot(m,lane,"teacher_forced_final144",journal::Sink::Store(&mut t.committed,t5::CACHE_RECORDS))?==144,"actual final cursor");
        o.phase=Phase::Committed;Ok(())
    })?.context("T5 diagnostic missing")
}
/// Model-free half of [`t5_lane_end`]: the lane's decode inputs are exactly
/// the frozen continuation (ordinary 128..143, n-gram 129+a..143).
fn t5_lane_inputs_checked(lane:usize)->Result<()> {with(|a|{
    let o=a.owner.as_ref().context("owner")?;let t=a.t5.as_mut().context("T5 state")?;
    transition(o.phase,if t.arm==t5::Arm::OrdinaryGlobal {Phase::Ready}else{Phase::Committed})?;ensure!(o.lane==lane,"T5 owner lane");
    let inputs=t5::decode_inputs(t.arm,&o.canonical,(t.arm==t5::Arm::NgramTransaction).then_some(o.accepted))?;
    ensure!(t.lane(lane)?.decode_inputs.iter().map(|r|(r.position,r.token)).eq(inputs.iter().copied()),"teacher-forced inputs differ from the frozen continuation");
    Ok(())
})?.context("T5 diagnostic missing")}
/// The n-gram lane's real accepted count after its committed transaction.
fn t5_accepted(lane:usize)->Result<usize> {with(|a|{
    let o=a.owner.as_ref().context("owner")?;transition(o.phase,Phase::Committed)?;ensure!(o.lane==lane,"T5 owner lane");Ok(o.accepted)
})?.context("T5 diagnostic missing")}
/// Source-owned serial epoch, installed BEFORE the actual initial forward.
pub fn routing_before_initial()->Result<()> {with(|a|{let o=a.owner.as_ref().context("owner")?;
    transition(o.phase,Phase::Ready)?;super::gemma_routing_witness::serial(
        super::gemma_routing_witness::Namespace::T5Initial,o.lane,0,128)})?;Ok(())}
fn initial_owner(a:&Active,prompt:&[u32])->Result<(usize,u32)> {
    let o=a.owner.as_ref().context("owner")?;transition(o.phase,Phase::Ready)?;
    ensure!(prompt==&o.canonical[..128],"same actual first128 required");Ok((o.lane,o.canonical[128]))
}
pub fn initial(m:&MlxModelWeights,prompt:&[u32],id:u32)->Result<()> {
    with(|a|{let (lane,_)=initial_owner(a,prompt)?;
        match a.t5.as_mut() {
            None=>{a.journal.tensor(json!({"owner_epoch":lane,"phase":"initial_head","position":127}),&m.activations.logits)?;
                ensure!(a.journal.snapshot(m,lane,"before_real_verify",journal::Sink::Journal)?==128,"actual128 cursor");}
            Some(t)=>{
                // The localizer pause covered exactly G5-A1's identical initial call.
                ensure!(t.pause.take().is_some()==t.localizer,"localizer pause must cover exactly the initial prefill");
                t5::head_row(&mut t.prefork,m,t.vocab,lane,127)?;
                ensure!(a.journal.snapshot(m,lane,"before_real_verify",journal::Sink::Store(&mut t.prefork,t5::CACHE_RECORDS))?==128,"actual128 cursor");
            }
        }Ok(())})?;
    initial_decided(prompt,id)
}
/// Model-free tail of [`initial`]: the initial decision, its relation to
/// frozen[128] and, in the n-gram T5 arm, the Class B row-127 refusal.
fn initial_decided(prompt:&[u32],id:u32)->Result<()> {let refusal=with(|a|{
    let (lane,frozen)=initial_owner(a,prompt)?;let mut refusal=None;
    if let Some(t)=a.t5.as_mut() {
        let l=t.lane(lane)?;l.initial_argmax=Some(id);l.initial_equals_frozen=Some(id==frozen);
        refusal=t5::initial_refusal(t.arm,lane,id,frozen);
        if refusal.is_some() {t.class_b=refusal.clone();}
    }
    a.journal.event("initial_gpu_selection",json!({"owner_epoch":lane,"actual_gpu_id":id,"canonical_id":frozen,"source":"actual ordinary returned GPU argmax"}))?;
    a.owner.as_mut().context("owner")?.phase=Phase::Initial;Ok(refusal)})?.flatten();super::gemma_routing_witness::end_checked()?;
    if refusal.is_some() {anyhow::bail!(t5::CLASS_B);}Ok(())}
pub fn proposal(output:&[u32],original:&[u32],k:u32)->Result<Option<Vec<u32>>> {with(|a|{
    let o=a.owner.as_mut().context("owner")?;transition(o.phase,Phase::Initial)?;
    ensure!(k==3 && output.len()==129,"bounded single real K3 transaction");o.output_before=output.to_vec();
    a.journal.event("proposal",json!({"owner_epoch":o.lane,"actual_ngram_proposer_ran":true,
        "actual_production_ngram_proposal":original,"injected_proposal":o.fixture,"input_history":output,
        "fixture_descriptor":"canonical129/130 plus deterministic nonEOS mutation of canonical131; proposal only, no forced acceptance",
        "proposal_origin":"InjectedDiagnosticFixtureNotProductionProposalQuality"}))?;
    if let Some(t)=a.t5.as_mut() {let x=t.lane(o.lane)?.transaction.as_mut().context("n-gram transaction record")?;
        x.proposer_ran=true;x.proposed=original.to_vec();x.injected=o.fixture.clone();}
    o.phase=Phase::Proposed;Ok(o.fixture.clone())})}
pub fn before_verify(prefix:&[u32],start:usize)->Result<()> {with(|a|{
    let o=a.owner.as_mut().context("owner")?;transition(o.phase,Phase::Proposed)?;
    let mut expected=o.output_before.clone();expected.extend(&o.fixture);
    ensure!(start==0 && prefix==expected && prefix.len()==132,"actual OptionC prefix/geometry");
    if a.t5.as_ref().is_some_and(|t|t.localizer) {
        // Segment-1 verify rows are owned (lane, 0..131) by the serial lease.
        localizer::verify_owners(o.lane,start,prefix.len()).map_err(anyhow::Error::msg)?;
        ensure!(super::gemma_prefill::current_serial_lane()==Some(o.lane) && !super::family::paused(),
            "verify re-prefill stage rows need the lane lease and no pause");
    }
    a.journal.event("verify_input",json!({"owner_epoch":o.lane,"actual_prefix":prefix,"start":start,"route":"production OptionC full re-prefill, no xlen/mounted substitution"}))?;
    super::gemma_routing_witness::serial(super::gemma_routing_witness::Namespace::T5Verify,o.lane,0,132)?;
    o.phase=Phase::Verifying;Ok(())})?;Ok(())}
pub fn after_verify(m:&MlxModelWeights)->Result<()> {with(|a|{
    let o=a.owner.as_mut().context("owner")?;transition(o.phase,Phase::Verifying)?;
    ensure!(a.journal.snapshot(m,o.lane,"after_checked_verifier_before_accept",events_sink(&a.t5))?==132,"actual132 cursor");
    o.phase=Phase::Verified;Ok(())})?;super::gemma_routing_witness::end_checked()?;Ok(())}
pub fn argmax_row(row:usize,b:&MlxBuffer,id:u32)->Result<()> {with(|a|{
    let o=a.owner.as_mut().context("owner")?;
    if o.phase!=Phase::Verified {return Ok(());}
    ensure!(row==o.rows.len() && row<4 && b.data_byte_len()==262144*4,"all actual verifier head rows");
    match a.t5.as_mut() {
        None=>a.journal.tensor(json!({"owner_epoch":o.lane,"phase":"actual_verifier_logits","verify_row":row,"position":128+row,"actual_gpu_argmax":id}),b)?,
        Some(t)=>{let data=b.as_logical_slice::<u8>()?;
            ensure!(data.len()==t.vocab*4 && t.verifier.len()==row*t.vocab*4 &&
                data.chunks_exact(4).all(|x|f32::from_le_bytes(x.try_into().unwrap()).is_finite()),"finite post-softcap verifier row");
            t.verifier.extend_from_slice(data);}
    }
    o.rows.push(id);Ok(())})?;Ok(())}
pub fn argmax_matrix(b:&MlxBuffer,ids:&[u32])->Result<()> {
    if !ACTIVE.with(|a|a.borrow().as_ref().and_then(|a|a.owner.as_ref()).is_some_and(|o|o.phase==Phase::Verified)) {return Ok(());}
    for (row,&id) in ids.iter().enumerate(){let view=b.slice_view((row*262144*4)as u64,262144).with_shape(vec![1,262144])?;argmax_row(row,&view,id)?;}Ok(())
}
pub fn decision(drafts:&[u32],argmaxes:&[u32],accepted:usize,fallback:u32)->Result<()> {with(|a|{
    let o=a.owner.as_mut().context("owner")?;transition(o.phase,Phase::Verified)?;
    ensure!(o.rows==argmaxes && argmaxes.len()==4 && drafts==o.fixture && accepted<=drafts.len(),"actual GPU vectors/production acceptance join");
    o.accepted=accepted;o.rejected=drafts.len()-accepted;
    a.journal.event("acceptance",json!({"owner_epoch":o.lane,"actual_argmaxes":argmaxes,"actual_accepted_count":accepted,
        "actual_accepted_ids":&drafts[..accepted],"actual_rejected_suffix":&drafts[accepted..],"actual_fallback":fallback,
        "producer":"production accept_prefix_argmax; no diagnostic acceptance override"}))?;
    if let Some(t)=a.t5.as_mut() {
        t5::require_accepted(accepted)?;
        if let Some(frozen)=t.frozen_accepted {ensure!(frozen[o.lane]==accepted,
            "observed accepted count {accepted} differs from the frozen localizer plan {}",frozen[o.lane]);}
        let rows=std::mem::take(&mut t.verifier);
        // All four post-softcap rows to raw-verifier; rows 128..128+a are committed.
        t5::raw_record(t.raw.as_mut().context("raw-verifier store")?,&rows,t.vocab,o.lane)?;
        for i in 0..=accepted {t5::verifier_row(&mut t.committed,&rows,t.vocab,o.lane,i)?;}
        let x=t.lane(o.lane)?.transaction.as_mut().context("n-gram transaction record")?;
        x.verifier_argmax=argmaxes.to_vec();x.accepted=Some(accepted);x.rejected=Some(drafts.len()-accepted);x.fallback=Some(fallback);
    }
    o.phase=Phase::Decided;Ok(())})?;Ok(())}
fn rollback_owner(a:&Active,output:&[u32],rejected:usize)->Result<usize> {
    let o=a.owner.as_ref().context("owner")?;transition(o.phase,Phase::Decided)?;
    ensure!(rejected==o.rejected && output.len()==130+o.accepted,"actual acceptance output cardinality");Ok(o.lane)
}
pub fn after_rollback(m:&MlxModelWeights,output:&[u32],rejected:usize)->Result<bool> {
    let Some(pos)=with(|a|{let lane=rollback_owner(a,output,rejected)?;a.journal.snapshot(m,lane,"after_actual_rollback",events_sink(&a.t5))})? else {return Ok(false)};
    rollback_committed(output,rejected,pos)
}
/// Model-free tail of [`after_rollback`]: the real cursor `pos` must be the
/// committed one (`132 - rejected`; in T5 mode exactly 129+a).
fn rollback_committed(output:&[u32],rejected:usize,pos:usize)->Result<bool> {Ok(with(|a|{
    rollback_owner(a,output,rejected)?;let o=a.owner.as_mut().context("owner")?;require_commit(pos,output.len(),rejected)?;
    if let Some(t)=a.t5.as_mut() {t5::require_cursor(pos,o.accepted)?;
        t.lane(o.lane)?.transaction.as_mut().context("n-gram transaction record")?.rollback_cursor=Some(pos);}
    a.journal.event("commit",json!({"owner_epoch":o.lane,"actual_output":output,"actual_rejected_count":rejected,
        "actual_cached_input_count":pos,"unprocessed_fallback_id":output.last(),
        "stop_after_one_transaction":"explicit test-only finite observation boundary"}))?;o.phase=Phase::Committed;Ok(true)})?.unwrap_or(false))}
