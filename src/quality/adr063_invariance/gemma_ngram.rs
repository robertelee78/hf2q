//! Test-only observer over REAL Gemma n-gram proposal/verify/accept/rollback.
use anyhow::{ensure,Context,Result};
use mlx_native::MlxBuffer;
use serde_json::{json,Value};
use std::cell::RefCell;
use crate::inference::models::gemma4::MlxModelWeights;
#[path="gemma_ngram/journal.rs"]mod journal;
#[path="gemma_ngram/runner.rs"]mod runner;
#[cfg(test)]#[path="gemma_ngram/controls.rs"]mod controls;
pub use journal::{PAYLOAD,METADATA};
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
struct Active {journal:journal::Journal,owner:Option<Owner>}
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
impl Scope {
    fn install(root:&std::path::Path,header:Value)->Result<Self> {
        ACTIVE.with(|a|{ensure!(a.borrow().is_none(),"nested Gemma ngram diagnostic");Ok::<_,anyhow::Error>(())})?;
        let mut journal=journal::Journal::new(root)?;journal.event("header",header)?;
        ACTIVE.with(|a|*a.borrow_mut()=Some(Active {journal,owner:None}));Ok(Self)
    }
    fn owner(&self,lane:usize,canonical:&[u32])->Result<()> {
        with(|a|{ensure!(lane<4 && a.owner.as_ref().is_none_or(|o|o.phase==Phase::Committed),"one serial owner/transaction");
            let injected=fixture(canonical,262144)?;
            a.owner=Some(Owner {lane,canonical:canonical.to_vec(),fixture:injected,phase:Phase::Ready,rows:Vec::new(),output_before:Vec::new(),accepted:0,rejected:0});Ok(())})?.context("diagnostic missing")?;Ok(())
    }
    fn observed_end(&self,m:&MlxModelWeights,output:&[u32],carry:u32)->Result<()> {
        with(|a|{let o=a.owner.as_ref().context("owner")?;transition(o.phase,Phase::Committed)?;
            ensure!(output.len()==144,"complete actual144 inputs, not forced dataset");
            let actual=a.journal.snapshot(m,o.lane,"ordinary_final144")?;ensure!(actual==144,"actual final cursor");
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
/// Source-owned serial epoch, installed BEFORE the actual initial forward.
pub fn routing_before_initial()->Result<()> {with(|a|{let o=a.owner.as_ref().context("owner")?;
    transition(o.phase,Phase::Ready)?;super::gemma_routing_witness::serial(
        super::gemma_routing_witness::Namespace::T5Initial,o.lane,0,128)})?;Ok(())}
pub fn initial(m:&MlxModelWeights,prompt:&[u32],id:u32)->Result<()> {with(|a|{
    let o=a.owner.as_mut().context("owner")?;transition(o.phase,Phase::Ready)?;
    ensure!(prompt==&o.canonical[..128],"same actual first128 required");
    a.journal.tensor(json!({"owner_epoch":o.lane,"phase":"initial_head","position":127}),&m.activations.logits)?;
    ensure!(a.journal.snapshot(m,o.lane,"before_real_verify")?==128,"actual128 cursor");
    a.journal.event("initial_gpu_selection",json!({"owner_epoch":o.lane,"actual_gpu_id":id,"canonical_id":o.canonical[128],"source":"actual ordinary returned GPU argmax"}))?;
    o.phase=Phase::Initial;Ok(())})?;super::gemma_routing_witness::end_checked()?;Ok(())}
pub fn proposal(output:&[u32],original:&[u32],k:u32)->Result<Option<Vec<u32>>> {with(|a|{
    let o=a.owner.as_mut().context("owner")?;transition(o.phase,Phase::Initial)?;
    ensure!(k==3 && output.len()==129,"bounded single real K3 transaction");o.output_before=output.to_vec();
    a.journal.event("proposal",json!({"owner_epoch":o.lane,"actual_ngram_proposer_ran":true,
        "actual_production_ngram_proposal":original,"injected_proposal":o.fixture,"input_history":output,
        "fixture_descriptor":"canonical129/130 plus deterministic nonEOS mutation of canonical131; proposal only, no forced acceptance",
        "proposal_origin":"InjectedDiagnosticFixtureNotProductionProposalQuality"}))?;o.phase=Phase::Proposed;Ok(o.fixture.clone())})}
pub fn before_verify(prefix:&[u32],start:usize)->Result<()> {with(|a|{
    let o=a.owner.as_mut().context("owner")?;transition(o.phase,Phase::Proposed)?;
    let mut expected=o.output_before.clone();expected.extend(&o.fixture);
    ensure!(start==0 && prefix==expected && prefix.len()==132,"actual OptionC prefix/geometry");
    a.journal.event("verify_input",json!({"owner_epoch":o.lane,"actual_prefix":prefix,"start":start,"route":"production OptionC full re-prefill, no xlen/mounted substitution"}))?;
    super::gemma_routing_witness::serial(super::gemma_routing_witness::Namespace::T5Verify,o.lane,0,132)?;
    o.phase=Phase::Verifying;Ok(())})?;Ok(())}
pub fn after_verify(m:&MlxModelWeights)->Result<()> {with(|a|{
    let o=a.owner.as_mut().context("owner")?;transition(o.phase,Phase::Verifying)?;
    ensure!(a.journal.snapshot(m,o.lane,"after_checked_verifier_before_accept")?==132,"actual132 cursor");
    o.phase=Phase::Verified;Ok(())})?;super::gemma_routing_witness::end_checked()?;Ok(())}
pub fn argmax_row(row:usize,b:&MlxBuffer,id:u32)->Result<()> {with(|a|{
    let o=a.owner.as_mut().context("owner")?;
    if o.phase!=Phase::Verified {return Ok(());}
    ensure!(row==o.rows.len() && row<4 && b.data_byte_len()==262144*4,"all actual verifier head rows");
    a.journal.tensor(json!({"owner_epoch":o.lane,"phase":"actual_verifier_logits","verify_row":row,"position":128+row,"actual_gpu_argmax":id}),b)?;
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
        "producer":"production accept_prefix_argmax; no diagnostic acceptance override"}))?;o.phase=Phase::Decided;Ok(())})?;Ok(())}
pub fn after_rollback(m:&MlxModelWeights,output:&[u32],rejected:usize)->Result<bool> {Ok(with(|a|{
    let o=a.owner.as_mut().context("owner")?;transition(o.phase,Phase::Decided)?;
    ensure!(rejected==o.rejected && output.len()==130+o.accepted,"actual acceptance output cardinality");
    let pos=a.journal.snapshot(m,o.lane,"after_actual_rollback")?;require_commit(pos,output.len(),rejected)?;
    a.journal.event("commit",json!({"owner_epoch":o.lane,"actual_output":output,"actual_rejected_count":rejected,
        "actual_cached_input_count":pos,"unprocessed_fallback_id":output.last(),
        "stop_after_one_transaction":"explicit test-only finite observation boundary"}))?;o.phase=Phase::Committed;Ok(true)})?.unwrap_or(false))}
