//! Partial Q38 ordinary target graph: singleton attention, shared RMS/FFN/head.
//! No scheduler, speculation, sampling or environment activation lives here.
use super::*;
use crate::serve::scheduler::SlotHandle;
use std::{cell::Cell,collections::BTreeSet};
mod graph;
mod retained;
#[derive(Clone,Debug)]pub(crate) struct Row {pub request:u64,pub handle:SlotHandle,pub position:u32,pub token:u32}
/// Supplied by the actual owning worker immediately before submission.
/// `speculative` means an ACTIVE proposal/verifier, NOT an MTP continuation.
#[derive(Clone,Debug)]pub(crate) struct Owner {pub request:u64,pub handle:SlotHandle,pub position:u32,pub cancelled:bool,pub speculative:bool,pub vision:bool}
#[derive(Debug)]pub(crate) struct OperatorMap {pub layer:Option<usize>,pub operator:&'static str,pub rows:Vec<Row>,pub physical_rows:usize}
pub(crate) struct Output {
    pub logits:Vec<Vec<f32>>,
    /// Real durable post-output-RMS rows in the exact input owner order.
    /// Views may share one parent; retain that parent until ALL owners release.
    pub nextn_hidden:Vec<MlxBuffer>,
    pub maps:Vec<OperatorMap>,pub whole_body_b4:bool,
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]enum State {Fresh,Encoding,Completed,FailedStop}
pub(crate) struct Invocation {state:State}
impl Invocation {
    pub(crate) fn new()->Self{Self{state:State::Fresh}}
    pub(crate) fn is_failed_stop(&self)->bool{self.state==State::FailedStop}
    pub(crate) fn completed(&self)->bool{self.state==State::Completed}
    fn begin(&mut self)->Result<()>{ensure!(self.state==State::Fresh,"one-shot invocation");self.state=State::Encoding;Ok(())}
    fn settle(&mut self,healthy:bool){self.state=if healthy{State::Completed}else{State::FailedStop};}
}
thread_local! {
    static CHECKED:Cell<bool>=const{Cell::new(false)};
    static FAILED:Cell<bool>=const{Cell::new(false)};
}
/// Default false. Only this graph's actual owning call enables checked helper
/// commits, so every causal CB status is inspected rather than inferred from
/// an unrelated terminal anchor. No teacher/quantization/sampling policy changes.
pub(crate) fn checked_active()->bool{CHECKED.with(Cell::get)}
struct CheckedScope{_same_thread:std::marker::PhantomData<std::rc::Rc<()>>}
impl CheckedScope {
    fn enter()->Result<Self>{
        ensure!(!FAILED.with(Cell::get),"shared graph worker failed; process recovery required");
        CHECKED.with(|c| ->Result<()> {ensure!(!c.replace(true),"reentrant shared checked scope");Ok(())})?;
        Ok(Self{_same_thread:std::marker::PhantomData})
    }
}
impl Drop for CheckedScope{fn drop(&mut self){CHECKED.with(|v|v.set(false));}}
/// Test-only access to the SAME causal-command checks used by run(). Never
/// nest this around run(): the shared producer installs its own scope.
#[cfg(test)]
pub(crate) fn checked_fixture<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    let _scope = CheckedScope::enter()?;
    let result = f();
    if result.is_err() { failed(); }
    result
}

fn failed(){FAILED.with(|v|v.set(true));super::super::decode_pool::profile::quarantine();}
/// Pure descriptor refusal. No target/cursor, proposal, RNG or GPU mutation.
pub(crate) fn preflight(rows:&[Row],owners:&[Owner],n_seqs:u32,max:u32,vocab:u32)->Result<()> {
    ensure!((2..=4).contains(&rows.len())&&rows.len()==owners.len()&&n_seqs==4&&max==262144&&vocab==248320,"selected 2..4 independent rows/four-slot Q38 profile");
    let mut slots=BTreeSet::new();let mut requests=BTreeSet::new();
    for r in rows {
        ensure!(requests.insert(r.request)&&slots.insert(r.handle.slot_id.0),"duplicate owner/slot; temporal rows are not independent");
        ensure!(r.handle.slot_id.0<n_seqs&&r.position<max&&r.token<vocab,"row bounds");
        let matches=owners.iter().filter(|o|o.request==r.request).collect::<Vec<_>>();ensure!(matches.len()==1,"exact one actual owner");let o=matches[0];
        ensure!(o.handle==r.handle&&o.position==r.position&&!o.cancelled&&!o.speculative&&!o.vision,"stale/cancelled/unsupported owner");
    }Ok(())
}
/// Physical input order comes from the actual admitted descriptors, not width.
fn native_rows(rows:&[Row])->crate::inference::native_observation::Input {
    crate::inference::native_observation::ordered_input("qwen35.shared_ordinary.actual_tokens",rows.len(),rows.iter().map(|r|(r.handle.slot_id.0,r.position as usize,1)))
}
/// Proven GPU-only helper CB: retain exact encoder until the existing row fence.
pub(crate) fn deferral_active()->bool{retained::active()}
pub(crate) fn submit_retained(enc:mlx_native::CommandEncoder,label:&'static str)->Result<()> {retained::submit(enc,label)}
/// Only healthy checked copies may retire a row's temporary helper suffix.
fn retire_row_scratch<T>(roots:&mut Vec<T>,start:usize,reset:impl FnOnce()->Result<()>)->Result<()> {
    ensure!(start<=roots.len(),"row-root boundary invalid");roots.truncate(start);reset()
}
/// Internal target producer. Known descriptor/config incompatibility refuses
/// BEFORE mutation with Invocation fresh. Once operational work begins ANY
/// error is sticky worker fail-stop: no scalar replay/reset/further GPU work.
/// The owning caller must preserve replies for its existing fatal fanout path.
pub(crate) fn run(model:&Qwen35Model,cache:&mut HybridKvCache,invocation:&mut Invocation,rows:&[Row],owners:&[Owner])->Result<Output> {
    ensure!(invocation.state==State::Fresh,"one-shot failed/completed invocation cannot revive");
    retained::require_retained_mode(mlx_native::command_buffer_uses_unretained_references())?;
    preflight(rows,owners,cache.n_seqs,cache.max_seq_len,model.cfg.vocab_size)?;
    ensure!(matches!(model.cfg.variant,super::super::Qwen35Variant::Dense)&&model.cfg.hidden_size==5120&&model.cfg.num_hidden_layers==64&&model.cfg.intermediate_size==Some(17408)&&model.glp.is_none()&&model.token_embd_native.is_some(),"selected native Q38 DenseQ unsteered model only");
    ensure!(cache.tq_kv_active&&!cache.la_capture_active(),"selected TQ8 ordinary operation; not active verifier capture");
    for r in rows{cache.validate_sequence_len_for_slot(r.handle.slot_id,r.position as usize)?;}
    for(i,layer)in model.layers.iter().enumerate(){match(layer.kind(),cache.slot_index_for_layer(i as u32)){
        (super::super::Qwen35LayerKind::FullAttention,Some(super::super::kv_cache::LayerSlot::Full(rank)))=>ensure!((rank as usize)<cache.full_attn.len(),"full cache rank"),
        (super::super::Qwen35LayerKind::LinearAttention,Some(super::super::kv_cache::LayerSlot::Linear(rank)))=>ensure!((rank as usize)<cache.linear_attn.len(),"linear cache rank"),
        _=>anyhow::bail!("actual layer/cache schema mismatch"),
    }}
    let retained=(rows.len()as u64).checked_mul(5120*4).and_then(|n|n.checked_mul(model.layers.len()as u64*8+3)).and_then(|n|n.checked_add(rows.len()as u64*248320*4)).context("retained source bound overflow")?;
    ensure!(retained<=128*1024*1024,"partial stable outputs bound; not helper scratch or physical fit");
    invocation.begin()?;
    let result=(|| ->Result<Output> {
        let operation=super::super::decode_pool::profile::Operation::enter(model as *const _ as usize)?;
        let _checked=CheckedScope::enter()?;
        let _retained=retained::GraphDeferralScope::enter()?;
        model.ensure_gpu_cache_primed()?;
        let output=GPU_CACHE.with(|cell| ->Result<Output> {
            let mut borrowed=cell.try_borrow_mut().context("reentrant actual GPU cache")?;let c=borrowed.as_mut().context("unprimed actual cache")?;
            ensure!(gpu_cache_matches_model(c,model)&&c.layer_weights.len()==model.layers.len(),"actual model/cache generation mismatch");
            for layer in &c.layer_weights{let f=match layer{LayerWeightsGpu::FullAttn{ffn,..}|LayerWeightsGpu::LinearAttn{ffn,..}=>ffn};ensure!(matches!(f,FfnWeightsGpu::DenseQ(_)),"native DenseQ all layers");}
            super::super::decode_pool::checked_reset_decode_pool()?;
            let mut roots=Vec::new();let encoded=graph::encode(model,c,cache,rows,&mut roots);
            if encoded.is_err(){std::mem::forget(roots);} // Unknown roots retained, no refund/replay.
            encoded
        })?;
        retained::require_empty()?;
        operation.complete()?; // Required ON24 authority completion, never Drop-quarantine healthy work.
        Ok(output)
    })();
    match result {Ok(output)=>{invocation.settle(true);Ok(output)},Err(e)=>{invocation.settle(false);failed();Err(e.context("shared target failed; owning worker must stop, no scalar replay"))}}
}

#[cfg(test)]mod controls {
    use super::*;
    fn owners()->(Vec<Row>,Vec<Owner>){let rows=(0..4).map(|s|Row{request:s+1,handle:SlotHandle{slot_id:SlotId(s as u32),generation:7},position:128+s as u32,token:1000+s as u32}).collect::<Vec<_>>();let owners=rows.iter().map(|r|Owner{request:r.request,handle:r.handle,position:r.position,cancelled:false,speculative:false,vision:false}).collect();(rows,owners)}
    #[test]fn actual_descriptors_allow_heterogeneous_reordered_rows_and_ordinary_mtp(){let(mut r,o)=owners();r.reverse();preflight(&r,&o,4,262144,248320).unwrap();}
    #[test]fn actual_scheduler_request_zero_is_valid_but_duplicate_zero_is_not(){let(mut r,mut o)=owners();r[0].request=0;o[0].request=0;preflight(&r,&o,4,262144,248320).unwrap();r[1].request=0;o[1].request=0;assert!(preflight(&r,&o,4,262144,248320).is_err());}
    #[test]fn actual_descriptors_refuse_before_mutation(){let(r,o)=owners();for n in 0..8{let(mut r,mut o)=(r.clone(),o.clone());match n{0=>r[0].handle.generation+=1,1=>r[1].handle.slot_id=r[0].handle.slot_id,2=>o[0].cancelled=true,3=>o[0].speculative=true,4=>o[0].vision=true,5=>r[0].position=262144,6=>r[0].token=248320,_=>o[1].request=o[0].request};assert!(preflight(&r,&o,4,262144,248320).is_err());}}
    #[test]fn scope_is_default_none_restores_after_unwind_and_nested_refuses(){assert!(!checked_active());{let g=CheckedScope::enter().unwrap();assert!(checked_active());assert!(CheckedScope::enter().is_err());drop(g);}assert!(!checked_active());let r=std::panic::catch_unwind(||{let _g=CheckedScope::enter().unwrap();panic!("pure unwind control, no GPU");});assert!(r.is_err());assert!(!checked_active());}
    #[test]fn retire_suffix_before_pool_reset_preserves_dedicated_roots(){let dedicated=std::rc::Rc::new(());let suffix=std::rc::Rc::new(());let mut roots=vec![dedicated.clone(),suffix.clone()];retire_row_scratch(&mut roots,1,||{assert_eq!(std::rc::Rc::strong_count(&suffix),1,"suffix dropped BEFORE reset");assert_eq!(std::rc::Rc::strong_count(&dedicated),2,"dedicated root retained");Ok(())}).unwrap();assert_eq!(roots.len(),1);}
    #[test]fn actual_invocation_failure_is_sticky_and_complete_cannot_reenter(){let mut i=Invocation::new();i.begin().unwrap();i.settle(false);assert!(i.is_failed_stop());assert!(i.begin().is_err());let mut j=Invocation::new();j.begin().unwrap();j.settle(true);assert!(j.completed());assert!(j.begin().is_err());}

}

// Final diagnostic-only injected Err after a real healthy row copy. Not a
// Metal command-buffer error and never enabled by production/default calls.
#[cfg(test)]
thread_local! {
    static FIXTURE_INJECT: Cell<bool> = const { Cell::new(false) };
    static FIXTURE_INJECT_HIT: Cell<bool> = const { Cell::new(false) };
}
#[cfg(test)]
pub(crate) fn fixture_arm_final_injection() -> Result<()> {
    ensure!(!checked_active() && !FAILED.with(Cell::get), "final injection requires healthy idle scope");
    FIXTURE_INJECT.with(|v| -> Result<()> {ensure!(!v.replace(true), "one final diagnostic injection");Ok(())})
}
#[cfg(test)]
fn take_fixture_injection(armed: &Cell<bool>, layer:usize, row:usize) -> bool {
    layer == 0 && row == 0 && armed.replace(false)
}
#[cfg(test)]
fn fixture_after_checked_attention_copy(layer:usize,row:usize) -> Result<()> {
    if FIXTURE_INJECT.with(|v|take_fixture_injection(v,layer,row)) {
        FIXTURE_INJECT_HIT.with(|v|v.set(true));
        anyhow::bail!("S3_FINAL_DIAGNOSTIC_INJECTION_AFTER_HEALTHY_CHECKED_ATTENTION_COPY");
    }
    Ok(())
}
#[cfg(test)]
pub(crate) fn fixture_final_injection_observed() -> bool {FIXTURE_INJECT_HIT.with(Cell::get)}
#[cfg(test)]
mod injection_controls {
    use super::*;
    #[test]fn actual_injection_selector_is_one_shot_only_first_layer_first_row() {
        let armed=Cell::new(true);
        assert!(!take_fixture_injection(&armed,1,0));
        assert!(!take_fixture_injection(&armed,0,1));
        assert!(take_fixture_injection(&armed,0,0));
        assert!(!take_fixture_injection(&armed,0,0));
    }
}
