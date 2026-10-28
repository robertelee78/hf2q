//! Real owning-worker cancellation after checked Gemma tail publication, before
//! Ready installs/seed emission. No forward-error or direct-restore substitute.
use super::*;
use std::{cell::{Cell,RefCell},rc::Rc};
struct Hook {handle:SlotHandle,end:usize,close:Box<dyn FnOnce()>,fired:Rc<Cell<bool>>}
thread_local! {static HOOK:RefCell<Option<Hook>>=const{RefCell::new(None)};}
struct Armed {fired:Rc<Cell<bool>>}
impl Drop for Armed {fn drop(&mut self){HOOK.with(|s|{s.borrow_mut().take();});}}
fn arm(handle:SlotHandle,end:usize,close:impl FnOnce()+'static)->Result<Armed> {
    let fired=Rc::new(Cell::new(false));
    HOOK.with(|s|{let mut s=s.borrow_mut();anyhow::ensure!(s.is_none(),"nested Gemma cancel hook");
        *s=Some(Hook{handle,end,close:Box::new(close),fired:fired.clone()});Ok(())})?;
    Ok(Armed{fired})
}
pub(super) fn after_actual_advance(success:bool,handle:SlotHandle,cursor:Option<usize>) ->Result<()> {
    if !success {return Ok(());}
    HOOK.with(|s|{let mut s=s.borrow_mut();
        if let Some(h)=s.as_ref() {anyhow::ensure!(h.handle==handle && cursor==Some(h.end),
            "Gemma cancel hook actual handle/generation/checked cursor mismatch");}
        if let Some(h)=s.take(){h.fired.set(true);(h.close)();}Ok(())})
}
#[derive(serde::Serialize)]
pub(crate) struct Evidence {
    slot:u32,generation:u64,checked_prefix:usize,actual_unpublished_end:usize,
    recovered_cursor:usize,retained_tokens:usize,real_receiver_closed_after_checked_advance:bool,
    scheduler_in_flight_after:u32,supervisor_healthy:bool,
}
pub(crate) fn run_loaded(model:&mut GemmaLoadedModel,lane:usize,prompt:&[u32],
    mut observe:impl FnMut(bool,&[crate::inference::models::gemma4::kv_cache::MultiSeqHybridKvBuffers])->Result<()>)
    ->Result<Evidence> {
    use crate::adr063_invariance::gemma_endpoint;
    anyhow::ensure!(lane<4 && prompt.len()==128 && model.weights.kv_graft_len()==0,
        "Gemma actual frozen128/no graft scope");
    let mut guard=Gemma4KvGuard::take(model)?;
    let slot=SlotId(lane as u32);validate_gemma4_slot_cursor(&guard,slot,0)?;
    let params=SamplingParams {temperature:0.0,max_tokens:8192,
        stable_prompt_prefix_tokens:Some(64),..SamplingParams::default()};
    let state=Gemma4PrefillState::new(slot,prompt.to_vec(),Vec::new(),params,0,None,Some(64))?;
    let mut scheduler=InflightBatchedScheduler::new(4,4);
    let admitted=scheduler.admit(AdmitRequest {prompt_tokens:128,max_tokens:8192,
        kv_bytes_needed:0,prompt_kv_bytes:0,preferred_slot:Some(slot)})
        .map_err(|e|anyhow::anyhow!("Gemma Q1 admission refused: {e:?}"))?;
    let handle=admitted.handle.context("Gemma Q1 unexpectedly queued")?;
    anyhow::ensure!(handle.slot_id==slot,"actual Gemma preferred slot mismatch");
    let (send,receive)=oneshot::channel();
    let mut slots:Vec<Option<Gemma4Slot>>=(0..4).map(|_|None).collect();
    slots[lane]=Some((Gemma4SlotWork::Prefill(state),SlotReply::Unary(send),handle));
    let mut retained=vec![Vec::new();4];
    let mut anchors:Vec<Option<Gemma4PromptAnchor>>=(0..4).map(|_|None).collect();
    let stats=Arc::new(Mutex::new(scheduler.stats()));let supervisor=EngineSupervisor::new();
    if let Some(fatal)=advance_gemma4_prefill(&mut guard,&mut scheduler,&mut slots,&mut retained,
        &mut anchors,None,&stats,handle,64,0,&supervisor) {
        return Err(fatal.error.context("Gemma Q1 first actual advance failed"));
    }
    let Some((Gemma4SlotWork::Prefill(state),reply,installed))=slots[lane].as_ref() else {
        anyhow::bail!("Gemma Q1 first actual64 transaction not Pending");
    };
    anyhow::ensure!(*installed==handle && !reply.client_closed() && state.progress.cursor()==64
        && state.pending_anchor.as_ref().is_some_and(|a|a.prompt_tokens==prompt[..64]),
        "Gemma actual stable64 anchor/owner missing");
    validate_gemma4_slot_cursor(&guard,slot,64)?;
    gemma_endpoint::end_call().map_err(anyhow::Error::msg)?;
    observe(false,guard.hybrid.as_ref().context("actual Gemma Hybrid absent")?)?;
    let armed=arm(handle,128,move||drop(receive))?;
    // Canonical rows64..127 are captured on committed replay, not this
    // unpublished transaction. Pause never changes the production GPU path.
    let pause=gemma_endpoint::pause_unpublished_tail().map_err(anyhow::Error::msg)?;
    let failure=advance_gemma4_prefill(&mut guard,&mut scheduler,&mut slots,&mut retained,
        &mut anchors,None,&stats,handle,64,0,&supervisor);
    drop(pause);
    if let Some(fatal)=failure {return Err(fatal.error.context("Gemma actual cancellation recovery failed"));}
    anyhow::ensure!(armed.fired.get() && slots[lane].is_none(),"actual Gemma cancel not fired/work remained");
    validate_gemma4_slot_cursor(&guard,slot,64)?;
    anyhow::ensure!(retained[lane]==prompt[..64]
        && anchors[lane].as_ref().is_some_and(|a|a.prompt_tokens==prompt[..64])
        && supervisor.is_healthy() && scheduler.stats().in_flight_slots==0,
        "Gemma real recovery did not restore prefix/release healthy reservation");
    observe(true,guard.hybrid.as_ref().context("actual Gemma Hybrid absent after recovery")?)?;
    Ok(Evidence {slot:slot.0,generation:handle.generation,checked_prefix:64,actual_unpublished_end:128,
        recovered_cursor:64,retained_tokens:retained[lane].len(),real_receiver_closed_after_checked_advance:armed.fired.get(),
        scheduler_in_flight_after:scheduler.stats().in_flight_slots,supervisor_healthy:true})
}
#[cfg(test)]
#[path="q1_gemma_cancel_controls.rs"]
mod controls;
