//! Owning-worker collector. No waiting for a cohort; existing scalar tail/reply
//! handling remains in the parent. All waiters stay in the slot table during
//! shared submission, so existing fatal fanout cannot strand a taken reply.
use super::*;
use super::super::engine_qwen35::{Qwen35DecodeState,Qwen35TickOutcome};
use super::super::engine_qwen35::ordinary_prepared::{Preparation,PreparedOrdinaryTick,PreparedTargetResult};
use crate::inference::models::qwen35::forward_gpu::shared_dense as graph;
use std::collections::HashMap;
use mlx_native::MlxBuffer;

fn effective_tq8(active:bool,raw:u32)->bool {active && (if matches!(raw,5|6|8){raw}else{8})==8}
fn same_row(a:&graph::Row,b:&graph::Row)->bool {
    a.request==b.request&&a.handle==b.handle&&a.position==b.position&&a.token==b.token
}
fn validate_maps(maps:&[graph::OperatorMap],rows:&[graph::Row])->Result<()> {
    let n=rows.len();
    anyhow::ensure!((2..=4).contains(&n)&&maps.len()==64*(n+1)+2,"shared operator map population");
    let packed=|m:&graph::OperatorMap,layer:Option<usize>,name:&str|->bool {
        m.layer==layer&&m.operator==name&&m.physical_rows==n&&m.rows.len()==n
            &&m.rows.iter().zip(rows).all(|(a,b)|same_row(a,b))
    };
    anyhow::ensure!(packed(&maps[0],None,"packed_embedding_gather_not_weight_matmul"),"shared embedding row lineage");
    for layer in 0..64 {
        let at=1+layer*(n+1);
        for (i,row) in rows.iter().enumerate() {
            let m=&maps[at+i];
            anyhow::ensure!(m.layer==Some(layer)&&m.operator=="singleton_entire_attention_including_QKV_output"
                &&m.physical_rows==1&&m.rows.len()==1&&same_row(&m.rows[0],row),"singleton attention lineage");
        }
        anyhow::ensure!(packed(&maps[at+n],Some(layer),"packed_postattn_RMS_dense_FFN_residual"),"packed FFN lineage");
    }
    anyhow::ensure!(packed(maps.last().expect("nonempty"),None,"packed_output_RMS_full_vocabulary_head"),"packed head lineage");
    Ok(())
}

pub(super) enum Staged {
    Ordinary { prepared:PreparedOrdinaryTick,result:Option<PreparedTargetResult>,shared:bool },
    HistoryProposals(Vec<u32>),
}
impl Staged {
    pub(super) fn was_shared(&self)->bool {matches!(self,Self::Ordinary {shared:true,..})}
    pub(super) fn discard_for_cancel(self) {drop(self);}
    pub(super) fn execute(self,state:&mut Qwen35DecodeState,q:&mut super::super::engine_qwen35::Qwen35LoadedModel,
        kv:&mut crate::inference::models::qwen35::kv_cache::HybridKvCache,s:&EngineSupervisor)->Result<Qwen35TickOutcome> {
        match self {
            Self::HistoryProposals(d)=>state.decode_tick_history_proposals(q,kv,s,d),
            Self::Ordinary {mut prepared,result,..}=>{
                let result=match result {Some(r)=>Ok(r),None=>{
                    let target_started=Instant::now();
                    let produced=state.produce_prepared_scalar(q,kv,s,&prepared);
                    if produced.is_ok() {
                        if let Err(e)=prepared.measured_target(target_started.elapsed(),1) {prepared.discard(state);return Err(e);}
                        q.shared_ordinary_width_counts[1]=q.shared_ordinary_width_counts[1].saturating_add(1);
                    }
                    produced
                }};
                match result {
                    Ok(r)=>state.finish_prepared_ordinary(q,kv,s,prepared,r),
                    Err(e)=>{prepared.discard(state);Err(e)}
                }
            }
        }
    }
}
pub(super) struct Batch {
    entries:HashMap<SlotHandle,Staged>,
    retained_rows:Vec<MlxBuffer>,
    submitted:bool,
}
impl Batch {
    pub(super) fn new()->Self {Self {entries:HashMap::new(),retained_rows:Vec::new(),submitted:false}}
    pub(super) fn contains(&self,h:SlotHandle)->bool {self.entries.contains_key(&h)}
    pub(super) fn take(&mut self,h:SlotHandle)->Option<Staged> {self.entries.remove(&h)}
    pub(super) fn quarantine(&mut self) {
        if self.submitted {
            crate::inference::models::qwen35::decode_pool::profile::quarantine();
            // Unknown completion of a later catchup cannot refund the shared
            // durable parent. Worker fatal fanout owns all replies; OS restarts.
            std::mem::forget(std::mem::take(&mut self.retained_rows));
        }
    }
    fn fatal(&mut self,error:anyhow::Error)->Qwen35FatalFailure {
        self.quarantine();
        let kind=if is_fatal_command_buffer_error(&error) {Qwen35WorkerFatalKind::Gpu}else{Qwen35WorkerFatalKind::Invariant};
        Qwen35FatalFailure {handle:None,reply:Qwen35FatalReply::None,
            error:error.context("shared ordinary fail-stop: no reset/replay"),kind}
    }
    pub(super) fn prepare_prefix(&mut self,guard:&mut Qwen35KvGuard<'_>,slots:&mut [Option<Qwen35Slot>],
        handles:&[SlotHandle],scheduler:&InflightBatchedScheduler,supervisor:&EngineSupervisor)
        ->std::result::Result<(),Qwen35FatalFailure> {
        let kv=guard.kv.as_ref().expect("kv Some");
        // Same read/write/dequant source authority as the actual cache paths;
        // never read ambient env or infer bits from one-byte packed storage.
        if !effective_tq8(kv.tq_kv_active,crate::debug::INVESTIGATION_ENV.tq_codebook_bits) || kv.la_capture_active()
            || guard.model.model.glp.is_some() || guard.model.model.kv_graft.is_some() {return Ok(());}
        let mut selected=Vec::new();
        for &handle in handles.iter().take(4) {
            if self.entries.contains_key(&handle) {break;}
            let Some((Qwen35SlotWork::Decode(state),reply,installed))=slots.get_mut(handle.slot_id.0 as usize).and_then(Option::as_mut) else {break;};
            if !state.shared_ordinary_semantics_supported() {break;}
            let preview=state.ordinary_leaf_preview(handle,*installed,reply.client_closed());
            use super::super::engine_qwen35::ordinary_action::{OwnerPreview,OrdinaryLeaf};
            if matches!(preview,OwnerPreview::RefusedOwner|OwnerPreview::Cancelled)
                ||preview.known_leaf()==Some(OrdinaryLeaf::GpuGreedy) {break;}
            let Some(request)=scheduler.admitted_request_id(handle) else {break;};
            let preparation=state.prepare_shared_ordinary_tick(guard.kv.as_ref().expect("kv Some"))
                .map_err(|e|self.fatal(e))?;
            match preparation {
                Preparation::Unsupported=>break,
                Preparation::HistoryProposals(d)=>{self.entries.insert(handle,Staged::HistoryProposals(d));break;}
                Preparation::Ordinary(prepared)=>{
                    let (token,position,slot)=prepared.input();
                    let share=prepared.can_share()&&slot==handle.slot_id&&position>=0;
                    self.entries.insert(handle,Staged::Ordinary {prepared,result:None,shared:false});
                    if !share {break;}
                    selected.push(graph::Row {request:request.0,handle,position:position as u32,token});
                }
            }
        }
        if selected.len()<2 {return Ok(());}
        let mut owners=Vec::new();
        for row in &selected {
            let Some((Qwen35SlotWork::Decode(_),reply,installed))=slots[row.handle.slot_id.0 as usize].as_ref() else {
                return Err(self.fatal(anyhow::anyhow!("shared prepared owner disappeared before submit")));
            };
            if *installed!=row.handle || scheduler.admitted_request_id(row.handle).map(|r|r.0)!=Some(row.request) {
                return Err(self.fatal(anyhow::anyhow!("shared prepared owner generation changed")));
            }
            // Cancel BEFORE shared mutation: all open peers keep their prepared
            // scalar action, not a re-proposal/re-consumption of forced tokens.
            if reply.client_closed() {return Ok(());}
            if let Err(e)=guard.kv.as_ref().expect("kv Some").validate_sequence_len_for_slot(row.handle.slot_id,row.position as usize) {
                return Err(self.fatal(e));
            }
            owners.push(graph::Owner {request:row.request,handle:row.handle,position:row.position,
                cancelled:false,speculative:false,vision:false});
        }
        let cohort=selected.iter().map(|r|r.handle).collect::<Vec<_>>();
        let _scope=scheduler.trace_transaction("qwen35","shared_ordinary_target_partial",&cohort);
        let target_started=Instant::now();
        let lease=supervisor.arm("Qwen35 shared ordinary target",std::time::Duration::from_secs(30))
            .map_err(|e|self.fatal(e))?;
        let mut invocation=graph::Invocation::new();
        let produced=graph::run(&guard.model.model,guard.kv.as_mut().expect("kv Some"),&mut invocation,&selected,&owners);
        if let Err(e)=lease.finish() {self.submitted=true;return Err(self.fatal(e));}
        let output=match produced {
            Ok(o)=>o,
            Err(e) if !invocation.is_failed_stop()=>{
                tracing::debug!(error=%e,"shared ordinary descriptor refused before mutation; prepared scalar only");
                return Ok(());
            }
            Err(e)=>{self.submitted=true;return Err(self.fatal(e));}
        };
        self.submitted=true;
        self.retained_rows.extend(output.nextn_hidden.iter().cloned());
        if !invocation.completed()||output.whole_body_b4||output.logits.len()!=selected.len()||output.nextn_hidden.len()!=selected.len() {
            return Err(self.fatal(anyhow::anyhow!("shared target completion/row extent mismatch")));
        }
        let n=selected.len();
        if let Err(e)=validate_maps(&output.maps,&selected) {return Err(self.fatal(e));}
        let singleton=output.maps.iter().filter(|m|m.physical_rows==1).count();
        let shared=output.maps.iter().filter(|m|m.physical_rows==n).count();
        let mut results=Vec::new();
        for ((row,logits),hidden) in selected.iter().zip(output.logits).zip(output.nextn_hidden) {
            let Some(Staged::Ordinary {prepared,..})=self.entries.get(&row.handle) else {unreachable!("prepared selected row")};
            let result=Qwen35DecodeState::supplied_ordinary_result(prepared,logits,hidden);
            let Some((Qwen35SlotWork::Decode(state),_,installed))=slots[row.handle.slot_id.0 as usize].as_ref() else {unreachable!("validated owner")};
            if *installed!=row.handle {return Err(self.fatal(anyhow::anyhow!("owner changed after shared target")));}
            if let Err(e)=state.validate_prepared_ordinary(guard.model,guard.kv.as_ref().expect("kv Some"),prepared,&result) {
                return Err(self.fatal(e));
            }
            results.push((row.handle,result));
        }
        let raw_target_elapsed=target_started.elapsed();
        // All target/cursor/state/output boundaries validated BEFORE ANY tail.
        for (handle,result) in results {
            if let Some(Staged::Ordinary {prepared,result:r,shared,..})=self.entries.get_mut(&handle) {
                if let Err(e)=prepared.measured_target(raw_target_elapsed,n) {return Err(self.fatal(e));}
                *r=Some(result);*shared=true;
            }
        }
        guard.model.shared_ordinary_width_counts[n]=guard.model.shared_ordinary_width_counts[n].saturating_add(1);
        tracing::info!(width=n,raw_target_ns=raw_target_elapsed.as_nanos() as u64,owners=?selected,singleton_attention_maps=singleton,
            shared_operator_maps=shared,width_counts=?guard.model.shared_ordinary_width_counts,
            "Qwen shared ordinary target completed; partial operators, not whole-body B4");
        Ok(())
    }
}

#[cfg(test)]
#[path="qwen_shared_ordinary_controls.rs"]
mod controls;
