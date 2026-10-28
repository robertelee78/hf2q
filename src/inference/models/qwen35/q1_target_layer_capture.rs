//! Test-only snapshots on the ORIGINAL owning encoders. No added GPU waits.
//! Missing native operator-route evidence remains explicit; no Q1/perf PASS.
use anyhow::{Context,Result};
use crate::adr063_invariance::{receipt::*,store::Store};
use mlx_native::{CommandEncoder,DType,MlxBuffer,MlxDevice};
use std::{cell::RefCell,collections::BTreeSet};
use super::{Qwen35Config,Qwen35Variant};
#[path="q1_target_layer_capture/plan.rs"]
mod plan;
pub(crate) use plan::{expected,ARM_BYTES};

struct Sink {
    id:u64,store:Store,forced:bool,wanted:[u32;2],entered:BTreeSet<usize>,
    completed:BTreeSet<usize>,metadata:Vec<serde_json::Value>,metadata_bytes:usize,failed:bool,
}
thread_local! {static ACTIVE:RefCell<Option<Sink>>=const{RefCell::new(None)};}
static NEXT:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(1);
pub(crate) struct Scope {id:u64,_same_thread:std::marker::PhantomData<std::rc::Rc<()>>}
pub(crate) fn selected()->bool {ACTIVE.with(|s|s.borrow().is_some())}
pub(crate) fn install(mut store:Store,forced:bool,wanted:[u32;2])->Result<Scope> {
    store.missing_capability("native per-projection kernel route not observed; HF branch/codec only; raw blit graph capture unsupported");
    ACTIVE.with(|s| {
        anyhow::ensure!(s.borrow().is_none(),"nested target-layer sink");
        let id=NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
        *s.borrow_mut()=Some(Sink{id,store,forced,wanted,entered:BTreeSet::new(),
            completed:BTreeSet::new(),metadata:vec![],metadata_bytes:0,failed:false});
        Ok(Scope{id,_same_thread:std::marker::PhantomData})
    })
}
impl Scope {
    pub(crate) fn finish(self,failure:Option<&str>)->Result<serde_json::Value> {
        ACTIVE.with(|s| {
            let mut sink=s.borrow_mut().take().context("target-layer sink absent")?;
            anyhow::ensure!(sink.id==self.id,"target-layer sink owner changed");
            if let Some(error)=failure {sink.store.fail(error);}
            if sink.failed || sink.completed!=BTreeSet::from([129,130]) {
                sink.store.fail("missing/uncertain target forward retirement");
            }
            let trace=sink.store.finish().map_err(anyhow::Error::msg)?;
            anyhow::ensure!(trace.coverage.errors.is_empty() && trace.coverage.captured.len()==258,
                "incomplete target-layer trace retained, no completion");
            Ok(serde_json::json!({"schema":"hf2q.q1.target-layer-snapshots.v1",
                "records":258,"payload_bytes":ARM_BYTES,"logical_lane":0,"physical_slot":0,
                "slot_generation":null,"owner_scope":sink.id,
                "slot_generation_scope":"fresh independently acquired model, not serving scheduler generation",
                "metadata":sink.metadata,"numeric_policy":"Unspecified","whole_q1_qualified":false,
                "native_operator_route":"UnknownNotObserved","graph_capture":"UnsupportedFailClosed",
                "additional_checked_waits":0,"blit_changes_compute_encoder_partitions":true,
                "original_first_fork_endpoint_equivalence":"PendingArtifactCheck"}))
        })
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|s| {
            let mut active=s.borrow_mut();
            if active.as_ref().is_some_and(|a|a.id==self.id) {active.take();}
        });
    }
}

pub(crate) struct Forward {
    id:u64,positions:Vec<usize>,tokens:Vec<u32>,axes:Vec<i32>,seq_rows:usize,bank:Option<MlxBuffer>,
    holds:Vec<MlxBuffer>,points:BTreeSet<usize>,metadata:Vec<serde_json::Value>,metadata_bytes:usize,drained:bool,
}
// Actual public blit validates composed logical offsets and rejects captured
// graph encoders. This helper neither opens/commits a CB nor waits for the GPU.
fn copy_rows(enc:&mut CommandEncoder,src:&MlxBuffer,dst:&MlxBuffer,rows:usize,
    width:usize,destination_offset:u64)->Result<()> {
    let bytes=rows.checked_mul(width).and_then(|n|n.checked_mul(4)).context("copy overflow")?;
    anyhow::ensure!(src.dtype()==DType::F32 && dst.dtype()==DType::F32 &&
        bytes<=src.data_byte_len() && destination_offset.checked_add(bytes as u64)
            .is_some_and(|end|end<=dst.data_byte_len() as u64),"snapshot logical bounds");
    enc.blit_copy_bytes(src,0,dst,destination_offset,bytes as u64)?;Ok(())
}
pub(crate) fn begin(device:&MlxDevice,cfg:&Qwen35Config,tokens:&[u32],positions:&[i32],
    slot:usize,nextn_all:bool,supported:bool)->Result<Option<Forward>> {
    ACTIVE.with(|s| {
        let mut active=s.borrow_mut();let Some(sink)=active.as_mut() else {return Ok(None)};
        let Some(selected)=plan::select(sink.forced,tokens,positions,sink.wanted)? else {return Ok(None)};
        anyhow::ensure!(!sink.failed && slot==0 && nextn_all && supported &&
            cfg.variant==Qwen35Variant::Dense && cfg.hidden_size as usize==plan::WIDTH &&
            cfg.num_hidden_layers as usize==plan::LAYERS,"unsupported selected target graph");
        anyhow::ensure!(selected.iter().all(|p|!sink.entered.contains(p)),"duplicate target input forward");
        sink.entered.extend(selected.iter().copied());
        let bytes=selected.len()*plan::POINTS*plan::WIDTH*4;
        anyhow::ensure!(bytes as u64<=ARM_BYTES,"snapshot bank source bound");
        let bank=device.alloc_buffer(bytes,DType::F32,vec![plan::POINTS,selected.len(),plan::WIDTH])?;
        Ok(Some(Forward{id:sink.id,positions:selected,tokens:tokens.to_vec(),axes:positions.to_vec(),seq_rows:tokens.len(),bank:Some(bank),
            holds:vec![],points:BTreeSet::new(),metadata:vec![],metadata_bytes:0,drained:false}))
    })
}
impl Forward {
    pub(crate) fn snapshot(&mut self,enc:&mut CommandEncoder,buffer:&MlxBuffer,
        layer:usize,stage:Stage,route:serde_json::Value)->Result<()> {
        let point=plan::visit(&mut self.points,layer,stage)?;
        anyhow::ensure!(buffer.element_count()>=self.seq_rows*plan::WIDTH,
            "noncontiguous source point");
        let destination=(point*self.positions.len()*plan::WIDTH*4) as u64;
        copy_rows(enc,buffer,self.bank.as_ref().context("snapshot bank absent")?,
            self.positions.len(),plan::WIDTH,destination)?;
        self.holds.push(buffer.clone());
        let record=serde_json::json!({"layer":layer,"stage":stage,
            "absolute_token_positions":self.positions,"physical_source_rows":if self.positions.len()==2 {vec![0,1]}else{vec![0]},
            "logical_width":plan::WIDTH,"source_seq_rows":self.seq_rows,
            "actual_forward_tokens":self.tokens,"actual_axis_major_positions":self.axes,
            "source_view_offset":buffer.byte_offset(),"source_data_bytes":buffer.data_byte_len(),
            "source_parent_bytes":buffer.byte_len(),"source_parent_identity":buffer.contents_ptr() as usize as u64,
            "destination_offset":destination,"copied_bytes":self.positions.len()*plan::WIDTH*4,
            "host_encode_metadata_not_gpu_completion":true,"route_and_state":route});
        let bytes=serde_json::to_vec(&record)?.len();
        self.metadata_bytes=self.metadata_bytes.checked_add(bytes).context("metadata overflow")?;
        anyhow::ensure!(bytes<=8*1024 && self.metadata_bytes<=768*1024,"target metadata quota exceeded");
        self.metadata.push(record);Ok(())
    }
    /// Called ONLY after the existing All+nextn output-head checked drain.
    /// CPU reads never happen at a per-layer encode point.
    pub(crate) fn finish_after_checked_output_head(mut self)->Result<()> {
        self.drained=true;
        plan::complete(&self.points)?;
        let bank=self.bank.as_ref().context("snapshot bank absent")?;
        let data=bank.as_logical_slice::<f32>()?;
        ACTIVE.with(|s| {
            let mut active=s.borrow_mut();let sink=active.as_mut().context("target sink disappeared")?;
            anyhow::ensure!(sink.id==self.id && !sink.failed,"target sink lineage mismatch");
            for point in 0..plan::POINTS {
                let (layer,stage)=if point==0 {(0,Stage::Input)} else if point%2==1 {
                    (((point-1)/2) as u32,Stage::Residual)
                } else {(((point-2)/2) as u32,Stage::LayerOutput)};
                for (row,&position) in self.positions.iter().enumerate() {
                    let begin=(point*self.positions.len()+row)*plan::WIDTH;
                    let end=begin+plan::WIDTH;
                    sink.store.capture(plan::at(position,layer,stage),Scalar::F32,vec![1,plan::WIDTH],
                        bank.contents_ptr() as usize as u64,bank.byte_len() as u64,
                        bank.byte_offset()+(begin*4) as u64,None,
                        "same owning encoder snapshot; actual existing target output-head checked terminal",
                        bytemuck::cast_slice(&data[begin..end])).map_err(anyhow::Error::msg)?;
                }
            }
            sink.metadata_bytes=sink.metadata_bytes.checked_add(self.metadata_bytes).context("arm metadata overflow")?;
            anyhow::ensure!(sink.metadata_bytes<=768*1024,"arm metadata quota exceeded");
            sink.completed.extend(self.positions.iter().copied());
            sink.metadata.append(&mut self.metadata);Ok(())
        })
    }
}
impl Drop for Forward {
    fn drop(&mut self) {
        if !self.drained {
            // Test-only failure: no refund/free against unknown GPU retirement.
            // One failed forward poisons this fresh process's sink; the bounded
            // bank+source owners remain until process exit, never reused.
            if let Some(bank)=self.bank.take() {std::mem::forget(bank);}
            for source in self.holds.drain(..) {std::mem::forget(source);}
            ACTIVE.with(|s| {if let Some(sink)=s.borrow_mut().as_mut() {
                if sink.id==self.id {sink.failed=true;sink.store.fail("unknown snapshot forward retirement; owners retained until process exit");}
            }});
        }
    }
}
#[cfg(test)]
#[path="q1_target_layer_capture/tests.rs"]
mod tests;
