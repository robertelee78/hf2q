//! Q1 Gemma T5 (design §2.3): same-route ordinary-global arm versus one real
//! injected K3 n-gram transaction, teacher-forced over the frozen 144 inputs.
//! Source-owned store extents; the bound supervisor manifest is exported from
//! these SAME functions and never typed by hand.
use super::*;
use crate::adr063_invariance::{receipt::*,store::Store};
use crate::serve::config::Gemma4Config;
use serde::{Deserialize,Serialize};
pub(super) const PREFORK:&str="prefork";
pub(super) const COMMITTED:&str="committed";
pub(super) const RAW:&str="raw-verifier";
pub(super) const STAGES:&str="stages";
pub(super) const SCHEMA:&str="hf2q.q1.gemma-t5-phase.v1";
pub(super) const LOCALIZER_SCHEMA:&str="hf2q.q1.gemma-t5-localizer.v1";
pub(super) const MARKER:&str="Q1_GEMMA_T5_COMPLETE";
pub(super) const LOCALIZER_MARKER:&str="Q1_GEMMA_T5_LOCALIZER_COMPLETE";
pub(super) const CLASS_B:&str="RouteSensitive: global-route initial decision differs from frozen at row 127";
pub(super) const INJECTED_ORIGIN:&str="InjectedDiagnosticFixtureNotProductionProposalQuality";
/// Dense companion heads live in their own namespace: live heads are 0..,
/// restored heads 1000.. (`gemma_cache::at`).
pub(super) const DENSE_SEGMENT:u32=2000;
/// Records per lane per cache boundary: 630 Hybrid (210 heads x K/Vpacked/
/// Vnorm) + 420 dense companion (210 heads x K/V) on the selected 30 layers.
pub(super) const CACHE_RECORDS:usize=1050;
#[derive(Clone,Copy,Debug,PartialEq,Eq,Serialize,Deserialize)]
pub(super) enum Arm {OrdinaryGlobal,NgramTransaction}
impl Arm {
    pub(super) fn label(self)->&'static str {match self {Arm::OrdinaryGlobal=>"OrdinaryGlobal",Arm::NgramTransaction=>"NgramTransaction"}}
}
pub(super) fn trajectory(arm:Arm,store:&str)->String {format!("GemmaT5:{}:{store}",arm.label())}
pub(super) fn logits_at(lane:usize,row:usize)->Coordinate {
    Coordinate {logical_step:0,layer:u32::MAX,stage:Stage::Logits,agent:format!("agent-{lane}"),logical_row_start:row as u64,segment:0}
}
fn logits(cfg:&Gemma4Config,lane:usize,row:usize)->ExpectedTensor {
    ExpectedTensor {at:logits_at(lane,row),scalar:Scalar::F32,shape:vec![1,cfg.vocab_size]}
}
/// Global-route Hybrid roles, then dense companions, per layer, in the journal
/// snapshot's capture order (role -> head).
pub(super) fn cache(cfg:&Gemma4Config,lane:usize,position:usize)->Result<Vec<ExpectedTensor>,String> {
    if lane>=4 || ![128,144].contains(&position) || cfg.num_hidden_layers!=30 || cfg.sliding_window<148 {
        return Err("Gemma T5 global cache boundary/geometry scope".into());
    }
    let mut out=Vec::new();
    for layer in 0..cfg.num_hidden_layers {
        let hd=cfg.head_dim_for_layer(layer);let nkv=cfg.num_kv_heads_for_layer(layer);
        if ![256,512].contains(&hd) || nkv==0 || nkv>=1000 {return Err("selected Gemma head geometry unsupported".into());}
        for (stage,scalar,width,segment) in [(Stage::CacheKey,Scalar::F16,hd,0),(Stage::CacheValuePacked,Scalar::U8,hd,0),
            (Stage::CacheValueNorm,Scalar::F32,hd/256,0),(Stage::CacheKey,Scalar::F32,hd,DENSE_SEGMENT),(Stage::CacheValue,Scalar::F32,hd,DENSE_SEGMENT)] {
            for head in 0..nkv {
                let mut at=crate::adr063_invariance::gemma_cache::at(lane,position,layer,stage,head,false);at.segment+=segment;
                out.push(ExpectedTensor {at,scalar,shape:vec![1,position,width]});
            }
        }
    }
    Ok(out)
}
pub(super) fn prefork(cfg:&Gemma4Config)->Result<Vec<ExpectedTensor>,String> {
    let mut out=Vec::new();
    for lane in 0..4 {out.push(logits(cfg,lane,127));out.extend(cache(cfg,lane,128)?);}
    Ok(out)
}
pub(super) fn committed(cfg:&Gemma4Config)->Result<Vec<ExpectedTensor>,String> {
    let mut out=Vec::new();
    for lane in 0..4 {out.extend((128..144).map(|row|logits(cfg,lane,row)));out.extend(cache(cfg,lane,144)?);}
    Ok(out)
}
/// All four verifier head rows (input positions 128..131), post-softcap.
pub(super) fn raw(cfg:&Gemma4Config)->Vec<ExpectedTensor> {
    (0..4).map(|lane|ExpectedTensor {at:Coordinate {logical_step:128,layer:u32::MAX,stage:Stage::SpecVerifierLogits,
        agent:format!("agent-{lane}"),logical_row_start:0,segment:0},scalar:Scalar::F32,shape:vec![4,cfg.vocab_size]}).collect()
}
pub(super) fn stores(cfg:&Gemma4Config,arm:Arm)->Result<Vec<(&'static str,Vec<ExpectedTensor>)>,String> {
    let mut out=vec![(PREFORK,prefork(cfg)?),(COMMITTED,committed(cfg)?)];
    if arm==Arm::NgramTransaction {out.push((RAW,raw(cfg)));}
    Ok(out)
}
pub(super) fn payload(expected:&[ExpectedTensor])->Result<u64,String> {
    expected.iter().try_fold(0u64,|sum,e|e.shape.iter().try_fold(e.scalar.bytes() as u64,|n,&d|n.checked_mul(d as u64))
        .and_then(|n|sum.checked_add(n))).ok_or_else(||"Gemma T5 payload overflow".into())
}
pub(super) fn witness_charge()->u64 {
    use crate::adr063_invariance::gemma_routing_witness as w;
    w::PAYLOAD_BYTES+w::METADATA_BYTES+w::CPU_LEDGER_BYTES
}
/// The extent of record: every T5 store's bytes, the exact routing charge and
/// the localizer's stage plan bound (0 outside localizer mode).
pub(super) fn host_payload_limit(cfg:&Gemma4Config,arm:Arm,stage_limit:u64)->Result<u64,String> {
    let stores=stores(cfg,arm)?;
    let mut total=witness_charge().checked_add(stage_limit).ok_or("Gemma T5 host overflow")?;
    for (_,e) in &stores {total=total.checked_add(payload(e)?).ok_or("Gemma T5 host overflow")?;}
    if total>PAYLOAD {return Err("Gemma T5 host payload exceeds the 4 GiB journal ceiling".into());}
    Ok(total)
}
/// Teacher-forced decode inputs `(position, frozen[position])`: ordinary
/// 128..143; n-gram 129+a..143 after the verifier committed rows 128..128+a.
pub(super) fn decode_inputs(arm:Arm,frozen:&[u32],accepted:Option<usize>)->Result<Vec<(usize,u32)>> {
    ensure!(frozen.len()==144,"frozen complete144 lineage");
    let start=match (arm,accepted) {
        (Arm::OrdinaryGlobal,None)=>128,
        (Arm::NgramTransaction,Some(a))=>{require_accepted(a)?;129+a}
        _=>anyhow::bail!("accepted count exists only for the n-gram arm"),
    };
    Ok((start..144).map(|p|(p,frozen[p])).collect())
}
pub(super) fn require_accepted(a:usize)->Result<()> {
    ensure!(a<=2,"accepted count {a} outside {{0,1,2}}: the injected mutation of 131 cannot be accepted");Ok(())
}
/// The real rollback cursor must be exactly 129+a.
pub(super) fn require_cursor(pos:usize,accepted:usize)->Result<()> {
    require_accepted(accepted)?;
    ensure!(pos==129+accepted,"actual rollback cursor {pos} differs from 129+a={}",129+accepted);Ok(())
}
/// Class B: the n-gram arm refuses when its global-route initial decision is
/// not frozen[128]; the verify input would differ from the ordinary arm.
pub(super) fn initial_refusal(arm:Arm,lane:usize,initial:u32,frozen128:u32)->Option<Value> {
    (arm==Arm::NgramTransaction && initial!=frozen128).then(||json!({"row":127,"reason":CLASS_B,"lane":lane}))
}
#[derive(Clone,Debug,Serialize,PartialEq,Eq)]
pub(super) struct Row {pub(super) position:usize,pub(super) token:u32}
#[derive(Clone,Debug,Serialize,Default)]
pub(super) struct Transaction {
    pub(super) proposer_ran:bool,pub(super) proposed:Vec<u32>,pub(super) injected:Vec<u32>,pub(super) injected_origin:&'static str,
    pub(super) verifier_argmax:Vec<u32>,pub(super) accepted:Option<usize>,pub(super) rejected:Option<usize>,
    pub(super) fallback:Option<u32>,pub(super) rollback_cursor:Option<usize>,
}
#[derive(Clone,Debug,Serialize)]
pub(super) struct Lane {
    pub(super) lane:usize,pub(super) initial_argmax:Option<u32>,pub(super) frozen_128:u32,pub(super) initial_equals_frozen:Option<bool>,
    pub(super) decode_inputs:Vec<Row>,pub(super) decode_argmax:Vec<Row>,pub(super) transaction:Option<Transaction>,
}
pub(super) struct State {
    pub(super) arm:Arm,pub(super) vocab:usize,pub(super) prefork:Store,pub(super) committed:Store,pub(super) raw:Option<Store>,
    pub(super) lanes:Vec<Lane>,pub(super) verifier:Vec<u8>,pub(super) frozen_accepted:Option<[usize;4]>,
    pub(super) pause:Option<crate::adr063_invariance::family::Pause>,pub(super) class_b:Option<Value>,
    pub(super) localizer:bool,
}
impl State {
    pub(super) fn lane(&mut self,lane:usize)->Result<&mut Lane> {
        self.lanes.last_mut().filter(|l|l.lane==lane).context("T5 lane decision record")
    }
    pub(super) fn begin(&mut self,lane:usize,frozen:&[u32])->Result<()> {
        ensure!(frozen.len()==144 && self.lanes.iter().all(|l|l.lane<lane) && lane<4,"T5 serial lanes in order");
        self.verifier.clear();
        self.lanes.push(Lane {lane,initial_argmax:None,frozen_128:frozen[128],initial_equals_frozen:None,decode_inputs:Vec::new(),
            decode_argmax:Vec::new(),transaction:(self.arm==Arm::NgramTransaction).then(||Transaction {injected_origin:INJECTED_ORIGIN,..Default::default()})});
        Ok(())
    }
}
const DRAIN:&str="actual Gemma global-route checked return; host-visible head row";
/// The current head logits row (single-row decode or prefill last row).
pub(super) fn head_row(store:&mut Store,m:&MlxModelWeights,vocab:usize,lane:usize,row:usize)->Result<()> {
    let view=m.activations.logits.slice_view(0,vocab).with_shape(vec![1,vocab])?;
    crate::adr063_invariance::capture::after_checked_drain(store,logits_at(lane,row),&view,&[1,vocab],None,DRAIN)
        .map_err(anyhow::Error::msg)
}
/// Host bytes of verifier row `i` (post-softcap) committed as `Logits` row 128+i.
pub(super) fn verifier_row(store:&mut Store,rows:&[u8],vocab:usize,lane:usize,i:usize)->Result<()> {
    let n=vocab*4;ensure!(rows.len()==4*n && i<4,"four verifier rows");
    store.capture(logits_at(lane,128+i),Scalar::F32,vec![1,vocab],rows.as_ptr() as usize as u64,rows.len() as u64,(i*n) as u64,None,
        "actual Gemma OptionC verifier head checked finish; CPU copy of the post-softcap row",&rows[i*n..(i+1)*n]).map_err(anyhow::Error::msg)
}
pub(super) fn raw_record(store:&mut Store,rows:&[u8],vocab:usize,lane:usize)->Result<()> {
    ensure!(rows.len()==4*vocab*4,"four verifier rows");
    store.capture(Coordinate {logical_step:128,layer:u32::MAX,stage:Stage::SpecVerifierLogits,agent:format!("agent-{lane}"),
        logical_row_start:0,segment:0},Scalar::F32,vec![4,vocab],rows.as_ptr() as usize as u64,rows.len() as u64,0,None,
        "actual Gemma OptionC verifier head checked finish; CPU copy of all four post-softcap rows",rows).map_err(anyhow::Error::msg)
}
