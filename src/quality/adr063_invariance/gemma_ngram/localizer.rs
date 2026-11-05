//! gemma_ngram_localizer (design §2.6): the verifier-route stage localizer of
//! the T5 adapter. Reuses T2Loc's owned-row capture, role sets, `row` /
//! `ordered` helpers and quotas; only the extent differs. Per lane, in
//! execution order: the captured prefill (ordinary initial rows 0..127 at
//! segment 0, or the n-gram verify re-prefill rows 0..131 at segment 1), then
//! single-row decode (ordinary 128..143, n-gram 129+a..143) at segment 0.
use super::t5::{require_accepted,Arm};
use crate::adr063_invariance::{family::OWNED_AGENT,gemma_cohort_localizer::{self as cohort,Targets,DECODE_STAGES,STAGE_PAYLOAD_CAP},
    gemma_localizer::PREFILL_STAGES,receipt::*};
use crate::serve::config::Gemma4Config;
use serde::{Deserialize,Serialize};
/// Verify re-prefill namespace: keeps verify row 131 (input mut131) apart
/// from continuation decode row 131 (input f131).
pub(super) const VERIFY_SEGMENT:u32=1;
pub(super) const VERIFY_ROWS:usize=132;
pub(super) const RECORD_CAP:usize=65536;
pub(super) const TARGET_CAP:usize=4096;
pub(super) const PASS2_LAYER_CAP:usize=8;
/// Pass 1: all 30 layers, these roles for both prefill and decode.
pub(super) const PASS1_STAGES:[Stage;3]=[Stage::RouterLogits,Stage::RouterIds,Stage::LayerOutput];
/// The packet's `stage_localizer` block: T2Loc's `Plan` plus the per-lane
/// accepted count frozen from the T5 n-gram receipt (null for branch P).
#[derive(Clone,Debug,PartialEq,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Plan {
    pub(super) rerun_id:String,
    pub(super) layers:Vec<u32>,
    pub(super) prefill_stages:Vec<Stage>,
    pub(super) decode_stages:Vec<Stage>,
    pub(super) host_payload_limit:u64,
    pub(super) accepted:Option<[u8;4]>,
}
impl Plan {
    pub(super) fn accepted(&self)->Result<Option<[usize;4]>,String> {
        let Some(a)=self.accepted else {return Ok(None)};
        let a=a.map(usize::from);
        for &x in &a {require_accepted(x).map_err(|e|e.to_string())?;}
        Ok(Some(a))
    }
}
/// `(lane, absolute position)` of the 132 verify rows; the verify is a full
/// re-prefill from 0, never a resumed or longer stream.
pub(super) fn verify_owners(lane:usize,start:usize,rows:usize)->Result<Vec<(usize,usize)>,String> {
    if lane>=4 || start!=0 || rows==0 || rows>VERIFY_ROWS {
        return Err("n-gram verify owners are (lane, 0..131) from start 0 only".into());
    }
    Ok((0..rows).map(|p|(lane,p)).collect())
}
fn at(layer:u32,stage:Stage,agent:String,row:usize,segment:u32)->Coordinate {
    Coordinate {logical_step:0,layer,stage,agent,logical_row_start:row as u64,segment}
}
/// `(targets, expected, planned bytes)`. `accepted` is required for the
/// n-gram arm and ignored by the ordinary arm, whose extent is a-independent.
pub(super) fn manifest(cfg:&Gemma4Config,layers:&[u32],prefill_stages:&[Stage],decode_stages:&[Stage],
    arm:Arm,accepted:Option<[usize;4]>)->Result<(Targets,Vec<ExpectedTensor>,u64),String> {
    if layers.is_empty() || layers.windows(2).any(|w|w[0]>=w[1]) || layers.iter().any(|&l|l as usize>=cfg.num_hidden_layers) {
        return Err("localizer layers must be ascending distinct actual Gemma layers".into());
    }
    let prefill=cohort::ordered(&PREFILL_STAGES,prefill_stages)?;
    let decode=cohort::ordered(&DECODE_STAGES,decode_stages)?;
    if prefill.is_empty() && decode.is_empty() {return Err("localizer needs at least one prefill or decode stage".into());}
    let (rows,segment,starts)=match (arm,accepted) {
        (Arm::OrdinaryGlobal,_)=>(128,0,[128;4]),
        (Arm::NgramTransaction,Some(a))=>{
            for &x in &a {require_accepted(x).map_err(|e|e.to_string())?;}
            (VERIFY_ROWS,VERIFY_SEGMENT,a.map(|x|129+x))
        }
        (Arm::NgramTransaction,None)=>return Err("n-gram localizer needs the frozen per-lane accepted count".into()),
    };
    let mut targets=Targets::new();
    for &layer in layers {for &stage in &prefill {
        targets.insert((0,layer,stage),CaptureTarget {at:at(layer,stage,OWNED_AGENT.into(),0,segment),normalize_rows:true});
    }}
    let first=*starts.iter().min().unwrap();
    for position in first..144 {for &layer in layers {for &stage in &decode {
        targets.insert((position as u64,layer,stage),CaptureTarget {at:at(layer,stage,OWNED_AGENT.into(),position,0),normalize_rows:true});
    }}}
    let mut expected=Vec::new();
    for (lane,&start) in starts.iter().enumerate() {
        let agent=format!("agent-{lane}");
        let mut push=|layer:u32,stage:Stage,row:usize,segment:u32|->Result<(),String> {
            let (scalar,axes)=cohort::row(cfg,layer as usize,stage)?;
            expected.push(ExpectedTensor {at:at(layer,stage,agent.clone(),row,segment),scalar,shape:std::iter::once(1).chain(axes).collect()});
            Ok(())
        };
        for &layer in layers {for &stage in &prefill {for row in 0..rows {push(layer,stage,row,segment)?;}}}
        for position in start..144 {for &layer in layers {for &stage in &decode {push(layer,stage,position,0)?;}}}
    }
    if targets.len()>TARGET_CAP || expected.len()>RECORD_CAP {return Err("localizer target/record quota exceeded".into());}
    let bytes=super::t5::payload(&expected)?;
    if bytes>STAGE_PAYLOAD_CAP {return Err("localizer payload exceeds the 1 GiB stage cap".into());}
    Ok((targets,expected,bytes))
}
fn file_safe(id:&str)->bool {!id.is_empty() && id.len()<=64 && id.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-'||b==b'_')}
/// Admission like T2Loc's `admit`: the Plan's source extent must equal the
/// stage extent of the bound manifest file, within its own bound and the cap.
pub(super) fn admit(cfg:&Gemma4Config,plan:&Plan,arm:Arm,bound:&[ExpectedTensor])->Result<(Targets,u64),String> {
    if !file_safe(&plan.rerun_id) {return Err("file-safe localizer rerun id required".into());}
    let (targets,expected,bytes)=manifest(cfg,&plan.layers,&plan.prefill_stages,&plan.decode_stages,arm,plan.accepted()?)?;
    if expected!=bound {return Err("Plan stage extent differs from the bound manifest".into());}
    if bytes>plan.host_payload_limit || plan.host_payload_limit>STAGE_PAYLOAD_CAP {
        return Err("stage payload exceeds the plan bound or the 1 GiB cap".into());
    }
    Ok((targets,bytes))
}
/// Pass shape: pass 1 is all layers with [`PASS1_STAGES`]; pass 2 is at most
/// eight layers with every prefill and decode role.
pub(super) fn require_pass(plan:&Plan,pass:u8)->Result<(),String> {
    let ok=match pass {
        1=>plan.layers==(0..30).collect::<Vec<u32>>() && plan.prefill_stages==PASS1_STAGES && plan.decode_stages==PASS1_STAGES,
        2=>(1..=PASS2_LAYER_CAP).contains(&plan.layers.len()) && plan.prefill_stages==PREFILL_STAGES && plan.decode_stages==DECODE_STAGES,
        _=>false,
    };
    if ok {Ok(())} else {Err(format!("plan is not the frozen pass-{pass} shape"))}
}
/// Frozen pass-2 rule: the union over lanes of {L*-1, L*} ({0} when L* = 0).
/// A lane whose onset is the output head (no pass-1 group differs) is `None`
/// and contributes no layer. At most eight layers.
pub(super) fn pass2_layers(onsets:&[Option<u32>;4])->Result<Vec<u32>,String> {
    let mut layers=std::collections::BTreeSet::new();
    for &l in onsets.iter().flatten() {
        if l>=30 {return Err("pass-1 onset outside the 30 Gemma layers".into());}
        if l>0 {layers.insert(l-1);}
        layers.insert(l);
    }
    if layers.is_empty() {return Err("no lane has a pass-1 layer onset; pass 2 does not run".into());}
    if layers.len()>PASS2_LAYER_CAP {return Err("pass-2 layer union exceeds eight layers".into());}
    Ok(layers.into_iter().collect())
}
