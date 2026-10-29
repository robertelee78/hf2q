//! Diagnostic-only property reads after existing healthy checks. DefaultNone.
use anyhow::{ensure,Context,Result};
use std::{cell::RefCell,time::Instant};
use serde::Serialize;
#[derive(Serialize)]pub(crate) struct Event{
    pub stage:&'static str,pub layer:Option<usize>,pub owner:Option<u32>,
    pub original_host_submit_or_wait_ns:u64,pub gpu_duration_ns_truncated:Option<u64>,
    pub healthy_result:bool,
}
pub(crate) const GROUPED_PROFILE:&str="qwen38-n2-one-layer-checked-group-v1";
#[derive(Serialize)]pub(crate) struct GroupedEvent {
 pub event_schema:&'static str,pub timing_profile:&'static str,pub event_ordinal:usize,
 pub stage:&'static str,pub layer:Option<usize>,pub owner:Option<u32>,
 pub host_boundary:&'static str,pub host_clock:&'static str,pub host_elapsed_ns:u64,
 pub completed_cb_gpu_duration_ns_truncated:Option<u64>,pub gpu_observation_boundary:&'static str,
 pub healthy_result:bool,pub sum_is_gpu_union:bool,
}
pub(crate) const THREE_LA_PROFILE:&str="qwen38-n2-three-la-checked-group-v1";
struct State{three_la:bool,events:Vec<Event>,grouped:Option<Vec<GroupedEvent>>,context:(Option<usize>,Option<u32>)}
thread_local!{static STATE:RefCell<Option<State>>=const{RefCell::new(None)};}
const CAP:usize=1024;
pub(crate) struct Scope{armed:bool}
impl Scope{
 pub(crate) fn enter()->Result<Self>{STATE.with(|s|->Result<()>{let mut s=s.try_borrow_mut().context("timing reentry")?;ensure!(s.is_none(),"nested timing");let mut events=Vec::new();events.try_reserve_exact(CAP)?;*s=Some(State{three_la:false,events,grouped:None,context:(None,None)});Ok(())})?;Ok(Self{armed:true})}
 pub(crate) fn take(mut self)->Result<Vec<Event>>{let x=STATE.with(|s|s.try_borrow_mut().map_err(|_|anyhow::anyhow!("timing borrowed"))?.take().context("timing scope missing"))?;self.armed=false;ensure!(x.grouped.is_none(),"legacy take cannot consume grouped events");Ok(x.events)}
}
impl Drop for Scope{fn drop(&mut self){if self.armed{STATE.with(|s|{if let Ok(mut s)=s.try_borrow_mut(){s.take();}});}}}

/// Separate opt-in stream; the original Event and Scope::enter/take stay V1.
pub(crate) struct GroupedScope{scope:Scope}
impl GroupedScope {
 pub(crate) fn enter()->Result<Self>{let scope=Scope::enter()?;STATE.with(|s|->Result<()>{
   let mut s=s.try_borrow_mut().context("grouped timing reentry")?;let s=s.as_mut().context("missing timing")?;
   let mut events=Vec::new();events.try_reserve_exact(CAP)?;s.grouped=Some(events);Ok(())})?;Ok(Self{scope})}
 pub(crate) fn take(mut self)->Result<Vec<GroupedEvent>>{let x=STATE.with(|s|s.try_borrow_mut().map_err(|_|anyhow::anyhow!("grouped timing borrowed"))?.take().context("grouped timing missing"))?;
  self.scope.armed=false;ensure!(x.events.is_empty(),"legacy/grouped mixing refused");x.grouped.context("missing grouped stream")}
}
pub(crate) struct ThreeLaScope{scope:GroupedScope}
impl ThreeLaScope {
 pub(crate) fn enter()->Result<Self>{let scope=GroupedScope::enter()?;STATE.with(|s|->Result<()>{s.try_borrow_mut().context("three LA timing borrow")?.as_mut().context("missing timing")?.three_la=true;Ok(())})?;Ok(Self{scope})}
 pub(crate) fn take(self)->Result<Vec<GroupedEvent>>{self.scope.take()}
}
pub(crate) fn three_la_active()->bool {STATE.with(|s|s.try_borrow().map(|s|s.as_ref().is_some_and(|s|s.three_la)).unwrap_or(false))}
pub(crate) fn require_grouped_or_inactive()->Result<()>{STATE.with(|s|->Result<()>{
 let s=s.try_borrow().context("timing profile borrow")?;if let Some(s)=s.as_ref(){ensure!(s.grouped.is_some(),"one-layer group refuses legacy timing profile");}Ok(())})}
fn grouped_boundary(stage:&str)->Result<(&'static str,&'static str)>{match stage{
 "shared-FFN"=>Ok(("ffn-anchor-checked","commit_and_wait_after_encoding_including_queued_upstream")),
 "retained-Delta-terminal-check"=>Ok(("retained-delta-checked","wait_until_completed_after_ffn_anchor")),
 "retained-FFN-terminal-check"=>Ok(("retained-ffn-checked","wait_until_completed_after_ffn_anchor")),
 "row-copy"=>Ok(("retained-row-copy-checked","wait_until_completed_after_ffn_anchor")),
 _=>anyhow::bail!("unknown grouped timing stage"),}}
pub(crate) fn context(layer:usize,owner:u32)->Result<()>{STATE.with(|s|->Result<()>{let mut s=s.try_borrow_mut().context("timing context reentry")?;if let Some(s)=s.as_mut(){s.context=(Some(layer),Some(owner));}Ok(())})}
pub(crate) fn start()->Result<Option<Instant>>{STATE.with(|s|Ok(s.try_borrow().map_err(|_|anyhow::anyhow!("timing clock reentry"))?.as_ref().map(|_|Instant::now())))}
pub(crate) fn checked(stage:&'static str,e:&mlx_native::CommandEncoder,start:Option<Instant>,at:Option<(usize,Option<u32>)>)->Result<()>{
 let Some(start)=start else{return Ok(());};
 let host=u64::try_from(start.elapsed().as_nanos())?;
 // Caller has ALREADY checked exact CB Result. No wait/recommit/raw escape.
 let gpu=e.completed_gpu_interval_ns();
 STATE.with(|s|->Result<()>{let mut s=s.try_borrow_mut().context("timing callback reentry")?;let s=s.as_mut().context("lost timing scope")?;let(layer,owner)=at.map_or(s.context,|(l,o)|(Some(l),o));
 let three_la=s.three_la;
 if let Some(events)=s.grouped.as_mut(){ensure!(events.len()<CAP,"grouped timing cap");let(stage,host_boundary)=grouped_boundary(stage)?;
  events.push(GroupedEvent{event_schema:if three_la{"hf2q.s3.three-la-cb-event.v3"}else{"hf2q.s3.layer-group-cb-event.v2"},timing_profile:if three_la{THREE_LA_PROFILE}else{GROUPED_PROFILE},event_ordinal:events.len(),
   stage,layer,owner,host_boundary,host_clock:"rust.std.time.Instant.duration",host_elapsed_ns:host,
   completed_cb_gpu_duration_ns_truncated:(gpu>0).then_some(gpu),gpu_observation_boundary:"after_own_healthy_Result",
   healthy_result:true,sum_is_gpu_union:false});
 }else{ensure!(s.events.len()<CAP,"timing event overflow refuses case");s.events.push(Event{stage,layer,owner,original_host_submit_or_wait_ns:host,gpu_duration_ns_truncated:(gpu>0).then_some(gpu),healthy_result:true});}Ok(())})
}
#[cfg(test)]mod controls{
 use super::*;
 #[test]fn inactive_has_no_clocks_and_scope_restores(){assert!(start().unwrap().is_none());let s=Scope::enter().unwrap();assert!(start().unwrap().is_some());assert!(Scope::enter().is_err());assert!(s.take().unwrap().is_empty());assert!(start().unwrap().is_none());}
}

#[cfg(test)]mod grouped_controls{
 use super::*;
 #[test]fn profile_scope_is_explicit_and_legacy_not_reinterpreted(){require_grouped_or_inactive().unwrap();let legacy=Scope::enter().unwrap();assert!(require_grouped_or_inactive().is_err());assert!(legacy.take().unwrap().is_empty());let grouped=GroupedScope::enter().unwrap();require_grouped_or_inactive().unwrap();assert!(grouped.take().unwrap().is_empty());require_grouped_or_inactive().unwrap();}
 #[test]fn actual_grouped_boundary_mapping_refuses_submit_health(){assert_eq!(grouped_boundary("row-copy").unwrap(),("retained-row-copy-checked","wait_until_completed_after_ffn_anchor"));assert_eq!(grouped_boundary("shared-FFN").unwrap().0,"ffn-anchor-checked");assert!(grouped_boundary("submitted-healthy").is_err());}
}
