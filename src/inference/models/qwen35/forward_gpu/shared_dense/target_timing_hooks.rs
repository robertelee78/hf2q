//! Diagnostic-only property reads after existing healthy checks. DefaultNone.
use anyhow::{ensure,Context,Result};
use std::{cell::RefCell,time::Instant};
use serde::Serialize;
#[derive(Serialize)]pub(crate) struct Event{
    pub stage:&'static str,pub layer:Option<usize>,pub owner:Option<u32>,
    pub original_host_submit_or_wait_ns:u64,pub gpu_duration_ns_truncated:Option<u64>,
    pub healthy_result:bool,
}
struct State{events:Vec<Event>,context:(Option<usize>,Option<u32>)}
thread_local!{static STATE:RefCell<Option<State>>=const{RefCell::new(None)};}
const CAP:usize=1024;
pub(crate) struct Scope{armed:bool}
impl Scope{
 pub(crate) fn enter()->Result<Self>{STATE.with(|s|->Result<()>{let mut s=s.try_borrow_mut().context("timing reentry")?;ensure!(s.is_none(),"nested timing");let mut events=Vec::new();events.try_reserve_exact(CAP)?;*s=Some(State{events,context:(None,None)});Ok(())})?;Ok(Self{armed:true})}
 pub(crate) fn take(mut self)->Result<Vec<Event>>{let x=STATE.with(|s|s.try_borrow_mut().map_err(|_|anyhow::anyhow!("timing borrowed"))?.take().context("timing scope missing"))?;self.armed=false;Ok(x.events)}
}
impl Drop for Scope{fn drop(&mut self){if self.armed{STATE.with(|s|{if let Ok(mut s)=s.try_borrow_mut(){s.take();}});}}}
pub(crate) fn context(layer:usize,owner:u32)->Result<()>{STATE.with(|s|->Result<()>{let mut s=s.try_borrow_mut().context("timing context reentry")?;if let Some(s)=s.as_mut(){s.context=(Some(layer),Some(owner));}Ok(())})}
pub(crate) fn start()->Result<Option<Instant>>{STATE.with(|s|Ok(s.try_borrow().map_err(|_|anyhow::anyhow!("timing clock reentry"))?.as_ref().map(|_|Instant::now())))}
pub(crate) fn checked(stage:&'static str,e:&mlx_native::CommandEncoder,start:Option<Instant>,at:Option<(usize,Option<u32>)>)->Result<()>{
 let Some(start)=start else{return Ok(());};
 let host=u64::try_from(start.elapsed().as_nanos())?;
 // Caller has ALREADY checked exact CB Result. No wait/recommit/raw escape.
 let gpu=e.completed_gpu_interval_ns();
 STATE.with(|s|->Result<()>{let mut s=s.try_borrow_mut().context("timing callback reentry")?;let s=s.as_mut().context("lost timing scope")?;ensure!(s.events.len()<CAP,"timing event overflow refuses case");let(layer,owner)=at.map_or(s.context,|(l,o)|(Some(l),o));s.events.push(Event{stage,layer,owner,original_host_submit_or_wait_ns:host,gpu_duration_ns_truncated:(gpu>0).then_some(gpu),healthy_result:true});Ok(())})
}
#[cfg(test)]mod controls{
 use super::*;
 #[test]fn inactive_has_no_clocks_and_scope_restores(){assert!(start().unwrap().is_none());let s=Scope::enter().unwrap();assert!(start().unwrap().is_some());assert!(Scope::enter().is_err());assert!(s.take().unwrap().is_empty());assert!(start().unwrap().is_none());}
}
