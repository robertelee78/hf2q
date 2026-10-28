//! Opt-in actual Gemma endpoint capture. No device or submission is created here.
use super::{capture, receipt::*, store::Store};
use mlx_native::MlxBuffer;
use std::cell::RefCell;
struct Active { store: Store, lane: usize, start: usize, rows: usize, next: usize }
thread_local! { static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) }; }
pub struct Scope;
impl Drop for Scope { fn drop(&mut self) { ACTIVE.with(|s| { s.borrow_mut().take(); }); } }
pub fn install(store: Store) -> Result<Scope, String> {
    ACTIVE.with(|s| { let mut s=s.borrow_mut();
        if s.is_some() { return Err("nested Gemma endpoint capture".into()); }
        *s=Some(Active { store,lane:0,start:0,rows:0,next:0 }); Ok(Scope)
    })
}
pub fn begin_call(lane:usize,start:usize,rows:usize)->Result<(),String> {
    if lane>=4 || rows==0 || start.checked_add(rows).is_none_or(|n| n>144) {
        return Err("Gemma actual row scope outside frozen workload".into());
    }
    ACTIVE.with(|s| { let mut s=s.borrow_mut();let a=s.as_mut().ok_or("missing Gemma capture")?;
        if a.next!=a.rows { return Err("previous Gemma endpoint rows incomplete".into()); }
        a.lane=lane;a.start=start;a.rows=rows;a.next=0;Ok(()) })
}
/// Only ordinary single-sequence head loops may expand. T2 must use its own
/// actual packed-input/membership adapter and cannot be inferred from this.
pub fn head_rows(start:usize,rows:usize,multi:bool)->Result<Option<Vec<usize>>,String> {
    ACTIVE.with(|s| { let s=s.borrow(); let Some(a)=s.as_ref() else {return Ok(None)};
        if multi || a.start!=start || a.rows!=rows || a.next!=0 {
            return Err("Gemma endpoint head/source row scope mismatch".into());
        }
        Ok(Some((0..rows).collect())) })
}
pub fn logits(row:usize,buffer:&MlxBuffer,vocab:usize)->Result<(),String> {
    ACTIVE.with(|s| { let mut s=s.borrow_mut(); let Some(a)=s.as_mut() else{return Ok(())};
        if row!=a.next || row>=a.rows {return Err("Gemma duplicate/omitted head row".into());}
        let view=buffer.slice_view(0,vocab).with_shape(vec![1,vocab]).map_err(|e|e.to_string())?;
        capture::after_checked_drain(&mut a.store,Coordinate { logical_step:0,layer:u32::MAX,
            stage:Stage::Logits,agent:format!("agent-{}",a.lane),logical_row_start:(a.start+row) as u64,segment:0 },
            &view,&[1,vocab],None,"actual Gemma endpoint checked head finish/slot-aware return")?;
        a.next+=1;Ok(()) })
}
pub fn end_call()->Result<(),String> { ACTIVE.with(|s| {
    let s=s.borrow();let a=s.as_ref().ok_or("missing Gemma endpoint")?;
    if a.next!=a.rows {Err("Gemma actual endpoint did not capture all source rows".into())}else{Ok(())}
}) }
pub fn with_store<T>(f:impl FnOnce(&mut Store)->Result<T,String>)->Result<T,String> {
    ACTIVE.with(|s| {let mut s=s.borrow_mut();f(&mut s.as_mut().ok_or("missing Gemma endpoint")?.store)})
}
pub fn finish()->Result<Trace,String> {
    let incomplete=end_call().err();
    ACTIVE.with(|s| {
        let mut a=s.borrow_mut().take().ok_or("missing Gemma endpoint")?;
        if let Some(error)=incomplete {a.store.fail(error);}
        a.store.finish()
    })
}
