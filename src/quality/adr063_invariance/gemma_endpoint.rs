//! Opt-in actual Gemma endpoint capture. No device or submission is created here.
use super::{capture, receipt::*, store::Store};
use mlx_native::MlxBuffer;
use std::cell::RefCell;
struct Active { store: Store, lane: usize, start: usize, rows: usize, next: usize, owners: Option<Vec<(usize,usize)>> }
thread_local! { static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) }; }
pub struct Scope;
impl Drop for Scope { fn drop(&mut self) { ACTIVE.with(|s| { s.borrow_mut().take(); }); } }
pub fn install(store: Store) -> Result<Scope, String> {
    if PAUSED.with(|p|p.get()) {return Err("cannot install a canonical capture during unpublished pause".into());}
    ACTIVE.with(|s| { let mut s=s.borrow_mut();
        if s.is_some() { return Err("nested Gemma endpoint capture".into()); }
        *s=Some(Active { store,lane:0,start:0,rows:0,next:0,owners:None }); Ok(Scope)
    })
}
pub fn begin_call(lane:usize,start:usize,rows:usize)->Result<(),String> {
    if lane>=4 || rows==0 || start.checked_add(rows).is_none_or(|n| n>144) {
        return Err("Gemma actual row scope outside frozen workload".into());
    }
    ACTIVE.with(|s| { let mut s=s.borrow_mut();let a=s.as_mut().ok_or("missing Gemma capture")?;
        if a.next!=a.rows { return Err("previous Gemma endpoint rows incomplete".into()); }
        a.lane=lane;a.start=start;a.rows=rows;a.next=0;a.owners=None;Ok(()) })
}
/// Exact source descriptor for one packed row stream: slot,start,length,offset.
pub type PackedLayout = Vec<(u32,usize,usize,usize)>;
pub fn packed_owners(layout:&[(u32,usize,usize,usize)])->Result<Vec<(usize,usize)>,String> {
    if layout.len()!=4 {return Err("Gemma T2 requires four independent source sequences".into());}
    let mut owners=Vec::new();let mut seen=std::collections::BTreeSet::new();
    for &(slot,start,len,offset) in layout {
        if slot>=4 || !seen.insert(slot) || start!=0 || len!=128 || offset!=owners.len() {
            return Err("Gemma packed source slots/offsets/frozen lengths mismatch".into());
        }
        owners.extend((0..len).map(|row|(slot as usize,start+row)));
    }Ok(owners)
}
pub fn begin_packed(layout:&[(u32,usize,usize,usize)])->Result<(),String> {
    let owners=packed_owners(layout)?;
    ACTIVE.with(|s| {let mut s=s.borrow_mut();let a=s.as_mut().ok_or("missing Gemma capture")?;
        if a.next!=a.rows {return Err("previous Gemma endpoint incomplete".into());}
        a.start=0;a.rows=owners.len();a.next=0;a.owners=Some(owners);Ok(())})
}
/// Source descriptor must exactly join the independently frozen call scope.
pub fn head_rows(start:usize,rows:usize,multi:Option<PackedLayout>)->Result<Option<Vec<usize>>,String> {
    ACTIVE.with(|s| { let s=s.borrow(); let Some(a)=s.as_ref() else {return Ok(None)};
        let actual=multi.as_ref().map(|m|packed_owners(m)).transpose()?;
        if a.start!=start || a.rows!=rows || a.next!=0 || a.owners!=actual {
            return Err("Gemma endpoint head/source row scope mismatch".into());
        }
        Ok(Some((0..rows).collect())) })
}
pub fn logits(row:usize,buffer:&MlxBuffer,vocab:usize)->Result<(),String> {
    ACTIVE.with(|s| { let mut s=s.borrow_mut(); let Some(a)=s.as_mut() else{return Ok(())};
        if row!=a.next || row>=a.rows {return Err("Gemma duplicate/omitted head row".into());}
        let view=buffer.slice_view(0,vocab).with_shape(vec![1,vocab]).map_err(|e|e.to_string())?;
        let (lane,position)=a.owners.as_ref().map_or((a.lane,a.start+row),|m|m[row]);
        capture::after_checked_drain(&mut a.store,Coordinate { logical_step:0,layer:u32::MAX,
            stage:Stage::Logits,agent:format!("agent-{lane}"),logical_row_start:position as u64,segment:0 },
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

/// Actual CPU Vec returned after the existing batched head's checked wait.
/// Parent identity here is explicitly the downloaded host Vec, not a GPU root.
pub fn downloaded_logits(lane:usize,position:usize,values:&[f32])->Result<(),String> {
    let bytes:Vec<u8>=values.iter().flat_map(|v|v.to_le_bytes()).collect();
    with_store(|store|store.capture(Coordinate {logical_step:0,layer:u32::MAX,stage:Stage::Logits,
        agent:format!("agent-{lane}"),logical_row_start:position as u64,segment:0},Scalar::F32,
        vec![1,values.len()],values.as_ptr() as usize as u64,bytes.len() as u64,0,None,
        "actual Gemma batched body/head checked return; CPU downloaded row",&bytes))
}

thread_local! {static PAUSED:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};}
/// Same-thread unpublished-tail lease. Does not submit, wait, alter a buffer,
/// or relabel required canonical rows as captured.
pub struct UnpublishedPause {
    active:Option<Active>,
    _same_thread:std::marker::PhantomData<std::rc::Rc<()>>,
}
pub fn pause_unpublished_tail()->Result<UnpublishedPause,String> {
    end_call()?;
    if PAUSED.with(|p|p.replace(true)) {return Err("nested unpublished capture pause".into());}
    let active=ACTIVE.with(|s|s.borrow_mut().take());
    if active.is_none() {PAUSED.with(|p|p.set(false));return Err("missing actual capture to pause".into());}
    Ok(UnpublishedPause {active,_same_thread:std::marker::PhantomData})
}
impl Drop for UnpublishedPause {
    fn drop(&mut self) {
        ACTIVE.with(|s| {let mut s=s.borrow_mut();assert!(s.is_none(),"capture installed during unpublished pause");
            *s=self.active.take();});
        PAUSED.with(|p|p.set(false));
    }
}
