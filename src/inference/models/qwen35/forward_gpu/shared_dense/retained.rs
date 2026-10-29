//! Isolated one-layer group policy, public native0.15.12 only. Default caller OFF.
//! Retain exact encoders before submission; never recycle pool epochs in flight.
use super::*;
use mlx_native::CommandEncoder;
use std::cell::RefCell;
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
enum Kind { Delta,Copy }
struct Entry { enc:CommandEncoder,#[cfg(test)] kind:Kind,#[cfg(test)] layer:usize,#[cfg(test)] owner:u32 }
#[derive(Clone,Copy,Debug)]
struct Epoch { layer:usize,width:usize,linear:bool,copies:usize,deltas:usize,owner:Option<u32> }
impl Epoch {
    fn begin_row(&mut self,index:usize,owner:u32)->Result<()> {
        ensure!(self.owner.is_none()&&index==self.copies&&index<self.width,"row epoch order/duplicate refusal");
        self.owner=Some(owner); Ok(())
    }
    fn accept(&mut self,kind:Kind)->Result<u32> {
        let owner=self.owner.context("submission without selected row")?;
        match kind {
            Kind::Delta=>{ensure!(self.linear&&self.deltas==self.copies,"duplicate/nonlinear Delta");self.deltas+=1;}
            Kind::Copy=>{ensure!(self.deltas==if self.linear{self.copies+1}else{0},"copy before Delta");self.copies+=1;self.owner=None;}
        } Ok(owner)
    }
    fn complete(&self)->Result<()> {ensure!(self.copies==self.width&&self.owner.is_none()&&self.deltas==if self.linear{self.width}else{0},"incomplete layer epoch");Ok(())}
}
struct Pending { encoders:Vec<Entry>,submitted:usize,width:usize,epoch:Option<Epoch> }
thread_local! {static PENDING:RefCell<Option<Pending>>=const{RefCell::new(None)};}
pub(super) fn active()->bool {PENDING.with(|p|p.try_borrow().map(|p|p.is_some()).unwrap_or(false))}
pub(super) fn require_width(width:usize)->Result<()> {ensure!(width==2,"one-layer grouped spike admits ONLY width2");Ok(())}
pub(super) struct GraphDeferralScope {_same_thread:std::marker::PhantomData<std::rc::Rc<()>>}
impl GraphDeferralScope {
    pub(super) fn enter(width:usize)->Result<Self> {
        ensure!(checked_active(),"graph group requires checked causal scope");require_width(width)?;
        require_retained_mode(mlx_native::command_buffer_uses_unretained_references())?;
        #[cfg(test)]super::target_timing_hooks::require_grouped_or_inactive()?;
        ensure!(!mlx_native::kernel_profile::is_enabled(),"group refuses sync-redirecting MLX_PROFILE_CB");
        PENDING.with(|p|->Result<()> {let mut p=p.try_borrow_mut().context("group scope reentry")?;
            ensure!(p.is_none(),"previous group scope remains");let mut encoders=Vec::new();encoders.try_reserve_exact(2*width)?;
            *p=Some(Pending{encoders,submitted:0,width,epoch:None});Ok(())})?;
        Ok(Self{_same_thread:std::marker::PhantomData})
    }
}
impl Drop for GraphDeferralScope {fn drop(&mut self){end_scope();}}
pub(super) fn require_retained_mode(unretained:bool)->Result<()> {ensure!(!unretained,"group refuses unretained-reference mode");Ok(())}
fn admit(current:usize,total:usize,width:usize)->Result<()> {
    ensure!((2..=4).contains(&width)&&current<2*width&&total<112*width,"finite layer/invocation CB cap exhausted");Ok(())
}
pub(super) fn begin_layer(layer:usize,linear:bool)->Result<()> {
    PENDING.with(|p|->Result<()> {let mut p=p.try_borrow_mut().context("layer group reentry")?;let p=p.as_mut().context("layer outside group")?;
        ensure!(layer<64&&p.encoders.is_empty()&&p.epoch.is_none(),"previous layer not retired");
        p.epoch=Some(Epoch{layer,width:p.width,linear,copies:0,deltas:0,owner:None});Ok(())})
}
pub(super) fn begin_row(index:usize,owner:u32)->Result<()> {
    PENDING.with(|p|p.try_borrow_mut().context("row group reentry")?.as_mut().context("row outside group")?.epoch.as_mut().context("row outside layer")?.begin_row(index,owner))
}
/// Required BEFORE any pool reset or helper-root truncation. Existing free roots
/// from an earlier healthy epoch may still be allocated; issued roots cannot reenter
/// the free list, so parameter CPU writes cannot target an in-flight issued root.
pub(super) fn require_reset_allowed()->Result<()> {
    PENDING.with(|p|->Result<()> {let p=p.try_borrow().context("reset/group borrow conflict")?;
        if let Some(p)=p.as_ref(){ensure!(p.encoders.is_empty()&&p.epoch.is_none(),"pool reset/truncate while layer issued or pending");}Ok(())})
}
pub(super) fn submit(enc:CommandEncoder,_label:&'static str)->Result<()> {submit_kind(enc,Kind::Delta)}
pub(super) fn submit_copy(enc:CommandEncoder)->Result<()> {submit_kind(enc,Kind::Copy)}
fn submit_kind(enc:CommandEncoder,kind:Kind)->Result<()> {
    PENDING.with(|p|->Result<()> {let mut p=p.try_borrow_mut().context("submission group reentry")?;let p=p.as_mut().context("submission outside group")?;
        admit(p.encoders.len(),p.submitted,p.width)?;
        let epoch=p.epoch.as_mut().context("submission outside layer")?;let owner=epoch.accept(kind)?;let layer=epoch.layer;
        // Ownership enters pre-reserved bounded storage BEFORE exactly one commit.
        p.encoders.push(Entry{enc,#[cfg(test)] kind,#[cfg(test)] layer,#[cfg(test)] owner});
        #[cfg(not(test))]let _=(layer,owner);p.submitted+=1;
        p.encoders.last_mut().context("missing retained encoder")?.enc.commit();Ok(())})
}
fn check_all<T>(entries:&[T],mut check:impl FnMut(&T)->Result<()>)->Result<()> {
    let mut first=None;for e in entries {if let Err(error)=check(e){if first.is_none(){first=Some(error);}}}
    match first{Some(error)=>Err(error),None=>Ok(())}
}
#[cfg(test)]
fn release_after_checks<T>(entries:&mut Vec<T>,check:impl FnMut(&T)->Result<()>)->Result<()> {check_all(entries,check)?;entries.clear();Ok(())}
fn release_after_anchor<T>(entries:&mut Vec<T>,anchor:Result<()>,epoch:Result<()>,check:impl FnMut(&T)->Result<()>)->Result<()> {
    // Even an actual anchor Error or metadata refusal cannot skip another CB.
    let checked=check_all(entries,check);anchor?;epoch?;checked?;entries.clear();Ok(())
}
/// Existing FFN Result is the anchor, never health authority for any other CB.
pub(super) fn after_layer_fence(anchor:Result<()>)->Result<()> {
    PENDING.with(|p|->Result<()> {let mut p=p.try_borrow_mut().context("fence group reentry")?;let p=p.as_mut().context("fence outside group")?;
        let epoch=p.epoch.as_ref().context("fence without layer")?.complete();
        release_after_anchor(&mut p.encoders,anchor,epoch,|entry| {
            #[cfg(test)]let clock=super::target_timing_hooks::start();
            let health=entry.enc.wait_until_completed().map_err(anyhow::Error::from);
            // Real status is inspected even when timing metadata itself failed.
            #[cfg(test)]let health=health.and_then(|()|super::target_timing_hooks::checked(
                match entry.kind{Kind::Delta=>"retained-Delta-terminal-check",Kind::Copy=>"row-copy"},
                &entry.enc,clock?,Some((entry.layer,Some(entry.owner)))));
            health
        })?;
        p.epoch=None;Ok(())})
}
pub(super) fn require_empty()->Result<()> {require_reset_allowed()}
fn end_scope(){PENDING.with(|p|{let Ok(mut p)=p.try_borrow_mut()else{failed();return;};
    if let Some(p)=p.take(){if !p.encoders.is_empty()||p.epoch.is_some(){failed();std::mem::forget(p);}}});}
#[cfg(test)]
mod controls {
    use super::*;
    #[test]
    fn scalar_checked_fixture_never_starts_graph_deferral() {
        checked_fixture(|| {
            assert!(checked_active());
            assert!(!active());
            Ok(())
        })
        .unwrap();
        assert!(!checked_active());
        assert!(!active());
    }
    #[test]
    fn graph_authority_is_separate_and_scoped() {
        let checked = CheckedScope::enter().unwrap();
        assert!(!active());
        {
            let graph = GraphDeferralScope::enter(2).unwrap();
            assert!(active());
            assert!(GraphDeferralScope::enter(2).is_err());
            drop(graph);
        }
        assert!(!active());
        assert!(checked_active());
        drop(checked);
    }
    #[test]
    fn actual_mode_guard_refuses_unretained() {
        assert!(require_retained_mode(false).is_ok());
        assert!(require_retained_mode(true).is_err());
    }
    #[test]
    fn admission_is_before_finite_commit_boundary() {
        assert!(admit(0,0,2).is_ok());
        assert!(admit(3,223,2).is_ok());
        assert!(admit(4,0,2).is_err());
        assert!(admit(0,224,2).is_err());
        assert!(admit(7,447,4).is_ok());
        assert!(admit(8,0,4).is_err());
        assert!(require_width(4).is_err());
    }
    #[test]
    fn actual_check_loop_checks_every_result_and_keeps_first_error() {
        let mut seen = Vec::new();
        let result = check_all(&[0, 1, 2], |i| {
            seen.push(*i);
            if *i != 1 {
                anyhow::bail!("CB{i}");
            }
            Ok(())
        });
        assert_eq!(seen, vec![0, 1, 2]);
        assert!(result.unwrap_err().to_string().contains("CB0"));
    }
    #[test]
    fn failed_check_retains_every_owner_until_quarantine() {
        let owner = std::rc::Rc::new(());
        let mut entries = vec![owner.clone(), owner.clone()];
        assert!(release_after_checks(&mut entries, |_| anyhow::bail!("earlier Error")).is_err());
        assert_eq!(entries.len(), 2);
        assert_eq!(std::rc::Rc::strong_count(&owner), 3);
        let mut healthy = vec![owner.clone()];
        release_after_checks(&mut healthy, |_| Ok(())).unwrap();
        assert!(healthy.is_empty());
        assert_eq!(entries.len(), 2);
        assert_eq!(std::rc::Rc::strong_count(&owner), 3);
    }
    #[test]
    fn empty_and_all_healthy_check_loops_succeed() {
        assert!(check_all::<u8>(&[], |_| anyhow::bail!("not called")).is_ok());
        assert!(check_all(&[0, 1], |_| Ok(())).is_ok());
    }
    #[test]
    fn actual_epoch_requires_ordered_delta_copy_and_all_rows() {
        let mut e=Epoch{layer:0,width:2,linear:true,copies:0,deltas:0,owner:None};
        assert!(e.complete().is_err());e.begin_row(0,3).unwrap();
        assert!(e.accept(Kind::Copy).is_err());e.accept(Kind::Delta).unwrap();
        assert!(e.accept(Kind::Delta).is_err());e.accept(Kind::Copy).unwrap();
        assert!(e.begin_row(0,3).is_err());e.begin_row(1,1).unwrap();
        e.accept(Kind::Delta).unwrap();e.accept(Kind::Copy).unwrap();e.complete().unwrap();
        assert!(e.begin_row(2,0).is_err());
    }
    #[test]
    fn actual_reset_and_truncate_gate_refuses_open_epoch_before_first_commit() {
        let checked=CheckedScope::enter().unwrap();let graph=GraphDeferralScope::enter(2).unwrap();
        require_reset_allowed().unwrap();begin_layer(0,false).unwrap();
        assert!(require_reset_allowed().is_err());
        let mut roots=vec![1,2];let mut reset_called=false;
        assert!(retire_row_scratch(&mut roots,1,||{reset_called=true;Ok(())}).is_err());
        assert_eq!(roots,vec![1,2]);assert!(!reset_called);
        // Host-only teardown: no encoder was created/submitted. Do not invent
        // actual completion for this model-free negative gate test.
        PENDING.with(|p|p.borrow_mut().as_mut().unwrap().epoch=None);
        require_reset_allowed().unwrap();drop(graph);drop(checked);
    }
    #[test]
    fn actual_anchor_error_checks_all_and_releases_no_owners() {
        let owner=std::rc::Rc::new(());let mut entries=vec![owner.clone(),owner.clone(),owner.clone()];
        let mut seen=0;
        let result=release_after_anchor(&mut entries,Err(anyhow::anyhow!("anchor Error")),Ok(()),|_|{seen+=1;Ok(())});
        assert!(result.is_err());assert_eq!(seen,3);assert_eq!(entries.len(),3);
        assert_eq!(std::rc::Rc::strong_count(&owner),4);
        let mut index=0;
        assert!(release_after_anchor(&mut entries,Ok(()),Ok(()),|_|{index+=1;if index==2{anyhow::bail!("retained Error");}Ok(())}).is_err());
        assert_eq!(index,3);assert_eq!(entries.len(),3);
        release_after_anchor(&mut entries,Ok(()),Ok(()),|_|Ok(())).unwrap();
        assert!(entries.is_empty());assert_eq!(std::rc::Rc::strong_count(&owner),1);
    }

}
