//! Isolated one-layer group policy, public native0.15.12 only. Default caller OFF.
//! Retain exact encoders before submission; never recycle pool epochs in flight.
use super::*;
use mlx_native::CommandEncoder;
use std::cell::RefCell;
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
enum Kind { Delta,Copy,Aux,Sdpa,Ops14,Ffn }
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) enum PlainAsyncBoundary { Fence,Carry,Unlabeled,Sdpa,Ops14 }
struct Entry { enc:CommandEncoder,#[cfg(test)] kind:Kind,#[cfg(test)] layer:usize,#[cfg(test)] owner:Option<u32> }
#[derive(Clone,Copy,Debug)]
struct Epoch { layer:usize,width:usize,linear:bool,copies:usize,deltas:usize,aux:usize,sdpa:usize,ops14:usize,owner:Option<u32> }
impl Epoch {
    fn begin_row(&mut self,index:usize,owner:u32)->Result<()> {
        ensure!(self.owner.is_none()&&index==self.copies&&index<self.width,"row epoch order/duplicate refusal");
        self.owner=Some(owner); Ok(())
    }
    fn accept(&mut self,kind:Kind)->Result<u32> {
        let owner=self.owner.context("submission without selected row")?;
        match kind {
            Kind::Ffn=>anyhow::bail!("FFN is block-owned, not row-owned"),
            Kind::Delta=>{ensure!(self.linear&&self.aux==0&&self.sdpa==0&&self.ops14==0&&self.deltas==self.copies,"duplicate/nonlinear Delta");self.deltas+=1;}
            Kind::Copy=>{ensure!(self.deltas==if self.linear{self.copies+1}else{0},"copy before Delta");
                ensure!(self.aux==if self.linear{0}else{self.copies+1},"full-attention copy before actual Plain Aux");self.copies+=1;self.owner=None;}
            Kind::Aux=>{ensure!(!self.linear&&self.aux==self.copies&&self.sdpa==self.copies+1,"duplicate or linear Plain Aux");self.aux+=1;}
            Kind::Sdpa=>{ensure!(!self.linear&&self.sdpa==self.copies&&self.aux==self.copies&&self.ops14==self.copies+1,"duplicate or out-of-order SDPA");self.sdpa+=1;}
            Kind::Ops14=>{ensure!(!self.linear&&self.ops14==self.copies&&self.sdpa==self.copies&&self.aux==self.copies,"duplicate or out-of-order ops1-4");self.ops14+=1;}
        } Ok(owner)
    }
    fn complete(&self)->Result<()> {ensure!(self.copies==self.width&&self.owner.is_none()&&(self.deltas==if self.linear{self.width}else{0})&&(self.aux==if self.linear{0}else{self.width})&&(self.sdpa==if self.linear{0}else{self.width})&&(self.ops14==if self.linear{0}else{self.width}),"incomplete layer epoch");Ok(())}
}
#[derive(Clone,Copy,Debug)]
struct Block {first:usize,last:usize,next:usize}
impl Block {
 fn accepts(&self,layer:usize,linear:bool)->Result<()> {ensure!(linear&&layer==self.next&&layer<=self.last,"nonadjacent/nonlinear block continuation");Ok(())}
 fn advance_ffn(&mut self,layer:usize,linear:bool)->Result<()> {ensure!(linear&&layer==self.next&&layer>=self.first&&layer<self.last,"intermediate FFN outside three LA block");self.next=layer+1;Ok(())}
}
struct Pending { encoders:Vec<Entry>,submitted:usize,width:usize,epoch:Option<Epoch>,block:Option<Block>,three:bool }
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
            ensure!(p.is_none(),"previous group scope remains");let mut encoders=Vec::new();encoders.try_reserve_exact(14)?;
            #[cfg(test)]let three=super::target_timing_hooks::three_la_active();
            #[cfg(not(test))]let three=false;
            *p=Some(Pending{encoders,submitted:0,width,epoch:None,block:None,three});Ok(())})?;
        Ok(Self{_same_thread:std::marker::PhantomData})
    }
}
impl Drop for GraphDeferralScope {fn drop(&mut self){end_scope();}}
pub(super) fn require_retained_mode(unretained:bool)->Result<()> {ensure!(!unretained,"group refuses unretained-reference mode");Ok(())}
fn admit(current:usize,total:usize,width:usize)->Result<()> {
    ensure!((2..=4).contains(&width)&&current<4*width&&total<160*width,"finite layer/invocation CB cap exhausted");Ok(())
}
fn block_last(layer:usize,three:bool)->usize {if three&&layer%4<3{layer-layer%4+2}else{layer}}
pub(super) fn begin_layer(layer:usize,linear:bool)->Result<()> {
    PENDING.with(|p|->Result<()> {let mut p=p.try_borrow_mut().context("layer group reentry")?;let p=p.as_mut().context("layer outside group")?;
        ensure!(layer<64&&p.epoch.is_none(),"previous layer incomplete");
        if let Some(block)=p.block {block.accepts(layer,linear)?;}
        else {ensure!(p.encoders.is_empty(),"previous block not retired");ensure!(!p.three||layer%4==0||layer%4==3,"three LA block must begin at actual first layer");let last=block_last(layer,p.three);
            ensure!(!p.three||linear==(layer%4<3),"selected three LA pattern drift");
            p.block=Some(Block{first:layer,last,next:layer});}
        p.epoch=Some(Epoch{layer,width:p.width,linear,copies:0,deltas:0,aux:0,sdpa:0,ops14:0,owner:None});Ok(())})
}
pub(super) fn intermediate()->Result<bool> {PENDING.with(|p|{let p=p.try_borrow().context("block borrow")?;let p=p.as_ref().context("block absent")?;
    Ok(p.epoch.as_ref().context("epoch absent")?.layer<p.block.context("block absent")?.last)})}
fn admit_block(current:usize,total:usize,three:bool,width:usize)->Result<()> {
    if three {ensure!(width==2&&current<14&&total<352,"three LA finite CB cap exhausted");Ok(())}
    else {admit(current,total,width)}
}
pub(super) fn submit_ffn(enc:CommandEncoder)->Result<()> {submit_block_owned(enc,Kind::Ffn)}
fn submit_block_owned(enc:CommandEncoder,kind:Kind)->Result<()> {
    ensure!(kind==Kind::Ffn,"only block-owned FFN admitted");
    PENDING.with(|p|->Result<()> {let mut p=p.try_borrow_mut().context("FFN borrow")?;let p=p.as_mut().context("FFN outside group")?;
        admit_block(p.encoders.len(),p.submitted,p.three,p.width)?;
        let epoch=p.epoch.as_ref().context("FFN outside epoch")?;epoch.complete()?;
        let layer=epoch.layer;let block=p.block.as_mut().context("FFN outside block")?;
        let mut next=*block;next.advance_ffn(layer,epoch.linear)?;
        p.submitted+=1;store_before_submit(&mut p.encoders,Entry{enc,#[cfg(test)]kind,#[cfg(test)]layer,#[cfg(test)]owner:None},|entry,_|{entry.enc.commit();Ok(())})?;
        *block=next;p.epoch=None;Ok(())})
}
pub(super) fn begin_row(index:usize,owner:u32)->Result<()> {
    PENDING.with(|p|p.try_borrow_mut().context("row group reentry")?.as_mut().context("row outside group")?.epoch.as_mut().context("row outside layer")?.begin_row(index,owner))
}
/// Required BEFORE any pool reset or helper-root truncation. Existing free roots
/// from an earlier healthy epoch may still be allocated; issued roots cannot reenter
/// the free list, so parameter CPU writes cannot target an in-flight issued root.
pub(super) fn require_reset_allowed()->Result<()> {
    PENDING.with(|p|->Result<()> {let p=p.try_borrow().context("reset/group borrow conflict")?;
        if let Some(p)=p.as_ref(){ensure!(p.encoders.is_empty()&&p.epoch.is_none()&&p.block.is_none(),"pool reset/truncate while layer issued or pending");}Ok(())})
}
pub(super) fn submit(enc:CommandEncoder,_label:&'static str)->Result<()> {submit_kind(enc,Kind::Delta)}
pub(super) fn submit_copy(enc:CommandEncoder)->Result<()> {submit_kind(enc,Kind::Copy)}
#[cfg(test)]
fn chosen_timing_stage(kind:Kind)->Option<&'static str> {
    match kind {Kind::Ffn=>Some("retained-FFN-terminal-check"),Kind::Delta=>Some("retained-Delta-terminal-check"),Kind::Copy=>Some("row-copy"),Kind::Aux|Kind::Sdpa|Kind::Ops14=>None}
}
fn eligible_aux(boundary:PlainAsyncBoundary,label:&str,epoch:Option<&Epoch>)->bool {
    ((boundary==PlainAsyncBoundary::Fence&&label=="layer.full_attn.ops6-7")||
        (boundary==PlainAsyncBoundary::Sdpa&&label=="layer.full_attn.sdpa_kv")||
        (boundary==PlainAsyncBoundary::Ops14&&label=="layer.full_attn.ops1-4"))&&
        epoch.is_some_and(|e|!e.linear&&e.owner.is_some())
}
/// Only an originally-async Plain boundary inside the selected FA row epoch.
/// Every other site/epoch retains its prior healthy wait behavior.
pub(super) fn select_aux(boundary:PlainAsyncBoundary,label:&str)->Result<bool> {
    PENDING.with(|p|->Result<bool>{let p=p.try_borrow().context("Aux selection borrow conflict")?;
        Ok(eligible_aux(boundary,label,p.as_ref().and_then(|p|p.epoch.as_ref())))})
}
pub(super) fn submit_aux(enc:CommandEncoder)->Result<()> {submit_kind(enc,Kind::Aux)}
pub(super) fn submit_sdpa(enc:CommandEncoder)->Result<()> {submit_kind(enc,Kind::Sdpa)}
pub(super) fn completed_aux(sdpa:bool)->Result<()> {
    PENDING.with(|p|->Result<()> {let mut p=p.try_borrow_mut().context("aux completion borrow")?;
        p.as_mut().context("aux completion outside group")?.epoch.as_mut().context("aux completion outside layer")?.accept(if sdpa{Kind::Sdpa}else{Kind::Aux})?;Ok(())})
}

pub(super) fn submit_ops14(enc:CommandEncoder)->Result<()> {submit_kind(enc,Kind::Ops14)}
pub(super) fn select_ops14(seq:u32,head:u32,tq:bool)->Result<bool> {
    if seq!=1||head!=256||!tq{return Ok(false);}
    select_aux(PlainAsyncBoundary::Ops14,"layer.full_attn.ops1-4")
}
pub(super) fn may_defer_ops14(selected:bool,dump_enabled:bool)->bool {selected&&!dump_enabled}
/// Called only after the original checked wait succeeded, before any CPU dump.
pub(super) fn completed_ops14()->Result<()> {
    PENDING.with(|p|->Result<()> {let mut p=p.try_borrow_mut().context("ops14 completion borrow")?;
        p.as_mut().context("ops14 completion outside group")?.epoch.as_mut().context("ops14 completion outside layer")?.accept(Kind::Ops14)?;Ok(())})
}

pub(super) fn select_sdpa(head_dim:u32,tq:bool)->Result<bool> {
    if head_dim!=256||!tq{return Ok(false);}
    select_aux(PlainAsyncBoundary::Sdpa,"layer.full_attn.sdpa_kv")
}
fn store_before_submit<T>(entries:&mut Vec<T>,entry:T,submit:impl FnOnce(&mut T,usize)->Result<()>)->Result<()> {
    entries.push(entry);let count=entries.len();submit(entries.last_mut().context("owned entry missing")?,count)
}

fn submit_kind(enc:CommandEncoder,kind:Kind)->Result<()> {
    PENDING.with(|p|->Result<()> {let mut p=p.try_borrow_mut().context("submission group reentry")?;let p=p.as_mut().context("submission outside group")?;
        admit_block(p.encoders.len(),p.submitted,p.three,p.width)?;
        let epoch=p.epoch.as_mut().context("submission outside layer")?;let owner=Some(epoch.accept(kind)?);let layer=epoch.layer;
        // Ownership enters pre-reserved bounded storage BEFORE exactly one commit.
        p.submitted+=1;
        store_before_submit(&mut p.encoders,Entry{enc,#[cfg(test)] kind,#[cfg(test)] layer,#[cfg(test)] owner},|entry,_|{
            entry.enc.commit();Ok(())
        })?;
        #[cfg(not(test))]let _=(layer,owner);Ok(())})

}
fn ensure_terminal(layer:usize,block:Block)->Result<()> {ensure!(layer==block.last,"nonterminal block anchor");Ok(())}
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
        let epoch=p.epoch.as_ref().context("fence without layer")?;
        let boundary=ensure_terminal(epoch.layer,p.block.context("fence without block")?);
        let epoch=boundary.and_then(|()|epoch.complete());
        release_after_anchor(&mut p.encoders,anchor,epoch,|entry| {
            #[cfg(test)]let clock=chosen_timing_stage(entry.kind).map(|stage|(stage,super::target_timing_hooks::start()));
            let health=entry.enc.wait_until_completed().map_err(anyhow::Error::from);
            // Real status is inspected even when timing metadata itself failed.
            #[cfg(test)]let health=match clock{None=>health,Some((stage,clock))=>health.and_then(|()|super::target_timing_hooks::checked(
                stage,
                &entry.enc,clock?,Some((entry.layer,entry.owner))))};
            health
        })?;
        p.epoch=None;p.block=None;Ok(())})
}
pub(super) fn require_empty()->Result<()> {require_reset_allowed()}
fn end_scope(){PENDING.with(|p|{let Ok(mut p)=p.try_borrow_mut()else{failed();return;};
    if let Some(p)=p.take(){if !p.encoders.is_empty()||p.epoch.is_some()||p.block.is_some(){failed();std::mem::forget(p);}}});}
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
        assert!(admit(7,319,2).is_ok());
        assert!(admit(8,0,2).is_err());
        assert!(admit(0,320,2).is_err());
        assert!(admit(15,639,4).is_ok());
        assert!(admit(16,0,4).is_err());
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
        let mut e=Epoch{layer:0,width:2,linear:true,copies:0,deltas:0,aux:0,sdpa:0,ops14:0,owner:None};
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
        PENDING.with(|p|{let mut p=p.borrow_mut();let p=p.as_mut().unwrap();p.epoch=None;p.block=None;});
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

    #[test]
    fn aux_selection_is_exact_original_async_fa_boundary_only() {
        let mut e=Epoch{layer:3,width:2,linear:false,copies:0,deltas:0,aux:0,sdpa:0,ops14:0,owner:None};
        assert!(!eligible_aux(PlainAsyncBoundary::Fence,"layer.full_attn.ops6-7",None));
        assert!(!eligible_aux(PlainAsyncBoundary::Fence,"layer.full_attn.ops6-7",Some(&e)));
        e.begin_row(0,3).unwrap();
        assert!(eligible_aux(PlainAsyncBoundary::Sdpa,"layer.full_attn.sdpa_kv",Some(&e)));
        assert!(!eligible_aux(PlainAsyncBoundary::Sdpa,"layer.full_attn.ops6-7",Some(&e)));
        assert!(eligible_aux(PlainAsyncBoundary::Fence,"layer.full_attn.ops6-7",Some(&e)));
        for boundary in [PlainAsyncBoundary::Carry,PlainAsyncBoundary::Unlabeled] {
            assert!(!eligible_aux(boundary,"layer.full_attn.ops6-7",Some(&e)));
        }
        assert!(!eligible_aux(PlainAsyncBoundary::Fence,"other",Some(&e)));
        e.linear=true;
        assert!(!eligible_aux(PlainAsyncBoundary::Fence,"layer.full_attn.ops6-7",Some(&e)));
    }
    #[test]
    fn fa_aux_is_required_once_per_row_without_delta_or_copy_progress() {
        let mut e=Epoch{layer:3,width:2,linear:false,copies:0,deltas:0,aux:0,sdpa:0,ops14:0,owner:None};
        assert!(e.accept(Kind::Aux).is_err());
        for (index,owner) in [(0,3),(1,0)] {
            e.begin_row(index,owner).unwrap();
            assert!(e.accept(Kind::Copy).is_err());
            assert!(e.accept(Kind::Aux).is_err());
            assert!(e.accept(Kind::Sdpa).is_err());
            e.accept(Kind::Ops14).unwrap();
            assert!(e.accept(Kind::Ops14).is_err());
            e.accept(Kind::Sdpa).unwrap();
            assert!(e.accept(Kind::Sdpa).is_err());
            e.accept(Kind::Aux).unwrap();
            assert_eq!(e.copies,index);assert_eq!(e.deltas,0);
            assert!(e.accept(Kind::Aux).is_err());
            e.accept(Kind::Copy).unwrap();
            assert!(e.accept(Kind::Aux).is_err());
        }
        e.complete().unwrap();
        e.linear=true;e.owner=Some(3);assert!(e.accept(Kind::Aux).is_err());
    }
    #[test]
    fn entry_is_owned_before_single_submit_and_error_keeps_it() {
        let mut entries=Vec::new();let mut calls=0;
        store_before_submit(&mut entries,7,|entry,count|{calls+=1;assert_eq!(count,1);assert_eq!(*entry,7);Ok(())}).unwrap();
        assert_eq!(calls,1);assert_eq!(entries,vec![7]);
        assert!(store_before_submit(&mut entries,8,|_,count|{assert_eq!(count,2);anyhow::bail!("submit diagnostic")}).is_err());
        assert_eq!(entries,vec![7,8]);
    }
    #[test]
    fn auxiliary_health_has_no_chosen_timing_record() {
        assert_eq!(chosen_timing_stage(Kind::Aux),None);
        assert_eq!(chosen_timing_stage(Kind::Sdpa),None);
        assert_eq!(chosen_timing_stage(Kind::Ops14),None);
        assert_eq!(chosen_timing_stage(Kind::Copy),Some("row-copy"));
        assert_eq!(chosen_timing_stage(Kind::Delta),Some("retained-Delta-terminal-check"));
    }

    #[test]
    fn sdpa_selected_geometry_and_outside_scope_keep_old_wait() {
        assert!(!select_sdpa(256,true).unwrap());
        assert!(!select_sdpa(512,true).unwrap());
        assert!(!select_sdpa(256,false).unwrap());
        let checked=CheckedScope::enter().unwrap();let graph=GraphDeferralScope::enter(2).unwrap();
        begin_layer(3,false).unwrap();begin_row(0,3).unwrap();
        assert!(select_sdpa(256,true).unwrap());
        assert!(!select_sdpa(512,true).unwrap());assert!(!select_sdpa(256,false).unwrap());
        assert!(require_reset_allowed().is_err());
        PENDING.with(|p|{let mut p=p.borrow_mut();let p=p.as_mut().unwrap();p.epoch=None;p.block=None;});
        drop(graph);drop(checked);
    }

    #[test]
    fn ops14_dump_fallback_requires_healthy_completion_and_keeps_reset_closed() {
        assert!(!select_ops14(1,256,true).unwrap());
        let checked=CheckedScope::enter().unwrap();let graph=GraphDeferralScope::enter(2).unwrap();
        begin_layer(3,false).unwrap();begin_row(0,3).unwrap();
        assert!(select_ops14(1,256,true).unwrap());
        assert!(!select_ops14(2,256,true).unwrap());assert!(!select_ops14(1,512,true).unwrap());
        assert!(!select_ops14(1,256,false).unwrap());
        assert!(may_defer_ops14(true,false));assert!(!may_defer_ops14(true,true));
        assert!(!may_defer_ops14(false,false));
        let mut called=false;
        let failed_wait:Result<()>=Err(anyhow::anyhow!("actual wait Error"));
        assert!(failed_wait.and_then(|()|{called=true;completed_ops14()}).is_err());
        assert!(!called);assert!(require_reset_allowed().is_err());
        completed_ops14().unwrap();assert!(completed_ops14().is_err());
        assert!(completed_aux(false).is_err());
        completed_aux(true).unwrap();assert!(completed_aux(true).is_err());
        completed_aux(false).unwrap();assert!(completed_aux(false).is_err());
        assert!(require_reset_allowed().is_err());
        PENDING.with(|p|{let mut p=p.borrow_mut();let p=p.as_mut().unwrap();p.epoch=None;p.block=None;});
        drop(graph);drop(checked);
    }

    #[test]
    fn three_la_block_order_caps_and_terminal_are_actual_helpers() {
        let mut b=Block{first:0,last:block_last(0,true),next:0};
        assert_eq!(b.last,2);assert_eq!(block_last(3,true),3);assert_eq!(block_last(0,false),0);
        b.accepts(0,true).unwrap();assert!(b.accepts(1,true).is_err());assert!(b.accepts(0,false).is_err());
        b.advance_ffn(0,true).unwrap();assert!(b.advance_ffn(0,true).is_err());
        b.accepts(1,true).unwrap();assert!(ensure_terminal(1,b).is_err());b.advance_ffn(1,true).unwrap();
        b.accepts(2,true).unwrap();ensure_terminal(2,b).unwrap();assert!(b.advance_ffn(2,true).is_err());
        assert!(admit_block(13,351,true,2).is_ok());assert!(admit_block(14,0,true,2).is_err());
        assert!(admit_block(0,352,true,2).is_err());assert!(admit_block(0,0,true,4).is_err());
        assert!(admit_block(7,319,false,2).is_ok());assert!(admit_block(8,0,false,2).is_err());
    }
    #[test]
    fn three_la_open_block_blocks_reset_between_epochs_even_without_entries() {
        let timing=super::super::target_timing_hooks::ThreeLaScope::enter().unwrap();
        let checked=CheckedScope::enter().unwrap();let graph=GraphDeferralScope::enter(2).unwrap();
        begin_layer(0,true).unwrap();
        PENDING.with(|p|{let mut p=p.borrow_mut();let p=p.as_mut().unwrap();p.epoch=None;p.block.as_mut().unwrap().next=1;});
        assert!(require_reset_allowed().is_err());assert!(begin_layer(2,true).is_err());begin_layer(1,true).unwrap();
        PENDING.with(|p|{let mut p=p.borrow_mut();let p=p.as_mut().unwrap();p.epoch=None;p.block=None;});
        drop(graph);drop(checked);assert!(timing.take().unwrap().is_empty());
    }
    #[test]
    fn three_la_anchor_error_checks_all_fourteen_and_keeps_all_owners() {
        let owner=std::rc::Rc::new(());let mut entries=vec![owner.clone();14];let mut seen=0;
        assert!(release_after_anchor(&mut entries,Err(anyhow::anyhow!("anchor")),Ok(()),|_|{seen+=1;if seen==7{anyhow::bail!("retained");}Ok(())}).is_err());
        assert_eq!(seen,14);assert_eq!(entries.len(),14);assert_eq!(std::rc::Rc::strong_count(&owner),15);
        assert_eq!(chosen_timing_stage(Kind::Ffn),Some("retained-FFN-terminal-check"));
    }

}
