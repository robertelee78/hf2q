//! Test-owned live allocation ownership, separate from disk payload and the
//! externally calibrated model/drafter working set. Reservations precede reads
//! and allocations and survive every overlapping arm until its owner drops.
use super::*;
use super::{matrix_plan::Budget, state_layout::StateLayout, trace::State};
use std::{cell::RefCell, ops::Deref, rc::{Rc, Weak}};

pub(super) struct Reservation { budget: Weak<RefCell<Budget>>, bytes: u64, raw: bool }
impl std::fmt::Debug for Reservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveReservation").field("bytes", &self.bytes).finish()
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        if let Some(budget) = self.budget.upgrade() {
            if self.raw { budget.borrow_mut().release_raw(self.bytes); }
            else { budget.borrow_mut().release_live(self.bytes); }
        }
    }
}
pub(super) fn reserve(budget: &Rc<RefCell<Budget>>, bytes: u64) -> Result<Reservation> {
    budget.borrow_mut().reserve_live(bytes)?;
    Ok(Reservation { budget: Rc::downgrade(budget), bytes, raw: false })
}

thread_local! { static CURRENT: RefCell<Option<Weak<RefCell<Budget>>>> = const { RefCell::new(None) }; }
pub(super) struct Bound { budget: Weak<RefCell<Budget>> }
pub(super) fn bind(budget: &Rc<RefCell<Budget>>) -> Result<Bound> {
    CURRENT.with(|cell| {
        let mut current = cell.borrow_mut();
        ensure!(current.as_ref().and_then(Weak::upgrade).is_none(), "another matrix RAM scope is live");
        let weak = Rc::downgrade(budget);
        *current = Some(weak.clone());
        Ok(Bound { budget: weak })
    })
}
impl Drop for Bound {
    fn drop(&mut self) {
        CURRENT.with(|cell| {
            let mut current = cell.borrow_mut();
            if current.as_ref().is_some_and(|weak| Weak::ptr_eq(weak, &self.budget)) { current.take(); }
        });
    }
}

// Prefix-authority/history scopes have no generation session. Their full raw
// observation copies still owe the same aggregate raw and live-RAM admission.
pub(super) fn observe_raw(owner: &mut Option<Reservation>, bytes: u64) -> Result<()> {
    let budget = CURRENT.with(|cell| cell.borrow().as_ref().and_then(Weak::upgrade));
    let Some(budget) = budget else { return Ok(()) };
    let previous = owner.as_ref().map_or(0, |reservation| reservation.bytes);
    let total = previous.checked_add(bytes).context("prefix raw observation RAM overflow")?;
    if let Some(reservation) = owner.as_ref() {
        ensure!(reservation.raw && Weak::ptr_eq(&reservation.budget, &Rc::downgrade(&budget)),
            "raw observation outlived its matrix RAM scope");
    }
    budget.borrow_mut().retain_raw(bytes)?;
    if let Some(reservation) = owner.as_mut() { reservation.bytes = total; }
    else { *owner = Some(Reservation { budget: Rc::downgrade(&budget), bytes: total, raw: true }); }
    Ok(())
}

#[derive(Debug)]
pub(super) struct OwnedState { state: State, _ram: Reservation }
impl Deref for OwnedState {
    type Target = State;
    fn deref(&self) -> &State { &self.state }
}
impl PartialEq for OwnedState {
    fn eq(&self, other: &Self) -> bool { self.state == other.state }
}
impl Eq for OwnedState {}
impl OwnedState {
    pub(super) fn same(&self, other: &Self, context: &str) -> Result<()> {
        self.state.same(&other.state, context)
    }
    pub(super) fn read(budget: &Rc<RefCell<Budget>>, layout: &StateLayout,
        cache: &HybridKvCache, slot: SlotId) -> Result<Self> {
        let payload = layout.logical_bytes()?;
        budget.borrow_mut().state_readback(payload)?;
        // State::read allocates its output once; per-head scratch is appended
        // directly. Include container capacity and one page per possible region.
        let regions = layout.cursors.checked_mul(6).and_then(|n| layout.parity.checked_mul(2)
            .and_then(|p| n.checked_add(p))).context("state region-container bound overflow")?;
        let overhead = regions.checked_mul(4096 + std::mem::size_of::<Vec<u8>>())
            .and_then(|n| n.checked_add(std::mem::size_of::<State>()))
            .context("state container reservation overflow")?;
        let bytes = payload.checked_add(u64::try_from(overhead)?).context("state RAM reservation overflow")?;
        let ram = reserve(budget, bytes)?;
        let state = State::read(cache, slot)?;
        Ok(Self { state, _ram: ram })
    }
}

pub(super) fn read(fixture: &Fixture, cache: &HybridKvCache, slot: SlotId) -> Result<OwnedState> {
    let layout = StateLayout::from_cache(cache, slot, 0)?;
    OwnedState::read(&fixture.matrix_budget()?, &layout, cache, slot)
}

pub(super) fn lane_envelope(fixture: &Fixture) -> Result<u64> {
    let LoadedModel::Qwen35(qwen) = fixture.loaded.as_ref() else { unreachable!() };
    let cache = HybridKvCache::initial_cache_payload_bytes(&qwen.model.cfg,
        fixture.packet.capacity, SLOTS, qwen.tq_kv_active)?;
    // All four stores may own K=4 committed plus one pending image. One whole
    // four-slot cache bounds their single-plane image geometry, conservatively
    // including KV and both ping-pong planes. No residency is inferred here.
    let rows = qwen.vocab_size.checked_add(qwen.hidden_size)
        .and_then(|n| n.checked_add(fixture.packet.capacity as usize))
        .and_then(|n| n.checked_mul(4)).context("anchor row reservation overflow")?;
    let one = rows.checked_add(std::mem::size_of::<Qwen35Anchor>())
        .and_then(|n| n.checked_add(std::mem::size_of::<Qwen35SpecPrefixBoundary>()))
        .and_then(|n| n.checked_add(256)).context("anchor metadata reservation overflow")?;
    cache.checked_mul(6).and_then(|n| u64::try_from(one).ok()
        .and_then(|one| one.checked_mul(21)).and_then(|rows| n.checked_add(rows)))
        .context("four-store cache/anchor reservation overflow")
}

pub(super) fn metadata_envelope(budget: &Budget) -> Result<u64> {
    // Ledger/plan containers, prefix authority and one bounded streaming JSON
    // value are covered independently of file payload sizes.
    u64::try_from(budget.manifest.grants.events)?.checked_mul(1024)
        .and_then(|n| n.checked_add(budget.manifest.grants.metadata_bytes))
        .context("metadata live reservation overflow")
}

pub(super) fn preflight(fixture: &Fixture) -> Result<Reservation> {
    let budget = fixture.matrix_budget()?;
    let LoadedModel::Qwen35(qwen) = fixture.loaded.as_ref() else { unreachable!() };
    let cache = HybridKvCache::initial_cache_payload_bytes(&qwen.model.cfg,
        fixture.packet.capacity, SLOTS, qwen.tq_kv_active)?;
    let lane = lane_envelope(fixture)?;
    let metadata = metadata_envelope(&budget.borrow())?;
    // Three lane owners; three retained peer images plus actual/twin/single
    // final states and one transient read. Model/drafter scratch remains under
    // the separately admitted source-bound Q5/owned-RSS guard.
    let state = cache.checked_div(u64::from(SLOTS)).and_then(|n| n.checked_add(1024 * 1024))
        .context("state preflight bound overflow")?;
    let peak = lane.checked_mul(3).and_then(|n| state.checked_mul(7).and_then(|states| n.checked_add(states)))
        .and_then(|n| n.checked_add(metadata)).and_then(|n| n.checked_add(budget.borrow().manifest.grants.retained_raw_bytes))
        .context("whole-case live RAM preflight overflow")?;
    budget.borrow_mut().preflight_live(peak)?;
    reserve(&budget, metadata)
}

#[cfg(test)]
mod controls {
    use super::*;
    #[test]
    fn c2_live_ram_rejects_overlapping_arms_and_remains_failed_after_release() {
        let (_root, writer) = super::super::transaction_trace::cpu_fixture::writer("actual");
        let context = writer.context();
        let limit = context.budget.borrow().manifest.grants.live_ram_bytes;
        let first = reserve(&context.budget, limit - 1).unwrap();
        assert!(reserve(&context.budget, 2).is_err(), "separately small arms exceed their overlapping live cap");
        drop(first);
        assert!(reserve(&context.budget, 1).is_err(), "release does not erase a failed RAM admission");
    }

    #[test]
    fn c2_prefix_raw_growth_cannot_release_prior_live_observations() {
        let (_root, writer) = super::super::transaction_trace::cpu_fixture::writer("actual");
        let context = writer.context();
        let _scope = bind(&context.budget).unwrap();
        let mut owner = None;
        observe_raw(&mut owner, 1000).unwrap();
        observe_raw(&mut owner, 1000).unwrap();
        let limit = context.budget.borrow().manifest.grants.retained_raw_bytes;
        assert!(observe_raw(&mut owner, limit - 1999).is_err(),
            "growing an existing observation owner must not refund its earlier buffers");
        drop(owner);
        assert!(reserve(&context.budget, 1).is_err());
    }
}
