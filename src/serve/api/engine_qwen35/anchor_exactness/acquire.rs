//! Thin test-only bridge from existing execution boundaries to required tickets.
//! Inactive diagnostic scopes do not acquire evidence. A required C2 scope can
//! never silently lose its writer, session, native guard or terminal settlement.
use super::*;
use super::super::{forward_plan::SavedHidden, obligations::Ticket,
    registration::{NativeHead, Session, StepWork}, transaction_trace::EventKind};

fn registered<T>(operation: impl FnOnce(&mut Session, &mut Transactions) -> Result<T>) -> Result<Option<T>> {
    ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut();
        let Some(active) = active.as_mut() else { return Ok(None) };
        let Some(session) = active.session.as_mut() else { return Ok(None) };
        let result = operation(session, &mut active.transactions);
        if result.is_err() { session.invalidate(); }
        result.map(Some)
    })
}

fn owned<T>(id: u64, operation: impl FnOnce(&mut Session) -> Result<T>) -> Result<T> {
    ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut();
        let session = active.as_mut().and_then(|active| active.session.as_mut())
            .filter(|session| session.id == id).context("acquisition guard outlived its owning scope")?;
        let result = operation(session);
        if result.is_err() { session.invalidate(); }
        result
    })
}

fn invalidate(id: u64) {
    ACTIVE.with(|cell| {
        if let Some(session) = cell.borrow_mut().as_mut().and_then(|active| active.session.as_mut()) {
            if session.id == id { session.invalidate(); }
        }
    });
}

pub(crate) struct NativeGuard { owner: Option<(u64, u64)> }
impl NativeGuard {
    pub(crate) fn finish(mut self) -> Result<()> {
        if let Some((owner, call)) = self.owner.take() { owned(owner, |session| session.close_native(call))?; }
        Ok(())
    }
}
impl Drop for NativeGuard {
    fn drop(&mut self) { if let Some((owner, _)) = self.owner { invalidate(owner); } }
}

pub(crate) fn native_begin() -> Result<NativeGuard> {
    let owner = registered(|session, _| Ok((session.id, session.open_native()?)))?;
    Ok(NativeGuard { owner })
}

pub(crate) fn native_contract(heads: impl IntoIterator<Item = NativeHead>) -> Result<()> {
    registered(|session, _| session.native_contract(heads))?;
    Ok(())
}

pub(crate) fn plan_native() -> Result<()> {
    registered(|session, _| session.plan_native())?;
    Ok(())
}

pub(crate) struct StepGuard { owner: Option<(u64, Ticket)> }
impl StepGuard {
    pub(crate) fn finish(mut self) -> Result<()> {
        if let Some((owner, ticket)) = self.owner.take() { owned(owner, |session| session.finish_step(ticket))?; }
        Ok(())
    }
}
impl Drop for StepGuard {
    fn drop(&mut self) { if let Some((owner, _)) = self.owner { invalidate(owner); } }
}

pub(crate) fn begin_step(work: StepWork) -> Result<StepGuard> {
    let owner = registered(|session, transactions| {
        let ticket = session.begin_step(work)?;
        let state = session.state_ticket()?;
        let Transactions::Recording(writer) = transactions else {
            anyhow::bail!("required request step lost its transaction writer");
        };
        writer.begin(state, EventKind::Transaction)?;
        Ok((session.id, ticket))
    })?;
    Ok(StepGuard { owner })
}

pub(crate) fn enabled() -> bool {
    ACTIVE.with(|cell| cell.borrow().as_ref().is_some_and(|active| active.session.is_some()))
}

pub(super) fn capture_state(cache: &HybridKvCache, slot: SlotId) -> Result<()> {
    if slot != SLOT { return Ok(()); }
    ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut();
        let active = active.as_mut().context("required state scope absent")?;
        let session = active.session.as_mut().context("required state session absent")?;
        let result = (|| {
            let ticket = session.state_before_capture()?;
            let Transactions::Recording(writer) = &mut active.transactions else {
                anyhow::bail!("required state capture lost its writer");
            };
            ensure!(writer.current()?.ticket == ticket, "state stream belongs to another required step");
            writer.capture(cache, slot, &active.published_ledger)?;
            session.observed_state(ticket)
        })();
        if result.is_err() { session.invalidate(); }
        result
    })
}

pub(crate) fn settle(cache: &HybridKvCache, ledger: &[u32], emitted: &[u32]) -> Result<()> {
    ACTIVE.with(|cell| -> Result<()> {
        let active = cell.borrow();
        let active = active.as_ref().context("completed request has no acquisition scope")?;
        ensure!(active.observations.decisions.len() >= emitted.len()
            && active.observations.decisions.iter().take(emitted.len()).enumerate()
                .all(|(index, &(observed_index, token))| observed_index == index && token == emitted[index]),
            "settled production tokens differ from their required semantic observations");
        Ok(())
    })?;
    settle_outcome(cache, ledger, super::super::request_kind::Outcome::Completed { emitted_tokens: emitted.len() })
}
pub(crate) fn settle_outcome(cache: &HybridKvCache, ledger: &[u32],
    outcome: super::super::request_kind::Outcome) -> Result<()> {
    registered(|session, transactions| {
        let ticket = session.settlement_before(outcome)?;
        let Transactions::Recording(writer) = transactions else {
            anyhow::bail!("required settlement lost its writer");
        };
        writer.begin(ticket, EventKind::Settlement)?;
        writer.capture(cache, SLOT, ledger)?;
        session.settlement_after(ticket)
    })?.context("required settlement has no acquisition scope")?;
    Ok(())
}

pub(crate) fn require_decision(slot: SlotId, index: usize, vocab: usize, hidden_width: usize) -> Result<()> {
    if slot != SLOT || COMPARISON.with(|cell| cell.get()) { return Ok(()); }
    registered(|session, _| session.require_decision(index, vocab, hidden_width))?;
    Ok(())
}

pub(crate) fn saved_checkpoint(head: NativeHead, witness: u64, hidden: SavedHidden,
    binding: &acquisition::FileBinding) -> Result<()> {
    registered(|session, _| session.saved_checkpoint(head, witness, hidden, binding))?
        .context("saved consumption has no required acquisition scope")?;
    Ok(())
}

#[cfg(test)]
#[path = "acquire_controls.rs"]
mod controls;
