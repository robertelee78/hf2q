//! Required execution tickets are declared by the request/native API boundary,
//! not by the lower plan/capture hooks. Omitting both hooks leaves a required
//! ticket open. This ledger contains no tensor values or inferred extents.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(super) struct Ticket(u32);

impl Ticket {
    pub(super) fn index(self) -> usize { self.0 as usize }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(super) enum Kind {
    Case,
    Request,
    Step,
    NativeCall,
    Projection,
    Selection,
    State,
    Checkpoint,
    SavedCheckpointConsumption,
    Settlement,
}

impl Kind {
    fn branch(self) -> bool {
        matches!(self, Self::Case | Self::Request | Self::Step | Self::NativeCall)
    }
    fn accepts(self, child: Self) -> bool {
        match self {
            Self::Case => child == Self::Request,
            Self::Request => matches!(child, Self::Step | Self::Settlement),
            Self::Step => matches!(child, Self::NativeCall | Self::Selection | Self::State
                | Self::Checkpoint | Self::SavedCheckpointConsumption),
            Self::NativeCall => matches!(child, Self::Projection | Self::Selection),
            _ => false,
        }
    }
    fn bit(self) -> u16 { 1 << self as u16 }
    fn required_children(self) -> u16 {
        match self {
            Self::Case => Self::Request.bit(),
            Self::Request => Self::Settlement.bit(),
            Self::Step => Self::State.bit(),
            Self::NativeCall => Self::Projection.bit(),
            _ => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(super) enum Phase { Required, Running, Planned, Complete }

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Entry {
    pub parent: Option<Ticket>,
    pub kind: Kind,
    pub phase: Phase,
    remaining: u32,
    children: u16,
    required_children: u16,
}

pub(super) fn validate_complete(entries: &[Entry]) -> Result<()> {
    ensure!(!entries.is_empty() && entries.len() <= 65536, "finite completed obligation tree required");
    let mut children = vec![0_u16; entries.len()];
    for (index, entry) in entries.iter().enumerate() {
        ensure!(entry.phase == Phase::Complete && entry.remaining == 0,
            "Incomplete: published execution has an open obligation");
        if let Some(parent) = entry.parent {
            ensure!(parent.index() < index && entries[parent.index()].kind.accepts(entry.kind),
                "execution obligation has a cyclic or incompatible parent");
            children[parent.index()] |= entry.kind.bit();
        } else {
            ensure!(index == 0 && entry.kind == Kind::Case, "execution has another root");
        }
        ensure!(entry.required_children & entry.kind.required_children() == entry.kind.required_children(),
            "execution weakened a mandatory child-kind requirement");
    }
    for (index, entry) in entries.iter().enumerate() {
        ensure!(entry.children == children[index]
            && children[index] & entry.required_children == entry.required_children,
            "Incomplete: completed execution omitted its required child kinds");
    }
    Ok(())
}

pub(super) struct Obligations {
    entries: Vec<Entry>,
    limit: usize,
    failed: bool,
}

impl Obligations {
    pub(super) fn new(limit: usize) -> Result<Self> {
        ensure!((1..=65536).contains(&limit), "finite C2 obligation grant required");
        let mut entries = Vec::with_capacity(limit);
        entries.push(Entry { parent: None, kind: Kind::Case, phase: Phase::Required,
            remaining: 0, children: 0, required_children: Kind::Case.required_children() });
        Ok(Self { entries, limit, failed: false })
    }
    pub(super) fn case(&self) -> Ticket { Ticket(0) }
    pub(super) fn entries(&self) -> &[Entry] { &self.entries }
    pub(super) fn failed(&self) -> bool { self.failed }
    pub(super) fn invalidate(&mut self) { self.failed = true; }

    // Every error is sticky, even if a caller ignores the Result.
    fn checked<T>(&mut self, operation: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        ensure!(!self.failed, "Incomplete: C2 obligation ledger previously failed");
        let result = operation(self);
        if result.is_err() { self.failed = true; }
        result
    }

    /// Called by the upper execution boundary before the work it requires.
    /// Case requests and native-call heads are frozen before their parent starts;
    /// request steps and step calls follow genuine production scheduling.
    pub(super) fn require(&mut self, parent: Ticket, kind: Kind) -> Result<Ticket> {
        self.checked(|this| {
            ensure!(this.entries.len() < this.limit, "C2 obligation grant exhausted");
            let parent_entry = this.entries.get_mut(parent.index())
                .ok_or_else(|| anyhow::anyhow!("unknown C2 parent ticket"))?;
            ensure!(parent_entry.kind.accepts(kind), "invalid C2 obligation parent/kind");
            ensure!(parent_entry.phase == Phase::Required
                || (parent_entry.phase == Phase::Running
                    && matches!(parent_entry.kind, Kind::Request | Kind::Step)),
                "C2 parent cannot acquire late obligations");
            parent_entry.remaining = parent_entry.remaining.checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("C2 child count overflow"))?;
            parent_entry.children |= kind.bit();
            let ticket = Ticket(u32::try_from(this.entries.len())?);
            this.entries.push(Entry { parent: Some(parent), kind, phase: Phase::Required,
                remaining: 0, children: 0, required_children: kind.required_children() });
            Ok(ticket)
        })
    }

    /// The selected production work may require additional child kinds, e.g.
    /// a native decode plus selection, or saved equality consumption. This is
    /// declared before the step starts, independently of lower hooks.
    pub(super) fn require_children(&mut self, ticket: Ticket, kinds: &[Kind]) -> Result<()> {
        self.checked(|this| {
            let entry = this.entries.get_mut(ticket.index())
                .ok_or_else(|| anyhow::anyhow!("unknown C2 obligation ticket"))?;
            ensure!(entry.kind.branch() && entry.phase == Phase::Required,
                "C2 work contract must precede execution");
            for &kind in kinds {
                ensure!(entry.kind.accepts(kind), "invalid required C2 child kind");
                entry.required_children |= kind.bit();
            }
            Ok(())
        })
    }

    pub(super) fn start(&mut self, ticket: Ticket) -> Result<()> {
        self.checked(|this| {
            let entry = this.entries.get(ticket.index())
                .ok_or_else(|| anyhow::anyhow!("unknown C2 obligation ticket"))?;
            ensure!(entry.kind.branch() && entry.phase == Phase::Required,
                "C2 parent execution is one-shot");
            if let Some(parent) = entry.parent {
                ensure!(this.entries[parent.index()].phase == Phase::Running,
                    "C2 parent must run before its child");
            }
            // Dynamic steps/calls may be added later, but the independent
            // terminal/state/head obligation must already exist.
            let initial = entry.kind.required_children();
            ensure!(entry.children & initial == initial,
                "C2 execution lacks its independent required child");
            this.entries[ticket.index()].phase = Phase::Running;
            Ok(())
        })
    }

    /// A lower pre-capture planner acknowledges an already-required ticket.
    /// Requested tensor geometry is checked by the source-derived plan builder.
    pub(super) fn plan(&mut self, ticket: Ticket) -> Result<()> {
        self.checked(|this| {
            let entry = this.entries.get(ticket.index())
                .ok_or_else(|| anyhow::anyhow!("unknown C2 obligation ticket"))?;
            ensure!(!entry.kind.branch() && entry.phase == Phase::Required,
                "C2 evidence plan is missing or duplicated");
            let parent = entry.parent.ok_or_else(|| anyhow::anyhow!("C2 evidence lacks parent"))?;
            ensure!(this.entries[parent.index()].phase == Phase::Running,
                "C2 evidence plan precedes execution boundary");
            this.entries[ticket.index()].phase = Phase::Planned;
            Ok(())
        })
    }

    /// Leaf completion follows actual observation/store closure; parent
    /// completion reconciles its entire required set, not only observed hooks.
    pub(super) fn complete(&mut self, ticket: Ticket) -> Result<()> {
        self.checked(|this| {
            let entry = this.entries.get(ticket.index())
                .ok_or_else(|| anyhow::anyhow!("unknown C2 obligation ticket"))?;
            if entry.kind.branch() {
                ensure!(entry.phase == Phase::Running && entry.remaining == 0,
                    "Incomplete: C2 parent has unfinished required work");
                ensure!(entry.children & entry.required_children == entry.required_children,
                    "Incomplete: selected C2 work omitted a required child kind");
            } else {
                ensure!(entry.phase == Phase::Planned,
                    "Incomplete: C2 observation lacks its plan or completed twice");
            }
            let parent = entry.parent;
            this.entries[ticket.index()].phase = Phase::Complete;
            if let Some(parent) = parent {
                let parent = &mut this.entries[parent.index()];
                ensure!(parent.phase == Phase::Running && parent.remaining > 0,
                    "C2 evidence completed outside its parent lifetime");
                parent.remaining -= 1;
            }
            Ok(())
        })
    }

    pub(super) fn finish(&mut self) -> Result<()> {
        self.checked(|this| {
            ensure!(this.entries[0].phase == Phase::Complete,
                "Incomplete: C2 case obligations are unfinished");
            Ok(())
        })
    }
}

#[cfg(test)]
mod controls {
    use super::*;

    fn native_case() -> (Obligations, Ticket, Ticket, Ticket, Ticket, Ticket) {
        let mut ledger = Obligations::new(32).unwrap();
        let request = ledger.require(ledger.case(), Kind::Request).unwrap();
        let settlement = ledger.require(request, Kind::Settlement).unwrap();
        ledger.start(ledger.case()).unwrap();
        ledger.start(request).unwrap();
        let step = ledger.require(request, Kind::Step).unwrap();
        ledger.require_children(step, &[Kind::NativeCall]).unwrap();
        let state = ledger.require(step, Kind::State).unwrap();
        ledger.start(step).unwrap();
        let call = ledger.require(step, Kind::NativeCall).unwrap();
        let projection = ledger.require(call, Kind::Projection).unwrap();
        ledger.start(call).unwrap();
        (ledger, request, settlement, step, state, projection)
    }

    #[test]
    fn c2_obligations_symmetric_lower_plan_and_observation_omission_is_fatal() {
        for _arm in ["cold", "resumed"] {
            let (mut ledger, _, _, _, _, projection) = native_case();
            let call = ledger.entries()[projection.index()].parent.unwrap();
            // Drop the entire lower path: neither plan nor observation runs.
            assert!(ledger.complete(call).is_err());
            assert!(ledger.failed());
            assert!(ledger.plan(projection).is_err(), "an ignored failure cannot be repaired");
            assert!(ledger.finish().is_err());
        }
    }

    #[test]
    fn c2_obligations_missing_settlement_cannot_publish_a_case() {
        let (mut ledger, request, _, step, state, projection) = native_case();
        let call = ledger.entries()[projection.index()].parent.unwrap();
        ledger.plan(projection).unwrap();
        ledger.complete(projection).unwrap();
        ledger.complete(call).unwrap();
        ledger.plan(state).unwrap();
        ledger.complete(state).unwrap();
        ledger.complete(step).unwrap();
        assert!(ledger.complete(request).is_err());
        assert!(ledger.finish().is_err());
    }

    #[test]
    fn c2_obligations_duplicate_observation_and_unplanned_observation_are_fatal() {
        let (mut ledger, _, _, _, _, projection) = native_case();
        assert!(ledger.complete(projection).is_err());
        assert!(ledger.finish().is_err());
        let (mut ledger, _, _, _, _, projection) = native_case();
        ledger.plan(projection).unwrap();
        ledger.complete(projection).unwrap();
        assert!(ledger.complete(projection).is_err());
        assert!(ledger.finish().is_err());
    }

    #[test]
    fn c2_obligations_native_work_cannot_drop_its_whole_call() {
        let mut ledger = Obligations::new(16).unwrap();
        let request = ledger.require(ledger.case(), Kind::Request).unwrap();
        ledger.require(request, Kind::Settlement).unwrap();
        ledger.start(ledger.case()).unwrap();
        ledger.start(request).unwrap();
        let step = ledger.require(request, Kind::Step).unwrap();
        ledger.require_children(step, &[Kind::NativeCall]).unwrap();
        let state = ledger.require(step, Kind::State).unwrap();
        ledger.start(step).unwrap();
        ledger.plan(state).unwrap();
        ledger.complete(state).unwrap();
        assert!(ledger.complete(step).is_err());
        assert!(ledger.finish().is_err());
    }
}

#[test]
fn c2_obligations_ignored_premature_case_finish_is_sticky() {
    let mut ledger = Obligations::new(8).unwrap();
    let request = ledger.require(ledger.case(), Kind::Request).unwrap();
    assert!(ledger.finish().is_err());
    assert!(ledger.require(request, Kind::Settlement).is_err());
}

#[test]
fn c2_obligations_complete_case_cannot_publish_after_duplicate_evidence() {
    let mut ledger = Obligations::new(32).unwrap();
    let request = ledger.require(ledger.case(), Kind::Request).unwrap();
    let settlement = ledger.require(request, Kind::Settlement).unwrap();
    ledger.start(ledger.case()).unwrap();
    ledger.start(request).unwrap();
    let step = ledger.require(request, Kind::Step).unwrap();
    ledger.require_children(step, &[Kind::NativeCall]).unwrap();
    let state = ledger.require(step, Kind::State).unwrap();
    ledger.start(step).unwrap();
    let call = ledger.require(step, Kind::NativeCall).unwrap();
    let projection = ledger.require(call, Kind::Projection).unwrap();
    ledger.start(call).unwrap();
    ledger.plan(projection).unwrap();
    ledger.complete(projection).unwrap();
    ledger.complete(call).unwrap();
    ledger.plan(state).unwrap();
    ledger.complete(state).unwrap();
    ledger.complete(step).unwrap();
    ledger.plan(settlement).unwrap();
    ledger.complete(settlement).unwrap();
    ledger.complete(request).unwrap();
    ledger.complete(ledger.case()).unwrap();
    ledger.finish().unwrap();
    assert!(ledger.complete(projection).is_err(), "duplicate evidence is not another complete case");
    assert!(ledger.finish().is_err(), "ignored duplicate evidence must invalidate publication");
}
