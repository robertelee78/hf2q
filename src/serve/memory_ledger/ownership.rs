use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{DeclaredBytes, Error, FrozenIdentity, Limits};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RootId(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HandleId(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ReservationId(pub u64);

/// A handle represents any owning reference, including a queued command,
/// captured graph, pool registration or view. Completion alone does not release
/// a retained command-buffer owner. Only explicit Release ends that ownership;
/// dropping the original wrapper must not release another owning handle.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Event {
    Allocate {
        root: RootId,
        handle: HandleId,
        bytes: DeclaredBytes,
    },
    Retain {
        source: HandleId,
        handle: HandleId,
    },
    /// Observation only: the completed command-buffer object may still retain
    /// resources. The owner must remain charged until explicit Release.
    Complete {
        handle: HandleId,
    },
    Release {
        handle: HandleId,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct Transition {
    pub identity: FrozenIdentity,
    pub current: DeclaredBytes,
    pub reserved: DeclaredBytes,
    pub peak: DeclaredBytes,
    pub allocated: DeclaredBytes,
    pub released: DeclaredBytes,
    /// Overlapping diagnostic subset of current roots; never add to current.
    pub completed_retained: DeclaredBytes,
    pub event_count: u64,
}

/// Pure simulation. &mut serializes all changes; no global production state or
/// locks exist here. Each failed operation leaves every field unchanged.
#[derive(Clone)]
pub struct Ownership {
    identity: FrozenIdentity,
    roots: BTreeMap<RootId, DeclaredBytes>,
    handles: BTreeMap<HandleId, RootId>,
    completed: BTreeSet<HandleId>,
    reservation: Option<(ReservationId, DeclaredBytes)>,
    seen_roots: BTreeSet<RootId>,
    seen_handles: BTreeSet<HandleId>,
    seen_reservations: BTreeSet<ReservationId>,
    peak: DeclaredBytes,
    allocated: DeclaredBytes,
    released: DeclaredBytes,
    event_count: u64,
}

impl Ownership {
    pub fn new(identity: FrozenIdentity) -> Result<Self, Error> {
        identity.validate()?;
        Ok(Self {
            identity,
            roots: BTreeMap::new(),
            handles: BTreeMap::new(),
            completed: BTreeSet::new(),
            reservation: None,
            seen_roots: BTreeSet::new(),
            seen_handles: BTreeSet::new(),
            seen_reservations: BTreeSet::new(),
            peak: DeclaredBytes::ZERO,
            allocated: DeclaredBytes::ZERO,
            released: DeclaredBytes::ZERO,
            event_count: 0,
        })
    }

    pub fn require_identity(&self, identity: &FrozenIdentity) -> Result<(), Error> {
        identity.validate()?;
        if identity != &self.identity {
            return Err(Error::Identity(
                "transition differs from frozen plan".into(),
            ));
        }
        Ok(())
    }

    fn roots_bytes(&self) -> Result<DeclaredBytes, Error> {
        self.roots
            .values()
            .try_fold(DeclaredBytes::ZERO, |sum, &bytes| sum.checked_add(bytes))
    }

    fn current(&self) -> Result<DeclaredBytes, Error> {
        self.roots_bytes()?
            .checked_add(self.reservation.map_or(DeclaredBytes::ZERO, |(_, b)| b))
    }

    fn observe(&mut self, limits: &Limits) -> Result<(), Error> {
        let current = self.current()?;
        limits.project(current)?;
        self.peak.owned = self.peak.owned.max(current.owned);
        self.peak.metal = self.peak.metal.max(current.metal);
        Ok(())
    }

    fn event(&mut self, event: Event) -> Result<(), Error> {
        match event {
            Event::Allocate {
                root,
                handle,
                bytes,
            } => {
                bytes.validate()?;
                if !self.seen_roots.insert(root) || !self.seen_handles.insert(handle) {
                    return Err(Error::Ownership("root or handle identity reused"));
                }
                self.allocated = self.allocated.checked_add(bytes)?;
                self.roots.insert(root, bytes);
                self.handles.insert(handle, root);
            }
            Event::Retain { source, handle } => {
                let root = *self
                    .handles
                    .get(&source)
                    .ok_or(Error::Ownership("retaining unknown source"))?;
                if !self.seen_handles.insert(handle) {
                    return Err(Error::Ownership("handle identity reused"));
                }
                self.handles.insert(handle, root);
            }
            Event::Release { handle } => {
                let root = self
                    .handles
                    .remove(&handle)
                    .ok_or(Error::Ownership("releasing unknown handle"))?;
                self.completed.remove(&handle);
                if !self.handles.values().any(|other| *other == root) {
                    let bytes = self
                        .roots
                        .remove(&root)
                        .ok_or(Error::Ownership("root missing while referenced"))?;
                    self.released = self.released.checked_add(bytes)?;
                }
            }
            Event::Complete { handle } => {
                if !self.handles.contains_key(&handle) {
                    return Err(Error::Ownership("completion of unknown owning handle"));
                }
                self.completed.insert(handle);
            }
        }
        self.event_count = self.event_count.checked_add(1).ok_or(Error::Overflow)?;
        Ok(())
    }

    /// Apply one atomic declared event batch. Existing owners must remain in
    /// the trace until explicit completion/release, including old growth roots.
    pub fn apply(
        &mut self,
        identity: &FrozenIdentity,
        events: &[Event],
        limits: &Limits,
    ) -> Result<(), Error> {
        self.require_identity(identity)?;
        let mut next = self.clone();
        next.observe(limits)?;
        for event in events {
            next.event(event.clone())?;
            next.observe(limits)?;
        }
        *self = next;
        Ok(())
    }

    /// One overlapping allocation transition permit. Reserve its complete
    /// additional peak, not a per-slot share checked against a stale total.
    pub fn reserve(
        &mut self,
        identity: &FrozenIdentity,
        id: ReservationId,
        bytes: DeclaredBytes,
        limits: &Limits,
    ) -> Result<(), Error> {
        self.require_identity(identity)?;
        if self.reservation.is_some() {
            return Err(Error::Ownership("overlapping transition already reserved"));
        }
        bytes.validate()?;
        let mut next = self.clone();
        if !next.seen_reservations.insert(id) {
            return Err(Error::Ownership("reservation identity reused"));
        }
        next.reservation = Some((id, bytes));
        next.observe(limits)?;
        *self = next;
        Ok(())
    }

    /// Replace a reservation with actual roots atomically. Intermediate events
    /// must fit its entire prior envelope. Do not release an old root merely
    /// because a cache pointer was swapped; queued handles keep it charged.
    pub fn commit(
        &mut self,
        identity: &FrozenIdentity,
        id: ReservationId,
        events: &[Event],
        limits: &Limits,
    ) -> Result<(), Error> {
        self.require_identity(identity)?;
        if self.reservation.map(|(other, _)| other) != Some(id) {
            return Err(Error::Ownership("unknown reservation"));
        }
        let envelope = self.current()?;
        let mut next = self.clone();
        next.reservation = None;
        next.observe(limits)?;
        for event in events {
            next.event(event.clone())?;
            if !envelope.covers(next.current()?) {
                return Err(Error::Ownership("transition exceeds reserved envelope"));
            }
            next.observe(limits)?;
        }
        *self = next;
        Ok(())
    }

    /// Simulation of cancelling unused capacity. Resources already allocated
    /// are separate roots and remain charged until their last owning release.
    pub fn cancel(&mut self, identity: &FrozenIdentity, id: ReservationId) -> Result<(), Error> {
        self.require_identity(identity)?;
        if self.reservation.map(|(other, _)| other) != Some(id) {
            return Err(Error::Ownership("unknown reservation"));
        }
        self.reservation = None;
        Ok(())
    }

    pub fn snapshot(&self) -> Transition {
        let completed_roots: BTreeSet<_> = self
            .completed
            .iter()
            .filter_map(|handle| self.handles.get(handle))
            .copied()
            .collect();
        let completed_retained = completed_roots
            .iter()
            .try_fold(DeclaredBytes::ZERO, |sum, root| {
                sum.checked_add(self.roots[root])
            })
            .expect("subset of validated ownership sums");
        // All mutation paths maintain checked sums before publication.
        Transition {
            identity: self.identity.clone(),
            current: self.current().expect("validated ownership sums"),
            reserved: self.reservation.map_or(DeclaredBytes::ZERO, |(_, b)| b),
            peak: self.peak,
            allocated: self.allocated,
            released: self.released,
            completed_retained,
            event_count: self.event_count,
        }
    }
}
