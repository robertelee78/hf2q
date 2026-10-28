//! C1 normative oracle. Production restore integration remains a separate C2 gate.
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
#[path = "anchor_contract/tests.rs"]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Identity {
    pub artifact: [u8; 32],
    pub family_schema: String,
    pub tokenizer: [u8; 32],
    pub template: [u8; 32],
    pub configuration: [u8; 32],
    pub conversation: u64,
    pub slot_generation: u64,
    pub lineage_generation: u64,
    pub graft_generation: u64,
    pub graft_digest: [u8; 32],
    pub graft_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Owner {
    pub id: u64,
    pub extent: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Component {
    pub role: String,
    pub shape: Vec<usize>,
    pub element_bytes: usize,
    pub representation: String,
    pub cursor: usize,
    pub capacity: usize,
    pub bytes: Vec<u8>,
}

impl Component {
    fn validate(&self, expected: &Self) -> Result<(), &'static str> {
        let bytes = self
            .shape
            .iter()
            .try_fold(self.element_bytes, |n, d| n.checked_mul(*d));
        if self.role != expected.role
            || self.shape != expected.shape
            || self.element_bytes != expected.element_bytes
            || self.representation != expected.representation
            || bytes != Some(self.bytes.len())
            || self.capacity != expected.capacity
            || self.cursor > self.capacity
            || self.cursor != expected.cursor
        {
            return Err("incompatible component");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LiveRows {
    // Half-open byte offsets in a declared physical owner. Family adapters
    // translate their row/slot axes; these are not logical token positions.
    pub owner: u64,
    pub start: usize,
    pub end: usize,
}

impl LiveRows {
    fn overlaps(&self, other: &Self) -> bool {
        self.owner == other.owner && self.start < other.end && other.start < self.end
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Anchor {
    pub id: u64,
    pub request: u64,
    pub identity: Identity,
    pub logical_tokens: usize,
    pub prefix_digest: [u8; 32],
    pub components: Vec<Component>,
    pub owners: Vec<Owner>,
    pub required_live_rows: Vec<LiveRows>,
}

/// A ticket authorizes an unchanged committed payload; preflight does not duplicate it.
pub(crate) struct RestoreTicket {
    origin: u64,
    epoch: u64,
    anchor: u64,
}

fn next_origin() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .expect("fixture origin exhausted")
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Store {
    identity: Identity,
    keep: usize,
    quota: usize,
    cache: Vec<Owner>,
    committed: Vec<Anchor>,
    pending: Option<Anchor>,
    constructing: Option<(Anchor, Vec<Owner>)>,
    retired_readers: BTreeMap<u64, Vec<Owner>>,
    invalid: BTreeSet<u64>,
    completed: BTreeSet<u64>,
    poisoned: bool,
    origin: u64,
    epoch: u64,
    restored: Option<u64>,
    quarantine: Vec<Owner>,
}

pub(crate) fn unique_bytes<'a>(
    owners: impl IntoIterator<Item = &'a Owner>,
) -> Result<usize, &'static str> {
    let mut roots = BTreeMap::new();
    for owner in owners {
        if let Some(old) = roots.insert(owner.id, owner.extent) {
            if old != owner.extent {
                return Err("owner extent disagreement");
            }
        }
    }
    roots.values().try_fold(0usize, |n, bytes| {
        n.checked_add(*bytes).ok_or("owner sum overflow")
    })
}

pub(crate) fn physical_cursor(
    logical: usize,
    graft: usize,
    unconsumed: usize,
) -> Result<usize, &'static str> {
    // Callers choose the family's actual unconsumed-tail contract; no family fallback.
    logical
        .checked_add(graft)
        .and_then(|n| n.checked_sub(unconsumed))
        .ok_or("cursor overflow/underflow")
}

impl Store {
    pub fn new(
        identity: Identity,
        keep: usize,
        quota: usize,
        cache: Vec<Owner>,
    ) -> Result<Self, &'static str> {
        if keep == 0 || unique_bytes(&cache)? > quota {
            return Err("invalid store grant");
        }
        Ok(Self {
            identity,
            keep,
            quota,
            cache,
            committed: Vec::new(),
            pending: None,
            constructing: None,
            retired_readers: BTreeMap::new(),
            invalid: BTreeSet::new(),
            completed: BTreeSet::new(),
            poisoned: false,
            origin: next_origin(),
            epoch: 0,
            restored: None,
            quarantine: Vec::new(),
        })
    }

    fn changed(&mut self) {
        self.epoch = self.epoch.saturating_add(1);
        if self.epoch == u64::MAX {
            self.poisoned = true;
        }
    }

    fn owners(&self) -> impl Iterator<Item = &Owner> {
        self.cache
            .iter()
            .chain(self.committed.iter().flat_map(|a| a.owners.iter()))
            .chain(self.pending.iter().flat_map(|a| a.owners.iter()))
            .chain(
                self.constructing
                    .iter()
                    .flat_map(|(a, temporary)| a.owners.iter().chain(temporary.iter())),
            )
            .chain(self.retired_readers.values().flatten())
            .chain(self.quarantine.iter())
    }

    pub fn reserved(&self) -> Result<usize, &'static str> {
        unique_bytes(self.owners())
    }

    pub fn begin_capture(
        &mut self,
        anchor: Anchor,
        constructing: &[Owner],
    ) -> Result<usize, &'static str> {
        if self.poisoned
            || anchor.identity != self.identity
            || self.pending.is_some()
            || self.constructing.is_some()
            || self.invalid.contains(&anchor.id)
            || self.completed.contains(&anchor.request)
            || self.committed.iter().any(|a| a.id == anchor.id)
        {
            return Err("capture refused");
        }
        for range in &anchor.required_live_rows {
            let owner = self
                .cache
                .iter()
                .chain(anchor.owners.iter())
                .find(|o| o.id == range.owner)
                .ok_or("live dependency owner missing")?;
            if range.start >= range.end || range.end > owner.extent {
                return Err("live dependency outside owner");
            }
        }
        let payload = anchor.components.iter().try_fold(0usize, |n, c| {
            n.checked_add(c.bytes.len()).ok_or("payload overflow")
        })?;
        let cache_ids: BTreeSet<_> = self.cache.iter().map(|o| o.id).collect();
        if unique_bytes(anchor.owners.iter().filter(|o| !cache_ids.contains(&o.id)))? < payload {
            return Err("payload owner omitted");
        }
        // Include OLD committed owners and all construction roots before publication.
        let peak = unique_bytes(
            self.owners()
                .chain(anchor.owners.iter())
                .chain(constructing.iter()),
        )?;
        if peak > self.quota {
            return Err("capture quota");
        }
        self.changed();
        self.constructing = Some((anchor, constructing.to_vec()));
        Ok(peak)
    }

    pub fn finish_capture(&mut self, drain_known: bool) -> Result<(), &'static str> {
        if !drain_known {
            self.partial_failure(false);
            return Err("capture retirement unknown");
        }
        let (anchor, _) = self.constructing.take().ok_or("no capture")?;
        self.changed();
        self.pending = Some(anchor);
        Ok(())
    }

    pub fn stage(&mut self, anchor: Anchor) -> Result<usize, &'static str> {
        let peak = self.begin_capture(anchor, &[])?;
        self.finish_capture(true)?;
        Ok(peak)
    }

    pub fn cancel(&mut self, request: u64, executed_writes: &[LiveRows], drain_known: bool) {
        if !drain_known {
            self.partial_failure(false);
            return;
        }
        self.prune_overlapping(executed_writes);
        if self.pending.as_ref().map(|a| a.request) == Some(request) {
            self.pending = None;
        }
        if self.constructing.as_ref().map(|(a, _)| a.request) == Some(request) {
            self.constructing = None;
        }
        self.completed.insert(request);
        self.changed();
    }

    pub fn complete(&mut self, request: u64, success: bool) -> Result<(), &'static str> {
        if self.completed.contains(&request)
            || self.pending.as_ref().map(|a| a.request) != Some(request)
        {
            return Err("completion identity");
        }
        let anchor = self.pending.take().unwrap();
        self.changed();
        self.completed.insert(request);
        if success {
            self.committed.push(anchor);
            if self.committed.len() > self.keep {
                let evicted = self.committed.remove(0);
                self.invalid.insert(evicted.id);
            }
        }
        Ok(())
    }

    pub fn preflight(
        &self,
        id: u64,
        identity: &Identity,
        logical_tokens: usize,
        prefix_digest: &[u8; 32],
        required: &[Component],
    ) -> Result<RestoreTicket, &'static str> {
        if self.poisoned
            || self.pending.is_some()
            || self.constructing.is_some()
            || identity != &self.identity
            || self.invalid.contains(&id)
        {
            return Err("restore identity/lineage/transaction");
        }
        let anchor = self
            .committed
            .iter()
            .find(|a| a.id == id)
            .ok_or("anchor unavailable")?;
        if anchor.identity != *identity
            || anchor.logical_tokens != logical_tokens
            || &anchor.prefix_digest != prefix_digest
            || anchor.components.len() != required.len()
        {
            return Err("component count/prefix/identity");
        }
        let mut roles = BTreeSet::new();
        for (component, expected) in anchor.components.iter().zip(required) {
            if !roles.insert(&component.role) {
                return Err("duplicate component");
            }
            component.validate(expected)?;
        }
        Ok(RestoreTicket {
            origin: self.origin,
            epoch: self.epoch,
            anchor: id,
        })
    }

    pub fn apply(
        &mut self,
        ticket: RestoreTicket,
        live: &mut Vec<Component>,
    ) -> Result<(), &'static str> {
        if self.poisoned
            || ticket.origin != self.origin
            || ticket.epoch != self.epoch
            || !self.committed.iter().any(|a| a.id == ticket.anchor)
            || self.invalid.contains(&ticket.anchor)
        {
            return Err("stale restore ticket");
        }
        // The supplied live destination is part of the separately declared cache owner.
        // This finite oracle does not model GPU copy/allocator internals.
        let components = &self
            .committed
            .iter()
            .find(|a| a.id == ticket.anchor)
            .unwrap()
            .components;
        live.clone_from(components);
        self.changed();
        self.restored = Some(ticket.anchor);
        Ok(())
    }

    fn prune_overlapping(&mut self, writes: &[LiveRows]) {
        let overlaps = |a: &Anchor| {
            a.required_live_rows
                .iter()
                .any(|r| writes.iter().any(|w| r.overlaps(w)))
        };
        let mut invalidated = Vec::new();
        self.committed.retain(|a| {
            if overlaps(a) {
                invalidated.push(a.id);
                false
            } else {
                true
            }
        });
        for a in [
            self.pending.as_ref(),
            self.constructing.as_ref().map(|(a, _)| a),
        ]
        .into_iter()
        .flatten()
        {
            if overlaps(a) {
                invalidated.push(a.id);
                self.completed.insert(a.request);
            }
        }
        if self.pending.as_ref().is_some_and(&overlaps) {
            self.pending = None;
        }
        if self.constructing.as_ref().is_some_and(|(a, _)| overlaps(a)) {
            self.constructing = None;
        }
        self.invalid.extend(invalidated);
    }

    /// Must happen BEFORE the first divergent write may execute, including on later cancellation.
    pub fn before_writes(
        &mut self,
        restored: u64,
        writes: &[LiveRows],
    ) -> Result<(), &'static str> {
        if self.restored != Some(restored) || self.constructing.is_some() {
            return Err("write without validated restore or while capture reads are live");
        }
        let index = self
            .committed
            .iter()
            .position(|a| a.id == restored)
            .ok_or("restore branch")?;
        for anchor in &self.committed[index + 1..] {
            self.invalid.insert(anchor.id);
        }
        self.committed.truncate(index + 1);
        self.prune_overlapping(writes);
        self.restored = None;
        self.changed();
        Ok(())
    }

    pub fn partial_failure(&mut self, drain_known: bool) {
        // Unknown GPU retirement keeps every owner charged in quarantine.
        if !drain_known {
            let quarantine = self.owners().cloned().collect();
            self.quarantine = quarantine;
        }
        self.invalid.extend(self.committed.iter().map(|a| a.id));
        self.committed.clear();
        self.pending = None;
        self.constructing = None;
        self.poisoned = true;
        self.restored = None;
        self.changed();
    }

    pub fn retain_reader(&mut self, reader: u64, anchor_id: u64) -> Result<(), &'static str> {
        if self.retired_readers.contains_key(&reader) {
            return Err("reader identity");
        }
        let owners = self
            .committed
            .iter()
            .find(|a| a.id == anchor_id)
            .ok_or("reader needs committed anchor")?
            .owners
            .clone();
        let peak = unique_bytes(self.owners().chain(owners.iter()))?;
        if peak > self.quota {
            return Err("reader quota");
        }
        self.retired_readers.insert(reader, owners);
        self.changed();
        Ok(())
    }

    pub fn finish_reader(&mut self, reader: u64) -> Result<(), &'static str> {
        self.retired_readers
            .remove(&reader)
            .ok_or("unknown reader")?;
        self.changed();
        Ok(())
    }

    pub fn replace_identity(
        &mut self,
        identity: Identity,
        drain_known: bool,
    ) -> Result<(), &'static str> {
        if !drain_known {
            self.partial_failure(false);
            return Err("identity retirement unknown");
        }
        if identity.slot_generation < self.identity.slot_generation
            || identity.graft_generation < self.identity.graft_generation
            || identity.lineage_generation < self.identity.lineage_generation
        {
            return Err("generation regressed");
        }
        if identity.slot_generation <= self.identity.slot_generation
            && identity.graft_generation <= self.identity.graft_generation
            && identity.lineage_generation <= self.identity.lineage_generation
        {
            return Err("generation did not advance");
        }
        for a in self
            .pending
            .iter()
            .chain(self.constructing.iter().map(|(a, _)| a))
        {
            self.invalid.insert(a.id);
            self.completed.insert(a.request);
        }
        self.invalid.extend(self.committed.iter().map(|a| a.id));
        self.committed.clear();
        self.pending = None;
        self.constructing = None;
        self.identity = identity;
        self.restored = None;
        self.changed();
        Ok(())
    }

    pub fn retire_quarantine(&mut self, known_drain: bool) -> Result<(), &'static str> {
        if !known_drain {
            return Err("retirement proof missing");
        }
        self.quarantine.clear();
        self.changed();
        Ok(())
    }

    pub fn reset(&mut self, identity: Identity) -> Result<(), &'static str> {
        if !self.quarantine.is_empty() || !self.retired_readers.is_empty() {
            return Err("readers still live");
        }
        self.replace_identity(identity, true)?;
        self.poisoned = false;
        self.epoch = 0;
        self.origin = next_origin();
        Ok(())
    }
}
