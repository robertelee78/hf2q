//! Each required state event registers allocator-derived extents before readback
//! and streams one ordinary strict Q1 child store. Settlement is a distinct
//! event, not an invented native call or extra scheduler transaction.
use super::*;
use anyhow::Context as _;
use super::{obligations::Ticket, state_layout::StateLayout, trace::State};
use std::{cell::RefCell, collections::BTreeSet, rc::Rc};

pub(super) struct Context {
    pub root: PathBuf,
    pub case: String,
    pub arm: String,
    pub agent: String,
    pub identity: Identity,
    pub acquisition: acquisition::Acquisition,
    pub registration_sha256: String,
    pub budget: Rc<RefCell<super::matrix_plan::Budget>>,
    pub payload_limit: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(super) enum EventKind { Transaction, Settlement }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(super) struct Event {
    pub ordinal: u64,
    pub ticket: Ticket,
    pub kind: EventKind,
}

#[derive(Serialize)]
struct Child {
    event: Event,
    root: String,
    plan: String,
    plan_sha256: String,
    trace_sha256: String,
    logical_bytes: u64,
}

pub(super) struct Capture {
    store: Store,
    layout: StateLayout,
    event: Event,
    root: PathBuf,
    plan_sha256: String,
    logical_bytes: u64,
}

#[derive(Serialize)]
struct StatePlan<'a> {
    schema: &'static str,
    identity: &'a Identity,
    scope_acquisition: &'a acquisition::Acquisition,
    registration_sha256: &'a str,
    case: &'a str,
    arm: &'a str,
    slot: u32,
    event: Event,
    store_root: &'a Path,
    layout: &'a StateLayout,
    expected: &'a [ExpectedTensor],
    derived_before_state_read: bool,
}

#[derive(Serialize)]
struct Index<'a> {
    schema: &'static str,
    identity: &'a Identity,
    scope_acquisition: &'a acquisition::Acquisition,
    registration_sha256: &'a str,
    case: &'a str,
    arm: &'a str,
    agent: &'a str,
    slot: u32,
    declared_events: &'a [Event],
    children: &'a [Child],
    all_declared_state_events_completed: bool,
}

pub(super) struct Writer {
    context: Rc<Context>,
    root: PathBuf,
    declared: Vec<Event>,
    children: Vec<Child>,
    tickets: BTreeSet<usize>,
    current: Option<Event>,
    failed: bool,
}

impl Writer {
    pub(super) fn new(fixture: &Fixture, case: &str, arm: &str) -> Result<Self> {
        ensure!(fixture.selected(case)? && matches!(arm, "actual" | "twin" | "single"),
            "transaction stream case/arm was not registered");
        let context = Rc::new(Context { root: fixture.packet.output.clone(), case: case.into(),
            arm: arm.into(), agent: format!("{case}/slot0"), identity: fixture.packet.identity.clone(),
            acquisition: fixture.observed(&fixture.packet.output)?, budget: fixture.matrix_budget()?,
            registration_sha256: fixture.packet.registration.as_ref()
                .context("state stream requires its frozen registration binding")?.sha256.clone(),
            payload_limit: fixture.packet.host_payload_limit });
        Self::with_context(context)
    }
    fn with_context(context: Rc<Context>) -> Result<Self> {
        let root = context.root.join(format!("{}-{}-transactions", context.case, context.arm));
        fs::create_dir(&root)?;
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))?;
        Ok(Self { context, root, declared: Vec::new(), children: Vec::new(),
            tickets: BTreeSet::new(), current: None, failed: false })
    }
    pub(super) fn context(&self) -> Rc<Context> { Rc::clone(&self.context) }

    /// Upper request/step code declares the state event before executing work.
    pub(super) fn begin(&mut self, ticket: Ticket, kind: EventKind) -> Result<()> {
        let result = (|| {
            ensure!(!self.failed && self.current.is_none() && self.tickets.insert(ticket.index()),
                "state event is overlapping, duplicated or previously failed");
            ensure!(self.declared.len() < self.context.budget.borrow().manifest.grants.events,
                "state event grant exhausted");
            let event = Event { ordinal: u64::try_from(self.declared.len())?, ticket, kind };
            self.declared.push(event);
            self.current = Some(event);
            Ok(())
        })();
        if result.is_err() { self.failed = true; }
        result
    }
    pub(super) fn current(&self) -> Result<Event> {
        ensure!(!self.failed, "state stream previously failed");
        self.current.context("state observation has no independent parent obligation")
    }

    fn metadata(&self, path: &Path, value: &impl Serialize) -> Result<String> {
        super::metadata::write(path, self.context.budget.as_ref(), value)
    }

    pub(super) fn prepare(&self, layout: StateLayout) -> Result<Capture> {
        let event = self.current()?;
        let bytes = layout.logical_bytes()?;
        let expected = layout.expected(&self.context.agent, 0)?;
        let count = expected.len();
        {
            let mut budget = self.context.budget.borrow_mut();
            budget.state_readback(bytes)?;
            budget.reserve_store(count, bytes)?;
        }
        let root = self.root.join(format!("event-{:06}", event.ordinal));
        let plan = StatePlan { schema: "hf2q.c2.state-plan.v1", identity: &self.context.identity,
            scope_acquisition: &self.context.acquisition, registration_sha256: &self.context.registration_sha256,
            case: &self.context.case, arm: &self.context.arm, slot: 0, event, store_root: &root,
            layout: &layout, expected: &expected, derived_before_state_read: true };
        let plan_sha256 = self.metadata(
            &self.root.join(format!("event-{:06}-plan.json", event.ordinal)), &plan)?;
        let mut store = Store::new(&root, self.context.identity.clone(),
            format!("C2-ClassA-state-{}-{:?}-{}", self.context.arm, event.kind, event.ordinal),
            expected, self.context.payload_limit, count).map_err(anyhow::Error::msg)?;
        store.bind_acquisition(self.context.acquisition.clone().for_store_root(&root)?)
            .map_err(anyhow::Error::msg)?;
        Ok(Capture { store, layout, event, root, plan_sha256, logical_bytes: bytes })
    }

    pub(super) fn record(&mut self, mut capture: Capture, state: &State, ledger: &[u32]) -> Result<()> {
        super::state_layout::capture(&mut capture.store, &capture.layout, state, ledger,
            &self.context.agent, 0)?;
        let trace = capture.store.finish().map_err(anyhow::Error::msg)?;
        ensure!(trace.coverage.errors.is_empty()
            && trace.coverage.expected.len() == trace.coverage.captured.len(),
            "Incomplete: child state coverage differs from its independent layout");
        let trace_sha256 = super::metadata::hash_file(&capture.root.join("trace.json"),
            self.context.budget.as_ref())?;
        self.children.push(Child { event: capture.event,
            root: format!("event-{:06}", capture.event.ordinal),
            plan: format!("event-{:06}-plan.json", capture.event.ordinal),
            plan_sha256: capture.plan_sha256, trace_sha256, logical_bytes: capture.logical_bytes });
        ensure!(self.current == Some(capture.event), "state event changed during capture");
        self.current = None;
        Ok(())
    }

    pub(super) fn capture(&mut self, cache: &HybridKvCache, slot: SlotId, ledger: &[u32]) -> Result<()> {
        let result = (|| {
            ensure!(slot == SLOT, "transaction state belongs to another slot");
            // Only allocation descriptors and CPU metadata are inspected here.
            let layout = StateLayout::from_cache(cache, slot, ledger.len())?;
            let capture = self.prepare(layout)?;
            let state = super::live_ram::OwnedState::read(&self.context.budget, &capture.layout, cache, slot)?;
            self.record(capture, &state, ledger)
        })();
        if result.is_err() { self.failed = true; }
        result
    }

    pub(super) fn finish(self) -> Result<()> {
        ensure!(!self.failed && self.current.is_none() && !self.declared.is_empty()
            && self.declared.len() == self.children.len()
            && self.declared.iter().zip(&self.children).all(|(required, child)| *required == child.event),
            "Incomplete: required transaction/settlement state events were omitted or failed");
        let index_path = self.root.join("index.json");
        let sha256 = self.metadata(&index_path, &Index {
            schema: "hf2q.c2.transaction-index.v1", identity: &self.context.identity,
            scope_acquisition: &self.context.acquisition, registration_sha256: &self.context.registration_sha256,
            case: &self.context.case, arm: &self.context.arm, agent: &self.context.agent, slot: 0,
            declared_events: &self.declared, children: &self.children,
            all_declared_state_events_completed: true,
        })?;
        self.context.budget.borrow_mut().record_artifact(&self.context.case,
            &format!("{}-{}-transactions", self.context.case, self.context.arm),
            &self.context.root, &index_path, sha256)?;
        Ok(())
    }
}

#[cfg(test)]
pub(super) mod cpu_fixture {
    use super::*;
    use super::super::matrix_plan::{self, Budget, Grants, Manifest, Mode};

    // Real bounded child stores with explicitly synthetic, unqualified CPU data.
    pub(in super::super) fn writer(arm: &str) -> (tempfile::TempDir, Writer) {
        writer_for_case(arm, "rank0-eqfalse-peersfalse")
    }
    pub(in super::super) fn writer_for_case(arm: &str, case: &str) -> (tempfile::TempDir, Writer) {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().canonicalize().unwrap();
        let identity = Identity { family: Family::Qwen38, source_commit: "a".repeat(40),
            binary_sha256: "b".repeat(64), native_version: env!("HF2Q_MLX_NATIVE_VERSION").into(),
            native_checksum: "c".repeat(64), model_sha256: "d".repeat(64),
            tokenizer_sha256: "e".repeat(64), template_sha256: "f".repeat(64),
            config_sha256: "1".repeat(64), token_vectors_sha256: "2".repeat(64) };
        let cases = matrix_plan::catalog(Mode::Lifecycle, false);
        let manifest = Manifest { schema: "hf2q.c2.matrix-registration.v1".into(), identity: identity.clone(),
            mode: Mode::Lifecycle, catalog_sha256: hash(&serde_json::to_vec(&cases).unwrap()),
            cases: vec![case.into()], grants: Grants { events: 128, child_stores: 8, records: 128,
                payload_bytes: 4096, metadata_bytes: 1024 * 1024, retained_raw_bytes: 4096,
                state_readback_bytes: 4096, live_ram_bytes: 2 * 1024 * 1024 } };
        manifest.validate(&identity, Mode::Lifecycle, false).unwrap();
        let context = Rc::new(Context { acquisition: acquisition::synthetic(&root, &identity),
            root, case: case.into(), arm: arm.into(), agent: format!("{case}/slot0"),
            identity, registration_sha256: "3".repeat(64),
            budget: Rc::new(RefCell::new(Budget::new(manifest))), payload_limit: 4096 });
        let writer = Writer::with_context(context).unwrap();
        (parent, writer)
    }

}
