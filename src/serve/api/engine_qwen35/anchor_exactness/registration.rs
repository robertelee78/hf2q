//! Mandatory request/API obligations are distinct from lower planning and
//! observation. Plans are durable before projection/selection; values never
//! determine their coordinates. This module does not execute model work.
use super::*;
use anyhow::Context as _;
use super::{forward_plan::{DecisionRequest, ForwardRequest, Origin}, obligations::{Kind, Obligations, Ticket},
    transaction_trace::Context};
use super::request_kind::{Outcome, Purpose};
use super::native_inputs::InputBinding;
use std::{collections::BTreeMap, rc::Rc, sync::atomic::{AtomicU64, Ordering}};

static NEXT: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum StepWork { Native, StateOnly, SavedCheckpoint }

#[derive(Clone, Debug)]
pub(crate) struct NativeHead {
    pub slot: SlotId,
    pub start: usize,
    pub input_rows: usize,
    pub projected_rows: usize,
    pub vocab: usize,
    pub hidden_width: usize,
    pub gpu_argmax: bool,
    pub input: InputBinding,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(super) enum PlanKind { Forward, Decision, GpuArgmax }

#[derive(Serialize, Deserialize)]
pub(super) struct PlanBinding {
    pub kind: PlanKind,
    pub ticket: Ticket,
    pub file: acquisition::FileBinding,
}

#[derive(Serialize)]
pub(super) struct HeaderRef<'a> {
    pub identity: &'a Identity,
    pub acquisition: &'a acquisition::Acquisition,
    pub registration_sha256: &'a str,
    pub case: &'a str,
    pub arm: &'a str,
    pub slot: u32,
}

impl Context {
    pub(super) fn header(&self) -> HeaderRef<'_> {
        HeaderRef { identity: &self.identity, acquisition: &self.acquisition,
            registration_sha256: &self.registration_sha256, case: &self.case, arm: &self.arm, slot: 0 }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Header {
    pub identity: Identity,
    pub acquisition: acquisition::Acquisition,
    pub registration_sha256: String,
    pub case: String,
    pub arm: String,
    pub slot: u32,
}

#[derive(Serialize)]
struct ForwardPlan<'a> {
    schema: &'static str,
    context: HeaderRef<'a>,
    ticket: Ticket,
    request: ForwardRequest,
    expected: &'a [ExpectedTensor],
    input: &'a InputBinding,
    original_witness: Option<&'a acquisition::FileBinding>,
}

#[derive(Serialize)]
struct DecisionPlan<'a> {
    schema: &'static str,
    context: HeaderRef<'a>,
    ticket: Ticket,
    request: DecisionRequest,
    raw_expected: ExpectedTensor,
    canonical_expected: [ExpectedTensor; 3],
}

#[derive(Serialize)]
struct GpuPlan<'a> {
    schema: &'static str,
    context: HeaderRef<'a>,
    ticket: Ticket,
    position: usize,
}

struct Head {
    ticket: Ticket,
    request: ForwardRequest,
    input: InputBinding,
    observed: bool,
}
struct Gpu {
    ticket: Ticket,
    position: usize,
    observed: bool,
}
struct Frame {
    id: u64,
    declared: bool,
    parent: Option<Ticket>,
    heads: Vec<Head>,
    gpu: Vec<Gpu>,
}
struct Step {
    ticket: Ticket,
    state: Ticket,
    work: StepWork,
    saved: Option<Head>,
}

#[derive(Serialize)]
struct ExecutionIndex<'a> {
    schema: &'static str,
    context: HeaderRef<'a>,
    parents: &'a [acquisition::FileBinding],
    plans: &'a [PlanBinding],
    obligations: &'a [super::obligations::Entry],
    all_required_execution_obligations_closed: bool,
    unforced: bool,
    whole_matrix_or_release_qualified: bool,
}

pub(super) struct Session {
    pub id: u64,
    context: Rc<Context>,
    root: PathBuf,
    ledger: Obligations,
    request: Ticket,
    settlement: Ticket,
    origin: usize,
    purpose: Purpose,
    step: Option<Step>,
    frames: Vec<Frame>,
    next_native: u64,
    next_forward: u64,
    parents: Vec<acquisition::FileBinding>,
    plans: Vec<PlanBinding>,
    decisions: BTreeMap<usize, Ticket>,
    dimensions: Option<(usize, usize)>,
    finished: bool,
}

impl Session {
    pub(super) fn new(context: Rc<Context>, origin: usize) -> Result<Self> {
        let limit = context.budget.borrow().manifest.grants.events;
        let purpose = Purpose::for_case(&context.case, &context.arm)?;
        if purpose.raw_required() { context.budget.borrow_mut().reserve_child()?; }
        if context.case.ends_with("-peerstrue") && matches!(context.arm.as_str(), "actual" | "single") {
            context.budget.borrow_mut().reserve_child()?; // canonical parity arm
        }
        let mut ledger = Obligations::new(limit)?;
        let request = ledger.require(ledger.case(), Kind::Request)?;
        let settlement = ledger.require(request, Kind::Settlement)?;
        ledger.start(ledger.case())?;
        ledger.start(request)?;
        let root = context.root.join(format!("{}-{}-plans", context.case, context.arm));
        fs::create_dir(&root)?;
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))?;
        let mut session = Self { id: NEXT.fetch_add(1, Ordering::Relaxed), context, root, ledger,
            request, settlement, origin, step: None, frames: Vec::new(), next_native: 0, next_forward: 0,
            parents: Vec::new(), plans: Vec::new(), decisions: BTreeMap::new(), dimensions: None, purpose, finished: false };
        let binding = session.publish("request.json", &serde_json::json!({
            "schema":"hf2q.c2.required-request.v1","context":session.context.header(),
            "request":request,"required_settlement":settlement,"prediction_origin":origin,"purpose":purpose,
        }))?;
        session.parents.push(binding);
        Ok(session)
    }
    fn publish(&self, name: &str, value: &impl Serialize) -> Result<acquisition::FileBinding> {
        let path = self.root.join(name);
        let sha256 = super::metadata::write(&path, self.context.budget.as_ref(), value)?;
        Ok(acquisition::FileBinding { path: path.strip_prefix(&self.context.root)?.into(), sha256 })
    }
    pub(super) fn invalidate(&mut self) {
        self.ledger.invalidate();
        self.context.budget.borrow_mut().invalidate();
    }
    fn checked<T>(&mut self, operation: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        ensure!(!self.ledger.failed() && !self.finished, "C2 registration previously failed or finished");
        let result = operation(self);
        if result.is_err() { self.invalidate(); }
        result
    }
    pub(super) fn begin_step(&mut self, work: StepWork) -> Result<Ticket> {
        self.checked(|this| {
            ensure!(this.step.is_none() && this.frames.is_empty(), "overlapping C2 request steps");
            let ticket = this.ledger.require(this.request, Kind::Step)?;
            let state = this.ledger.require(ticket, Kind::State)?;
            match work {
                StepWork::Native => this.ledger.require_children(ticket, &[Kind::NativeCall])?,
                StepWork::SavedCheckpoint => this.ledger.require_children(ticket, &[Kind::SavedCheckpointConsumption])?,
                StepWork::StateOnly => {}
            }
            this.ledger.start(ticket)?;
            let binding = this.publish(&format!("step-{:06}.json", ticket.index()), &serde_json::json!({
                "schema":"hf2q.c2.required-step.v1","context":this.context.header(),
                "ticket":ticket,"work":work,"required_state":state,
            }))?;
            this.parents.push(binding);
            this.step = Some(Step { ticket, state, work, saved: None });
            Ok(ticket)
        })
    }
    pub(super) fn state_before_capture(&mut self) -> Result<Ticket> {
        self.checked(|this| {
            let ticket = this.step.as_ref().context("state lacks required request step")?.state;
            this.ledger.plan(ticket)?;
            Ok(ticket)
        })
    }
    pub(super) fn observed_state(&mut self, ticket: Ticket) -> Result<()> {
        self.checked(|this| {
            ensure!(this.step.as_ref().is_some_and(|step| step.state == ticket), "state belongs to another step");
            this.ledger.complete(ticket)
        })
    }
    pub(super) fn finish_step(&mut self, ticket: Ticket) -> Result<()> {
        self.checked(|this| {
            ensure!(this.frames.is_empty(), "native API guard remains open at step closure");
            let step = this.step.take().context("required step absent")?;
            ensure!(step.ticket == ticket, "request step identity changed");
            this.ledger.complete(ticket)
        })
    }
    pub(super) fn open_native(&mut self) -> Result<u64> {
        self.checked(|this| {
            ensure!(this.step.is_some(), "native API call has no independent request step");
            let id = this.next_native;
            this.next_native = id.checked_add(1).context("native call identity overflow")?;
            this.frames.push(Frame { id, declared: false, parent: None, heads: Vec::new(), gpu: Vec::new() });
            Ok(id)
        })
    }
    pub(super) fn native_contract(&mut self, heads: impl IntoIterator<Item = NativeHead>) -> Result<()> {
        self.checked(|this| {
            let mut frame = this.frames.pop().context("native metadata has no API parent guard")?;
            ensure!(!frame.declared, "native API contract declared twice");
            let step = this.step.as_ref().context("native contract lacks request step")?;
            let mut other_slots = 0_u8;
            for head in heads {
                ensure!(head.slot.0 < SLOTS, "native requested slot outside C2 allocation");
                if head.slot != SLOT { other_slots |= 1 << head.slot.0; continue; }
                ensure!(this.purpose.raw_required(), "a no-advance/reference request invented target execution");
                ensure!(step.work != StepWork::SavedCheckpoint, "equality consumption invented a target forward");
                head.input.geometry(head.input_rows)?;
                let parent = if let Some(parent) = frame.parent { parent } else {
                    let parent = this.ledger.require(step.ticket, Kind::NativeCall)?;
                    frame.parent = Some(parent);
                    parent
                };
                let ticket = this.ledger.require(parent, Kind::Projection)?;
                let request = ForwardRequest { sequence: this.next_forward, slot: head.slot.0,
                    start: head.start, input_rows: head.input_rows, projected_rows: head.projected_rows,
                    vocab: head.vocab, hidden_width: head.hidden_width, origin: Origin::Native };
                this.next_forward = this.next_forward.checked_add(1).context("forward identity overflow")?;
                let bytes = request.logical_bytes()?;
                let records = head.projected_rows.checked_mul(2).context("projection record count overflow")?;
                {
                    let mut budget = this.context.budget.borrow_mut();
                    budget.reserve_records(records, bytes)?;
                    let retained = u64::try_from(head.input_rows)?.checked_mul(4)
                        .and_then(|n| bytes.checked_add(n)).and_then(|n| n.checked_add(256))
                        .context("retained observation grant overflow")?;
                    budget.retain_raw(retained)?;
                }
                if let Some(dimensions) = this.dimensions {
                    ensure!(dimensions == (head.vocab, head.hidden_width), "native model projection geometry changed");
                } else { this.dimensions = Some((head.vocab, head.hidden_width)); }
                if head.gpu_argmax {
                    let first = request.first_row()?;
                    for row in 0..head.projected_rows {
                        frame.gpu.push(Gpu { ticket: this.ledger.require(parent, Kind::Selection)?,
                            position: first.checked_add(row).context("GPU selector position overflow")?, observed: false });
                    }
                }
                frame.heads.push(Head { ticket, request, input: head.input, observed: false });
            }
            ensure!(frame.parent.is_some() || other_slots != 0, "native call omitted its selected input contract");
            if let Some(parent) = frame.parent { this.ledger.start(parent)?; }
            frame.declared = true;
            let required: Vec<_> = frame.heads.iter().map(|head| (head.ticket, head.request, &head.input)).collect();
            let gpu: Vec<_> = frame.gpu.iter().map(|row| (row.ticket, row.position)).collect();
            let binding = this.publish(&format!("native-{:06}.json", frame.id), &serde_json::json!({
                "schema":"hf2q.c2.required-native-call.v1","context":this.context.header(),
                "call":frame.id,"parent":frame.parent,"other_slots":other_slots,
                "required_projections":required,"required_gpu_selectors":gpu,
            }))?;
            this.parents.push(binding);
            this.frames.push(frame);
            Ok(())
        })
    }
    pub(super) fn plan_native(&mut self) -> Result<()> {
        self.checked(|this| {
            let frame = this.frames.pop().context("lower native planner lacks API guard")?;
            ensure!(frame.declared, "lower planner ran before the independent API contract");
            for head in &frame.heads {
                this.ledger.plan(head.ticket)?;
                let expected = head.request.expected(&this.context.agent)?;
                let file = this.publish(&format!("forward-{:06}.json", head.request.sequence), &ForwardPlan {
                    schema: "hf2q.c2.forward-plan.v1", context: this.context.header(), ticket: head.ticket,
                    request: head.request, expected: &expected, input: &head.input, original_witness: None,
                })?;
                this.plans.push(PlanBinding { kind: PlanKind::Forward, ticket: head.ticket, file });
            }
            for row in &frame.gpu {
                this.ledger.plan(row.ticket)?;
                let file = this.publish(&format!("gpu-{:06}.json", row.ticket.index()), &GpuPlan {
                    schema: "hf2q.c2.gpu-selector-plan.v1", context: this.context.header(),
                    ticket: row.ticket, position: row.position,
                })?;
                this.plans.push(PlanBinding { kind: PlanKind::GpuArgmax, ticket: row.ticket, file });
            }
            this.frames.push(frame);
            Ok(())
        })
    }
    pub(super) fn close_native(&mut self, id: u64) -> Result<()> {
        self.checked(|this| {
            let frame = this.frames.pop().context("native API guard absent")?;
            ensure!(frame.id == id && frame.declared, "native API omitted its mandatory requested contract");
            ensure!(frame.heads.iter().all(|head| head.observed) && frame.gpu.iter().all(|row| row.observed),
                "Incomplete: native API omitted a required projection/selector observation");
            if let Some(parent) = frame.parent { this.ledger.complete(parent)?; }
            Ok(())
        })
    }
    pub(super) fn forward(&mut self, slot: SlotId, start: usize, input: &[u32],
        vocab: usize, hidden_width: usize, logits: usize, hidden: usize) -> Result<(u64, String)> {
        self.checked(|this| {
            let input_rows = input.len();
            let head = if let Some(frame) = this.frames.last_mut() {
                let mut matches = frame.heads.iter_mut().filter(|head| head.request.slot == slot.0
                    && head.request.start == start && head.request.input_rows == input_rows);
                let head = matches.next().context("actual forward has no required API projection ticket")?;
                ensure!(matches.next().is_none(), "ambiguous native projection source identity");
                head
            } else { this.step.as_mut().and_then(|step| step.saved.as_mut())
                .context("actual forward has neither native nor saved-checkpoint obligation")? };
            let request = head.request;
            ensure!(!head.observed && request.slot == slot.0 && request.start == start
                && request.input_rows == input_rows && request.vocab == vocab && request.hidden_width == hidden_width
                && request.projected_rows.checked_mul(vocab) == Some(logits)
                && request.projected_rows.checked_mul(hidden_width) == Some(hidden),
                "Incomplete: returned full rows differ from pre-call requested geometry");
            let input_sha256 = head.input.check(input)?;
            this.ledger.complete(head.ticket)?;
            head.observed = true;
            Ok((request.sequence, input_sha256))
        })
    }
    pub(super) fn gpu_argmax(&mut self, position: usize) -> Result<()> {
        self.checked(|this| {
            let frame = this.frames.last_mut().context("actual GPU selector lacks native parent")?;
            let mut matches = frame.gpu.iter_mut().filter(|row| row.position == position);
            let row = matches.next().context("GPU selector position was not requested")?;
            ensure!(matches.next().is_none() && !row.observed, "GPU selector was ambiguous or repeated");
            this.ledger.complete(row.ticket)?;
            row.observed = true;
            Ok(())
        })
    }
    pub(super) fn saved_checkpoint(&mut self, head: NativeHead, witness: u64,
        hidden: super::forward_plan::SavedHidden, binding: &acquisition::FileBinding) -> Result<()> {
        self.checked(|this| {
            ensure!(this.frames.is_empty(), "saved checkpoint was mislabeled as a native call");
            let step = this.step.as_ref().context("saved checkpoint lacks request step")?;
            ensure!(step.work == StepWork::SavedCheckpoint && step.saved.is_none()
                && head.slot == SLOT && !head.gpu_argmax,
                "saved checkpoint consumption does not match its declared phase");
            head.input.geometry(head.input_rows)?;
            ensure!(matches!(&head.input, InputBinding::Host { .. }), "saved checkpoint logical input must be pre-bound");
            ensure!(binding.path.starts_with(&this.context.root),
                "original checkpoint witness escaped its acquisition scope");
            acquisition::checked_file(binding, 1024 * 1024)?;
            let ticket = this.ledger.require(step.ticket, Kind::SavedCheckpointConsumption)?;
            let request = ForwardRequest { sequence: this.next_forward, slot: SLOT.0, start: head.start,
                input_rows: head.input_rows, projected_rows: head.projected_rows, vocab: head.vocab,
                hidden_width: head.hidden_width, origin: Origin::SavedCheckpoint { witness, hidden } };
            this.next_forward = this.next_forward.checked_add(1).context("saved event identity overflow")?;
            let bytes = request.logical_bytes()?;
            let expected = request.expected(&this.context.agent)?;
            {
                let mut budget = this.context.budget.borrow_mut();
                budget.reserve_records(expected.len(), bytes)?;
                budget.retain_raw(bytes.checked_add(260).context("saved observation grant overflow")?)?;
            }
            if let Some(dimensions) = this.dimensions {
                ensure!(dimensions == (head.vocab, head.hidden_width), "saved model geometry changed");
            } else { this.dimensions = Some((head.vocab, head.hidden_width)); }
            this.ledger.plan(ticket)?;
            let file = this.publish(&format!("forward-{:06}.json", request.sequence), &ForwardPlan {
                schema: "hf2q.c2.forward-plan.v1", context: this.context.header(), ticket, request,
                expected: &expected, input: &head.input, original_witness: Some(binding),
            })?;
            this.plans.push(PlanBinding { kind: PlanKind::Forward, ticket, file });
            this.step.as_mut().expect("checked required step").saved =
                Some(Head { ticket, request, input: head.input, observed: false });
            Ok(())
        })
    }
    pub(super) fn require_decision(&mut self, index: usize, vocab: usize, hidden_width: usize) -> Result<()> {
        self.checked(|this| {
            ensure!(!this.decisions.contains_key(&index), "semantic decision required twice");
            let parent = this.step.as_ref().context("semantic selector lacks request step")?.ticket;
            ensure!(vocab > 0 && hidden_width > 0, "semantic selector model geometry absent");
            if let Some(dimensions) = this.dimensions {
                ensure!(dimensions == (vocab, hidden_width), "semantic selector model geometry changed");
            } else { this.dimensions = Some((vocab, hidden_width)); }
            let ticket = this.ledger.require(parent, Kind::Selection)?;
            let request = DecisionRequest { semantic_index: u64::try_from(index)?,
                physical_position: u64::try_from(this.origin.checked_add(index).context("semantic position overflow")?)?,
                vocab, hidden_width };
            let canonical = this.context.case.ends_with("-peerstrue")
                && matches!(this.context.arm.as_str(), "actual" | "single");
            {
                let mut budget = this.context.budget.borrow_mut();
                budget.reserve_records(1, 4)?;
                if canonical {
                    let bytes = vocab.checked_add(hidden_width).and_then(|n| n.checked_mul(4))
                        .and_then(|n| n.checked_add(4)).context("canonical payload grant overflow")?;
                    budget.reserve_records(3, u64::try_from(bytes)?)?;
                }
            }
            this.ledger.plan(ticket)?;
            let file = this.publish(&format!("decision-{index:06}.json"), &DecisionPlan {
                schema: "hf2q.c2.semantic-selector-plan.v1", context: this.context.header(), ticket,
                request, raw_expected: request.raw_expected(&this.context.agent),
                canonical_expected: request.canonical_expected(&format!("{}-single-slot-parity/slot0", this.context.case))?,
            })?;
            this.plans.push(PlanBinding { kind: PlanKind::Decision, ticket, file });
            this.decisions.insert(index, ticket);
            Ok(())
        })
    }
    pub(super) fn decision(&mut self, index: usize) -> Result<()> {
        self.checked(|this| this.ledger.complete(*this.decisions.get(&index)
            .context("semantic observation lacks its independent required selector")?))
    }
    pub(super) fn settlement_before(&mut self, outcome: Outcome) -> Result<Ticket> {
        self.checked(|this| {
            ensure!(this.step.is_none() && this.frames.is_empty(), "request settled with open execution guards");
            outcome.validate(this.purpose)?;
            if let Outcome::Completed { emitted_tokens } = outcome {
                ensure!(!this.decisions.is_empty() && this.decisions.len() >= emitted_tokens
                    && this.decisions.len() <= emitted_tokens.checked_add(1).context("terminal decision count overflow")?
                    && this.decisions.keys().copied().eq(0..this.decisions.len()),
                    "completed generation omitted emitted/intermediate/terminal semantic work");
            }
            this.ledger.plan(this.settlement)?;
            let binding = this.publish("settlement.json", &serde_json::json!({
                "schema":"hf2q.c2.required-settlement.v1","context":this.context.header(),
                "ticket":this.settlement,"outcome":outcome,
            }))?;
            this.parents.push(binding);
            Ok(this.settlement)
        })
    }
    pub(super) fn settlement_after(&mut self, ticket: Ticket) -> Result<()> {
        self.checked(|this| {
            ensure!(ticket == this.settlement, "request settlement identity changed");
            this.ledger.complete(ticket)
        })
    }
    pub(super) fn finish(mut self) -> Result<()> {
        ensure!(self.step.is_none() && self.frames.is_empty(), "C2 execution guards remain open");
        ensure!(self.ledger.entries().iter().any(|entry| entry.kind == Kind::Step),
            "request omitted its complete selected-work/state phase");
        ensure!(if self.purpose.raw_required() { self.next_forward > 0 && !self.decisions.is_empty() }
            else { self.next_forward == 0 && self.decisions.is_empty() },
            "request raw work differs from its declared generation/cancellation/reference purpose");
        self.ledger.complete(self.request)?;
        self.ledger.complete(self.ledger.case())?;
        self.ledger.finish()?;
        let path = self.root.join("execution.json");
        let sha256 = super::metadata::write(&path, self.context.budget.as_ref(), &ExecutionIndex {
            schema: "hf2q.c2.execution-index.v1", context: self.context.header(),
            parents: &self.parents, plans: &self.plans, obligations: self.ledger.entries(),
            all_required_execution_obligations_closed: true, unforced: true,
            whole_matrix_or_release_qualified: false,
        })?;
        self.context.budget.borrow_mut().record_artifact(&self.context.case,
            &format!("{}-{}-execution", self.context.case, self.context.arm),
            &self.context.root, &path, sha256)?;
        self.finished = true;
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.finished { self.context.budget.borrow_mut().invalidate(); }
    }
}

impl Session {
    pub(super) fn state_ticket(&self) -> Result<Ticket> {
        self.step.as_ref().map(|step| step.state).context("required state step is absent")
    }
}
