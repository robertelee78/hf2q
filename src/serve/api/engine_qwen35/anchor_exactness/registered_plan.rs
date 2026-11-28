//! Reload immutable pre-call registrations, not captured counts, for strict Q1.
use super::*;
use super::{forward_plan::{DecisionRequest, ForwardRequest, Origin},
    obligations::{Entry, Kind, Ticket}, registration::{Header, PlanBinding, PlanKind, StepWork}};
use std::collections::{BTreeMap, BTreeSet};
use super::request_kind::{Outcome, Purpose};
use super::native_inputs::InputBinding;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Execution {
    schema: String,
    context: Header,
    parents: Vec<acquisition::FileBinding>,
    plans: Vec<PlanBinding>,
    obligations: Vec<Entry>,
    all_required_execution_obligations_closed: bool,
    unforced: bool,
    whole_matrix_or_release_qualified: bool,
}
#[derive(Deserialize)]
#[serde(tag = "schema", deny_unknown_fields)]
enum Parent {
    #[serde(rename = "hf2q.c2.required-request.v1")]
    Request { context: Header, request: Ticket, required_settlement: Ticket, prediction_origin: usize, purpose: Purpose },
    #[serde(rename = "hf2q.c2.required-step.v1")]
    Step { context: Header, ticket: Ticket, work: StepWork, required_state: Ticket },
    #[serde(rename = "hf2q.c2.required-native-call.v1")]
    Native { context: Header, call: u64, parent: Option<Ticket>, other_slots: u8,
        required_projections: Vec<(Ticket, ForwardRequest, InputBinding)>, required_gpu_selectors: Vec<(Ticket, usize)> },
    #[serde(rename = "hf2q.c2.required-settlement.v1")]
    Settlement { context: Header, ticket: Ticket, outcome: Outcome },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ForwardPlan {
    schema: String, context: Header, ticket: Ticket, request: ForwardRequest,
    expected: Vec<ExpectedTensor>, original_witness: Option<acquisition::FileBinding>,
    input: InputBinding,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionPlan {
    schema: String, context: Header, ticket: Ticket, request: DecisionRequest,
    raw_expected: ExpectedTensor, canonical_expected: [ExpectedTensor; 3],
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GpuPlan { schema: String, context: Header, ticket: Ticket, position: usize }

pub(super) struct Bundle {
    pub execution: acquisition::FileBinding,
    pub context: Header,
    pub forwards: BTreeMap<u64, ForwardRequest>,
    pub inputs: BTreeMap<u64, InputBinding>,
    pub decisions: BTreeMap<usize, DecisionRequest>,
    pub gpu_positions: Vec<usize>,
    pub raw_expected: Vec<ExpectedTensor>,
    pub canonical_expected: Vec<ExpectedTensor>,
}

pub(super) fn same_acquisition(a: &acquisition::Acquisition, b: &acquisition::Acquisition) -> bool {
    a.packet_sha256 == b.packet_sha256 && a.nonce == b.nonce && a.output_root == b.output_root
        && a.actual_binary_sha256 == b.actual_binary_sha256 && a.composition_sha256 == b.composition_sha256
        && a.independent_review_sha256 == b.independent_review_sha256 && a.opened_model == b.opened_model
        && a.observed_model_opens == b.observed_model_opens && a.required_model_opens == b.required_model_opens
        && a.device_registry_id == b.device_registry_id && a.queue_identity == b.queue_identity
        && a.model_instance == b.model_instance && a.provenance == b.provenance
}

pub(super) fn resolve(root: &Path, binding: &acquisition::FileBinding) -> Result<acquisition::FileBinding> {
    ensure!(!binding.path.is_absolute() && binding.path.components()
        .all(|part| matches!(part, std::path::Component::Normal(_))), "plan binding is not scope-relative");
    let path = root.join(&binding.path);
    ensure!(path.canonicalize()? == path, "plan binding aliases or escapes its private scope");
    Ok(acquisition::FileBinding { path, sha256: binding.sha256.clone() })
}

pub(super) struct ExpectedScope<'a> {
    pub root: &'a Path,
    pub identity: &'a Identity,
    pub acquisition: &'a acquisition::Acquisition,
    pub registration_sha256: &'a str,
    pub case: &'a str,
    pub arm: &'a str,
}
fn header(value: &Header, scope: &ExpectedScope<'_>) -> Result<()> {
    ensure!(value.identity == *scope.identity && value.case == scope.case && value.arm == scope.arm && value.slot == 0
        && value.registration_sha256 == scope.registration_sha256
        && same_acquisition(&value.acquisition, scope.acquisition), "registered plan has another source/scope/case/arm/slot");
    Ok(())
}
fn entry(entries: &[Entry], ticket: Ticket, kind: Kind, parent: Option<Ticket>) -> Result<()> {
    let value = entries.get(ticket.index()).context("plan names an unknown required ticket")?;
    ensure!(value.kind == kind && value.parent == parent, "plan ticket has another required kind/parent");
    Ok(())
}
fn owned_read<T: serde::de::DeserializeOwned>(root: &Path, binding: &acquisition::FileBinding) -> Result<T> {
    Ok(serde_json::from_slice(&acquisition::checked_file(&resolve(root, binding)?, 16 * 1024 * 1024)?)?)
}

pub(super) fn load(fixture: &Fixture, case: &str, arm: &str) -> Result<Bundle> {
    let budget = fixture.matrix_budget()?;
    let borrowed = budget.borrow();
    let binding = borrowed.artifact(&format!("{case}-{arm}-execution"))?;
    let observed = fixture.observed(&fixture.packet.output)?;
    load_checked(&ExpectedScope { root: &fixture.packet.output, identity: &fixture.packet.identity,
        acquisition: &observed, registration_sha256: &fixture.packet.registration.as_ref()
            .context("matrix registration missing")?.sha256, case, arm }, binding)
}

pub(super) fn load_checked(scope: &ExpectedScope<'_>, binding: &acquisition::FileBinding) -> Result<Bundle> {
    let case = scope.case;
    let purpose = Purpose::for_case(scope.case, scope.arm)?;
    let execution: Execution = owned_read(scope.root, binding)?;
    ensure!(execution.schema == "hf2q.c2.execution-index.v1" && execution.all_required_execution_obligations_closed
        && execution.unforced && !execution.whole_matrix_or_release_qualified,
        "registered execution is incomplete or overclaims qualification");
    header(&execution.context, scope)?;
    super::obligations::validate_complete(&execution.obligations)?;
    let entries = &execution.obligations;
    let mut parent_tickets = BTreeSet::new();
    let mut calls = BTreeSet::new();
    let mut required_forwards = BTreeMap::new();
    let mut required_gpu = BTreeMap::new();
    let mut request = None;
    let mut origin = None;
    let mut outcome = None;
    // The constructor publishes the request before any steps or native work.
    for binding in &execution.parents {
        match owned_read::<Parent>(scope.root, binding)? {
            Parent::Request { context, request: ticket, required_settlement, prediction_origin, purpose: declared_purpose } => {
                header(&context, scope)?;
                ensure!(declared_purpose == purpose, "required request changed cancellation/generation purpose");
                ensure!(request.is_none() && ticket.index() < entries.len(), "required request duplicated");
                entry(entries, ticket, Kind::Request, entries[ticket.index()].parent)?;
                ensure!(entries[ticket.index()].parent.is_some_and(|parent| parent.index() == 0), "request lacks case parent");
                entry(entries, required_settlement, Kind::Settlement, Some(ticket))?;
                ensure!(parent_tickets.insert(ticket.index()), "required request repeated");
                request = Some(ticket); origin = Some(prediction_origin);
            }
            Parent::Step { context, ticket, work, required_state } => {
                header(&context, scope)?;
                entry(entries, ticket, Kind::Step, Some(request.context("step preceded required request")?))?;
                entry(entries, required_state, Kind::State, Some(ticket))?;
                ensure!(parent_tickets.insert(ticket.index()), "required step repeated");
                let children = entries.iter().filter(|value| value.parent == Some(ticket));
                match work {
                    StepWork::Native => ensure!(children.clone().any(|value| value.kind == Kind::NativeCall), "required native work omitted"),
                    StepWork::SavedCheckpoint => ensure!(children.clone().any(|value| value.kind == Kind::SavedCheckpointConsumption)
                        && !children.clone().any(|value| value.kind == Kind::NativeCall), "saved checkpoint replaced by native work"),
                    StepWork::StateOnly => {}
                }
            }
            Parent::Native { context, call, parent, other_slots, required_projections, required_gpu_selectors } => {
                header(&context, scope)?;
                ensure!(calls.insert(call), "native API contract repeated");
                if let Some(parent) = parent {
                    ensure!(parent.index() < entries.len(), "native parent ticket absent");
                    entry(entries, parent, Kind::NativeCall, entries[parent.index()].parent)?;
                    ensure!(parent_tickets.insert(parent.index()) && !required_projections.is_empty(), "native parent lacks required projections");
                    for (ticket, projection, input) in required_projections {
                        entry(entries, ticket, Kind::Projection, Some(parent))?;
                        input.geometry(projection.input_rows)?;
                        ensure!(projection.origin == Origin::Native
                            && required_forwards.insert(ticket.index(), (projection, input)).is_none(),
                            "required projection duplicated or changed origin");
                    }
                    for (ticket, position) in required_gpu_selectors {
                        entry(entries, ticket, Kind::Selection, Some(parent))?;
                        ensure!(required_gpu.insert(ticket.index(), position).is_none(), "required GPU selector repeated");
                    }
                } else {
                    ensure!(other_slots != 0 && required_projections.is_empty() && required_gpu_selectors.is_empty(),
                        "native contract omitted its actual selected input slots");
                }
            }
            Parent::Settlement { context, ticket, outcome: value } => {
                header(&context, scope)?;
                value.validate(purpose)?;
                entry(entries, ticket, Kind::Settlement, Some(request.context("terminal preceded required request")?))?;
                ensure!(outcome.replace(value).is_none() && parent_tickets.insert(ticket.index()),
                    "terminal request outcome was repeated");
            }
        }
    }
    ensure!(request.is_some() && outcome.is_some() && entries.iter().any(|entry| entry.kind == Kind::Step),
        "required request/work/terminal metadata missing");
    for (index, value) in entries.iter().enumerate() {
        if matches!(value.kind, Kind::Request | Kind::Step | Kind::NativeCall | Kind::Settlement) {
            ensure!(parent_tickets.contains(&index), "whole required API/step contract omitted");
        }
    }
    let mut result = Bundle { execution: binding.clone(), context: execution.context,
        forwards: BTreeMap::new(), decisions: BTreeMap::new(),
        inputs: BTreeMap::new(), gpu_positions: Vec::new(), raw_expected: Vec::new(), canonical_expected: Vec::new() };
    let mut planned = BTreeSet::new();
    let agent = format!("{case}/slot0");
    for binding in execution.plans {
        ensure!(planned.insert(binding.ticket.index()), "lower plan repeated a required ticket");
        match binding.kind {
            PlanKind::Forward => {
                let value: ForwardPlan = owned_read(scope.root, &binding.file)?;
                header(&value.context, scope)?;
                ensure!(value.schema == "hf2q.c2.forward-plan.v1" && value.ticket == binding.ticket,
                    "forward plan schema/ticket differs");
                let required = entries.get(value.ticket.index()).context("projection ticket absent")?;
                match value.request.origin {
                    Origin::Native => {
                        ensure!(required.kind == Kind::Projection && value.original_witness.is_none(),
                            "lower projection changed native origin");
                        let (request, input) = required_forwards.remove(&value.ticket.index())
                            .context("lower projection lacks required native input/geometry")?;
                        ensure!(request == value.request && input == value.input,
                            "lower projection differs from its required native input/geometry");
                    }
                    Origin::SavedCheckpoint { witness, .. } => {
                        ensure!(required.kind == Kind::SavedCheckpointConsumption && witness > 0,
                            "saved consumption lacks its required source event");
                        ensure!(matches!(&value.input, InputBinding::Host { .. }), "saved logical input is not pre-bound");
                        let original = value.original_witness.as_ref().context("original checkpoint receipt absent")?;
                        ensure!(original.path.canonicalize()?.starts_with(scope.root), "saved receipt escaped acquisition root");
                        acquisition::checked_file(original, 1024 * 1024)?;
                    }
                }
                ensure!(value.expected == value.request.expected(&agent)? && value.request.slot == 0
                    && result.forwards.insert(value.request.sequence, value.request).is_none(),
                    "forward extents differ from the immutable source geometry");
                value.input.geometry(value.request.input_rows)?;
                result.inputs.insert(value.request.sequence, value.input);
                result.raw_expected.extend(value.expected);
            }
            PlanKind::Decision => {
                let value: DecisionPlan = owned_read(scope.root, &binding.file)?;
                header(&value.context, scope)?;
                let required = entries.get(value.ticket.index()).context("semantic selector ticket absent")?;
                ensure!(value.schema == "hf2q.c2.semantic-selector-plan.v1" && value.ticket == binding.ticket
                    && required.kind == Kind::Selection && required.parent.is_some_and(|parent| entries[parent.index()].kind == Kind::Step),
                    "semantic selector has no required source parent");
                ensure!(value.request.physical_position == u64::try_from(origin.context("prediction origin absent")?)?
                    .checked_add(value.request.semantic_index).context("semantic position overflow")?
                    && value.raw_expected == value.request.raw_expected(&agent)
                    && value.canonical_expected == value.request.canonical_expected(&format!("{case}-single-slot-parity/slot0"))?,
                    "semantic selector changed its registered coordinates");
                ensure!(result.decisions.insert(usize::try_from(value.request.semantic_index)?, value.request).is_none(), "semantic selector repeated");
                result.raw_expected.push(value.raw_expected);
                result.canonical_expected.extend(value.canonical_expected);
            }
            PlanKind::GpuArgmax => {
                let value: GpuPlan = owned_read(scope.root, &binding.file)?;
                header(&value.context, scope)?;
                ensure!(value.schema == "hf2q.c2.gpu-selector-plan.v1" && value.ticket == binding.ticket
                    && required_gpu.remove(&value.ticket.index()) == Some(value.position),
                    "GPU selector differs from required native metadata");
                result.gpu_positions.push(value.position);
            }
        }
    }
    ensure!(required_forwards.is_empty() && required_gpu.is_empty(), "whole lower plan/observer pair omitted");
    for (index, value) in entries.iter().enumerate() {
        if matches!(value.kind, Kind::Projection | Kind::Selection | Kind::SavedCheckpointConsumption) {
            ensure!(planned.contains(&index), "required evidence leaf has no durable pre-call plan");
        }
    }
    ensure!(result.decisions.keys().copied().eq(0..result.decisions.len())
        && result.forwards.keys().copied().eq(0..u64::try_from(result.forwards.len())?),
        "required semantic/native work was reordered or omitted");
    ensure!(if purpose.raw_required() { !result.decisions.is_empty() && !result.forwards.is_empty() }
        else { result.decisions.is_empty() && result.forwards.is_empty() },
        "request raw work differs from its source-declared purpose");
    if let Some(Outcome::Completed { emitted_tokens }) = outcome {
        ensure!(result.decisions.len() >= emitted_tokens && result.decisions.len() <= emitted_tokens
            .checked_add(1).context("terminal semantic count overflow")?, "terminal omitted semantic decisions");
    }
    Ok(result)
}
