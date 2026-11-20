//! Same-route twins retain the actually executed prefix chunks and K3 steps,
//! but never replay overwritten history or restore an image in the reference.
use super::*;
use super::trace::{Observations, State};
use std::{collections::BTreeMap, sync::{Arc, atomic::Ordering}};
#[path = "matrix_dispatch.rs"]
mod matrix_dispatch;
#[path = "canonical_pair.rs"]
mod canonical_pair;
#[path = "prefix_script.rs"]
mod prefix_script;
pub(crate) fn completed(slot: SlotId, tokens: &[u32]) { matrix_dispatch::completed(slot, tokens); }

struct Checkpoint {
    tokens: Vec<u32>,
    logits: Vec<f32>,
    // Diagnostic provenance only, never installed into model/cache state.
    hidden: Arc<Vec<f32>>,
}
pub(super) struct Lane {
    pub(super) slot: SlotId,
    pub(super) cache: matrix_dispatch::Cache,
    pub(super) store: Qwen35AnchorStore,
    pub(super) retained: Vec<u32>,
    pub(super) spec: Option<Qwen35SpecPrefixBoundary>,
    driver: matrix_dispatch::Driver,
    checkpoint: Option<Checkpoint>,
    hidden_witnesses: BTreeMap<u64, Arc<Vec<f32>>>,
    prefix_witnesses: BTreeMap<u64, prefix_script::Turn>,
    admission: Option<crate::serve::api::qwen35_speculation::QwenSpeculationDecision>,
}
impl Lane {
    pub(super) fn new(fixture: &Fixture) -> Result<Self> { Ok(Self {
        cache: matrix_dispatch::Cache::new(fixture.cache()?),
        store: Qwen35AnchorStore::new(4)?, retained: Vec::new(), spec: None,
        slot: SLOT, checkpoint: None, hidden_witnesses: BTreeMap::new(),
        prefix_witnesses: BTreeMap::new(),
        driver: matrix_dispatch::Driver::new()?, admission: None }) }
    pub(super) fn restore(&mut self, id: u64, supervisor: &EngineSupervisor) -> Result<()> {
        let anchor = self.store.anchor(id).cloned();
        let restored = restore_committed_anchor(&mut self.cache, self.slot, &mut self.store, id, supervisor);
        self.prune_witnesses();
        restored?;
        let anchor = anchor.context("successful restore lost its committed anchor")?;
        let hidden = self.hidden_witnesses.get(&id).context("original checkpoint normalized-head witness missing")?.clone();
        self.checkpoint = Some(Checkpoint { tokens: anchor.prompt_tokens.clone(),
            logits: anchor.prefill_logits, hidden });
        self.retained = anchor.prompt_tokens;
        self.spec = anchor.spec;
        self.driver.set_prefix(self.slot, &self.retained, self.spec.as_ref());
        Ok(())
    }
    pub(super) fn admit(&mut self, prompt: usize, limit: usize) -> Result<u64> {
        self.driver.admit(self.slot, prompt, limit)
    }
    pub(super) fn cancel_admission(&mut self) -> Result<()> {
        self.driver.cancel(self.slot);
        self.driver.set_prefix(self.slot, &self.retained, self.spec.as_ref());
        self.prune_witnesses();
        self.checkpoint = self.store.deepest_prefix_of(&self.retained, None, None)
            .filter(|anchor| anchor.prompt_tokens == self.retained)
            .map(|anchor| -> Result<Checkpoint> {
                let hidden = self.hidden_witnesses.get(&anchor.id)
                    .context("recovered checkpoint lost its original normalized-head witness")?.clone();
                Ok(Checkpoint { tokens: anchor.prompt_tokens.clone(),
                    logits: anchor.prefill_logits.clone(), hidden })
            }).transpose()?;
        Ok(())
    }
    pub(super) fn prune_witnesses(&mut self) {
        let store = &self.store;
        self.hidden_witnesses.retain(|id, _| store.anchor(*id).is_some()
            || store.pending().is_some_and(|pending| pending.id == *id));
        self.prefix_witnesses.retain(|id, _| store.anchor(*id).is_some()
            || store.pending().is_some_and(|pending| pending.id == *id));
    }
    pub(super) fn cold(&mut self) -> Result<()> {
        self.store.reset();
        self.driver.set_prefix(self.slot, &[], None);
        self.cache.reset_for_slot(self.slot)?;
        self.retained.clear(); self.spec = None; self.checkpoint = None;
        self.hidden_witnesses.clear(); self.prefix_witnesses.clear(); Ok(())
    }
}
pub(super) struct Served {
    pub(super) state: Qwen35DecodeState,
    pub(super) anchor: u64,
    pub(super) request: Option<u64>,
}
pub(super) fn prefill(fixture: &mut Fixture, lane: &mut Lane, prompt: &[u32], cached: usize,
    publish: bool) -> Result<Served> {
    prefill_limit(fixture, lane, prompt, cached, publish, DECODE_LIMIT)
}
pub(super) fn prefill_limit(fixture: &mut Fixture, lane: &mut Lane, prompt: &[u32], cached: usize,
    publish: bool, limit: usize) -> Result<Served> {
    if cached == 0 { lane.driver.set_prefix(lane.slot, &[], None); }
    let request = lane.admit(prompt.len(), limit)?;
    prefill_transaction(fixture, lane, prompt, cached, publish, limit, Some(request))
}
fn prefill_transaction(fixture: &mut Fixture, lane: &mut Lane, prompt: &[u32], cached: usize,
    publish: bool, limit: usize, request: Option<u64>) -> Result<Served> {
    let _observation_scope = if trace::enabled() { None } else { Some(trace::Scope::begin(None, 0)?) };
    let supervisor = fixture.supervisor.clone();
    let qwen = fixture.qwen();
    let equality = if cached == prompt.len() {
        let checkpoint = lane.checkpoint.as_ref().context("equality checkpoint logits absent")?;
        ensure!(checkpoint.tokens == prompt, "equality seed belongs to a different logical prefix");
        Some(checkpoint.logits.clone())
    } else { None };
    let params = SamplingParams { max_tokens: limit, ..SamplingParams::default() };
    let retained = crate::serve::api::engine::Qwen35RetainedPrefix {
        tokens: lane.retained.clone(), spec: lane.spec.clone() };
    let preference = crate::serve::api::engine::Qwen35TestSlotPreference {
        cached_tokens: cached, anchor_id: None, restore_prompt_anchor: false,
        use_prefill_logits: cached == prompt.len() };
    let (decision, cached_spec) = crate::serve::api::engine::test_qwen35_admission_speculation(
        qwen, &lane.cache, lane.slot, &preference, &lane.store, &retained, prompt.len(), &params, None)?;
    lane.admission = Some(decision);
    let mut pending = Qwen35PrefillState::begin(prompt.to_vec(), params, None,
        &mut lane.cache, lane.slot, cached, equality, cached_spec, None,
        qwen.hidden_size, qwen.model.kv_graft.as_ref())?;
    let (state, logits) = loop {
        let outcome = pending.advance(qwen, None, &mut lane.cache, CHUNK, &supervisor, None)?;
        trace::transaction(&lane.cache, lane.slot)?;
        match outcome {
            Qwen35PrefillAdvance::Pending { state, .. } => pending = state,
            Qwen35PrefillAdvance::Ready { state, prefill_logits, .. } => break (state, prefill_logits),
        }
    };
    let physical = lane.cache.sequence_len_for_slot(lane.slot)? as usize;
    let hidden = if cached == prompt.len() {
        let witness = lane.checkpoint.as_ref().context("equality hidden witness absent")?.hidden.clone();
        let consumed = state.mtp.as_ref().map(|mtp| mtp.verifier_hidden.as_logical_slice::<f32>()).transpose()?;
        let (source, actual_hidden) = match consumed {
            Some(hidden) => ("production-saved-logits-equality/consumed-spec-hidden", hidden),
            None => ("production-saved-logits-equality/original-checkpoint-normalized-head-witness", witness.as_slice()),
        };
        trace::forward(lane.slot, physical - 1, &prompt[prompt.len() - 1..],
            source, &logits, qwen.vocab_size, Some(actual_hidden), qwen.hidden_size)?;
        witness
    } else { Arc::new(trace::checkpoint_hidden(lane.slot, physical - 1)?) };
    lane.checkpoint = Some(Checkpoint { tokens: prompt.to_vec(), logits: logits.clone(), hidden: hidden.clone() });
    let mut anchor = 0;
    if publish {
        let request = request.context("checkpoint capture requires an actual admitted request")?;
        let physical = lane.cache.sequence_len_for_slot(lane.slot)? as usize;
        let ticket = lane.store.begin_capture()?;
        let image = match lane.cache.snapshot_slot_anchor(lane.slot, physical) {
            Ok(image) => image,
            Err(error) => { lane.store.abort_transaction(ticket)?; return Err(error); }
        };
        let spec = state.spec_prefix_candidate(prompt.len()).map(|spec| {
            qwen.model.with_gpu_cache_mut(|device, _| owned_spec_boundary(
                device, spec.token_count, &spec.pending_target_hidden))
        }).transpose()?;
        lane.store.finish_capture(ticket)?;
        anchor = lane.store.stage(request, Qwen35Anchor { id: 0, request,
            prompt_tokens: prompt.to_vec(), kv: image, prefill_logits: logits,
            vision_fingerprint: None, spec })?;
        lane.hidden_witnesses.insert(anchor, hidden);
        lane.prune_witnesses();
    }
    Ok(Served { state, anchor, request })
}
fn ready(lane: &Lane, state: Qwen35DecodeState) -> Result<matrix_dispatch::Ready> {
    let seed_logits = if state.finished_at_seed() {
        Some(lane.checkpoint.as_ref().context("seed completion lost actual prefill logits")?.logits.clone())
    } else { None };
    Ok(matrix_dispatch::Ready { state, seed_logits })
}
pub(super) fn tick(fixture: &mut Fixture, lane: &mut Lane, state: Qwen35DecodeState) -> Result<()> {
    let ready = ready(lane, state)?;
    lane.driver.run(fixture, &mut lane.cache, vec![ready], &mut lane.store, Some(1))?;
    Ok(())
}
pub(super) fn finish_success(fixture: &mut Fixture, lane: &mut Lane, served: Served)
    -> Result<Vec<u32>> {
    let ready = ready(lane, served.state)?;
    let outcome = lane.driver.run(fixture, &mut lane.cache, vec![ready],
        &mut lane.store, None)?;
    let published = outcome.retained.context("completed target omitted its actual retained-prefix publication")?;
    lane.retained = published.tokens;
    lane.spec = published.spec;
    let cursor = lane.cache.sequence_len_for_slot(lane.slot)? as usize;
    let graft = fixture.qwen().model.kv_graft.as_ref().map_or(0, |g| g.bank.n_slots as usize);
    ensure!(lane.retained.len() + graft == cursor, "production retained ledger/physical cursor join");
    if served.anchor != 0 {
        ensure!(lane.store.anchor(served.anchor).is_some() && lane.store.pending().is_none(),
            "production success did not commit its pending capture");
    }
    lane.prune_witnesses();
    outcome.tokens.into_iter().find_map(|(slot, tokens)| (slot == lane.slot).then_some(tokens))
        .context("successful serving settlement omitted its target token witness")
}
// Grow on the live retained prefix. No rewind is used in checkpoint creation:
// every prior decode/K3 transaction is an ancestor of every later checkpoint.
pub(super) fn build(fixture: &mut Fixture, lane: &mut Lane, through: usize) -> Result<Vec<u64>> {
    let tokens = fixture.packet.workload.lanes[lane.slot.0 as usize].clone();
    let mut ids = Vec::new();
    for index in 0..=through {
        let cached = lane.retained.len();
        let mut prompt = lane.retained.clone();
        prompt.extend_from_slice(&tokens[index * 32..(index + 1) * 32]);
        let scope = trace::Scope::begin(None, 0)?;
        let served = prefill(fixture, lane, &prompt, cached, true)?;
        let anchor = served.anchor;
        ids.push(anchor);
        let prefill = prefix_script::Signature::read(&scope)?;
        let tokens = finish_success(fixture, lane, served)?;
        let complete = prefix_script::Signature::read(&scope)?;
        lane.prefix_witnesses.insert(anchor, prefix_script::Turn {
            prompt, prefill, complete, tokens, retained: lane.retained.clone() });
    }
    Ok(ids)
}
fn replay_script(fixture: &mut Fixture, lane: &mut Lane, script: &[prefix_script::Turn],
    through: usize, checkpoint_only: bool) -> Result<Vec<u64>> {
    let mut ids = Vec::new();
    for (index, turn) in script[..=through].iter().enumerate() {
        let prompt = &turn.prompt;
        let expected_retained = if index == 0 { &[][..] } else { script[index - 1].retained.as_slice() };
        ensure!(lane.retained == expected_retained && prompt.starts_with(expected_retained),
            "independent replay departed from the complete frozen retained ledger");
        let scope = trace::Scope::begin(None, 0)?;
        let cached = lane.retained.len();
        let served = if checkpoint_only && index == through {
            // This is a prefix-transaction replay, not an admitted generation:
            // its decode tail is deliberately absent. The eventual suffix gets
            // its own real admission with the full suffix request dimensions.
            prefill_transaction(fixture, lane, prompt, cached, false, DECODE_LIMIT, None)?
        } else { prefill(fixture, lane, prompt, cached, !checkpoint_only)? };
        turn.check_prefill(&scope)?;
        if checkpoint_only && index == through {
            // Retain the returned logits/hidden boundary, not an anchor image.
            // The sampled seed has not been fed; no decode transaction executes.
            lane.retained = prompt.clone();
            lane.spec = served.state.spec_prefix_candidate(prompt.len());
            break;
        }
        ids.push(served.anchor);
        let tokens = finish_success(fixture, lane, served)?;
        turn.check_complete(&scope, &tokens, &lane.retained)?;
        if checkpoint_only {
            // A genuine seed-completion takes the real prefill settlement,
            // which may publish its own prior checkpoint. The reference never
            // selects it; release it without changing any cache/ledger bytes.
            lane.store.reset();
            lane.prune_witnesses();
        }
    }
    Ok(ids)
}
pub(super) fn independent_checkpoint(fixture: &mut Fixture, script: &[prefix_script::Turn], selected: usize)
    -> Result<Lane> {
    let mut lane = Lane::new(fixture)?;
    replay_script(fixture, &mut lane, script, selected, true)?;
    lane.driver.set_prefix(lane.slot, &lane.retained, lane.spec.as_ref());
    ensure!(lane.store.is_empty(), "reference unexpectedly captured an image");
    Ok(lane)
}
pub(super) fn script(lane: &mut Lane, ids: &[u64]) -> Result<Vec<prefix_script::Turn>> {
    ids.iter().map(|&id| {
        let anchor = lane.store.anchor(id).context("script checkpoint disappeared")?;
        let turn = lane.prefix_witnesses.remove(&id).context("independent complete prefix witness disappeared")?;
        ensure!(turn.prompt == anchor.prompt_tokens, "prefix input authority differs from captured prompt");
        Ok(turn)
    }).collect()
}
fn divergent_prompt(tokens: &[u32], prefix: &[u32], equality: bool) -> Vec<u32> {
    let mut prompt = prefix.to_vec();
    if !equality { prompt.extend_from_slice(&tokens[128..144]); }
    prompt
}
#[derive(Serialize)]
struct CaseEvidence {
    schema: &'static str,
    class: &'static str,
    case: String,
    route: Route,
    rank: usize,
    equality: bool,
    compared_forward_rows: usize,
    compared_hidden_rows: usize,
    target_tokens: Vec<u32>,
    actual_lane0_completed_k3: u64,
    actual_lane0_completed_joint_mtp: u64,
    twin_lane0_completed_k3: u64,
    twin_lane0_completed_joint_mtp: u64,
    actual_route_events: Vec<&'static str>,
    twin_route_events: Vec<&'static str>,
    actual_lane0_admission_decision: &'static str,
    twin_lane0_admission_decision: &'static str,
    single_lane0_admission_decision: &'static str,
    single_lane0_completed_k3: u64,
    single_lane0_completed_joint_mtp: u64,
    single_route_events: Vec<&'static str>,
    single_slot_parity_rows: Option<usize>,
    same_concurrency_raw_twin: bool,
    checkpoint_depth: usize,
    prefix_script_sha256: String,
    single_slot_twin: bool,
    state_exact: bool,
    peer_restore_unchanged: bool,
    resolved_speculation: &'static str,
    literal_k3_capability: &'static str,
}
fn prepare_peers(fixture: &mut Fixture, lane: &mut Lane) -> Result<()> {
    let supervisor = fixture.supervisor.clone();
    let mut states = Vec::new();
    for index in 1..4 {
        let tokens = fixture.packet.workload.lanes[index][..128].to_vec();
        lane.driver.admit(SlotId(index as u32), tokens.len(), DECODE_LIMIT)?;
        let qwen = fixture.qwen();
        let mut pending = Qwen35PrefillState::begin(tokens, SamplingParams {
            max_tokens: DECODE_LIMIT, ..SamplingParams::default()
        }, None, &mut lane.cache, SlotId(index as u32), 0, None, None, None,
            qwen.hidden_size, qwen.model.kv_graft.as_ref())?;
        loop {
            match pending.advance(qwen, None, &mut lane.cache, CHUNK, &supervisor, None)? {
                Qwen35PrefillAdvance::Pending { state, .. } => pending = state,
                Qwen35PrefillAdvance::Ready { state, prefill_logits, .. } => {
                    let seed_logits = state.finished_at_seed().then_some(prefill_logits);
                    states.push(matrix_dispatch::Ready { state, seed_logits });
                    break;
                }
            }
        }
    }
    lane.driver.run(fixture, &mut lane.cache, states, &mut lane.store, Some(1))?;
    ensure!((1..SLOTS).all(|slot| lane.driver.active(SlotId(slot))),
        "concurrency fixture must retain all three actually decoding peers");
    Ok(())
}
pub(super) fn compare_final(fixture: &mut Fixture, lane: &mut Lane, prompt: &[u32], cached: usize,
    peers: bool, name: &str, arm: &str) -> Result<(Observations, State, Vec<u32>)> {
    lane.admit(prompt.len(), DECODE_LIMIT)?;
    if peers { prepare_peers(fixture, lane)?; }
    compare_final_prepared(fixture, lane, prompt, cached, name, arm)
}
fn compare_final_prepared(fixture: &mut Fixture, lane: &mut Lane, prompt: &[u32], cached: usize,
    name: &str, arm: &str) -> Result<(Observations, State, Vec<u32>)> {
    let scope = trace::Scope::begin(None, 0)?;
    scope.attach(super::transaction_trace::Writer::new(fixture, name, arm)?, &lane.retained)?;
    let served = prefill(fixture, lane, prompt, cached, true)?;
    let readiness = served.state.joint_decode_readiness(&lane.cache).map_err(|reason| reason.label());
    let tokens = finish_success(fixture, lane, served)?;
    scope.publish(&lane.retained);
    trace::transaction(&lane.cache, lane.slot)?;
    scope.finish_transactions()?;
    let mut observed = scope.take()?;
    observed.readiness = Some(readiness);
    drop(scope);
    let decision = lane.admission.context("completed generation lost its production admission decision")?;
    let (k3, joint_mtp) = completed_routes(&observed);
    fs::write(fixture.packet.output.join(format!("{name}-{arm}-participation.json")),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"hf2q.c2.lane-participation.v1","case":name,"arm":arm,"slot":lane.slot.0,
            "route":fixture.packet.route,"admission_decision":decision.metric_reason(),
            "completed_k3":k3,"completed_joint_mtp":joint_mtp,"route_events":observed.route_events,
        }))?)?;
    if fixture.packet.route == Route::MtpAuto {
        if decision == crate::serve::api::qwen35_speculation::QwenSpeculationDecision::Eligible {
            ensure!(k3 + joint_mtp > 0, "eligible AUTO target {name}/{arm} never completed K3/joint-MTP");
        } else {
            ensure!(k3 + joint_mtp == 0, "ineligible AUTO target {name}/{arm} completed MTP");
        }
    }
    Ok((observed, State::read(&lane.cache, lane.slot)?, tokens))
}
fn completed_routes(observed: &Observations) -> (u64, u64) {
    let count = |name| observed.route_events.iter().filter(|&&route| route == name).count() as u64;
    (count("K3-actual-solo-completed"), count("JointMTP-actual-window-completed"))
}
fn compare_case(fixture: &mut Fixture, rank: usize, equality: bool, peers: bool) -> Result<()> {
    let selected = 3 - rank;
    let name = format!("rank{rank}-eq{equality}-peers{peers}");
    // Establish the input authority before any restore. Only this never-restored
    // lane determines the nested input script; restored output cannot redefine it.
    let mut authority = Lane::new(fixture)?;
    let authority_ids = build(fixture, &mut authority, 3)?;
    let script = script(&mut authority, &authority_ids)?;
    let script_bytes = serde_json::to_vec_pretty(&script)?;
    let prefix_script_sha256 = hash(&script_bytes);
    fs::write(fixture.packet.output.join(format!("{name}-prefix-script.json")), script_bytes)?;
    drop(authority);
    let depth = script[selected].prompt.len();
    let prompt = divergent_prompt(&fixture.packet.workload.lanes[0], &script[selected].prompt, equality);
    let mut resumed = Lane::new(fixture)?;
    let ids = replay_script(fixture, &mut resumed, &script, 3, false)?;
    // Reserve slot zero before admitting peers, exactly as a serving admission
    // does. Their live requests/replies survive restore and all decode turns.
    resumed.admit(prompt.len(), DECODE_LIMIT)?;
    if peers { prepare_peers(fixture, &mut resumed)?; }
    let before_peers = (1..4).map(|i| State::read(&resumed.cache, SlotId(i))).collect::<Result<Vec<_>>>()?;
    resumed.restore(ids[selected], &fixture.supervisor)?;
    for (index, before) in before_peers.iter().enumerate() {
        before.same(&State::read(&resumed.cache, SlotId(index as u32 + 1))?, "peer changed across target restore")?;
    }
    let (actual, actual_state, actual_tokens) = compare_final_prepared(fixture, &mut resumed,
        &prompt, depth, &name, "actual")?;
    let mut cold = independent_checkpoint(fixture, &script, selected)?;
    // First twin: preserve every raw transaction, including rejected verifier
    // tails. Default cap8 allocates different depths at different lane widths.
    let (expected, expected_state, expected_tokens) = compare_final(fixture, &mut cold,
        &prompt, depth, peers, &name, "twin")?;
    super::trace_pair::write_pair(fixture, &name, &actual, &expected)?;
    super::state_pair::write(fixture, &name, &actual_state, &expected_state,
        &resumed.retained, &cold.retained)?;
    ensure!(actual == expected, "DivergedExactInvariant: all actual logits/hidden/decision rows or routes");
    actual_state.same(&expected_state, "resumed target/MTP/KV/TQ/DeltaNet/parity versus same-concurrency cold twin")?;
    ensure!(actual_tokens == expected_tokens && resumed.retained == cold.retained,
        "DivergedExactInvariant: tokens or retained ledger");
    // Second twin: independently replay a single active slot. Canonical
    // per-position parity covers every accepted decision, its full vocabulary
    // and actual hidden row, plus the complete final semantic state/ledger.
    let mut single_lane = if peers { Some(independent_checkpoint(fixture, &script, selected)?) } else { None };
    let single = if let Some(lane) = single_lane.as_mut() {
        Some(compare_final(fixture, lane, &prompt, depth, false, &name, "single")?)
    } else { None };
    let mut single_slot_parity_rows = None;
    if let (Some((observed, state, tokens)), Some(lane)) = (&single, &single_lane) {
        super::trace_pair::write_arm(fixture, &name, "single", observed)?;
        let parity_case = format!("{name}-single-slot-parity");
        super::state_pair::write(fixture, &parity_case, &actual_state, state,
            &resumed.retained, &lane.retained)?;
        actual_state.same(state, "concurrent target versus independent single-slot semantic cache")?;
        ensure!(resumed.retained == lane.retained, "DivergedExactInvariant: single-slot retained ledger");
        let graft = fixture.qwen().model.kv_graft.as_ref().map_or(0, |graft| graft.bank.n_slots as usize);
        single_slot_parity_rows = Some(canonical_pair::write_and_compare(fixture, &name, &prompt,
            &actual, &actual_tokens, observed, tokens, graft)?);
    }
    let (actual_k3, actual_joint) = completed_routes(&actual);
    let (twin_k3, twin_joint) = completed_routes(&expected);
    let single_observed = single.as_ref().map_or(&expected, |result| &result.0);
    let (single_k3, single_joint) = completed_routes(single_observed);
    let single_route_events = single_observed.route_events.clone();
    let graft_bound = fixture.packet.graft.is_some();
    use crate::serve::api::qwen35_speculation::QwenSpeculationDecision;
    let expected_admission = cold.admission.context("independent admission decision absent")?;
    ensure!(resumed.admission == Some(expected_admission), "restored admission eligibility differs from cold reference");
    if let Some(lane) = &single_lane {
        ensure!(lane.admission == Some(expected_admission), "single-slot admission eligibility differs");
    }
    let eligible_auto = fixture.packet.route == Route::MtpAuto && !graft_bound;
    if eligible_auto {
        ensure!(expected_admission == QwenSpeculationDecision::Eligible,
            "nongrafted AUTO cold checkpoint unexpectedly lost its independently replayed MTP boundary");
        ensure!(actual_k3 + actual_joint > 0, "restored target lane never completed eligible K3/joint-MTP");
        ensure!(twin_k3 + twin_joint > 0, "independent target twin never completed eligible K3/joint-MTP");
        ensure!(single_k3 + single_joint > 0, "single-slot target never completed eligible K3/joint-MTP");
    }
    if fixture.packet.route == Route::MtpAuto && graft_bound {
        let required = if equality { QwenSpeculationDecision::PromptCacheHit }
            else { QwenSpeculationDecision::RuntimeUnavailable };
        ensure!(expected_admission == required, "actual graft/equality capability boundary changed");
        ensure!(actual_k3 + actual_joint + twin_k3 + twin_joint + single_k3 + single_joint == 0,
            "ineligible target arm completed ungrafted K3/joint-MTP");
        ensure!(resumed.spec.is_none() && cold.spec.is_none(),
            "ineligible AUTO retained ungrafted speculative metadata");
        if let Some(lane) = &single_lane { ensure!(lane.spec.is_none(), "single-slot graft retained stale MTP"); }
    }
    if fixture.packet.route == Route::CurrentDefault {
        let ran_default = |observed: &Observations| observed.route_events.iter()
            .any(|&route| route == "DFlash2-actual-runner-completed");
        ensure!(ran_default(&actual) == !graft_bound && ran_default(&expected) == !graft_bound
            && ran_default(single_observed) == !graft_bound,
            "default actual completed route/graft capability did not match cold/resumed cells");
        if graft_bound { ensure!(resumed.spec.is_none() && cold.spec.is_none(),
            "grafted default retained ungrafted speculative metadata"); }
    }
    let evidence = CaseEvidence { schema: "hf2q.c2.exactness-case.v1", class: "ClassA",
        case: name.clone(), route: fixture.packet.route, rank, equality,
        compared_forward_rows: actual.forwards.iter().map(|f| f.logits.len() / f.vocab).sum(),
        compared_hidden_rows: actual.forwards.iter().map(|f| if f.hidden_width == 0 {0} else {f.hidden.len() / f.hidden_width}).sum(),
        target_tokens: actual_tokens,
        actual_lane0_completed_k3: actual_k3, actual_lane0_completed_joint_mtp: actual_joint,
        twin_lane0_completed_k3: twin_k3, twin_lane0_completed_joint_mtp: twin_joint,
        actual_route_events: actual.route_events, twin_route_events: expected.route_events,
        single_lane0_completed_k3: single_k3, single_lane0_completed_joint_mtp: single_joint,
        single_route_events, single_slot_parity_rows, same_concurrency_raw_twin: true,
        actual_lane0_admission_decision: resumed.admission.context("actual admission absent")?.metric_reason(),
        twin_lane0_admission_decision: expected_admission.metric_reason(),
        single_lane0_admission_decision: match &single_lane {
            Some(lane) => lane.admission.context("single-slot admission absent")?,
            None => expected_admission, // The raw twin itself is single-slot.
        }.metric_reason(),
        checkpoint_depth: depth, prefix_script_sha256, single_slot_twin: true,
        state_exact: true, peer_restore_unchanged: true,
        resolved_speculation: if fixture.packet.route == Route::MtpAuto {
            if equality && graft_bound { "PromptCacheHit/ordinary-grafted" }
            else if graft_bound { "RuntimeUnavailable/ordinary-grafted" }
            else { fixture.packet.route.label() }
        } else { fixture.packet.route.label() },
        literal_k3_capability: if equality && graft_bound && fixture.packet.route == Route::MtpAuto { "PromptCacheHit" }
            else if graft_bound && fixture.packet.route == Route::MtpAuto { "ExpectedCapabilityNegativeControl-ADR059-no-grafted-MTP" }
            else if eligible_auto { "ObservedActualK3" } else { "NotDeclaredK3" } };
    fs::write(fixture.packet.output.join(format!("{name}.json")), serde_json::to_vec_pretty(&evidence)?)?;
    Ok(())
}
fn failure_cases(fixture: &mut Fixture) -> Result<()> {
    for fault in [AnchorRestoreFault::Preflight, AnchorRestoreFault::Encode] {
        let mut lane = Lane::new(fixture)?;
        let ids = build(fixture, &mut lane, 3)?;
        let before = State::read(&lane.cache, SLOT)?;
        let committed: Vec<_> = lane.store.committed().iter().map(|a| a.id).collect();
        lane.store.restore_fault = Some(fault);
        let refused = lane.restore(ids[1], &fixture.supervisor);
        ensure!(refused.is_err(), "injected restore failure did not execute");
        if matches!(fault, AnchorRestoreFault::Preflight) {
            before.same(&State::read(&lane.cache, SLOT)?, "Phase A mutated cache")?;
            ensure!(committed == lane.store.committed().iter().map(|a| a.id).collect::<Vec<_>>()
                && !lane.store.is_poisoned(), "Phase A mutated store");
        } else {
            ensure!(lane.store.is_poisoned() && lane.store.is_empty(), "encode failure must poison/clear without shallower fallback");
        }
        lane.cold()?;
        let mut cold = Lane::new(fixture)?;
        let prompt = fixture.packet.workload.lanes[0][..128].to_vec();
        let name = format!("failure-{fault:?}");
        let actual = compare_final(fixture, &mut lane, &prompt, 0, false, &name, "actual")?;
        let expected = compare_final(fixture, &mut cold, &prompt, 0, false, &name, "twin")?;
        super::trace_pair::write_pair(fixture, &name, &actual.0, &expected.0)?;
        super::state_pair::write(fixture, &name, &actual.1, &expected.1,
            &lane.retained, &cold.retained)?;
        ensure!(actual == expected, "DivergedExactInvariant: failure recovery cold twin");
    }
    Ok(())
}
pub(super) fn run(family: Family) -> Result<()> {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut fixture = Fixture::load(family)?;
    // Root runs absent and bound-same-graft packets separately, never two models.
    for rank in [0, 2] { for equality in [false, true] { for peers in [false, true] {
        compare_case(&mut fixture, rank, equality, peers)?;
    }}}
    super::cancellation::run(&mut fixture)?;
    failure_cases(&mut fixture)?;
    super::graft::changed_identity(&mut fixture)?;
    fixture.finish()
}

pub(super) fn o1(family: Family) -> Result<()> {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut fixture = Fixture::load(family)?;
    let mut authority = Lane::new(&fixture)?;
    let authority_ids = build(&mut fixture, &mut authority, 2)?;
    let script = script(&mut authority, &authority_ids)?;
    fs::write(fixture.packet.output.join("o1-prefix-script.json"), serde_json::to_vec_pretty(&script)?)?;
    drop(authority);
    let a = script[2].prompt.clone();
    let a0 = &script[0].prompt;
    let c = divergent_prompt(&fixture.packet.workload.lanes[0], &a, false);
    for shallower in [false, true] {
        for disabled_pruning in [false, true] {
            let mut lane = Lane::new(&fixture)?;
            let ids = replay_script(&mut fixture, &mut lane, &script, 2, false)?;
            let a_id = ids[2];
            lane.store.pruning_disabled = disabled_pruning;
            let tombstone_before = if shallower {
                ANCHOR_TELEMETRY.tombstoned_cursor.load(Ordering::Relaxed)
            } else { ANCHOR_TELEMETRY.tombstoned_cold_reset.load(Ordering::Relaxed) };
            let shared = if shallower { a0.len() } else { 0 };
            let source = &fixture.packet.workload.lanes[1];
            let offset = source.iter().position(|&token| token != a[shared])
                .context("frozen O1 workload has no token diverging at the required boundary")?;
            let mut b = a[..shared].to_vec();
            b.extend(source.iter().cycle().skip(offset).take(a.len() + 16 - shared).copied());
            ensure!(b.len() > a.len() && b[..shared] == a[..shared] && b[shared] != a[shared],
                "O1 B must overwrite below A and complete above its actual checkpoint depth");
            if shallower {
                lane.restore(ids[0], &fixture.supervisor)?;
                let served = prefill(&mut fixture, &mut lane, &b, a0.len(), true)?;
                finish_success(&mut fixture, &mut lane, served)?;
            } else {
                // Fault injection suppresses ONLY store pruning. Reset still
                // advances cache generation, so its independent guard MUST refuse.
                if !disabled_pruning { lane.store.reset(); }
                lane.cache.reset_for_slot(SLOT)?;
                lane.retained.clear(); lane.spec = None;
                let served = prefill(&mut fixture, &mut lane, &b, 0, true)?;
                finish_success(&mut fixture, &mut lane, served)?;
            }
            ensure!(lane.cache.sequence_len_for_slot(SLOT)? as usize > a.len(), "B did not complete at a high cursor");
            let tombstone_after = if shallower {
                ANCHOR_TELEMETRY.tombstoned_cursor.load(Ordering::Relaxed)
            } else { ANCHOR_TELEMETRY.tombstoned_cold_reset.load(Ordering::Relaxed) };
            if !disabled_pruning {
                ensure!(tombstone_after > tombstone_before, "O1 did not observe its actual tombstone cause");
            }
            let refused_before = ANCHOR_TELEMETRY.restore_refused.load(Ordering::Relaxed);
            let restore = lane.restore(a_id, &fixture.supervisor);
            let negative = shallower && disabled_pruning;
            if negative {
                ensure!(restore.is_ok(), "pruning-only negative control did not reuse overwritten A");
            } else {
                ensure!(restore.is_err(), "overwritten A remained restorable");
                ensure!(ANCHOR_TELEMETRY.restore_refused.load(Ordering::Relaxed) > refused_before,
                    "O1 refusal counter is vacuous");
                if !shallower && disabled_pruning {
                    ensure!(format!("{:#}", restore.as_ref().err().expect("checked refusal"))
                        .contains("slot lineage changed"), "cold-reset negative control did not hit generation guard");
                }
                lane.cold()?;
            }
            let mut cold = if negative {
                independent_checkpoint(&mut fixture, &script, 2)?
            } else { Lane::new(&fixture)? };
            let name = format!("o1-shallower{shallower}-pruningdisabled{disabled_pruning}");
            let cached = if negative { a.len() } else { 0 };
            let actual = compare_final(&mut fixture, &mut lane, &c, cached, false, &name, "actual")?;
            let expected = compare_final(&mut fixture, &mut cold, &c, cached, false, &name, "twin")?;
            let equal = actual == expected && lane.retained == cold.retained;
            ensure!(equal != negative, "O1 sensitivity failed: expected {}", if negative {"DivergedExactInvariant"} else {"ExactEqual"});
            super::trace_pair::write_pair(&fixture, &name, &actual.0, &expected.0)?;
            super::state_pair::write(&fixture, &name, &actual.1, &expected.1,
                &lane.retained, &cold.retained)?;
            fs::write(fixture.packet.output.join(format!("{name}.json")), serde_json::to_vec_pretty(&serde_json::json!({
                "schema":"hf2q.c2.o1.v1", "shallower":shallower, "pruning_disabled":disabled_pruning,
                "classification":if equal {"ExactEqual"} else {"DivergedExactInvariant"},
                "cold_reset_negative_control":if !shallower && disabled_pruning {Some("RefusedByIndependentCacheGeneration")} else {None},
                "b_completed_high_cursor":true, "second_fault":false
                ,"checkpoint_depth":a.len(),"shallower_depth":a0.len(),"b_prompt_depth":b.len(),
                "b_shared_prefix":shared,"twin_image_restores":0
                ,"tombstoned":tombstone_after-tombstone_before,
                "restore_refused":ANCHOR_TELEMETRY.restore_refused.load(Ordering::Relaxed)-refused_before
            }))?)?;
        }
    }
    fixture.finish()
}
