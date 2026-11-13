//! Qwen DFlash2 serving transaction.
//!
//! One step drafts anchor-plus-seven proposal paths for every active physical
//! slot, verifies all selected rows in one fixed width-32 target tile, commits
//! only each lane's accepted prefix, and queues semantic output for the same
//! in-turn drain used by joint MTP. DFlash2 is optional and fail-closed: a lane
//! joins only when its target cursor and persistent draft-ring cursor agree.

use std::time::Instant;

use anyhow::{Context, Result};

use crate::inference::models::qwen35::joint_decode_routes::{JointDenseFfnRoute, JointRouteMode};
use crate::inference::models::qwen35::joint_step_options::JointStepOptions;
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::verify_decision::DECISION_SLOTS;
use crate::inference::models::qwen35::{
    Qwen35JointOutput, Qwen35JointRowOutput, Qwen35JointVerify, Qwen35JointVerifyLane,
};
use crate::inference::spec_decode::dflash2::{
    adapters::{DFlash2TargetFamily, QwenDFlash2Adapter},
    route_profile,
    verify_budget::VerifyBudgetLane,
};
use crate::serve::api::engine_supervisor::EngineSupervisor;

use super::{
    plan_qwen35_verified_block, Qwen35CanonicalDecision, Qwen35DecisionSource, Qwen35DecodeState,
    Qwen35LoadedModel, Qwen35SpecSemanticState, Qwen35TickOutcome, Qwen35VerifiedBlockPlan,
    QWEN35_WORKER_TRANSACTION_TIMEOUT,
};

const DFLASH2_MAX_ROWS_PER_LANE: usize = 8;
const DFLASH2_PROPOSALS_PER_LANE: usize = DFLASH2_MAX_ROWS_PER_LANE - 1;

/// Opt-in switch of the context-first commit order; exact value `1`.
pub(in crate::serve::api) const DFLASH2_CONTEXT_FIRST_ENV: &str = "HF2Q_DFLASH2_CONTEXT_FIRST";

/// Whether a step submits its draft-context commit before it encodes the
/// DeltaNet replay, as this model's routes resolved it.
pub(in crate::serve::api) fn context_first() -> bool {
    const SWITCH: route_profile::Switch = route_profile::switch(DFLASH2_CONTEXT_FIRST_ENV);
    route_profile::is_one(SWITCH)
}

/// Opt-in switch of the draft prefetch; exact value `1`.
pub(in crate::serve::api) const DFLASH2_DRAFT_PREFETCH_ENV: &str = "HF2Q_DFLASH2_DRAFT_PREFETCH";

/// Whether a step that settles acceptance on the host commits the next
/// step's draft command before it waits for its own replay, as this model's
/// routes resolved it.
pub(in crate::serve::api) fn draft_prefetch() -> bool {
    const SWITCH: route_profile::Switch = route_profile::switch(DFLASH2_DRAFT_PREFETCH_ENV);
    route_profile::is_one(SWITCH)
}

/// Switch of the GPU decision candidates of host-sampled verify rows:
/// `1` uses them, `check` computes them beside the host sampler's decision
/// over the full logits and compares the two plans (a diagnostic).
pub(in crate::serve::api) const DFLASH2_GPU_DECISIONS_ENV: &str = "HF2Q_DFLASH2_GPU_DECISIONS";

/// How a step decides the rows of its lanes that sample on the host.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::serve::api) enum DecisionRoute {
    /// The host sampler over every row's logits.
    #[default]
    HostLogits,
    /// The decision candidates; a row they cannot decide takes the host
    /// sampler over that row's logits.
    Candidates,
    /// Both, compared; the host sampler's plan is the one used.
    Check,
}

impl DecisionRoute {
    pub(in crate::serve::api) fn label(self) -> &'static str {
        match self {
            Self::HostLogits => "host_logits",
            Self::Candidates => "candidates",
            Self::Check => "check",
        }
    }
}

fn parse_decision_route(raw: Option<&str>) -> DecisionRoute {
    match raw {
        Some("1") => DecisionRoute::Candidates,
        Some("check") => DecisionRoute::Check,
        _ => DecisionRoute::HostLogits,
    }
}

/// The decision route as this model's routes resolved it.
pub(in crate::serve::api) fn decision_route() -> DecisionRoute {
    const SWITCH: route_profile::Switch = route_profile::switch(DFLASH2_GPU_DECISIONS_ENV);
    route_profile::with_value(SWITCH, parse_decision_route)
}

/// Development diagnostic for measuring the two tail switches inside one
/// process: a positive block length `n` runs the switches that are on for
/// `n` cycles, then the stock route for `n` cycles, and so on, and marks
/// every cycle in the cycle timeline (`tail_block.on` / `tail_block.off`).
pub(in crate::serve::api) const DFLASH2_TAIL_BLOCKS_ENV: &str = "HF2Q_DFLASH2_TAIL_BLOCKS";

/// The arm of the next cycle under [`DFLASH2_TAIL_BLOCKS_ENV`]: `None` when
/// the diagnostic is off, else whether the tail switches apply.
pub(in crate::serve::api) fn tail_block_arm() -> Option<bool> {
    static BLOCK: LazyLock<Option<u64>> = LazyLock::new(|| {
        std::env::var(DFLASH2_TAIL_BLOCKS_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|&block| block > 0)
    });
    static CYCLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let block = (*BLOCK)?;
    let cycle = CYCLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some((cycle / block) % 2 == 0)
}

/// Plans of a `check` run whose candidate plan was not the host sampler's.
static DECISION_MISMATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The draft call the step after this one will make, as far as this step
/// can tell: every lane that verified goes on from the last token of its
/// plan unless the plan ended it or it reached its length limit.
struct NextDraft {
    anchors: Vec<u32>,
    positions: Vec<u32>,
    maximum_rows: Vec<u32>,
    lane_order: Vec<u32>,
    gpu_target_tokens: bool,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::serve::api) struct Qwen35DFlash2Forward {
    pub mode: JointRouteMode,
    pub dense_ffn: JointDenseFfnRoute,
    pub options: JointStepOptions,
    /// Commit the next step's draft before waiting for this step's replay
    /// ([`draft_prefetch`]).
    pub draft_prefetch: bool,
    /// How host-sampled rows are decided ([`decision_route`]).
    pub decisions: DecisionRoute,
    /// The cycle's arm under [`DFLASH2_TAIL_BLOCKS_ENV`], for its timeline
    /// mark; the two fields above already follow it.
    pub tail_block: Option<bool>,
}

struct Qwen35DFlash2Window {
    pending_token: u32,
    position: i32,
    greedy: bool,
    maximum_rows: usize,
    /// How the window's rows are decided; [`DecisionRoute::HostLogits`] for
    /// a greedy window (the GPU argmax decides it) and for a request the
    /// candidates do not represent.
    decisions: DecisionRoute,
}

impl Qwen35DFlash2Window {
    fn output(&self) -> Qwen35JointOutput {
        if self.greedy {
            Qwen35JointOutput::GreedyTokenAndHidden
        } else if self.decisions == DecisionRoute::Candidates {
            Qwen35JointOutput::Hidden
        } else {
            Qwen35JointOutput::LogitsAndHidden
        }
    }

    fn tokens(&self, proposals: &[u32], verify_rows: usize) -> Result<Vec<u32>> {
        anyhow::ensure!(
            (1..=self.maximum_rows).contains(&verify_rows),
            "Qwen DFlash2 selected {verify_rows} rows outside 1..={}",
            self.maximum_rows
        );
        anyhow::ensure!(
            verify_rows - 1 <= proposals.len(),
            "Qwen DFlash2 selected {verify_rows} rows from {} proposals",
            proposals.len()
        );
        let mut tokens = Vec::with_capacity(verify_rows);
        tokens.push(self.pending_token);
        tokens.extend_from_slice(&proposals[..verify_rows - 1]);
        Ok(tokens)
    }
}

struct Qwen35DFlash2Plan {
    plan: Qwen35VerifiedBlockPlan,
    /// Rows the decision candidates decided, and rows that had candidates
    /// but took the host sampler over their own logits.
    candidate_rows: usize,
    fallback_rows: usize,
}

impl Qwen35DFlash2Plan {
    fn accepted(&self) -> usize {
        self.plan.valid_input_tokens
    }
}

enum Verified {
    Window {
        plan: Qwen35DFlash2Plan,
        tokens: Vec<u32>,
    },
    Failed(anyhow::Error),
}

impl Qwen35DecodeState {
    fn prepare_dflash2_window(
        &self,
        kv_cache: &HybridKvCache,
        draft_context: usize,
    ) -> Result<Qwen35DFlash2Window> {
        self.joint_decode_readiness(kv_cache)
            .map_err(|reason| anyhow::anyhow!("Qwen DFlash2 lane refused: {}", reason.label()))?;
        let pending_token = *self
            .generated_tokens
            .last()
            .context("Qwen DFlash2 window needs a seeded output token")?;
        let position = (self.decode_position_base + self.generated_tokens.len() - 1) as i32;
        let target_cursor = kv_cache.sequence_len_for_slot(self.slot_id)? as usize;
        anyhow::ensure!(
            draft_context == target_cursor,
            "Qwen DFlash2 slot {} draft cursor {draft_context} != target cursor {target_cursor}",
            self.slot_id.0
        );
        let remaining = self.max_tokens.saturating_sub(self.generated_tokens.len());
        anyhow::ensure!(remaining > 0, "Qwen DFlash2 window past the length limit");
        Ok(Qwen35DFlash2Window {
            pending_token,
            position,
            greedy: self.is_greedy && !self.want_logprobs,
            maximum_rows: remaining.min(DFLASH2_MAX_ROWS_PER_LANE),
            decisions: DecisionRoute::HostLogits,
        })
    }

    /// Whether the decision candidates represent this request's sampler: a
    /// host-sampled window at temperature zero with nothing between the
    /// logits and the argmax but the repetition penalty (no logit bias,
    /// top-k, top-p, min-p, seed or logprobs), whose repetition window fits
    /// the candidates' history. A grammar is admitted; the rows it
    /// constrains fall back to the host sampler one by one.
    fn decision_candidates_eligible(&self, window: &Qwen35DFlash2Window) -> bool {
        !window.greedy
            && !self.want_logprobs
            && super::is_qwen_server_speculation_exact_eligible(&self.params)
            && f64::from(self.params.temperature) < crate::serve::sampler_pure::SAMPLING_EPS
            && self.sampling_history.len() <= DECISION_SLOTS
    }

    fn gpu_greedy_acceptance_eligible(&self, window: &Qwen35DFlash2Window) -> bool {
        window.greedy
            && super::shadow_draft::verify_transforms_clear(self)
            && super::is_qwen_server_speculation_exact_eligible(&self.params)
    }

    fn plan_dflash2_window(
        &self,
        qwen: &Qwen35LoadedModel,
        window: &Qwen35DFlash2Window,
        drafts: &[u32],
        mut rows: Vec<Qwen35JointRowOutput>,
    ) -> Result<Qwen35DFlash2Plan> {
        self.plan_dflash2_rows(qwen, window, drafts, &mut rows, window.decisions)
    }

    /// [`Self::plan_dflash2_window`] over borrowed rows with an explicit
    /// decision route (`check` plans the same rows twice).
    fn plan_dflash2_rows(
        &self,
        qwen: &Qwen35LoadedModel,
        window: &Qwen35DFlash2Window,
        drafts: &[u32],
        rows: &mut [Qwen35JointRowOutput],
        decisions: DecisionRoute,
    ) -> Result<Qwen35DFlash2Plan> {
        anyhow::ensure!(
            rows.len() == drafts.len() + 1,
            "Qwen DFlash2 window verified {} rows for {} drafts",
            rows.len(),
            drafts.len()
        );
        let mut semantic = Qwen35SpecSemanticState::from_decode(self);
        let penalty = super::effective_repetition_penalty(&self.params);
        let vocab = qwen.vocab_size;
        let (mut candidate_rows, mut fallback_rows) = (0, 0);
        let mut canonical_at = |row: usize| {
            let output = &mut rows[row];
            if window.greedy {
                let token = output
                    .greedy_token
                    .context("Qwen DFlash2 greedy row returned no argmax")?;
                return semantic.select_and_observe_from(
                    qwen,
                    &self.params,
                    Qwen35DecisionSource::Argmax(token),
                );
            }
            let candidates = match decisions {
                DecisionRoute::HostLogits => None,
                DecisionRoute::Candidates | DecisionRoute::Check => output.decision.as_ref(),
            };
            if let Some(candidates) = candidates {
                // The sampler is a penalised argmax exactly when no grammar
                // constrains this row: none at all (and then no token table,
                // which the sampler refuses), or one still waiting for its
                // trigger (with the table the sampler checks first).
                let unconstrained = match (
                    semantic.grammar_runtime.as_ref(),
                    self.params.token_bytes.as_deref(),
                ) {
                    (None, None) => true,
                    (Some(grammar), Some(table)) => {
                        grammar.is_awaiting_trigger() && table.len() == vocab
                    }
                    _ => false,
                };
                let token = unconstrained
                    .then(|| candidates.greedy_token(&semantic.sampling_history, penalty))
                    .flatten();
                if let Some(token) = token {
                    candidate_rows += 1;
                    return semantic.select_and_observe_from(
                        qwen,
                        &self.params,
                        Qwen35DecisionSource::Argmax(token),
                    );
                }
                fallback_rows += 1;
                let mut logits = candidates
                    .logits()
                    .context("Qwen DFlash2 decision fallback logits")?;
                return semantic.select_and_observe_from(
                    qwen,
                    &self.params,
                    Qwen35DecisionSource::Logits(&mut logits),
                );
            }
            // The host sampler changes the logits it is given (penalty,
            // grammar mask): sample from a copy when the row is planned
            // twice.
            let mut copy;
            let logits = output
                .logits
                .as_deref_mut()
                .context("Qwen DFlash2 row returned no logits")?;
            let logits = if window.decisions == DecisionRoute::Check {
                copy = logits.to_vec();
                copy.as_mut_slice()
            } else {
                logits
            };
            semantic.select_and_observe_from(
                qwen,
                &self.params,
                Qwen35DecisionSource::Logits(logits),
            )
        };
        let plan = if drafts.is_empty() {
            let decision: Qwen35CanonicalDecision = canonical_at(0)?;
            Qwen35VerifiedBlockPlan {
                output: if decision.terminal {
                    Default::default()
                } else {
                    [decision.token].into()
                },
                terminal_after_pending: decision.terminal,
                matched_drafts: 0,
                rejected_drafts: 0,
                valid_input_tokens: 1,
                carry_hidden_row: 0,
            }
        } else {
            plan_qwen35_verified_block(drafts, &mut canonical_at)?
        };
        Ok(Qwen35DFlash2Plan {
            plan,
            candidate_rows,
            fallback_rows,
        })
    }

    fn plan_selected_dflash2_window(
        &self,
        qwen: &Qwen35LoadedModel,
        window: &Qwen35DFlash2Window,
        tokens: &[u32],
        mut rows: Vec<Qwen35JointRowOutput>,
    ) -> Result<Qwen35DFlash2Plan> {
        anyhow::ensure!(!tokens.is_empty(), "Qwen DFlash2 selected no target rows");
        rows.truncate(tokens.len());
        if window.decisions != DecisionRoute::Check {
            return self.plan_dflash2_window(qwen, window, &tokens[1..], rows);
        }
        // Diagnostic: the candidates' plan beside the host sampler's over
        // the full logits. The host sampler's is the one used.
        let drafts = &tokens[1..];
        let candidate =
            self.plan_dflash2_rows(qwen, window, drafts, &mut rows, DecisionRoute::Candidates);
        let mut host =
            self.plan_dflash2_rows(qwen, window, drafts, &mut rows, DecisionRoute::HostLogits)?;
        match candidate {
            Ok(candidate) if candidate.plan == host.plan => {
                host.candidate_rows = candidate.candidate_rows;
                host.fallback_rows = candidate.fallback_rows;
            }
            other => {
                let total =
                    DECISION_MISMATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                tracing::error!(
                    target: "hf2q::serve::api::engine_qwen35::progress",
                    slot = self.slot_id.0,
                    mismatches = total,
                    host = ?host.plan,
                    candidate = ?other.map(|plan| plan.plan),
                    "Qwen DFlash2 decision candidates disagree with the host sampler"
                );
            }
        }
        Ok(host)
    }

    fn settle_dflash2_window(&mut self, plan: Qwen35DFlash2Plan) {
        let Qwen35DFlash2Plan { plan, .. } = plan;
        self.mtp = None;
        self.history_lookup = None;
        self.pending_speculation_output = plan.output;
        self.terminal_after_pending = plan.terminal_after_pending;
    }
}

impl Qwen35LoadedModel {
    /// Run one DFlash2 draft/verify/commit transaction over distinct physical
    /// scheduler slots. The optional runtime is always restored before return.
    pub(in crate::serve::api) fn run_dflash2_step(
        &mut self,
        kv_cache: &mut HybridKvCache,
        states: &mut [&mut Qwen35DecodeState],
        forward: Qwen35DFlash2Forward,
        supervisor: &EngineSupervisor,
        is_fatal: &dyn Fn(&anyhow::Error) -> bool,
    ) -> Result<Vec<Result<Qwen35TickOutcome>>> {
        let mut runtime = self
            .dflash2
            .take()
            .context("Qwen DFlash2 runtime is not provisioned")?;
        let result = self.run_dflash2_step_inner(
            &mut runtime,
            kv_cache,
            states,
            forward,
            supervisor,
            is_fatal,
        );
        self.dflash2 = Some(runtime);
        result
    }

    fn run_dflash2_step_inner(
        &self,
        runtime: &mut crate::inference::spec_decode::dflash2::runtime::DFlash2Runtime,
        kv_cache: &mut HybridKvCache,
        states: &mut [&mut Qwen35DecodeState],
        forward: Qwen35DFlash2Forward,
        supervisor: &EngineSupervisor,
        is_fatal: &dyn Fn(&anyhow::Error) -> bool,
    ) -> Result<Vec<Result<Qwen35TickOutcome>>> {
        anyhow::ensure!(!states.is_empty(), "Qwen DFlash2 step requires a lane");
        anyhow::ensure!(
            states.len() <= runtime.workspace.max_lanes,
            "Qwen DFlash2 step has {} lanes, runtime supports {}",
            states.len(),
            runtime.workspace.max_lanes
        );
        let _stage_round =
            crate::inference::stage_profile::Round::begin("qwen35", "dflash2", states.len());
        let lanes = states.len();
        let mut ticks: Vec<Option<Result<Qwen35TickOutcome>>> = (0..lanes).map(|_| None).collect();
        let mut windows: Vec<Option<Qwen35DFlash2Window>> = Vec::with_capacity(lanes);
        for (index, state) in states.iter_mut().enumerate() {
            state.consume_planned_tick();
            let slot = state.slot_id.0 as usize;
            let prepared = runtime
                .lane_lengths()
                .get(slot)
                .context("Qwen DFlash2 slot exceeds runtime lanes")
                .and_then(|&draft_context| state.prepare_dflash2_window(kv_cache, draft_context));
            match prepared {
                Ok(mut window) => {
                    if forward.decisions != DecisionRoute::HostLogits
                        && state.decision_candidates_eligible(&window)
                    {
                        window.decisions = forward.decisions;
                    }
                    windows.push(Some(window));
                }
                Err(error) => {
                    ticks[index] = Some(Err(error));
                    windows.push(None);
                }
            }
        }
        let active = (0..lanes)
            .filter(|&index| windows[index].is_some())
            .collect::<Vec<_>>();
        if active.is_empty() {
            return Ok(finish_ticks(ticks));
        }
        let trace_timing = tracing::enabled!(
            target: "hf2q::serve::api::engine_qwen35::progress",
            tracing::Level::DEBUG
        );
        let cycle_started = trace_timing.then(Instant::now);
        let mut timeline = super::dflash2_timeline::Cycle::begin();
        let timeline_on = timeline.is_some();
        let timeline_mark = mlx_native::command_timeline::mark;
        match forward.tail_block {
            Some(true) => timeline_mark("tail_block.on"),
            Some(false) => timeline_mark("tail_block.off"),
            None => {}
        }
        // Native encode counters and this worker thread's CPU time at cycle
        // entry; the DEBUG record reports the dispatches, command buffers,
        // host waits, barriers and host CPU one cycle costs. CPU time close
        // to the cycle's wall time means the host, not the GPU, bounds it.
        let native_counters = || {
            let mut cpu = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            // SAFETY: `cpu` is a valid, exclusively borrowed timespec.
            let cpu_us =
                if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut cpu) } == 0 {
                    cpu.tv_sec as u64 * 1_000_000 + cpu.tv_nsec as u64 / 1_000
                } else {
                    0
                };
            (
                mlx_native::dispatch_count(),
                mlx_native::commit_count(),
                mlx_native::sync_count(),
                mlx_native::barrier_count(),
                cpu_us,
            )
        };
        let counters_at_entry = trace_timing.then(native_counters);

        let physical_lanes = runtime.workspace.max_lanes;
        let mut anchors = vec![0_u32; physical_lanes];
        let mut temperatures = vec![0.0_f32; physical_lanes];
        // The native selector validates every physical lane before dispatch,
        // including lanes omitted from this scheduler step. Give inactive
        // lanes the smallest valid bound; their draft/verify output is ignored.
        let mut maximum_rows = vec![1_u32; physical_lanes];
        for &index in &active {
            let state = &states[index];
            let slot = state.slot_id.0 as usize;
            let window = windows[index].as_ref().expect("active window");
            anchors[slot] = window.pending_token;
            temperatures[slot] = 0.0;
            maximum_rows[slot] = u32::try_from(window.maximum_rows)?;
        }
        let positions = runtime
            .lane_lengths()
            .iter()
            .map(|&position| u32::try_from(position).context("Qwen DFlash2 position exceeds u32"))
            .collect::<Result<Vec<_>>>()?;
        let uniforms = vec![0.5_f32; physical_lanes * DFLASH2_PROPOSALS_PER_LANE];
        let gpu_target_tokens = runtime.gpu_target_tokens()
            && active
                .iter()
                .all(|&index| maximum_rows[states[index].slot_id.0 as usize] == 8);
        let target_lane_order = active
            .iter()
            .map(|&index| states[index].slot_id.0)
            .collect::<Vec<_>>();
        let adapter_config = runtime.config.clone();
        let adapter = QwenDFlash2Adapter::new(&self.model, &adapter_config)?;
        timeline_mark("draft_begin");
        let draft_started = trace_timing.then(Instant::now);
        let lease = supervisor.arm("Qwen35 DFlash2 draft", QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
        let drafted = self.model.with_gpu_cache_mut(|device, registry| {
            if gpu_target_tokens {
                runtime.run_draft_for_gpu_target(
                    device,
                    registry,
                    &adapter,
                    &anchors,
                    &positions,
                    &temperatures,
                    0,
                    &maximum_rows,
                    &target_lane_order,
                    &uniforms,
                )
            } else {
                runtime.run_draft(
                    device,
                    registry,
                    &adapter,
                    &anchors,
                    &positions,
                    &temperatures,
                    0,
                    &maximum_rows,
                    &uniforms,
                )
            }
        });
        let drafted = lease.finish().and(drafted);
        let draft_micros = draft_started.map(|started| started.elapsed().as_micros());
        timeline_mark("draft_end");
        if let Err(error) = drafted {
            if is_fatal(&error) {
                return Err(error);
            }
            distribute_error(&mut ticks, &active, error);
            return Ok(finish_ticks(ticks));
        }
        if !gpu_target_tokens {
            let mut active_lanes = [0; 4];
            for (lane, &index) in active.iter().enumerate() {
                active_lanes[lane] = states[index].slot_id.0 as usize;
            }
            runtime.select_verify_rows(&active_lanes[..active.len()], &maximum_rows)?;
        }
        let mut tokens: Vec<Option<Vec<u32>>> = (0..lanes).map(|_| None).collect();
        if gpu_target_tokens {
            for &index in &active {
                let window = windows[index].as_ref().expect("active window");
                tokens[index] = Some(vec![window.pending_token; window.maximum_rows]);
            }
        } else {
            let proposals = runtime.proposals()?;
            let verify_rows = runtime.verify_rows()?;
            for &index in &active {
                let slot = states[index].slot_id.0 as usize;
                let proposal_start = slot * DFLASH2_PROPOSALS_PER_LANE;
                let proposal_end = proposal_start + DFLASH2_PROPOSALS_PER_LANE;
                match windows[index].as_ref().expect("active window").tokens(
                    &proposals[proposal_start..proposal_end],
                    verify_rows[slot] as usize,
                ) {
                    Ok(lane_tokens) => tokens[index] = Some(lane_tokens),
                    Err(error) => ticks[index] = Some(Err(error)),
                }
            }
        }
        let verifying = active
            .iter()
            .copied()
            .filter(|&index| tokens[index].is_some())
            .collect::<Vec<_>>();
        if verifying.is_empty() {
            return Ok(finish_ticks(ticks));
        }
        let lane_indices = verifying
            .iter()
            .map(|&index| states[index].slot_id.0 as usize)
            .collect::<Vec<_>>();
        let gpu_acceptance = gpu_target_tokens
            && verifying.iter().all(|&index| {
                states[index].gpu_greedy_acceptance_eligible(
                    windows[index].as_ref().expect("verified window"),
                )
            });
        let verify_lanes = verifying
            .iter()
            .map(|&index| {
                let state = &states[index];
                let window = windows[index].as_ref().expect("verifying window");
                Qwen35JointVerifyLane {
                    slot_id: state.slot_id,
                    position: window.position,
                    tokens: tokens[index].as_deref().expect("verifying tokens"),
                    output: window.output(),
                }
            })
            .collect::<Vec<_>>();
        let selected_rows = (trace_timing || timeline_on).then(|| {
            verify_lanes
                .iter()
                .map(|lane| lane.tokens.len())
                .sum::<usize>()
        });
        let adapter = QwenDFlash2Adapter::new(&self.model, &runtime.config)?;
        let capture = adapter.target_capture(&runtime.config, &runtime.workspace, &lane_indices)?;
        timeline_mark("verify_begin");
        let verify_started = Instant::now();
        let lease = supervisor.arm("Qwen35 DFlash2 verify", QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
        // Base repetition windows of the lanes whose rows are decided from
        // candidates (none when the route is off or no lane is eligible).
        let decision_histories = verifying
            .iter()
            .map(|&index| {
                let window = windows[index].as_ref().expect("verifying window");
                (window.decisions != DecisionRoute::HostLogits)
                    .then_some(states[index].sampling_history.as_slice())
            })
            .collect::<Vec<_>>();
        let verify = if decision_histories.iter().any(Option::is_some) {
            // Never together with GPU acceptance, which needs every lane
            // greedy; candidates are for lanes that are not.
            self.model.forward_gpu_dflash2_verify_with_decisions(
                &verify_lanes,
                gpu_target_tokens.then_some(&runtime.workspace.target_tokens),
                kv_cache,
                forward.mode,
                forward.dense_ffn,
                forward.options,
                Some(capture),
                &decision_histories,
            )
        } else if gpu_target_tokens {
            let mut output_tail =
                |encoder: &mut mlx_native::CommandEncoder,
                 registry: &mut mlx_native::KernelRegistry,
                 device: &mlx_native::MlxDevice,
                 target_tokens: &mlx_native::MlxBuffer| {
                    runtime.encode_gpu_acceptance_and_context(
                        encoder,
                        registry,
                        device,
                        target_tokens,
                        &target_lane_order,
                        &maximum_rows,
                        &self.eos_token_ids,
                    )
                };
            self.model.forward_gpu_dflash2_verify_from_gpu_tokens(
                &verify_lanes,
                &runtime.workspace.target_tokens,
                kv_cache,
                forward.mode,
                forward.dense_ffn,
                forward.options,
                Some(capture),
                gpu_acceptance.then_some(&mut output_tail),
            )
        } else {
            self.model.forward_gpu_dflash2_verify(
                &verify_lanes,
                kv_cache,
                forward.mode,
                forward.dense_ffn,
                forward.options,
                Some(capture),
            )
        };
        drop(decision_histories);
        let verify = lease.finish().and(verify);
        let verify_elapsed = verify_started.elapsed();
        let verify_micros = trace_timing.then(|| verify_elapsed.as_micros());
        timeline_mark("verify_end");
        if timeline_on {
            // Every command committed before the output head has completed:
            // a previous cycle that deferred its line can write it now.
            super::dflash2_timeline::settle_deferred();
        }
        // GPU acceptance has already committed the draft context when verify
        // returns. Any later failure must stop the worker rather than continue
        // with host and draft cursors that may differ.
        let Qwen35JointVerify {
            rows,
            hidden: _,
            pending,
        } = match verify {
            Ok(verify) => verify,
            Err(error) if gpu_acceptance || is_fatal(&error) => return Err(error),
            Err(error) => {
                distribute_error(&mut ticks, &verifying, error);
                return Ok(finish_ticks(ticks));
            }
        };
        drop(verify_lanes);
        if gpu_target_tokens {
            let proposals = runtime.proposals()?;
            let verify_rows = runtime.verify_rows()?;
            for &index in &verifying {
                let slot = states[index].slot_id.0 as usize;
                let proposal_start = slot * DFLASH2_PROPOSALS_PER_LANE;
                let proposal_end = proposal_start + DFLASH2_PROPOSALS_PER_LANE;
                let window = windows[index].as_ref().expect("verified window");
                tokens[index] = Some(
                    window
                        .tokens(
                            &proposals[proposal_start..proposal_end],
                            verify_rows[slot] as usize,
                        )
                        .with_context(|| {
                            format!("bind GPU-produced DFlash2 target tokens for slot {slot}")
                        })?,
                );
            }
        }
        let plan_started = trace_timing.then(Instant::now);
        timeline_mark("plan_begin");

        let mut accepted = Vec::with_capacity(verifying.len());
        let mut verified = Vec::with_capacity(verifying.len());
        let (mut candidate_rows, mut fallback_rows) = (0_usize, 0_usize);
        for ((&index, rows), lane_tokens) in verifying.iter().zip(rows).zip(
            verifying
                .iter()
                .map(|&index| tokens[index].take().expect("verified tokens")),
        ) {
            let step = states[index]
                .plan_selected_dflash2_window(
                    self,
                    windows[index].as_ref().expect("verified window"),
                    &lane_tokens,
                    rows,
                )
                .map(|plan| Verified::Window {
                    plan,
                    tokens: lane_tokens,
                })
                .unwrap_or_else(Verified::Failed);
            accepted.push(match &step {
                Verified::Window { plan, .. } => {
                    candidate_rows += plan.candidate_rows;
                    fallback_rows += plan.fallback_rows;
                    plan.accepted()
                }
                Verified::Failed(_) => 0,
            });
            verified.push(step);
        }
        if gpu_acceptance {
            let (gpu_retained, gpu_accepted) = runtime.gpu_acceptance_counts()?;
            for (&index, step) in verifying.iter().zip(&verified) {
                let slot = states[index].slot_id.0 as usize;
                let (host_retained, host_accepted) = match step {
                    Verified::Window { plan, .. } => (plan.accepted(), plan.plan.matched_drafts),
                    Verified::Failed(_) => (0, 0),
                };
                anyhow::ensure!(
                    gpu_retained[slot] as usize == host_retained
                        && gpu_accepted[slot] as usize == host_accepted,
                    "Qwen DFlash2 GPU acceptance mismatch for slot {slot}: \
                     GPU retained/accepted={}/{}, host={host_retained}/{host_accepted}",
                    gpu_retained[slot],
                    gpu_accepted[slot]
                );
            }
        }
        let plan_micros = plan_started.map(|started| started.elapsed().as_micros());

        let mut retained = vec![0_u32; physical_lanes];
        for (&index, &count) in verifying.iter().zip(&accepted) {
            retained[states[index].slot_id.0 as usize] = u32::try_from(count)?;
        }
        // Context first: the draft-context commit needs only the retained
        // counts, so it goes to the GPU before the replay is encoded. The
        // GPU, idle since the output head, starts it while the host encodes
        // the replay. The commit's host checks run first, so only a failing
        // replay command can leave a submitted context without its replay.
        let commit_valid = !gpu_acceptance
            && self
                .model
                .validate_joint_verify_commit(kv_cache, &pending, &accepted)
                .is_ok();
        let prefetch = commit_valid && forward.draft_prefetch;
        let context_first = commit_valid && !prefetch && context_first();
        // The verify-budget observation reads this step's draft confidences,
        // which a prefetched draft overwrites: it runs first when the step
        // prefetches, and after both commits otherwise.
        let mut budget_observed = false;
        if prefetch {
            self.observe_dflash2_verify_budget(
                runtime,
                states,
                &verifying,
                &verified,
                verify_elapsed,
            );
            budget_observed = true;
        }
        let next_draft = if prefetch {
            next_draft(runtime, states, &verifying, &verified, &retained)
        } else {
            None
        };
        timeline_mark("target_commit_begin");
        let target_commit_started = trace_timing.then(Instant::now);
        let lease = supervisor.arm("Qwen35 DFlash2 commit", QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
        // Waits this step makes that mlx-native's sync counter does not see
        // (it counts `commit_and_wait` only).
        let mut uncounted_waits = 0_u64;
        let (committed, pipelined_context) = if prefetch {
            // Prefetch: the context commit goes to the GPU first, reading
            // its own copy of the step's positions; the replay is committed
            // behind it (they share no resource, so they overlap); then the
            // next step's draft, whose ring reads Metal orders behind the
            // context commit's ring writes because the rings are tracked
            // buffers bound directly to both commands
            // (`submit_target_context_detached`). The step waits for its
            // replay and its context commit while the draft runs on. The
            // draft is committed last because a wait on a command returns
            // only once every command committed before it has completed:
            // behind the draft, the replay's wait would last until the
            // draft's end.
            let submitted = self.model.with_gpu_cache_mut(|device, registry| {
                runtime.submit_target_context_detached(device, registry, &retained)
            });
            timeline_mark("context_submitted");
            let step_lanes = verifying
                .iter()
                .map(|&index| states[index].slot_id.0 as usize)
                .collect::<Vec<_>>();
            // No early return from here to `settle_submitted_step`, which
            // follows the waits on both commands.
            let replay = match &submitted {
                Ok(_) => self
                    .model
                    .submit_joint_verify_commit(kv_cache, pending, &accepted),
                Err(_) => Err(anyhow::anyhow!(
                    "Qwen DFlash2 replay not submitted: the context commit was not"
                )),
            };
            uncounted_waits += u64::from(submitted.is_ok()) + u64::from(replay.is_ok());
            if let (Ok(_), Ok(_), Some(next)) = (&submitted, &replay, &next_draft) {
                let prefetched =
                    QwenDFlash2Adapter::new(&self.model, &runtime.config).and_then(|adapter| {
                        let temperatures = vec![0.0_f32; physical_lanes];
                        self.model.with_gpu_cache_mut(|device, registry| {
                            runtime.prefetch_draft(
                                device,
                                registry,
                                &adapter,
                                &next.anchors,
                                &next.positions,
                                &temperatures,
                                0,
                                &next.maximum_rows,
                                &next.lane_order,
                                &uniforms,
                                !next.gpu_target_tokens,
                            )
                        })
                    });
                timeline_mark("draft.prefetch_committed");
                if let Err(error) = prefetched {
                    // Not a failure of this step: the next one drafts itself.
                    tracing::warn!(
                        target: "hf2q::serve::api::engine_qwen35::progress",
                        error = %format!("{error:#}"),
                        "Qwen DFlash2 draft prefetch failed"
                    );
                }
            }
            let committed = replay
                .and_then(|submitted| self.model.finish_joint_verify_commit(kv_cache, submitted));
            let context = submitted
                .and_then(|submitted| runtime.finish_target_context(submitted, &retained, false));
            let context = runtime.settle_submitted_step(committed.is_ok(), context, &step_lanes);
            (committed, Some(context))
        } else if context_first {
            let submitted = self.model.with_gpu_cache_mut(|device, registry| {
                runtime.submit_target_context(device, registry, &retained)
            });
            timeline_mark("context_submitted");
            uncounted_waits += u64::from(submitted.is_ok());
            let step_lanes = verifying
                .iter()
                .map(|&index| states[index].slot_id.0 as usize)
                .collect::<Vec<_>>();
            // The replay and the context commit overlap on the GPU; they
            // share no resource, and `settle_context_first` waits for the
            // context command itself after the replay's wait, also when the
            // replay failed. No early return may come between the
            // submission above and that call.
            let (committed, context) =
                runtime.settle_context_first(submitted, &retained, &step_lanes, || {
                    self.model.commit_joint_verify(kv_cache, pending, &accepted)
                });
            (committed, Some(context))
        } else {
            (
                self.model.commit_joint_verify(kv_cache, pending, &accepted),
                None,
            )
        };
        let committed = lease.finish().and(committed);
        let target_commit_micros =
            target_commit_started.map(|started| started.elapsed().as_micros());
        timeline_mark("target_commit_end");
        timeline_mark("draft_commit_begin");
        let draft_commit_started = trace_timing.then(Instant::now);
        if let Err(error) = committed {
            if let Some(Err(cleanup)) = &pipelined_context {
                // The step failed after its context commit was submitted and
                // the draft rings could not be waited for or cleared: stop
                // the worker rather than draft from them.
                return Err(error.context(format!(
                    "Qwen DFlash2 draft rings left unsettled by a failed step: {cleanup:#}"
                )));
            }
            if gpu_acceptance || is_fatal(&error) {
                return Err(error);
            }
            distribute_error(&mut ticks, &verifying, error);
            return Ok(finish_ticks(ticks));
        }
        let context_committed = match pipelined_context {
            Some(result) => result,
            None => {
                let lease = supervisor.arm(
                    "Qwen35 DFlash2 target-context commit",
                    QWEN35_WORKER_TRANSACTION_TIMEOUT,
                )?;
                let context_committed = if gpu_acceptance {
                    runtime.settle_gpu_context_lengths(&retained)
                } else {
                    self.model.with_gpu_cache_mut(|device, registry| {
                        runtime.commit_target_context(device, registry, &retained)
                    })
                };
                lease.finish().and(context_committed)
            }
        };
        let draft_commit_micros = draft_commit_started.map(|started| started.elapsed().as_micros());
        timeline_mark("draft_commit_end");
        if let Err(error) = context_committed {
            if gpu_acceptance || is_fatal(&error) {
                return Err(error);
            }
            distribute_error(&mut ticks, &verifying, error);
            return Ok(finish_ticks(ticks));
        }
        if !budget_observed {
            self.observe_dflash2_verify_budget(
                runtime,
                states,
                &verifying,
                &verified,
                verify_elapsed,
            );
        }
        for ((&index, step), &count) in verifying.iter().zip(&verified).zip(&accepted) {
            if let Verified::Window { tokens, .. } = step {
                let slot = states[index].slot_id.0 as usize;
                runtime
                    .record_lane_tokens(slot, &tokens[..count])
                    .with_context(|| format!("bind DFlash2 slot {slot} retained token prefix"))?;
            }
        }
        if let (Some(started), Some(entry)) = (cycle_started, counters_at_entry) {
            let now = native_counters();
            tracing::debug!(
                target: "hf2q::serve::api::engine_qwen35::progress",
                active_lanes = verifying.len(),
                selected_rows = selected_rows.unwrap_or_default(),
                accepted_rows = accepted.iter().sum::<usize>(),
                gpu_acceptance,
                draft_us = draft_micros.unwrap_or_default(),
                target_verify_us = verify_micros.unwrap_or_default(),
                host_plan_us = plan_micros.unwrap_or_default(),
                target_commit_us = target_commit_micros.unwrap_or_default(),
                draft_commit_us = draft_commit_micros.unwrap_or_default(),
                total_us = started.elapsed().as_micros(),
                dispatches = now.0.wrapping_sub(entry.0),
                command_buffers = now.1.wrapping_sub(entry.1),
                host_waits = now.2.wrapping_sub(entry.2) + uncounted_waits,
                barriers = now.3.wrapping_sub(entry.3),
                host_cpu_us = now.4.wrapping_sub(entry.4),
                candidate_rows,
                fallback_rows,
                prefetched_drafts = runtime.prefetch_counts().prefetched,
                adopted_drafts = runtime.prefetch_counts().adopted,
                discarded_drafts = runtime.prefetch_counts().discarded,
                "Qwen DFlash2 stage timing"
            );
        }
        if let Some(mut timeline) = timeline.take() {
            if runtime.prefetch_outstanding() {
                timeline.defer_line();
            }
            timeline.finish(super::dflash2_timeline::CycleSummary {
                active_lanes: verifying.len(),
                selected_rows: selected_rows.unwrap_or_default(),
                accepted_rows: accepted.iter().sum::<usize>(),
                gpu_acceptance,
            });
        }

        for (&index, step) in verifying.iter().zip(verified) {
            ticks[index] = Some(match step {
                Verified::Window { plan, tokens } => {
                    let drafted = tokens.len().saturating_sub(1);
                    tracing::debug!(
                        target: "hf2q::serve::api::engine_qwen35::progress",
                        slot = states[index].slot_id.0,
                        drafted,
                        accepted = plan.accepted(),
                        "Qwen DFlash2 verified window"
                    );
                    states[index].settle_dflash2_window(plan);
                    states[index].joint_mtp_drain_tick(self)
                }
                Verified::Failed(error) => Err(error),
            });
        }
        Ok(finish_ticks(ticks))
    }
}

impl Qwen35LoadedModel {
    /// Feed a verified step to the MoE target's verify-row budget. Reads the
    /// step's draft confidences from the runtime's workspace.
    fn observe_dflash2_verify_budget(
        &self,
        runtime: &mut crate::inference::spec_decode::dflash2::runtime::DFlash2Runtime,
        states: &[&mut Qwen35DecodeState],
        verifying: &[usize],
        verified: &[Verified],
        verify_elapsed: std::time::Duration,
    ) {
        if self.model.cfg.moe.is_none() {
            return;
        }
        let mut observations = [VerifyBudgetLane::default(); 4];
        let mut observed = 0;
        for (&index, step) in verifying.iter().zip(verified) {
            if let Verified::Window { plan, tokens } = step {
                observations[observed] = VerifyBudgetLane {
                    lane_index: states[index].slot_id.0 as usize,
                    verify_rows: tokens.len(),
                    accepted: plan.plan.matched_drafts,
                    labels: true,
                };
                observed += 1;
            }
        }
        if observed == verifying.len() {
            if let Err(error) = runtime.observe_verify_budget(
                &observations[..observed],
                verify_elapsed.as_secs_f64() * 1000.0,
            ) {
                tracing::warn!(
                    target: "hf2q::serve::api::engine_qwen35::progress",
                    error = %error,
                    "Qwen DFlash2 verify-budget observation discarded"
                );
            }
        }
    }
}

/// The draft call of the step after one whose lanes `verifying` planned
/// `verified` and retain `retained` rows; `None` when a lane failed or no
/// lane goes on. A lane goes on from the last token of its plan unless the
/// plan ended on a terminal decision or the lane reached its length limit;
/// anything else that ends or changes a lane before the next step (a stop
/// string, a closed client, a new lane) makes that step's inputs differ
/// from these, and the runtime then drops the prefetched command.
fn next_draft(
    runtime: &crate::inference::spec_decode::dflash2::runtime::DFlash2Runtime,
    states: &[&mut Qwen35DecodeState],
    verifying: &[usize],
    verified: &[Verified],
    retained: &[u32],
) -> Option<NextDraft> {
    let physical_lanes = runtime.workspace.max_lanes;
    let mut next = NextDraft {
        anchors: vec![0; physical_lanes],
        positions: Vec::with_capacity(physical_lanes),
        maximum_rows: vec![1; physical_lanes],
        lane_order: Vec::with_capacity(verifying.len()),
        gpu_target_tokens: false,
    };
    for (&length, &retained) in runtime.lane_lengths().iter().zip(retained) {
        next.positions
            .push(u32::try_from(length.checked_add(retained as usize)?).ok()?);
    }
    for (&index, step) in verifying.iter().zip(verified) {
        let Verified::Window { plan, .. } = step else {
            return None;
        };
        let state = &states[index];
        let generated = state.generated_tokens.len() + plan.plan.output.len();
        let remaining = state.max_tokens.saturating_sub(generated);
        let (false, Some(&anchor), true) = (
            plan.plan.terminal_after_pending,
            plan.plan.output.back(),
            remaining > 0,
        ) else {
            continue;
        };
        let slot = state.slot_id.0;
        next.anchors[slot as usize] = anchor;
        next.maximum_rows[slot as usize] = remaining.min(DFLASH2_MAX_ROWS_PER_LANE) as u32;
        next.lane_order.push(slot);
    }
    if next.lane_order.is_empty() {
        return None;
    }
    next.gpu_target_tokens = runtime.gpu_target_tokens()
        && next
            .lane_order
            .iter()
            .all(|&slot| next.maximum_rows[slot as usize] == 8);
    Some(next)
}

fn distribute_error(
    ticks: &mut [Option<Result<Qwen35TickOutcome>>],
    indices: &[usize],
    error: anyhow::Error,
) {
    let message = format!("{error:#}");
    let mut first = Some(error);
    for &index in indices {
        ticks[index] = Some(Err(first
            .take()
            .unwrap_or_else(|| anyhow::anyhow!("{message}"))));
    }
}

fn finish_ticks(ticks: Vec<Option<Result<Qwen35TickOutcome>>>) -> Vec<Result<Qwen35TickOutcome>> {
    ticks
        .into_iter()
        .map(|tick| {
            tick.unwrap_or_else(|| Err(anyhow::anyhow!("Qwen DFlash2 lane produced no tick")))
        })
        .collect()
}

#[cfg(test)]
#[path = "dflash2_decision_tests.rs"]
mod decision_tests;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::super::speculation_qualification_tests::loaded_model;
    use super::*;
    use crate::serve::api::engine::SamplingParams;
    use crate::serve::api::qwen35_speculation::QwenSpeculationPolicy;
    use crate::serve::multi_seq_kv::SlotId;

    fn decode_state(qwen: &Qwen35LoadedModel, max_tokens: usize) -> Qwen35DecodeState {
        let prompt = vec![1, 2, 3, 4];
        let mut logits = vec![0.0_f32; qwen.vocab_size];
        logits[8] = 10.0;
        let mut state = Qwen35DecodeState::from_prefill_logits(
            qwen,
            prompt.clone(),
            SamplingParams {
                max_tokens,
                ..SamplingParams::default()
            },
            None,
            SlotId(0),
            0,
            &logits,
            Duration::ZERO,
            prompt.len(),
            None,
        )
        .expect("seed decode state");
        state.history_lookup = None;
        state
    }

    fn greedy(token: u32) -> Qwen35JointRowOutput {
        Qwen35JointRowOutput {
            greedy_token: Some(token),
            logits: None,
            nextn_hidden: None,
            decision: None,
        }
    }

    fn logits(token: usize, vocab: usize) -> Qwen35JointRowOutput {
        let mut values = vec![-10.0_f32; vocab];
        values[token] = 10.0;
        Qwen35JointRowOutput {
            greedy_token: None,
            logits: Some(values),
            nextn_hidden: None,
            decision: None,
        }
    }

    #[test]
    fn target_rejection_commits_matching_prefix_and_canonical_replacement() {
        let qwen = loaded_model(QwenSpeculationPolicy::Off);
        let state = decode_state(&qwen, 8);
        let window = Qwen35DFlash2Window {
            pending_token: 8,
            position: 4,
            greedy: true,
            maximum_rows: 4,
            decisions: DecisionRoute::HostLogits,
        };
        let plan = state
            .plan_dflash2_window(
                &qwen,
                &window,
                &[9, 10, 55],
                vec![greedy(9), greedy(10), greedy(11), greedy(12)],
            )
            .expect("plan DFlash2 rejection");
        assert_eq!(
            plan.plan.output.into_iter().collect::<Vec<_>>(),
            [9, 10, 11]
        );
        assert_eq!(plan.plan.matched_drafts, 2);
        assert_eq!(plan.plan.rejected_drafts, 1);
        assert_eq!(plan.plan.valid_input_tokens, 3);
    }

    #[test]
    fn full_acceptance_includes_target_bonus_and_all_input_rows() {
        let qwen = loaded_model(QwenSpeculationPolicy::Off);
        let state = decode_state(&qwen, 8);
        let window = Qwen35DFlash2Window {
            pending_token: 8,
            position: 4,
            greedy: true,
            maximum_rows: 4,
            decisions: DecisionRoute::HostLogits,
        };
        let plan = state
            .plan_dflash2_window(
                &qwen,
                &window,
                &[9, 10, 11],
                vec![greedy(9), greedy(10), greedy(11), greedy(12)],
            )
            .expect("plan DFlash2 full acceptance");
        assert_eq!(
            plan.plan.output.into_iter().collect::<Vec<_>>(),
            [9, 10, 11, 12]
        );
        assert_eq!(plan.plan.matched_drafts, 3);
        assert_eq!(plan.plan.rejected_drafts, 0);
        assert_eq!(plan.plan.valid_input_tokens, 4);
    }
    #[test]
    fn selected_gpu_budget_excludes_unverified_target_rows() {
        let qwen = loaded_model(QwenSpeculationPolicy::Off);
        let state = decode_state(&qwen, 8);
        let window = Qwen35DFlash2Window {
            pending_token: 8,
            position: 4,
            greedy: true,
            maximum_rows: 8,
            decisions: DecisionRoute::HostLogits,
        };
        let plan = state
            .plan_selected_dflash2_window(
                &qwen,
                &window,
                &[8, 9, 10],
                vec![
                    greedy(9),
                    greedy(10),
                    greedy(11),
                    greedy(99),
                    greedy(99),
                    greedy(99),
                    greedy(99),
                    greedy(99),
                ],
            )
            .expect("plan selected GPU DFlash2 rows");
        assert_eq!(
            plan.plan.output.into_iter().collect::<Vec<_>>(),
            [9, 10, 11]
        );
        assert_eq!(plan.plan.valid_input_tokens, 3);
    }

    #[test]
    fn gpu_acceptance_requires_transform_free_exact_greedy_semantics() {
        let qwen = loaded_model(QwenSpeculationPolicy::Off);
        let mut state = decode_state(&qwen, 8);
        let window = Qwen35DFlash2Window {
            pending_token: 8,
            position: 4,
            greedy: true,
            maximum_rows: 8,
            decisions: DecisionRoute::HostLogits,
        };
        assert!(state.gpu_greedy_acceptance_eligible(&window));

        state.params.repetition_penalty = 1.05;
        assert!(!state.gpu_greedy_acceptance_eligible(&window));

        state.params.repetition_penalty = 1.0;
        state.params.stop_strings.push("stop".to_owned());
        assert!(!state.gpu_greedy_acceptance_eligible(&window));
    }

    #[test]
    fn terminal_draft_is_not_streamed_or_retained() {
        let qwen = loaded_model(QwenSpeculationPolicy::Off);
        let state = decode_state(&qwen, 8);
        let window = Qwen35DFlash2Window {
            pending_token: 8,
            position: 4,
            greedy: true,
            maximum_rows: 4,
            decisions: DecisionRoute::HostLogits,
        };
        let plan = state
            .plan_dflash2_window(
                &qwen,
                &window,
                &[9, 63, 11],
                vec![greedy(9), greedy(63), greedy(11), greedy(12)],
            )
            .expect("plan DFlash2 terminal draft");
        assert_eq!(plan.plan.output.into_iter().collect::<Vec<_>>(), [9]);
        assert_eq!(plan.plan.matched_drafts, 2);
        assert_eq!(plan.plan.rejected_drafts, 0);
        assert_eq!(plan.plan.valid_input_tokens, 2);
        assert!(plan.plan.terminal_after_pending);
    }

    #[test]
    fn logprob_request_uses_full_logits_for_each_canonical_decision() {
        let qwen = loaded_model(QwenSpeculationPolicy::Off);
        let mut state = decode_state(&qwen, 8);
        state.want_logprobs = true;
        let window = Qwen35DFlash2Window {
            pending_token: 8,
            position: 4,
            greedy: false,
            maximum_rows: 4,
            decisions: DecisionRoute::HostLogits,
        };
        let plan = state
            .plan_dflash2_window(
                &qwen,
                &window,
                &[9, 10, 55],
                vec![
                    logits(9, qwen.vocab_size),
                    logits(10, qwen.vocab_size),
                    logits(11, qwen.vocab_size),
                    logits(12, qwen.vocab_size),
                ],
            )
            .expect("plan DFlash2 logprob rejection");
        assert_eq!(
            plan.plan.output.into_iter().collect::<Vec<_>>(),
            [9, 10, 11]
        );
        assert_eq!(plan.plan.valid_input_tokens, 3);
    }
}
