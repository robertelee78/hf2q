//! Slot-ledger half of the joint MTP verify route (`HF2Q_QWEN_JOINT_MTP`,
//! design H3/H4).
//!
//! In a joint MTP step every lane of the group contributes one window to a
//! single joint verify forward (`forward_gpu_joint_verify`):
//!
//! - a lane with MTP state contributes its pending token plus the drafts
//!   of one batched MTP forward per depth across all drafting lanes
//!   ([`Qwen35LoadedModel::draft_joint_mtp`]), at the cap8 depths
//!   ([`cap8_draft_depths`]);
//! - a lane without MTP state contributes the spec-off ordinary row and
//!   commits it through the joint decode step's own path
//!   (`prepare_joint_tick` / `finish_joint_tick`).
//!
//! Every verify row is bitwise the spec-off row at its position, so the
//! window's canonical decisions are the spec-off decisions as long as the
//! decision function is spec-off's: a forced thinking-budget token first,
//! then the GPU argmax for greedy requests without logprobs, else the
//! constrained host sampler over the row's logits, each against the state
//! after the earlier rows ([`Qwen35DecodeState::plan_joint_mtp_window`]).
//! The accept walk is the K3 one (`plan_qwen35_verified_block`): drafts are
//! accepted while they equal the previous row's decision, and the first
//! mismatch (or the bonus row) commits the canonical token. Stops (EOS,
//! grammar end) inside the run end it; the window never holds more rows
//! than the request has tokens left, so the length limit cannot be passed.
//!
//! After the shared commit (replay of the DeltaNet prefix, cursor
//! truncation) each window lane catches its MTP cache up over its accepted
//! rows, carries the last accepted verifier hidden row and queues the
//! accepted tokens; the scheduler drains them all within the same turn
//! ([`Qwen35DecodeState::joint_mtp_drain_tick`]).

use anyhow::{Context, Result};
use mlx_native::MlxBuffer;

use crate::inference::models::qwen35::joint_decode_core::rows_view;
use crate::inference::models::qwen35::joint_decode_routes::{
    JointDenseFfnRoute, JointRouteMode, MAX_JOINT_DECODE_ROWS,
};
use crate::inference::models::qwen35::joint_step_options::JointStepOptions;
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::mtp::MtpDraftLane;
use crate::inference::models::qwen35::spec_decode::{nth_hidden_row, positions_for_range};
use crate::inference::models::qwen35::{
    Qwen35JointOutput, Qwen35JointRowOutput, Qwen35JointVerify, Qwen35JointVerifyLane,
    Qwen35JointVerifyPending,
};
use crate::serve::api::engine_supervisor::EngineSupervisor;
use crate::serve::api::qwen35_speculation::{record_proposer_outcome, QwenSpeculationProposer};
use crate::serve::multi_seq_kv::SlotId;

use super::{
    mtp_cursor_for_slot, plan_qwen35_verified_block, take_pending_speculation_output,
    Qwen35CanonicalDecision, Qwen35DecisionSource, Qwen35DecodeState, Qwen35JointLane,
    Qwen35LoadedModel, Qwen35SpecSemanticState, Qwen35TickOutcome, Qwen35VerifiedBlockPlan,
    QWEN35_WORKER_TRANSACTION_TIMEOUT,
};

/// Row budget of one joint MTP step (cap8): every projection pass stays on
/// the `m <= 8` route.
pub(in crate::serve::api) const JOINT_MTP_MAX_ROWS: usize = MAX_JOINT_DECODE_ROWS;
/// Deepest window: the pending token plus three drafts.
pub(in crate::serve::api) const JOINT_MTP_MAX_DRAFTS: usize = 3;

/// The cap8 draft depths of one step. `caps[i]` is lane `i`'s largest
/// admissible draft count (0 for a lane that cannot draft). Every lane
/// holds one row; the remaining budget of [`JOINT_MTP_MAX_ROWS`] rows is
/// granted one draft at a time, round-robin in lane order, up to each cap.
/// With every lane able to draft three: 4 lanes → 1 each, 3 → 2/2/1,
/// 2 → 3/3, 1 → 3.
pub(in crate::serve::api) fn cap8_draft_depths(caps: &[usize]) -> Vec<usize> {
    let mut depths = vec![0; caps.len()];
    let mut budget = JOINT_MTP_MAX_ROWS.saturating_sub(caps.len());
    while budget > 0 {
        let mut granted = false;
        for (depth, &cap) in depths.iter_mut().zip(caps) {
            if budget > 0 && *depth < cap.min(JOINT_MTP_MAX_DRAFTS) {
                *depth += 1;
                budget -= 1;
                granted = true;
            }
        }
        if !granted {
            break;
        }
    }
    depths
}

/// Accept walk over one window: [`plan_qwen35_verified_block`], plus the
/// draft-free window, whose only row is the bonus.
fn plan_qwen35_window(
    drafts: &[u32],
    mut canonical_at: impl FnMut(usize) -> Result<Qwen35CanonicalDecision>,
) -> Result<Qwen35VerifiedBlockPlan> {
    if !drafts.is_empty() {
        return plan_qwen35_verified_block(drafts, canonical_at);
    }
    let decision = canonical_at(0)?;
    Ok(Qwen35VerifiedBlockPlan {
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
    })
}

/// Entry of a lane with MTP state: its window before drafting.
pub(in crate::serve::api) struct Qwen35JointMtpWindow {
    pending_token: u32,
    position: i32,
    /// Target and MTP cursor at entry (equal).
    prior: u32,
    /// The carry row the MTP catch-up shifts in front of the window.
    verifier_hidden: MlxBuffer,
    /// Decision source: GPU argmax (greedy, no logprobs) or host sampler.
    greedy: bool,
    /// Largest admissible draft count.
    draft_cap: usize,
}

/// A lane's role in a joint MTP step.
pub(in crate::serve::api) enum Qwen35JointMtpRole {
    /// No MTP state: the spec-off ordinary row, committed through the joint
    /// decode step's path.
    Ordinary(Qwen35JointLane),
    /// MTP state: a window of the pending token and its drafts.
    Window(Qwen35JointMtpWindow),
}

impl Qwen35JointMtpRole {
    /// Largest draft count the cap8 allocator may grant this lane.
    pub(in crate::serve::api) fn draft_cap(&self) -> usize {
        match self {
            Self::Ordinary(_) => 0,
            Self::Window(window) => window.draft_cap,
        }
    }
}

/// A verified window's accept plan, between the accept walk and the shared
/// commit.
pub(in crate::serve::api) struct Qwen35JointMtpPlan {
    plan: Qwen35VerifiedBlockPlan,
}

impl Qwen35JointMtpPlan {
    /// Verify rows the window keeps (its valid input tokens).
    pub(in crate::serve::api) fn accepted(&self) -> usize {
        self.plan.valid_input_tokens
    }
}

/// One drafting lane of [`Qwen35LoadedModel::draft_joint_mtp`].
pub(in crate::serve::api) struct Qwen35JointMtpDraftRequest<'a> {
    pub slot: SlotId,
    pub token: u32,
    pub position: i32,
    pub hidden: &'a MlxBuffer,
    pub depth: usize,
    pub prior_mtp: u32,
}

impl Qwen35JointMtpWindow {
    pub(in crate::serve::api) fn draft_request(
        &self,
        slot: SlotId,
        depth: usize,
    ) -> Qwen35JointMtpDraftRequest<'_> {
        Qwen35JointMtpDraftRequest {
            slot,
            token: self.pending_token,
            position: self.position,
            hidden: &self.verifier_hidden,
            depth,
            prior_mtp: self.prior,
        }
    }

    /// The verify row output this window asks for.
    pub(in crate::serve::api) fn output(&self) -> Qwen35JointOutput {
        if self.greedy {
            Qwen35JointOutput::GreedyTokenAndHidden
        } else {
            Qwen35JointOutput::LogitsAndHidden
        }
    }

    /// The pending token, then the drafts, in row order.
    pub(in crate::serve::api) fn tokens(&self, drafts: &[u32]) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(drafts.len() + 1);
        tokens.push(self.pending_token);
        tokens.extend_from_slice(drafts);
        tokens
    }
}

impl Qwen35DecodeState {
    /// Prepare this lane's share of a joint MTP step. A lane with MTP state
    /// becomes a window (entry cursors checked equal); any other lane takes
    /// the joint decode step's ordinary preparation unchanged.
    pub(in crate::serve::api) fn prepare_joint_mtp(
        &mut self,
        kv_cache: &HybridKvCache,
    ) -> Result<Qwen35JointMtpRole> {
        let Some(mtp) = self.mtp.as_ref() else {
            return Ok(Qwen35JointMtpRole::Ordinary(self.prepare_joint_tick(kv_cache)?));
        };
        let pending_token = *self
            .generated_tokens
            .last()
            .context("Qwen joint MTP window needs a seeded output token")?;
        let position = (self.decode_position_base + self.generated_tokens.len() - 1) as i32;
        let prior = kv_cache.sequence_len_for_slot(self.slot_id)?;
        let prior_mtp = mtp_cursor_for_slot(kv_cache, self.slot_id)?;
        anyhow::ensure!(
            prior == prior_mtp,
            "Qwen joint MTP window requires equal entry cursors (target={prior}, mtp={prior_mtp})"
        );
        let remaining = self.max_tokens.saturating_sub(self.generated_tokens.len());
        anyhow::ensure!(remaining > 0, "Qwen joint MTP window past the length limit");
        Ok(Qwen35JointMtpRole::Window(Qwen35JointMtpWindow {
            pending_token,
            position,
            prior,
            verifier_hidden: mtp.verifier_hidden.clone(),
            greedy: self.is_greedy && !self.want_logprobs,
            // A window commits at most one token per row.
            draft_cap: (remaining - 1).min(JOINT_MTP_MAX_DRAFTS),
        }))
    }

    /// Accept walk over a verified window: row `r`'s canonical decision is
    /// spec-off's decision function on row `r` against the state after the
    /// earlier rows' decisions. Mutates nothing.
    pub(in crate::serve::api) fn plan_joint_mtp_window(
        &self,
        qwen: &Qwen35LoadedModel,
        window: &Qwen35JointMtpWindow,
        drafts: &[u32],
        mut rows: Vec<Qwen35JointRowOutput>,
    ) -> Result<Qwen35JointMtpPlan> {
        anyhow::ensure!(
            rows.len() == drafts.len() + 1,
            "Qwen joint MTP window verified {} rows for {} drafts",
            rows.len(),
            drafts.len()
        );
        let mut semantic = Qwen35SpecSemanticState::from_decode(self);
        plan_qwen35_window(drafts, |row| {
            let output = &mut rows[row];
            let source = if window.greedy {
                Qwen35DecisionSource::Argmax(
                    output
                        .greedy_token
                        .context("Qwen joint MTP greedy row returned no argmax")?,
                )
            } else {
                Qwen35DecisionSource::Logits(
                    output
                        .logits
                        .as_deref_mut()
                        .context("Qwen joint MTP row returned no logits")?,
                )
            };
            semantic.select_and_observe_from(qwen, &self.params, source)
        })
        .map(|plan| Qwen35JointMtpPlan { plan })
    }

    /// After the shared commit kept `plan.valid_input_tokens` rows of this
    /// window: catch the MTP cache up over them (target and MTP cursors end
    /// equal), carry the last accepted verifier hidden row and queue the
    /// accepted output for the in-turn drain. On error the caller resets
    /// the slot.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::serve::api) fn settle_joint_mtp_window(
        &mut self,
        qwen: &Qwen35LoadedModel,
        kv_cache: &mut HybridKvCache,
        supervisor: &EngineSupervisor,
        window: Qwen35JointMtpWindow,
        tokens: &[u32],
        hidden_rows: &MlxBuffer,
        plan: Qwen35JointMtpPlan,
    ) -> Result<()> {
        let Qwen35JointMtpPlan { plan } = plan;
        let accepted = plan.valid_input_tokens;
        anyhow::ensure!(
            (1..=tokens.len()).contains(&accepted),
            "Qwen joint MTP window accepted {accepted} of {} rows",
            tokens.len()
        );
        let hidden_size = qwen.model.cfg.hidden_size;
        let mtp = qwen
            .model
            .mtp
            .as_ref()
            .context("Qwen joint MTP catch-up weights missing")?;
        let accepted_tokens = &tokens[..accepted];
        let shared_embed = qwen
            .model
            .embed_tokens_gpu(accepted_tokens)
            .context("Qwen joint MTP catch-up embeddings")?;
        let target_rows = rows_view(hidden_rows, 0, accepted, hidden_size as usize)?;
        let lease = supervisor.arm(
            "Qwen35 joint MTP catch-up",
            QWEN35_WORKER_TRANSACTION_TIMEOUT,
        )?;
        let caught_up = qwen.model.with_gpu_cache_mut(|device, registry| {
            mtp.process_target_batch(
                accepted_tokens,
                Some(&window.verifier_hidden),
                &target_rows,
                &shared_embed,
                kv_cache,
                self.slot_id,
                &positions_for_range(window.position, accepted),
                device,
                registry,
                &qwen.model.cfg,
            )
        });
        lease
            .finish()
            .and(caught_up)
            .context("Qwen joint MTP catch-up")?;
        kv_cache
            .validate_speculative_cursors_for_slot(self.slot_id, window.prior as usize + accepted)
            .context("Qwen joint MTP committed cursor equality")?;
        let carry = nth_hidden_row(hidden_rows, hidden_size, plan.carry_hidden_row as u64)?;
        self.mtp
            .as_mut()
            .context("Qwen joint MTP state vanished")?
            .verifier_hidden = carry;
        let drafted = tokens.len() - 1;
        if drafted > 0 {
            record_proposer_outcome(
                QwenSpeculationProposer::Mtp,
                drafted,
                plan.matched_drafts,
                plan.rejected_drafts,
                1,
                0,
            );
        }
        self.pending_speculation_output = plan.output;
        self.terminal_after_pending = plan.terminal_after_pending;
        Ok(())
    }

    /// Whether verified output is still queued for this turn's drain.
    pub(in crate::serve::api) fn has_pending_speculation_output(&self) -> bool {
        !self.pending_speculation_output.is_empty()
    }

    /// Emit one queued verified token (the joint MTP in-turn drain), or end
    /// the request when the window's run ended on a stop with nothing left.
    pub(in crate::serve::api) fn joint_mtp_drain_tick(
        &mut self,
        qwen: &Qwen35LoadedModel,
    ) -> Result<Qwen35TickOutcome> {
        match take_pending_speculation_output(&mut self.pending_speculation_output) {
            Some(token) => self.commit_speculation_output(qwen, token),
            None => {
                anyhow::ensure!(
                    self.terminal_after_pending,
                    "Qwen joint MTP drain without verified output"
                );
                self.finish_reason = "stop";
                Ok(Qwen35TickOutcome {
                    fragment: String::new(),
                    is_reasoning: false,
                    finished: true,
                })
            }
        }
    }
}

impl Qwen35LoadedModel {
    /// Batched MTP drafting for one joint MTP step (H3): one MTP forward per
    /// draft depth over every lane still drafting at that depth. Each
    /// lane's MTP cursor is restored to its entry value afterwards, whether
    /// drafting succeeded or not. Returns each lane's drafts.
    pub(in crate::serve::api) fn draft_joint_mtp(
        &self,
        kv_cache: &mut HybridKvCache,
        lanes: &[Qwen35JointMtpDraftRequest<'_>],
        supervisor: &EngineSupervisor,
    ) -> Result<Vec<Vec<u32>>> {
        let max_depth = lanes.iter().map(|lane| lane.depth).max().unwrap_or(0);
        if max_depth == 0 {
            return Ok(vec![Vec::new(); lanes.len()]);
        }
        let mtp = self
            .model
            .mtp
            .as_ref()
            .context("Qwen joint MTP draft weights missing")?;
        let lease = supervisor.arm("Qwen35 joint MTP draft", QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
        let drafted = self.model.with_gpu_cache_mut(|device, registry| {
            let mut drafts = vec![Vec::with_capacity(max_depth); lanes.len()];
            let mut hidden: Vec<Option<MlxBuffer>> = vec![None; lanes.len()];
            let mut tokens = lanes.iter().map(|lane| lane.token).collect::<Vec<_>>();
            for depth in 0..max_depth {
                let active = (0..lanes.len())
                    .filter(|&index| lanes[index].depth > depth)
                    .collect::<Vec<_>>();
                let active_tokens = active.iter().map(|&index| tokens[index]).collect::<Vec<_>>();
                let shared_embed =
                    self.model
                        .embed_tokens_gpu_in_context(&active_tokens, device, registry)?;
                let rows = active
                    .iter()
                    .map(|&index| MtpDraftLane {
                        slot_id: lanes[index].slot,
                        prev_hidden: hidden[index].as_ref().unwrap_or(lanes[index].hidden),
                        token: tokens[index],
                        position: lanes[index].position + depth as i32,
                    })
                    .collect::<Vec<_>>();
                let outputs = mtp.forward_draft_greedy_batch(
                    &rows,
                    &shared_embed,
                    kv_cache,
                    device,
                    registry,
                    &self.model.cfg,
                )?;
                drop(rows);
                for (&index, (token, next_hidden)) in active.iter().zip(outputs) {
                    drafts[index].push(token);
                    tokens[index] = token;
                    hidden[index] = Some(next_hidden);
                }
            }
            Ok(drafts)
        });
        let supervision = lease.finish();
        let mut restore = Ok(());
        for lane in lanes {
            if let Err(error) = kv_cache.truncate_mtp_to_for_slot(lane.slot, lane.prior_mtp) {
                restore = Err(error.context(format!(
                    "Qwen joint MTP discard draft cache slot {}",
                    lane.slot.0
                )));
            }
        }
        supervision.context("Qwen joint MTP draft supervision")?;
        let drafts = drafted.context("Qwen joint MTP draft")?;
        restore?;
        Ok(drafts)
    }

    /// One supervised joint verify forward over the lanes' windows.
    pub(in crate::serve::api) fn forward_joint_verify(
        &self,
        lanes: &[Qwen35JointVerifyLane<'_>],
        kv_cache: &mut HybridKvCache,
        mode: JointRouteMode,
        dense_ffn: JointDenseFfnRoute,
        options: JointStepOptions,
        supervisor: &EngineSupervisor,
    ) -> Result<Qwen35JointVerify> {
        let lease = supervisor.arm("Qwen35 joint MTP verify", QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
        let verify = self
            .model
            .forward_gpu_joint_verify(lanes, kv_cache, mode, dense_ffn, options);
        lease.finish()?;
        verify.with_context(|| {
            format!(
                "Qwen35Model::forward_gpu_joint_verify ({} lanes, {} mode, {} dense FFN, {options:?})",
                lanes.len(),
                mode.label(),
                dense_ffn.label()
            )
        })
    }

    /// The shared commit of a joint verify step (DeltaNet replay, cursor
    /// truncation). On error the caller resets every lane of the step.
    pub(in crate::serve::api) fn commit_joint_verify(
        &self,
        kv_cache: &mut HybridKvCache,
        pending: Qwen35JointVerifyPending,
        accepted: &[usize],
        supervisor: &EngineSupervisor,
    ) -> Result<()> {
        let lease = supervisor.arm("Qwen35 joint MTP commit", QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
        let committed = self.model.commit_joint_verify(kv_cache, pending, accepted);
        lease.finish()?;
        committed.context("Qwen joint MTP commit")
    }
}

/// The forward choices of a joint MTP step (the worker's joint config).
#[derive(Clone, Copy, Debug)]
pub(in crate::serve::api) struct Qwen35JointMtpForward {
    pub mode: JointRouteMode,
    pub dense_ffn: JointDenseFfnRoute,
    pub options: JointStepOptions,
}

/// The drafter of a joint MTP step: production calls
/// [`Qwen35LoadedModel::draft_joint_mtp`]; tests force draft outcomes.
pub(in crate::serve::api) type Qwen35JointMtpDrafter<'a> = dyn FnMut(
        &Qwen35LoadedModel,
        &mut HybridKvCache,
        &[Qwen35JointMtpDraftRequest<'_>],
    ) -> Result<Vec<Vec<u32>>>
    + 'a;

/// A verified lane between its accept walk and its commit.
enum Verified {
    Ordinary(Qwen35JointRowOutput),
    Window {
        plan: Qwen35JointMtpPlan,
        tokens: Vec<u32>,
    },
    Failed(anyhow::Error),
}

/// One joint MTP step over `states` (distinct slots, at most
/// [`JOINT_MTP_MAX_ROWS`] lanes): roles, cap8 depths, batched drafting,
/// one joint verify forward, the accept walks, the shared commit and each
/// lane's settlement. Returns every lane's tick (a window lane's first
/// verified token; the rest stay queued for
/// [`Qwen35DecodeState::joint_mtp_drain_tick`]).
///
/// Lane-local failures (preparation, accept walk, catch-up) fail only that
/// lane; a failed draft fails the drafting lanes; a failed verify or commit
/// fails every lane that submitted it. A failure `is_fatal` accepts (an
/// unhealthy supervisor, a fatal command buffer) is returned as `Err`: the
/// worker fail-stops.
pub(in crate::serve::api) fn run_joint_mtp_step(
    qwen: &Qwen35LoadedModel,
    kv_cache: &mut HybridKvCache,
    states: &mut [&mut Qwen35DecodeState],
    forward: Qwen35JointMtpForward,
    supervisor: &EngineSupervisor,
    drafter: &mut Qwen35JointMtpDrafter<'_>,
    is_fatal: &dyn Fn(&anyhow::Error) -> bool,
) -> Result<Vec<Result<Qwen35TickOutcome>>> {
    let lanes = states.len();
    let mut ticks: Vec<Option<Result<Qwen35TickOutcome>>> = (0..lanes).map(|_| None).collect();
    let fail_all = |ticks: &mut Vec<Option<Result<Qwen35TickOutcome>>>,
                    indices: &[usize],
                    error: anyhow::Error| {
        let message = format!("{error:#}");
        let mut first = Some(error);
        for &index in indices {
            ticks[index] = Some(Err(first
                .take()
                .unwrap_or_else(|| anyhow::anyhow!("{message}"))));
        }
    };

    // 1. Roles, as each lane's own tick would see its state.
    let mut roles: Vec<Option<Qwen35JointMtpRole>> = Vec::with_capacity(lanes);
    for (index, state) in states.iter_mut().enumerate() {
        match state.prepare_joint_mtp(kv_cache) {
            Ok(role) => roles.push(Some(role)),
            Err(error) => {
                ticks[index] = Some(Err(error));
                roles.push(None);
            }
        }
    }

    // 2. cap8 depths and batched drafting.
    let caps = roles
        .iter()
        .map(|role| role.as_ref().map_or(0, Qwen35JointMtpRole::draft_cap))
        .collect::<Vec<_>>();
    let depths = cap8_draft_depths(&caps);
    let mut drafts = vec![Vec::new(); lanes];
    let drafting = (0..lanes)
        .filter(|&index| depths[index] > 0)
        .collect::<Vec<_>>();
    if !drafting.is_empty() {
        let requests = drafting
            .iter()
            .filter_map(|&index| match roles[index].as_ref() {
                Some(Qwen35JointMtpRole::Window(window)) => {
                    Some(window.draft_request(states[index].slot_id, depths[index]))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let drafted = drafter(qwen, kv_cache, &requests).and_then(|outputs| {
            anyhow::ensure!(
                outputs.len() == requests.len()
                    && outputs
                        .iter()
                        .zip(&requests)
                        .all(|(drafts, request)| drafts.len() == request.depth),
                "Qwen joint MTP drafter returned the wrong draft counts"
            );
            Ok(outputs)
        });
        drop(requests);
        match drafted {
            Ok(outputs) => {
                for (&index, output) in drafting.iter().zip(outputs) {
                    drafts[index] = output;
                }
            }
            Err(error) if is_fatal(&error) => return Err(error),
            Err(error) => {
                for &index in &drafting {
                    roles[index] = None;
                }
                fail_all(&mut ticks, &drafting, error);
            }
        }
    }

    // 3. One joint verify forward over every window.
    let verifying = (0..lanes)
        .filter(|&index| roles[index].is_some())
        .collect::<Vec<_>>();
    if !verifying.is_empty() {
        let tokens = verifying
            .iter()
            .map(|&index| match roles[index].as_ref() {
                Some(Qwen35JointMtpRole::Window(window)) => window.tokens(&drafts[index]),
                Some(Qwen35JointMtpRole::Ordinary(lane)) => vec![lane.row.token],
                None => unreachable!("verifying lanes hold a role"),
            })
            .collect::<Vec<_>>();
        let verify_lanes = verifying
            .iter()
            .zip(&tokens)
            .map(|(&index, tokens)| match roles[index].as_ref() {
                Some(Qwen35JointMtpRole::Window(window)) => Qwen35JointVerifyLane {
                    slot_id: states[index].slot_id,
                    position: window.position,
                    tokens,
                    output: window.output(),
                },
                Some(Qwen35JointMtpRole::Ordinary(lane)) => Qwen35JointVerifyLane {
                    slot_id: lane.row.slot_id,
                    position: lane.row.position,
                    tokens,
                    output: lane.row.output,
                },
                None => unreachable!("verifying lanes hold a role"),
            })
            .collect::<Vec<_>>();
        let verify = qwen.forward_joint_verify(
            &verify_lanes,
            kv_cache,
            forward.mode,
            forward.dense_ffn,
            forward.options,
            supervisor,
        );
        drop(verify_lanes);
        let Qwen35JointVerify {
            rows,
            hidden,
            pending,
        } = match verify {
            Ok(verify) => verify,
            Err(error) if is_fatal(&error) => return Err(error),
            Err(error) => {
                fail_all(&mut ticks, &verifying, error);
                return Ok(finish_ticks(ticks));
            }
        };

        // 4. Accept walks with spec-off's decision function.
        let mut accepted = Vec::with_capacity(verifying.len());
        let mut verified = Vec::with_capacity(verifying.len());
        for ((&index, lane_rows), tokens) in verifying.iter().zip(rows).zip(tokens) {
            let step = match roles[index].as_ref() {
                Some(Qwen35JointMtpRole::Ordinary(_)) => match lane_rows.into_iter().next() {
                    Some(output) => Verified::Ordinary(output),
                    None => Verified::Failed(anyhow::anyhow!(
                        "Qwen joint MTP ordinary row returned no output"
                    )),
                },
                Some(Qwen35JointMtpRole::Window(window)) => {
                    match states[index].plan_joint_mtp_window(
                        qwen,
                        window,
                        &drafts[index],
                        lane_rows,
                    ) {
                        Ok(plan) => Verified::Window { plan, tokens },
                        Err(error) => Verified::Failed(error),
                    }
                }
                None => unreachable!("verifying lanes hold a role"),
            };
            // An ordinary row is the spec-off step itself; a failed walk
            // discards its window (the lane is reset at settlement).
            accepted.push(match &step {
                Verified::Ordinary(_) => 1,
                Verified::Window { plan, .. } => plan.accepted(),
                Verified::Failed(_) => 0,
            });
            verified.push(step);
        }

        // 5. Shared commit, then each lane's settlement.
        match qwen.commit_joint_verify(kv_cache, pending, &accepted, supervisor) {
            Ok(()) => {
                for ((&index, step), hidden_rows) in verifying.iter().zip(verified).zip(&hidden) {
                    let role = roles[index].take().expect("verifying lanes hold a role");
                    let state = &mut *states[index];
                    ticks[index] = Some(match (role, step) {
                        (Qwen35JointMtpRole::Ordinary(lane), Verified::Ordinary(output)) => state
                            .finish_joint_tick(qwen, kv_cache, supervisor, lane.target, output),
                        (Qwen35JointMtpRole::Window(window), Verified::Window { plan, tokens }) => {
                            state
                                .settle_joint_mtp_window(
                                    qwen, kv_cache, supervisor, window, &tokens, hidden_rows,
                                    plan,
                                )
                                .and_then(|()| state.joint_mtp_drain_tick(qwen))
                        }
                        (_, Verified::Failed(error)) => Err(error),
                        _ => Err(anyhow::anyhow!("Qwen joint MTP lane role changed")),
                    });
                }
            }
            Err(error) if is_fatal(&error) => return Err(error),
            Err(error) => fail_all(&mut ticks, &verifying, error),
        }
    }
    Ok(finish_ticks(ticks))
}

fn finish_ticks(
    ticks: Vec<Option<Result<Qwen35TickOutcome>>>,
) -> Vec<Result<Qwen35TickOutcome>> {
    ticks
        .into_iter()
        .map(|tick| tick.unwrap_or_else(|| Err(anyhow::anyhow!("Qwen joint MTP lane produced no tick"))))
        .collect()
}

#[cfg(test)]
#[path = "joint_mtp_tests.rs"]
mod tests;
