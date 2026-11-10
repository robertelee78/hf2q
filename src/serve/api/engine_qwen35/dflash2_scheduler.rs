//! Qwen DFlash2 serving transaction.
//!
//! One step drafts anchor-plus-seven proposal paths for every active physical
//! slot, verifies all selected rows in one fixed width-32 target tile, commits
//! only each lane's accepted prefix, and queues semantic output for the same
//! in-turn drain used by joint MTP. DFlash2 is optional and fail-closed: a lane
//! joins only when its target cursor and persistent draft-ring cursor agree.

use anyhow::{Context, Result};

use crate::inference::models::qwen35::joint_decode_routes::{JointDenseFfnRoute, JointRouteMode};
use crate::inference::models::qwen35::joint_step_options::JointStepOptions;
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::{
    Qwen35JointOutput, Qwen35JointRowOutput, Qwen35JointVerify, Qwen35JointVerifyLane,
};
use crate::inference::spec_decode::dflash2::adapters::QwenDFlash2Adapter;
use crate::serve::api::engine_supervisor::EngineSupervisor;

use super::{
    plan_qwen35_verified_block, Qwen35CanonicalDecision, Qwen35DecisionSource, Qwen35DecodeState,
    Qwen35LoadedModel, Qwen35SpecSemanticState, Qwen35TickOutcome, Qwen35VerifiedBlockPlan,
    QWEN35_WORKER_TRANSACTION_TIMEOUT,
};

const DFLASH2_MAX_ROWS_PER_LANE: usize = 8;
const DFLASH2_PROPOSALS_PER_LANE: usize = DFLASH2_MAX_ROWS_PER_LANE - 1;

#[derive(Clone, Copy, Debug)]
pub(in crate::serve::api) struct Qwen35DFlash2Forward {
    pub mode: JointRouteMode,
    pub dense_ffn: JointDenseFfnRoute,
    pub options: JointStepOptions,
}

struct Qwen35DFlash2Window {
    pending_token: u32,
    position: i32,
    greedy: bool,
    maximum_rows: usize,
}

impl Qwen35DFlash2Window {
    fn output(&self) -> Qwen35JointOutput {
        if self.greedy {
            Qwen35JointOutput::GreedyTokenAndHidden
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
        })
    }

    fn plan_dflash2_window(
        &self,
        qwen: &Qwen35LoadedModel,
        window: &Qwen35DFlash2Window,
        drafts: &[u32],
        mut rows: Vec<Qwen35JointRowOutput>,
    ) -> Result<Qwen35DFlash2Plan> {
        anyhow::ensure!(
            rows.len() == drafts.len() + 1,
            "Qwen DFlash2 window verified {} rows for {} drafts",
            rows.len(),
            drafts.len()
        );
        let mut semantic = Qwen35SpecSemanticState::from_decode(self);
        let mut canonical_at = |row: usize| {
            let output = &mut rows[row];
            let source = if window.greedy {
                Qwen35DecisionSource::Argmax(
                    output
                        .greedy_token
                        .context("Qwen DFlash2 greedy row returned no argmax")?,
                )
            } else {
                Qwen35DecisionSource::Logits(
                    output
                        .logits
                        .as_deref_mut()
                        .context("Qwen DFlash2 row returned no logits")?,
                )
            };
            semantic.select_and_observe_from(qwen, &self.params, source)
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
            plan_qwen35_verified_block(drafts, canonical_at)?
        };
        Ok(Qwen35DFlash2Plan { plan })
    }

    fn settle_dflash2_window(&mut self, plan: Qwen35DFlash2Plan) {
        let Qwen35DFlash2Plan { plan } = plan;
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
                Ok(window) => windows.push(Some(window)),
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
        let adapter_config = runtime.config.clone();
        let adapter = QwenDFlash2Adapter::new(&self.model, &adapter_config)?;
        let lease = supervisor.arm("Qwen35 DFlash2 draft", QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
        let drafted = self.model.with_gpu_cache_mut(|device, registry| {
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
        });
        let drafted = lease.finish().and(drafted);
        if let Err(error) = drafted {
            if is_fatal(&error) {
                return Err(error);
            }
            distribute_error(&mut ticks, &active, error);
            return Ok(finish_ticks(ticks));
        }
        let proposals = runtime.proposals()?.to_vec();
        let verify_rows = runtime.verify_rows()?.to_vec();
        let mut tokens: Vec<Option<Vec<u32>>> = (0..lanes).map(|_| None).collect();
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
        let adapter_config = runtime.config.clone();
        let adapter = QwenDFlash2Adapter::new(&self.model, &adapter_config)?;
        let capture = runtime.qwen_target_capture(&adapter, &lane_indices)?;
        let lease = supervisor.arm("Qwen35 DFlash2 verify", QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
        let verify = self.model.forward_gpu_dflash2_verify(
            &verify_lanes,
            kv_cache,
            forward.mode,
            forward.dense_ffn,
            forward.options,
            Some(capture),
        );
        let verify = lease.finish().and(verify);
        let Qwen35JointVerify {
            rows,
            hidden: _,
            pending,
        } = match verify {
            Ok(verify) => verify,
            Err(error) if is_fatal(&error) => return Err(error),
            Err(error) => {
                distribute_error(&mut ticks, &verifying, error);
                return Ok(finish_ticks(ticks));
            }
        };
        drop(verify_lanes);

        let mut accepted = Vec::with_capacity(verifying.len());
        let mut verified = Vec::with_capacity(verifying.len());
        for ((&index, rows), lane_tokens) in verifying.iter().zip(rows).zip(
            verifying
                .iter()
                .map(|&index| tokens[index].take().expect("verified tokens")),
        ) {
            let drafts = &lane_tokens[1..];
            let step = states[index]
                .plan_dflash2_window(
                    self,
                    windows[index].as_ref().expect("verified window"),
                    drafts,
                    rows,
                )
                .map(|plan| Verified::Window {
                    plan,
                    tokens: lane_tokens,
                })
                .unwrap_or_else(Verified::Failed);
            accepted.push(match &step {
                Verified::Window { plan, .. } => plan.accepted(),
                Verified::Failed(_) => 0,
            });
            verified.push(step);
        }

        let lease = supervisor.arm("Qwen35 DFlash2 commit", QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
        let committed = self.model.commit_joint_verify(kv_cache, pending, &accepted);
        let committed = lease.finish().and(committed);
        if let Err(error) = committed {
            if is_fatal(&error) {
                return Err(error);
            }
            distribute_error(&mut ticks, &verifying, error);
            return Ok(finish_ticks(ticks));
        }
        let mut retained = vec![0_u32; physical_lanes];
        for (&index, &count) in verifying.iter().zip(&accepted) {
            retained[states[index].slot_id.0 as usize] = u32::try_from(count)?;
        }
        let lease = supervisor.arm(
            "Qwen35 DFlash2 target-context commit",
            QWEN35_WORKER_TRANSACTION_TIMEOUT,
        )?;
        let context_committed = self.model.with_gpu_cache_mut(|device, registry| {
            runtime.commit_target_context(device, registry, &retained)
        });
        let context_committed = lease.finish().and(context_committed);
        if let Err(error) = context_committed {
            if is_fatal(&error) {
                return Err(error);
            }
            distribute_error(&mut ticks, &verifying, error);
            return Ok(finish_ticks(ticks));
        }
        for ((&index, step), &count) in verifying.iter().zip(&verified).zip(&accepted) {
            if let Verified::Window { tokens, .. } = step {
                let slot = states[index].slot_id.0 as usize;
                runtime
                    .record_lane_tokens(slot, &tokens[..count])
                    .with_context(|| format!("bind DFlash2 slot {slot} retained token prefix"))?;
            }
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
}
