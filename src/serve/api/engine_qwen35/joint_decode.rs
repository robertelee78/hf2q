//! Slot-ledger half of Qwen joint B-row decode.
//!
//! A joint step replaces one slot's ordinary target forward with a packed
//! forward shared by several slots. Each slot prepares exactly the inputs
//! its serial tick would feed the target (`prepare_joint_tick`) and later
//! applies the target's result through the serial tick's own commit code
//! (`finish_joint_tick`):
//!
//! - slots without MTP state take the ordinary path of `decode_tick`
//!   (thinking-budget forcing, GPU argmax for greedy, constrained sampling);
//! - slots with MTP state take the coherent ordinary path of
//!   `decode_tick_mtp_warmup`: after the shared target forward they run
//!   their own MTP catch-up over the same token, so target and MTP cursors
//!   stay equal and a later solo turn can resume K3 drafting.
//!
//! Phase 1 suppresses MTP drafting and history proposals for joint rows; a
//! history-lookup ledger catches up lazily from the committed tokens on the
//! slot's next solo round, exactly as after a serial lookup miss.

use anyhow::{Context, Result};
use std::time::Instant;

use crate::inference::models::qwen35::joint_decode_routes::{JointDenseFfnRoute, JointRouteMode};
use crate::inference::models::qwen35::joint_step_options::JointStepOptions;
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::{Qwen35JointOutput, Qwen35JointRow, Qwen35JointRowOutput};
use crate::serve::api::engine_supervisor::EngineSupervisor;
use crate::serve::api::qwen35_speculation::QwenSpeculationRoute;

use super::{
    mtp_cursor_for_slot, rollback_slot_mtp_error, Qwen35CoherentTarget, Qwen35DecodeState,
    Qwen35LoadedModel, Qwen35TickKind, Qwen35TickOutcome, QWEN35_WORKER_TRANSACTION_TIMEOUT,
};

/// How a prepared joint row commits its target result.
pub(in crate::serve::api) enum Qwen35JointTarget {
    /// `decode_tick`'s ordinary target decision.
    Ordinary {
        forced_token: Option<(u32, bool)>,
        started: Instant,
    },
    /// `decode_tick_mtp_warmup`'s coherent decision (target + MTP catch-up).
    Coherent(Qwen35CoherentTarget),
}

/// One slot's prepared contribution to a joint step.
pub(in crate::serve::api) struct Qwen35JointLane {
    pub(in crate::serve::api) row: Qwen35JointRow,
    pub(in crate::serve::api) target: Qwen35JointTarget,
}

impl Qwen35DecodeState {
    /// Whether this slot's next tick is a target forward that a joint step
    /// can serve: no CPU-only verified output is queued, the step bound is
    /// not reached, and the cache cursor (and MTP cursor, when MTP state is
    /// held) equals the committed ledger minus the unfed newest token.
    pub(in crate::serve::api) fn joint_decode_ready(&self, kv_cache: &HybridKvCache) -> bool {
        if self.step >= self.max_tokens
            || !self.pending_speculation_output.is_empty()
            || self.generated_tokens.is_empty()
        {
            return false;
        }
        let fed = self.prompt_tokens.len() + self.generated_tokens.len() - 1;
        let Ok(cursor) = kv_cache.sequence_len_for_slot(self.slot_id) else {
            return false;
        };
        if cursor as usize != fed {
            return false;
        }
        match self.mtp {
            Some(_) => mtp_cursor_for_slot(kv_cache, self.slot_id).is_ok_and(|mtp| mtp == cursor),
            None => true,
        }
    }

    /// Whether this slot's serial tick is the ordinary target forward: no
    /// MTP state and no history lookup, so a one-lane joint step replaces
    /// exactly that forward and suppresses no speculation (#717).
    pub(in crate::serve::api) fn ordinary_target_only(&self) -> bool {
        self.mtp.is_none() && self.history_lookup.is_none()
    }

    /// Counterfactual joint-verify eligibility for the E4 interval observer.
    /// Transform blockers are definitive. Proposer availability is evaluated
    /// from request-local state only and does not advance any live ledger.
    pub(in crate::serve::api) fn joint_verify_eligibility(&self) -> &'static str {
        if !super::shadow_draft::verify_transforms_clear(self) {
            return "blocked_transform";
        }
        let remaining = self.max_tokens.saturating_sub(self.generated_tokens.len());
        if self.mtp.is_some() && remaining >= super::K3_VERIFY_ROWS {
            return "eligible";
        }
        let history = self.history_lookup.as_ref().is_some_and(|live| {
            let mut lookup = live.clone();
            let expected = self.prompt_tokens.len() + self.generated_tokens.len();
            if lookup.verified_len() < expected {
                let Some(cursor) = lookup.verified_len().checked_sub(self.prompt_tokens.len())
                else {
                    return false;
                };
                lookup.extend_verified(&self.generated_tokens[cursor..]);
            }
            lookup.verified_len() == expected && !lookup.propose().is_empty()
        });
        if history {
            "eligible"
        } else {
            "no_draft"
        }
    }

    /// Joint-partition class of a joint-ready lane: `Some(ordinary)` lets it
    /// join, `None` keeps it on the serial loop.
    /// - Repeatable route: only a lane whose planned tick is `Ordinary` joins,
    ///   as an ordinary row (alone included in padded mode); a lane that
    ///   speculates, drains verified output or has no plan stays serial, so
    ///   its route never depends on the cohort.
    /// - Adaptive route (phase 1): any joint-ready lane joins a group of two
    ///   or more as an M1 row; alone it joins only without speculation state.
    pub(in crate::serve::api) fn joint_lane_class(&self) -> Option<bool> {
        match self.speculation_route() {
            QwenSpeculationRoute::Repeatable => {
                matches!(self.planned_tick(), Some(Qwen35TickKind::Ordinary)).then_some(true)
            }
            QwenSpeculationRoute::Adaptive => Some(self.ordinary_target_only()),
        }
    }

    /// Prepare the target input this slot's serial tick would feed. Mutates
    /// only what the serial tick mutates before its forward (the thinking
    /// budget's forced cursor on the ordinary path).
    pub(in crate::serve::api) fn prepare_joint_tick(
        &mut self,
        kv_cache: &HybridKvCache,
    ) -> Result<Qwen35JointLane> {
        if self.mtp.is_some() {
            let target = self.coherent_ordinary_target(kv_cache)?;
            return Ok(Qwen35JointLane {
                row: Qwen35JointRow {
                    slot_id: self.slot_id,
                    token: target.next_token,
                    position: target.next_pos,
                    output: Qwen35JointOutput::LogitsAndHidden,
                },
                target: Qwen35JointTarget::Coherent(target),
            });
        }
        let forced_token = self.next_ordinary_forced_token();
        let token = *self
            .generated_tokens
            .last()
            .context("Qwen joint ordinary decode needs a seeded output token")?;
        let position = (self.decode_position_base + self.step - 1) as i32;
        let output = if self.is_greedy && !self.want_logprobs {
            Qwen35JointOutput::GreedyToken
        } else {
            Qwen35JointOutput::Logits
        };
        Ok(Qwen35JointLane {
            row: Qwen35JointRow {
                slot_id: self.slot_id,
                token,
                position,
                output,
            },
            target: Qwen35JointTarget::Ordinary {
                forced_token,
                started: Instant::now(),
            },
        })
    }

    /// Commit this slot's share of a healthy joint target forward through
    /// the serial tick's commit path.
    pub(in crate::serve::api) fn finish_joint_tick(
        &mut self,
        qwen: &Qwen35LoadedModel,
        kv_cache: &mut HybridKvCache,
        supervisor: &EngineSupervisor,
        target: Qwen35JointTarget,
        output: Qwen35JointRowOutput,
    ) -> Result<Qwen35TickOutcome> {
        match target {
            Qwen35JointTarget::Ordinary {
                forced_token,
                started,
            } => {
                let token = match (output.greedy_token, output.logits) {
                    (Some(predicted), _) => forced_token.map_or(predicted, |(token, _)| token),
                    (None, Some(logits)) => {
                        self.select_ordinary_logits_token(qwen, logits, forced_token)?
                    }
                    (None, None) => anyhow::bail!("Qwen joint row returned no target decision"),
                };
                self.commit_ordinary_target_token(qwen, token, started.elapsed())
            }
            Qwen35JointTarget::Coherent(target) => {
                let (Some(logits), Some(nextn_hidden)) = (output.logits, output.nextn_hidden)
                else {
                    return rollback_slot_mtp_error(
                        kv_cache,
                        self.slot_id,
                        target.prior_target,
                        target.prior_mtp,
                        anyhow::anyhow!("Qwen joint coherent row returned no logits/hidden"),
                        "Qwen joint coherent ordinary target",
                    );
                };
                self.finish_coherent_ordinary(
                    qwen,
                    kv_cache,
                    supervisor,
                    target,
                    logits,
                    nextn_hidden,
                )
            }
        }
    }
}

impl Qwen35LoadedModel {
    /// One supervised joint target forward over prepared rows.
    pub(in crate::serve::api) fn forward_joint_decode(
        &self,
        rows: &[Qwen35JointRow],
        kv_cache: &mut HybridKvCache,
        mode: JointRouteMode,
        dense_ffn: JointDenseFfnRoute,
        options: JointStepOptions,
        supervisor: &EngineSupervisor,
    ) -> Result<Vec<Qwen35JointRowOutput>> {
        let lease = supervisor.arm("Qwen35 joint decode", QWEN35_WORKER_TRANSACTION_TIMEOUT)?;
        let forward = self
            .model
            .forward_gpu_decode_joint_with_options(rows, kv_cache, mode, dense_ffn, options);
        lease.finish()?;
        let outputs = forward.with_context(|| {
            format!(
                "Qwen35Model::forward_gpu_decode_joint_with_options ({} rows, {} mode, {} dense FFN, {options:?})",
                rows.len(),
                mode.label(),
                dense_ffn.label()
            )
        })?;
        anyhow::ensure!(
            outputs.len() == rows.len(),
            "Qwen joint decode returned {} outputs for {} rows",
            outputs.len(),
            rows.len()
        );
        Ok(outputs)
    }
}
