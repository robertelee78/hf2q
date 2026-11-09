//! ADR-065 J1: the fixed-depth-three MTP (K3) transaction, split at its
//! state boundaries so the solo tick and a joint verify window share it.
//!
//! `mtp_k3_round` runs, in order: [`Qwen35DecodeState::k3_entry`],
//! [`Qwen35DecodeState::k3_draft`], [`Qwen35DecodeState::k3_verify_solo`],
//! [`Qwen35DecodeState::k3_verify_to_plan`] and
//! [`Qwen35DecodeState::k3_commit_solo`]. The solo commit runs
//! [`Qwen35DecodeState::k3_commit_cursors`], the DeltaNet restore and
//! [`Qwen35DecodeState::k3_commit_settle`] inside one fail-closed scope, then
//! [`Qwen35DecodeState::k3_commit_finish`] after it.
//!
//! Each unit owns exactly the mutations, failure handling and error contexts
//! the monolithic tick had at the same points; the `#[cfg(test)]` Q1 hooks
//! keep their positions relative to capture arming, clearing and cursor
//! mutation. `plan_qwen35_verified_block` stays the canonical accept walk.
//!
//! A joint caller (J4) must not call `k3_commit_settle` per lane before J2:
//! `clear_la_capture` is cache-global and would disarm every other lane.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use mlx_native::MlxBuffer;

use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::spec_decode::{nth_hidden_row, positions_for_range};
use crate::serve::api::engine_supervisor::EngineSupervisor;
use crate::serve::api::qwen35_speculation::{record_proposer_outcome, QwenSpeculationProposer};
use crate::serve::multi_seq_kv::SlotId;

#[cfg(test)]
use super::q1_speculation;
use super::{
    mtp_cursor_for_slot, plan_qwen35_verified_block, rollback_slot_mtp_draft_error,
    Qwen35DecodeState, Qwen35LoadedModel, Qwen35SpecSemanticState, Qwen35VerifiedBlockPlan,
    QWEN35_WORKER_TRANSACTION_TIMEOUT,
};

/// MTP draft steps per K3 round.
pub(in crate::serve::api) const K3_DRAFT_DEPTH: usize = 3;
/// Verifier rows per K3 round: the seeded token plus every draft.
pub(in crate::serve::api) const K3_VERIFY_ROWS: usize = K3_DRAFT_DEPTH + 1;

/// Entry cursors of one K3 round; `prior_target_len == prior_mtp_len` checked.
pub(in crate::serve::api) struct K3Entry {
    pub next_token: u32,
    pub next_pos: i32,
    pub prior_target_len: u32,
    pub prior_mtp_len: u32,
}

/// Draft result. The draft MTP K/V is already discarded and capture is armed
/// for `K3_VERIFY_ROWS` rows.
pub(in crate::serve::api) struct K3Draft {
    pub entry: K3Entry,
    pub drafts: [u32; K3_DRAFT_DEPTH],
    pub verify_input: [u32; K3_VERIFY_ROWS],
    pub verify_positions: Vec<i32>,
}

/// Verifier rows for one window: `K3_VERIFY_ROWS` logits rows and
/// post-output-norm hidden rows, in the solo route's layout.
pub(in crate::serve::api) struct K3VerifyRows {
    pub logits: MlxBuffer,
    pub hidden: MlxBuffer,
    /// Wall time of the verify forward. A joint window fills it with its
    /// share or `Duration::ZERO`.
    pub verify_elapsed: Duration,
}

/// Planned transaction. `plan` is the unchanged `Qwen35VerifiedBlockPlan`.
pub(in crate::serve::api) struct K3Plan {
    pub draft: K3Draft,
    pub hidden: MlxBuffer,
    pub(super) plan: Qwen35VerifiedBlockPlan,
    pub committed_cursor: u32,
    pub timing: K3PlanTiming,
}

/// Profile-line durations measured inside `k3_verify_to_plan`; `verify` is
/// copied from `K3VerifyRows`.
#[derive(Clone, Copy, Debug)]
pub(in crate::serve::api) struct K3PlanTiming {
    pub verify: Duration,
    pub embed: Duration,
    pub catchup: Duration,
    pub semantic: Duration,
}

/// DeltaNet restore owed after cursor settlement; `None` on full acceptance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::serve::api) struct K3LaRestore {
    pub slot: SlotId,
    pub capture_row: u32,
}

/// Outcome of `k3_commit_finish`: the committed verifier rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::serve::api) struct K3Committed {
    pub valid_input_tokens: usize,
}

/// Outcome of the solo commit with the two profile-line durations measured
/// at the fail-closed scope.
#[derive(Clone, Copy, Debug)]
pub(in crate::serve::api) struct K3SoloCommit {
    pub committed: K3Committed,
    pub state_elapsed: Duration,
    pub rollback_elapsed: Duration,
}
impl K3Plan {
    /// Bind a canonical plan to its round; the committed cursor is the entry
    /// cursor plus the plan's valid verifier rows.
    pub(super) fn new(
        draft: K3Draft,
        hidden: MlxBuffer,
        plan: Qwen35VerifiedBlockPlan,
        timing: K3PlanTiming,
    ) -> Self {
        let committed_cursor = draft.entry.prior_target_len + plan.valid_input_tokens as u32;
        Self {
            draft,
            hidden,
            plan,
            committed_cursor,
            timing,
        }
    }
}

impl Qwen35DecodeState {
    /// Read the round's entry token, position and cursors. No mutation.
    pub(in crate::serve::api) fn k3_entry(&self, kv_cache: &HybridKvCache) -> Result<K3Entry> {
        let next_token = *self
            .generated_tokens
            .last()
            .context("Qwen SlotAware MTP K3 needs a seeded output token")?;
        let next_pos = (self.decode_position_base + self.generated_tokens.len() - 1) as i32;
        let prior_target_len = kv_cache
            .sequence_len_for_slot(self.slot_id)
            .context("Qwen SlotAware MTP K3 read target cursor")?;
        let prior_mtp_len = mtp_cursor_for_slot(kv_cache, self.slot_id)?;
        anyhow::ensure!(
            prior_target_len == prior_mtp_len,
            "Qwen SlotAware MTP K3 requires equal entry cursors (target={prior_target_len}, mtp={prior_mtp_len})"
        );
        Ok(K3Entry {
            next_token,
            next_pos,
            prior_target_len,
            prior_mtp_len,
        })
    }

    /// Arm recurrent capture, run three greedy MTP draft steps and discard
    /// their MTP K/V back to `prior_mtp_len`.
    pub(in crate::serve::api) fn k3_draft(
        &mut self,
        qwen: &Qwen35LoadedModel,
        kv_cache: &mut HybridKvCache,
        supervisor: &EngineSupervisor,
        entry: K3Entry,
    ) -> Result<K3Draft> {
        let next_token = entry.next_token;
        let next_pos = entry.next_pos;
        let prior_mtp_len = entry.prior_mtp_len;
        let mtp = qwen
            .model
            .mtp
            .as_ref()
            .context("Qwen SlotAware MTP K3 weights missing")?;
        let initial_hidden = &self
            .mtp
            .as_ref()
            .context("Qwen SlotAware MTP K3 state missing")?
            .verifier_hidden;
        if let Err(error) = qwen.model.with_gpu_cache_mut(|device, _registry| {
            kv_cache
                .ensure_la_capture(&qwen.model.cfg, device, K3_VERIFY_ROWS as u32)
                .context("Qwen SlotAware MTP K3 allocate recurrent capture")
        }) {
            kv_cache.clear_la_capture();
            return Err(error);
        }
        let draft_lease = match supervisor.arm(
            "Qwen35 SlotAware MTP K3 draft",
            QWEN35_WORKER_TRANSACTION_TIMEOUT,
        ) {
            Ok(lease) => lease,
            Err(error) => {
                kv_cache.clear_la_capture();
                return Err(error.context("Qwen SlotAware MTP K3 draft admission"));
            }
        };
        let drafted = qwen.model.with_gpu_cache_mut(|device, registry| {
            let mut drafts = Vec::with_capacity(K3_DRAFT_DEPTH);
            let mut chain_hidden: Option<MlxBuffer> = None;
            let mut chain_token = next_token;
            for depth in 0..K3_DRAFT_DEPTH {
                let shared_embed =
                    qwen.model
                        .embed_tokens_gpu_in_context(&[chain_token], device, registry)?;
                let previous = chain_hidden.as_ref().unwrap_or(initial_hidden);
                let (token, next_hidden) = mtp.forward_draft_greedy_for_token(
                    previous,
                    chain_token,
                    &shared_embed,
                    kv_cache,
                    self.slot_id,
                    &[next_pos + depth as i32; 4],
                    device,
                    registry,
                    &qwen.model.cfg,
                )?;
                drafts.push(token);
                chain_token = token;
                chain_hidden = Some(next_hidden);
            }
            Ok::<_, anyhow::Error>(drafts)
        });
        let draft_supervision = draft_lease.finish();
        if let Err(error) = draft_supervision.and(
            drafted
                .as_ref()
                .map(|_| ())
                .map_err(|e| anyhow::anyhow!("{e:#}")),
        ) {
            kv_cache.clear_la_capture();
            return rollback_slot_mtp_draft_error(
                kv_cache,
                self.slot_id,
                prior_mtp_len,
                error,
                "Qwen SlotAware MTP K3 draft",
            );
        }
        let drafts = drafted.expect("draft result checked above");
        if let Err(error) = kv_cache.truncate_mtp_to_for_slot(self.slot_id, prior_mtp_len) {
            kv_cache.clear_la_capture();
            return rollback_slot_mtp_draft_error(
                kv_cache,
                self.slot_id,
                prior_mtp_len,
                error,
                "Qwen SlotAware MTP K3 discard speculative draft cache",
            );
        }
        #[cfg(test)]
        q1_speculation::first_fork::after_draft_discard(next_pos, kv_cache)?;
        #[cfg(test)]
        let drafts =
            q1_speculation::after_drafts(next_pos, entry.prior_target_len, prior_mtp_len, drafts)?;
        let drafts: [u32; K3_DRAFT_DEPTH] = drafts
            .try_into()
            .expect("the draft chain yields K3_DRAFT_DEPTH tokens");
        Ok(K3Draft {
            entry,
            drafts,
            verify_input: [next_token, drafts[0], drafts[1], drafts[2]],
            verify_positions: positions_for_range(next_pos, K3_VERIFY_ROWS),
        })
    }

    /// The solo route's four-row target verify forward. Writes four target
    /// K/V rows and four capture rows, then flips this slot's parity once.
    pub(in crate::serve::api) fn k3_verify_solo(
        &self,
        qwen: &Qwen35LoadedModel,
        kv_cache: &mut HybridKvCache,
        supervisor: &EngineSupervisor,
        draft: &K3Draft,
    ) -> Result<K3VerifyRows> {
        let verify_lease = match supervisor.arm(
            "Qwen35 SlotAware MTP K3 verify",
            QWEN35_WORKER_TRANSACTION_TIMEOUT,
        ) {
            Ok(lease) => lease,
            Err(error) => {
                kv_cache.clear_la_capture();
                return Err(error.context("Qwen SlotAware MTP K3 verify admission"));
            }
        };
        let verify_started = Instant::now();
        let verified = {
            let _native_role = crate::inference::native_observation::target_verify();
            qwen.model.forward_gpu_with_nextn_hidden_buffer(
                &draft.verify_input,
                &draft.verify_positions,
                kv_cache,
                self.slot_id,
            )
        };
        let verify_supervision = verify_lease.finish();
        let (logits, hidden) = match (verify_supervision, verified) {
            (Ok(()), Ok(value)) => value,
            (supervision, forward) => {
                let error = supervision
                    .err()
                    .or_else(|| forward.err())
                    .expect("failed verification has an error");
                kv_cache.clear_la_capture();
                kv_cache.reset_for_slot(self.slot_id).with_context(|| {
                    format!("Qwen SlotAware MTP K3 verify failed ({error:#}); fail-closed reset")
                })?;
                return Err(error.context("Qwen SlotAware MTP K3 verify"));
            }
        };
        #[cfg(test)]
        q1_speculation::first_fork::verifier_pre_catchup(draft.entry.next_pos, &logits, &hidden)?;
        let verify_elapsed = verify_started.elapsed();
        Ok(K3VerifyRows {
            logits,
            hidden,
            verify_elapsed,
        })
    }

    /// Check the verifier rows, catch the MTP K/V up over them and run the
    /// canonical accept walk. Every failure clears capture and resets the
    /// slot.
    pub(in crate::serve::api) fn k3_verify_to_plan(
        &mut self,
        qwen: &Qwen35LoadedModel,
        kv_cache: &mut HybridKvCache,
        supervisor: &EngineSupervisor,
        draft: K3Draft,
        rows: K3VerifyRows,
    ) -> Result<K3Plan> {
        let K3VerifyRows {
            logits: mut verify_logits,
            hidden: verify_hidden,
            verify_elapsed,
        } = rows;
        #[cfg(test)]
        let next_pos = draft.entry.next_pos;
        let mtp = qwen
            .model
            .mtp
            .as_ref()
            .expect("MTP weights checked by k3_draft");
        let initial_hidden = &self
            .mtp
            .as_ref()
            .expect("MTP state checked by k3_draft")
            .verifier_hidden;
        let vocab = qwen.vocab_size;
        if verify_logits.element_count() != K3_VERIFY_ROWS * vocab
            || verify_hidden.element_count() != K3_VERIFY_ROWS * qwen.model.cfg.hidden_size as usize
        {
            let error = anyhow::anyhow!(
                "Qwen SlotAware MTP K3 verify shape mismatch: logits={} expected={}, hidden={} expected={}",
                verify_logits.element_count(),
                K3_VERIFY_ROWS * vocab,
                verify_hidden.element_count(),
                K3_VERIFY_ROWS * qwen.model.cfg.hidden_size as usize,
            );
            kv_cache.clear_la_capture();
            kv_cache
                .reset_for_slot(self.slot_id)
                .context("Qwen SlotAware MTP K3 shape failure reset")?;
            return Err(error);
        }

        let embed_started = Instant::now();
        let shared_embed_rows = match qwen.model.embed_tokens_gpu(&draft.verify_input) {
            Ok(rows) => rows,
            Err(error) => {
                kv_cache.clear_la_capture();
                kv_cache
                    .reset_for_slot(self.slot_id)
                    .context("Qwen SlotAware MTP K3 embedding failure reset")?;
                return Err(error.context("Qwen SlotAware MTP K3 verifier embeddings"));
            }
        };
        let embed_elapsed = embed_started.elapsed();
        let catchup_lease = match supervisor.arm(
            "Qwen35 SlotAware MTP K3 target catch-up",
            QWEN35_WORKER_TRANSACTION_TIMEOUT,
        ) {
            Ok(lease) => lease,
            Err(error) => {
                kv_cache.clear_la_capture();
                kv_cache
                    .reset_for_slot(self.slot_id)
                    .context("Qwen SlotAware MTP K3 catch-up admission reset")?;
                return Err(error.context("Qwen SlotAware MTP K3 catch-up admission"));
            }
        };
        let catchup_started = Instant::now();
        let caught_up = qwen.model.with_gpu_cache_mut(|device, registry| {
            mtp.process_target_batch(
                &draft.verify_input,
                Some(initial_hidden),
                &verify_hidden,
                &shared_embed_rows,
                kv_cache,
                self.slot_id,
                &draft.verify_positions,
                device,
                registry,
                &qwen.model.cfg,
            )
        });
        let catchup_supervision = catchup_lease.finish();
        if let Err(error) = catchup_supervision.and(caught_up) {
            kv_cache.clear_la_capture();
            kv_cache.reset_for_slot(self.slot_id).with_context(|| {
                format!("Qwen SlotAware MTP K3 catch-up failed ({error:#}); fail-closed reset")
            })?;
            return Err(error.context("Qwen SlotAware MTP K3 target catch-up"));
        }
        let catchup_elapsed = catchup_started.elapsed();

        #[cfg(test)]
        q1_speculation::verified_buffers(next_pos, &verify_logits, &verify_hidden)?;
        let readback = crate::inference::stage_profile::CpuSpan::begin("logit_readback");
        let verify_logits = match verify_logits.as_mut_slice::<f32>() {
            Ok(logits) => logits,
            Err(error) => {
                kv_cache.clear_la_capture();
                kv_cache
                    .reset_for_slot(self.slot_id)
                    .context("Qwen SlotAware MTP K3 logits view failure reset")?;
                return Err(anyhow::anyhow!("{error}").context("Qwen SlotAware MTP K3 logits view"));
            }
        };
        drop(readback);
        let host_selection = crate::inference::stage_profile::CpuSpan::begin("host_selection");
        let semantic_started = Instant::now();
        let mut semantic = Qwen35SpecSemanticState::from_decode(self);
        let plan = match plan_qwen35_verified_block(&draft.drafts, |row| {
            let start = row * vocab;
            let decision = semantic.select_and_observe(
                qwen,
                &self.params,
                &mut verify_logits[start..start + vocab],
            )?;
            #[cfg(test)]
            q1_speculation::canonical_decision(next_pos, row, decision.token, decision.terminal)?;
            Ok(decision)
        }) {
            Ok(plan) => plan,
            Err(error) => {
                kv_cache.clear_la_capture();
                kv_cache
                    .reset_for_slot(self.slot_id)
                    .context("Qwen SlotAware MTP K3 semantic failure reset")?;
                return Err(error.context("Qwen SlotAware MTP K3 canonical accept walk"));
            }
        };
        let semantic_elapsed = semantic_started.elapsed();
        drop(host_selection);

        Ok(K3Plan::new(
            draft,
            verify_hidden,
            plan,
            K3PlanTiming {
                verify: verify_elapsed,
                embed: embed_elapsed,
                catchup: catchup_elapsed,
                semantic: semantic_elapsed,
            },
        ))
    }

    /// Truncate the target, then the MTP cursor to the committed prefix.
    /// Returns the DeltaNet restore a partial plan owes. Errors belong to the
    /// caller's fail-closed scope.
    pub(in crate::serve::api) fn k3_commit_cursors(
        &self,
        kv_cache: &mut HybridKvCache,
        plan: &K3Plan,
    ) -> Result<Option<K3LaRestore>> {
        kv_cache
            .truncate_full_attn_to_for_slot(self.slot_id, plan.committed_cursor)
            .context("Qwen SlotAware MTP K3 target truncate")?;
        kv_cache
            .truncate_mtp_to_for_slot(self.slot_id, plan.committed_cursor)
            .context("Qwen SlotAware MTP K3 MTP truncate")?;
        Ok(
            (plan.plan.valid_input_tokens < K3_VERIFY_ROWS).then_some(K3LaRestore {
                slot: self.slot_id,
                capture_row: plan.plan.carry_hidden_row as u32,
            }),
        )
    }

    /// Disarm capture (cache-global until J2) and check that both cursors sit
    /// on the committed prefix. Errors belong to the caller's fail-closed
    /// scope.
    pub(in crate::serve::api) fn k3_commit_settle(
        &self,
        kv_cache: &mut HybridKvCache,
        plan: &K3Plan,
    ) -> Result<()> {
        kv_cache.clear_la_capture();
        kv_cache
            .validate_speculative_cursors_for_slot(self.slot_id, plan.committed_cursor as usize)
            .context("Qwen SlotAware MTP K3 committed cursor equality")
    }

    /// Install the carry hidden row and the verified output ledger. Runs
    /// after the fail-closed scope: capture is already cleared, so a carry
    /// failure only resets the slot.
    pub(in crate::serve::api) fn k3_commit_finish(
        &mut self,
        qwen: &Qwen35LoadedModel,
        kv_cache: &mut HybridKvCache,
        plan: K3Plan,
    ) -> Result<K3Committed> {
        let carry_hidden = match nth_hidden_row(
            &plan.hidden,
            qwen.model.cfg.hidden_size,
            plan.plan.carry_hidden_row as u64,
        ) {
            Ok(hidden) => hidden,
            Err(error) => {
                kv_cache
                    .reset_for_slot(self.slot_id)
                    .context("Qwen SlotAware MTP K3 carry-hidden failure reset")?;
                return Err(error.context("Qwen SlotAware MTP K3 carry hidden"));
            }
        };
        self.mtp
            .as_mut()
            .expect("MTP state checked above")
            .verifier_hidden = carry_hidden;
        #[cfg(test)]
        q1_speculation::state_committed(plan.draft.entry.next_pos, kv_cache, &plan.plan)?;
        let committed = K3Committed {
            valid_input_tokens: plan.plan.valid_input_tokens,
        };
        self.pending_speculation_output = plan.plan.output;
        self.terminal_after_pending = plan.plan.terminal_after_pending;
        record_proposer_outcome(
            QwenSpeculationProposer::Mtp,
            K3_DRAFT_DEPTH,
            plan.plan.matched_drafts,
            plan.plan.rejected_drafts,
            1,
            0,
        );
        Ok(committed)
    }

    /// The solo commit: cursors, the configured DeltaNet restore and
    /// settlement inside one fail-closed scope (any error clears capture and
    /// resets the slot), then the ledger.
    pub(in crate::serve::api) fn k3_commit_solo(
        &mut self,
        qwen: &Qwen35LoadedModel,
        kv_cache: &mut HybridKvCache,
        supervisor: &EngineSupervisor,
        plan: K3Plan,
    ) -> Result<K3SoloCommit> {
        let state_started = Instant::now();
        let mut rollback_elapsed = Duration::ZERO;
        let state_result = (|| -> Result<()> {
            let restore = {
                let _rollback_cpu = crate::inference::stage_profile::CpuSpan::begin("rollback");
                self.k3_commit_cursors(kv_cache, &plan)?
            };
            if let Some(restore) = restore {
                let rollback_started = Instant::now();
                let lease = supervisor.arm(
                    "Qwen35 SlotAware MTP K3 restore",
                    QWEN35_WORKER_TRANSACTION_TIMEOUT,
                )?;
                let restored = kv_cache
                    .rollback_la(restore.slot, restore.capture_row)
                    .context("Qwen SlotAware MTP K3 recurrent rollback");
                lease.finish().and(restored)?;
                rollback_elapsed = rollback_started.elapsed();
            }
            self.k3_commit_settle(kv_cache, &plan)
        })();
        let state_elapsed = state_started.elapsed();
        if let Err(error) = state_result {
            kv_cache.clear_la_capture();
            kv_cache.reset_for_slot(self.slot_id).with_context(|| {
                format!("Qwen SlotAware MTP K3 commit failed ({error:#}); fail-closed reset")
            })?;
            return Err(error);
        }
        let committed = self.k3_commit_finish(qwen, kv_cache, plan)?;
        Ok(K3SoloCommit {
            committed,
            state_elapsed,
            rollback_elapsed,
        })
    }
}

#[cfg(test)]
#[path = "k3_transaction_tests.rs"]
mod tests;
