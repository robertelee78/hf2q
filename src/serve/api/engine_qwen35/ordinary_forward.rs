//! Typed ordinary target boundary; no shared producer or policy activation.
//! Preparation is before target mutation; completion consumes it exactly once.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DecodeFingerprint { slot: SlotId, step: usize, generated_len: usize, last_token: u32, position_base: usize }

/// Request-local continuation, NOT scheduler/generation authority. The owning
/// worker must retain and validate the actual installed handle separately.
pub(crate) struct PreparedCoherentOrdinaryTarget {
    fingerprint: DecodeFingerprint,
    next_pos: i32,
    prior_target: u32,
    prior_mtp: u32,
    pending_hidden: MlxBuffer,
}
impl PreparedCoherentOrdinaryTarget {
    pub(crate) fn input(&self) -> (u32, i32, SlotId) {
        (self.fingerprint.last_token, self.next_pos, self.fingerprint.slot)
    }
}
/// Source-owned actual full vocabulary and post-output-RMSNorm h_nextn row.
/// This type does not certify normalized bytes: only the actual producing head
/// and its checked completion do. Do not substitute the pre-norm residual.
pub(crate) struct CoherentOrdinaryTargetResult { logits: Vec<f32>, nextn_hidden: MlxBuffer }
impl CoherentOrdinaryTargetResult {
    pub(crate) fn from_actual_forward(logits: Vec<f32>, nextn_hidden: MlxBuffer) -> Self {
        Self { logits, nextn_hidden }
    }
}

/// Called before ANY catchup/selection. The target has advanced exactly once;
/// the MTP cursor has NOT advanced yet. Never reread target as its old prior.
fn validate_boundary(
    prepared: DecodeFingerprint, current: DecodeFingerprint,
    prior: (u32, u32), now: (u32, u32),
    output: (usize, usize, bool), expected: (usize, usize), state_mtp_present: bool,
) -> Result<()> {
    anyhow::ensure!(prepared == current, "coherent ordinary prepared owner/step/input changed");
    let next = prior.0.checked_add(1).context("coherent ordinary prior cursor overflow")?;
    anyhow::ensure!(prior.0 == prior.1 && now == (next, prior.1),
        "coherent ordinary supplied result requires target=prior+1 and MTP=prior");
    anyhow::ensure!(state_mtp_present, "coherent ordinary prepared MTP state missing");
    anyhow::ensure!((output.0, output.1) == expected && output.2 && expected.0 > 0 && expected.1 > 0,
        "coherent ordinary supplied full logits/normalized hidden extent");
    Ok(())
}

impl Qwen35DecodeState {
    fn coherent_target_fingerprint(&self) -> Result<DecodeFingerprint> {
        Ok(DecodeFingerprint { slot: self.slot_id, step: self.step,
            generated_len: self.generated_tokens.len(), position_base: self.decode_position_base, last_token: *self.generated_tokens.last()
                .context("Qwen SlotAware coherent ordinary decode needs a seed")? })
    }
    pub(crate) fn prepare_coherent_ordinary_target(
        &self, kv_cache: &HybridKvCache,
    ) -> Result<PreparedCoherentOrdinaryTarget> {
        let fingerprint = self.coherent_target_fingerprint()?;
        let next_pos = (self.decode_position_base + fingerprint.generated_len - 1) as i32;
        let prior_target = kv_cache.sequence_len_for_slot(self.slot_id)?;
        let prior_mtp = mtp_cursor_for_slot(kv_cache, self.slot_id)?;
        anyhow::ensure!(
            prior_target == prior_mtp,
            "Qwen coherent ordinary decode cursor mismatch (target={prior_target}, mtp={prior_mtp})"
        );
        let pending_hidden = self
            .mtp
            .as_ref()
            .context("Qwen coherent ordinary decode missing MTP state")?
            .verifier_hidden
            .clone();
        Ok(PreparedCoherentOrdinaryTarget {
            fingerprint, next_pos,
            prior_target, prior_mtp, pending_hidden,
        })
    }

    pub(super) fn produce_scalar_coherent_target(
        &mut self, qwen: &mut Qwen35LoadedModel, kv_cache: &mut HybridKvCache,
        supervisor: &EngineSupervisor, prepared: &PreparedCoherentOrdinaryTarget,
    ) -> Result<CoherentOrdinaryTargetResult> {
        let (next_token, next_pos, _) = prepared.input();
        let (prior_target, prior_mtp) = (prepared.prior_target, prepared.prior_mtp);
        let lease = supervisor.arm(
            "Qwen35 coherent ordinary target",
            QWEN35_WORKER_TRANSACTION_TIMEOUT,
        )?;
        let forward = qwen.model.forward_gpu_with_nextn_hidden(
            &[next_token],
            &[next_pos; 4],
            kv_cache,
            self.slot_id,
        );
        if let Err(error) = lease.finish() {
            return rollback_slot_mtp_error(
                kv_cache,
                self.slot_id,
                prior_target,
                prior_mtp,
                error,
                "Qwen coherent ordinary target supervision",
            );
        }
        let (logits, nextn_hidden) = match forward {
            Ok(value) => value,
            Err(error) => {
                return rollback_slot_mtp_error(
                    kv_cache,
                    self.slot_id,
                    prior_target,
                    prior_mtp,
                    error,
                    "Qwen coherent ordinary target",
                );
            }
        };
        Ok(CoherentOrdinaryTargetResult::from_actual_forward(logits, nextn_hidden))
    }

    /// Future shared caller checks ALL members here before any member tail.
    /// On a shared error, do not invoke scalar rollback/replay as fallback.
    pub(crate) fn validate_coherent_ordinary_result(
        &self, qwen: &Qwen35LoadedModel, kv_cache: &HybridKvCache,
        prepared: &PreparedCoherentOrdinaryTarget, target: &CoherentOrdinaryTargetResult,
    ) -> Result<()> {
        validate_boundary(
            prepared.fingerprint, self.coherent_target_fingerprint()?,
            (prepared.prior_target, prepared.prior_mtp),
            (kv_cache.sequence_len_for_slot(self.slot_id)?, mtp_cursor_for_slot(kv_cache, self.slot_id)?),
            (target.logits.len(), target.nextn_hidden.element_count(),
                target.nextn_hidden.dtype() == mlx_native::DType::F32),
            (qwen.vocab_size, qwen.model.cfg.hidden_size as usize), self.mtp.is_some(),
        )
    }

    /// Default scalar and future supplied target share this exact completion.
    /// A shared caller must validate ALL participants after target completion
    /// before any member tail, and fail-stop on uncertain shared mutation;
    /// this local continuation does not grant cohort rollback/replay authority.
    pub(in crate::serve::api) fn finish_coherent_ordinary_target(
        &mut self, qwen: &mut Qwen35LoadedModel, kv_cache: &mut HybridKvCache,
        supervisor: &EngineSupervisor, prepared: PreparedCoherentOrdinaryTarget,
        target: CoherentOrdinaryTargetResult, ordinary_decision_started: Instant,
    ) -> Result<Qwen35TickOutcome> {
        let (next_token, next_pos, _) = prepared.input();
        let (prior_target, prior_mtp) = (prepared.prior_target, prepared.prior_mtp);
        let boundary = self.validate_coherent_ordinary_result(qwen, kv_cache, &prepared, &target);
        if let Err(error) = boundary {
            return rollback_slot_mtp_error(kv_cache, self.slot_id, prior_target, prior_mtp,
                error, "Qwen coherent ordinary prepared result boundary");
        }
        let CoherentOrdinaryTargetResult { mut logits, nextn_hidden } = target;
        let pending_hidden = prepared.pending_hidden;
        let shared_embed = match qwen.model.embed_tokens_gpu(&[next_token]) {
            Ok(embed) => embed,
            Err(error) => {
                return rollback_slot_mtp_error(
                    kv_cache,
                    self.slot_id,
                    prior_target,
                    prior_mtp,
                    error,
                    "Qwen coherent ordinary shared embedding",
                );
            }
        };
        let mtp = match qwen.model.mtp.as_ref() {
            Some(mtp) => mtp,
            None => {
                return rollback_slot_mtp_error(
                    kv_cache,
                    self.slot_id,
                    prior_target,
                    prior_mtp,
                    anyhow::anyhow!("Qwen coherent ordinary decode MTP weights missing"),
                    "Qwen coherent ordinary MTP lookup",
                );
            }
        };
        let lease = match supervisor.arm(
            "Qwen35 coherent ordinary MTP catch-up",
            QWEN35_WORKER_TRANSACTION_TIMEOUT,
        ) {
            Ok(lease) => lease,
            Err(error) => {
                return rollback_slot_mtp_error(
                    kv_cache,
                    self.slot_id,
                    prior_target,
                    prior_mtp,
                    error,
                    "Qwen coherent ordinary MTP catch-up admission",
                );
            }
        };
        let caught_up = qwen.model.with_gpu_cache_mut(|device, registry| {
            mtp.process_target_batch(
                &[next_token],
                Some(&pending_hidden),
                &nextn_hidden,
                &shared_embed,
                kv_cache,
                self.slot_id,
                &[next_pos; 4],
                device,
                registry,
                &qwen.model.cfg,
            )
        });
        if let Err(error) = lease.finish().and(caught_up) {
            return rollback_slot_mtp_error(
                kv_cache,
                self.slot_id,
                prior_target,
                prior_mtp,
                error,
                "Qwen coherent ordinary MTP catch-up",
            );
        }
        if let Err(error) = kv_cache
            .validate_speculative_cursors_for_slot(self.slot_id, (prior_target + 1) as usize)
        {
            return rollback_slot_mtp_error(
                kv_cache,
                self.slot_id,
                prior_target,
                prior_mtp,
                error,
                "Qwen coherent ordinary committed cursor equality",
            );
        }
        let carry_hidden = match crate::inference::models::qwen35::spec_decode::nth_hidden_row(
            &nextn_hidden,
            qwen.model.cfg.hidden_size,
            0,
        ) {
            Ok(hidden) => hidden,
            Err(error) => {
                return rollback_slot_mtp_error(
                    kv_cache,
                    self.slot_id,
                    prior_target,
                    prior_mtp,
                    error,
                    "Qwen coherent ordinary carry hidden",
                );
            }
        };
        self.mtp
            .as_mut()
            .expect("MTP state checked above")
            .verifier_hidden = carry_hidden;

        let mut semantic = Qwen35SpecSemanticState::from_decode(self);
        let decision = match semantic.select_and_observe(qwen, &self.params, &mut logits) {
            Ok(decision) => decision,
            Err(error) => {
                return rollback_slot_mtp_error(
                    kv_cache,
                    self.slot_id,
                    prior_target,
                    prior_mtp,
                    error,
                    "Qwen coherent ordinary canonical decision",
                );
            }
        };
        let ordinary_decision_elapsed = ordinary_decision_started.elapsed();
        self.mtp_cost
            .observe_ordinary_target(ordinary_decision_elapsed);
        self.history_cost
            .observe_ordinary_target(ordinary_decision_elapsed);
        super::super::qwen35_speculation::record_outcome(0, 0, 0, 1, 0);
        if decision.terminal {
            self.finish_reason = "stop";
            return Ok(Qwen35TickOutcome {
                fragment: String::new(),
                is_reasoning: false,
                finished: true,
            });
        }
        self.pending_speculation_output.push_back(decision.token);
        let token = self
            .pending_speculation_output
            .pop_front()
            .expect("coherent ordinary decision queued above");
        self.commit_speculation_output(qwen, token)
    }
}

#[cfg(test)]
#[path = "ordinary_forward_controls.rs"]
mod controls;
