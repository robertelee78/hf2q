//! Transactional fresh-prefill aggregation for Qwen generative families.
//!
//! The scheduler remains in `engine.rs`. This module owns the family-local
//! semantic transaction: all selected target and MTP cache state advances
//! together, checkpoints are validated before publication, and a caller may
//! still roll the prepared cohort back when cancellation races GPU work.

use super::super::anchor_store::StagePending;
use super::*;

const MAX_RECTANGULAR_PREFILL_ROWS: usize = 128;

pub(crate) fn rectangular_cold_request_rows(
    prompt_len: usize,
    stable_boundary: Option<usize>,
) -> Option<usize> {
    rectangular_stable_boundary_rows(
        0,
        0,
        prompt_len,
        stable_boundary,
        MAX_RECTANGULAR_PREFILL_ROWS,
    )
}

fn rectangular_stable_boundary_rows(
    cached_tokens: usize,
    next_token_index: usize,
    prompt_len: usize,
    stable_boundary: Option<usize>,
    requested_tokens: usize,
) -> Option<usize> {
    if cached_tokens != 0 || next_token_index != 0 {
        return None;
    }
    let boundary = stable_boundary?;
    if boundary >= prompt_len {
        return None;
    }
    let row_limit = requested_tokens.min(MAX_RECTANGULAR_PREFILL_ROWS);
    let end = qwen35_next_prefill_end(0, prompt_len, row_limit, stable_boundary);
    (end == boundary && (16..=MAX_RECTANGULAR_PREFILL_ROWS).contains(&end)).then_some(end)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen35RectangularPrefillPlan {
    pub(crate) rows: usize,
    pub(crate) mtp_prefill: bool,
    pub(crate) checkpoint_at_end: bool,
}

fn rectangular_execution_matches(
    candidate: Qwen35RectangularPrefillPlan,
    admitted: Qwen35RectangularPrefillPlan,
) -> bool {
    candidate.rows == admitted.rows
        && candidate.mtp_prefill == admitted.mtp_prefill
        && (!admitted.checkpoint_at_end || candidate.checkpoint_at_end)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen35MtpPrefillOutcome {
    NotRequested,
    Succeeded,
    OrdinaryReplay,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen35CheckpointStagingOutcome {
    Staged,
    CapacitySuppressed { lane: usize, outcome: StagePending },
    InvariantViolation { lane: usize, outcome: StagePending },
}

fn checkpoint_staging_rejection(
    lane: usize,
    outcome: StagePending,
) -> Option<Qwen35CheckpointStagingOutcome> {
    match outcome {
        StagePending::Staged => None,
        StagePending::NoCommittedCapacity | StagePending::BudgetExceeded { .. } => {
            Some(Qwen35CheckpointStagingOutcome::CapacitySuppressed { lane, outcome })
        }
        StagePending::PendingOccupied => {
            Some(Qwen35CheckpointStagingOutcome::InvariantViolation { lane, outcome })
        }
    }
}

pub(crate) struct Qwen35PreparedPrefillCohort {
    advances: Vec<Qwen35PrefillAdvance>,
    transactions: Vec<(SlotId, HybridKvSlotTransaction)>,
    retry_states: Vec<Qwen35PrefillState>,
    mtp_outcome: Qwen35MtpPrefillOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen35MtpCatchupFailureDisposition {
    FailStop,
    ReplayOrdinaryTarget,
}

fn classify_mtp_catchup_failure(
    supervisor: &EngineSupervisor,
    error: &anyhow::Error,
) -> Qwen35MtpCatchupFailureDisposition {
    if !supervisor.is_healthy() || super::super::engine::is_fatal_command_buffer_error(error) {
        Qwen35MtpCatchupFailureDisposition::FailStop
    } else {
        Qwen35MtpCatchupFailureDisposition::ReplayOrdinaryTarget
    }
}

impl Qwen35PreparedPrefillCohort {
    pub(crate) fn advances(&self) -> &[Qwen35PrefillAdvance] {
        &self.advances
    }

    pub(crate) fn mtp_outcome(&self) -> Qwen35MtpPrefillOutcome {
        self.mtp_outcome
    }

    pub(crate) fn commit(self) -> Vec<Qwen35PrefillAdvance> {
        self.advances
    }

    pub(crate) fn take_checkpoints(&mut self) -> Result<Vec<Qwen35StablePromptCheckpoint>> {
        self.advances
            .iter_mut()
            .map(|advance| match advance {
                Qwen35PrefillAdvance::Pending { checkpoint, .. } => checkpoint
                    .take()
                    .context("rectangular Qwen advance omitted its stable checkpoint"),
                Qwen35PrefillAdvance::Ready { .. } => {
                    anyhow::bail!("rectangular Qwen stable-boundary advance completed a prompt")
                }
            })
            .collect()
    }

    pub(crate) fn stage_optional_checkpoints(
        &mut self,
        mut stage: impl FnMut(usize, Qwen35StablePromptCheckpoint) -> StagePending,
    ) -> Result<Qwen35CheckpointStagingOutcome> {
        for (lane, checkpoint) in self.take_checkpoints()?.into_iter().enumerate() {
            let outcome = stage(lane, checkpoint);
            if let Some(rejection) = checkpoint_staging_rejection(lane, outcome) {
                return Ok(rejection);
            }
        }
        Ok(Qwen35CheckpointStagingOutcome::Staged)
    }

    pub(crate) fn rollback_for_retry(
        self,
        kv_cache: &mut HybridKvCache,
        reason: &str,
    ) -> Result<Vec<Qwen35PrefillState>> {
        rollback_rectangular_transactions(kv_cache, &self.transactions, reason)?;
        Ok(self.retry_states)
    }
}

impl Qwen35PrefillState {
    /// Return the exact cold scalar slice shape that a rectangular execution
    /// may replace. This is a no-wait eligibility decision: callers group only
    /// already-admitted FIFO peers with an identical plan.
    pub(crate) fn rectangular_prefill_plan(
        &self,
        qwen: &Qwen35LoadedModel,
        kv_cache: &HybridKvCache,
        requested_tokens: usize,
    ) -> Option<Qwen35RectangularPrefillPlan> {
        if self.cached_prefill_logits.is_some()
            || self.vision.is_some()
            || self.mtp_pending_hidden.is_some()
        {
            return None;
        }
        let end = rectangular_stable_boundary_rows(
            self.cached_tokens,
            self.next_token_index,
            self.prompt_tokens.len(),
            self.stable_prompt_prefix_tokens,
            requested_tokens,
        )?;
        let mtp_prefill = !self.speculation_unavailable
            && qwen.speculation.policy()
                == super::super::qwen35_speculation::QwenSpeculationPolicy::Auto
            && is_qwen_server_speculation_exact_eligible(&self.params)
            && qwen.model.mtp.is_some()
            && kv_cache.mtp_slot.is_some();
        Some(Qwen35RectangularPrefillPlan {
            rows: end,
            mtp_prefill,
            checkpoint_at_end: true,
        })
    }

    /// Execute one exact rectangular target transaction and prepare, but do
    /// not publish, every lane's scheduler-visible advance.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::serve::api) fn advance_rectangular_prefill(
        mut states: Vec<Self>,
        plan: Qwen35RectangularPrefillPlan,
        qwen: &mut Qwen35LoadedModel,
        kv_cache: &mut HybridKvCache,
        reservations: &[Option<Qwen35CompoundCheckpointReservation>],
        supervisor: &EngineSupervisor,
    ) -> Result<Qwen35PreparedPrefillCohort> {
        anyhow::ensure!(
            (2..=4).contains(&states.len()),
            "rectangular Qwen prefill requires two through four states"
        );
        anyhow::ensure!(
            states.len() == reservations.len(),
            "rectangular Qwen prefill state/reservation count mismatch"
        );
        for state in &states {
            let candidate = state.rectangular_prefill_plan(qwen, kv_cache, plan.rows);
            anyhow::ensure!(
                candidate.is_some_and(|candidate| rectangular_execution_matches(candidate, plan)),
                "rectangular Qwen prefill state no longer matches its admitted plan"
            );
        }
        if plan.checkpoint_at_end {
            anyhow::ensure!(
                reservations.iter().all(|reservation| {
                    reservation.is_some_and(|reservation| reservation.boundary == plan.rows)
                }),
                "rectangular Qwen stable-boundary prefill requires an admitted checkpoint reservation for every lane"
            );
        }

        let mut retry_states = states
            .iter()
            .map(Qwen35PrefillState::clone_cold_rectangular_retry)
            .collect::<Result<Vec<_>>>()?;
        let slot_ids = states.iter().map(|state| state.slot_id).collect::<Vec<_>>();
        qwen.model
            .validate_gpu_rectangular_prefill(kv_cache, &slot_ids, plan.rows)
            .context("preflight rectangular Qwen target route")?;
        if plan.mtp_prefill {
            for slot_id in &slot_ids {
                kv_cache
                    .validate_speculative_cursors_for_slot(*slot_id, 0)
                    .with_context(|| {
                        format!(
                            "rectangular Qwen slot {} MTP cache is not at the cold transaction boundary",
                            slot_id.0
                        )
                    })?;
            }
        }
        let tokens = states
            .iter()
            .flat_map(|state| state.prompt_tokens[..plan.rows].iter().copied())
            .collect::<Vec<_>>();
        let mut positions = Vec::with_capacity(4 * tokens.len());
        for axis in 0..4 {
            for _state in &states {
                positions.extend((0..plan.rows).map(|position| position as i32));
            }
            debug_assert_eq!(positions.len(), (axis + 1) * tokens.len());
        }

        let mut transactions = capture_rectangular_transactions(kv_cache, &slot_ids)?;
        let lease = supervisor.arm(
            "Qwen35 rectangular bounded prefill",
            QWEN35_WORKER_TRANSACTION_TIMEOUT,
        )?;
        let target_result = qwen.model.forward_gpu_rectangular_prefill_last_logits(
            &tokens,
            &positions,
            kv_cache,
            &slot_ids,
            plan.mtp_prefill,
        );
        let target_result = target_result.and_then(|outputs| {
            qwen35_state_failpoint(Qwen35StateFailpoint::PrefillCohortTarget)?;
            Ok(outputs)
        });
        let mut outputs = match target_result {
            Ok(outputs) => outputs,
            Err(target) => {
                let error = match lease.finish() {
                    Ok(()) => target,
                    Err(supervision) => anyhow::anyhow!("{target:#}; supervisor: {supervision:#}"),
                };
                return rollback_rectangular_error(
                    kv_cache,
                    &transactions,
                    error,
                    "rectangular Qwen target prefill",
                );
            }
        };
        let mut mtp_outcome = Qwen35MtpPrefillOutcome::NotRequested;
        if outputs.len() != states.len() {
            let output_count = outputs.len();
            drop(outputs);
            let supervision = lease.finish();
            let mismatch = anyhow::anyhow!(
                "rectangular Qwen target returned {output_count} outputs for {} states; supervisor: {supervision:?}",
                states.len()
            );
            return rollback_rectangular_error(
                kv_cache,
                &transactions,
                mismatch,
                "rectangular Qwen target output cardinality",
            );
        }

        if plan.mtp_prefill {
            let catchup = catch_up_rectangular_mtp(&states, &outputs, qwen, kv_cache, plan.rows);
            let supervision = lease.finish();
            if let Err(supervision) = supervision {
                return rollback_rectangular_error(
                    kv_cache,
                    &transactions,
                    supervision,
                    "rectangular Qwen target/MTP supervision",
                );
            }
            match catchup {
                Ok(hidden_rows) => {
                    mtp_outcome = Qwen35MtpPrefillOutcome::Succeeded;
                    for (state, hidden) in states.iter_mut().zip(hidden_rows) {
                        state.mtp_pending_hidden = Some(hidden);
                        super::super::qwen35_speculation::record_outcome(0, 0, 0, 1, 0);
                    }
                }
                Err(catchup_error) => {
                    if classify_mtp_catchup_failure(supervisor, &catchup_error)
                        == Qwen35MtpCatchupFailureDisposition::FailStop
                    {
                        return rollback_rectangular_error(
                            kv_cache,
                            &transactions,
                            catchup_error,
                            "rectangular Qwen fatal MTP prompt catch-up",
                        );
                    }
                    tracing::warn!(
                        lanes = states.len(),
                        error = %catchup_error,
                        "Qwen rectangular MTP prompt catch-up unavailable; replaying the cohort through ordinary target prefill"
                    );
                    mtp_outcome = Qwen35MtpPrefillOutcome::OrdinaryReplay;
                    rollback_rectangular_transactions(
                        kv_cache,
                        &transactions,
                        "rollback rectangular MTP prompt catch-up",
                    )?;
                    for state in &mut states {
                        state.mtp_pending_hidden = None;
                        state.speculation_unavailable = true;
                        super::super::qwen35_speculation::record_fallback(
                            super::super::qwen35_speculation::QwenSpeculationDecision::RuntimeUnavailable,
                        );
                    }
                    for state in &mut retry_states {
                        state.speculation_unavailable = true;
                    }
                    transactions = capture_rectangular_transactions(kv_cache, &slot_ids)?;
                    let replay_lease = supervisor.arm(
                        "Qwen35 rectangular ordinary prefill replay",
                        QWEN35_WORKER_TRANSACTION_TIMEOUT,
                    )?;
                    let replay = qwen.model.forward_gpu_rectangular_prefill_last_logits(
                        &tokens, &positions, kv_cache, &slot_ids, false,
                    );
                    outputs = match (replay, replay_lease.finish()) {
                        (Ok(outputs), Ok(())) => outputs,
                        (Err(error), Ok(())) => {
                            return rollback_rectangular_error(
                                kv_cache,
                                &transactions,
                                error,
                                "rectangular Qwen ordinary replay",
                            );
                        }
                        (Ok(_), Err(error)) => {
                            return rollback_rectangular_error(
                                kv_cache,
                                &transactions,
                                error,
                                "rectangular Qwen ordinary replay supervision",
                            );
                        }
                        (Err(replay), Err(supervision)) => {
                            return rollback_rectangular_error(
                                kv_cache,
                                &transactions,
                                anyhow::anyhow!("{replay:#}; supervisor: {supervision:#}"),
                                "rectangular Qwen ordinary replay and supervision failure",
                            );
                        }
                    };
                }
            }
        } else if let Err(supervision) = lease.finish() {
            return rollback_rectangular_error(
                kv_cache,
                &transactions,
                supervision,
                "rectangular Qwen target supervision",
            );
        }
        if outputs.len() != states.len() {
            let mismatch = anyhow::anyhow!(
                "rectangular Qwen target returned {} final outputs for {} states",
                outputs.len(),
                states.len()
            );
            return rollback_rectangular_error(
                kv_cache,
                &transactions,
                mismatch,
                "rectangular Qwen final output cardinality",
            );
        }

        let advances_result = (|| -> Result<Vec<Qwen35PrefillAdvance>> {
            let mut advances = Vec::with_capacity(states.len());
            for ((mut state, output), reservation) in states
                .into_iter()
                .zip(outputs)
                .zip(reservations.iter().copied())
            {
                kv_cache
                    .validate_sequence_len_for_slot(state.slot_id, plan.rows)
                    .with_context(|| {
                        format!(
                            "validate rectangular Qwen slot {} target cursor",
                            state.slot_id.0
                        )
                    })?;
                if plan.mtp_prefill && !state.speculation_unavailable {
                    kv_cache
                        .validate_speculative_cursors_for_slot(state.slot_id, plan.rows)
                        .with_context(|| {
                            format!(
                                "validate rectangular Qwen slot {} target/MTP cursor equality",
                                state.slot_id.0
                            )
                        })?;
                }

                let checkpoint = if plan.checkpoint_at_end
                    && reservation.is_some_and(|reservation| reservation.boundary == plan.rows)
                {
                    let capture_started = Instant::now();
                    let spec = state.mtp_pending_hidden.as_ref().and_then(|hidden| {
                    let boundary = Qwen35SpecPrefixBoundary {
                        token_count: plan.rows,
                        pending_target_hidden: hidden.clone(),
                    };
                    match boundary.clone_owned(&qwen.model) {
                        Ok(boundary) => Some(boundary),
                        Err(error) => {
                            tracing::warn!(
                                slot = state.slot_id.0,
                                error = %error,
                                "Qwen rectangular checkpoint dropped speculative metadata because its hidden row could not be detached"
                            );
                            super::super::qwen35_speculation::record_fallback(
                                super::super::qwen35_speculation::QwenSpeculationDecision::RuntimeUnavailable,
                            );
                            None
                        }
                    }
                });
                    let checkpoint = Qwen35StablePromptCheckpoint {
                        prompt_tokens: state.prompt_tokens[..plan.rows].to_vec(),
                        kv: kv_cache
                            .snapshot_slot_anchor(state.slot_id, plan.rows)
                            .with_context(|| {
                                format!(
                                    "capture rectangular Qwen slot {} stable boundary",
                                    state.slot_id.0
                                )
                            })?,
                        prefill_logits: output.logits.clone(),
                        vision_fingerprint: state.params.vision_fingerprint,
                        spec,
                        capture_duration: capture_started.elapsed(),
                    };
                    let reservation = reservation.expect("checkpoint branch requires reservation");
                    anyhow::ensure!(
                        checkpoint.owned_bytes() <= reservation.retained_bytes,
                        "rectangular Qwen checkpoint exact bytes {} exceeded reserved bytes {}",
                        checkpoint.owned_bytes(),
                        reservation.retained_bytes
                    );
                    Some(checkpoint)
                } else {
                    None
                };
                state.next_token_index = plan.rows;
                let advanced_tokens = plan.rows;
                advances.push(Qwen35PrefillAdvance::Pending {
                    state,
                    advanced_tokens,
                    checkpoint,
                });
            }
            qwen35_state_failpoint(Qwen35StateFailpoint::PrefillCohortCheckpoint)
                .map_err(|error| error.context("rectangular Qwen post-checkpoint failpoint"))?;
            Ok(advances)
        })();
        let advances = match advances_result {
            Ok(advances) => advances,
            Err(error) => {
                return rollback_rectangular_error(
                    kv_cache,
                    &transactions,
                    error,
                    "prepare rectangular Qwen publication state",
                );
            }
        };

        Ok(Qwen35PreparedPrefillCohort {
            advances,
            transactions,
            retry_states,
            mtp_outcome,
        })
    }

    fn clone_cold_rectangular_retry(&self) -> Result<Self> {
        anyhow::ensure!(
            self.cached_tokens == 0
                && self.next_token_index == 0
                && self.cached_prefill_logits.is_none()
                && self.vision.is_none()
                && self.mtp_pending_hidden.is_none(),
            "rectangular Qwen retry clone requires a cold text prefill state"
        );
        Ok(Self {
            slot_id: self.slot_id,
            prompt_tokens: self.prompt_tokens.clone(),
            params: self.params.clone(),
            cached_tokens: 0,
            next_token_index: 0,
            cached_prefill_logits: None,
            stable_prompt_prefix_tokens: self.stable_prompt_prefix_tokens,
            vision: None,
            mtp_pending_hidden: None,
            speculation_unavailable: self.speculation_unavailable,
            prefill_started: self.prefill_started,
        })
    }
}

fn capture_rectangular_transactions(
    kv_cache: &HybridKvCache,
    slot_ids: &[SlotId],
) -> Result<Vec<(SlotId, HybridKvSlotTransaction)>> {
    slot_ids
        .iter()
        .copied()
        .map(|slot_id| {
            begin_slot_state_transaction(kv_cache, slot_id, 0)
                .map(|transaction| (slot_id, transaction))
        })
        .collect()
}

fn rollback_rectangular_transactions(
    kv_cache: &mut HybridKvCache,
    transactions: &[(SlotId, HybridKvSlotTransaction)],
    reason: &str,
) -> Result<()> {
    kv_cache.clear_la_capture();
    let mut failures = Vec::new();
    for (slot_id, transaction) in transactions.iter().rev() {
        if let Err(error) = kv_cache.rollback_slot_transaction(*slot_id, transaction) {
            failures.push(format!("slot {}: {error:#}", slot_id.0));
        }
    }
    if failures.is_empty() {
        return Ok(());
    }
    let mut reset_failures = Vec::new();
    for (slot_id, _) in transactions {
        if let Err(error) = kv_cache.reset_for_slot(*slot_id) {
            reset_failures.push(format!("slot {}: {error:#}", slot_id.0));
        }
    }
    anyhow::bail!(
        "{reason}: all-lane rollback failed: {}; fail-closed reset failures: {}",
        failures.join("; "),
        if reset_failures.is_empty() {
            "none".to_string()
        } else {
            reset_failures.join("; ")
        }
    )
}

fn rollback_rectangular_error<T>(
    kv_cache: &mut HybridKvCache,
    transactions: &[(SlotId, HybridKvSlotTransaction)],
    error: anyhow::Error,
    context: &'static str,
) -> Result<T> {
    let original = error.context(context);
    match rollback_rectangular_transactions(kv_cache, transactions, context) {
        Ok(()) => Err(original),
        Err(rollback) => Err(original.context(format!(
            "rectangular Qwen rollback also failed: {rollback:#}"
        ))),
    }
}

fn catch_up_rectangular_mtp(
    states: &[Qwen35PrefillState],
    outputs: &[crate::inference::models::qwen35::forward_gpu::RectangularPrefillLaneOutput],
    qwen: &Qwen35LoadedModel,
    kv_cache: &mut HybridKvCache,
    rows: usize,
) -> Result<Vec<MlxBuffer>> {
    anyhow::ensure!(
        states.len() == outputs.len(),
        "rectangular Qwen MTP state/output count mismatch"
    );
    let mtp = qwen
        .model
        .mtp
        .as_ref()
        .context("Qwen rectangular MTP prompt catch-up weights missing")?;
    let mut hidden_rows = Vec::with_capacity(states.len());
    for (state, output) in states.iter().zip(outputs) {
        let chunk = &state.prompt_tokens[..rows];
        let positions = prefill_positions_from(0, rows);
        let target_nextn = output
            .nextn_hidden
            .as_ref()
            .context("Qwen rectangular target omitted required MTP hidden rows")?;
        let shared_embed_rows = qwen.model.embed_tokens_gpu(chunk)?;
        qwen.model.with_gpu_cache_mut(|device, registry| {
            mtp.process_target_batch(
                chunk,
                state.mtp_pending_hidden.as_ref(),
                target_nextn,
                &shared_embed_rows,
                kv_cache,
                state.slot_id,
                &positions,
                device,
                registry,
                &qwen.model.cfg,
            )
        })?;
        kv_cache
            .validate_speculative_cursors_for_slot(state.slot_id, rows)
            .with_context(|| {
                format!(
                    "Qwen rectangular slot {} target/MTP cursor equality",
                    state.slot_id.0
                )
            })?;
        hidden_rows.push(
            crate::inference::models::qwen35::spec_decode::last_hidden_row(
                target_nextn,
                qwen.model.cfg.hidden_size,
            )?,
        );
        anyhow::ensure!(
            hidden_rows
                .last()
                .is_some_and(|hidden| hidden.element_count() == qwen.model.cfg.hidden_size as usize),
            "Qwen rectangular MTP prefill hidden must be one row"
        );
    }
    qwen35_state_failpoint(Qwen35StateFailpoint::PrefillCohortMtp)?;
    qwen35_state_failpoint(Qwen35StateFailpoint::PrefillCohortMtpFatal)?;
    Ok(hidden_rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::api::anchor_store::{AnchorEntry, AnchorPublicationDisposition, AnchorStore};
    use crate::serve::api::qwen35_anchor_store::discard_cohort_pending;
    use mlx_native::MlxDevice;

    #[derive(Clone, Debug)]
    struct FakeAnchor {
        token_count: usize,
        epoch: u64,
        bytes: u64,
        publication_disposition: AnchorPublicationDisposition,
    }

    impl FakeAnchor {
        fn new(token_count: usize, bytes: u64) -> Self {
            Self {
                token_count,
                epoch: u64::MAX,
                bytes,
                publication_disposition: AnchorPublicationDisposition::Unpublished,
            }
        }
    }

    impl AnchorEntry for FakeAnchor {
        fn token_count(&self) -> usize {
            self.token_count
        }

        fn lineage_epoch(&self) -> u64 {
            self.epoch
        }

        fn set_lineage_epoch(&mut self, epoch: u64) {
            self.epoch = epoch;
        }

        fn owned_bytes(&self) -> u64 {
            self.bytes
        }

        fn publication_disposition(&self) -> AnchorPublicationDisposition {
            self.publication_disposition
        }

        fn set_publication_disposition(&mut self, disposition: AnchorPublicationDisposition) {
            self.publication_disposition = disposition;
        }
    }

    fn rectangular_test_model_with_mtp(mtp_enabled: bool) -> Qwen35LoadedModel {
        let mut cfg =
            crate::inference::models::qwen35::forward_gpu::tests::physical_batch_hybrid_cfg();
        cfg.max_position_embeddings = 256;
        cfg.mtp_num_hidden_layers = u32::from(mtp_enabled);
        let hidden_size = cfg.hidden_size as usize;
        let vocab_size = cfg.vocab_size as usize;
        let mut model =
            crate::inference::models::qwen35::forward_gpu::tests::deterministic_dense_model(
                cfg.clone(),
            );
        if mtp_enabled {
            model.ensure_gpu_cache_primed().expect("prime target cache");
            let mtp = model
                .with_gpu_cache_mut(|device, _registry| {
                    Ok(
                        crate::inference::models::qwen35::mtp::tests::load_nonzero_mtp_fixture_for_cfg(
                            device, &cfg,
                        ),
                    )
                })
                .expect("load D256 MTP fixture on target device");
            model.mtp = Some(mtp);
        }
        Qwen35LoadedModel {
            model,
            tokenizer: tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            chat_template: "{{ messages }}".to_string(),
            model_id: "rectangular-state-test".to_string(),
            model_path: PathBuf::from("rectangular-state-test.gguf"),
            eos_token_ids: vec![127],
            hidden_size,
            vocab_size,
            context_length: Some(256),
            quant_type: Some("test-f32".to_string()),
            load_duration: Duration::ZERO,
            provenance: crate::core::provenance::Provenance::External,
            expected_projector_sha256: None,
            vision_projector_profile: None,
            vision_deepstack_output_count: None,
            vision_special_tokens_present: false,
            prompt_cache: HybridPromptCache::new(),
            lcp_registry: crate::serve::kv_persist::lcp_registry::LcpRegistry::new(1),
            kv_metrics_sink: None,
            disk_persistor: None,
            lcp_hydrated_for_cfg: std::collections::HashSet::new(),
            tq_kv_active: true,
            persistent_kv_cache: None,
            speculation: super::super::super::qwen35_speculation::QwenSpeculationController::new(
                if mtp_enabled {
                    super::super::super::qwen35_speculation::QwenSpeculationPolicy::Auto
                } else {
                    super::super::super::qwen35_speculation::QwenSpeculationPolicy::Off
                },
            ),
        }
    }

    fn rectangular_test_model() -> Qwen35LoadedModel {
        rectangular_test_model_with_mtp(false)
    }

    fn rectangular_test_states(
        qwen: &Qwen35LoadedModel,
        kv: &mut HybridKvCache,
        slots: &[SlotId],
    ) -> (
        Vec<Qwen35PrefillState>,
        Qwen35RectangularPrefillPlan,
        Vec<Option<Qwen35CompoundCheckpointReservation>>,
    ) {
        let mut states = Vec::with_capacity(slots.len());
        for (lane, slot) in slots.iter().copied().enumerate() {
            let prompt = (0..128)
                .map(|row| 3 + ((lane * 29 + row * 7) as u32 % 124))
                .collect::<Vec<_>>();
            let mut params = SamplingParams::default();
            params.max_tokens = 8;
            params.stable_prompt_prefix_tokens = Some(121);
            states.push(
                Qwen35PrefillState::begin(
                    prompt,
                    params,
                    None,
                    kv,
                    slot,
                    0,
                    None,
                    None,
                    None,
                    qwen.hidden_size,
                )
                .expect("state"),
            );
        }
        let plan = states[0]
            .rectangular_prefill_plan(qwen, kv, 2_048)
            .expect("plan");
        assert!(states
            .iter()
            .all(|state| state.rectangular_prefill_plan(qwen, kv, 2_048) == Some(plan)));
        let reservations = states
            .iter()
            .map(|state| {
                state
                    .compound_checkpoint_reservation(
                        kv,
                        plan.rows,
                        qwen.hidden_size,
                        qwen.vocab_size,
                    )
                    .expect("reservation")
            })
            .collect();
        (states, plan, reservations)
    }

    #[test]
    fn stable_boundary_shape_gate_matches_the_measured_tiled_route() {
        for boundary in [16, 121, 128] {
            assert_eq!(
                rectangular_stable_boundary_rows(0, 0, boundary + 7, Some(boundary), 2_048),
                Some(boundary)
            );
        }
        for (cached, cursor, prompt, boundary, requested) in [
            (0, 0, 128, None, 128),
            (0, 0, 128, Some(15), 128),
            (0, 0, 130, Some(129), 2_048),
            (0, 0, 121, Some(121), 2_048),
            (1, 1, 128, Some(121), 2_048),
            (0, 0, 128, Some(121), 120),
        ] {
            assert_eq!(
                rectangular_stable_boundary_rows(cached, cursor, prompt, boundary, requested),
                None,
                "unexpected admission for cached={cached} cursor={cursor} prompt={prompt} boundary={boundary:?} requested={requested}"
            );
        }
    }

    #[test]
    fn fatal_mtp_catchup_never_selects_ordinary_gpu_replay() {
        let supervisor = EngineSupervisor::new();
        let fatal = anyhow::Error::new(mlx_native::MlxError::CommandBufferError(
            "injected rectangular MTP timeout".to_string(),
        ))
        .context("MTP catch-up");
        assert_eq!(
            classify_mtp_catchup_failure(&supervisor, &fatal),
            Qwen35MtpCatchupFailureDisposition::FailStop
        );
        assert_eq!(
            classify_mtp_catchup_failure(
                &supervisor,
                &anyhow::anyhow!("ordinary MTP feature-unavailable error"),
            ),
            Qwen35MtpCatchupFailureDisposition::ReplayOrdinaryTarget
        );
        supervisor.poison_now("test rectangular MTP supervisor poison");
        assert_eq!(
            classify_mtp_catchup_failure(
                &supervisor,
                &anyhow::anyhow!("otherwise recoverable MTP error"),
            ),
            Qwen35MtpCatchupFailureDisposition::FailStop
        );
    }

    #[test]
    fn pending_occupied_is_an_invariant_not_capacity_pressure() {
        assert_eq!(
            checkpoint_staging_rejection(1, StagePending::PendingOccupied),
            Some(Qwen35CheckpointStagingOutcome::InvariantViolation {
                lane: 1,
                outcome: StagePending::PendingOccupied,
            })
        );
        assert!(matches!(
            checkpoint_staging_rejection(
                1,
                StagePending::BudgetExceeded {
                    needed_bytes: 2,
                    budget_bytes: 1,
                },
            ),
            Some(Qwen35CheckpointStagingOutcome::CapacitySuppressed { lane: 1, .. })
        ));
    }

    #[test]
    fn all_lane_rollback_restores_selected_slots_and_not_the_peer() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let mut cfg = crate::inference::models::qwen35::mtp::tests::tiny_nonzero_mtp_cfg();
        cfg.mtp_num_hidden_layers = 1;
        let mut kv = HybridKvCache::new(&cfg, &device, 32, 3).expect("cache");
        let selected = [SlotId(2), SlotId(0)];
        let transactions = capture_rectangular_transactions(&kv, &selected).expect("capture");
        let peer_before = (
            kv.full_attn
                .iter()
                .map(|full| full.current_len[1])
                .collect::<Vec<_>>(),
            kv.mtp_slot.as_ref().expect("MTP slot").current_len[1],
            kv.linear_attn
                .iter()
                .map(|linear| linear.pp_flipped[1])
                .collect::<Vec<_>>(),
        );

        for full in &mut kv.full_attn {
            full.current_len[2] = 17;
            full.current_len[0] = 17;
        }
        kv.mtp_slot.as_mut().expect("MTP slot").current_len[2] = 17;
        kv.mtp_slot.as_mut().expect("MTP slot").current_len[0] = 17;
        for linear in &mut kv.linear_attn {
            linear.swap_for_slot(SlotId(2));
            linear.swap_for_slot(SlotId(0));
        }

        rollback_rectangular_transactions(&mut kv, &transactions, "model-free cohort rollback")
            .expect("rollback");
        for slot in selected {
            assert_eq!(kv.sequence_len_for_slot(slot).expect("cursor"), 0);
            assert_eq!(
                kv.mtp_slot.as_ref().expect("MTP slot").current_len[slot.0 as usize],
                0
            );
            assert!(kv
                .linear_attn
                .iter()
                .all(|linear| !linear.pp_flipped[slot.0 as usize]));
        }
        assert_eq!(
            (
                kv.full_attn
                    .iter()
                    .map(|full| full.current_len[1])
                    .collect::<Vec<_>>(),
                kv.mtp_slot.as_ref().expect("MTP slot").current_len[1],
                kv.linear_attn
                    .iter()
                    .map(|linear| linear.pp_flipped[1])
                    .collect::<Vec<_>>(),
            ),
            peer_before,
        );
    }

    #[test]
    fn rectangular_state_executor_prepares_all_checkpoints_and_can_retry_atomically() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let mut qwen = rectangular_test_model();
        let mut kv =
            HybridKvCache::new_with_options(&qwen.model.cfg, &device, 256, 3, true).expect("cache");
        let slots = [SlotId(2), SlotId(0)];
        let (states, plan, reservations) = rectangular_test_states(&qwen, &mut kv, &slots);
        let supervisor = EngineSupervisor::new();
        let mut prepared = Qwen35PrefillState::advance_rectangular_prefill(
            states,
            plan,
            &mut qwen,
            &mut kv,
            &reservations,
            &supervisor,
        )
        .expect("prepare");
        assert_eq!(prepared.advances().len(), slots.len());
        for (advance, slot) in prepared.advances().iter().zip(slots) {
            advance
                .validate_checkpoint_for_install(slot, qwen.vocab_size, qwen.hidden_size)
                .expect("checkpoint validation");
            assert_eq!(kv.sequence_len_for_slot(slot).expect("cursor"), 121);
        }
        assert_eq!(prepared.take_checkpoints().expect("checkpoints").len(), 2);
        let retry = prepared
            .rollback_for_retry(&mut kv, "test post-submit cancellation")
            .expect("rollback");
        assert_eq!(retry.len(), slots.len());
        for (state, slot) in retry.iter().zip(slots) {
            assert_eq!(kv.sequence_len_for_slot(slot).expect("cursor"), 0);
            assert_eq!(
                state.rectangular_prefill_plan(&qwen, &kv, 2_048),
                Some(plan)
            );
        }
    }

    #[test]
    fn rectangular_state_executor_keeps_the_cohort_when_checkpointing_is_suppressed() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let mut qwen = rectangular_test_model();
        let mut kv =
            HybridKvCache::new_with_options(&qwen.model.cfg, &device, 256, 3, true).expect("cache");
        let slots = [SlotId(2), SlotId(0)];
        let (states, mut plan, _) = rectangular_test_states(&qwen, &mut kv, &slots);
        plan.checkpoint_at_end = false;
        let reservations = vec![None; slots.len()];
        let supervisor = EngineSupervisor::new();

        let prepared = Qwen35PrefillState::advance_rectangular_prefill(
            states,
            plan,
            &mut qwen,
            &mut kv,
            &reservations,
            &supervisor,
        )
        .expect("checkpoint capacity is optional to rectangular execution");

        for (advance, slot) in prepared.advances().iter().zip(slots) {
            let Qwen35PrefillAdvance::Pending { checkpoint, .. } = advance else {
                panic!("stable-boundary rectangular slice must remain pending");
            };
            assert!(checkpoint.is_none());
            assert_eq!(
                kv.sequence_len_for_slot(slot).expect("cursor"),
                plan.rows as u32
            );
        }
        assert_eq!(prepared.commit().len(), slots.len());
    }

    #[test]
    fn partial_checkpoint_staging_preserves_lineage_and_commits_target_advances() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let mut qwen = rectangular_test_model();
        let mut kv =
            HybridKvCache::new_with_options(&qwen.model.cfg, &device, 256, 3, true).expect("cache");
        let slots = [SlotId(2), SlotId(0)];
        let (states, plan, reservations) = rectangular_test_states(&qwen, &mut kv, &slots);
        let supervisor = EngineSupervisor::new();
        let mut prepared = Qwen35PrefillState::advance_rectangular_prefill(
            states,
            plan,
            &mut qwen,
            &mut kv,
            &reservations,
            &supervisor,
        )
        .expect("prepare");

        let mut stores = (0..slots.len())
            .map(|_| AnchorStore::with_committed_capacity(4))
            .collect::<Vec<AnchorStore<FakeAnchor>>>();
        for (lane, store) in stores.iter_mut().enumerate() {
            assert_eq!(
                store.stage_pending(FakeAnchor::new(32 + lane, 40 + lane as u64), 4, 10_000),
                StagePending::Staged
            );
            store.publish_pending(4).expect("seed committed lineage");
        }
        let committed_before = stores
            .iter()
            .map(|store| {
                (
                    store.committed_token_counts(),
                    store.lineage_epoch(),
                    store.owned_bytes(),
                )
            })
            .collect::<Vec<_>>();

        let staging = prepared
            .stage_optional_checkpoints(|lane, checkpoint| {
                let budget = if lane == 0 {
                    10_000
                } else {
                    stores[lane].owned_bytes()
                };
                stores[lane].stage_pending(
                    FakeAnchor::new(checkpoint.prompt_tokens.len(), 73),
                    4,
                    budget,
                )
            })
            .expect("checkpoint extraction");
        assert!(matches!(
            staging,
            Qwen35CheckpointStagingOutcome::CapacitySuppressed {
                lane: 1,
                outcome: StagePending::BudgetExceeded { .. }
            }
        ));
        assert!(stores[0].has_pending());
        assert!(!stores[1].has_pending());
        assert_eq!(
            discard_cohort_pending(&mut stores, &[0, 1]).expect("cohort pending cleanup"),
            1
        );
        for (store, expected) in stores.iter().zip(&committed_before) {
            assert!(!store.has_pending());
            assert_eq!(store.committed_token_counts(), expected.0);
            assert_eq!(store.lineage_epoch(), expected.1);
            assert_eq!(store.owned_bytes(), expected.2);
        }

        let advances = prepared.commit();
        assert_eq!(advances.len(), slots.len());
        for (advance, slot) in advances.iter().zip(slots) {
            let Qwen35PrefillAdvance::Pending { checkpoint, .. } = advance else {
                panic!("rectangular target advance must remain publishable");
            };
            assert!(checkpoint.is_none());
            assert_eq!(
                kv.sequence_len_for_slot(slot).expect("cursor"),
                plan.rows as u32
            );
        }
    }

    #[test]
    fn nonfatal_mtp_failure_rewinds_every_lane_and_replays_once() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let mut qwen = rectangular_test_model_with_mtp(true);
        let mut kv =
            HybridKvCache::new_with_options(&qwen.model.cfg, &device, 256, 3, true).expect("cache");
        let slots = [SlotId(2), SlotId(0)];
        let (states, plan, reservations) = rectangular_test_states(&qwen, &mut kv, &slots);
        assert!(
            plan.mtp_prefill,
            "fixture must enter the real MTP catch-up path"
        );

        let tokens = states
            .iter()
            .flat_map(|state| state.prompt_tokens[..plan.rows].iter().copied())
            .collect::<Vec<_>>();
        let mut positions = Vec::with_capacity(4 * tokens.len());
        for _axis in 0..4 {
            for _state in &states {
                positions.extend((0..plan.rows).map(|position| position as i32));
            }
        }
        let mut ordinary_kv =
            HybridKvCache::new_with_options(&qwen.model.cfg, &device, 256, 3, true)
                .expect("ordinary cache");
        let ordinary_outputs = qwen
            .model
            .forward_gpu_rectangular_prefill_last_logits(
                &tokens,
                &positions,
                &mut ordinary_kv,
                &slots,
                false,
            )
            .expect("ordinary rectangular control");
        let ordinary_anchors = slots
            .iter()
            .map(|slot| {
                ordinary_kv
                    .snapshot_slot_anchor(*slot, plan.rows)
                    .expect("ordinary anchor")
            })
            .collect::<Vec<_>>();

        crate::inference::models::qwen35::forward_gpu::reset_test_rectangular_target_calls();
        QWEN35_STATE_FAILPOINT.store(
            Qwen35StateFailpoint::PrefillCohortMtp as u8,
            std::sync::atomic::Ordering::SeqCst,
        );
        let supervisor = EngineSupervisor::new();
        let mut prepared = Qwen35PrefillState::advance_rectangular_prefill(
            states,
            plan,
            &mut qwen,
            &mut kv,
            &reservations,
            &supervisor,
        )
        .expect("nonfatal MTP failure must replay ordinary target");
        QWEN35_STATE_FAILPOINT.store(0, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            prepared.mtp_outcome(),
            Qwen35MtpPrefillOutcome::OrdinaryReplay
        );
        assert_eq!(
            crate::inference::models::qwen35::forward_gpu::test_rectangular_target_calls(),
            2,
            "one MTP-capturing target plus exactly one ordinary replay"
        );
        for (advance, slot) in prepared.advances().iter().zip(slots) {
            let Qwen35PrefillAdvance::Pending { state, .. } = advance else {
                panic!("stable-boundary replay must remain pending")
            };
            assert!(state.speculation_unavailable);
            assert_eq!(
                kv.sequence_len_for_slot(slot).expect("target cursor"),
                plan.rows as u32
            );
            assert_eq!(
                kv.mtp_slot.as_ref().expect("MTP slot").current_len[slot.0 as usize],
                0,
                "ordinary replay must leave the rewound MTP cursor uncommitted"
            );
        }
        let checkpoints = prepared.take_checkpoints().expect("checkpoints");
        for (((checkpoint, ordinary), ordinary_anchor), slot) in checkpoints
            .iter()
            .zip(&ordinary_outputs)
            .zip(&ordinary_anchors)
            .zip(slots)
        {
            assert_eq!(
                checkpoint.prefill_logits, ordinary.logits,
                "slot {slot:?} logits"
            );
            assert!(
                checkpoint.kv.payload_eq(ordinary_anchor),
                "slot {slot:?} cache"
            );
            assert!(
                checkpoint.spec.is_none(),
                "fallback anchor must not claim MTP state"
            );
        }
        prepared
            .rollback_for_retry(&mut kv, "test cleanup")
            .expect("cleanup rollback");
    }

    #[test]
    fn typed_mtp_failure_rewinds_every_lane_without_gpu_replay() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let mut qwen = rectangular_test_model_with_mtp(true);
        let mut kv =
            HybridKvCache::new_with_options(&qwen.model.cfg, &device, 256, 3, true).expect("cache");
        let slots = [SlotId(2), SlotId(0)];
        let (states, plan, reservations) = rectangular_test_states(&qwen, &mut kv, &slots);
        assert!(plan.mtp_prefill);
        crate::inference::models::qwen35::forward_gpu::reset_test_rectangular_target_calls();
        QWEN35_STATE_FAILPOINT.store(
            Qwen35StateFailpoint::PrefillCohortMtpFatal as u8,
            std::sync::atomic::Ordering::SeqCst,
        );
        let supervisor = EngineSupervisor::new();
        let result = Qwen35PrefillState::advance_rectangular_prefill(
            states,
            plan,
            &mut qwen,
            &mut kv,
            &reservations,
            &supervisor,
        );
        QWEN35_STATE_FAILPOINT.store(0, std::sync::atomic::Ordering::SeqCst);
        let error = match result {
            Ok(_) => panic!("typed MTP command-buffer failure must escape"),
            Err(error) => error,
        };
        assert!(error.chain().any(|source| {
            matches!(
                source.downcast_ref::<mlx_native::MlxError>(),
                Some(mlx_native::MlxError::CommandBufferError(_))
            )
        }));
        assert_eq!(
            crate::inference::models::qwen35::forward_gpu::test_rectangular_target_calls(),
            1,
            "fatal MTP catch-up must not submit an ordinary target replay"
        );
        for slot in slots {
            assert_eq!(kv.sequence_len_for_slot(slot).expect("target cursor"), 0);
            assert_eq!(
                kv.mtp_slot.as_ref().expect("MTP slot").current_len[slot.0 as usize],
                0
            );
            assert!(kv
                .linear_attn
                .iter()
                .all(|linear| !linear.pp_flipped[slot.0 as usize]));
        }
    }

    #[test]
    fn rectangular_post_checkpoint_failure_rolls_back_every_lane() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let mut qwen = rectangular_test_model();
        let mut kv =
            HybridKvCache::new_with_options(&qwen.model.cfg, &device, 256, 3, true).expect("cache");
        let slots = [SlotId(2), SlotId(0)];
        let (states, plan, reservations) = rectangular_test_states(&qwen, &mut kv, &slots);
        let supervisor = EngineSupervisor::new();
        QWEN35_STATE_FAILPOINT.store(
            Qwen35StateFailpoint::PrefillCohortCheckpoint as u8,
            std::sync::atomic::Ordering::SeqCst,
        );
        let result = Qwen35PrefillState::advance_rectangular_prefill(
            states,
            plan,
            &mut qwen,
            &mut kv,
            &reservations,
            &supervisor,
        );
        QWEN35_STATE_FAILPOINT.store(0, std::sync::atomic::Ordering::SeqCst);
        let error = match result {
            Ok(_) => panic!("checkpoint failpoint must reject the prepared cohort"),
            Err(error) => error,
        };
        assert!(format!("{error:#}").contains("post-checkpoint failpoint"));
        for slot in slots {
            assert_eq!(kv.sequence_len_for_slot(slot).expect("cursor"), 0);
            assert!(kv
                .linear_attn
                .iter()
                .all(|linear| !linear.pp_flipped[slot.0 as usize]));
        }
    }
}
