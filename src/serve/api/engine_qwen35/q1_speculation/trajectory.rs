//! Real owning-state calls; explicit MTP selector, no fabricated cost/acceptance.
use super::*;
struct CacheLease<'a> {
    model: &'a mut Qwen35LoadedModel,
    cache: Option<HybridKvCache>,
}
impl Drop for CacheLease<'_> {
    fn drop(&mut self) {
        self.model.persistent_kv_cache = self.cache.take();
    }
}
fn check_fresh(cache: &mut HybridKvCache) -> Result<usize> {
    cache.reset_for_slot(SlotId(0))?;
    anyhow::ensure!(
        cache.n_seqs == 1
            && cache.sequence_len_for_slot(SlotId(0))? == 0
            && !cache.la_capture_active(),
        "fresh independent physical slot0 required"
    );
    let mut elements = 0usize;
    for state in &cache.linear_attn {
        for buffer in [
            &state.conv_state,
            &state.conv_state_scratch,
            &state.recurrent,
            &state.recurrent_scratch,
        ] {
            anyhow::ensure!(buffer.dtype() == DType::F32, "state dtype");
            let values = buffer.as_logical_slice::<f32>()?;
            anyhow::ensure!(
                values.iter().all(|&v| v == 0.0),
                "reset left stale current/scratch state"
            );
            elements = elements
                .checked_add(values.len())
                .context("zero state count overflow")?;
        }
    }
    Ok(elements)
}
fn exact_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        max_tokens: 8192,
        ..SamplingParams::default()
    }
}
pub(super) fn run_lane(
    model: &mut Qwen35LoadedModel,
    mode: Mode,
    prefix: &[u32],
    lane: usize,
) -> Result<usize> {
    run_lane_until(model, mode, prefix, lane, 144)
}
pub(super) fn run_lane_until(
    model: &mut Qwen35LoadedModel, mode: Mode, prefix: &[u32], lane: usize, end: usize,
) -> Result<usize> {
    anyhow::ensure!(end == 131 || end == 144, "explicit diagnostic end");
    anyhow::ensure!(
        prefix.len() == 128
            && model.speculation.policy()
                == super::super::super::qwen35_speculation::QwenSpeculationPolicy::Auto
            && model.model.mtp.is_some()
            && model.model.kv_graft.is_none()
            && model.tq_kv_active,
        "selected QwenAuto/MTP/TQ8/ungrafted path required"
    );
    let cache = model
        .persistent_kv_cache
        .take()
        .context("persistent cache absent")?;
    let mut lease = CacheLease {
        model,
        cache: Some(cache),
    };
    let zero_elements = check_fresh(lease.cache.as_mut().unwrap())?;
    enter_lane(lane)?;
    let params = exact_params();
    anyhow::ensure!(
        is_qwen_server_speculation_exact_eligible(&params),
        "canonical greedy semantics unsupported"
    );
    let prefill = Qwen35PrefillState::begin(
        prefix.to_vec(),
        params,
        None,
        lease.cache.as_mut().unwrap(),
        SlotId(0),
        0,
        None,
        None,
        None,
        lease.model.hidden_size,
        None,
    )?;
    let supervisor = EngineSupervisor::new();
    let advance = prefill.advance(
        lease.model,
        None,
        lease.cache.as_mut().unwrap(),
        128,
        &supervisor,
    )?;
    let Qwen35PrefillAdvance::Ready { mut state, .. } = advance else {
        anyhow::bail!("one128-row prefill didn't reach actual Ready");
    };
    anyhow::ensure!(
        !state.finished_at_seed() && state.mtp.is_some(),
        "preparation seed ended or real MTP initialization unavailable"
    );
    lease
        .cache
        .as_ref()
        .unwrap()
        .validate_speculative_cursors_for_slot(SlotId(0), 128)?;
    seed(
        *state
            .generated_tokens
            .first()
            .context("actual canonical seed absent")?,
        false,
    )?;
    for input_position in 128..end {
        if mode == Mode::ForcedProposal && input_position == 129 {
            anyhow::ensure!(
                state.pending_speculation_output.is_empty() && state.mtp_cost.may_speculate(),
                "real ordinary baseline not measured; no cost override"
            );
            let before = state.generated_tokens.len();
            let outcome = state.mtp_k3_round(
                lease.model,
                lease.cache.as_mut().unwrap(),
                &supervisor,
            )?;
            anyhow::ensure!(
                !outcome.finished && state.generated_tokens.len() == before + 1,
                "actual K3 first output missing/terminal"
            );
            observe_k3_return(*state.generated_tokens.last().unwrap())?;
            let pending = take_pending_speculation_output(&mut state.pending_speculation_output)
                .context("real corrected pending output absent")?;
            observe_pending(pending)?;
            let outcome = state.commit_speculation_output(lease.model, pending)?;
            anyhow::ensure!(
                !outcome.finished && state.pending_speculation_output.is_empty(),
                "exactly two real K3 canonical outputs required"
            );
            lease
                .cache
                .as_ref()
                .unwrap()
                .validate_speculative_cursors_for_slot(SlotId(0), 131)?;
            continue;
        }
        if mode == Mode::ForcedProposal && input_position == 130 {
            continue;
        }
        anyhow::ensure!(
            lease
                .cache
                .as_ref()
                .unwrap()
                .sequence_len_for_slot(SlotId(0))? as usize
                == input_position,
            "actual ordinary cursor/order drift"
        );
        let outcome = state.decode_tick_mtp_warmup(
            lease.model,
            lease.cache.as_mut().unwrap(),
            &supervisor,
        )?;
        anyhow::ensure!(
            !outcome.finished || input_position == 143,
            "canonical trajectory ended before requiredg15 input"
        );
    }
    anyhow::ensure!(supervisor.is_healthy(), "GPU supervision failed");
    lease
        .cache
        .as_ref()
        .unwrap()
        .validate_speculative_cursors_for_slot(SlotId(0), end)?;
    anyhow::ensure!(
        !lease.cache.as_ref().unwrap().la_capture_active(),
        "capture should be inactive; storage retained/charged"
    );
    Ok(zero_elements)
}
