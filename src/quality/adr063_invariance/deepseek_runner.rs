//! Actual DeepSeek verifier transactions. This is deliberately separate from
//! the stage localizer: grouped/cohort execution cannot be relabelled as the
//! per-layer diagnostic route, and a cancelled transaction is not a token.
use super::protocol::{ExecutionEvidence, Trajectory, Workload};
use crate::inference::models::deepseek4::{cache::Deepseek4Cache, Deepseek4Model};
use anyhow::{bail, ensure, Result};
use mlx_native::MlxBuffer;

/// Existing actual APIs only. Callback runs after their checked GPU drain and
/// successful cursor publication, before any subsequent transaction can write.
/// A caller must not run the per-layer localizer concurrently with this route.
pub fn run(
    model: &mut Deepseek4Model,
    work: &Workload,
    trajectory: Trajectory,
    capacity: usize,
    mut observe: impl FnMut(usize, usize, &MlxBuffer, &MlxBuffer, &Deepseek4Cache) -> Result<()>,
) -> Result<ExecutionEvidence> {
    work.validate().map_err(anyhow::Error::msg)?;
    ensure!(
        work.family == super::receipt::Family::Deepseek4,
        "DS adapter family mismatch"
    );
    ensure!(
        work.vocab_size == model.cfg.vocab_size,
        "DS vocabulary mismatch"
    );
    ensure!(
        capacity >= 144,
        "Q1 configured cache capacity cannot shrink below workload"
    );
    ensure!(
        !super::family::enabled(super::receipt::Family::Deepseek4),
        "grouped endpoint runner must not silently enable the single-layer localizer"
    );
    if trajectory == Trajectory::T5GenuineSpeculation {
        bail!("LocalizationIncomplete: DS genuine proposal/verifier adapter not yet connected; no fake acceptance");
    }
    let mut evidence = ExecutionEvidence {
        trajectory,
        actual_route: "DeepSeek actual verifier transactions".into(),
        positions_per_lane: vec![0; 4],
        checked_terminal_calls: 0,
        restored_at: None,
        cancelled_at: None,
        accepted_draft_rows: None,
        rejected_draft_rows: None,
        physical_cohort_rows: None,
        missing_adapter_capabilities: vec![],
    };
    let mut caches = Vec::new();
    for _ in 0..4 {
        caches.push(model.allocate_cache(capacity)?);
    }
    if trajectory == Trajectory::T2ActualCohort {
        let batches: Vec<&[u32]> = work.lanes.iter().map(|t| &t[..128]).collect();
        let mut refs: Vec<_> = caches.iter_mut().collect();
        let states =
            model
                .forward_verifier_prefill_cohort_with_commit_gate(&batches, &mut refs, || Ok(()))?;
        ensure!(states.len() == 4, "actual cohort state cardinality");
        evidence.checked_terminal_calls += 1;
        evidence.physical_cohort_rows = Some(vec![4]);
        evidence.actual_route="DS cooperative prefill row-packed FFN; native four-lane decode chain (not one shared target graph)".into();
        for lane in 0..4 {
            ensure!(caches[lane].position() == 128, "cohort cursor publication");
            observe_state(model, lane, 128, &states[lane], &caches[lane], &mut observe)?;
        }
        for position in 128..144 {
            let ids = std::array::from_fn(|lane| work.lanes[lane][position]);
            let [a, b, c, d] = caches.as_mut_slice() else {
                unreachable!()
            };
            let mut refs = [a, b, c, d];
            let state =
                model.forward_verifier_decode_cohort_with_commit_gate(ids, &mut refs, || Ok(()))?;
            ensure!(
                state.shape().first() == Some(&4),
                "native packed cohort extent"
            );
            let per_lane = state.element_count() / 4;
            for lane in 0..4 {
                let offset = lane
                    .checked_mul(per_lane)
                    .and_then(|n| n.checked_mul(state.dtype().size_of()))
                    .ok_or_else(|| anyhow::anyhow!("cohort offset overflow"))?;
                let mut shape = state.shape().to_vec();
                shape[0] = 1;
                let row = state
                    .slice_view(offset as u64, per_lane)
                    .with_shape(shape)?;
                observe_state(model, lane, position + 1, &row, &caches[lane], &mut observe)?;
            }
            evidence.checked_terminal_calls += 1;
        }
    } else {
        let order = if trajectory == Trajectory::T4ReverseOrder {
            [3, 2, 1, 0]
        } else {
            [0, 1, 2, 3]
        };
        for lane in order {
            let tokens = &work.lanes[lane];
            let cache = &mut caches[lane];
            if matches!(
                trajectory,
                Trajectory::T1Resume64 | Trajectory::T3Chunk64 | Trajectory::T6CancelRestore
            ) {
                let first = model.forward_verifier_prefill(&tokens[..64], cache)?;
                evidence.checked_terminal_calls += 1;
                observe_state(model, lane, 64, &first, cache, &mut observe)?;
                if matches!(
                    trajectory,
                    Trajectory::T1Resume64 | Trajectory::T6CancelRestore
                ) {
                    let checkpoint = cache.snapshot()?;
                    if trajectory == Trajectory::T6CancelRestore {
                        let rejected = model.forward_verifier_prefill_with_commit_gate(
                            &tokens[64..128],
                            cache,
                            || {
                                Err(anyhow::anyhow!(
                                    "Q1 deliberate rejection after GPU drain before publication"
                                ))
                            },
                        );
                        ensure!(rejected.is_err(), "cancel negative unexpectedly published");
                        ensure!(cache.position() == 64, "cancel changed logical cursor");
                        evidence.checked_terminal_calls += 1;
                        evidence.cancelled_at = Some(64);
                    } else {
                        *cache = model.allocate_cache(capacity)?;
                    }
                    cache.restore(&checkpoint)?;
                    ensure!(cache.position() == 64, "actual restore cursor");
                    evidence.restored_at = Some(64);
                }
                let state = model.forward_verifier_prefill(&tokens[64..128], cache)?;
                evidence.checked_terminal_calls += 1;
                observe_state(model, lane, 128, &state, cache, &mut observe)?;
            } else {
                let state = model.forward_verifier_prefill(&tokens[..128], cache)?;
                evidence.checked_terminal_calls += 1;
                observe_state(model, lane, 128, &state, cache, &mut observe)?;
            }
            for position in 128..144 {
                let state = model.forward_verifier_one(tokens[position], cache)?;
                evidence.checked_terminal_calls += 1;
                observe_state(model, lane, position + 1, &state, cache, &mut observe)?;
            }
        }
    }
    evidence.positions_per_lane = caches.iter().map(|c| c.position()).collect();
    evidence.validate().map_err(anyhow::Error::msg)?;
    Ok(evidence)
}

fn observe_state(
    model: &mut Deepseek4Model,
    lane: usize,
    position: usize,
    state: &MlxBuffer,
    cache: &Deepseek4Cache,
    observe: &mut impl FnMut(usize, usize, &MlxBuffer, &MlxBuffer, &Deepseek4Cache) -> Result<()>,
) -> Result<()> {
    // Existing full-row native head. No argmax/sampling changes or last-row
    // shortcut: every state row returned by this actual transaction is scored.
    let logits = model.forward_logits(state)?;
    observe(lane, position, state, &logits, cache)
}
