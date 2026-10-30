//! Load-time warmup of the production SlotAware Gemma prefill route.
//!
//! `warmup_once` drives the legacy single-sequence one-token path. The first
//! real cold prefill therefore paid first use of the slot-mounted batched
//! route (multi-row MoE, sliding flash attention, global tensor-mm) and the
//! first touch of most expert pages: 264–392 ms for the first 458-token
//! prefill versus 188 ms for the next three in the 2026-10-30 four-agent
//! runs. This primes that route once in a clean idle slot and returns the
//! slot to the never-used state, so no request can observe it.
use super::*;

/// Above the 64-row global tensor-mm and 32-row batched-prefill thresholds,
/// and long enough to route through most experts.
const SLOT_WARMUP_TOKENS: u32 = 256;
/// Synthetic ids start above the special/control range.
const SLOT_WARMUP_FIRST_ID: u32 = 1_024;

/// A slot no request owns, retains or can restore: its rows are unreachable.
fn clean_idle_slot(
    slots: &[Option<Gemma4Slot>],
    retained_tokens: &[Vec<u32>],
    prompt_anchors: &[Option<Gemma4PromptAnchor>],
) -> Option<SlotId> {
    (0..slots.len())
        .find(|&index| {
            slots[index].is_none()
                && retained_tokens.get(index).is_some_and(Vec::is_empty)
                && prompt_anchors.get(index).is_some_and(Option::is_none)
        })
        .and_then(|index| u32::try_from(index).ok())
        .map(SlotId)
}

fn synthetic_prompt(vocab_size: usize) -> Option<Vec<u32>> {
    let vocab = u32::try_from(vocab_size).ok()?;
    let span = vocab
        .checked_sub(SLOT_WARMUP_FIRST_ID.checked_mul(2)?)?
        .max(1);
    Some(
        (0..SLOT_WARMUP_TOKENS)
            .map(|i| SLOT_WARMUP_FIRST_ID + (i.wrapping_mul(7_919) % span))
            .collect(),
    )
}

/// Legacy warmup, then one slot-aware cold prefill into a clean idle slot.
pub(super) fn warmup(
    guard: &mut Gemma4KvGuard<'_>,
    slots: &[Option<Gemma4Slot>],
    retained_tokens: &[Vec<u32>],
    prompt_anchors: &[Option<Gemma4PromptAnchor>],
    supervisor: &EngineSupervisor,
) -> Result<()> {
    warmup_once(guard.model, supervisor)?;
    // The kill-switch arm must equal stock serving, including load warmup.
    if crate::debug::investigation_env::gemma_full_input::mode()
        == crate::debug::investigation_env::gemma_full_input::Mode::Off
    {
        return Ok(());
    }
    let Some(slot) = clean_idle_slot(slots, retained_tokens, prompt_anchors) else {
        tracing::debug!("Gemma4 slot-route warmup skipped: no clean idle slot");
        return Ok(());
    };
    // A bound graft owns rows below every cold cursor; Hybrid is the only
    // production slot route. Neither case is primed here.
    if guard.hybrid.is_none() || guard.model.weights.kv_graft_len() > 0 {
        return Ok(());
    }
    let Some(prompt) = synthetic_prompt(guard.model.config.vocab_size) else {
        return Ok(());
    };
    let started = Instant::now();
    install_gemma4_slot_cursor(guard, slot, 0)?;
    clear_gemma4_self_mounts(guard.model);
    let result = supervised_gemma4_gpu_call(supervisor, "gemma4_slot_warmup_prefill", || {
        guard
            .model
            .weights
            .forward_prefill_with_soft_tokens_slot_aware(
                &prompt,
                &[],
                1,
                &mut guard.model.ctx,
                slot,
                &mut guard.kv,
                guard.hybrid.as_mut(),
                guard.dense.as_mut(),
                guard.mlx.as_mut(),
            )
    });
    clear_gemma4_self_mounts(guard.model);
    // Never-used state on every outcome: cursor zero, no ledger, no anchor.
    install_gemma4_slot_cursor(guard, slot, 0)?;
    if let Err(error) = result {
        // Only a worker-fatal GPU fault stops the load; anything else leaves
        // the first request to pay first use, exactly as before this warmup.
        if is_worker_fatal(supervisor, &error) {
            return Err(error);
        }
        tracing::warn!(slot = slot.0, error = %error, "Gemma4 slot-route warmup failed; continuing");
        return Ok(());
    }
    tracing::info!(
        slot = slot.0,
        tokens = prompt.len(),
        elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
        "Gemma4 SlotAware prefill route warmed"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_slot_warmup_only_uses_a_slot_no_request_can_observe() {
        let none: Vec<Option<Gemma4Slot>> = (0..3).map(|_| None).collect();
        let anchors: Vec<Option<Gemma4PromptAnchor>> = (0..3).map(|_| None).collect();
        let retained = vec![vec![1, 2], Vec::new(), Vec::new()];
        // Slot 0 retains a live prefix that a continuation may reuse.
        assert_eq!(clean_idle_slot(&none, &retained, &anchors), Some(SlotId(1)));
        let all_retained = vec![vec![1], vec![2], vec![3]];
        assert_eq!(clean_idle_slot(&none, &all_retained, &anchors), None);
        assert_eq!(clean_idle_slot(&[], &[], &[]), None);
    }

    #[test]
    fn gemma_slot_warmup_prompt_is_deterministic_and_in_vocabulary() {
        let prompt = synthetic_prompt(262_144).unwrap();
        assert_eq!(prompt.len(), SLOT_WARMUP_TOKENS as usize);
        assert_eq!(prompt, synthetic_prompt(262_144).unwrap());
        assert!(prompt
            .iter()
            .all(|&id| (SLOT_WARMUP_FIRST_ID..262_144).contains(&id)));
        assert!(synthetic_prompt(1_000).is_none());
    }
}
