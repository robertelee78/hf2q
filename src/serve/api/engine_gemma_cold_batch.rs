//! Cold-batch admission policy, before any slot reset or GPU submission.
use super::*;

#[derive(Debug, Eq, PartialEq)]
pub(super) struct Plan {
    pub(super) has_tiny_prefill: bool,
    transaction_too_large: bool,
    has_reusable_prefix: bool,
}
impl Plan {
    pub(super) fn requires_slot_admission(&self) -> bool {
        self.has_tiny_prefill || self.transaction_too_large || self.has_reusable_prefix
    }
}

/// The caller supplies the actual request-aware retained-slot preference.
/// No generation cue (S == P) is not proof that a request has no reusable KV.
/// Every lane falls back in its original order if any lane needs slot reuse.
pub(super) fn plan(candidates: impl IntoIterator<Item = (usize, usize)>) -> Plan {
    let mut total_rows = Some(0usize);
    let mut result = Plan {
        has_tiny_prefill: false,
        transaction_too_large: false,
        has_reusable_prefix: false,
    };
    for (rows, cached) in candidates {
        total_rows = total_rows.and_then(|total| total.checked_add(rows));
        result.has_tiny_prefill |=
            !crate::serve::forward_prefill::gemma_slot_prefill_batched_work_eligible(rows, false);
        result.has_reusable_prefix |= cached > 0;
    }
    result.transaction_too_large =
        total_rows.is_none_or(|rows| rows > GEMMA4_SLOT_PREFILL_CHUNK_TOKENS as usize);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preference(
        retained: &[Vec<u32>],
        anchors: &[Vec<u32>],
        prompt: &[u32],
    ) -> Option<Gemma4SlotPreference> {
        // Actual production affinity primitive, with CPU-only token owners;
        // no fabricated GPU cache or snapshot is required to test selection.
        gemma4_slot_affinity(
            retained.iter().zip(anchors).enumerate().map(|(i, (live, anchor))| {
                (i, live.as_slice(), Some(anchor.as_slice()), None, false, false)
            }),
            prompt,
        )
        .map(Gemma4SlotAffinity::preference)
    }

    fn histories() -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let stable: Vec<Vec<u32>> = (0..4)
            .map(|agent| (0..458).map(|token| agent * 1000 + token).collect())
            .collect();
        let live = stable.iter().map(|s| {
            let mut p = s.clone();
            p.extend([8000; 25]);
            p
        }).collect();
        let prompts = stable.iter().map(|s| {
            let mut p = s.clone();
            p.extend([9000; 56]);
            p
        }).collect();
        (live, stable, prompts)
    }

    #[test]
    fn gemma_cold_batch_preserves_four_reusable_458_token_anchors_without_a_cue() {
        let (live, stable, prompts) = histories();
        let candidates = prompts.iter().enumerate().map(|(i, prompt)| {
            assert_eq!(prompt.len(), 514);
            // Deciding failed-run shape: current S == P, thus no strict cue
            // boundary and all four requests classify as the cold cohort.
            assert!(gemma4_cross_batch_eligible(514, 8192, false, false, false));
            let selected = preference(&live, &stable, prompt).unwrap();
            assert_eq!(selected.slot, SlotId(i as u32));
            assert_eq!(selected.cached_tokens, 458);
            assert!(selected.restore_prompt_anchor);
            (prompt.len(), selected.cached_tokens)
        });
        let decision = plan(candidates);
        assert!(decision.requires_slot_admission());
        assert!(!decision.transaction_too_large);
        assert!(!decision.has_tiny_prefill);
    }

    #[test]
    fn gemma_cold_batch_still_batches_true_cold_and_divergent_histories() {
        let (live, stable, _) = histories();
        let prompts = vec![vec![42_000; 514]; 4];
        let decision = plan(prompts.iter().map(|prompt| {
            assert!(preference(&live, &stable, prompt).is_none());
            (prompt.len(), 0)
        }));
        assert!(!decision.requires_slot_admission());
    }

    #[test]
    fn gemma_cold_batch_mixed_warm_cold_falls_back_without_reordering() {
        let (live, stable, prompts) = histories();
        let cold = vec![42_000; 514];
        let requests = [cold.as_slice(), prompts[2].as_slice(), prompts[0].as_slice()];
        let mut visited = Vec::new();
        let decision = plan(requests.iter().enumerate().map(|(i, prompt)| {
            visited.push(i);
            (prompt.len(), preference(&live, &stable, prompt).map_or(0, |p| p.cached_tokens))
        }));
        assert!(decision.requires_slot_admission());
        assert_eq!(visited, [0, 1, 2]);
    }

    #[test]
    fn gemma_cold_batch_preserves_existing_tiny_aggregate_and_overflow_refusals() {
        assert!(plan([(1, 0), (514, 0)]).has_tiny_prefill);
        assert!(plan([(2048, 0), (2049, 0)]).requires_slot_admission());
        assert!(plan([(usize::MAX, 0), (1, 0)]).requires_slot_admission());
        assert!(!plan([(2048, 0), (2048, 0)]).requires_slot_admission());
    }
}
