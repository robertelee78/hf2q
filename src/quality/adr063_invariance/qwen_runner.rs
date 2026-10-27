//! Actual Qwen native forward/cache adapter; no peer, alternate loader, or fake
//! batching. Q38 dense and Q36 MoE share the production model implementation,
//! but the configured variant is verified before any trajectory executes.
use super::{
    protocol::{ExecutionEvidence, Trajectory, Workload},
    receipt::Family,
};
use crate::inference::models::qwen35::{
    kv_cache::HybridKvCache, model::Qwen35Model, spec_decode::positions_for_range, Qwen35Variant,
};
use crate::serve::multi_seq_kv::SlotId;
use anyhow::{bail, ensure, Result};

pub fn run(
    model: &Qwen35Model,
    work: &Workload,
    trajectory: Trajectory,
    capacity: u32,
    mut observe: impl FnMut(usize, usize, &[f32], &HybridKvCache) -> Result<()>,
) -> Result<ExecutionEvidence> {
    work.validate().map_err(anyhow::Error::msg)?;
    ensure!(
        matches!(
            (work.family, model.cfg.variant),
            (Family::Qwen38, Qwen35Variant::Dense) | (Family::Qwen36, Qwen35Variant::Moe)
        ),
        "exact Qwen family/variant mismatch"
    );
    ensure!(
        work.vocab_size == model.cfg.vocab_size && capacity >= 144,
        "Qwen diagnostic shape/config mismatch"
    );
    match trajectory {
        Trajectory::T2ActualCohort=>bail!("SourceCapabilityRefusal: ordinary Qwen decode_batch loops per handle; no shared four-lane target graph. Do not replace by concatenated independent sequences"),
        Trajectory::T5GenuineSpeculation=>bail!("LocalizationIncomplete: genuine Qwen speculative accepted/rejected transaction adapter pending; no fabricated proposals"),
        Trajectory::T6CancelRestore=>bail!("LocalizationIncomplete: owning-worker cancellation gate required; forward_gpu parity mutation is not a publication transaction"),
        _=>{}
    }
    model.ensure_gpu_cache_primed()?;
    let mut evidence = ExecutionEvidence {
        trajectory,
        actual_route: "Qwen actual forward_gpu All + TQ8 independent serial slot0 caches".into(),
        positions_per_lane: vec![0; 4],
        checked_terminal_calls: 0,
        restored_at: None,
        cancelled_at: None,
        accepted_draft_rows: None,
        rejected_draft_rows: None,
        physical_cohort_rows: None,
        missing_adapter_capabilities: vec![],
    };
    let order = if trajectory == Trajectory::T4ReverseOrder {
        [3, 2, 1, 0]
    } else {
        [0, 1, 2, 3]
    };
    for lane in order {
        let mut cache = model.with_gpu_cache_mut(|device, _| {
            HybridKvCache::new_with_options(&model.cfg, device, capacity, 1, true)
        })?;
        let tokens = &work.lanes[lane];
        if matches!(trajectory, Trajectory::T1Resume64 | Trajectory::T3Chunk64) {
            forward(model, lane, 0, &tokens[..64], &mut cache, &mut observe)?;
            evidence.checked_terminal_calls += 1;
            if trajectory == Trajectory::T1Resume64 {
                let checkpoint =
                    model.with_gpu_cache_mut(|device, _| cache.snapshot_prefix(device, 64))?;
                cache = model.with_gpu_cache_mut(|device, _| {
                    HybridKvCache::new_with_options(&model.cfg, device, capacity, 1, true)
                })?;
                cache.restore_partial(&checkpoint, 64)?;
                ensure!(
                    cache.sequence_len_for_slot(SlotId(0))? == 64,
                    "Qwen actual restore cursor"
                );
                evidence.restored_at = Some(64);
            }
            forward(model, lane, 64, &tokens[64..128], &mut cache, &mut observe)?;
            evidence.checked_terminal_calls += 1;
        } else {
            forward(model, lane, 0, &tokens[..128], &mut cache, &mut observe)?;
            evidence.checked_terminal_calls += 1;
        }
        for position in 128..144 {
            forward(
                model,
                lane,
                position,
                &tokens[position..position + 1],
                &mut cache,
                &mut observe,
            )?;
            evidence.checked_terminal_calls += 1;
        }
        evidence.positions_per_lane[lane] = cache.sequence_len_for_slot(SlotId(0))? as usize;
    }
    evidence.validate().map_err(anyhow::Error::msg)?;
    Ok(evidence)
}
fn forward(
    model: &Qwen35Model,
    lane: usize,
    start: usize,
    tokens: &[u32],
    cache: &mut HybridKvCache,
    observe: &mut impl FnMut(usize, usize, &[f32], &HybridKvCache) -> Result<()>,
) -> Result<()> {
    ensure!(
        cache.sequence_len_for_slot(SlotId(0))? as usize == start,
        "Qwen forward starts at wrong logical position"
    );
    let positions = positions_for_range(i32::try_from(start)?, tokens.len());
    let logits = model.forward_gpu(tokens, &positions, cache, SlotId(0))?;
    let end = start
        .checked_add(tokens.len())
        .ok_or_else(|| anyhow::anyhow!("Qwen position overflow"))?;
    ensure!(
        logits.len()
            == tokens
                .len()
                .checked_mul(model.cfg.vocab_size as usize)
                .ok_or_else(|| anyhow::anyhow!("Qwen logits overflow"))?,
        "actual Qwen full-row logits omitted rows"
    );
    ensure!(
        cache.sequence_len_for_slot(SlotId(0))? as usize == end,
        "Qwen terminal cursor mismatch"
    );
    observe(lane, end, &logits, cache)
}
