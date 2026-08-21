//! Fail-closed cache/config/call state admission for source execution.

use anyhow::{bail, ensure, Context, Result};

use crate::inference::models::qwen35::kv_cache::{HybridKvCache, LayerSlot};
use crate::inference::models::qwen35::Qwen35LayerKind;

use super::super::{PreparedQwen35SourceAttentionV1, PreparedQwen35SourceTeacherV1};

pub(super) struct CallPreflight {
    pub(super) end_position: u32,
    pub(super) linear_parity: bool,
}

pub(super) fn preflight_source_teacher_state(
    teacher: &PreparedQwen35SourceTeacherV1,
    cache: &HybridKvCache,
    expected_position: u32,
) -> Result<bool> {
    let config = &teacher.config;
    ensure!(
        teacher.layers.len() == config.layer_types.len()
            && cache
                .slot_index_for_layer(u32::try_from(config.layer_types.len())?)
                .is_none()
            && cache.n_seqs == 1
            && cache.mtp_slot.is_none()
            && !cache.tq_kv_active
            && teacher.device.registry_id() == cache_device_registry_id(cache)?,
        "source teacher graph/cache profile differs from prepared config"
    );
    let mut full_rank = 0usize;
    let mut linear_rank = 0usize;
    let mut linear_parity = None;
    for (layer_index, kind) in config.layer_types.iter().copied().enumerate() {
        match (kind, &teacher.layers[layer_index].attention) {
            (Qwen35LayerKind::FullAttention, PreparedQwen35SourceAttentionV1::Full(_)) => {
                ensure!(
                    cache.slot_index_for_layer(u32::try_from(layer_index)?)
                        == Some(LayerSlot::Full(full_rank as u32)),
                    "source teacher full-attention cache rank differs at layer {layer_index}"
                );
                let slot = &cache.full_attn[full_rank];
                ensure!(
                    slot.tq.is_none()
                        && slot.k.is_some()
                        && slot.v.is_some()
                        && slot.current_len.as_slice() == [expected_position],
                    "source teacher full-attention cursor/profile differs at layer {layer_index}"
                );
                full_rank += 1;
            }
            (Qwen35LayerKind::LinearAttention, PreparedQwen35SourceAttentionV1::Linear(_)) => {
                ensure!(
                    cache.slot_index_for_layer(u32::try_from(layer_index)?)
                        == Some(LayerSlot::Linear(linear_rank as u32)),
                    "source teacher Delta cache rank differs at layer {layer_index}"
                );
                let slot = &cache.linear_attn[linear_rank];
                ensure!(
                    slot.capture_states.is_none()
                        && slot.conv_capture_states.is_none()
                        && slot.pp_flipped.len() == 1,
                    "source teacher Delta cache profile differs at layer {layer_index}"
                );
                let parity = slot.pp_flipped[0];
                ensure!(
                    linear_parity.is_none_or(|expected| expected == parity),
                    "source teacher Delta cache parity is desynchronized"
                );
                linear_parity = Some(parity);
                linear_rank += 1;
            }
            _ => bail!("source teacher layer schedule differs at layer {layer_index}"),
        }
    }
    ensure!(
        full_rank == cache.full_attn.len() && linear_rank == cache.linear_attn.len(),
        "source teacher cache cardinality differs from layer schedule"
    );
    linear_parity.context("source teacher cache lacks Delta state")
}

pub(super) fn preflight_source_teacher_call(
    teacher: &PreparedQwen35SourceTeacherV1,
    cache: &HybridKvCache,
    token_ids: &[u32],
    first_position: u32,
) -> Result<CallPreflight> {
    let linear_parity = preflight_source_teacher_state(teacher, cache, first_position)?;
    let sequence_length = u32::try_from(token_ids.len())?;
    ensure!(
        if first_position == 0 {
            sequence_length >= 16
        } else {
            sequence_length == 1
        },
        "source teacher v1 admits one >=16-token fresh prefill followed by one-token continuations"
    );
    ensure!(
        teacher.config.vocab_size > 0
            && teacher.config.vocab_size % 2 == 0
            && token_ids
                .iter()
                .all(|token_id| *token_id < teacher.config.vocab_size),
        "source teacher tokens/output vocabulary differ from the v1 output route"
    );
    let end_position = first_position
        .checked_add(sequence_length)
        .context("source teacher call position overflow")?;
    ensure!(
        end_position <= cache.max_seq_len,
        "source teacher call exceeds cache capacity"
    );
    Ok(CallPreflight {
        end_position,
        linear_parity,
    })
}

pub(super) fn postflight_source_teacher_call(
    cache: &HybridKvCache,
    preflight: &CallPreflight,
) -> Result<()> {
    ensure!(
        cache
            .full_attn
            .iter()
            .all(|slot| slot.current_len.as_slice() == [preflight.end_position])
            && cache.linear_attn.iter().all(|slot| {
                slot.pp_flipped.len() == 1 && slot.pp_flipped[0] != preflight.linear_parity
            }),
        "source teacher cache did not advance exactly once"
    );
    Ok(())
}

pub(super) fn cache_device_registry_id(cache: &HybridKvCache) -> Result<u64> {
    if let Some(slot) = cache.full_attn.first() {
        return Ok(slot
            .k
            .as_ref()
            .context("source teacher full cache K is absent")?
            .metal_buffer()
            .device()
            .registry_id());
    }
    let slot = cache
        .linear_attn
        .first()
        .context("source teacher cache has no layer state")?;
    Ok(slot.conv_state.metal_buffer().device().registry_id())
}
