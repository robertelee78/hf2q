//! Complete active Qwen cache reads, no snapshots/new Metal roots. Reads are
//! allowed only after the caller's actual checked forward terminal. Native
//! packed codes and norms remain distinct tensors; residency aliases do not
//! add data. This first adapter refuses multi-slot layouts explicitly.
use super::{
    capture,
    receipt::{Coordinate, Stage},
    store::Store,
};
use crate::inference::models::qwen35::kv_cache::{FullAttnKvSlot, HybridKvCache, LayerSlot};
use crate::serve::multi_seq_kv::SlotId;
use anyhow::{ensure, Result};
use mlx_native::MlxBuffer;
pub fn after_checked_terminal(
    store: &mut Store,
    cache: &HybridKvCache,
    layer_count: u32,
    agent: &str,
    boundary: usize,
) -> Result<()> {
    ensure!(cache.n_seqs==1,"Q1 cache adapter currently requires independently owned serial caches; no cross-slot guess");
    ensure!(
        cache.sequence_len_for_slot(SlotId(0))? as usize == boundary,
        "actual Qwen visible cache boundary mismatch"
    );
    for layer in 0..layer_count {
        match cache
            .slot_index_for_layer(layer)
            .ok_or_else(|| anyhow::anyhow!("actual Qwen layer mapping missing"))?
        {
            LayerSlot::Full(index) => full(
                store,
                &cache.full_attn[index as usize],
                agent,
                boundary,
                layer,
            )?,
            LayerSlot::Linear(index) => {
                let state = &cache.linear_attn[index as usize];
                for (stage, buffer) in [
                    (Stage::CacheConv, state.conv_bufs_for_slot(SlotId(0)).0),
                    (
                        Stage::CacheRecurrent,
                        state.recurrent_bufs_for_slot(SlotId(0)).0,
                    ),
                ] {
                    capture::after_checked_drain(
                        store,
                        at(agent, boundary, layer, stage, 0),
                        buffer,
                        buffer.shape(),
                        None,
                        "Qwen actual terminal + visible cursor + per-slot current ping-pong getter",
                    )
                    .map_err(anyhow::Error::msg)?;
                }
            }
        }
    }
    if let Some(ref mtp) = cache.mtp_slot {
        // MTP has its own cursor. It is not assumed advanced by target-only
        // ordinary forwards, and an unexecuted MTP capacity tail is not read.
        let valid = mtp.current_len[0] as usize;
        if valid > 0 {
            full(store, mtp, agent, valid, layer_count)?;
        }
    }
    Ok(())
}
fn full(
    store: &mut Store,
    slot: &FullAttnKvSlot,
    agent: &str,
    valid: usize,
    layer: u32,
) -> Result<()> {
    ensure!(
        slot.current_len.as_slice() == [valid as u32],
        "Qwen cache role cursor mismatch"
    );
    for (stage, buffer) in [
        (Stage::CacheKey, slot.k.as_ref()),
        (Stage::CacheValue, slot.v.as_ref()),
    ] {
        if let Some(buffer) = buffer {
            heads(store, buffer, agent, valid, layer, stage)?;
        }
    }
    if let Some(ref tq) = slot.tq {
        for (stage, buffer) in [
            (Stage::CacheKeyPacked, &tq.k_packed),
            (Stage::CacheValuePacked, &tq.v_packed),
            (Stage::CacheKeyNorm, &tq.k_norms),
            (Stage::CacheValueNorm, &tq.v_norms),
        ] {
            heads(store, buffer, agent, valid, layer, stage)?;
        }
    }
    Ok(())
}
fn heads(
    store: &mut Store,
    buffer: &MlxBuffer,
    agent: &str,
    valid: usize,
    layer: u32,
    stage: Stage,
) -> Result<()> {
    let shape = buffer.shape();
    ensure!(
        shape.len() == 4 && shape[0] == 1 && valid <= shape[2],
        "Qwen actual cache geometry mismatch"
    );
    let per_head = shape[2]
        .checked_mul(shape[3])
        .ok_or_else(|| anyhow::anyhow!("Qwen head span overflow"))?;
    let elements = valid
        .checked_mul(shape[3])
        .ok_or_else(|| anyhow::anyhow!("Qwen active head span overflow"))?;
    for head in 0..shape[1] {
        let offset = head
            .checked_mul(per_head)
            .and_then(|n| n.checked_mul(buffer.dtype().size_of()))
            .ok_or_else(|| anyhow::anyhow!("Qwen head physical offset overflow"))?;
        let view = buffer
            .slice_view(offset as u64, elements)
            .with_shape(vec![valid, shape[3]])?;
        capture::rows_after_checked_drain(store,at(agent,valid,layer,stage,u32::try_from(head)?),&view,None,
            "Qwen actual checked terminal; complete visible prefix per head, excludes inactive capacity")
            .map_err(anyhow::Error::msg)?;
    }
    Ok(())
}
fn at(agent: &str, boundary: usize, layer: u32, stage: Stage, segment: u32) -> Coordinate {
    Coordinate {
        logical_step: boundary as u64,
        layer,
        stage,
        agent: agent.into(),
        logical_row_start: 0,
        segment,
    }
}
