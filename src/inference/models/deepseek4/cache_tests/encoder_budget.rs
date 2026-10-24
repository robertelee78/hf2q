//! Scoped test-owned host containers; not opaque native/driver metadata.
use super::{model, packet::Packet};
use anyhow::{ensure, Result};
use model::cache::{
    CacheBufferPlan, CacheSpan, Deepseek4CachePlan, LayerCachePlan, LayerCacheSpan,
};
use std::mem::size_of;
fn shape(plan: &CacheBufferPlan) -> usize {
    plan.shape.capacity() * size_of::<usize>()
}
pub(super) fn check(
    cfg: &model::Deepseek4Config,
    plans: &[Deepseek4CachePlan; 2],
    spans: &[CacheSpan; 2],
    packet: &Packet,
) -> Result<usize> {
    let mut bytes = cfg.compress_ratios.capacity() * size_of::<u32>()
        + (cfg.swiglu_clamp_experts.capacity() + cfg.swiglu_clamp_shared.capacity())
            * size_of::<f32>()
        + packet.roots.capacity() * size_of::<mlx_native::MlxBuffer>();
    for plan in plans {
        bytes += plan.layers.capacity() * size_of::<LayerCachePlan>();
        for layer in &plan.layers {
            bytes += shape(&layer.attention_kv) + shape(&layer.window_kv);
            for value in [
                &layer.compressed_kv,
                &layer.indexer_kv,
                &layer.main_kv_state,
                &layer.main_score_state,
                &layer.indexer_kv_state,
                &layer.indexer_score_state,
            ]
            .into_iter()
            .flatten()
            {
                bytes += shape(value);
            }
        }
    }
    for span in spans {
        bytes += span.layers.capacity() * size_of::<LayerCacheSpan>();
    }
    bytes += super::packet::shape_argument_bytes();
    // Conservative finite allowance for <=512 local root/view handles, their
    // rank<=4 shape arguments, semantic slice-reference containers and stack
    // descriptors. Arc internals/allocator/native tracking remain Unknown.
    bytes += 512 * (size_of::<mlx_native::MlxBuffer>() + 4 * size_of::<usize>())
        + 7 * size_of::<&[u8]>();
    ensure!(bytes <= 1_048_576, "test-owned bookkeeping exceeded1MiB");
    Ok(bytes)
}
