//! Exact active DS cache extents at a SOURCE-OWNED checked per-layer drain.
//! No capacity tail is read. Ring rows are compared in physical ring order;
//! logical token position remains in the coordinate/configuration evidence.
use super::{
    family,
    receipt::{Family, Stage},
};
use crate::inference::models::deepseek4::cache::{Deepseek4Cache, LayerCacheSpan};
use mlx_native::MlxBuffer;
fn rows(buffer: &MlxBuffer, valid: usize) -> Result<MlxBuffer, String> {
    let shape = buffer.shape();
    if shape.is_empty() || valid > shape[0] {
        return Err("DS active rows exceed source buffer".into());
    }
    let per_row = shape[1..]
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or("DS cache row overflow")?;
    let count = per_row
        .checked_mul(valid)
        .ok_or("DS active cache extent overflow")?;
    let mut view_shape = shape.to_vec();
    view_shape[0] = valid;
    buffer
        .slice_view(0, count)
        .with_shape(view_shape)
        .map_err(|e| e.to_string())
}
pub fn layer_after_drain(
    cache: &Deepseek4Cache,
    span: &LayerCacheSpan,
    start: usize,
    count: usize,
) -> Result<(), String> {
    if !family::enabled(Family::Deepseek4) {
        return Ok(());
    }
    let layer = span.layer_index;
    let end = start
        .checked_add(count)
        .ok_or("DS cache position overflow")?;
    let cache_layer = cache
        .layers()
        .get(layer)
        .ok_or("DS actual cache layer missing")?;
    let read = |stage, buffer: &MlxBuffer| {
        family::drained(
            Family::Deepseek4,
            start as u64,
            layer as u32,
            stage,
            buffer,
            "DS layer checked session finish + MOE status, before cursor publication",
        )
    };
    let window = rows(
        &cache_layer.window_kv,
        end.min(cache_layer.window_kv.shape()[0]),
    )?;
    read(Stage::CacheWindow, &window)?;
    if let Some(ref buffer) = cache_layer.compressed_kv {
        let valid = cache
            .graft_slots()
            .checked_add(span.compressed_valid_after)
            .ok_or("graft extent overflow")?;
        if valid > 0 {
            read(Stage::CacheCompressed, &rows(buffer, valid)?)?;
        }
    }
    if let Some(ref buffer) = cache_layer.indexer_kv {
        if span.indexer_valid_after > 0 {
            read(
                Stage::CacheIndexer,
                &rows(buffer, span.indexer_valid_after)?,
            )?;
        }
    }
    for (stage, buffer) in [
        (Stage::CacheMainKv, &cache_layer.main_kv_state),
        (Stage::CacheMainScore, &cache_layer.main_score_state),
        (Stage::CacheIndexerState, &cache_layer.indexer_kv_state),
        (Stage::CacheIndexerScore, &cache_layer.indexer_score_state),
    ] {
        if let Some(buffer) = buffer {
            read(stage, buffer)?;
        }
    }
    Ok(())
}
