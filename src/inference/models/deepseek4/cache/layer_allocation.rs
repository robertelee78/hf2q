//! Shared production layer allocation/state initialization, without cursor publication.
use super::super::cache_buffers::{
    allocate_buffer, allocate_optional, allocate_overwrite_buffer, fill_state, view_buffer,
};
use super::{CacheError, CacheKind, LayerCache, LayerCachePlan};
use mlx_native::MlxDevice;

pub(in crate::inference::models::deepseek4) fn allocate_layer_from_plan(
    device: &MlxDevice,
    layer: &LayerCachePlan,
    lazy_append_storage: bool,
) -> Result<LayerCache, CacheError> {
    let attention_kv = if lazy_append_storage {
        allocate_overwrite_buffer(
            &device,
            layer.layer_index,
            CacheKind::AttentionKv,
            &layer.attention_kv,
        )?
    } else {
        allocate_buffer(
            &device,
            layer.layer_index,
            CacheKind::AttentionKv,
            &layer.attention_kv,
        )?
    };
    let window_kv = view_buffer(
        &attention_kv,
        0,
        layer.layer_index,
        CacheKind::WindowKv,
        &layer.window_kv,
    )?;
    let compressed_kv = layer
        .compressed_kv
        .as_ref()
        .map(|buffer| {
            view_buffer(
                &attention_kv,
                layer.window_kv.bytes,
                layer.layer_index,
                CacheKind::CompressedKv,
                buffer,
            )
        })
        .transpose()?;
    let indexer_kv = layer
        .indexer_kv
        .as_ref()
        .map(|buffer| {
            if lazy_append_storage {
                allocate_overwrite_buffer(&device, layer.layer_index, CacheKind::IndexerKv, buffer)
            } else {
                allocate_buffer(&device, layer.layer_index, CacheKind::IndexerKv, buffer)
            }
        })
        .transpose()?;
    let main_kv_state = allocate_optional(
        &device,
        layer.layer_index,
        CacheKind::MainKvState,
        layer.main_kv_state.as_ref(),
    )?;
    let main_score_state = allocate_optional(
        &device,
        layer.layer_index,
        CacheKind::MainScoreState,
        layer.main_score_state.as_ref(),
    )?;
    let indexer_kv_state = allocate_optional(
        &device,
        layer.layer_index,
        CacheKind::IndexerKvState,
        layer.indexer_kv_state.as_ref(),
    )?;
    let indexer_score_state = allocate_optional(
        &device,
        layer.layer_index,
        CacheKind::IndexerScoreState,
        layer.indexer_score_state.as_ref(),
    )?;
    Ok(LayerCache {
        attention_kv,
        window_kv,
        compressed_kv,
        indexer_kv,
        main_kv_state,
        main_score_state,
        indexer_kv_state,
        indexer_score_state,
    })
}

pub(in crate::inference::models::deepseek4) fn initialize_layer_state(
    layer: &mut LayerCache,
    layer_index: usize,
) -> Result<(), CacheError> {
    fill_state(
        layer.main_kv_state.as_mut(),
        0.0,
        layer_index,
        CacheKind::MainKvState,
    )?;
    fill_state(
        layer.main_score_state.as_mut(),
        f32::NEG_INFINITY,
        layer_index,
        CacheKind::MainScoreState,
    )?;
    fill_state(
        layer.indexer_kv_state.as_mut(),
        0.0,
        layer_index,
        CacheKind::IndexerKvState,
    )?;
    fill_state(
        layer.indexer_score_state.as_mut(),
        f32::NEG_INFINITY,
        layer_index,
        CacheKind::IndexerScoreState,
    )?;
    Ok(())
}
