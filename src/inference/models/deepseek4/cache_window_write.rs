//! Compressed production window-write segment, including original range checks.
use super::cache::{LayerCache, LayerCacheSpan, LayerCacheStep};
use anyhow::{Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy;
use mlx_native::{KernelRegistry, MlxBuffer, MlxDevice};

#[cfg(test)]
thread_local! { static OMIT_COPY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
#[cfg(test)]
pub(super) fn set_omit_copy(value: bool) {
    OMIT_COPY.with(|flag| flag.set(value));
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_window_write(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    layer_cache: &LayerCache,
    window_cache_input: &MlxBuffer,
    window_write_inputs: &[&MlxBuffer],
    layer_span: Option<&LayerCacheSpan>,
    layer_step: Option<&LayerCacheStep>,
    window_write_count: usize,
    window_capacity: usize,
    head_dim: usize,
) -> Result<()> {
    session.barrier_between(window_write_inputs, &[&layer_cache.window_kv]);
    #[cfg(test)]
    if OMIT_COPY.with(std::cell::Cell::get) {
        return Ok(());
    }
    dispatch_kv_cache_copy(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        window_cache_input,
        &layer_cache.window_kv,
        u32::try_from(
            layer_span
                .map(|span| span.window_write_start)
                .or_else(|| layer_step.map(|step| step.window_write_slot))
                .context("DeepSeek-V4 compressed window write slot is missing")?,
        )
        .context("DeepSeek-V4 compressed window write slot exceeds u32")?,
        head_dim as u32,
        u32::try_from(window_write_count)
            .context("DeepSeek-V4 compressed window write count exceeds u32")?,
        window_capacity as u32,
        true,
    )?;
    Ok(())
}
