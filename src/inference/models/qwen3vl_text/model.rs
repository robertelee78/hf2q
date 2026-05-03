//! Qwen3-VL text-LM bundle: config + weights + GPU context owner.
//!
//! Mirrors the role of [`crate::inference::models::qwen35::model::Qwen35Model`]
//! for the Qwen3-VL text family. iter-228a holds only load surface;
//! iter-228b adds the `forward_*` methods.

use anyhow::{Context, Result};
use mlx_native::gguf::GgufFile;

use crate::serve::gpu::GpuContext;
use crate::serve::header::LoadProgress;

use super::{Qwen3VlTextConfig, Qwen3VlTextWeights};

/// All artifacts the SERVE worker needs to run a Qwen3-VL text LM.
///
/// Owned by-value so the worker thread can move it across the
/// per-request dispatch arms without refcounting.
pub struct Qwen3VlTextModel {
    /// Architecture config parsed from GGUF metadata.
    pub cfg: Qwen3VlTextConfig,
    /// Loaded weights (per-layer + global).
    pub weights: Qwen3VlTextWeights,
    /// GPU context owning the Metal device + command queue + scratch
    /// pool. iter-228b's forward path acquires command buffers from
    /// this context.
    pub ctx: GpuContext,
}

impl Qwen3VlTextModel {
    /// Open a Qwen3-VL text-LM GGUF and load every weight + initialize
    /// the GPU context.
    ///
    /// Mirrors [`crate::inference::models::qwen35::model::Qwen35Model::load_from_gguf`]
    /// in shape: parse config, init GPU, load all weights against the
    /// Metal device, return the bundle.
    ///
    /// `progress` is driven layer-by-layer. Pass an
    /// [`LoadProgress::new(false, 0, n_layers)`] for the silent
    /// (non-TTY / `-v`) path.
    pub fn load_from_gguf(
        gguf: &GgufFile,
        progress: &mut LoadProgress,
    ) -> Result<Self> {
        let cfg = Qwen3VlTextConfig::from_gguf(gguf)
            .context("Qwen3VlTextConfig::from_gguf")?;
        let ctx = GpuContext::new()
            .map_err(|e| anyhow::anyhow!("mlx-native init failed: {e}"))?;
        // Borrow the device from the context for the load pass; the
        // context owns the Metal device handle for the lifetime of
        // the bundle.
        let weights =
            Qwen3VlTextWeights::load_from_gguf(gguf, &cfg, ctx.device(), progress)
                .context("Qwen3VlTextWeights::load_from_gguf")?;
        Ok(Self { cfg, weights, ctx })
    }

    /// Parse the config without loading any weights. Used by callers
    /// that need to size the [`LoadProgress`] denominator before
    /// starting the per-layer load loop.
    pub fn load_config_only(gguf: &GgufFile) -> Result<Qwen3VlTextConfig> {
        Qwen3VlTextConfig::from_gguf(gguf)
    }
}
