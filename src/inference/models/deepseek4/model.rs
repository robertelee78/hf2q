//! Native DeepSeek-V4 model ownership and load boundary.

use anyhow::{Context, Result};
use mlx_native::gguf::GgufFile;

use crate::serve::gpu::GpuContext;

use super::cache::{CacheError, Deepseek4Cache, Deepseek4CachePlan};
use super::{Deepseek4Config, Deepseek4Weights};

/// Architecture-specific weights, configuration, and Metal execution state.
///
/// Cache allocation is deliberately request-sized instead of eagerly using
/// the checkpoint's one-million-token maximum.
pub struct Deepseek4Model {
    pub cfg: Deepseek4Config,
    pub weights: Deepseek4Weights,
    pub ctx: GpuContext,
    /// ADR-053: optional GLP runtime steering. When bound, each steered
    /// layer's post-layer HC state (`[rows, 4, hidden]`) is projected
    /// per-stream (one direction per stream index) after the FFN's
    /// `dispatch_hc_post` expand. Off unless `--glp`/`--uncensor` bound one.
    pub glp: Option<crate::inference::glp::BoundGlp>,
}

impl Deepseek4Model {
    /// Parse and validate the complete verifier artifact before returning a
    /// runnable model bundle. DSpark draft tensors are rejected by the exact
    /// verifier catalog and belong in their own artifact.
    pub fn load_from_gguf(gguf: &GgufFile) -> Result<Self> {
        let cfg = Deepseek4Config::from_gguf(gguf).context("Deepseek4Config::from_gguf")?;
        let ctx = GpuContext::new()
            .map_err(|source| anyhow::anyhow!("DeepSeek-V4 Metal initialization: {source}"))?;
        let weights = Deepseek4Weights::load_from_gguf(gguf, &cfg, ctx.device().clone())
            .context("Deepseek4Weights::load_from_gguf")?;
        Ok(Self { cfg, weights, ctx, glp: None })
    }

    /// Parse only the strict architecture metadata without allocating Metal
    /// buffers. Callers use this for admission control and progress sizing.
    pub fn load_config_only(gguf: &GgufFile) -> Result<Deepseek4Config> {
        Deepseek4Config::from_gguf(gguf)
    }

    /// Plan a cache without allocating it, for request admission control.
    pub fn cache_plan(&self, context_length: usize) -> Result<Deepseek4CachePlan, CacheError> {
        Deepseek4CachePlan::for_context(&self.cfg, context_length)
    }

    /// Allocate a coherent cache on the same Metal device and residency set
    /// that own the model weights.
    pub fn allocate_cache(&self, context_length: usize) -> Result<Deepseek4Cache, CacheError> {
        let plan = self.cache_plan(context_length)?;
        Deepseek4Cache::allocate(&plan, self.ctx.device().clone())
    }

    /// Reserve a full logical cache whose append-only KV rows are committed
    /// on first write. Used by multi-agent slots so logical context capacity
    /// is independent from aggregate physical residency.
    pub fn allocate_logical_cache(
        &self,
        context_length: usize,
    ) -> Result<Deepseek4Cache, CacheError> {
        let plan = self.cache_plan(context_length)?;
        Deepseek4Cache::allocate_logical(&plan, self.ctx.device().clone())
    }
}
