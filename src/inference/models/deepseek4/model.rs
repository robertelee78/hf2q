//! Native DeepSeek-V4 model ownership and load boundary.

use anyhow::{Context, Result};
use mlx_native::gguf::GgufFile;
use mlx_native::{DType, MlxBuffer};

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
    /// Sticky device receipt for invalid MoE state. Route sanitization keeps
    /// expert pointer arithmetic safe, while activation/reduction kernels keep
    /// non-finites contained; a transaction is publishable only while this
    /// word remains zero.
    pub(super) invalid_moe_status: MlxBuffer,
    #[cfg(test)]
    inject_invalid_moe_route_once: bool,
}

fn validate_moe_status(status: &[u32]) -> Result<()> {
    anyhow::ensure!(
        status.first().copied() == Some(0),
        "DeepSeek-V4 inference rejected invalid MoE state"
    );
    Ok(())
}

impl Deepseek4Model {
    /// Parse and validate the complete verifier artifact before returning a
    /// runnable model bundle. DSpark draft tensors are rejected by the exact
    /// verifier catalog and belong in their own artifact.
    pub fn load_from_gguf(gguf: &GgufFile) -> Result<Self> {
        let cfg = Deepseek4Config::from_gguf(gguf).context("Deepseek4Config::from_gguf")?;
        let mut ctx = GpuContext::new()
            .map_err(|source| anyhow::anyhow!("DeepSeek-V4 Metal initialization: {source}"))?;
        let weights = Deepseek4Weights::load_from_gguf(gguf, &cfg, ctx.device().clone())
            .context("Deepseek4Weights::load_from_gguf")?;
        let native_bf16 = weights
            .native_bf16_dense_matrices(cfg.output_groups as usize)
            .context("inventory DeepSeek-V4 native BF16 dense projections")?;
        ctx.activate_native_bf16_dense(&native_bf16)
            .context("activate DeepSeek-V4 native BF16 dense projections")?;
        let native_scalar_experts = weights
            .native_scalar_expert_matrices(&cfg)
            .context("inventory DeepSeek-V4 native scalar expert projections")?;
        ctx.activate_native_scalar_experts(&native_scalar_experts)
            .context("activate DeepSeek-V4 native scalar expert projections")?;
        let mut invalid_moe_status = ctx
            .device()
            .alloc_buffer(4, DType::U32, vec![1])
            .map_err(|source| anyhow::anyhow!("allocate DeepSeek-V4 MoE status: {source}"))?;
        invalid_moe_status
            .as_mut_slice::<u32>()
            .context("initialize DeepSeek-V4 MoE status")?[0] = 0;
        Ok(Self {
            cfg,
            weights,
            ctx,
            invalid_moe_status,
            #[cfg(test)]
            inject_invalid_moe_route_once: false,
        })
    }

    pub(super) fn begin_moe_transaction(&mut self) -> Result<()> {
        self.invalid_moe_status
            .as_mut_slice::<u32>()
            .context("reset DeepSeek-V4 MoE status")?[0] = 0;
        #[cfg(test)]
        if std::mem::take(&mut self.inject_invalid_moe_route_once) {
            self.inject_invalid_moe_route_for_test()?;
        }
        Ok(())
    }

    pub(super) fn ensure_moe_transaction_clean(&self) -> Result<()> {
        let status = self
            .invalid_moe_status
            .as_slice::<u32>()
            .context("read DeepSeek-V4 MoE status")?;
        validate_moe_status(status)?;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn force_invalid_moe_status_once_for_test(&mut self) {
        self.inject_invalid_moe_route_once = true;
    }

    #[cfg(test)]
    pub(super) fn seed_invalid_moe_status_for_test(&mut self) -> Result<()> {
        self.invalid_moe_status
            .as_mut_slice::<u32>()
            .context("seed DeepSeek-V4 invalid MoE status")?[0] = 1;
        Ok(())
    }

    #[cfg(test)]
    fn inject_invalid_moe_route_for_test(&mut self) -> Result<()> {
        use mlx_native::ops::deepseek_moe_routing::dispatch_deepseek_moe_sanitize_indices;

        let device = self.ctx.device().clone();
        let mut indices = device
            .alloc_buffer(6 * 4, DType::I32, vec![1, 6])
            .context("allocate injected DeepSeek-V4 route indices")?;
        indices
            .as_mut_slice::<i32>()
            .context("write injected DeepSeek-V4 route indices")?
            .copy_from_slice(&[-1, 0, 1, 2, 3, 4]);
        let safe_indices = device
            .alloc_buffer(6 * 4, DType::U32, vec![1, 6])
            .context("allocate injected DeepSeek-V4 safe route indices")?;
        let mut encoder = device
            .command_encoder()
            .context("begin injected DeepSeek-V4 route sanitizer")?;
        dispatch_deepseek_moe_sanitize_indices(
            &mut encoder,
            &mut self.ctx.registry,
            &device,
            &indices,
            &safe_indices,
            &self.invalid_moe_status,
            1,
        )
        .context("encode injected DeepSeek-V4 route sanitizer")?;
        encoder
            .commit_and_wait()
            .context("execute injected DeepSeek-V4 route sanitizer")?;
        Ok(())
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

#[cfg(test)]
mod route_status_tests {
    use super::validate_moe_status;

    #[test]
    fn moe_status_rejects_missing_or_nonzero_device_receipts() {
        assert!(validate_moe_status(&[0]).is_ok());
        assert!(validate_moe_status(&[]).is_err());
        assert!(validate_moe_status(&[1]).is_err());
        assert!(validate_moe_status(&[u32::MAX]).is_err());
    }
}
