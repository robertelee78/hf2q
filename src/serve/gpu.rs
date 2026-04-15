//! mlx-native integration layer for hf2q inference.
//!
//! Provides [`GpuContext`] — a thin wrapper that holds the [`GraphExecutor`]
//! and [`KernelRegistry`] for the mlx-native backend.
//!
//! # ADR-008: candle divorce
//!
//! All candle bridge functions have been removed.  Weights are loaded
//! directly from GGUF via `mlx_native::gguf::GgufFile` into `MlxBuffer`s.
//! The `QuantWeightInfo` struct now uses `mlx_native::GgmlType` directly.

use mlx_native::{GraphExecutor, KernelRegistry, MlxDevice};

/// GPU context for the mlx-native backend.
///
/// Owns the graph executor (which in turn owns the Metal device and command
/// queue) and the pre-warmed kernel registry.  Created once at model load;
/// lives for the duration of inference.
pub struct GpuContext {
    /// Batched dispatch executor — one `CommandEncoder` per forward pass.
    /// Also owns the `MlxDevice`.
    pub executor: GraphExecutor,
    /// Pre-compiled shader pipeline cache.
    pub registry: KernelRegistry,
}

// SAFETY: The metal::DeviceRef is Send+Sync (MTLDevice is thread-safe).
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl GpuContext {
    /// Initialize the mlx-native GPU context.
    ///
    /// Creates the Metal device, graph executor, and an empty kernel registry.
    /// Kernel pipelines are compiled lazily on first use (typically during the
    /// warmup forward passes).
    ///
    /// # Errors
    ///
    /// Returns an error if no Metal device is available.
    pub fn new() -> mlx_native::Result<Self> {
        let device = MlxDevice::new()?;
        let gpu_name = device.name();
        let executor = GraphExecutor::new(device);
        let mut registry = KernelRegistry::new();
        // Register all inference kernels.
        mlx_native::ops::hadamard_quantize_kv::register(&mut registry);
        mlx_native::ops::flash_attn_vec_tq::register(&mut registry);
        // F16 SDPA reduce kernels — reused by TQ SDPA with NWG>1.
        mlx_native::ops::flash_attn_vec::register(&mut registry);
        // Standalone FWHT for TQ SDPA pre/post rotation.
        let fwht_src = mlx_native::ops::fwht_standalone::FWHT_STANDALONE_SHADER_SOURCE;
        registry.register_source("fwht_standalone_f32_d256", fwht_src);
        registry.register_source("fwht_standalone_f32_d512", fwht_src);
        tracing::info!("mlx-native GpuContext initialized on {}", gpu_name);
        Ok(Self { executor, registry })
    }

    /// Borrow the underlying `MlxDevice`.
    #[inline]
    pub fn device(&self) -> &MlxDevice {
        self.executor.device()
    }

    /// Human-readable GPU name (e.g. "Apple M5 Max").
    pub fn gpu_name(&self) -> String {
        self.device().name()
    }

    /// Split borrow: returns (&GraphExecutor, &mut KernelRegistry) to avoid
    /// conflicting borrows when methods need both the device (from executor)
    /// and mutable access to the registry.
    #[inline]
    pub fn split(&mut self) -> (&GraphExecutor, &mut KernelRegistry) {
        (&self.executor, &mut self.registry)
    }
}

// ---------------------------------------------------------------------------
// Quantized weight metadata
// ---------------------------------------------------------------------------

/// Information about a quantized weight loaded from GGUF.
#[derive(Debug, Clone, Copy)]
pub struct QuantWeightInfo {
    /// GGML quantization type (Q4_0, Q6_K, Q8_0, etc.).
    pub ggml_dtype: mlx_native::GgmlType,
    /// Number of output rows (N dimension of the weight matrix).
    pub rows: usize,
    /// Number of input columns (K dimension of the weight matrix).
    pub cols: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_init() {
        let ctx = GpuContext::new().expect("GpuContext::new should succeed on Apple Silicon");
        assert!(!ctx.gpu_name().is_empty());
        println!("GpuContext GPU: {}", ctx.gpu_name());
    }
}
