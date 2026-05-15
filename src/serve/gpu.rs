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
    /// Secondary pre-warmed kernel registry for the parallel-encode worker
    /// thread (ADR-031 Phase B, Option A).  `Some` only when
    /// `HF2Q_PARALLEL_ENCODE=1` was set at process start; `None` otherwise,
    /// keeping the default path zero-cost.
    ///
    /// Life-cycle: `take_worker_registry` moves it out for one
    /// `forward_decode` call; `encode_parallel_layers_chunked` returns it
    /// via mpsc; `put_worker_registry` stores it back so the next token
    /// finds it here again.
    pub worker_registry: Option<KernelRegistry>,
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
    /// When `HF2Q_PARALLEL_ENCODE=1` is set at process start, also allocates
    /// and registers an identical secondary `KernelRegistry` for the
    /// parallel-encode worker thread.  One-time ~5 ms startup cost; paid only
    /// on opt-in.
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
        // ADR-011 Phase 2 Wave 4 (flash_attn_prefill wire-up):
        //   Flash-attention tiled prefill kernels replace sdpa/sdpa_sliding for
        //   batched prefill. Three registration calls cover (1) the D=256
        //   main kernel (bf16 Q/K/V/O, BQ=32, BK=16), (2) the D=512 NSG=8
        //   llama.cpp-derived main kernel (bf16, NQPSG=8, NCPSG=64), (3) the
        //   SWA / causal mask builder (Wave 2D, shape [qL, kL] broadcast
        //   across batch + heads), and (4) the tile-skip pre-pass classifier
        //   (Wave 2E, one byte per (qtile, ktile) from the mask). See
        //   docs/ADR-011-phase2-wave4-wire-up-verification.md.
        mlx_native::ops::flash_attn_prefill::register(&mut registry);
        mlx_native::ops::flash_attn_prefill_d512::register(&mut registry);
        mlx_native::ops::flash_attn_prefill_mask::register(&mut registry);
        mlx_native::ops::flash_attn_prefill_blk::register(&mut registry);

        // ADR-031 Phase B (Option A): allocate a second identical registry for
        // the parallel-encode worker thread, but ONLY when opt-in is set.
        // Using std::env::var directly here (not INVESTIGATION_ENV) because
        // LazyLock semantics allow either init order, and env::var is cheaper
        // and sufficient for this single binary decision at model load.
        let worker_registry = if std::env::var("HF2Q_PARALLEL_ENCODE").as_deref() == Ok("1") {
            let mut wreg = KernelRegistry::new();
            // Mirror the EXACT same registrations as the main registry above.
            // Chesterton's fence: if a new kernel family is added to the main
            // registry block, it MUST also be added here to keep the worker
            // registry warm for all decode-hot kernels.
            mlx_native::ops::hadamard_quantize_kv::register(&mut wreg);
            mlx_native::ops::flash_attn_vec_tq::register(&mut wreg);
            mlx_native::ops::flash_attn_vec::register(&mut wreg);
            wreg.register_source("fwht_standalone_f32_d256", fwht_src);
            wreg.register_source("fwht_standalone_f32_d512", fwht_src);
            mlx_native::ops::flash_attn_prefill::register(&mut wreg);
            mlx_native::ops::flash_attn_prefill_d512::register(&mut wreg);
            mlx_native::ops::flash_attn_prefill_mask::register(&mut wreg);
            mlx_native::ops::flash_attn_prefill_blk::register(&mut wreg);
            tracing::info!("mlx-native GpuContext: worker KernelRegistry pre-warmed (HF2Q_PARALLEL_ENCODE=1)");
            Some(wreg)
        } else {
            None
        };

        tracing::info!("mlx-native GpuContext initialized on {}", gpu_name);
        Ok(Self { executor, registry, worker_registry })
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

    /// Move the worker registry out for use by `encode_parallel_layers_chunked`.
    ///
    /// Returns `None` if `HF2Q_PARALLEL_ENCODE=1` was not set at process start
    /// (i.e. the worker registry was never allocated) or if it has already been
    /// taken and not yet returned (panic-safe: caller gets `None` and can error
    /// cleanly via the `ok_or_else` pattern in B3).
    #[inline]
    pub fn take_worker_registry(&mut self) -> Option<KernelRegistry> {
        self.worker_registry.take()
    }

    /// Return the worker registry after `encode_parallel_layers_chunked`
    /// completes.  Called unconditionally on every `PARALLEL=ON` forward_decode
    /// return path so the next token's parallel split finds the registry here.
    #[inline]
    pub fn put_worker_registry(&mut self, reg: KernelRegistry) {
        self.worker_registry = Some(reg);
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
        assert!(ctx.worker_registry.is_none(), "worker_registry should be None when HF2Q_PARALLEL_ENCODE is unset");
        println!("GpuContext GPU: {}", ctx.gpu_name());
    }

    #[test]
    fn test_worker_registry_round_trip() {
        // Simulate take/put without actually setting HF2Q_PARALLEL_ENCODE
        // (worker_registry will be None in this test env).
        let mut ctx = GpuContext::new().expect("GpuContext::new");
        assert!(ctx.take_worker_registry().is_none());
        // put_worker_registry with a fresh registry still works.
        let reg = KernelRegistry::new();
        ctx.put_worker_registry(reg);
        assert!(ctx.worker_registry.is_some());
        let _reg = ctx.take_worker_registry();
        assert!(ctx.worker_registry.is_none());
    }
}
