//! mlx-native integration layer for hf2q inference.
//!
//! Provides [`GpuContext`] — a thin wrapper that holds the [`GraphExecutor`]
//! and [`KernelRegistry`] for the mlx-native backend.
//!
//! # ADR-006 Phase 5: per-op cutover
//!
//! During migration, the forward pass mixes candle ops and mlx-native ops.
//! Both use `StorageModeShared` Metal buffers on Apple Silicon unified memory,
//! so the same physical memory is accessible by both frameworks without copies.
//!
//! # Weight Format Incompatibility (BLOCKER for zero-copy cutover)
//!
//! candle's `QMatMul` uses **GGML block format** where quantized values, scales,
//! and metadata are interleaved within fixed-size block structs:
//!
//! - `block_q4_0`: 18 bytes per 32 values — `[half d, uint8 qs[16]]`
//! - `block_q6_K`: 210 bytes per 256 values — `[uint8 ql[128], uint8 qh[64],
//!    int8 scales[16], half d]`
//! - `block_q8_0`: 34 bytes per 32 values — `[half d, int8 qs[32]]`
//!
//! mlx-native's `quantized_matmul` uses **MLX affine format** with three
//! separate contiguous buffers:
//!
//! - **weights**: packed uint32 `[N, packed_k]` — 8 values per uint32 for 4-bit
//! - **scales**: bf16 `[N, num_groups]` — one per group
//! - **biases**: bf16 `[N, num_groups]` — one per group
//! - Dequantization: `float_val = scale * quant_val + bias`
//!
//! These formats are **fundamentally incompatible** at the byte level:
//!
//! 1. **Interleaved vs separated**: GGML packs scales inside each block struct;
//!    MLX stores all scales in a separate contiguous buffer.
//! 2. **Scale format**: GGML uses f16 (IEEE half); MLX uses bf16 (bfloat16).
//! 3. **Dequantization formula**: GGML Q4_0 uses `d * (quant - 8)` (symmetric,
//!    no bias); MLX uses `scale * quant + bias` (affine). GGML Q6_K uses a
//!    two-level super-block scheme with nested scales.
//! 4. **Packing order**: GGML Q6_K splits quant bits across `ql` (low 4 bits)
//!    and `qh` (high 2 bits); MLX packs all 6 bits contiguously.
//!
//! **Consequence**: zero-copy buffer sharing between candle's GGUF weights and
//! mlx-native's quantized_matmul is NOT possible.  A conversion step at model
//! load time is required.  See the Phase 5 step 1b plan below.

#[cfg(feature = "mlx-native-backend")]
use mlx_native::{GraphExecutor, KernelRegistry, MlxDevice};

/// GPU context for the mlx-native backend.
///
/// Owns the graph executor (which in turn owns the Metal device and command
/// queue) and the pre-warmed kernel registry.  Created once at model load;
/// lives on `Gemma4Model` alongside (or eventually replacing) the candle
/// `Device`.
///
/// Access the underlying `MlxDevice` via `self.executor.device()`.
#[cfg(feature = "mlx-native-backend")]
pub struct GpuContext {
    /// Batched dispatch executor — one `CommandEncoder` per forward pass.
    /// Also owns the `MlxDevice`.
    pub executor: GraphExecutor,
    /// Pre-compiled shader pipeline cache.
    pub registry: KernelRegistry,
}

#[cfg(feature = "mlx-native-backend")]
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
        let registry = KernelRegistry::new();
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
}

#[cfg(test)]
#[cfg(feature = "mlx-native-backend")]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_init() {
        let ctx = GpuContext::new().expect("GpuContext::new should succeed on Apple Silicon");
        assert!(!ctx.gpu_name().is_empty());
        println!("GpuContext GPU: {}", ctx.gpu_name());
    }
}
