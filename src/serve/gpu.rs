//! mlx-native integration layer for hf2q inference.
//!
//! Provides [`GpuContext`] — a thin wrapper that holds the [`GraphExecutor`]
//! and [`KernelRegistry`] for the mlx-native backend.
//!
//! # ADR-006 Phase 5: per-op cutover
//!
//! During migration, the forward pass can use either candle or mlx-native as
//! the GPU compute backend.  The `--backend candle|mlx-native` CLI flag
//! selects the backend at runtime.
//!
//! # Bridge: candle weights -> MlxBuffer (load-time copy)
//!
//! candle and mlx-native use different Rust bindings for Metal buffers
//! (`candle-metal-kernels::Buffer` vs `metal::Buffer` from the `metal` crate).
//! These types wrap the same Objective-C MTLBuffer protocol but are
//! incompatible at the Rust type level.
//!
//! Instead of zero-copy buffer sharing, the bridge copies weight bytes at
//! model load time.  For quantized weights (`QTensor`), this uses
//! `QTensor::data()`.  For dense weights (`Tensor`), this uses
//! `Tensor::to_vec1`.  Both are one-time costs at model load.
//!
//! At inference time, the MlxBuffer weights are used directly by mlx-native's
//! kernels — no per-token copies.

#[cfg(feature = "mlx-native-backend")]
use mlx_native::{GraphExecutor, KernelRegistry, MlxBuffer, MlxDevice};

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
    /// Raw pointer to the Metal device, cached to avoid borrow conflicts.
    /// SAFETY: the metal device lives as long as the executor (which owns it).
    /// This pointer is never dereferenced after the executor is dropped.
    metal_device_ptr: *const mlx_native::metal::DeviceRef,
}

// SAFETY: The metal::DeviceRef is Send+Sync (MTLDevice is thread-safe).
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

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
        let metal_device_ptr: *const mlx_native::metal::DeviceRef = device.metal_device();
        let executor = GraphExecutor::new(device);
        let mut registry = KernelRegistry::new();
        // Register TurboQuant KV cache kernels (ADR-007 Phase 1.2).
        mlx_native::ops::hadamard_quantize_kv::register(&mut registry);
        mlx_native::ops::flash_attn_vec_tq::register(&mut registry);
        tracing::info!("mlx-native GpuContext initialized on {}", gpu_name);
        Ok(Self { executor, registry, metal_device_ptr })
    }

    /// Get a reference to the metal device without borrowing executor.
    ///
    /// SAFETY: the returned reference is valid for the lifetime of this GpuContext.
    #[inline]
    pub fn metal_device_ref(&self) -> &mlx_native::metal::DeviceRef {
        unsafe { &*self.metal_device_ptr }
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
// Bridge: candle dense Tensor -> MlxBuffer (load-time copy)
// ---------------------------------------------------------------------------

/// Copy a candle `Tensor`'s data into a fresh `MlxBuffer`.
///
/// This is a load-time copy, not zero-copy.  Both candle and mlx-native use
/// `StorageModeShared` Metal buffers, but they use different Rust `metal`
/// bindings (`candle-metal-kernels::Buffer` vs `metal::Buffer`) that are
/// type-incompatible.
///
/// For F32 tensors (norm weights, layer scalars, etc.), this copies through
/// the CPU.  The cost is negligible for the small tensors involved.
#[cfg(feature = "mlx-native-backend")]
pub fn candle_tensor_to_mlx_buffer(
    tensor: &candle_core::Tensor,
    mlx_device: &MlxDevice,
) -> anyhow::Result<MlxBuffer> {
    // Force contiguous + F32 for a clean byte layout.
    let t = tensor.to_dtype(candle_core::DType::F32)?.contiguous()?;
    let data: Vec<f32> = t.to_vec1()
        .or_else(|_| {
            // Multi-dimensional: flatten first.
            t.flatten_all()?.to_vec1()
        })
        .map_err(|e| anyhow::anyhow!("candle_tensor_to_mlx_buffer: to_vec1 failed: {e}"))?;

    let byte_len = data.len() * std::mem::size_of::<f32>();
    let shape: Vec<usize> = tensor.dims().to_vec();

    let mut mlx_buf = mlx_device.alloc_buffer(
        byte_len,
        mlx_native::DType::F32,
        shape,
    ).map_err(|e| anyhow::anyhow!("MlxBuffer alloc failed: {e}"))?;

    // Write f32 data into the Metal buffer.
    let dst: &mut [f32] = mlx_buf.as_mut_slice()
        .map_err(|e| anyhow::anyhow!("MlxBuffer::as_mut_slice failed: {e}"))?;
    dst.copy_from_slice(&data);

    Ok(mlx_buf)
}

/// Copy a candle F16 `Tensor`'s data into a fresh `MlxBuffer` with F16 dtype.
#[cfg(feature = "mlx-native-backend")]
pub fn candle_tensor_f16_to_mlx_buffer(
    tensor: &candle_core::Tensor,
    mlx_device: &MlxDevice,
) -> anyhow::Result<MlxBuffer> {
    let t = tensor.to_dtype(candle_core::DType::F16)?.contiguous()?;
    // Read raw bytes through flatten -> to_vec1 as u16 (f16 bits)
    let flat = t.flatten_all()?;
    let (storage, layout) = flat.storage_and_layout();
    let byte_len = flat.elem_count() * 2; // f16 = 2 bytes

    let shape: Vec<usize> = tensor.dims().to_vec();

    let mut mlx_buf = mlx_device.alloc_buffer(
        byte_len,
        mlx_native::DType::F16,
        shape,
    ).map_err(|e| anyhow::anyhow!("MlxBuffer alloc failed: {e}"))?;

    // Use the raw contents pointer for a direct memcpy.
    // On Metal with StorageModeShared, contents() gives CPU-accessible memory.
    match &*storage {
        candle_core::Storage::Metal(ms) => {
            let src_ptr = ms.buffer().contents() as *const u8;
            let src_offset = layout.start_offset() * 2; // f16 = 2 bytes
            let dst_ptr = mlx_buf.contents_ptr() as *mut u8;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_ptr.add(src_offset),
                    dst_ptr,
                    byte_len,
                );
            }
        }
        _ => anyhow::bail!("candle_tensor_f16_to_mlx_buffer: expected Metal storage"),
    }

    Ok(mlx_buf)
}

// ---------------------------------------------------------------------------
// Bridge: candle QTensor -> MlxBuffer (load-time copy)
// ---------------------------------------------------------------------------

/// Load a candle `QTensor`'s raw quantized bytes into a fresh `MlxBuffer`.
///
/// The returned buffer contains the raw GGML block data that
/// `quantized_matmul_ggml` consumes directly.
///
/// This is NOT zero-copy — it reads the bytes via `QTensor::data()` and
/// writes them into a new mlx-native Metal allocation.  Acceptable because
/// it only happens once at model load time.
#[cfg(feature = "mlx-native-backend")]
pub fn load_qtensor_to_mlx_buffer(
    qtensor: &candle_core::quantized::QTensor,
    mlx_device: &MlxDevice,
) -> anyhow::Result<(MlxBuffer, QuantWeightInfo)> {
    let raw_bytes = qtensor.data()
        .map_err(|e| anyhow::anyhow!("QTensor::data() failed: {e}"))?;
    let byte_len = raw_bytes.len();
    let ggml_dtype = qtensor.dtype();
    let shape = qtensor.shape();

    // Allocate an MlxBuffer and copy the raw GGML block bytes into it.
    let mut mlx_buf = mlx_device.alloc_buffer(
        byte_len,
        mlx_native::DType::U8,
        vec![byte_len],
    ).map_err(|e| anyhow::anyhow!("MlxBuffer alloc failed: {e}"))?;

    // Write the raw bytes into the Metal buffer (CPU-accessible on unified memory).
    let dst: &mut [u8] = mlx_buf.as_mut_slice()
        .map_err(|e| anyhow::anyhow!("MlxBuffer::as_mut_slice failed: {e}"))?;
    dst.copy_from_slice(&raw_bytes);

    let info = QuantWeightInfo {
        ggml_dtype,
        rows: shape.dims()[0],
        cols: if shape.dims().len() > 1 { shape.dims()[1] } else { 1 },
    };

    Ok((mlx_buf, info))
}

/// Load a candle `QMatMul`'s quantized weights into a fresh `MlxBuffer`.
#[cfg(feature = "mlx-native-backend")]
pub fn load_qmatmul_to_mlx_buffer(
    qmatmul: &candle_core::quantized::QMatMul,
    mlx_device: &MlxDevice,
) -> anyhow::Result<(MlxBuffer, QuantWeightInfo)> {
    use candle_core::quantized::QMatMul;

    match qmatmul {
        QMatMul::QTensor(qt) => load_qtensor_to_mlx_buffer(qt, mlx_device),
        QMatMul::Tensor(_) | QMatMul::TensorF16(_) => {
            anyhow::bail!(
                "load_qmatmul_to_mlx_buffer: expected QTensor variant, got dequantized Tensor"
            )
        }
    }
}

/// Information about a quantized weight loaded from candle.
#[cfg(feature = "mlx-native-backend")]
#[derive(Debug, Clone, Copy)]
pub struct QuantWeightInfo {
    /// GGML quantization type (Q4_0, Q6_K, Q8_0, etc.).
    pub ggml_dtype: candle_core::quantized::GgmlDType,
    /// Number of output rows (N dimension of the weight matrix).
    pub rows: usize,
    /// Number of input columns (K dimension of the weight matrix).
    pub cols: usize,
}

// ---------------------------------------------------------------------------
// Type mapping
// ---------------------------------------------------------------------------

/// Map a candle `DType` to an mlx-native `DType`.
#[cfg(feature = "mlx-native-backend")]
pub fn candle_dtype_to_mlx(dt: candle_core::DType) -> mlx_native::DType {
    match dt {
        candle_core::DType::F32 => mlx_native::DType::F32,
        candle_core::DType::F16 => mlx_native::DType::F16,
        candle_core::DType::BF16 => mlx_native::DType::BF16,
        candle_core::DType::U8 => mlx_native::DType::U8,
        candle_core::DType::U32 => mlx_native::DType::U32,
        // Non-exhaustive fallback for types without a direct mapping.
        _ => mlx_native::DType::F32,
    }
}

/// Map a candle `GgmlDType` to an mlx-native `GgmlType`.
///
/// Only the types actually used in the model are mapped.
#[cfg(feature = "mlx-native-backend")]
pub fn candle_ggml_to_mlx(
    dt: candle_core::quantized::GgmlDType,
) -> anyhow::Result<mlx_native::GgmlType> {
    use candle_core::quantized::GgmlDType;
    match dt {
        GgmlDType::Q4_0 => Ok(mlx_native::GgmlType::Q4_0),
        GgmlDType::Q8_0 => Ok(mlx_native::GgmlType::Q8_0),
        GgmlDType::Q6K => Ok(mlx_native::GgmlType::Q6_K),
        other => anyhow::bail!(
            "candle_ggml_to_mlx: unsupported GGML type {:?}",
            other
        ),
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

    #[test]
    fn test_dtype_mapping() {
        assert_eq!(
            candle_dtype_to_mlx(candle_core::DType::F32),
            mlx_native::DType::F32
        );
        assert_eq!(
            candle_dtype_to_mlx(candle_core::DType::F16),
            mlx_native::DType::F16
        );
        assert_eq!(
            candle_dtype_to_mlx(candle_core::DType::BF16),
            mlx_native::DType::BF16
        );
    }

    #[test]
    fn test_ggml_type_mapping() {
        use candle_core::quantized::GgmlDType;
        assert!(candle_ggml_to_mlx(GgmlDType::Q4_0).is_ok());
        assert!(candle_ggml_to_mlx(GgmlDType::Q8_0).is_ok());
        assert!(candle_ggml_to_mlx(GgmlDType::Q6K).is_ok());
    }
}
