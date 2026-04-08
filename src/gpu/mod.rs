//! GPU compute backend via Candle.
//!
//! Provides device selection, tensor conversion from hf2q IR to Candle tensors,
//! transformer forward pass for activation capture, and tokenizer utilities.

pub mod forward;
pub mod tokenizer;

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use candle_core::{DType as CandleDType, Device, Tensor};
use tracing::info;

#[allow(dead_code)]
use crate::ir::{DType as IrDType, TensorMap, TensorRef};

/// GPU device wrapper indicating which accelerator is in use.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum GpuDevice {
    /// Apple Metal GPU.
    Metal,
    /// NVIDIA CUDA GPU.
    Cuda,
    /// CPU fallback.
    Cpu,
}

impl std::fmt::Display for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuDevice::Metal => write!(f, "Metal"),
            GpuDevice::Cuda => write!(f, "CUDA"),
            GpuDevice::Cpu => write!(f, "CPU"),
        }
    }
}

/// Select the best available Candle device.
///
/// Prefers Metal on macOS, CUDA on Linux/Windows, falls back to CPU.
pub fn select_device() -> Result<(Device, GpuDevice)> {
    // Try Metal first (macOS Apple Silicon)
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                info!("Using Metal GPU device");
                return Ok((device, GpuDevice::Metal));
            }
            Err(e) => {
                tracing::warn!("Metal device unavailable: {}", e);
            }
        }
    }

    // Try CUDA
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(device) => {
                info!("Using CUDA GPU device");
                return Ok((device, GpuDevice::Cuda));
            }
            Err(e) => {
                tracing::warn!("CUDA device unavailable: {}", e);
            }
        }
    }

    info!("Using CPU device (compile with --features metal or --features cuda for GPU)");
    Ok((Device::Cpu, GpuDevice::Cpu))
}

/// Map an hf2q IR dtype to a Candle dtype.
fn ir_dtype_to_candle(dtype: IrDType) -> Result<CandleDType> {
    match dtype {
        IrDType::F32 => Ok(CandleDType::F32),
        IrDType::F16 => Ok(CandleDType::F16),
        IrDType::BF16 => Ok(CandleDType::BF16),
        IrDType::U8 => Ok(CandleDType::U8),
        IrDType::U32 => Ok(CandleDType::U32),
        IrDType::I64 => Ok(CandleDType::I64),
        other => bail!("Unsupported IR dtype for Candle conversion: {}", other),
    }
}

/// Convert a single IR TensorRef into a Candle Tensor on the given device.
pub fn tensor_from_ir(tensor_ref: &TensorRef, device: &Device) -> Result<Tensor> {
    let candle_dtype = ir_dtype_to_candle(tensor_ref.dtype)
        .with_context(|| format!("tensor '{}'", tensor_ref.name))?;

    let shape = &tensor_ref.shape;

    // Validate data length matches shape * element_size
    let expected_bytes = tensor_ref.numel() * tensor_ref.dtype.element_size();
    if tensor_ref.data.len() != expected_bytes {
        bail!(
            "Tensor '{}': expected {} bytes ({} elements x {} bytes), got {}",
            tensor_ref.name,
            expected_bytes,
            tensor_ref.numel(),
            tensor_ref.dtype.element_size(),
            tensor_ref.data.len()
        );
    }

    Tensor::from_raw_buffer(&tensor_ref.data, candle_dtype, shape, device)
        .with_context(|| format!("Failed to create Candle tensor for '{}'", tensor_ref.name))
}

/// Convert an entire IR TensorMap into a HashMap of Candle Tensors.
///
/// All tensors are loaded onto the specified device. Tensors that cannot be
/// converted (unsupported dtype) are skipped with a warning.
pub fn load_tensor_map(
    tensor_map: &TensorMap,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let mut result = HashMap::with_capacity(tensor_map.len());

    for (name, tensor_ref) in tensor_map.tensors.iter() {
        match tensor_from_ir(tensor_ref, device) {
            Ok(tensor) => {
                result.insert(name.clone(), tensor);
            }
            Err(e) => {
                tracing::warn!("Skipping tensor '{}': {}", name, e);
            }
        }
    }

    info!(
        "Loaded {}/{} tensors onto {}",
        result.len(),
        tensor_map.len(),
        device_name(device),
    );

    Ok(result)
}

/// Human-readable name for a Candle device.
fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        Device::Metal(_) => "Metal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType as IrDType, TensorRef};

    #[test]
    fn test_select_device() {
        let result = select_device();
        assert!(result.is_ok());
        let (_device, gpu_device) = result.unwrap();
        // On CI without Metal/CUDA features, should fall back to CPU
        println!("Selected device: {}", gpu_device);
    }

    #[test]
    fn test_ir_dtype_to_candle_supported() {
        assert!(ir_dtype_to_candle(IrDType::F32).is_ok());
        assert!(ir_dtype_to_candle(IrDType::F16).is_ok());
        assert!(ir_dtype_to_candle(IrDType::BF16).is_ok());
        assert!(ir_dtype_to_candle(IrDType::U8).is_ok());
        assert!(ir_dtype_to_candle(IrDType::U32).is_ok());
        assert!(ir_dtype_to_candle(IrDType::I64).is_ok());
    }

    #[test]
    fn test_ir_dtype_to_candle_unsupported() {
        // Bool and I32 are not supported by Candle
        assert!(ir_dtype_to_candle(IrDType::Bool).is_err());
        assert!(ir_dtype_to_candle(IrDType::I32).is_err());
    }

    #[test]
    fn test_tensor_from_ir_f32() {
        let data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let tensor_ref = TensorRef {
            name: "test.weight".to_string(),
            shape: vec![2, 3],
            dtype: IrDType::F32,
            data,
        };

        let device = Device::Cpu;
        let tensor = tensor_from_ir(&tensor_ref, &device).unwrap();
        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(tensor.dtype(), CandleDType::F32);

        let vals: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor_from_ir_bad_size() {
        let tensor_ref = TensorRef {
            name: "bad.weight".to_string(),
            shape: vec![2, 3],
            dtype: IrDType::F32,
            data: vec![0u8; 10], // wrong size: should be 24
        };

        let device = Device::Cpu;
        let result = tensor_from_ir(&tensor_ref, &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_tensor_map() {
        let mut tmap = TensorMap::new();

        let data_f32: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        tmap.insert(TensorRef {
            name: "layer.weight".to_string(),
            shape: vec![2, 2],
            dtype: IrDType::F32,
            data: data_f32,
        });

        let device = Device::Cpu;
        let loaded = load_tensor_map(&tmap, &device).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains_key("layer.weight"));
    }
}
