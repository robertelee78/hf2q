//! Static round-to-nearest quantization: f16, q8, q4, q2.
//!
//! Implements the Quantizer trait for simple static quantization methods.
//! Non-weight tensors (layer norms, biases, scalars) are preserved at full precision.
//! Weight tensors are quantized with per-group scale factors.

use crate::ir::{DType, QuantizedTensor, TensorQuantInfo, TensorRef};
use crate::quantize::{LayerQuantConfig, QuantizeError, Quantizer};

/// Static quantizer implementing round-to-nearest with per-group scaling.
pub struct StaticQuantizer {
    /// Name for display
    name: String,
    /// Default bit width
    default_bits: u8,
}

impl StaticQuantizer {
    /// Create a new static quantizer for the given method.
    pub fn new(method: &str) -> Result<Self, QuantizeError> {
        let default_bits = match method {
            "f16" => 16,
            "q8" => 8,
            "q4" => 4,
            "q2" => 2,
            _ => {
                return Err(QuantizeError::UnsupportedMethod {
                    method: method.to_string(),
                })
            }
        };

        Ok(Self {
            name: method.to_string(),
            default_bits,
        })
    }
}

impl Quantizer for StaticQuantizer {
    fn name(&self) -> &str {
        &self.name
    }

    fn requires_calibration(&self) -> bool {
        false
    }

    fn quantize_tensor(
        &self,
        tensor: &TensorRef,
        config: &LayerQuantConfig,
    ) -> Result<QuantizedTensor, QuantizeError> {
        // Preserved tensors (norms, biases, etc.) pass through at original precision
        if config.preserve {
            return Ok(preserve_tensor(tensor));
        }

        // f16 mode: only bf16→f16 conversion, no quantization
        if self.default_bits == 16 {
            return convert_to_f16(tensor);
        }

        // Quantize weight tensor
        quantize_weight(tensor, config.bits, config.group_size)
    }
}

/// Preserve a tensor at full precision (for norms, biases, etc.)
fn preserve_tensor(tensor: &TensorRef) -> QuantizedTensor {
    // If bf16, convert to f16 first
    let (data, dtype) = if tensor.dtype == DType::BF16 {
        match tensor.to_f16() {
            Ok(converted) => (converted.data, DType::F16),
            Err(_) => (tensor.data.clone(), tensor.dtype),
        }
    } else {
        (tensor.data.clone(), tensor.dtype)
    };

    QuantizedTensor {
        name: tensor.name.clone(),
        shape: tensor.shape.clone(),
        original_dtype: tensor.dtype,
        data,
        quant_info: TensorQuantInfo {
            method: "passthrough".to_string(),
            bits: dtype.element_size() as u8 * 8,
            group_size: 0,
            preserved: true,
            scales: None,
            biases: None,
            ggml_type: None,
        },
    }
}

/// Convert a tensor to f16 (lossless for f16 input, bf16→f16 for bf16 input).
fn convert_to_f16(tensor: &TensorRef) -> Result<QuantizedTensor, QuantizeError> {
    let converted = match tensor.dtype {
        DType::BF16 => tensor.to_f16().map_err(QuantizeError::IrError)?,
        DType::F16 => tensor.clone(),
        DType::F32 => {
            // f32→f16 conversion
            let mut f16_data = Vec::with_capacity(tensor.numel() * 2);
            for chunk in tensor.data.chunks_exact(4) {
                let f32_val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let f16_val = half::f16::from_f32(f32_val);
                f16_data.extend_from_slice(&f16_val.to_le_bytes());
            }
            TensorRef {
                name: tensor.name.clone(),
                shape: tensor.shape.clone(),
                dtype: DType::F16,
                data: f16_data,
            }
        }
        _ => {
            return Err(QuantizeError::TensorQuantizeFailed {
                tensor: tensor.name.clone(),
                reason: format!("Cannot convert {} to f16", tensor.dtype),
            });
        }
    };

    Ok(QuantizedTensor {
        name: converted.name.clone(),
        shape: converted.shape.clone(),
        original_dtype: tensor.dtype,
        data: converted.data,
        quant_info: TensorQuantInfo {
            method: "f16".to_string(),
            bits: 16,
            group_size: 0,
            preserved: false,
            scales: None,
            biases: None,
            ggml_type: None,
        },
    })
}

/// Quantize a weight tensor to the specified bit width using round-to-nearest.
///
/// Uses symmetric min-max quantization per group:
/// 1. Divide tensor into groups along the last dimension
/// 2. For each group, find absmax
/// 3. Compute scale = absmax / (2^(bits-1) - 1)
/// 4. Quantize: q = round(x / scale)
/// 5. Store quantized values packed into bytes
fn quantize_weight(
    tensor: &TensorRef,
    bits: u8,
    group_size: usize,
) -> Result<QuantizedTensor, QuantizeError> {
    // First convert to f16 if bf16
    let tensor = if tensor.dtype == DType::BF16 {
        tensor.to_f16().map_err(QuantizeError::IrError)?
    } else {
        tensor.clone()
    };

    // Get f32 values for quantization math
    let f32_values = tensor_to_f32(&tensor)?;

    let total_elements = f32_values.len();

    // Pad to group_size boundary if needed
    let effective_group_size = if total_elements < group_size {
        total_elements
    } else {
        group_size
    };

    let num_groups = total_elements.div_ceil(effective_group_size);
    let qmax = ((1u32 << (bits - 1)) - 1) as f32;

    // Compute scales per group
    let mut scales_f16 = Vec::with_capacity(num_groups * 2);
    let mut quantized_values: Vec<i8> = Vec::with_capacity(total_elements);

    for g in 0..num_groups {
        let start = g * effective_group_size;
        let end = (start + effective_group_size).min(total_elements);
        let group = &f32_values[start..end];

        // Find absmax for symmetric quantization
        let absmax = group.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));

        let scale = if absmax == 0.0 {
            0.0f32
        } else {
            absmax / qmax
        };

        // Store scale as f16
        let scale_f16 = half::f16::from_f32(scale);
        scales_f16.extend_from_slice(&scale_f16.to_le_bytes());

        // Quantize each value in the group
        for &val in group {
            let q = if scale == 0.0 {
                0i8
            } else {
                let scaled = val / scale;
                scaled.round().clamp(-qmax, qmax) as i8
            };
            quantized_values.push(q);
        }
    }

    // Pack quantized values into bytes
    let packed_data = pack_quantized(&quantized_values, bits);

    Ok(QuantizedTensor {
        name: tensor.name.clone(),
        shape: tensor.shape.clone(),
        original_dtype: tensor.dtype,
        data: packed_data,
        quant_info: TensorQuantInfo {
            method: format!("q{}", bits),
            bits,
            group_size: effective_group_size,
            preserved: false,
            scales: Some(scales_f16),
            biases: None,
            ggml_type: None,
        },
    })
}

/// Convert tensor data to f32 values.
fn tensor_to_f32(tensor: &TensorRef) -> Result<Vec<f32>, QuantizeError> {
    match tensor.dtype {
        DType::F32 => {
            let values: Vec<f32> = tensor
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(values)
        }
        DType::F16 => {
            let values: Vec<f32> = tensor
                .data
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();
            Ok(values)
        }
        DType::BF16 => {
            let values: Vec<f32> = tensor
                .data
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect();
            Ok(values)
        }
        _ => Err(QuantizeError::TensorQuantizeFailed {
            tensor: tensor.name.clone(),
            reason: format!("Cannot quantize dtype {}", tensor.dtype),
        }),
    }
}

/// Pack quantized i8 values into a byte array with bit packing.
///
/// For 8-bit: 1 value per byte (straightforward)
/// For 4-bit: 2 values per byte (low nibble first, then high nibble)
/// For 2-bit: 4 values per byte
fn pack_quantized(values: &[i8], bits: u8) -> Vec<u8> {
    match bits {
        8 => {
            // 8-bit: each value is one byte (interpret as u8)
            values.iter().map(|&v| v as u8).collect()
        }
        4 => {
            // 4-bit: pack two values per byte
            let mut packed = Vec::with_capacity(values.len().div_ceil(2));
            for pair in values.chunks(2) {
                let lo = (pair[0] & 0x0F) as u8;
                let hi = if pair.len() > 1 {
                    ((pair[1] & 0x0F) as u8) << 4
                } else {
                    0
                };
                packed.push(lo | hi);
            }
            packed
        }
        2 => {
            // 2-bit: pack four values per byte
            let mut packed = Vec::with_capacity(values.len().div_ceil(4));
            for quad in values.chunks(4) {
                let mut byte = 0u8;
                for (i, &v) in quad.iter().enumerate() {
                    byte |= ((v & 0x03) as u8) << (i * 2);
                }
                packed.push(byte);
            }
            packed
        }
        _ => {
            // Fallback: store as bytes
            values.iter().map(|&v| v as u8).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_f16_tensor(name: &str, shape: Vec<usize>, values: &[f32]) -> TensorRef {
        let data: Vec<u8> = values
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        TensorRef {
            name: name.to_string(),
            shape,
            dtype: DType::F16,
            data,
        }
    }

    #[test]
    fn test_preserve_tensor() {
        let norm = TensorRef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            shape: vec![4],
            dtype: DType::F16,
            data: vec![0u8; 8],
        };

        let result = preserve_tensor(&norm);
        assert!(result.quant_info.preserved);
        assert_eq!(result.quant_info.method, "passthrough");
        assert_eq!(result.data.len(), 8);
    }

    #[test]
    fn test_f16_passthrough() {
        let tensor = make_f16_tensor("test", vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);

        let quantizer = StaticQuantizer::new("f16").unwrap();
        let config = LayerQuantConfig {
            bits: 16,
            group_size: 64,
            preserve: false,
        };

        let result = quantizer.quantize_tensor(&tensor, &config).unwrap();
        assert_eq!(result.quant_info.method, "f16");
        assert_eq!(result.quant_info.bits, 16);
        assert_eq!(result.data.len(), 8); // 4 f16 values * 2 bytes
    }

    #[test]
    fn test_q8_quantization() {
        let tensor = make_f16_tensor("weight", vec![2, 4], &[1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 0.0]);

        let result = quantize_weight(&tensor, 8, 4).unwrap();
        assert_eq!(result.quant_info.method, "q8");
        assert_eq!(result.quant_info.bits, 8);
        assert!(result.quant_info.scales.is_some());
        // 8 values at 8 bits = 8 bytes
        assert_eq!(result.data.len(), 8);
    }

    #[test]
    fn test_q4_quantization() {
        let tensor = make_f16_tensor("weight", vec![2, 4], &[1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 0.0]);

        let result = quantize_weight(&tensor, 4, 4).unwrap();
        assert_eq!(result.quant_info.method, "q4");
        assert_eq!(result.quant_info.bits, 4);
        assert!(result.quant_info.scales.is_some());
        // 8 values at 4 bits = 4 bytes
        assert_eq!(result.data.len(), 4);
    }

    #[test]
    fn test_q2_quantization() {
        let tensor = make_f16_tensor("weight", vec![2, 4], &[1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 0.0]);

        let result = quantize_weight(&tensor, 2, 4).unwrap();
        assert_eq!(result.quant_info.method, "q2");
        assert_eq!(result.quant_info.bits, 2);
        // 8 values at 2 bits = 2 bytes
        assert_eq!(result.data.len(), 2);
    }

    #[test]
    fn test_preserved_tensor_not_quantized() {
        let norm = TensorRef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            shape: vec![4],
            dtype: DType::F16,
            data: vec![0u8; 8],
        };

        let quantizer = StaticQuantizer::new("q4").unwrap();
        let config = LayerQuantConfig {
            bits: 4,
            group_size: 64,
            preserve: true, // this tensor should be preserved
        };

        let result = quantizer.quantize_tensor(&norm, &config).unwrap();
        assert!(result.quant_info.preserved);
        assert_eq!(result.data.len(), 8); // unchanged
    }

    #[test]
    fn test_pack_4bit() {
        let values: Vec<i8> = vec![3, -2, 1, 0];
        let packed = pack_quantized(&values, 4);
        assert_eq!(packed.len(), 2);
        // First byte: low nibble = 3 (0x03), high nibble = -2 & 0x0F = 0x0E → 0xE3
        assert_eq!(packed[0] & 0x0F, 0x03);
    }

    #[test]
    fn test_bf16_tensor_quantization() {
        // Create bf16 tensor
        let data: Vec<u8> = [1.0f32, -1.0, 0.5, -0.5]
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();

        let tensor = TensorRef {
            name: "weight".to_string(),
            shape: vec![2, 2],
            dtype: DType::BF16,
            data,
        };

        let result = quantize_weight(&tensor, 4, 4).unwrap();
        assert_eq!(result.quant_info.bits, 4);
    }

    #[test]
    fn test_all_zeros_quantization() {
        let tensor = make_f16_tensor("weight", vec![2, 2], &[0.0, 0.0, 0.0, 0.0]);
        let result = quantize_weight(&tensor, 4, 4).unwrap();
        // All zeros should quantize without error
        assert_eq!(result.quant_info.bits, 4);
    }
}
