//! MLX output backend — writes MLX-compatible safetensors with config.
//!
//! Output structure:
//! - Consolidated safetensors shards (N input → 4 output)
//! - MLX-compatible config.json
//! - tokenizer.json + tokenizer_config.json (copied from source)
//! - quantization_config.json with per-tensor quant metadata

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use serde_json::Value;
use tracing::{debug, info, warn};

use crate::backends::{BackendError, OutputBackend};
use crate::cli::DEFAULT_OUTPUT_SHARDS;
use crate::ir::{
    FormatWarning, OutputFile, OutputManifest, QuantizedModel, QuantizedTensor, WarningSeverity,
};
use crate::progress::ProgressReporter;

/// MLX output backend.
pub struct MlxBackend;

impl MlxBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MlxBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputBackend for MlxBackend {
    fn name(&self) -> &str {
        "mlx"
    }

    fn validate(&self, model: &QuantizedModel) -> Result<Vec<FormatWarning>, BackendError> {
        let mut warnings = Vec::new();

        if model.tensors.is_empty() {
            return Err(BackendError::ValidationFailed {
                reason: "No tensors to write".to_string(),
            });
        }

        // Check that all tensors have data
        for (name, tensor) in &model.tensors {
            if tensor.data.is_empty() && !tensor.shape.is_empty() {
                let non_zero_shape = tensor.shape.iter().all(|&d| d > 0);
                if non_zero_shape {
                    warnings.push(FormatWarning {
                        message: format!("Tensor '{}' has non-zero shape but empty data", name),
                        severity: WarningSeverity::Warning,
                    });
                }
            }
        }

        Ok(warnings)
    }

    fn write(
        &self,
        model: &QuantizedModel,
        input_dir: &Path,
        output_dir: &Path,
        progress: &ProgressReporter,
    ) -> Result<OutputManifest, BackendError> {
        // Create output directory
        fs::create_dir_all(output_dir).map_err(|e| BackendError::WriteFailed {
            reason: format!("Failed to create output directory: {}", e),
        })?;

        let mut files = Vec::new();

        // 1. Write quantized tensors as safetensors shards
        let shard_files = write_safetensors_shards(model, output_dir, progress)?;
        files.extend(shard_files);

        // 2. Write MLX config.json
        let config_size = write_mlx_config(model, output_dir)?;
        files.push(OutputFile {
            filename: "config.json".to_string(),
            size_bytes: config_size,
        });

        // 3. Copy tokenizer files from source
        let tokenizer_files = copy_tokenizer_files(input_dir, output_dir)?;
        files.extend(tokenizer_files);

        // 4. Write quantization_config.json
        let quant_config_size = write_quant_config(model, output_dir)?;
        files.push(OutputFile {
            filename: "quantization_config.json".to_string(),
            size_bytes: quant_config_size,
        });

        let total_size: u64 = files.iter().map(|f| f.size_bytes).sum();
        let shard_count = files
            .iter()
            .filter(|f| f.filename.ends_with(".safetensors"))
            .count();

        info!(
            output_dir = %output_dir.display(),
            total_size_mb = total_size / (1024 * 1024),
            shard_count,
            "MLX output written"
        );

        Ok(OutputManifest {
            output_dir: output_dir.display().to_string(),
            files,
            total_size_bytes: total_size,
            shard_count,
        })
    }
}

/// Write quantized tensors as consolidated safetensors shards.
/// Consolidates N input shards → DEFAULT_OUTPUT_SHARDS output shards.
fn write_safetensors_shards(
    model: &QuantizedModel,
    output_dir: &Path,
    progress: &ProgressReporter,
) -> Result<Vec<OutputFile>, BackendError> {
    let num_shards = DEFAULT_OUTPUT_SHARDS;

    // Sort tensors by name for deterministic output
    let mut sorted_tensors: Vec<(&String, &QuantizedTensor)> = model.tensors.iter().collect();
    sorted_tensors.sort_by_key(|(name, _)| name.as_str());

    // Distribute tensors across shards roughly evenly by data size
    let total_size: usize = sorted_tensors.iter().map(|(_, t)| t.data.len()).sum();
    let target_shard_size = (total_size / num_shards).max(1);

    let mut shards: Vec<Vec<(&String, &QuantizedTensor)>> = vec![Vec::new(); num_shards];
    let mut shard_sizes = vec![0usize; num_shards];
    let mut current_shard = 0;

    for tensor_pair in &sorted_tensors {
        shards[current_shard].push(*tensor_pair);
        shard_sizes[current_shard] += tensor_pair.1.data.len();

        // Move to next shard if we've exceeded target (but don't go past last shard)
        if shard_sizes[current_shard] >= target_shard_size && current_shard < num_shards - 1 {
            current_shard += 1;
        }
    }

    let pb = progress.bar(num_shards as u64, "Writing MLX shards");

    let mut files = Vec::new();
    let mut weight_map: BTreeMap<String, String> = BTreeMap::new();

    for (shard_idx, shard_tensors) in shards.iter().enumerate() {
        if shard_tensors.is_empty() {
            pb.inc(1);
            continue;
        }

        let shard_filename = if num_shards == 1 {
            "model.safetensors".to_string()
        } else {
            format!(
                "model-{:05}-of-{:05}.safetensors",
                shard_idx + 1,
                num_shards
            )
        };

        let shard_path = output_dir.join(&shard_filename);

        // Build safetensors file for this shard, collecting actual output tensor names
        let mut shard_tensor_names: Vec<String> = Vec::new();
        let shard_size = write_single_safetensors_file(
            shard_tensors,
            &shard_path,
            &mut shard_tensor_names,
        )?;

        // Track weight map for index.json using actual output names
        for tensor_name in &shard_tensor_names {
            weight_map.insert(tensor_name.clone(), shard_filename.clone());
        }

        files.push(OutputFile {
            filename: shard_filename,
            size_bytes: shard_size,
        });

        pb.inc(1);
    }

    // Write index.json if multiple shards
    if num_shards > 1 {
        let index = serde_json::json!({
            "metadata": {
                "total_size": total_size
            },
            "weight_map": weight_map
        });

        let index_path = output_dir.join("model.safetensors.index.json");
        let index_json = serde_json::to_string_pretty(&index)
            .map_err(BackendError::Serialization)?;
        fs::write(&index_path, &index_json).map_err(|e| BackendError::WriteFailed {
            reason: format!("Failed to write index.json: {}", e),
        })?;

        files.push(OutputFile {
            filename: "model.safetensors.index.json".to_string(),
            size_bytes: index_json.len() as u64,
        });
    }

    pb.finish_with_message("MLX shards written");

    Ok(files)
}

/// A serializable tensor entry for the safetensors file.
struct SafetensorEntry {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Write a single safetensors file from a set of tensors.
///
/// MLX-compatible format:
/// - Quantized weights stored as uint32 (8 × 4-bit values per u32)
/// - Scales and biases as sibling tensors: `{module}.scales`, `{module}.biases`
/// - Expert gate_up_proj split into switch_glu.gate_proj + switch_glu.up_proj
///
/// Safetensors binary layout:
/// - 8 bytes: header length (u64 LE)
/// - N bytes: JSON header
/// - remaining: tensor data (concatenated in order)
fn write_single_safetensors_file(
    tensors: &[(&String, &QuantizedTensor)],
    path: &Path,
    output_tensor_names: &mut Vec<String>,
) -> Result<u64, BackendError> {
    // Expand each QuantizedTensor into one or more SafetensorEntry values.
    // Quantized tensors produce weight + scales + biases entries.
    // Expert gate_up_proj tensors are split into gate_proj + up_proj.
    let mut entries: Vec<SafetensorEntry> = Vec::new();

    for (name, tensor) in tensors {
        if tensor.quant_info.preserved || tensor.quant_info.scales.is_none() {
            // Preserved / unquantized tensor — write as-is
            entries.push(SafetensorEntry {
                name: (*name).clone(),
                dtype: output_dtype_string(tensor),
                shape: tensor.shape.clone(),
                data: tensor.data.clone(),
            });
            continue;
        }

        // --- Quantized tensor: convert to MLX format ---
        let bits = tensor.quant_info.bits;
        let group_size = tensor.quant_info.group_size;
        let scales_raw = tensor.quant_info.scales.as_ref().unwrap();

        // 1. Convert signed packed nibbles to unsigned (for MLX affine mode).
        //    For N-bit: XOR each nibble's MSB. For 4-bit packed 2-per-byte: XOR 0x88.
        let unsigned_data = convert_signed_to_unsigned(&tensor.data, bits);

        // 2. Compute MLX shapes:
        //    weight: last dim becomes last_dim * bits / 32 (packed into u32)
        //    scales/biases: last dim becomes last_dim / group_size
        let weight_shape = mlx_weight_shape(&tensor.shape, bits);
        let scales_shape = mlx_scales_shape(&tensor.shape, group_size);

        // 3. Convert scales from f16 to f32 (MLX QuantizedLinear uses f32)
        let scales_f32 = f16_bytes_to_f32_bytes(scales_raw);

        // 4. Create biases (zero-point) tensor: constant 2^(bits-1) for all groups
        let zero_point = (1u32 << (bits - 1)) as f32;
        let num_scale_elements = scales_raw.len() / 2; // f16 = 2 bytes each
        let biases_f32 = vec_f32_to_bytes(zero_point, num_scale_elements);

        // 5. Handle expert tensor naming and gate_up_proj splitting
        if name.ends_with(".experts.gate_up_proj") {
            // Split fused gate+up into separate tensors along axis=-2
            let base = name.strip_suffix(".gate_up_proj").unwrap();
            let split_entries = split_expert_gate_up(
                base,
                &unsigned_data,
                &weight_shape,
                &scales_f32,
                &scales_shape,
                &biases_f32,
            );
            entries.extend(split_entries);
        } else if name.ends_with(".experts.down_proj") {
            // Rename to switch_glu.down_proj
            let base = name.strip_suffix(".down_proj").unwrap();
            let module = format!("{}.switch_glu.down_proj", base);
            entries.push(SafetensorEntry {
                name: format!("{}.weight", module),
                dtype: "U32",
                shape: weight_shape,
                data: unsigned_data,
            });
            entries.push(SafetensorEntry {
                name: format!("{}.scales", module),
                dtype: "F32",
                shape: scales_shape.clone(),
                data: scales_f32,
            });
            entries.push(SafetensorEntry {
                name: format!("{}.biases", module),
                dtype: "F32",
                shape: scales_shape,
                data: biases_f32,
            });
        } else {
            // Standard quantized tensor (e.g., self_attn.k_proj.weight)
            // Strip ".weight" suffix to get the module path for sibling naming
            let module = if let Some(m) = name.strip_suffix(".weight") {
                m.to_string()
            } else {
                (*name).clone()
            };
            entries.push(SafetensorEntry {
                name: format!("{}.weight", module),
                dtype: "U32",
                shape: weight_shape,
                data: unsigned_data,
            });
            entries.push(SafetensorEntry {
                name: format!("{}.scales", module),
                dtype: "F32",
                shape: scales_shape.clone(),
                data: scales_f32,
            });
            entries.push(SafetensorEntry {
                name: format!("{}.biases", module),
                dtype: "F32",
                shape: scales_shape,
                data: biases_f32,
            });
        }
    }

    // Sort entries by name for deterministic output
    entries.sort_by(|a, b| a.name.cmp(&b.name));

    // Collect output tensor names for the weight_map index
    output_tensor_names.extend(entries.iter().map(|e| e.name.clone()));

    // Build safetensors header and concatenate data
    let mut header_map: BTreeMap<String, Value> = BTreeMap::new();
    let mut current_offset = 0usize;

    for entry in &entries {
        let data_end = current_offset + entry.data.len();
        header_map.insert(
            entry.name.clone(),
            serde_json::json!({
                "dtype": entry.dtype,
                "shape": entry.shape,
                "data_offsets": [current_offset, data_end]
            }),
        );
        current_offset = data_end;
    }

    header_map.insert(
        "__metadata__".to_string(),
        serde_json::json!({"format": "mlx"}),
    );

    let header_json =
        serde_json::to_string(&header_map).map_err(BackendError::Serialization)?;
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;

    let mut file_data = Vec::with_capacity(8 + header_bytes.len() + current_offset);
    file_data.extend_from_slice(&header_size.to_le_bytes());
    file_data.extend_from_slice(header_bytes);
    for entry in &entries {
        file_data.extend_from_slice(&entry.data);
    }

    fs::write(path, &file_data).map_err(|e| BackendError::WriteFailed {
        reason: format!("Failed to write {}: {}", path.display(), e),
    })?;

    debug!(path = %path.display(), size = file_data.len(), "Wrote safetensors shard");

    Ok(file_data.len() as u64)
}

/// Convert signed quantized packed bytes to unsigned for MLX affine mode.
///
/// For 4-bit: each byte has two nibbles. XOR with 0x88 flips the MSB of
/// each nibble, converting signed [-8..7] to unsigned [0..15].
/// For 2-bit: XOR with 0xAA (flip MSB of each 2-bit field).
/// For 8-bit: XOR with 0x80 (flip sign bit).
fn convert_signed_to_unsigned(data: &[u8], bits: u8) -> Vec<u8> {
    let xor_mask: u8 = match bits {
        4 => 0x88,   // flip bit 3 of each nibble
        2 => 0xAA,   // flip bit 1 of each 2-bit field
        8 => 0x80,   // flip sign bit
        _ => 0x00,   // no conversion for unknown bit widths
    };
    data.iter().map(|&b| b ^ xor_mask).collect()
}

/// Compute MLX weight shape: last dimension packed into uint32.
/// For 4-bit with shape [64, 128]: result is [64, 16] (128 * 4 / 32 = 16).
fn mlx_weight_shape(original_shape: &[usize], bits: u8) -> Vec<usize> {
    if original_shape.is_empty() {
        return vec![];
    }
    let mut shape = original_shape.to_vec();
    let last = shape.last_mut().unwrap();
    *last = (*last * bits as usize) / 32;
    shape
}

/// Compute MLX scales/biases shape: last dimension divided by group_size.
/// For shape [64, 128] with group_size=64: result is [64, 2].
fn mlx_scales_shape(original_shape: &[usize], group_size: usize) -> Vec<usize> {
    if original_shape.is_empty() || group_size == 0 {
        return vec![];
    }
    let mut shape = original_shape.to_vec();
    let last = shape.last_mut().unwrap();
    *last = (*last).div_ceil(group_size);
    shape
}

/// Convert f16 byte array to f32 byte array (little-endian).
fn f16_bytes_to_f32_bytes(f16_data: &[u8]) -> Vec<u8> {
    let mut f32_data = Vec::with_capacity(f16_data.len() * 2);
    for chunk in f16_data.chunks_exact(2) {
        let f16_val = half::f16::from_le_bytes([chunk[0], chunk[1]]);
        let f32_val = f16_val.to_f32();
        f32_data.extend_from_slice(&f32_val.to_le_bytes());
    }
    f32_data
}

/// Create a byte buffer of `count` repeated f32 values (little-endian).
fn vec_f32_to_bytes(value: f32, count: usize) -> Vec<u8> {
    let bytes = value.to_le_bytes();
    let mut buf = Vec::with_capacity(count * 4);
    for _ in 0..count {
        buf.extend_from_slice(&bytes);
    }
    buf
}

/// Split a fused expert gate_up_proj tensor into separate gate_proj + up_proj
/// entries under the switch_glu namespace.
///
/// The fused tensor has shape (num_experts, hidden*2, in_features).
/// Split along axis=-2 to produce two (num_experts, hidden, in_features) tensors.
fn split_expert_gate_up(
    base: &str, // e.g., "model.language_model.layers.0.experts"
    weight_data: &[u8],
    weight_shape: &[usize],
    scales_data: &[u8],
    scales_shape: &[usize],
    biases_data: &[u8],
) -> Vec<SafetensorEntry> {
    let mut entries = Vec::with_capacity(6);

    // Split weight: shape is (num_experts, hidden*2, last_packed_dim)
    // Each half has hidden rows per expert.
    let (gate_w, up_w, gate_w_shape, up_w_shape) =
        split_tensor_axis_neg2(weight_data, weight_shape);

    // Split scales: shape is (num_experts, hidden*2, groups_per_row)
    let (gate_s, up_s, gate_s_shape, up_s_shape) =
        split_tensor_axis_neg2(scales_data, scales_shape);

    // Split biases: same shape as scales
    let (gate_b, up_b, gate_b_shape, up_b_shape) =
        split_tensor_axis_neg2(biases_data, scales_shape);

    let gate_module = format!("{}.switch_glu.gate_proj", base);
    let up_module = format!("{}.switch_glu.up_proj", base);

    entries.push(SafetensorEntry {
        name: format!("{}.weight", gate_module),
        dtype: "U32",
        shape: gate_w_shape,
        data: gate_w,
    });
    entries.push(SafetensorEntry {
        name: format!("{}.scales", gate_module),
        dtype: "F32",
        shape: gate_s_shape,
        data: gate_s,
    });
    entries.push(SafetensorEntry {
        name: format!("{}.biases", gate_module),
        dtype: "F32",
        shape: gate_b_shape,
        data: gate_b,
    });
    entries.push(SafetensorEntry {
        name: format!("{}.weight", up_module),
        dtype: "U32",
        shape: up_w_shape,
        data: up_w,
    });
    entries.push(SafetensorEntry {
        name: format!("{}.scales", up_module),
        dtype: "F32",
        shape: up_s_shape,
        data: up_s,
    });
    entries.push(SafetensorEntry {
        name: format!("{}.biases", up_module),
        dtype: "F32",
        shape: up_b_shape,
        data: up_b,
    });

    entries
}

/// Split a tensor's data in half along the second-to-last axis.
///
/// For shape [A, B, C], splits into two [A, B/2, C] tensors.
/// For shape [B, C], splits into two [B/2, C] tensors.
/// Data is assumed to be in row-major (C-contiguous) order.
fn split_tensor_axis_neg2(
    data: &[u8],
    shape: &[usize],
) -> (Vec<u8>, Vec<u8>, Vec<usize>, Vec<usize>) {
    if shape.len() < 2 {
        // Cannot split along axis -2 for 1D tensors; return halves
        let mid = data.len() / 2;
        let mut half_shape = shape.to_vec();
        if !half_shape.is_empty() {
            half_shape[0] /= 2;
        }
        return (
            data[..mid].to_vec(),
            data[mid..].to_vec(),
            half_shape.clone(),
            half_shape,
        );
    }

    let ndim = shape.len();
    let split_axis = ndim - 2;
    let split_dim = shape[split_axis];
    let half_dim = split_dim / 2;

    // Compute the stride of the split axis in bytes.
    // For the last axis (innermost), each element is `element_bytes` bytes.
    // But we don't know element_bytes from shape alone — compute from total data size.
    let total_elements: usize = shape.iter().product();
    if total_elements == 0 {
        let half_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(i, &d)| if i == split_axis { d / 2 } else { d })
            .collect();
        return (vec![], vec![], half_shape.clone(), half_shape);
    }
    let element_bytes = data.len() / total_elements;

    // Number of "outer" slabs (product of dims before split axis)
    let outer: usize = shape[..split_axis].iter().product();
    // Number of bytes in one row of the inner dimensions (split_dim * inner_stride)
    let inner_stride: usize = shape[split_axis + 1..].iter().product::<usize>() * element_bytes;
    let slab_bytes = split_dim * inner_stride;
    let half_slab_bytes = half_dim * inner_stride;

    let mut first_half = Vec::with_capacity(outer * half_slab_bytes);
    let mut second_half = Vec::with_capacity(outer * half_slab_bytes);

    for o in 0..outer {
        let slab_start = o * slab_bytes;
        first_half.extend_from_slice(&data[slab_start..slab_start + half_slab_bytes]);
        second_half
            .extend_from_slice(&data[slab_start + half_slab_bytes..slab_start + slab_bytes]);
    }

    let half_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .map(|(i, &d)| if i == split_axis { half_dim } else { d })
        .collect();

    (first_half, second_half, half_shape.clone(), half_shape)
}

/// Determine the safetensors dtype string for a quantized tensor.
fn output_dtype_string(tensor: &QuantizedTensor) -> &'static str {
    if tensor.quant_info.preserved {
        // Preserved tensors retain their dtype
        match tensor.original_dtype {
            crate::ir::DType::F32 => "F32",
            crate::ir::DType::F16 => "F16",
            crate::ir::DType::BF16 => "F16", // bf16 was converted to f16
            crate::ir::DType::I32 => "I32",
            crate::ir::DType::I64 => "I64",
            crate::ir::DType::U8 => "U8",
            crate::ir::DType::U16 => "U16",
            crate::ir::DType::U32 => "U32",
            crate::ir::DType::Bool => "BOOL",
        }
    } else if tensor.quant_info.bits == 16 {
        "F16"
    } else {
        // Quantized weight data — store as raw bytes (U8)
        // MLX will interpret these using the quantization config
        "U8"
    }
}

/// Write MLX-compatible config.json with quantization metadata.
///
/// Injects `quantization` key that mlx-lm uses to know how to load quantized weights:
/// `{"group_size": N, "bits": N}` — triggers nn.quantize() during model loading.
fn write_mlx_config(model: &QuantizedModel, output_dir: &Path) -> Result<u64, BackendError> {
    let mut config = model.metadata.raw_config.clone();

    // Build quantization config with per-layer overrides for mixed-bit models.
    // mlx-lm's load_model() uses this to call nn.quantize() with the right
    // bit width per layer. Global defaults apply to all layers; per-layer
    // entries override specific modules.
    let mut quant_config = serde_json::json!({
        "group_size": model.group_size,
        "bits": model.bits
    });

    // Add per-layer overrides for tensors whose bit width differs from the global default.
    // Convert tensor names to the module paths mlx-lm expects after sanitization:
    //   "model.language_model.layers.0.self_attn.k_proj.weight"
    //     → "language_model.model.layers.0.self_attn.k_proj"
    if let Some(quant_obj) = quant_config.as_object_mut() {
        for (name, tensor) in &model.tensors {
            if tensor.quant_info.preserved || tensor.quant_info.scales.is_none() {
                continue;
            }
            if tensor.quant_info.bits == model.bits
                && tensor.quant_info.group_size == model.group_size
            {
                continue; // matches global default, no override needed
            }
            // Convert to mlx-lm module path: strip "model." prefix, add extra
            // "model." after "language_model", strip ".weight" suffix, handle
            // expert renaming.
            let module_path = tensor_name_to_mlx_module_path(name);
            quant_obj.insert(
                module_path,
                serde_json::json!({
                    "group_size": tensor.quant_info.group_size,
                    "bits": tensor.quant_info.bits
                }),
            );
        }
    }

    if let Some(obj) = config.as_object_mut() {
        obj.insert("quantization".to_string(), quant_config);
    }

    let config_json = serde_json::to_string_pretty(&config)
        .map_err(BackendError::Serialization)?;

    let path = output_dir.join("config.json");
    fs::write(&path, &config_json).map_err(|e| BackendError::WriteFailed {
        reason: format!("Failed to write config.json: {}", e),
    })?;

    Ok(config_json.len() as u64)
}

/// Convert an hf2q tensor name to the module path mlx-lm expects after sanitization.
///
/// Examples:
///   "model.language_model.layers.0.self_attn.k_proj.weight"
///     → "language_model.model.layers.0.self_attn.k_proj"
///   "model.language_model.layers.0.mlp.gate_proj.weight"
///     → "language_model.model.layers.0.mlp.gate_proj"
///   "model.language_model.layers.0.experts.gate_up_proj"
///     → "language_model.model.layers.0.experts.switch_glu.gate_proj"
///     (Note: split tensors get two entries; we return the gate_proj path)
fn tensor_name_to_mlx_module_path(name: &str) -> String {
    let mut path = name.to_string();

    // Strip "model." prefix
    if let Some(stripped) = path.strip_prefix("model.") {
        path = stripped.to_string();
    }

    // Insert extra "model." after "language_model."
    // "language_model.layers.0..." → "language_model.model.layers.0..."
    if path.starts_with("language_model.") && !path.starts_with("language_model.model.") {
        path = path.replacen("language_model.", "language_model.model.", 1);
    }

    // Strip ".weight" suffix to get module path
    if let Some(stripped) = path.strip_suffix(".weight") {
        path = stripped.to_string();
    }

    // Handle expert renaming
    if path.ends_with(".experts.gate_up_proj") {
        path = path.replace(".experts.gate_up_proj", ".experts.switch_glu.gate_proj");
    } else if path.ends_with(".experts.down_proj") {
        path = path.replace(".experts.down_proj", ".experts.switch_glu.down_proj");
    }

    path
}

/// Copy tokenizer files from source to output directory.
fn copy_tokenizer_files(
    input_dir: &Path,
    output_dir: &Path,
) -> Result<Vec<OutputFile>, BackendError> {
    let mut files = Vec::new();

    let tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
    ];

    for filename in &tokenizer_files {
        let src = input_dir.join(filename);
        if src.exists() {
            let dst = output_dir.join(filename);
            fs::copy(&src, &dst).map_err(|e| BackendError::WriteFailed {
                reason: format!("Failed to copy {}: {}", filename, e),
            })?;

            let size = fs::metadata(&dst)
                .map(|m| m.len())
                .unwrap_or(0);

            files.push(OutputFile {
                filename: filename.to_string(),
                size_bytes: size,
            });

            debug!(file = %filename, "Copied tokenizer file");
        } else {
            if *filename == "tokenizer.json" || *filename == "tokenizer_config.json" {
                warn!(file = %filename, "Tokenizer file not found in source directory");
            }
        }
    }

    Ok(files)
}

/// Write quantization_config.json with per-tensor metadata.
fn write_quant_config(model: &QuantizedModel, output_dir: &Path) -> Result<u64, BackendError> {
    let mut per_tensor: HashMap<String, Value> = HashMap::new();

    for (name, tensor) in &model.tensors {
        per_tensor.insert(
            name.clone(),
            serde_json::json!({
                "method": tensor.quant_info.method,
                "bits": tensor.quant_info.bits,
                "group_size": tensor.quant_info.group_size,
                "preserved": tensor.quant_info.preserved,
            }),
        );
    }

    let quant_config = serde_json::json!({
        "quant_method": model.quant_method,
        "bits": model.bits,
        "group_size": model.group_size,
        "tensor_count": model.tensors.len(),
        "per_tensor": per_tensor,
    });

    let json = serde_json::to_string_pretty(&quant_config)
        .map_err(BackendError::Serialization)?;

    let path = output_dir.join("quantization_config.json");
    fs::write(&path, &json).map_err(|e| BackendError::WriteFailed {
        reason: format!("Failed to write quantization_config.json: {}", e),
    })?;

    Ok(json.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, ModelMetadata, QuantizedTensor, TensorQuantInfo};

    fn make_test_model() -> QuantizedModel {
        let mut tensors = HashMap::new();

        tensors.insert(
            "layer.weight".to_string(),
            QuantizedTensor {
                name: "layer.weight".to_string(),
                shape: vec![2, 2],
                original_dtype: DType::F16,
                data: vec![0u8; 4],
                quant_info: TensorQuantInfo {
                    method: "q4".to_string(),
                    bits: 4,
                    group_size: 64,
                    preserved: false,
                    scales: Some(vec![0u8; 2]),
                    biases: None,
                },
            },
        );

        tensors.insert(
            "layer.norm.weight".to_string(),
            QuantizedTensor {
                name: "layer.norm.weight".to_string(),
                shape: vec![2],
                original_dtype: DType::F16,
                data: vec![0u8; 4],
                quant_info: TensorQuantInfo {
                    method: "passthrough".to_string(),
                    bits: 16,
                    group_size: 0,
                    preserved: true,
                    scales: None,
                    biases: None,
                },
            },
        );

        QuantizedModel {
            metadata: ModelMetadata {
                architecture: "TestModel".to_string(),
                model_type: "test".to_string(),
                param_count: 100,
                hidden_size: 4,
                num_layers: 1,
                layer_types: vec!["attention".to_string()],
                num_attention_heads: 2,
                num_kv_heads: None,
                vocab_size: 100,
                dtype: "float16".to_string(),
                shard_count: 1,
                num_experts: None,
                top_k_experts: None,
                intermediate_size: None,
                raw_config: serde_json::json!({"model_type": "test"}),
            },
            tensors,
            quant_method: "q4".to_string(),
            group_size: 64,
            bits: 4,
        }
    }

    #[test]
    fn test_validate_non_empty_model() {
        let backend = MlxBackend::new();
        let model = make_test_model();
        let warnings = backend.validate(&model).unwrap();
        // No warnings expected for valid model
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_validate_empty_model_fails() {
        let backend = MlxBackend::new();
        let model = QuantizedModel {
            metadata: ModelMetadata {
                architecture: "Test".to_string(),
                model_type: "test".to_string(),
                param_count: 0,
                hidden_size: 0,
                num_layers: 0,
                layer_types: vec![],
                num_attention_heads: 0,
                num_kv_heads: None,
                vocab_size: 0,
                dtype: "float16".to_string(),
                shard_count: 0,
                num_experts: None,
                top_k_experts: None,
                intermediate_size: None,
                raw_config: serde_json::Value::Null,
            },
            tensors: HashMap::new(),
            quant_method: "q4".to_string(),
            group_size: 64,
            bits: 4,
        };
        assert!(backend.validate(&model).is_err());
    }

    #[test]
    fn test_write_mlx_output() {
        let tmp = tempfile::tempdir().unwrap();
        let input_dir = tmp.path().join("input");
        let output_dir = tmp.path().join("output");
        fs::create_dir_all(&input_dir).unwrap();

        // Create a minimal tokenizer file
        fs::write(input_dir.join("tokenizer.json"), "{}").unwrap();
        fs::write(input_dir.join("tokenizer_config.json"), "{}").unwrap();

        let backend = MlxBackend::new();
        let model = make_test_model();
        let progress = ProgressReporter::new();

        let manifest = backend.write(&model, &input_dir, &output_dir, &progress).unwrap();

        assert!(output_dir.join("config.json").exists());
        assert!(output_dir.join("quantization_config.json").exists());
        assert!(output_dir.join("tokenizer.json").exists());
        assert!(manifest.shard_count > 0);
        assert!(manifest.total_size_bytes > 0);
    }

    #[test]
    fn test_output_dtype_preserved() {
        let tensor = QuantizedTensor {
            name: "test".to_string(),
            shape: vec![4],
            original_dtype: DType::F16,
            data: vec![0u8; 8],
            quant_info: TensorQuantInfo {
                method: "passthrough".to_string(),
                bits: 16,
                group_size: 0,
                preserved: true,
                scales: None,
                biases: None,
            },
        };
        assert_eq!(output_dtype_string(&tensor), "F16");
    }

    #[test]
    fn test_convert_signed_to_unsigned_4bit() {
        // 4-bit: XOR with 0x88
        // Byte 0x37 = lo=7(signed +7), hi=3(signed +3)
        // After: lo=15(unsigned), hi=11(unsigned)
        let data = vec![0x37u8];
        let result = convert_signed_to_unsigned(&data, 4);
        assert_eq!(result, vec![0x37 ^ 0x88]);
        assert_eq!(result[0] & 0x0F, 15); // +7 → 15
        assert_eq!((result[0] >> 4) & 0x0F, 11); // +3 → 11

        // Zero nibbles: 0x00 → 0x88 (unsigned 8 = zero point)
        let zero = convert_signed_to_unsigned(&[0x00], 4);
        assert_eq!(zero[0] & 0x0F, 8);
        assert_eq!((zero[0] >> 4) & 0x0F, 8);
    }

    #[test]
    fn test_mlx_weight_shape() {
        // 4-bit: last dim * 4 / 32 = last_dim / 8
        assert_eq!(mlx_weight_shape(&[64, 128], 4), vec![64, 16]);
        assert_eq!(mlx_weight_shape(&[8, 64, 128], 4), vec![8, 64, 16]);
        // 2-bit: last dim * 2 / 32 = last_dim / 16
        assert_eq!(mlx_weight_shape(&[64, 128], 2), vec![64, 8]);
    }

    #[test]
    fn test_mlx_scales_shape() {
        assert_eq!(mlx_scales_shape(&[64, 128], 64), vec![64, 2]);
        assert_eq!(mlx_scales_shape(&[8, 64, 128], 64), vec![8, 64, 2]);
        assert_eq!(mlx_scales_shape(&[64, 128], 32), vec![64, 4]);
    }

    #[test]
    fn test_f16_to_f32_conversion() {
        // f16 value 1.0 = 0x3C00
        let f16_data = vec![0x00u8, 0x3C];
        let f32_data = f16_bytes_to_f32_bytes(&f16_data);
        assert_eq!(f32_data.len(), 4);
        let f32_val = f32::from_le_bytes([f32_data[0], f32_data[1], f32_data[2], f32_data[3]]);
        assert!((f32_val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_split_tensor_axis_neg2() {
        // 2D tensor [4, 2]: split along axis 0 → two [2, 2]
        let data: Vec<u8> = (0..8).collect();
        let (first, second, shape1, shape2) =
            split_tensor_axis_neg2(&data, &[4, 2]);
        assert_eq!(shape1, vec![2, 2]);
        assert_eq!(shape2, vec![2, 2]);
        assert_eq!(first, vec![0, 1, 2, 3]);
        assert_eq!(second, vec![4, 5, 6, 7]);

        // 3D tensor [2, 4, 2]: split axis=-2 (axis 1) → two [2, 2, 2]
        let data3d: Vec<u8> = (0..16).collect();
        let (first3, second3, s1, s2) =
            split_tensor_axis_neg2(&data3d, &[2, 4, 2]);
        assert_eq!(s1, vec![2, 2, 2]);
        assert_eq!(s2, vec![2, 2, 2]);
        // Expert 0: rows 0-3 → first half [0,1,2,3], second half [4,5,6,7]
        // Expert 1: rows 0-3 → first half [8,9,10,11], second half [12,13,14,15]
        assert_eq!(first3, vec![0, 1, 2, 3, 8, 9, 10, 11]);
        assert_eq!(second3, vec![4, 5, 6, 7, 12, 13, 14, 15]);
    }

    #[test]
    fn test_write_quantized_tensor_naming() {
        // Verify that a quantized tensor produces correct sibling names
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.safetensors");

        let tensor = QuantizedTensor {
            name: "model.layer.weight".to_string(),
            shape: vec![4, 128], // 4 output, 128 input
            original_dtype: DType::F16,
            data: vec![0u8; 4 * 128 / 2], // 4-bit: 2 values per byte
            quant_info: TensorQuantInfo {
                method: "q4".to_string(),
                bits: 4,
                group_size: 64,
                preserved: false,
                scales: Some(vec![0u8; 4 * 2 * 2]), // 4 rows * 2 groups * 2 bytes f16
                biases: None,
            },
        };

        let tensors: Vec<(&String, &QuantizedTensor)> = vec![(&tensor.name, &tensor)];
        let mut names = Vec::new();
        write_single_safetensors_file(&tensors, &path, &mut names).unwrap();

        // Should produce weight + scales + biases with proper sibling naming
        assert!(names.contains(&"model.layer.biases".to_string()));
        assert!(names.contains(&"model.layer.scales".to_string()));
        assert!(names.contains(&"model.layer.weight".to_string()));
        // Should NOT have "model.layer.weight.scales"
        assert!(!names.iter().any(|n| n.contains("weight.scales")));
    }

    #[test]
    fn test_config_json_includes_quantization() {
        let tmp = tempfile::tempdir().unwrap();
        let model = make_test_model();
        write_mlx_config(&model, tmp.path()).unwrap();

        let config_str = fs::read_to_string(tmp.path().join("config.json")).unwrap();
        let config: Value = serde_json::from_str(&config_str).unwrap();

        assert!(config.get("quantization").is_some());
        assert_eq!(config["quantization"]["bits"], 4);
        assert_eq!(config["quantization"]["group_size"], 64);
    }
}
