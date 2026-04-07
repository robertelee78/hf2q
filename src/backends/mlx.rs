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
    DType as IrDType, FormatWarning, ModelMetadata, OutputFile, OutputManifest, QuantizedModel,
    QuantizedTensor, TensorMap, TensorRef, WarningSeverity,
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

    fn requires_native_quantization(&self) -> bool {
        true
    }

    fn quantize_and_write(
        &self,
        tensor_map: &TensorMap,
        metadata: &ModelMetadata,
        bits: u8,
        group_size: usize,
        bit_overrides: Option<&HashMap<String, u8>>,
        input_dir: &Path,
        output_dir: &Path,
        progress: &ProgressReporter,
    ) -> Result<OutputManifest, BackendError> {
        fs::create_dir_all(output_dir).map_err(|e| BackendError::WriteFailed {
            reason: format!("Failed to create output directory: {}", e),
        })?;

        let mut files = Vec::new();
        let empty_overrides = HashMap::new();
        let overrides = bit_overrides.unwrap_or(&empty_overrides);

        // 1. Quantize f16 tensors using MLX affine algorithm and write shards
        let shard_files = write_natively_quantized_shards(
            tensor_map, metadata, bits, group_size, overrides, output_dir, progress,
        )?;
        files.extend(shard_files);

        // 2. Write config.json with quantization metadata + per-layer overrides
        let config_size = write_mlx_config_native(
            metadata, bits, group_size, overrides, output_dir,
        )?;
        files.push(OutputFile {
            filename: "config.json".to_string(),
            size_bytes: config_size,
        });

        // 3. Copy tokenizer files
        let tokenizer_files = copy_tokenizer_files(input_dir, output_dir)?;
        files.extend(tokenizer_files);

        // 4. Write quantization_config.json (metadata for downstream tools)
        let qc = serde_json::json!({
            "quant_method": "mlx_affine",
            "bits": bits,
            "group_size": group_size,
        });
        let qc_json = serde_json::to_string_pretty(&qc).map_err(BackendError::Serialization)?;
        let qc_path = output_dir.join("quantization_config.json");
        fs::write(&qc_path, &qc_json).map_err(|e| BackendError::WriteFailed {
            reason: format!("Failed to write quantization_config.json: {}", e),
        })?;
        files.push(OutputFile {
            filename: "quantization_config.json".to_string(),
            size_bytes: qc_json.len() as u64,
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
            "MLX native quantization output written"
        );

        Ok(OutputManifest {
            output_dir: output_dir.display().to_string(),
            files,
            total_size_bytes: total_size,
            shard_count,
        })
    }
}

/// Quantize f16 tensors using MLX affine algorithm and write as safetensors shards.
///
/// This is the native quantization path: original f16 weights go directly through
/// MLX's affine quantize algorithm (pure Rust port), producing bit-exact compatible
/// output for MLX's `quantized_matmul` Metal kernel.
///
/// Weight tensors (2D+ with "weight" in name, not norms/scalars) are quantized.
/// Everything else is preserved at original precision.
fn write_natively_quantized_shards(
    tensor_map: &TensorMap,
    _metadata: &ModelMetadata,
    bits: u8,
    group_size: usize,
    bit_overrides: &HashMap<String, u8>,
    output_dir: &Path,
    progress: &ProgressReporter,
) -> Result<Vec<OutputFile>, BackendError> {
    let num_shards = DEFAULT_OUTPUT_SHARDS;

    // Sort tensors by name for deterministic output
    let mut sorted_tensors: Vec<(&String, &TensorRef)> = tensor_map.tensors.iter().collect();
    sorted_tensors.sort_by_key(|(name, _)| name.as_str());

    // Distribute tensors across shards roughly evenly by data size
    let total_size: usize = sorted_tensors.iter().map(|(_, t)| t.data.len()).sum();
    let target_shard_size = (total_size / num_shards).max(1);

    let mut shards: Vec<Vec<(&String, &TensorRef)>> = vec![Vec::new(); num_shards];
    let mut shard_sizes = vec![0usize; num_shards];
    let mut current_shard = 0;

    for tensor_pair in &sorted_tensors {
        shards[current_shard].push(*tensor_pair);
        shard_sizes[current_shard] += tensor_pair.1.data.len();

        if shard_sizes[current_shard] >= target_shard_size && current_shard < num_shards - 1 {
            current_shard += 1;
        }
    }

    let pb = progress.bar(num_shards as u64, "Writing MLX shards (native quant)");
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
            format!("model-{:05}-of-{:05}.safetensors", shard_idx + 1, num_shards)
        };

        let shard_path = output_dir.join(&shard_filename);

        let mut shard_tensor_names: Vec<String> = Vec::new();
        let shard_size = write_native_safetensors_file(
            shard_tensors,
            bits,
            group_size,
            bit_overrides,
            &shard_path,
            &mut shard_tensor_names,
        )?;

        for tensor_name in &shard_tensor_names {
            weight_map.insert(tensor_name.clone(), shard_filename.clone());
        }

        files.push(OutputFile {
            filename: shard_filename,
            size_bytes: shard_size,
        });

        pb.inc(1);
    }

    // Write index.json
    if num_shards > 1 {
        let index = serde_json::json!({
            "metadata": { "total_size": total_size },
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

    pb.finish_with_message("MLX shards written (native quant)");
    Ok(files)
}

/// Write a single safetensors file, quantizing weight tensors with MLX affine algorithm.
///
/// For each tensor:
/// - If it's a weight tensor (2D+, not a norm/scalar): convert to f32, quantize, write
///   as (uint32 weight + bf16 scales + bf16 biases) with proper naming
/// - If it's a non-weight tensor: write as-is at original precision
fn write_native_safetensors_file(
    tensors: &[(&String, &TensorRef)],
    bits: u8,
    group_size: usize,
    bit_overrides: &HashMap<String, u8>,
    path: &Path,
    output_tensor_names: &mut Vec<String>,
) -> Result<u64, BackendError> {
    let mut entries: Vec<SafetensorEntry> = Vec::new();

    for (name, tensor) in tensors {
        // Determine if this tensor should be quantized.
        // Quantize all 2D+ weight tensors: Linear projections, Embeddings, expert weights.
        // Preserve: layer norms (1D), scalars (0D/1D), position embeddings, std_bias/scale.
        // This matches mlx-lm's nn.quantize which transforms both nn.Linear and nn.Embedding.
        let is_2d_plus = tensor.shape.len() >= 2;
        let is_norm_or_scalar = name.contains("layernorm")
            || name.contains("_norm.")
            || name.contains("layer_scalar")
            || name.contains("router.scale")
            || name.contains("per_expert_scale")
            || name.contains("std_bias")
            || name.contains("std_scale")
            || name.contains("position_embedding");
        // Skip vision tower entirely — text-only inference doesn't use it,
        // and quantizing it creates extra keys that may confuse the loader.
        let is_vision = name.contains("vision_tower") || name.contains("embed_vision");
        if is_vision {
            continue;
        }
        let should_quantize = is_2d_plus && !is_norm_or_scalar;

        if should_quantize {
            // Look up per-tensor bit override (e.g., 8-bit for MLP/router in MoE)
            let tensor_bits = bit_overrides.get(*name).copied().unwrap_or(bits);

            // Convert to f32 for quantization
            let f32_values = tensor_ref_to_f32(tensor);

            // Run MLX affine quantization at the right bit width
            let (mlx_packed, mlx_scales, mlx_biases) =
                mlx_affine_quantize(&f32_values, tensor_bits as u32, group_size);

            let weight_shape = mlx_weight_shape(&tensor.shape, tensor_bits);
            let scales_shape = mlx_scales_shape(&tensor.shape, group_size);

            // Handle expert naming
            // MLX stores ALL bit widths as uint32 in safetensors
            let weight_dtype: &'static str = "U32";
            let make_entries = |module: &str,
                                w_data: Vec<u8>, w_shape: Vec<usize>,
                                s_data: Vec<u8>, s_shape: Vec<usize>,
                                b_data: Vec<u8>| -> Vec<SafetensorEntry> {
                vec![
                    SafetensorEntry { name: format!("{}.weight", module), dtype: weight_dtype, shape: w_shape, data: w_data },
                    SafetensorEntry { name: format!("{}.scales", module), dtype: "BF16", shape: s_shape.clone(), data: s_data },
                    SafetensorEntry { name: format!("{}.biases", module), dtype: "BF16", shape: s_shape, data: b_data },
                ]
            };

            if name.ends_with(".experts.gate_up_proj") {
                let base = name.strip_suffix(".gate_up_proj").unwrap();
                let (gate_w, up_w, half_w_shape, _) = split_tensor_axis_neg2(&mlx_packed, &weight_shape);
                let (gate_s, up_s, half_s_shape, _) = split_tensor_axis_neg2(&mlx_scales, &scales_shape);
                let (gate_b, up_b, _, _) = split_tensor_axis_neg2(&mlx_biases, &scales_shape);

                entries.extend(make_entries(
                    &format!("{}.switch_glu.gate_proj", base),
                    gate_w, half_w_shape.clone(), gate_s, half_s_shape.clone(), gate_b,
                ));
                entries.extend(make_entries(
                    &format!("{}.switch_glu.up_proj", base),
                    up_w, half_w_shape, up_s, half_s_shape, up_b,
                ));
            } else if name.ends_with(".experts.down_proj") {
                let base = name.strip_suffix(".down_proj").unwrap();
                entries.extend(make_entries(
                    &format!("{}.switch_glu.down_proj", base),
                    mlx_packed, weight_shape, mlx_scales, scales_shape, mlx_biases,
                ));
            } else {
                let module = name.strip_suffix(".weight").unwrap_or(name);
                entries.extend(make_entries(
                    module, mlx_packed, weight_shape, mlx_scales, scales_shape, mlx_biases,
                ));
            }
        } else {
            // Non-weight tensor: preserve as-is
            let dtype_str = match tensor.dtype {
                IrDType::F32 => "F32",
                IrDType::F16 => "F16",
                IrDType::BF16 => "BF16", // preserved as bf16 for MLX native path
                IrDType::I32 => "I32",
                IrDType::I64 => "I64",
                IrDType::U8 => "U8",
                IrDType::U16 => "U16",
                IrDType::U32 => "U32",
                IrDType::Bool => "BOOL",
            };
            entries.push(SafetensorEntry {
                name: (*name).clone(),
                dtype: dtype_str,
                shape: tensor.shape.clone(),
                data: tensor.data.clone(),
            });
        }
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    output_tensor_names.extend(entries.iter().map(|e| e.name.clone()));

    // Build safetensors file
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

    header_map.insert("__metadata__".to_string(), serde_json::json!({"format": "mlx"}));

    let header_json = serde_json::to_string(&header_map).map_err(BackendError::Serialization)?;
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

    debug!(path = %path.display(), size = file_data.len(), "Wrote safetensors shard (native quant)");
    Ok(file_data.len() as u64)
}

/// Convert a TensorRef to f32 values for quantization.
fn tensor_ref_to_f32(tensor: &TensorRef) -> Vec<f32> {
    match tensor.dtype {
        IrDType::F32 => tensor.data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        IrDType::F16 => tensor.data.chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        IrDType::BF16 => tensor.data.chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        _ => vec![0.0; tensor.shape.iter().product()],
    }
}

/// Write MLX config.json for native quantization path with per-layer overrides.
fn write_mlx_config_native(
    metadata: &ModelMetadata,
    bits: u8,
    group_size: usize,
    bit_overrides: &HashMap<String, u8>,
    output_dir: &Path,
) -> Result<u64, BackendError> {
    let mut config = metadata.raw_config.clone();

    // Build quantization config with per-layer overrides
    let mut quant_config = serde_json::json!({
        "group_size": group_size,
        "bits": bits,
        "mode": "affine"
    });

    if let Some(quant_obj) = quant_config.as_object_mut() {
        for (tensor_name, &override_bits) in bit_overrides {
            // Convert tensor name to mlx-lm module path
            let paths = tensor_name_to_mlx_module_paths(tensor_name);
            for path in paths {
                quant_obj.insert(
                    path,
                    serde_json::json!({
                        "group_size": group_size,
                        "bits": override_bits
                    }),
                );
            }
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

        // --- Quantized tensor: re-quantize using MLX's affine algorithm ---
        //
        // hf2q uses symmetric quantization internally. MLX uses affine quantization
        // with a different packing and scale/bias convention. Rather than trying to
        // convert between formats, we dequantize back to f32 and then re-quantize
        // using MLX's exact algorithm (ported to pure Rust from mlx/backend/cpu/quantized.cpp).
        let bits = tensor.quant_info.bits;
        let group_size = tensor.quant_info.group_size;
        let scales_raw = tensor.quant_info.scales.as_ref().unwrap();

        // Step 1: Dequantize our packed data back to f32 using our symmetric scales
        let total_elements: usize = tensor.shape.iter().product();
        let int_values = unpack_symmetric(&tensor.data, bits, total_elements);
        let our_scales: Vec<f32> = scales_raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        let effective_gs = if total_elements < group_size { total_elements } else { group_size };
        let mut f32_weights = Vec::with_capacity(total_elements);
        for (i, &ival) in int_values.iter().enumerate() {
            let g = i / effective_gs;
            let scale = our_scales.get(g).copied().unwrap_or(0.0);
            f32_weights.push(scale * (ival as f32));
        }

        // Step 2: Re-quantize using MLX's affine algorithm
        let (mlx_packed, mlx_scales, mlx_biases) =
            mlx_affine_quantize(&f32_weights, bits as u32, group_size);

        // Step 3: Compute shapes for the MLX format
        let weight_shape = mlx_weight_shape(&tensor.shape, bits);
        let scales_shape = mlx_scales_shape(&tensor.shape, group_size);

        // Step 4: Emit entries with correct naming
        let make_entries = |module: &str,
                            w_data: Vec<u8>,
                            w_shape: Vec<usize>,
                            s_data: Vec<u8>,
                            s_shape: Vec<usize>,
                            b_data: Vec<u8>|
         -> Vec<SafetensorEntry> {
            vec![
                SafetensorEntry {
                    name: format!("{}.weight", module),
                    dtype: "U32",
                    shape: w_shape,
                    data: w_data,
                },
                SafetensorEntry {
                    name: format!("{}.scales", module),
                    dtype: "BF16",
                    shape: s_shape.clone(),
                    data: s_data,
                },
                SafetensorEntry {
                    name: format!("{}.biases", module),
                    dtype: "BF16",
                    shape: s_shape,
                    data: b_data,
                },
            ]
        };

        if name.ends_with(".experts.gate_up_proj") {
            // Split fused gate+up into separate tensors along axis=-2
            let base = name.strip_suffix(".gate_up_proj").unwrap();

            let (gate_w, up_w, half_w_shape, _) =
                split_tensor_axis_neg2(&mlx_packed, &weight_shape);
            let (gate_s, up_s, half_s_shape, _) =
                split_tensor_axis_neg2(&mlx_scales, &scales_shape);
            let (gate_b, up_b, _, _) =
                split_tensor_axis_neg2(&mlx_biases, &scales_shape);

            let gate_mod = format!("{}.switch_glu.gate_proj", base);
            let up_mod = format!("{}.switch_glu.up_proj", base);

            entries.extend(make_entries(
                &gate_mod, gate_w, half_w_shape.clone(), gate_s, half_s_shape.clone(), gate_b,
            ));
            entries.extend(make_entries(
                &up_mod, up_w, half_w_shape, up_s, half_s_shape, up_b,
            ));
        } else if name.ends_with(".experts.down_proj") {
            let base = name.strip_suffix(".down_proj").unwrap();
            let module = format!("{}.switch_glu.down_proj", base);
            entries.extend(make_entries(
                &module, mlx_packed, weight_shape, mlx_scales, scales_shape, mlx_biases,
            ));
        } else {
            let module = if let Some(m) = name.strip_suffix(".weight") {
                m.to_string()
            } else {
                (*name).clone()
            };
            entries.extend(make_entries(
                &module, mlx_packed, weight_shape, mlx_scales, scales_shape, mlx_biases,
            ));
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

/// Unpack hf2q's symmetric-quantized packed bytes into signed integers.
/// Inverse of `pack_quantized` in static_quant.rs.
fn unpack_symmetric(data: &[u8], bits: u8, total_elements: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(total_elements);
    match bits {
        4 => {
            for &byte in data {
                let lo = (byte & 0x0F) as i8;
                let lo = if lo & 0x08 != 0 { lo | !0x0F } else { lo };
                values.push(lo);
                let hi = ((byte >> 4) & 0x0F) as i8;
                let hi = if hi & 0x08 != 0 { hi | !0x0F } else { hi };
                values.push(hi);
            }
        }
        2 => {
            for &byte in data {
                for shift in (0..8).step_by(2) {
                    let val = ((byte >> shift) & 0x03) as i8;
                    let val = if val & 0x02 != 0 { val | !0x03 } else { val };
                    values.push(val);
                }
            }
        }
        8 => {
            values.extend(data.iter().map(|&b| b as i8));
        }
        _ => {
            values.extend(data.iter().map(|&b| b as i8));
        }
    }
    values.truncate(total_elements);
    values
}

/// Pure Rust implementation of MLX's affine quantization algorithm.
///
/// Ported from `mlx/backend/cpu/quantized.cpp::quantize()`.
/// For each group of `group_size` f32 values:
///   1. Find min/max
///   2. Compute scale = (max - min) / n_bins, with sign flip for numerical stability
///   3. Compute bias from the edge value
///   4. Quantize: uint_val = clamp(round((w - bias) / scale), 0, n_bins)
///   5. Pack values into output bytes
///
/// Packing format depends on bit width:
/// - Power-of-2 bits (2, 4, 8): pack into uint32, `el_per_int = 32/bits`
/// - Non-power-of-2 bits (3, 6): pack into 3-byte (24-bit) groups:
///   - 3-bit: 8 values per 3 bytes (8*3 = 24 bits)
///   - 6-bit: 4 values per 3 bytes (4*6 = 24 bits)
///   Stored as uint8 arrays (not uint32).
///
/// Dequantization formula: `w ≈ uint_val * scale + bias`
///
/// Returns (packed_bytes, scales_bf16_bytes, biases_bf16_bytes).
fn mlx_affine_quantize(
    weights: &[f32],
    bits: u32,
    group_size: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let n_bins = ((1u32 << bits) - 1) as f32;
    let eps = 1e-7f32;
    let total_elements = weights.len();
    let num_groups = total_elements.div_ceil(group_size);

    // All bit widths pack into uint32 words.
    // For N-bit: floor(32/N) values per uint32, and (group_size * N / 32) uint32s per group.
    let el_per_u32 = 32 / bits as usize; // 2-bit:16, 3-bit:10, 4-bit:8, 6-bit:5, 8-bit:4
    let u32s_per_group = (group_size * bits as usize) / 32;

    let mut packed = Vec::with_capacity(num_groups * u32s_per_group * 4);
    let mut scales_bf16 = Vec::with_capacity(num_groups * 2);
    let mut biases_bf16 = Vec::with_capacity(num_groups * 2);

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(total_elements);
        let group = &weights[start..end];

        // Find min/max
        let mut w_min = f32::INFINITY;
        let mut w_max = f32::NEG_INFINITY;
        for &v in group {
            w_min = w_min.min(v);
            w_max = w_max.max(v);
        }

        // Compute scale with sign flip for stability (matches MLX exactly)
        let mask = w_min.abs() > w_max.abs();
        let mut scale = ((w_max - w_min) / n_bins).max(eps);
        if !mask {
            scale = -scale;
        }

        let edge = if mask { w_min } else { w_max };
        let q0 = (edge / scale).round();
        let mut bias = 0.0f32;
        if q0 != 0.0 {
            scale = edge / q0;
            bias = edge;
        }

        // Store scale and bias as bf16
        let scale_bf16 = half::bf16::from_f32(scale);
        let bias_bf16 = half::bf16::from_f32(bias);
        scales_bf16.extend_from_slice(&scale_bf16.to_le_bytes());
        biases_bf16.extend_from_slice(&bias_bf16.to_le_bytes());

        // Quantize values and pack into uint32 words
        let actual_len = end - start;
        let mut val_idx = 0usize;
        for _j in 0..u32s_per_group {
            let mut out_el: u32 = 0;
            for k in 0..el_per_u32 {
                let w_el = if val_idx < actual_len { group[val_idx] } else { 0.0 };
                let q = ((w_el - bias) / scale).round().clamp(0.0, n_bins);
                out_el |= (q as u32) << (k as u32 * bits);
                val_idx += 1;
            }
            packed.extend_from_slice(&out_el.to_le_bytes());
        }
    }

    (packed, scales_bf16, biases_bf16)
}

/// Whether the given bit width uses uint8 triplet packing (3, 6) vs uint32 packing (2, 4, 8).
fn mlx_uses_u8_packing(bits: u8) -> bool {
    !((bits as u32).is_power_of_two())
}

/// Compute MLX weight shape after packing.
///
/// ALL bit widths are stored as uint32 in safetensors.
/// Shape: last_dim * bits / 32 (uint32 count).
///   Example: 4-bit [64, 128] -> [64, 16] (128 * 4 / 32 = 16)
///   Example: 6-bit [64, 128] -> [64, 24] (128 * 6 / 32 = 24)
///   Example: 3-bit [64, 128] -> [64, 12] (128 * 3 / 32 = 12)
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

/// Split a fused expert gate_up_proj tensor into separate gate_proj + up_proj
/// entries under the switch_glu namespace.
///
/// NOTE: This function is used by the legacy `write()` path (non-native backends).
/// The native quantization path uses inline splitting in `write_native_safetensors_file`.
#[allow(dead_code)]
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
            // Convert to mlx-lm module path(s). Expert gate_up_proj splits
            // into both gate_proj and up_proj — emit overrides for both.
            let module_paths = tensor_name_to_mlx_module_paths(name);
            let override_val = serde_json::json!({
                "group_size": tensor.quant_info.group_size,
                "bits": tensor.quant_info.bits
            });
            for path in module_paths {
                quant_obj.insert(path, override_val.clone());
            }
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

/// Convert an hf2q tensor name to the module path(s) mlx-lm expects after sanitization.
///
/// Returns multiple paths for fused tensors (gate_up_proj → gate_proj + up_proj).
fn tensor_name_to_mlx_module_paths(name: &str) -> Vec<String> {
    let mut path = name.to_string();

    // Strip "model." prefix
    if let Some(stripped) = path.strip_prefix("model.") {
        path = stripped.to_string();
    }

    // Insert extra "model." after "language_model."
    if path.starts_with("language_model.") && !path.starts_with("language_model.model.") {
        path = path.replacen("language_model.", "language_model.model.", 1);
    }

    // Strip ".weight" suffix
    if let Some(stripped) = path.strip_suffix(".weight") {
        path = stripped.to_string();
    }

    // Handle expert renaming — gate_up_proj produces TWO module paths
    if path.ends_with(".experts.gate_up_proj") {
        let base = path.replace(".experts.gate_up_proj", ".experts.switch_glu");
        vec![
            format!("{}.gate_proj", base),
            format!("{}.up_proj", base),
        ]
    } else if path.ends_with(".experts.down_proj") {
        vec![path.replace(".experts.down_proj", ".experts.switch_glu.down_proj")]
    } else {
        vec![path]
    }
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
        "chat_template.jinja",
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
    fn test_unpack_symmetric_4bit() {
        // Byte 0x37 = lo nibble 0x7 (signed +7), hi nibble 0x3 (signed +3)
        let data = vec![0x37u8];
        let result = unpack_symmetric(&data, 4, 2);
        assert_eq!(result, vec![7i8, 3i8]);

        // Byte 0xF9 = lo=0x9 (signed -7), hi=0xF (signed -1)
        let data2 = vec![0xF9u8];
        let result2 = unpack_symmetric(&data2, 4, 2);
        assert_eq!(result2, vec![-7i8, -1i8]);

        // Zero byte: both nibbles = 0
        let zero = unpack_symmetric(&[0x00], 4, 2);
        assert_eq!(zero, vec![0i8, 0i8]);
    }

    #[test]
    fn test_mlx_affine_quantize_roundtrip() {
        // Quantize known values and verify dequantization matches
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let (packed, scales_bf16, biases_bf16) = mlx_affine_quantize(&weights, 4, 64);

        // Verify output sizes
        assert_eq!(packed.len(), 64 * 4 / 32 * 4); // 64 values, 8 per uint32, 4 bytes each = 32 bytes
        assert_eq!(scales_bf16.len(), 2); // 1 group * 2 bytes bf16
        assert_eq!(biases_bf16.len(), 2);

        // Dequantize and check reconstruction error
        let scale = half::bf16::from_le_bytes([scales_bf16[0], scales_bf16[1]]).to_f32();
        let bias = half::bf16::from_le_bytes([biases_bf16[0], biases_bf16[1]]).to_f32();

        let mut max_error = 0.0f32;
        for (j, chunk) in packed.chunks_exact(4).enumerate() {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            for k in 0..8 {
                let idx = j * 8 + k;
                if idx >= 64 { break; }
                let uint_val = ((word >> (k * 4)) & 0xF) as f32;
                let dequant = uint_val * scale + bias;
                let error = (weights[idx] - dequant).abs();
                max_error = max_error.max(error);
            }
        }
        // 4-bit quantization of range [-3.2, 3.1] should have max error < 0.5
        assert!(max_error < 0.5, "Max reconstruction error too high: {}", max_error);
    }

    #[test]
    fn test_mlx_weight_shape() {
        // Power-of-2: packed into uint32
        // 4-bit: last dim * 4 / 32 = last_dim / 8
        assert_eq!(mlx_weight_shape(&[64, 128], 4), vec![64, 16]);
        assert_eq!(mlx_weight_shape(&[8, 64, 128], 4), vec![8, 64, 16]);
        // 2-bit: last dim * 2 / 32 = last_dim / 16
        assert_eq!(mlx_weight_shape(&[64, 128], 2), vec![64, 8]);
        // 8-bit: last dim * 8 / 32 = last_dim / 4
        assert_eq!(mlx_weight_shape(&[64, 128], 8), vec![64, 32]);

        // Non-power-of-2: also packed into uint32
        // 3-bit: last dim * 3 / 32
        assert_eq!(mlx_weight_shape(&[64, 128], 3), vec![64, 12]);
        // 6-bit: last dim * 6 / 32
        assert_eq!(mlx_weight_shape(&[64, 128], 6), vec![64, 24]);
    }

    #[test]
    fn test_mlx_scales_shape() {
        assert_eq!(mlx_scales_shape(&[64, 128], 64), vec![64, 2]);
        assert_eq!(mlx_scales_shape(&[8, 64, 128], 64), vec![8, 64, 2]);
        assert_eq!(mlx_scales_shape(&[64, 128], 32), vec![64, 4]);
    }

    #[test]
    fn test_mlx_affine_quantize_3bit_packing() {
        // 3-bit: 10 values per uint32 (30 bits used), stored as U32
        // group_size must be divisible by el_per_u32 — use 10 (=10*1 u32)
        // Actually group_size * 3 / 32 must be integer. Use 32 values: 32*3/32 = 3 u32s
        let weights: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 5.0).collect();
        let (packed, scales_bf16, biases_bf16) = mlx_affine_quantize(&weights, 3, 32);

        // 32 values * 3 bits / 32 = 3 uint32s = 12 bytes
        assert_eq!(packed.len(), 12, "3-bit packing of 32 values: 3 uint32s = 12 bytes");
        assert_eq!(scales_bf16.len(), 2); // 1 group
        assert_eq!(biases_bf16.len(), 2);
    }

    #[test]
    fn test_mlx_affine_quantize_6bit_packing() {
        // 6-bit: 5 values per uint32 (30 bits used), stored as U32
        // group_size=64: 64*6/32 = 12 uint32s
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let (packed, scales_bf16, biases_bf16) = mlx_affine_quantize(&weights, 6, 64);

        // 64 values * 6 bits / 32 = 12 uint32s = 48 bytes
        assert_eq!(packed.len(), 48, "6-bit packing of 64 values: 12 uint32s = 48 bytes");
        assert_eq!(scales_bf16.len(), 2); // 1 group
        assert_eq!(biases_bf16.len(), 2);

        // Verify dequantization roundtrip
        let scale = half::bf16::from_le_bytes([scales_bf16[0], scales_bf16[1]]).to_f32();
        let bias = half::bf16::from_le_bytes([biases_bf16[0], biases_bf16[1]]).to_f32();

        let mut max_error = 0.0f32;
        for (j, chunk) in packed.chunks_exact(4).enumerate() {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            for k in 0..5 { // 5 values per uint32 for 6-bit
                let idx = j * 5 + k;
                if idx >= 64 { break; }
                let uint_val = ((word >> (k * 6)) & 0x3F) as f32;
                let dequant = uint_val * scale + bias;
                let error = (weights[idx] - dequant).abs();
                max_error = max_error.max(error);
            }
        }
        assert!(max_error < 0.5, "6-bit max reconstruction error too high: {}", max_error);
    }

    #[test]
    fn test_mlx_weight_shape_3bit_6bit() {
        // All bit widths stored as uint32: last_dim * bits / 32
        assert_eq!(mlx_weight_shape(&[64, 128], 3), vec![64, 12]); // 128 * 3 / 32 = 12
        assert_eq!(mlx_weight_shape(&[64, 64], 3), vec![64, 6]);   // 64 * 3 / 32 = 6
        assert_eq!(mlx_weight_shape(&[64, 128], 6), vec![64, 24]); // 128 * 6 / 32 = 24
        assert_eq!(mlx_weight_shape(&[64, 64], 6), vec![64, 12]);  // 64 * 6 / 32 = 12
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
