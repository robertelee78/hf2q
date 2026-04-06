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

        // Build safetensors file for this shard
        let shard_size = write_single_safetensors_file(shard_tensors, &shard_path)?;

        // Track weight map for index.json
        for (name, _) in shard_tensors {
            weight_map.insert((*name).clone(), shard_filename.clone());
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

/// Write a single safetensors file from a set of tensors.
///
/// Safetensors format:
/// - 8 bytes: header length (u64 LE)
/// - N bytes: JSON header
/// - remaining: tensor data (concatenated in order)
fn write_single_safetensors_file(
    tensors: &[(&String, &QuantizedTensor)],
    path: &Path,
) -> Result<u64, BackendError> {
    // Build header and compute data offsets
    let mut header_map: BTreeMap<String, Value> = BTreeMap::new();
    let mut data_sections: Vec<(&[u8], &[u8])> = Vec::new(); // (quant_data, scale_data) pairs

    let mut current_offset = 0usize;

    for (name, tensor) in tensors {
        // The main tensor data
        let data = &tensor.data;
        let data_end = current_offset + data.len();

        // Determine the dtype string for the output
        // For quantized tensors, we store the raw quantized data
        // MLX expects specific dtypes — for quantized weights, we use the quantized format
        let dtype_str = output_dtype_string(tensor);

        let tensor_info = serde_json::json!({
            "dtype": dtype_str,
            "shape": tensor.shape,
            "data_offsets": [current_offset, data_end]
        });

        header_map.insert((*name).clone(), tensor_info);
        current_offset = data_end;

        // If there are scales, write them as a separate tensor
        if let Some(ref scales) = tensor.quant_info.scales {
            let scale_name = format!("{}.scales", name);
            let num_groups = scales.len() / 2; // f16 = 2 bytes each
            let scale_end = current_offset + scales.len();

            let scale_info = serde_json::json!({
                "dtype": "F16",
                "shape": [num_groups],
                "data_offsets": [current_offset, scale_end]
            });

            header_map.insert(scale_name, scale_info);
            data_sections.push((data, scales));
            current_offset = scale_end;
            continue;
        }

        data_sections.push((data, &[]));
    }

    // Add metadata
    let metadata = serde_json::json!({
        "format": "pt"
    });
    header_map.insert("__metadata__".to_string(), metadata);

    let header_json =
        serde_json::to_string(&header_map).map_err(BackendError::Serialization)?;
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;

    // Write the file
    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header_size.to_le_bytes());
    file_data.extend_from_slice(header_bytes);

    for (data, scales) in &data_sections {
        file_data.extend_from_slice(data);
        if !scales.is_empty() {
            file_data.extend_from_slice(scales);
        }
    }

    fs::write(path, &file_data).map_err(|e| BackendError::WriteFailed {
        reason: format!("Failed to write {}: {}", path.display(), e),
    })?;

    debug!(path = %path.display(), size = file_data.len(), "Wrote safetensors shard");

    Ok(file_data.len() as u64)
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

/// Write MLX-compatible config.json.
fn write_mlx_config(model: &QuantizedModel, output_dir: &Path) -> Result<u64, BackendError> {
    let config = &model.metadata.raw_config;

    let config_json = serde_json::to_string_pretty(config)
        .map_err(BackendError::Serialization)?;

    let path = output_dir.join("config.json");
    fs::write(&path, &config_json).map_err(|e| BackendError::WriteFailed {
        reason: format!("Failed to write config.json: {}", e),
    })?;

    Ok(config_json.len() as u64)
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
}
