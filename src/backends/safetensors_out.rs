//! Safetensors output backend — writes quantized models in the safetensors format.
//!
//! Target consumers: inferrs, Candle, vLLM.
//!
//! Features:
//! - Single-file output for small models (<5 GB)
//! - Multi-shard output for large models with index.json
//! - quantization_config.json sidecar with per-tensor metadata
//! - Scale tensors stored as separate entries (e.g., `tensor_name.scales`)

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use tracing::{debug, info};

use crate::backends::{BackendError, OutputBackend};
use crate::ir::{
    DType, FormatWarning, OutputFile, OutputManifest, QuantizedModel, WarningSeverity,
};
use crate::progress::ProgressReporter;

/// Maximum shard size in bytes (~4 GB).
const SHARD_SIZE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Safetensors output backend.
pub struct SafetensorsBackend;

impl SafetensorsBackend {
    pub fn new() -> Self {
        Self
    }

    /// Map our DType to safetensors dtype string.
    fn dtype_to_safetensors(dtype: DType) -> &'static str {
        match dtype {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::U8 => "U8",
            DType::U16 => "U16",
            DType::U32 => "U32",
            DType::Bool => "BOOL",
        }
    }

    /// Build the quantization_config.json content.
    fn build_quant_config(model: &QuantizedModel) -> serde_json::Value {
        let mut per_layer_bits = serde_json::Map::new();
        let mut quant_info_map = serde_json::Map::new();

        for (name, tensor) in &model.tensors {
            per_layer_bits.insert(
                name.clone(),
                serde_json::Value::Number(tensor.quant_info.bits.into()),
            );
            quant_info_map.insert(
                name.clone(),
                serde_json::json!({
                    "method": tensor.quant_info.method,
                    "bits": tensor.quant_info.bits,
                    "group_size": tensor.quant_info.group_size,
                    "preserved": tensor.quant_info.preserved,
                }),
            );
        }

        serde_json::json!({
            "quant_method": model.quant_method,
            "bits": model.bits,
            "group_size": model.group_size,
            "per_layer_bits": per_layer_bits,
            "quantization_info": quant_info_map,
        })
    }

    /// Build the __metadata__ map for safetensors header.
    fn build_header_metadata(model: &QuantizedModel) -> BTreeMap<String, String> {
        let mut meta = BTreeMap::new();
        meta.insert("format".to_string(), "hf2q".to_string());
        meta.insert("quant_method".to_string(), model.quant_method.clone());
        meta.insert("bits".to_string(), model.bits.to_string());
        meta.insert("group_size".to_string(), model.group_size.to_string());
        meta.insert(
            "architecture".to_string(),
            model.metadata.architecture.clone(),
        );
        meta.insert(
            "tensor_count".to_string(),
            model.tensors.len().to_string(),
        );
        meta
    }

    /// Collect all tensor entries (weights + scales) in sorted order, returning
    /// (name, dtype_str, shape, data_ref) tuples.
    fn collect_tensor_entries(
        model: &QuantizedModel,
    ) -> Vec<(String, &'static str, Vec<usize>, &[u8])> {
        let mut entries: Vec<(String, &str, Vec<usize>, &[u8])> = Vec::new();

        let mut names: Vec<&String> = model.tensors.keys().collect();
        names.sort();

        for name in names {
            let tensor = &model.tensors[name];

            // Determine the dtype for storage:
            // - Preserved tensors keep original dtype
            // - Quantized tensors are stored as U8 (packed bits)
            let dtype_str = if tensor.quant_info.preserved {
                Self::dtype_to_safetensors(tensor.original_dtype)
            } else {
                "U8"
            };

            let shape = if tensor.quant_info.preserved {
                tensor.shape.clone()
            } else {
                // Packed quantized data: store as flat byte array
                vec![tensor.data.len()]
            };

            entries.push((name.clone(), dtype_str, shape, tensor.data.as_slice()));

            // Add scales as a separate tensor if present
            if let Some(ref scales) = tensor.quant_info.scales {
                let scale_name = format!("{}.scales", name);
                // Scales are typically f16 or f32; store as raw U8 bytes
                entries.push((scale_name, "U8", vec![scales.len()], scales.as_slice()));
            }

            // Add biases/zero-points if present
            if let Some(ref biases) = tensor.quant_info.biases {
                let bias_name = format!("{}.biases", name);
                entries.push((bias_name, "U8", vec![biases.len()], biases.as_slice()));
            }
        }

        entries
    }

    /// Serialize a set of tensor entries into safetensors format bytes.
    /// Uses manual serialization compatible with the safetensors spec:
    /// [header_size: u64 LE] [header: JSON] [tensor data...]
    fn serialize_safetensors(
        entries: &[(String, &str, Vec<usize>, &[u8])],
        metadata: &BTreeMap<String, String>,
    ) -> Result<Vec<u8>, BackendError> {
        // Build header JSON: { "__metadata__": {...}, "tensor_name": { "dtype": ..., "shape": [...], "data_offsets": [start, end] } }
        let mut header = serde_json::Map::new();

        // Add metadata
        let meta_value = serde_json::to_value(metadata)
            .map_err(|e| BackendError::Serialization(e))?;
        header.insert("__metadata__".to_string(), meta_value);

        // Calculate data offsets
        let mut offset: usize = 0;
        let mut data_ranges: Vec<(usize, usize)> = Vec::with_capacity(entries.len());

        for (_, _, _, data) in entries {
            let start = offset;
            let end = start + data.len();
            data_ranges.push((start, end));
            offset = end;
        }

        // Add tensor metadata to header
        for (i, (name, dtype, shape, _)) in entries.iter().enumerate() {
            let (start, end) = data_ranges[i];
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }

        let header_json =
            serde_json::to_string(&header).map_err(|e| BackendError::Serialization(e))?;

        // Pad header to 8-byte alignment
        let header_bytes = header_json.as_bytes();
        let padding = (8 - (header_bytes.len() % 8)) % 8;
        let padded_header_len = header_bytes.len() + padding;

        // Total size: 8 (header size) + padded_header + data
        let total_data_size: usize = entries.iter().map(|(_, _, _, d)| d.len()).sum();
        let total_size = 8 + padded_header_len + total_data_size;

        let mut buffer = Vec::with_capacity(total_size);

        // Write header size as u64 LE
        buffer.extend_from_slice(&(padded_header_len as u64).to_le_bytes());

        // Write header JSON + padding
        buffer.extend_from_slice(header_bytes);
        buffer.extend(std::iter::repeat(b' ').take(padding));

        // Write tensor data
        for (_, _, _, data) in entries {
            buffer.extend_from_slice(data);
        }

        Ok(buffer)
    }
}

impl OutputBackend for SafetensorsBackend {
    fn name(&self) -> &str {
        "safetensors"
    }

    fn validate(&self, model: &QuantizedModel) -> Result<Vec<FormatWarning>, BackendError> {
        let mut warnings = Vec::new();

        for (name, tensor) in &model.tensors {
            // Warn if shape is empty
            if tensor.shape.is_empty() {
                warnings.push(FormatWarning {
                    message: format!("Tensor '{}' has empty shape", name),
                    severity: WarningSeverity::Warning,
                });
            }

            // Warn if preserved tensor has no data
            if tensor.quant_info.preserved && tensor.data.is_empty() {
                warnings.push(FormatWarning {
                    message: format!("Preserved tensor '{}' has no data", name),
                    severity: WarningSeverity::Warning,
                });
            }

            // Warn if quantized tensor has unexpected data size for its bit width
            if !tensor.quant_info.preserved && tensor.data.is_empty() {
                warnings.push(FormatWarning {
                    message: format!("Quantized tensor '{}' has no data", name),
                    severity: WarningSeverity::Warning,
                });
            }
        }

        if model.tensors.is_empty() {
            warnings.push(FormatWarning {
                message: "Model has no tensors".to_string(),
                severity: WarningSeverity::Warning,
            });
        }

        Ok(warnings)
    }

    fn write(
        &self,
        model: &QuantizedModel,
        _input_dir: &Path,
        output_dir: &Path,
        progress: &ProgressReporter,
    ) -> Result<OutputManifest, BackendError> {
        fs::create_dir_all(output_dir)?;

        let entries = Self::collect_tensor_entries(model);
        let metadata = Self::build_header_metadata(model);

        // Calculate total data size to decide on sharding
        let total_data_size: u64 = entries.iter().map(|(_, _, _, d)| d.len() as u64).sum();
        info!(
            total_size_mb = total_data_size / (1024 * 1024),
            tensor_count = entries.len(),
            "Writing safetensors output"
        );

        let pb = progress.bar(entries.len() as u64, "Writing safetensors");

        let mut output_files = Vec::new();

        if total_data_size > SHARD_SIZE_BYTES {
            // Multi-shard output
            let mut shards: Vec<Vec<(String, &str, Vec<usize>, &[u8])>> = Vec::new();
            let mut current_shard: Vec<(String, &str, Vec<usize>, &[u8])> = Vec::new();
            let mut current_size: u64 = 0;

            for entry in &entries {
                let entry_size = entry.3.len() as u64;
                if !current_shard.is_empty() && current_size + entry_size > SHARD_SIZE_BYTES {
                    shards.push(std::mem::take(&mut current_shard));
                    current_size = 0;
                }
                current_shard.push((
                    entry.0.clone(),
                    entry.1,
                    entry.2.clone(),
                    entry.3,
                ));
                current_size += entry_size;
            }
            if !current_shard.is_empty() {
                shards.push(current_shard);
            }

            let total_shards = shards.len();
            let mut weight_map: BTreeMap<String, String> = BTreeMap::new();

            for (shard_idx, shard_entries) in shards.iter().enumerate() {
                let shard_num = shard_idx + 1;
                let filename =
                    format!("model-{:05}-of-{:05}.safetensors", shard_num, total_shards);

                debug!(shard = shard_num, tensors = shard_entries.len(), "Writing shard");

                for (name, _, _, _) in shard_entries {
                    weight_map.insert(name.clone(), filename.clone());
                }

                let shard_bytes =
                    Self::serialize_safetensors(shard_entries, &metadata)?;

                let shard_path = output_dir.join(&filename);
                fs::write(&shard_path, &shard_bytes)?;

                output_files.push(OutputFile {
                    filename: filename.clone(),
                    size_bytes: shard_bytes.len() as u64,
                });

                pb.inc(shard_entries.len() as u64);
            }

            // Write index file
            let index = serde_json::json!({
                "metadata": {
                    "total_size": total_data_size,
                },
                "weight_map": weight_map,
            });
            let index_json = serde_json::to_string_pretty(&index)
                .map_err(|e| BackendError::Serialization(e))?;
            let index_path = output_dir.join("model.safetensors.index.json");
            fs::write(&index_path, &index_json)?;

            output_files.push(OutputFile {
                filename: "model.safetensors.index.json".to_string(),
                size_bytes: index_json.len() as u64,
            });

            info!(shards = total_shards, "Multi-shard safetensors output written");
        } else {
            // Single-file output
            let shard_bytes = Self::serialize_safetensors(&entries, &metadata)?;
            let output_path = output_dir.join("model.safetensors");
            fs::write(&output_path, &shard_bytes)?;

            output_files.push(OutputFile {
                filename: "model.safetensors".to_string(),
                size_bytes: shard_bytes.len() as u64,
            });

            pb.inc(entries.len() as u64);
            info!("Single-file safetensors output written");
        }

        // Write quantization_config.json sidecar
        let quant_config = Self::build_quant_config(model);
        let config_json = serde_json::to_string_pretty(&quant_config)
            .map_err(|e| BackendError::Serialization(e))?;
        let config_path = output_dir.join("quantization_config.json");
        fs::write(&config_path, &config_json)?;

        output_files.push(OutputFile {
            filename: "quantization_config.json".to_string(),
            size_bytes: config_json.len() as u64,
        });

        pb.finish_with_message("Safetensors output complete");

        let total_size_bytes: u64 = output_files.iter().map(|f| f.size_bytes).sum();
        let shard_count = output_files
            .iter()
            .filter(|f| f.filename.ends_with(".safetensors"))
            .count();

        Ok(OutputManifest {
            output_dir: output_dir.to_string_lossy().to_string(),
            files: output_files,
            total_size_bytes,
            shard_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::ir::{ModelMetadata, QuantizedModel, QuantizedTensor, TensorQuantInfo};

    fn make_metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: "TestArch".to_string(),
            model_type: "test".to_string(),
            param_count: 1000,
            hidden_size: 64,
            num_layers: 2,
            layer_types: vec!["attention".to_string()],
            num_attention_heads: 4,
            num_kv_heads: Some(4),
            vocab_size: 256,
            dtype: "float16".to_string(),
            shard_count: 1,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: Some(128),
            raw_config: serde_json::Value::Null,
        }
    }

    fn make_tensor(name: &str, data: Vec<u8>, preserved: bool) -> QuantizedTensor {
        QuantizedTensor {
            name: name.to_string(),
            shape: vec![4, 4],
            original_dtype: DType::F16,
            data,
            quant_info: TensorQuantInfo {
                method: if preserved {
                    "passthrough".to_string()
                } else {
                    "q4".to_string()
                },
                bits: if preserved { 16 } else { 4 },
                group_size: 32,
                preserved,
                scales: if preserved {
                    None
                } else {
                    Some(vec![0u8; 8])
                },
                biases: None,
            },
        }
    }

    fn make_model(tensors: Vec<QuantizedTensor>) -> QuantizedModel {
        let mut tensor_map = HashMap::new();
        for t in tensors {
            tensor_map.insert(t.name.clone(), t);
        }
        QuantizedModel {
            metadata: make_metadata(),
            tensors: tensor_map,
            quant_method: "q4_k".to_string(),
            group_size: 32,
            bits: 4,
        }
    }

    #[test]
    fn test_validate_well_formed_model() {
        let backend = SafetensorsBackend::new();
        let model = make_model(vec![
            make_tensor("weight_a", vec![1u8; 16], false),
            make_tensor("norm_b", vec![2u8; 32], true),
        ]);
        let warnings = backend.validate(&model).unwrap();
        assert!(warnings.is_empty(), "Well-formed model should produce no warnings");
    }

    #[test]
    fn test_validate_empty_model_warns() {
        let backend = SafetensorsBackend::new();
        let model = make_model(vec![]);
        let warnings = backend.validate(&model).unwrap();
        assert!(
            warnings.iter().any(|w| w.message.contains("no tensors")),
            "Empty model should produce a warning"
        );
    }

    #[test]
    fn test_write_single_file() {
        let backend = SafetensorsBackend::new();
        let model = make_model(vec![
            make_tensor("layer.0.weight", vec![0xAB; 32], false),
            make_tensor("layer.1.norm", vec![0xCD; 16], true),
        ]);

        let tmp = tempfile::tempdir().unwrap();
        let progress = ProgressReporter::new();
        let manifest = backend
            .write(&model, tmp.path(), tmp.path(), &progress)
            .unwrap();

        // Should produce model.safetensors + quantization_config.json
        assert_eq!(manifest.shard_count, 1);
        assert!(
            manifest.files.iter().any(|f| f.filename == "model.safetensors"),
            "Should write model.safetensors"
        );
        assert!(
            manifest
                .files
                .iter()
                .any(|f| f.filename == "quantization_config.json"),
            "Should write quantization_config.json"
        );

        // Verify the safetensors file is readable
        let st_path = tmp.path().join("model.safetensors");
        let st_bytes = fs::read(&st_path).unwrap();
        assert!(st_bytes.len() > 8, "Safetensors file should have header + data");

        // Verify header size is valid
        let header_size =
            u64::from_le_bytes(st_bytes[..8].try_into().unwrap()) as usize;
        assert!(header_size > 0, "Header size should be positive");
        assert!(
            8 + header_size <= st_bytes.len(),
            "Header size should not exceed file size"
        );

        // Verify header is valid JSON
        let header_str =
            std::str::from_utf8(&st_bytes[8..8 + header_size]).unwrap();
        let header: serde_json::Value =
            serde_json::from_str(header_str.trim()).unwrap();
        assert!(header.get("__metadata__").is_some(), "Should have __metadata__");
        assert!(
            header.get("layer.0.weight").is_some(),
            "Should have weight tensor in header"
        );

        // Verify quantization_config.json
        let config_path = tmp.path().join("quantization_config.json");
        let config: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(config["quant_method"], "q4_k");
        assert_eq!(config["bits"], 4);
        assert_eq!(config["group_size"], 32);
    }

    #[test]
    fn test_validate_empty_data_warns() {
        let backend = SafetensorsBackend::new();
        let model = make_model(vec![make_tensor("bad_tensor", vec![], false)]);
        let warnings = backend.validate(&model).unwrap();
        assert!(
            warnings
                .iter()
                .any(|w| w.message.contains("no data")),
            "Empty quantized data should warn"
        );
    }

    #[test]
    fn test_scales_written_as_separate_tensor() {
        let backend = SafetensorsBackend::new();
        let model = make_model(vec![make_tensor("proj.weight", vec![0xFF; 64], false)]);

        let tmp = tempfile::tempdir().unwrap();
        let progress = ProgressReporter::new();
        let manifest = backend
            .write(&model, tmp.path(), tmp.path(), &progress)
            .unwrap();

        let st_path = tmp.path().join("model.safetensors");
        let st_bytes = fs::read(&st_path).unwrap();
        let header_size =
            u64::from_le_bytes(st_bytes[..8].try_into().unwrap()) as usize;
        let header_str =
            std::str::from_utf8(&st_bytes[8..8 + header_size]).unwrap();
        let header: serde_json::Value =
            serde_json::from_str(header_str.trim()).unwrap();

        assert!(
            header.get("proj.weight.scales").is_some(),
            "Scales should be stored as separate tensor entry"
        );
        assert_eq!(manifest.shard_count, 1);
    }

    #[test]
    fn test_header_metadata_content() {
        let model = make_model(vec![make_tensor("w", vec![1; 8], false)]);
        let meta = SafetensorsBackend::build_header_metadata(&model);
        assert_eq!(meta.get("format").unwrap(), "hf2q");
        assert_eq!(meta.get("quant_method").unwrap(), "q4_k");
        assert_eq!(meta.get("bits").unwrap(), "4");
        assert_eq!(meta.get("architecture").unwrap(), "TestArch");
    }
}
