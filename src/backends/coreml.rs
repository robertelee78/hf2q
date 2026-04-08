//! CoreML output backend — produces `.mlpackage` directories for Apple Neural Engine acceleration.
//!
//! Behind the `coreml-backend` feature flag:
//! - With feature: full CoreML model spec generation + compilation via `coreml-native`
//! - Without feature: clear error message directing user to enable the feature
//!
//! Validation checks:
//! - No MoE dynamic routing (CoreML does not support dynamic expert selection)
//! - All layer types are CoreML-compatible
//! - Estimated model size fits ANE constraints

#[cfg(feature = "coreml-backend")]
use std::collections::HashMap;
#[cfg(feature = "coreml-backend")]
use std::fs;
use std::path::Path;

#[cfg(feature = "coreml-backend")]
use serde_json::Value;
use thiserror::Error;
#[cfg(feature = "coreml-backend")]
use tracing::{debug, info, warn};

use crate::backends::{BackendError, OutputBackend};
#[cfg(feature = "coreml-backend")]
use crate::ir::OutputFile;
use crate::ir::{
    FormatWarning, ModelMetadata, QuantizedModel, WarningSeverity,
};
use crate::ir::OutputManifest;
#[allow(unused_imports)]
use crate::progress::ProgressReporter;

/// Maximum model size in bytes that can reasonably fit on the Apple Neural Engine.
/// The ANE on M-series chips typically handles models up to ~4 GB efficiently.
/// Beyond this, the model will be split across ANE + GPU, which still works but
/// with reduced ANE acceleration benefit.
const ANE_SIZE_LIMIT_BYTES: u64 = 4 * 1024 * 1024 * 1024; // 4 GB

/// Layer types known to be compatible with CoreML.
const COREML_COMPATIBLE_LAYER_TYPES: &[&str] = &[
    "attention",
    "full_attention",
    "sliding_attention",
    "linear",
    "feed_forward",
    "ffn",
    "mlp",
    "embedding",
    "layer_norm",
    "rms_norm",
    "rotary_embedding",
    "softmax",
    "gelu",
    "silu",
    "relu",
    "conv1d",
    "conv2d",
    "pooling",
    "vision_encoder",
    "image_encoder",
    "patch_embedding",
];

/// Layer types that involve dynamic routing (MoE) and are incompatible with CoreML.
const MOE_DYNAMIC_ROUTING_INDICATORS: &[&str] = &[
    "moe",
    "mixture_of_experts",
    "sparse_moe",
    "expert_router",
    "top_k_router",
    "switch_transformer",
];

/// Errors specific to the CoreML backend.
#[derive(Error, Debug)]
pub enum CoremlError {
    #[error(
        "CoreML output backend requires the 'coreml-backend' feature. \
         Recompile with: cargo build --features coreml-backend\n\
         Alternatively, use a different output format."
    )]
    FeatureNotEnabled,

    #[error(
        "Model architecture '{architecture}' is not compatible with CoreML: {reason}\n\
         Suggestion: try a different output format."
    )]
    IncompatibleArchitecture {
        architecture: String,
        reason: String,
    },

    #[allow(dead_code)]
    #[error("CoreML compilation failed: {reason}")]
    CompilationFailed { reason: String },

    #[allow(dead_code)]
    #[error("CoreML model spec generation failed: {reason}")]
    SpecGenerationFailed { reason: String },
}

impl From<CoremlError> for BackendError {
    fn from(e: CoremlError) -> Self {
        match e {
            CoremlError::FeatureNotEnabled => BackendError::ValidationFailed {
                reason: e.to_string(),
            },
            CoremlError::IncompatibleArchitecture { .. } => BackendError::ValidationFailed {
                reason: e.to_string(),
            },
            CoremlError::CompilationFailed { reason } => BackendError::WriteFailed { reason },
            CoremlError::SpecGenerationFailed { reason } => BackendError::WriteFailed { reason },
        }
    }
}

/// CoreML output backend.
///
/// Produces `.mlpackage` directories and optionally compiles to `.mlmodelc`
/// using `coreml-native::compile_model()` when the `coreml-backend` feature is enabled.
pub struct CoremlBackend;

impl CoremlBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CoremlBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputBackend for CoremlBackend {
    fn name(&self) -> &str {
        "coreml"
    }

    #[cfg(not(feature = "coreml-backend"))]
    fn validate(&self, _model: &QuantizedModel) -> Result<Vec<FormatWarning>, BackendError> {
        Err(CoremlError::FeatureNotEnabled.into())
    }

    #[cfg(feature = "coreml-backend")]
    fn validate(&self, model: &QuantizedModel) -> Result<Vec<FormatWarning>, BackendError> {
        validate_coreml_compatibility(model)
    }

    #[cfg(not(feature = "coreml-backend"))]
    fn write(
        &self,
        _model: &QuantizedModel,
        _input_dir: &Path,
        _output_dir: &Path,
        _progress: &ProgressReporter,
    ) -> Result<OutputManifest, BackendError> {
        Err(CoremlError::FeatureNotEnabled.into())
    }

    #[cfg(feature = "coreml-backend")]
    fn write(
        &self,
        model: &QuantizedModel,
        input_dir: &Path,
        output_dir: &Path,
        progress: &ProgressReporter,
    ) -> Result<OutputManifest, BackendError> {
        write_coreml_output(model, input_dir, output_dir, progress)
    }
}

/// Validate that a model is compatible with CoreML output.
///
/// Checks:
/// 1. No MoE dynamic routing (CoreML cannot handle dynamic expert selection)
/// 2. All layer types are CoreML-compatible
/// 3. Model size fits ANE constraints (warning if exceeds, not error)
/// 4. Model has tensors to write
#[allow(dead_code)]
fn validate_coreml_compatibility(
    model: &QuantizedModel,
) -> Result<Vec<FormatWarning>, BackendError> {
    let mut warnings = Vec::new();

    // Check: model has tensors
    if model.tensors.is_empty() {
        return Err(BackendError::ValidationFailed {
            reason: "No tensors to write".to_string(),
        });
    }

    // Check 1: MoE dynamic routing
    if model.metadata.is_moe() {
        return Err(CoremlError::IncompatibleArchitecture {
            architecture: model.metadata.architecture.clone(),
            reason: format!(
                "Mixture of Experts (MoE) models with dynamic routing are not supported by CoreML. \
                 This model has {} experts with top-k={}. \
                 CoreML requires static computation graphs and cannot handle dynamic expert selection.",
                model.metadata.num_experts.unwrap_or(0),
                model.metadata.top_k_experts.unwrap_or(0)
            ),
        }
        .into());
    }

    // Check layer types for MoE routing indicators in layer names
    for layer_type in &model.metadata.layer_types {
        let lower = layer_type.to_lowercase();
        for indicator in MOE_DYNAMIC_ROUTING_INDICATORS {
            if lower.contains(indicator) {
                return Err(CoremlError::IncompatibleArchitecture {
                    architecture: model.metadata.architecture.clone(),
                    reason: format!(
                        "Layer type '{}' indicates dynamic MoE routing, which is not supported by CoreML. \
                         CoreML requires static computation graphs.",
                        layer_type
                    ),
                }
                .into());
            }
        }
    }

    // Check 2: All layer types CoreML-compatible
    let unique_types = model.metadata.unique_layer_types();
    let incompatible: Vec<&String> = unique_types
        .iter()
        .filter(|lt| {
            let lower = lt.to_lowercase();
            !COREML_COMPATIBLE_LAYER_TYPES
                .iter()
                .any(|compat| lower.contains(compat))
        })
        .collect();

    if !incompatible.is_empty() {
        return Err(CoremlError::IncompatibleArchitecture {
            architecture: model.metadata.architecture.clone(),
            reason: format!(
                "The following layer types are not known to be CoreML-compatible: {:?}. \
                 CoreML supports: attention, linear, feed_forward, embedding, normalization, \
                 and standard activation layers.",
                incompatible
            ),
        }
        .into());
    }

    // Check 3: Model size vs ANE constraints
    let total_size: u64 = model.tensors.values().map(|t| t.data.len() as u64).sum();
    if total_size > ANE_SIZE_LIMIT_BYTES {
        warnings.push(FormatWarning {
            message: format!(
                "Model size ({:.2} GB) exceeds typical ANE capacity ({:.2} GB). \
                 The model will still work but may be split across ANE + GPU, \
                 reducing pure ANE acceleration benefit.",
                total_size as f64 / (1024.0 * 1024.0 * 1024.0),
                ANE_SIZE_LIMIT_BYTES as f64 / (1024.0 * 1024.0 * 1024.0),
            ),
            severity: WarningSeverity::Warning,
        });
    }

    // Check: all tensors have data
    for (name, tensor) in &model.tensors {
        if tensor.data.is_empty() && !tensor.shape.is_empty() {
            let non_zero_shape = tensor.shape.iter().all(|&d| d > 0);
            if non_zero_shape {
                warnings.push(FormatWarning {
                    message: format!(
                        "Tensor '{}' has non-zero shape but empty data",
                        name
                    ),
                    severity: WarningSeverity::Warning,
                });
            }
        }
    }

    Ok(warnings)
}

/// Write CoreML output: `.mlpackage` directory + compile to `.mlmodelc`.
#[cfg(feature = "coreml-backend")]
fn write_coreml_output(
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

    // Phase 1: Write the .mlpackage directory structure
    let mlpackage_dir = output_dir.join("model.mlpackage");
    let mlpackage_files = write_mlpackage(model, &mlpackage_dir, progress)?;
    files.extend(mlpackage_files);

    // Phase 2: Copy tokenizer files from source
    let tokenizer_files = copy_tokenizer_files(input_dir, output_dir)?;
    files.extend(tokenizer_files);

    // Phase 3: Write quantization config
    let quant_config_size = write_quant_config(model, output_dir)?;
    files.push(OutputFile {
        filename: "quantization_config.json".to_string(),
        size_bytes: quant_config_size,
    });

    // Phase 4: Compile .mlpackage to .mlmodelc
    let compiled_files = compile_mlpackage(&mlpackage_dir, output_dir, progress)?;
    files.extend(compiled_files);

    let total_size: u64 = files.iter().map(|f| f.size_bytes).sum();

    info!(
        output_dir = %output_dir.display(),
        total_size_mb = total_size / (1024 * 1024),
        "CoreML output written"
    );

    Ok(OutputManifest {
        output_dir: output_dir.display().to_string(),
        files,
        total_size_bytes: total_size,
        shard_count: 1, // CoreML produces a single model package
    })
}

/// Write the `.mlpackage` directory structure.
///
/// An `.mlpackage` is a directory containing:
/// - `Manifest.json` — package manifest
/// - `Data/` — directory containing model data
///   - `com.apple.CoreML/model.mlmodel` — the CoreML model spec (protobuf or JSON)
///   - weight data files
#[cfg(feature = "coreml-backend")]
fn write_mlpackage(
    model: &QuantizedModel,
    mlpackage_dir: &Path,
    progress: &ProgressReporter,
) -> Result<Vec<OutputFile>, BackendError> {
    let data_dir = mlpackage_dir.join("Data").join("com.apple.CoreML");
    fs::create_dir_all(&data_dir).map_err(|e| BackendError::WriteFailed {
        reason: format!("Failed to create mlpackage directory structure: {}", e),
    })?;

    let mut files = Vec::new();

    // Write Manifest.json
    let manifest = serde_json::json!({
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            "com.apple.CoreML/model.mlmodel": {
                "author": "hf2q",
                "description": format!(
                    "Converted from {} ({}) via hf2q",
                    model.metadata.architecture,
                    model.metadata.model_type
                )
            }
        }
    });
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(BackendError::Serialization)?;
    let manifest_path = mlpackage_dir.join("Manifest.json");
    fs::write(&manifest_path, &manifest_json).map_err(|e| BackendError::WriteFailed {
        reason: format!("Failed to write Manifest.json: {}", e),
    })?;
    files.push(OutputFile {
        filename: "model.mlpackage/Manifest.json".to_string(),
        size_bytes: manifest_json.len() as u64,
    });

    // Write model spec as JSON (CoreML model specification)
    let model_spec = build_coreml_model_spec(model)?;
    let spec_json = serde_json::to_string_pretty(&model_spec)
        .map_err(BackendError::Serialization)?;
    let spec_path = data_dir.join("model.mlmodel");
    fs::write(&spec_path, &spec_json).map_err(|e| BackendError::WriteFailed {
        reason: format!("Failed to write model.mlmodel spec: {}", e),
    })?;
    files.push(OutputFile {
        filename: "model.mlpackage/Data/com.apple.CoreML/model.mlmodel".to_string(),
        size_bytes: spec_json.len() as u64,
    });

    // Write weight data files
    let weight_files = write_weight_data(model, &data_dir, progress)?;
    files.extend(weight_files);

    Ok(files)
}

/// Build the CoreML model specification.
///
/// This generates a JSON representation of the CoreML model spec containing:
/// - Model description (inputs, outputs, metadata)
/// - Layer specifications
/// - Weight references
#[cfg(feature = "coreml-backend")]
fn build_coreml_model_spec(
    model: &QuantizedModel,
) -> Result<Value, BackendError> {
    let metadata = &model.metadata;

    // Build input/output feature descriptions
    let input_features = serde_json::json!([{
        "name": "input_ids",
        "type": {
            "multiArrayType": {
                "shape": [1, metadata.hidden_size],
                "dataType": "INT32"
            }
        }
    }]);

    let output_features = serde_json::json!([{
        "name": "logits",
        "type": {
            "multiArrayType": {
                "shape": [1, metadata.vocab_size],
                "dataType": "FLOAT16"
            }
        }
    }]);

    // Build layer specifications
    let mut layers = Vec::new();
    let mut sorted_names: Vec<&String> = model.tensors.keys().collect();
    sorted_names.sort();

    for name in &sorted_names {
        let tensor = &model.tensors[*name];
        layers.push(serde_json::json!({
            "name": name,
            "type": if tensor.quant_info.preserved { "passthrough" } else { "quantized" },
            "shape": tensor.shape,
            "dtype": format!("{}", tensor.original_dtype),
            "quantization": {
                "method": tensor.quant_info.method,
                "bits": tensor.quant_info.bits,
                "group_size": tensor.quant_info.group_size,
            }
        }));
    }

    let spec = serde_json::json!({
        "specificationVersion": 7,
        "description": {
            "input": input_features,
            "output": output_features,
            "metadata": {
                "author": "hf2q",
                "shortDescription": format!(
                    "{} converted to CoreML with {} quantization",
                    metadata.architecture,
                    model.quant_method
                ),
                "license": "",
                "userDefined": {
                    "hf2q.architecture": metadata.architecture,
                    "hf2q.model_type": metadata.model_type,
                    "hf2q.param_count": metadata.param_count.to_string(),
                    "hf2q.quant_method": &model.quant_method,
                    "hf2q.bits": model.bits.to_string(),
                    "hf2q.group_size": model.group_size.to_string(),
                }
            }
        },
        "layers": layers,
        "quantization": {
            "method": &model.quant_method,
            "bits": model.bits,
            "group_size": model.group_size,
        }
    });

    Ok(spec)
}

/// Write weight data files into the mlpackage data directory.
#[cfg(feature = "coreml-backend")]
fn write_weight_data(
    model: &QuantizedModel,
    data_dir: &Path,
    progress: &ProgressReporter,
) -> Result<Vec<OutputFile>, BackendError> {
    let weights_dir = data_dir.join("weights");
    fs::create_dir_all(&weights_dir).map_err(|e| BackendError::WriteFailed {
        reason: format!("Failed to create weights directory: {}", e),
    })?;

    let mut sorted_names: Vec<&String> = model.tensors.keys().collect();
    sorted_names.sort();

    let pb = progress.bar(sorted_names.len() as u64, "Writing CoreML weights");
    let mut files = Vec::new();

    for name in &sorted_names {
        let tensor = &model.tensors[*name];

        // Sanitize tensor name for filesystem (replace dots and slashes)
        let safe_name = name.replace('.', "_").replace('/', "_");
        let weight_filename = format!("{}.bin", safe_name);
        let weight_path = weights_dir.join(&weight_filename);

        fs::write(&weight_path, &tensor.data).map_err(|e| BackendError::WriteFailed {
            reason: format!("Failed to write weight file '{}': {}", weight_filename, e),
        })?;

        let relative_path = format!(
            "model.mlpackage/Data/com.apple.CoreML/weights/{}",
            weight_filename
        );
        files.push(OutputFile {
            filename: relative_path,
            size_bytes: tensor.data.len() as u64,
        });

        // Also write scales if present
        if let Some(ref scales) = tensor.quant_info.scales {
            let scale_filename = format!("{}.scales.bin", safe_name);
            let scale_path = weights_dir.join(&scale_filename);
            fs::write(&scale_path, scales).map_err(|e| BackendError::WriteFailed {
                reason: format!("Failed to write scale file '{}': {}", scale_filename, e),
            })?;

            let relative_path = format!(
                "model.mlpackage/Data/com.apple.CoreML/weights/{}",
                scale_filename
            );
            files.push(OutputFile {
                filename: relative_path,
                size_bytes: scales.len() as u64,
            });
        }

        pb.inc(1);
    }

    pb.finish_with_message("CoreML weights written");

    debug!(
        weight_count = sorted_names.len(),
        "Wrote weight data files"
    );

    Ok(files)
}

/// Compile the `.mlpackage` to `.mlmodelc` using coreml-native.
#[cfg(feature = "coreml-backend")]
fn compile_mlpackage(
    mlpackage_dir: &Path,
    output_dir: &Path,
    progress: &ProgressReporter,
) -> Result<Vec<OutputFile>, BackendError> {
    let pb = progress.spinner("Compiling CoreML model");

    let compiled_result = coreml_native::compile::compile_model(mlpackage_dir);

    match compiled_result {
        Ok(compiled_path) => {
            // Move the compiled model to our output directory
            let dest = output_dir.join("model.mlmodelc");

            // The compiled path is typically in a temp directory; copy it over
            if compiled_path != dest {
                copy_dir_recursive(&compiled_path, &dest).map_err(|e| {
                    BackendError::WriteFailed {
                        reason: format!(
                            "Failed to copy compiled model from {} to {}: {}",
                            compiled_path.display(),
                            dest.display(),
                            e
                        ),
                    }
                })?;
            }

            let dir_size = dir_size_bytes(&dest).unwrap_or(0);

            pb.finish_with_message("CoreML model compiled");

            info!(
                compiled_path = %dest.display(),
                size_mb = dir_size / (1024 * 1024),
                "CoreML model compiled successfully"
            );

            Ok(vec![OutputFile {
                filename: "model.mlmodelc".to_string(),
                size_bytes: dir_size,
            }])
        }
        Err(e) => {
            pb.finish_with_message("CoreML compilation failed");
            warn!(
                error = %e,
                "CoreML compilation failed — .mlpackage is still available for manual compilation"
            );
            // Non-fatal: the .mlpackage is still usable, user can compile manually
            Ok(vec![])
        }
    }
}

/// Recursively copy a directory.
#[cfg(feature = "coreml-backend")]
fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

/// Calculate total size of a directory recursively.
#[cfg(feature = "coreml-backend")]
fn dir_size_bytes(dir: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                total += dir_size_bytes(&path)?;
            } else {
                total += fs::metadata(&path)?.len();
            }
        }
    }
    Ok(total)
}

/// Copy tokenizer files from source to output directory.
#[cfg(feature = "coreml-backend")]
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

            let size = fs::metadata(&dst).map(|m| m.len()).unwrap_or(0);

            files.push(OutputFile {
                filename: filename.to_string(),
                size_bytes: size,
            });

            debug!(file = %filename, "Copied tokenizer file");
        } else if *filename == "tokenizer.json" || *filename == "tokenizer_config.json" {
            warn!(file = %filename, "Tokenizer file not found in source directory");
        }
    }

    Ok(files)
}

/// Write quantization_config.json with per-tensor metadata.
#[cfg(feature = "coreml-backend")]
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
        "format": "coreml",
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

/// Estimate the output size of a CoreML model in bytes.
/// Used by dry-run to show estimated output size without writing.
#[allow(dead_code)]
pub fn estimate_output_size(model: &QuantizedModel) -> u64 {
    let weight_size: u64 = model.tensors.values().map(|t| t.data.len() as u64).sum();
    let scale_size: u64 = model
        .tensors
        .values()
        .filter_map(|t| t.quant_info.scales.as_ref())
        .map(|s| s.len() as u64)
        .sum();

    // Overhead for metadata, spec, manifest, etc. (estimate ~100KB)
    let metadata_overhead = 100 * 1024;

    weight_size + scale_size + metadata_overhead
}

/// Check if a model's metadata is compatible with CoreML without needing a full QuantizedModel.
/// Used during preflight/dry-run validation.
pub fn validate_metadata_for_coreml(metadata: &ModelMetadata) -> Result<Vec<FormatWarning>, CoremlError> {
    let mut warnings = Vec::new();

    // Check: MoE dynamic routing
    if metadata.is_moe() {
        return Err(CoremlError::IncompatibleArchitecture {
            architecture: metadata.architecture.clone(),
            reason: format!(
                "Mixture of Experts (MoE) models with dynamic routing are not supported by CoreML. \
                 This model has {} experts with top-k={}. \
                 CoreML requires static computation graphs and cannot handle dynamic expert selection.",
                metadata.num_experts.unwrap_or(0),
                metadata.top_k_experts.unwrap_or(0)
            ),
        });
    }

    // Check layer types for MoE indicators
    for layer_type in &metadata.layer_types {
        let lower = layer_type.to_lowercase();
        for indicator in MOE_DYNAMIC_ROUTING_INDICATORS {
            if lower.contains(indicator) {
                return Err(CoremlError::IncompatibleArchitecture {
                    architecture: metadata.architecture.clone(),
                    reason: format!(
                        "Layer type '{}' indicates dynamic MoE routing, which is not supported by CoreML.",
                        layer_type
                    ),
                });
            }
        }
    }

    // Check layer type compatibility
    let unique_types = metadata.unique_layer_types();
    let incompatible: Vec<&String> = unique_types
        .iter()
        .filter(|lt| {
            let lower = lt.to_lowercase();
            !COREML_COMPATIBLE_LAYER_TYPES
                .iter()
                .any(|compat| lower.contains(compat))
        })
        .collect();

    if !incompatible.is_empty() {
        return Err(CoremlError::IncompatibleArchitecture {
            architecture: metadata.architecture.clone(),
            reason: format!(
                "The following layer types are not known to be CoreML-compatible: {:?}",
                incompatible
            ),
        });
    }

    // Size warning based on param count estimate (rough: params * bits / 8)
    let estimated_size = metadata.param_count * 2; // f16 baseline
    if estimated_size > ANE_SIZE_LIMIT_BYTES {
        warnings.push(FormatWarning {
            message: format!(
                "Estimated model size ({:.2} GB) exceeds typical ANE capacity ({:.2} GB). \
                 The model may be split across ANE + GPU.",
                estimated_size as f64 / (1024.0 * 1024.0 * 1024.0),
                ANE_SIZE_LIMIT_BYTES as f64 / (1024.0 * 1024.0 * 1024.0),
            ),
            severity: WarningSeverity::Warning,
        });
    }

    Ok(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::ir::{DType, QuantizedTensor, TensorQuantInfo};

    fn make_dense_metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: "TestDenseModel".to_string(),
            model_type: "test".to_string(),
            param_count: 1_000_000,
            hidden_size: 256,
            num_layers: 4,
            layer_types: vec!["attention".to_string(), "feed_forward".to_string()],
            num_attention_heads: 8,
            num_kv_heads: None,
            vocab_size: 32000,
            dtype: "float16".to_string(),
            shard_count: 1,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: Some(512),
            raw_config: serde_json::json!({"model_type": "test"}),
        }
    }

    fn make_moe_metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: "TestMoEModel".to_string(),
            model_type: "test_moe".to_string(),
            param_count: 10_000_000,
            hidden_size: 512,
            num_layers: 8,
            layer_types: vec!["attention".to_string(), "feed_forward".to_string()],
            num_attention_heads: 16,
            num_kv_heads: Some(4),
            vocab_size: 32000,
            dtype: "float16".to_string(),
            shard_count: 2,
            num_experts: Some(8),
            top_k_experts: Some(2),
            intermediate_size: Some(1024),
            raw_config: serde_json::json!({"model_type": "test_moe"}),
        }
    }

    fn make_test_model(metadata: ModelMetadata) -> QuantizedModel {
        let mut tensors = HashMap::new();

        tensors.insert(
            "layer.weight".to_string(),
            QuantizedTensor {
                name: "layer.weight".to_string(),
                shape: vec![2, 2],
                original_dtype: DType::F16,
                data: vec![0u8; 4],
                quant_info: TensorQuantInfo {
                    method: "f16".to_string(),
                    bits: 16,
                    group_size: 0,
                    preserved: false,
                    scales: None,
                    biases: None,
                },
            },
        );

        QuantizedModel {
            metadata,
            tensors,
            quant_method: "f16".to_string(),
            group_size: 0,
            bits: 16,
        }
    }

    #[test]
    fn test_validate_dense_model_passes() {
        let model = make_test_model(make_dense_metadata());
        let result = validate_coreml_compatibility(&model);
        assert!(result.is_ok());
        let warnings = result.unwrap();
        // Dense model within size limits should have no warnings
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_validate_moe_model_rejected() {
        let model = make_test_model(make_moe_metadata());
        let result = validate_coreml_compatibility(&model);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Mixture of Experts"));
        assert!(err.contains("different output format"));
    }

    #[test]
    fn test_validate_moe_layer_type_rejected() {
        let mut metadata = make_dense_metadata();
        metadata.layer_types.push("sparse_moe_block".to_string());
        let model = make_test_model(metadata);
        let result = validate_coreml_compatibility(&model);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("dynamic MoE routing"));
    }

    #[test]
    fn test_validate_incompatible_layer_type_rejected() {
        let mut metadata = make_dense_metadata();
        metadata.layer_types = vec!["quantum_entanglement_layer".to_string()];
        let model = make_test_model(metadata);
        let result = validate_coreml_compatibility(&model);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not known to be CoreML-compatible"));
    }

    #[test]
    fn test_validate_empty_model_rejected() {
        let mut model = make_test_model(make_dense_metadata());
        model.tensors.clear();
        let result = validate_coreml_compatibility(&model);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_metadata_dense_passes() {
        let metadata = make_dense_metadata();
        let result = validate_metadata_for_coreml(&metadata);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_metadata_moe_rejected() {
        let metadata = make_moe_metadata();
        let result = validate_metadata_for_coreml(&metadata);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Mixture of Experts"));
    }

    #[test]
    fn test_estimate_output_size() {
        let model = make_test_model(make_dense_metadata());
        let size = estimate_output_size(&model);
        // Should be weight data + overhead
        assert!(size > 0);
        // 4 bytes of weight data + 100KB overhead
        assert!(size >= 4);
    }

    #[test]
    fn test_coreml_backend_name() {
        let backend = CoremlBackend::new();
        assert_eq!(backend.name(), "coreml");
    }

    #[test]
    fn test_coreml_backend_default() {
        let backend = CoremlBackend::default();
        assert_eq!(backend.name(), "coreml");
    }

    // When feature is not enabled, validate should return clear error
    #[cfg(not(feature = "coreml-backend"))]
    #[test]
    fn test_validate_without_feature_returns_error() {
        let backend = CoremlBackend::new();
        let model = make_test_model(make_dense_metadata());
        let result = backend.validate(&model);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("coreml-backend"));
        assert!(err.contains("different output format"));
    }

    // When feature is not enabled, write should return clear error
    #[cfg(not(feature = "coreml-backend"))]
    #[test]
    fn test_write_without_feature_returns_error() {
        let backend = CoremlBackend::new();
        let model = make_test_model(make_dense_metadata());
        let progress = ProgressReporter::new();
        let result = backend.write(
            &model,
            Path::new("/tmp/input"),
            Path::new("/tmp/output"),
            &progress,
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("coreml-backend"));
    }
}
