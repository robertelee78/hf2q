//! Pre-conversion validation (Epic 2, Story 2.1).
//!
//! Runs before any expensive work to catch configuration issues early.
//! All checks complete in under 5 seconds for any model.
//! Produces clear errors with actionable guidance on failure.

use std::path::Path;

use thiserror::Error;
use tracing::{debug, info};

use crate::cli::{ConvertConfig, OutputFormat, QuantMethod, UnsupportedLayerPolicy};
use crate::ir::ModelMetadata;

/// Errors from preflight validation.
#[derive(Error, Debug)]
pub enum PreflightError {
    #[error(
        "Unsupported layer types found: {layers:?}\n\
         \n\
         These layer types are not supported by the '{quant_method}' quantizer.\n\
         Affected layer indices: {indices:?}\n\
         \n\
         Options:\n\
         - Use --unsupported-layers=passthrough to pass these layers through at f16\n\
         - Use --quant f16 which handles all layer types"
    )]
    UnsupportedLayers {
        layers: Vec<String>,
        indices: Vec<usize>,
        quant_method: String,
    },

    #[error(
        "Output format '{format}' is not compatible with model architecture '{architecture}': {reason}\n\
         \n\
         Suggestion: {suggestion}"
    )]
    IncompatibleFormat {
        format: String,
        architecture: String,
        reason: String,
        suggestion: String,
    },

    #[error(
        "Sensitive layer range exceeds model layer count.\n\
         Specified range: {range}\n\
         Model has {max_layers} layers (valid range: 0-{max_index}).\n\
         \n\
         Fix: adjust --sensitive-layers to be within 0-{max_index}"
    )]
    InvalidLayerRange {
        range: String,
        max_layers: u32,
        max_index: u32,
    },

    #[error(
        "Insufficient disk space for conversion output.\n\
         Estimated output size: {needed_display}\n\
         Available disk space:  {available_display}\n\
         \n\
         Free up at least {shortfall_display} of disk space, or use a different --output directory."
    )]
    InsufficientDisk {
        needed_bytes: u64,
        available_bytes: u64,
        needed_display: String,
        available_display: String,
        shortfall_display: String,
    },

    #[error(
        "No safetensors files found in input directory: {path}\n\
         \n\
         Ensure the directory contains .safetensors files or a model.safetensors.index.json."
    )]
    NoSafetensorsFiles { path: String },

    #[error(
        "Input directory does not exist: {path}\n\
         \n\
         Provide a valid local model directory with --input, or use --repo to download from HuggingFace."
    )]
    InputNotFound { path: String },

    #[error(
        "No config.json found in input directory: {path}\n\
         \n\
         A HuggingFace model directory must contain config.json. Is this the correct path?"
    )]
    NoConfigJson { path: String },

    #[error(
        "Output directory already exists and is not empty: {path}\n\
         \n\
         Use a different --output path, or remove the existing directory first."
    )]
    OutputDirExists { path: String },
}

/// Result of a successful preflight validation.
#[derive(Debug)]
pub struct PreflightReport {
    /// Layer types that will be passed through at f16 (if --unsupported-layers=passthrough)
    pub passthrough_layers: Vec<PassthroughLayer>,
    /// Estimated output size in bytes
    pub estimated_output_bytes: u64,
    /// Available disk space in bytes
    pub available_disk_bytes: u64,
    /// Warnings (non-fatal)
    pub warnings: Vec<String>,
}

/// A layer that will be passed through at f16 instead of being quantized.
#[derive(Debug, Clone)]
pub struct PassthroughLayer {
    /// Layer type name
    pub layer_type: String,
    /// Index of the layer in the model
    pub layer_index: usize,
}

/// Run all preflight validation checks before conversion begins.
///
/// This must complete in under 5 seconds for any model.
/// Returns a PreflightReport on success, or a PreflightError with actionable guidance.
pub fn validate(
    config: &ConvertConfig,
    metadata: &ModelMetadata,
) -> Result<PreflightReport, PreflightError> {
    info!("Running preflight validation");
    let mut warnings = Vec::new();
    let mut passthrough_layers = Vec::new();

    // Check 1: Input directory exists and has required files
    validate_input_dir(&config.input_dir)?;

    // Check 2: Validate layer types are supported for chosen quantizer
    let unsupported = find_unsupported_layers(metadata, &config.quant);
    if !unsupported.is_empty() {
        match config.unsupported_layers {
            Some(UnsupportedLayerPolicy::Passthrough) => {
                // User opted in — record passthrough layers and warn
                for (idx, layer_type) in &unsupported {
                    passthrough_layers.push(PassthroughLayer {
                        layer_type: layer_type.clone(),
                        layer_index: *idx,
                    });
                    warnings.push(format!(
                        "Layer {} (type '{}') will be passed through at f16",
                        idx, layer_type
                    ));
                }
            }
            None => {
                let layers: Vec<String> = unsupported.iter().map(|(_, t)| t.clone()).collect();
                let indices: Vec<usize> = unsupported.iter().map(|(i, _)| *i).collect();
                let mut unique_layers = layers.clone();
                unique_layers.sort();
                unique_layers.dedup();
                return Err(PreflightError::UnsupportedLayers {
                    layers: unique_layers,
                    indices,
                    quant_method: config.quant.to_string(),
                });
            }
        }
    }

    // Check 3: Output format compatible with model architecture
    validate_format_compatibility(config.format, metadata)?;

    // Check 4: Sensitive layers range valid for model
    validate_sensitive_layers(&config.sensitive_layers, metadata)?;

    // Check 5: Output directory does not already exist with content
    validate_output_dir(&config.output_dir, &mut warnings)?;

    // Check 6: Disk space sufficient
    let estimated_output_bytes = estimate_output_size(metadata, &config.quant, config.bits);
    let available_disk_bytes = get_available_disk_space(&config.output_dir);

    if available_disk_bytes > 0 && estimated_output_bytes > available_disk_bytes {
        return Err(PreflightError::InsufficientDisk {
            needed_bytes: estimated_output_bytes,
            available_bytes: available_disk_bytes,
            needed_display: crate::progress::format_bytes(estimated_output_bytes),
            available_display: crate::progress::format_bytes(available_disk_bytes),
            shortfall_display: crate::progress::format_bytes(
                estimated_output_bytes - available_disk_bytes,
            ),
        });
    }

    debug!(
        estimated_output_mb = estimated_output_bytes / (1024 * 1024),
        available_disk_mb = available_disk_bytes / (1024 * 1024),
        passthrough_count = passthrough_layers.len(),
        warning_count = warnings.len(),
        "Preflight validation passed"
    );

    Ok(PreflightReport {
        passthrough_layers,
        estimated_output_bytes,
        available_disk_bytes,
        warnings,
    })
}

/// Check that the input directory exists and contains required files.
fn validate_input_dir(input_dir: &Path) -> Result<(), PreflightError> {
    if !input_dir.exists() {
        return Err(PreflightError::InputNotFound {
            path: input_dir.display().to_string(),
        });
    }

    if !input_dir.join("config.json").exists() {
        return Err(PreflightError::NoConfigJson {
            path: input_dir.display().to_string(),
        });
    }

    // Check for safetensors files
    let has_safetensors = input_dir.join("model.safetensors").exists()
        || input_dir.join("model.safetensors.index.json").exists()
        || std::fs::read_dir(input_dir)
            .map(|entries| {
                entries.filter_map(|e| e.ok()).any(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "safetensors")
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

    if !has_safetensors {
        return Err(PreflightError::NoSafetensorsFiles {
            path: input_dir.display().to_string(),
        });
    }

    Ok(())
}

/// Known supported layer types for quantization.
/// These are standard transformer layer types that quantizers know how to handle.
const SUPPORTED_LAYER_TYPES: &[&str] = &[
    "attention",
    "full_attention",
    "sliding_attention",
    "moe_attention",
    "linear",
    "mlp",
    "feedforward",
    "ffn",
    "dense",
    "self_attention",
    "cross_attention",
    "grouped_query_attention",
];

/// Find layer types in the model that are not supported by the chosen quantizer.
/// Returns (layer_index, layer_type) pairs for unsupported layers.
fn find_unsupported_layers(
    metadata: &ModelMetadata,
    quant: &QuantMethod,
) -> Vec<(usize, String)> {
    // f16 mode handles all layer types (it's just dtype conversion)
    if *quant == QuantMethod::F16 {
        return Vec::new();
    }

    let mut unsupported = Vec::new();

    for (idx, layer_type) in metadata.layer_types.iter().enumerate() {
        let normalized = layer_type.to_lowercase();
        let is_supported = SUPPORTED_LAYER_TYPES
            .iter()
            .any(|supported| normalized.contains(supported));

        if !is_supported && !normalized.is_empty() {
            unsupported.push((idx, layer_type.clone()));
        }
    }

    unsupported
}

/// Validate that the output format is compatible with the model architecture.
fn validate_format_compatibility(
    format: OutputFormat,
    metadata: &ModelMetadata,
) -> Result<(), PreflightError> {
    match format {
        OutputFormat::Coreml => {
            // CoreML does not support MoE dynamic routing
            if metadata.is_moe() {
                return Err(PreflightError::IncompatibleFormat {
                    format: "coreml".to_string(),
                    architecture: metadata.architecture.clone(),
                    reason: "CoreML does not support Mixture of Experts (MoE) models with dynamic routing".to_string(),
                    suggestion: "Use --format mlx instead, which supports MoE architectures.".to_string(),
                });
            }
        }
        OutputFormat::Mlx => {
            // MLX supports all architectures
        }
        _ => {
            // Future formats — validated at CLI level as not-yet-implemented
        }
    }

    Ok(())
}

/// Validate that sensitive layer ranges are within the model's layer count.
fn validate_sensitive_layers(
    ranges: &[std::ops::RangeInclusive<usize>],
    metadata: &ModelMetadata,
) -> Result<(), PreflightError> {
    if ranges.is_empty() || metadata.num_layers == 0 {
        return Ok(());
    }

    let max_layer = (metadata.num_layers as usize).saturating_sub(1);

    for range in ranges {
        if *range.end() > max_layer {
            return Err(PreflightError::InvalidLayerRange {
                range: format!("{}-{}", range.start(), range.end()),
                max_layers: metadata.num_layers,
                max_index: metadata.num_layers.saturating_sub(1),
            });
        }
    }

    Ok(())
}

/// Validate the output directory does not already exist with content.
fn validate_output_dir(output_dir: &Path, warnings: &mut Vec<String>) -> Result<(), PreflightError> {
    if output_dir.exists() {
        // Check if directory is non-empty
        let is_empty = std::fs::read_dir(output_dir)
            .map(|mut entries| entries.next().is_none())
            .unwrap_or(false);

        if !is_empty {
            return Err(PreflightError::OutputDirExists {
                path: output_dir.display().to_string(),
            });
        }

        warnings.push(format!(
            "Output directory '{}' already exists (but is empty)",
            output_dir.display()
        ));
    }

    Ok(())
}

/// Estimate the output size based on model metadata and quantization method.
fn estimate_output_size(metadata: &ModelMetadata, quant: &QuantMethod, bits: Option<u8>) -> u64 {
    // Rough estimation: param_count * bytes_per_param
    // Plus overhead for config, tokenizer, metadata files (~10MB)
    let overhead_bytes: u64 = 10 * 1024 * 1024;

    if metadata.param_count == 0 {
        return overhead_bytes;
    }

    let bits_per_param: f64 = match quant {
        QuantMethod::F16 => 16.0,
        QuantMethod::Q8 => {
            let b = bits.unwrap_or(8) as f64;
            // Scale factors add ~10% overhead
            b * 1.1
        }
        QuantMethod::Q4 => {
            let b = bits.unwrap_or(4) as f64;
            b * 1.1
        }
        QuantMethod::Q2 => {
            let b = bits.unwrap_or(2) as f64;
            b * 1.1
        }
        QuantMethod::Mixed26 | QuantMethod::Mixed36 | QuantMethod::Mixed46 => {
            // Average of low and high bits
            5.0 * 1.1
        }
        QuantMethod::DwqMixed46 => 5.0 * 1.1,
        QuantMethod::Q4Mxfp => 4.5 * 1.1,
        QuantMethod::Auto => 4.0 * 1.1, // Conservative estimate
    };

    let estimated_bytes = (metadata.param_count as f64 * bits_per_param / 8.0) as u64;
    estimated_bytes + overhead_bytes
}

/// Get available disk space at the target path (or its parent if it doesn't exist yet).
fn get_available_disk_space(target_path: &Path) -> u64 {
    // Find an existing ancestor directory to check disk space
    let check_path = if target_path.exists() {
        target_path.to_path_buf()
    } else if let Some(parent) = target_path.parent() {
        if parent.exists() {
            parent.to_path_buf()
        } else {
            // Fall back to current directory
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"))
        }
    } else {
        std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"))
    };

    // Use sysinfo to get disk space
    use sysinfo::Disks;
    let disks = Disks::new_with_refreshed_list();

    // Find the disk that contains our target path
    let mut best_match: Option<(usize, u64)> = None;
    for disk in disks.list() {
        let mount = disk.mount_point();
        if check_path.starts_with(mount) {
            let mount_len = mount.as_os_str().len();
            match best_match {
                Some((prev_len, _)) if mount_len > prev_len => {
                    best_match = Some((mount_len, disk.available_space()));
                }
                None => {
                    best_match = Some((mount_len, disk.available_space()));
                }
                _ => {}
            }
        }
    }

    best_match.map(|(_, space)| space).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ModelMetadata;

    fn make_test_metadata(num_layers: u32, layer_types: Vec<String>) -> ModelMetadata {
        ModelMetadata {
            architecture: "TestModel".to_string(),
            model_type: "test".to_string(),
            param_count: 1_000_000,
            hidden_size: 256,
            num_layers,
            layer_types,
            num_attention_heads: 8,
            num_kv_heads: Some(4),
            vocab_size: 32000,
            dtype: "float16".to_string(),
            shard_count: 1,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: Some(512),
            raw_config: serde_json::Value::Null,
        }
    }

    #[test]
    fn test_supported_layer_types_pass() {
        let metadata = make_test_metadata(
            4,
            vec![
                "attention".to_string(),
                "full_attention".to_string(),
                "sliding_attention".to_string(),
                "attention".to_string(),
            ],
        );
        let unsupported = find_unsupported_layers(&metadata, &QuantMethod::Q4);
        assert!(unsupported.is_empty());
    }

    #[test]
    fn test_unsupported_layer_types_detected() {
        let metadata = make_test_metadata(
            3,
            vec![
                "attention".to_string(),
                "novel_quantum_layer".to_string(),
                "attention".to_string(),
            ],
        );
        let unsupported = find_unsupported_layers(&metadata, &QuantMethod::Q4);
        assert_eq!(unsupported.len(), 1);
        assert_eq!(unsupported[0].0, 1);
        assert_eq!(unsupported[0].1, "novel_quantum_layer");
    }

    #[test]
    fn test_f16_supports_all_layers() {
        let metadata = make_test_metadata(
            2,
            vec![
                "exotic_layer".to_string(),
                "unknown_thing".to_string(),
            ],
        );
        let unsupported = find_unsupported_layers(&metadata, &QuantMethod::F16);
        assert!(unsupported.is_empty());
    }

    #[test]
    fn test_sensitive_layers_valid() {
        let metadata = make_test_metadata(32, vec!["attention".to_string(); 32]);
        let ranges = vec![13..=24];
        assert!(validate_sensitive_layers(&ranges, &metadata).is_ok());
    }

    #[test]
    fn test_sensitive_layers_out_of_range() {
        let metadata = make_test_metadata(10, vec!["attention".to_string(); 10]);
        let ranges = vec![5..=15]; // layer 15 > max layer 9
        let result = validate_sensitive_layers(&ranges, &metadata);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("exceeds model layer count"));
    }

    #[test]
    fn test_coreml_moe_incompatible() {
        let mut metadata = make_test_metadata(4, vec!["attention".to_string(); 4]);
        metadata.num_experts = Some(128);
        metadata.top_k_experts = Some(8);

        let result = validate_format_compatibility(OutputFormat::Coreml, &metadata);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("MoE"));
    }

    #[test]
    fn test_mlx_supports_moe() {
        let mut metadata = make_test_metadata(4, vec!["attention".to_string(); 4]);
        metadata.num_experts = Some(128);
        metadata.top_k_experts = Some(8);

        assert!(validate_format_compatibility(OutputFormat::Mlx, &metadata).is_ok());
    }

    #[test]
    fn test_estimate_output_size_q4() {
        let metadata = make_test_metadata(32, vec!["attention".to_string(); 32]);
        // 1M params at ~4.4 bits = ~550KB + 10MB overhead
        let estimated = estimate_output_size(&metadata, &QuantMethod::Q4, None);
        assert!(estimated > 10 * 1024 * 1024); // At least the overhead
        assert!(estimated < 100 * 1024 * 1024); // Not unreasonably large for 1M params
    }

    #[test]
    fn test_estimate_output_size_f16() {
        let metadata = make_test_metadata(32, vec!["attention".to_string(); 32]);
        // 1M params at 16 bits = 2MB + 10MB overhead
        let estimated = estimate_output_size(&metadata, &QuantMethod::F16, None);
        // 1M params * 16 bits / 8 = 2MB + 10MB overhead ≈ 12MB
        assert!(estimated > 11 * 1024 * 1024);
    }

    #[test]
    fn test_get_available_disk_space_works() {
        // Just verify it doesn't panic and returns something reasonable
        let space = get_available_disk_space(std::path::Path::new("/tmp"));
        // On any real system, /tmp should have some space
        assert!(space > 0);
    }

    #[test]
    fn test_validate_input_dir_missing() {
        let result = validate_input_dir(Path::new("/definitely/nonexistent/path"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn test_validate_input_dir_no_config() {
        let tmp = tempfile::tempdir().unwrap();
        let result = validate_input_dir(tmp.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("config.json"));
    }

    #[test]
    fn test_validate_input_dir_no_safetensors() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("config.json"), "{}").unwrap();
        let result = validate_input_dir(tmp.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("safetensors"));
    }

    #[test]
    fn test_validate_input_dir_valid() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("config.json"), "{}").unwrap();
        std::fs::write(tmp.path().join("model.safetensors"), &[0u8; 16]).unwrap();
        assert!(validate_input_dir(tmp.path()).is_ok());
    }

    #[test]
    fn test_output_dir_nonempty_fails() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("something.txt"), "data").unwrap();
        let mut warnings = Vec::new();
        let result = validate_output_dir(tmp.path(), &mut warnings);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not empty"));
    }

    #[test]
    fn test_output_dir_empty_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let mut warnings = Vec::new();
        assert!(validate_output_dir(tmp.path(), &mut warnings).is_ok());
        assert!(!warnings.is_empty()); // warns that it exists
    }

    #[test]
    fn test_output_dir_nonexistent_ok() {
        let mut warnings = Vec::new();
        assert!(validate_output_dir(Path::new("/tmp/nonexistent_hf2q_test_dir"), &mut warnings).is_ok());
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_full_preflight_validation() {
        let tmp = tempfile::tempdir().unwrap();
        let input_dir = tmp.path().join("input");
        std::fs::create_dir_all(&input_dir).unwrap();
        std::fs::write(input_dir.join("config.json"), "{}").unwrap();
        std::fs::write(input_dir.join("model.safetensors"), &[0u8; 16]).unwrap();

        let output_dir = tmp.path().join("output_nonexistent");

        let metadata = make_test_metadata(
            4,
            vec!["attention".to_string(); 4],
        );

        let config = ConvertConfig {
            input_dir,
            format: OutputFormat::Mlx,
            quant: QuantMethod::Q4,
            sensitive_layers: vec![0..=3],
            calibration_samples: 1024,
            bits: None,
            group_size: 64,
            output_dir,
            json_report: false,
            skip_quality: false,
            dry_run: false,
            yes: false,
            unsupported_layers: None,
        };

        let report = validate(&config, &metadata).unwrap();
        assert!(report.passthrough_layers.is_empty());
        assert!(report.warnings.is_empty());
        assert!(report.estimated_output_bytes > 0);
    }

    #[test]
    fn test_preflight_with_passthrough() {
        let tmp = tempfile::tempdir().unwrap();
        let input_dir = tmp.path().join("input");
        std::fs::create_dir_all(&input_dir).unwrap();
        std::fs::write(input_dir.join("config.json"), "{}").unwrap();
        std::fs::write(input_dir.join("model.safetensors"), &[0u8; 16]).unwrap();

        let output_dir = tmp.path().join("output_nonexistent");

        let metadata = make_test_metadata(
            3,
            vec![
                "attention".to_string(),
                "exotic_layer".to_string(),
                "attention".to_string(),
            ],
        );

        let config = ConvertConfig {
            input_dir,
            format: OutputFormat::Mlx,
            quant: QuantMethod::Q4,
            sensitive_layers: Vec::new(),
            calibration_samples: 1024,
            bits: None,
            group_size: 64,
            output_dir,
            json_report: false,
            skip_quality: false,
            dry_run: false,
            yes: false,
            unsupported_layers: Some(UnsupportedLayerPolicy::Passthrough),
        };

        let report = validate(&config, &metadata).unwrap();
        assert_eq!(report.passthrough_layers.len(), 1);
        assert_eq!(report.passthrough_layers[0].layer_index, 1);
        assert_eq!(report.passthrough_layers[0].layer_type, "exotic_layer");
    }

    #[test]
    fn test_preflight_rejects_unsupported_without_flag() {
        let tmp = tempfile::tempdir().unwrap();
        let input_dir = tmp.path().join("input");
        std::fs::create_dir_all(&input_dir).unwrap();
        std::fs::write(input_dir.join("config.json"), "{}").unwrap();
        std::fs::write(input_dir.join("model.safetensors"), &[0u8; 16]).unwrap();

        let output_dir = tmp.path().join("output_nonexistent");

        let metadata = make_test_metadata(
            2,
            vec!["exotic_layer".to_string(), "attention".to_string()],
        );

        let config = ConvertConfig {
            input_dir,
            format: OutputFormat::Mlx,
            quant: QuantMethod::Q4,
            sensitive_layers: Vec::new(),
            calibration_samples: 1024,
            bits: None,
            group_size: 64,
            output_dir,
            json_report: false,
            skip_quality: false,
            dry_run: false,
            yes: false,
            unsupported_layers: None,
        };

        let result = validate(&config, &metadata);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("unsupported-layers=passthrough"));
    }
}
