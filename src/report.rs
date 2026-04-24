//! JSON report generation and terminal summary.
//!
//! Produces a stable, documented JSON schema for CI pipeline consumption.
//!
//! ## JSON Report Schema (v1)
//!
//! ```json
//! {
//!   "schema_version": "1",
//!   "version": "0.1.0",
//!   "input": "/path/to/model",
//!   "output": "/path/to/output",
//!   "model": {
//!     "architecture": "Gemma4ForConditionalGeneration",
//!     "model_type": "gemma4",
//!     "param_count": 25805933872,
//!     "num_layers": 30,
//!     "hidden_size": 2816,
//!     "dtype": "bfloat16"
//!   },
//!   "hardware": {
//!     "chip_model": "Apple M5 Max",
//!     "total_memory_bytes": 137438953472,
//!     "available_memory_bytes": 107374182400,
//!     "total_cores": 18
//!   },
//!   "quantization": {
//!     "method": "q4",
//!     "bits": 4,
//!     "group_size": 64,
//!     "per_layer_bits": { "model.layers.0.self_attn.q_proj.weight": 4 }
//!   },
//!   "quality": {
//!     "kl_divergence": 0.0023,
//!     "per_layer_kl": [0.001, 0.002, ...],
//!     "perplexity_pre": 5.23,
//!     "perplexity_post": 5.41,
//!     "perplexity_delta": 0.18,
//!     "cosine_sim_average": 0.9987,
//!     "per_layer_cosine_sim": [0.999, 0.998, ...]
//!   },
//!   "output_files": [
//!     { "filename": "model-00001-of-00004.safetensors", "size_bytes": 1234567 }
//!   ],
//!   "total_output_bytes": 12345678,
//!   "input_size_bytes": 54760833024,
//!   "compression_ratio": 4.2,
//!   "timing": {
//!     "total_seconds": 123.4,
//!     "phases": {
//!       "read": 10.2,
//!       "quantize": 45.6,
//!       "quality": 30.1,
//!       "write": 37.5
//!     }
//!   }
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ir::{ModelMetadata, OutputManifest};
use crate::quality::regression::QualityGate;
use crate::quality::QualityReport;

/// Errors from report generation.
#[derive(Error, Debug)]
pub enum ReportError {
    #[error("Report generation failed: {reason}")]
    GenerationFailed { reason: String },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Current schema version for the JSON report.
pub const SCHEMA_VERSION: &str = "1";

/// Full conversion report for JSON output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionReport {
    /// Schema version for forward compatibility
    pub schema_version: String,
    /// hf2q version
    pub version: String,
    /// Input model path or repo
    pub input: String,
    /// Output directory
    pub output: String,
    /// Model metadata
    pub model: ModelSummary,
    /// Hardware profile (optional — populated when detected)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware: Option<HardwareSummary>,
    /// Quantization configuration
    pub quantization: QuantSummary,
    /// Quality metrics (optional — None when --skip-quality)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<QualitySummary>,
    /// Quality gate pass/fail for CI (optional — populated when quality measured)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_gate: Option<QualityGate>,
    /// Output file listing
    pub output_files: Vec<FileSummary>,
    /// Total output size in bytes
    pub total_output_bytes: u64,
    /// Input size in bytes
    pub input_size_bytes: u64,
    /// Compression ratio (input / output)
    pub compression_ratio: f64,
    /// Timing information
    pub timing: TimingSummary,
}

/// Model summary for reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    pub architecture: String,
    pub model_type: String,
    pub param_count: u64,
    pub num_layers: u32,
    pub hidden_size: u64,
    pub dtype: String,
    pub shard_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_experts: Option<u32>,
}

/// Hardware summary for reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSummary {
    pub chip_model: String,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub total_cores: u32,
}

/// Quantization summary for reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantSummary {
    pub method: String,
    pub bits: u8,
    pub group_size: usize,
    /// Per-layer bit allocation (layer name -> bits)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_layer_bits: Option<HashMap<String, u8>>,
}

/// Quality metrics summary for reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kl_divergence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_layer_kl: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perplexity_pre: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perplexity_post: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perplexity_delta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cosine_sim_average: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_layer_cosine_sim: Option<Vec<f64>>,
}

/// File summary for reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSummary {
    pub filename: String,
    pub size_bytes: u64,
}

/// Timing summary for reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingSummary {
    /// Total elapsed time in seconds
    pub total_seconds: f64,
    /// Per-phase timing (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phases: Option<HashMap<String, f64>>,
}

/// Builder for constructing a ConversionReport.
pub struct ReportBuilder {
    input: String,
    output: String,
    metadata: ModelMetadata,
    quant_method: String,
    bits: u8,
    group_size: usize,
    per_layer_bits: Option<HashMap<String, u8>>,
    quality: Option<QualityReport>,
    quality_gate: Option<QualityGate>,
    manifest: Option<OutputManifest>,
    input_size_bytes: u64,
    hardware: Option<HardwareSummary>,
    total_seconds: f64,
    phases: Option<HashMap<String, f64>>,
}

impl ReportBuilder {
    /// Create a new report builder.
    pub fn new(
        input: String,
        output: String,
        metadata: ModelMetadata,
        quant_method: String,
        bits: u8,
        group_size: usize,
    ) -> Self {
        Self {
            input,
            output,
            metadata,
            quant_method,
            bits,
            group_size,
            per_layer_bits: None,
            quality: None,
            quality_gate: None,
            manifest: None,
            input_size_bytes: 0,
            hardware: None,
            total_seconds: 0.0,
            phases: None,
        }
    }

    /// Set per-layer bit allocation.
    pub fn with_per_layer_bits(mut self, bits: HashMap<String, u8>) -> Self {
        self.per_layer_bits = Some(bits);
        self
    }

    /// Set quality metrics.
    pub fn with_quality(mut self, report: QualityReport) -> Self {
        self.quality = Some(report);
        self
    }

    /// Set quality gate result.
    pub fn with_quality_gate(mut self, gate: QualityGate) -> Self {
        self.quality_gate = Some(gate);
        self
    }

    /// Set output manifest.
    pub fn with_manifest(mut self, manifest: OutputManifest) -> Self {
        self.manifest = Some(manifest);
        self
    }

    /// Set input size.
    pub fn with_input_size(mut self, bytes: u64) -> Self {
        self.input_size_bytes = bytes;
        self
    }

    /// Set hardware profile.
    pub fn with_hardware(mut self, hw: HardwareSummary) -> Self {
        self.hardware = Some(hw);
        self
    }

    /// Set timing information.
    pub fn with_timing(mut self, total_seconds: f64, phases: Option<HashMap<String, f64>>) -> Self {
        self.total_seconds = total_seconds;
        self.phases = phases;
        self
    }

    /// Build the final report.
    pub fn build(self) -> ConversionReport {
        let output_files: Vec<FileSummary> = self
            .manifest
            .as_ref()
            .map(|m| {
                m.files
                    .iter()
                    .map(|f| FileSummary {
                        filename: f.filename.clone(),
                        size_bytes: f.size_bytes,
                    })
                    .collect()
            })
            .unwrap_or_default();

        let total_output_bytes = self
            .manifest
            .as_ref()
            .map(|m| m.total_size_bytes)
            .unwrap_or(0);

        let compression_ratio = if total_output_bytes > 0 {
            self.input_size_bytes as f64 / total_output_bytes as f64
        } else {
            0.0
        };

        let quality_summary = self.quality.as_ref().map(|q| QualitySummary {
            kl_divergence: q.kl_divergence,
            per_layer_kl: q.per_layer_kl.clone(),
            perplexity_pre: q.perplexity_pre,
            perplexity_post: q.perplexity_post,
            perplexity_delta: q.perplexity_delta,
            cosine_sim_average: q.cosine_sim_average,
            per_layer_cosine_sim: q.per_layer_cosine_sim.clone(),
        });

        ConversionReport {
            schema_version: SCHEMA_VERSION.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            input: self.input,
            output: self.output,
            model: ModelSummary {
                architecture: self.metadata.architecture.clone(),
                model_type: self.metadata.model_type.clone(),
                param_count: self.metadata.param_count,
                num_layers: self.metadata.num_layers,
                hidden_size: self.metadata.hidden_size,
                dtype: self.metadata.dtype.clone(),
                shard_count: self.metadata.shard_count,
                num_experts: self.metadata.num_experts,
            },
            hardware: self.hardware,
            quantization: QuantSummary {
                method: self.quant_method,
                bits: self.bits,
                group_size: self.group_size,
                per_layer_bits: self.per_layer_bits,
            },
            quality: quality_summary,
            quality_gate: self.quality_gate,
            output_files,
            total_output_bytes,
            input_size_bytes: self.input_size_bytes,
            compression_ratio,
            timing: TimingSummary {
                total_seconds: self.total_seconds,
                phases: self.phases,
            },
        }
    }
}

/// Serialize a report to a JSON string.
pub fn to_json(report: &ConversionReport) -> Result<String, ReportError> {
    serde_json::to_string_pretty(report).map_err(ReportError::Serialization)
}

/// Write a report to a file.
pub fn write_to_file(report: &ConversionReport, path: &std::path::Path) -> Result<(), ReportError> {
    let json = to_json(report)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Write a report to stdout (for --json-report --yes mode).
pub fn write_to_stdout(report: &ConversionReport) -> Result<(), ReportError> {
    let json = to_json(report)?;
    println!("{}", json);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: "TestModel".to_string(),
            model_type: "test".to_string(),
            param_count: 1_000_000,
            hidden_size: 256,
            num_layers: 4,
            layer_types: vec!["attention".to_string(); 4],
            num_attention_heads: 8,
            num_kv_heads: Some(4),
            vocab_size: 32000,
            dtype: "float16".to_string(),
            shard_count: 1,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: Some(512),
            raw_config: serde_json::Value::Null,
            explicit_layer_types: None,
            full_attention_interval: None,
            attn_output_gate: None,
            head_dim: None,
            partial_rotary_factor: None,
            rope_parameters: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_num_key_heads: None,
            linear_value_head_dim: None,
            linear_num_value_heads: None,
            mamba_ssm_dtype: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            output_router_logits: None,
            router_aux_loss_coef: None,
        }
    }

    #[test]
    fn test_report_builder_basic() {
        let report = ReportBuilder::new(
            "/input".to_string(),
            "/output".to_string(),
            make_metadata(),
            "q4".to_string(),
            4,
            64,
        )
        .with_input_size(2_000_000)
        .with_timing(10.5, None)
        .build();

        assert_eq!(report.schema_version, "1");
        assert_eq!(report.input, "/input");
        assert_eq!(report.output, "/output");
        assert_eq!(report.model.architecture, "TestModel");
        assert_eq!(report.quantization.method, "q4");
        assert_eq!(report.quantization.bits, 4);
        assert_eq!(report.timing.total_seconds, 10.5);
        assert!(report.quality.is_none()); // Not set
    }

    #[test]
    fn test_report_with_quality() {
        let quality = QualityReport {
            kl_divergence: Some(0.0023),
            per_layer_kl: Some(vec![0.001, 0.002, 0.003, 0.004]),
            perplexity_pre: Some(5.23),
            perplexity_post: Some(5.41),
            perplexity_delta: Some(0.18),
            per_layer_cosine_sim: Some(vec![0.999, 0.998, 0.997, 0.996]),
            cosine_sim_average: Some(0.9975),
            cosine_sim_min: Some(0.996),
            cosine_sim_min_layer: Some(3),
        };

        let report = ReportBuilder::new(
            "/input".to_string(),
            "/output".to_string(),
            make_metadata(),
            "q4".to_string(),
            4,
            64,
        )
        .with_quality(quality)
        .build();

        assert!(report.quality.is_some());
        let q = report.quality.as_ref().unwrap();
        assert_eq!(q.kl_divergence, Some(0.0023));
        assert_eq!(q.perplexity_delta, Some(0.18));
    }

    #[test]
    fn test_report_serialization() {
        let report = ReportBuilder::new(
            "/input".to_string(),
            "/output".to_string(),
            make_metadata(),
            "q4".to_string(),
            4,
            64,
        )
        .build();

        let json = to_json(&report).unwrap();
        assert!(json.contains("schema_version"));
        assert!(json.contains("TestModel"));
        assert!(json.contains("\"method\": \"q4\""));

        // Verify it can be deserialized
        let parsed: ConversionReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model.architecture, "TestModel");
    }

    #[test]
    fn test_report_compression_ratio() {
        let manifest = OutputManifest {
            output_dir: "/output".to_string(),
            files: vec![crate::ir::OutputFile {
                filename: "model.safetensors".to_string(),
                size_bytes: 500_000,
            }],
            total_size_bytes: 500_000,
            shard_count: 1,
        };

        let report = ReportBuilder::new(
            "/input".to_string(),
            "/output".to_string(),
            make_metadata(),
            "q4".to_string(),
            4,
            64,
        )
        .with_input_size(2_000_000)
        .with_manifest(manifest)
        .build();

        assert!((report.compression_ratio - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_report_with_per_layer_bits() {
        let mut per_layer = HashMap::new();
        per_layer.insert("layer.0.weight".to_string(), 4);
        per_layer.insert("layer.1.weight".to_string(), 6);

        let report = ReportBuilder::new(
            "/input".to_string(),
            "/output".to_string(),
            make_metadata(),
            "mixed-4-6".to_string(),
            4,
            64,
        )
        .with_per_layer_bits(per_layer)
        .build();

        let bits = report.quantization.per_layer_bits.as_ref().unwrap();
        assert_eq!(*bits.get("layer.0.weight").unwrap(), 4);
        assert_eq!(*bits.get("layer.1.weight").unwrap(), 6);
    }

    #[test]
    fn test_report_with_hardware() {
        let report = ReportBuilder::new(
            "/input".to_string(),
            "/output".to_string(),
            make_metadata(),
            "q4".to_string(),
            4,
            64,
        )
        .with_hardware(HardwareSummary {
            chip_model: "Apple M5 Max".to_string(),
            total_memory_bytes: 128 * 1024 * 1024 * 1024,
            available_memory_bytes: 100 * 1024 * 1024 * 1024,
            total_cores: 18,
        })
        .build();

        assert!(report.hardware.is_some());
        assert_eq!(report.hardware.unwrap().chip_model, "Apple M5 Max");
    }

    #[test]
    fn test_write_to_file() {
        let report = ReportBuilder::new(
            "/input".to_string(),
            "/output".to_string(),
            make_metadata(),
            "q4".to_string(),
            4,
            64,
        )
        .build();

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("report.json");

        write_to_file(&report, &path).unwrap();
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: ConversionReport = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed.model.architecture, "TestModel");
    }
}
