//! JSON report generation and terminal summary (Epic 4).
//!
//! Report structure is stable for CI pipeline consumption.

use thiserror::Error;

/// Errors from report generation.
#[derive(Error, Debug)]
pub enum ReportError {
    #[error("Report generation failed: {reason}")]
    GenerationFailed { reason: String },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Full conversion report for JSON output.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConversionReport {
    /// hf2q version
    pub version: String,
    /// Input model path or repo
    pub input: String,
    /// Output directory
    pub output: String,
    /// Model metadata
    pub model: ModelSummary,
    /// Quantization configuration
    pub quantization: QuantSummary,
    /// Output file listing
    pub output_files: Vec<FileSummary>,
    /// Total output size
    pub total_output_bytes: u64,
    /// Elapsed time in seconds
    pub elapsed_seconds: f64,
}

/// Model summary for reports.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelSummary {
    pub architecture: String,
    pub model_type: String,
    pub param_count: u64,
    pub num_layers: u32,
    pub dtype: String,
}

/// Quantization summary for reports.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantSummary {
    pub method: String,
    pub bits: u8,
    pub group_size: usize,
}

/// File summary for reports.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileSummary {
    pub filename: String,
    pub size_bytes: u64,
}
