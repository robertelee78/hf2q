//! Input module — owns all external model I/O.
//!
//! Nothing outside this module touches raw model files directly.
//! Sub-modules:
//! - `config_parser`: HF config.json -> ModelMetadata
//! - `safetensors`: Streaming mmap shard reader -> TensorMap
//! - `hf_download`: HF Hub download (Epic 3)

pub mod config_parser;
pub mod hf_download;
pub mod safetensors;

use std::path::Path;

use thiserror::Error;

use crate::ir::{ModelMetadata, TensorMap};
use crate::progress::ProgressReporter;

/// Errors from input operations.
#[derive(Error, Debug)]
pub enum InputError {
    #[error("Input directory does not exist: {path}")]
    DirectoryNotFound { path: String },

    #[error("No config.json found in {path}")]
    NoConfig { path: String },

    #[allow(dead_code)]
    #[error("No safetensors files found in {path}")]
    NoSafetensors { path: String },

    #[error("Config parse error: {0}")]
    ConfigParse(#[from] config_parser::ConfigParseError),

    #[error("Safetensors read error: {0}")]
    SafetensorsRead(#[from] safetensors::SafetensorsError),

    #[error("HF download error: {0}")]
    HfDownload(#[from] hf_download::DownloadError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Read a model from a local directory: parse config and load tensor map.
pub fn read_model(
    input_dir: &Path,
    progress: &ProgressReporter,
) -> Result<(ModelMetadata, TensorMap), InputError> {
    if !input_dir.exists() {
        return Err(InputError::DirectoryNotFound {
            path: input_dir.display().to_string(),
        });
    }

    let config_path = input_dir.join("config.json");
    if !config_path.exists() {
        return Err(InputError::NoConfig {
            path: input_dir.display().to_string(),
        });
    }

    // Parse model metadata from config.json
    let metadata = config_parser::parse_config(&config_path)?;

    // Load tensors from safetensors files
    let tensor_map = safetensors::read_tensors(input_dir, progress)?;

    Ok((metadata, tensor_map))
}
