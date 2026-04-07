//! Weight loading for inference — loads safetensors into Metal buffers via mlx-native.
//!
//! This module bridges hf2q's model loading with mlx-native's Metal buffer
//! infrastructure. It:
//! 1. Discovers safetensors files in the model directory
//! 2. Parses quantization_config.json for per-tensor quant metadata
//! 3. Loads each tensor into an `MlxBuffer` via mlx-native
//! 4. Builds a `WeightMap` keyed by tensor name

use std::collections::HashMap;
use std::path::Path;

use mlx_native::{DType, MlxBuffer, MlxDevice, QuantizationConfig, SafetensorsFile};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors from weight loading.
#[derive(Error, Debug)]
pub enum WeightLoadError {
    #[error("No safetensors files found in {path}")]
    NoSafetensors { path: String },

    #[error("Failed to open safetensors file '{path}': {reason}")]
    SafetensorsOpen { path: String, reason: String },

    #[error("Failed to load tensor '{tensor}': {reason}")]
    TensorLoad { tensor: String, reason: String },

    #[error("Failed to read quantization config: {reason}")]
    QuantConfig { reason: String },

    #[error("Metal device error: {reason}")]
    DeviceError { reason: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Per-tensor quantization metadata attached to loaded weights.
#[derive(Debug, Clone)]
pub struct TensorQuantMeta {
    /// Bit-width for this tensor (3, 4, 6, 8, or 16 for unquantized).
    pub bits: u8,
    /// Quantization group size.
    pub group_size: usize,
}

/// A loaded weight tensor with its Metal buffer and metadata.
#[derive(Debug)]
pub struct LoadedWeight {
    /// The Metal buffer containing the tensor data.
    pub buffer: MlxBuffer,
    /// Element data type.
    pub dtype: DType,
    /// Tensor shape (dimensions).
    pub shape: Vec<usize>,
    /// Quantization metadata (if the model has quantization_config.json).
    pub quant_meta: Option<TensorQuantMeta>,
}

/// Map of tensor names to loaded Metal buffers with metadata.
pub struct WeightMap {
    /// Tensors keyed by their full name path.
    pub weights: HashMap<String, LoadedWeight>,
}

impl std::fmt::Debug for WeightMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightMap")
            .field("tensor_count", &self.weights.len())
            .field(
                "total_bytes",
                &self
                    .weights
                    .values()
                    .map(|w| w.buffer.byte_len())
                    .sum::<usize>(),
            )
            .finish()
    }
}

impl WeightMap {
    /// Total memory used by all loaded weight buffers in bytes.
    pub fn total_bytes(&self) -> usize {
        self.weights
            .values()
            .map(|w| w.buffer.byte_len())
            .sum()
    }

    /// Number of tensors in the map.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the weight map is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Get a loaded weight by tensor name.
    pub fn get(&self, name: &str) -> Option<&LoadedWeight> {
        self.weights.get(name)
    }
}

/// Load all model weights from safetensors files into Metal buffers.
///
/// Discovers all `*.safetensors` files in `model_dir`, loads each tensor
/// into an `MlxBuffer`, and attaches per-tensor quantization metadata from
/// `quantization_config.json` (if present).
///
/// # Arguments
///
/// * `model_dir` - Path to the model directory
/// * `device` - Metal device for buffer allocation
pub fn load_weights(
    model_dir: &Path,
    device: &MlxDevice,
) -> Result<WeightMap, WeightLoadError> {
    // Discover safetensors files
    let safetensors_paths = discover_safetensors(model_dir)?;
    if safetensors_paths.is_empty() {
        return Err(WeightLoadError::NoSafetensors {
            path: model_dir.display().to_string(),
        });
    }

    info!(
        shard_count = safetensors_paths.len(),
        "Loading weight shards from {}",
        model_dir.display()
    );

    // Load quantization config if present
    let quant_config = load_quant_config(model_dir);

    // Load tensors from all shards
    let mut weights = HashMap::new();

    for sf_path in &safetensors_paths {
        debug!(file = %sf_path.display(), "Opening safetensors shard");

        let sf = SafetensorsFile::open(sf_path).map_err(|e| WeightLoadError::SafetensorsOpen {
            path: sf_path.display().to_string(),
            reason: format!("{}", e),
        })?;

        let tensor_names = sf.tensor_names().map_err(|e| WeightLoadError::SafetensorsOpen {
            path: sf_path.display().to_string(),
            reason: format!("Failed to list tensors: {}", e),
        })?;

        for name in &tensor_names {
            let (dtype, shape, buffer) =
                sf.load_tensor(name, device).map_err(|e| WeightLoadError::TensorLoad {
                    tensor: name.clone(),
                    reason: format!("{}", e),
                })?;

            let quant_meta = quant_config.as_ref().map(|qc| {
                let (bits, group_size) = qc.config_for_tensor(name);
                TensorQuantMeta { bits, group_size }
            });

            weights.insert(
                name.clone(),
                LoadedWeight {
                    buffer,
                    dtype,
                    shape,
                    quant_meta,
                },
            );
        }

        debug!(
            file = %sf_path.display(),
            tensors = tensor_names.len(),
            "Shard loaded"
        );
    }

    let total_bytes: usize = weights.values().map(|w| w.buffer.byte_len()).sum();

    info!(
        tensor_count = weights.len(),
        total_mb = total_bytes / (1024 * 1024),
        "All weights loaded into Metal buffers"
    );

    Ok(WeightMap { weights })
}

/// Discover all safetensors files in a directory, sorted by name.
fn discover_safetensors(dir: &Path) -> Result<Vec<std::path::PathBuf>, WeightLoadError> {
    let entries = std::fs::read_dir(dir)?;

    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            files.push(path);
        }
    }

    files.sort();
    Ok(files)
}

/// Load quantization_config.json if present. Returns None if the file
/// doesn't exist or can't be parsed (non-fatal).
fn load_quant_config(model_dir: &Path) -> Option<QuantizationConfig> {
    let config_path = model_dir.join("quantization_config.json");
    if !config_path.exists() {
        debug!("No quantization_config.json found — treating all tensors as unquantized");
        return None;
    }

    match QuantizationConfig::from_file(&config_path) {
        Ok(config) => {
            info!(
                bits = config.bits,
                group_size = config.group_size,
                per_tensor_overrides = config.per_tensor.len(),
                "Loaded quantization config"
            );
            Some(config)
        }
        Err(e) => {
            warn!("Failed to parse quantization_config.json: {}. Treating all tensors as unquantized.", e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_safetensors_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let files = discover_safetensors(tmp.path()).unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn test_discover_safetensors_with_files() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("model-00001.safetensors"), b"dummy").unwrap();
        std::fs::write(tmp.path().join("model-00002.safetensors"), b"dummy").unwrap();
        std::fs::write(tmp.path().join("config.json"), b"{}").unwrap();

        let files = discover_safetensors(tmp.path()).unwrap();
        assert_eq!(files.len(), 2);
        assert!(files[0].file_name().unwrap().to_str().unwrap().contains("00001"));
        assert!(files[1].file_name().unwrap().to_str().unwrap().contains("00002"));
    }

    #[test]
    fn test_discover_safetensors_nonexistent_dir() {
        let result = discover_safetensors(Path::new("/nonexistent/dir"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_quant_config_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let config = load_quant_config(tmp.path());
        assert!(config.is_none());
    }

    #[test]
    fn test_load_quant_config_valid() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("quantization_config.json"),
            r#"{"bits": 4, "group_size": 64}"#,
        )
        .unwrap();

        let config = load_quant_config(tmp.path());
        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 64);
    }

    #[test]
    fn test_load_quant_config_invalid_json() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("quantization_config.json"),
            "not json",
        )
        .unwrap();

        let config = load_quant_config(tmp.path());
        assert!(config.is_none()); // Non-fatal
    }

    #[test]
    fn test_no_safetensors_error() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("config.json"), b"{}").unwrap();

        // We'd need an MlxDevice for this, so just test the discovery part
        let result = discover_safetensors(tmp.path()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_weight_map_empty() {
        let wm = WeightMap {
            weights: HashMap::new(),
        };
        assert!(wm.is_empty());
        assert_eq!(wm.len(), 0);
        assert_eq!(wm.total_bytes(), 0);
        assert!(wm.get("nonexistent").is_none());
    }
}
