//! Streaming safetensors reader via memory-mapped I/O.
//!
//! Supports both sharded models (model.safetensors.index.json) and single-file models.
//! Uses memmap2 for lazy memory-mapped access to tensor data.
//! Memory usage bounded: only one shard is actively accessed at a time.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde_json::Value;
use thiserror::Error;
use tracing::{debug, info};

use crate::ir::{DType, TensorMap, TensorRef};
use crate::progress::ProgressReporter;

/// Errors from safetensors reading.
#[derive(Error, Debug)]
pub enum SafetensorsError {
    #[error("No safetensors files found in {path}")]
    NoFiles { path: String },

    #[error("Failed to read shard '{shard}': {source}")]
    ShardReadError {
        shard: String,
        source: std::io::Error,
    },

    #[error("Failed to memory-map shard '{shard}': {source}")]
    MmapError {
        shard: String,
        source: std::io::Error,
    },

    #[error("Failed to parse safetensors header in '{shard}': {reason}")]
    HeaderParseError { shard: String, reason: String },

    #[error("Failed to parse index.json: {0}")]
    IndexParseError(String),

    #[error("Unsupported dtype '{dtype}' in tensor '{tensor}'")]
    UnsupportedDtype { dtype: String, tensor: String },

    #[error("Tensor '{tensor}' data range [{start}..{end}] exceeds file size {file_size} in shard '{shard}'")]
    DataOutOfBounds {
        tensor: String,
        shard: String,
        start: usize,
        end: usize,
        file_size: usize,
    },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Read all tensors from safetensors files in a model directory.
///
/// Handles both:
/// 1. Sharded models with `model.safetensors.index.json`
/// 2. Single-file models with `model.safetensors`
///
/// Tensors are read via mmap, with data copied out per-shard to limit memory usage.
pub fn read_tensors(
    model_dir: &Path,
    progress: &ProgressReporter,
) -> Result<TensorMap, SafetensorsError> {
    let shard_paths = discover_shards(model_dir)?;

    if shard_paths.is_empty() {
        return Err(SafetensorsError::NoFiles {
            path: model_dir.display().to_string(),
        });
    }

    info!(
        shard_count = shard_paths.len(),
        "Discovered safetensors shards"
    );

    let pb = progress.bar(shard_paths.len() as u64, "Reading shards");

    let mut tensor_map = TensorMap::new();

    for shard_path in &shard_paths {
        let shard_name = shard_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| shard_path.display().to_string());

        debug!(shard = %shard_name, "Reading shard");

        let tensors = read_shard(shard_path, &shard_name)?;
        for tensor in tensors {
            tensor_map.insert(tensor);
        }

        pb.inc(1);
    }

    pb.finish_with_message(format!("Read {} tensors from {} shards", tensor_map.len(), shard_paths.len()));

    Ok(tensor_map)
}

/// Discover safetensors shard file paths in a model directory.
fn discover_shards(model_dir: &Path) -> Result<Vec<PathBuf>, SafetensorsError> {
    // Check for sharded model with index
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        return discover_shards_from_index(&index_path, model_dir);
    }

    // Check for single-file model
    let single_path = model_dir.join("model.safetensors");
    if single_path.exists() {
        return Ok(vec![single_path]);
    }

    // Fallback: find any .safetensors files
    let mut paths: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();

    paths.sort();
    Ok(paths)
}

/// Discover shard paths from model.safetensors.index.json.
fn discover_shards_from_index(
    index_path: &Path,
    model_dir: &Path,
) -> Result<Vec<PathBuf>, SafetensorsError> {
    let content = std::fs::read_to_string(index_path).map_err(|e| {
        SafetensorsError::IndexParseError(format!("Failed to read {}: {}", index_path.display(), e))
    })?;

    let index: Value = serde_json::from_str(&content).map_err(|e| {
        SafetensorsError::IndexParseError(format!(
            "Failed to parse {}: {}",
            index_path.display(),
            e
        ))
    })?;

    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            SafetensorsError::IndexParseError("index.json missing weight_map".to_string())
        })?;

    let mut shard_names: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    shard_names.sort();
    shard_names.dedup();

    let paths: Vec<PathBuf> = shard_names
        .into_iter()
        .map(|name| model_dir.join(&name))
        .collect();

    // Verify all shards exist
    for path in &paths {
        if !path.exists() {
            return Err(SafetensorsError::ShardReadError {
                shard: path.display().to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Shard file not found: {}", path.display()),
                ),
            });
        }
    }

    Ok(paths)
}

/// Read all tensors from a single safetensors shard.
///
/// Uses memory-mapped I/O: the file is mmap'd, tensors are parsed from the header,
/// and data is copied out. The mmap is dropped when this function returns,
/// keeping memory bounded.
fn read_shard(shard_path: &Path, shard_name: &str) -> Result<Vec<TensorRef>, SafetensorsError> {
    let file = File::open(shard_path).map_err(|e| SafetensorsError::ShardReadError {
        shard: shard_name.to_string(),
        source: e,
    })?;

    // Safety: we're reading only, the file exists, and we handle any errors.
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| SafetensorsError::MmapError {
            shard: shard_name.to_string(),
            source: e,
        })?
    };

    let file_size = mmap.len();
    if file_size < 8 {
        return Err(SafetensorsError::HeaderParseError {
            shard: shard_name.to_string(),
            reason: format!("File too small ({} bytes)", file_size),
        });
    }

    // Safetensors format: first 8 bytes = u64 LE header size, then JSON header, then data
    let header_size = u64::from_le_bytes(
        mmap[..8]
            .try_into()
            .map_err(|_| SafetensorsError::HeaderParseError {
                shard: shard_name.to_string(),
                reason: "Failed to read header size".to_string(),
            })?,
    ) as usize;

    if 8 + header_size > file_size {
        return Err(SafetensorsError::HeaderParseError {
            shard: shard_name.to_string(),
            reason: format!(
                "Header size ({}) exceeds file size ({})",
                header_size,
                file_size - 8
            ),
        });
    }

    let header_bytes = &mmap[8..8 + header_size];
    let header: HashMap<String, Value> =
        serde_json::from_slice(header_bytes).map_err(|e| SafetensorsError::HeaderParseError {
            shard: shard_name.to_string(),
            reason: format!("JSON parse error: {}", e),
        })?;

    let data_start = 8 + header_size;
    let mut tensors = Vec::new();

    for (name, info) in &header {
        // Skip __metadata__ key
        if name == "__metadata__" {
            continue;
        }

        let dtype_str = info
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SafetensorsError::HeaderParseError {
                shard: shard_name.to_string(),
                reason: format!("Tensor '{}' missing dtype", name),
            })?;

        let dtype = DType::from_safetensors_str(dtype_str).ok_or_else(|| {
            SafetensorsError::UnsupportedDtype {
                dtype: dtype_str.to_string(),
                tensor: name.clone(),
            }
        })?;

        let shape: Vec<usize> = info
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| SafetensorsError::HeaderParseError {
                shard: shard_name.to_string(),
                reason: format!("Tensor '{}' missing shape", name),
            })?
            .iter()
            .filter_map(|v| v.as_u64().map(|u| u as usize))
            .collect();

        let offsets = info
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| SafetensorsError::HeaderParseError {
                shard: shard_name.to_string(),
                reason: format!("Tensor '{}' missing data_offsets", name),
            })?;

        if offsets.len() != 2 {
            return Err(SafetensorsError::HeaderParseError {
                shard: shard_name.to_string(),
                reason: format!(
                    "Tensor '{}' has {} data_offsets, expected 2",
                    name,
                    offsets.len()
                ),
            });
        }

        let offset_start = offsets[0].as_u64().unwrap_or(0) as usize;
        let offset_end = offsets[1].as_u64().unwrap_or(0) as usize;

        let abs_start = data_start + offset_start;
        let abs_end = data_start + offset_end;

        if abs_end > file_size {
            return Err(SafetensorsError::DataOutOfBounds {
                tensor: name.clone(),
                shard: shard_name.to_string(),
                start: abs_start,
                end: abs_end,
                file_size,
            });
        }

        // Copy data from mmap — this is the bounded-memory approach.
        // Each shard's data is copied and then the mmap for the shard is freed.
        let data = mmap[abs_start..abs_end].to_vec();

        tensors.push(TensorRef {
            name: name.clone(),
            shape,
            dtype,
            data,
        });
    }

    debug!(
        shard = %shard_name,
        tensor_count = tensors.len(),
        "Parsed shard"
    );

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal valid safetensors file in memory.
    /// Format: 8-byte header length (LE u64) + JSON header + tensor data
    fn create_test_safetensors(
        tensors: &[(&str, &[usize], &str, &[u8])],
    ) -> Vec<u8> {
        let mut header_map = serde_json::Map::new();
        let mut current_offset = 0usize;

        for (name, shape, dtype, data) in tensors {
            let mut tensor_info = serde_json::Map::new();
            tensor_info.insert(
                "dtype".to_string(),
                serde_json::Value::String(dtype.to_string()),
            );
            tensor_info.insert(
                "shape".to_string(),
                serde_json::Value::Array(
                    shape.iter().map(|&s| serde_json::Value::Number(s.into())).collect(),
                ),
            );
            let end_offset = current_offset + data.len();
            tensor_info.insert(
                "data_offsets".to_string(),
                serde_json::Value::Array(vec![
                    serde_json::Value::Number(current_offset.into()),
                    serde_json::Value::Number(end_offset.into()),
                ]),
            );
            header_map.insert(name.to_string(), serde_json::Value::Object(tensor_info));
            current_offset = end_offset;
        }

        let header_json = serde_json::to_string(&header_map).unwrap();
        let header_bytes = header_json.as_bytes();
        let header_size = header_bytes.len() as u64;

        let mut file_data = Vec::new();
        file_data.extend_from_slice(&header_size.to_le_bytes());
        file_data.extend_from_slice(header_bytes);
        for (_, _, _, data) in tensors {
            file_data.extend_from_slice(data);
        }

        file_data
    }

    #[test]
    fn test_read_single_shard() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path();

        // Create a small test tensor: 2x3 F32
        let tensor_data: Vec<u8> = (0..6u32)
            .flat_map(|v| (v as f32).to_le_bytes())
            .collect();

        let safetensors_data =
            create_test_safetensors(&[("test_weight", &[2, 3], "F32", &tensor_data)]);

        std::fs::write(model_dir.join("model.safetensors"), &safetensors_data).unwrap();

        let progress = ProgressReporter::new();
        let tensor_map = read_tensors(model_dir, &progress).unwrap();

        assert_eq!(tensor_map.len(), 1);
        let tensor = tensor_map.get("test_weight").unwrap();
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.dtype, DType::F32);
        assert_eq!(tensor.data.len(), 24); // 6 * 4 bytes
    }

    #[test]
    fn test_read_multiple_tensors() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path();

        let weight_data: Vec<u8> = vec![0u8; 4 * 2]; // 4 F16 elements
        let bias_data: Vec<u8> = vec![0u8; 2 * 2]; // 2 F16 elements

        let safetensors_data = create_test_safetensors(&[
            ("layer.weight", &[2, 2], "F16", &weight_data),
            ("layer.bias", &[2], "F16", &bias_data),
        ]);

        std::fs::write(model_dir.join("model.safetensors"), &safetensors_data).unwrap();

        let progress = ProgressReporter::new();
        let tensor_map = read_tensors(model_dir, &progress).unwrap();

        assert_eq!(tensor_map.len(), 2);
        assert!(tensor_map.get("layer.weight").is_some());
        assert!(tensor_map.get("layer.bias").is_some());
    }

    #[test]
    fn test_no_safetensors_error() {
        let tmp = tempfile::tempdir().unwrap();
        let progress = ProgressReporter::new();
        let result = read_tensors(tmp.path(), &progress);
        assert!(result.is_err());
    }
}
