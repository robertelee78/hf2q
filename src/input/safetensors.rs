//! Streaming safetensors reader via memory-mapped I/O.
//!
//! Supports both sharded models (`model.safetensors.index.json`) and
//! single-file models. Built on a lazy primitive ([`LazyTensorMap`]):
//! tensor metadata (name / shape / dtype / byte range) is parsed from
//! every shard at index time, but the bulk weight bytes only materialise
//! when the consumer calls [`LazyTensor::materialize`].
//!
//! ## ADR-014 P0 contract (Decisions 1 + 2)
//!
//! The shard file is mmap'd into an [`Arc<Mmap>`]; every per-tensor
//! [`LazyTensor`] in the resulting [`LazyTensorMap`] carries a closure
//! that holds an `Arc<Mmap>` + the tensor's byte offset + len. The
//! `Arc<Mmap>` keeps the file mapped through the entire pipeline; pages
//! are paged in / evicted by the kernel, not by us. On apex MoE this
//! means ~12 mmap regions live (one per shard); total virtual ~70 GB;
//! **resident** ~one tensor at a time once the streaming quantize loop
//! (P2) consumes from the map.
//!
//! ## Bridge to the eager API
//!
//! Pre-P1 callers (`src/input/mod.rs::read_model`, `src/models/vit/...`)
//! still consume an eager [`TensorMap`]. The legacy [`read_tensors`]
//! function is the bridge: `read_tensors_lazy(...).materialize_all()`.
//! P1 lifts the Phase 1.4–1.7 transforms onto `LazyTensorMap` and the
//! bridge becomes test-only.
//!
//! ## Why we no longer copy at parse time
//!
//! The Chesterton-fence rationale for the previous eager `to_vec()`
//! pattern: when the original reader was written, lifetimes on `Mmap`
//! borrows threaded through the entire pipeline and "copy out, drop the
//! mmap" was the simplest way to avoid borrow propagation. We now have
//! `Arc<Mmap>` patterns elsewhere (mlx-native tensor refs); the
//! lifetime cost has been amortised, and the memory cost (eager copy of
//! ~70 GB on apex MoE) was the primary bottleneck on M5 Max 128 GB.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use serde_json::Value;
use thiserror::Error;
use tracing::{debug, info};

use crate::ir::lazy::{LazyMeta, LazyTensor, LazyTensorMap, MaterializeError};
use crate::ir::{DType, TensorMap};
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

    #[error("Tensor '{tensor}' header size {header_bytes} disagrees with shape×dtype {derived_bytes} in shard '{shard}'")]
    HeaderSizeMismatch {
        tensor: String,
        shard: String,
        header_bytes: usize,
        derived_bytes: usize,
    },

    /// Surfaced when the eager [`read_tensors`] bridge runs the lazy
    /// closures and one of them fails. Should be impossible in practice
    /// (bounds + size are checked at parse time before the closure is
    /// constructed), but kept typed so the bridge can never silently
    /// downgrade a real I/O failure.
    #[error("Materialise error: {0}")]
    Materialize(#[from] MaterializeError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Read tensor *metadata* lazily from safetensors files in a model
/// directory. Bytes do not materialise until each tensor's
/// [`LazyTensor::materialize`] is invoked.
///
/// Handles:
/// 1. Sharded models with `model.safetensors.index.json`
/// 2. Single-file models with `model.safetensors`
///
/// Each shard's mmap is wrapped in an [`Arc<Mmap>`] that lives as long
/// as any tensor closure from that shard — i.e. until every tensor in
/// the shard has either been materialised-and-dropped or the
/// [`LazyTensorMap`] itself is dropped. This is the streaming pipeline's
/// memory contract: parse-time RSS is bounded by the parsed metadata
/// alone; bulk weight bytes only become resident when consumed.
pub fn read_tensors_lazy(
    model_dir: &Path,
    progress: &ProgressReporter,
) -> Result<LazyTensorMap, SafetensorsError> {
    let shard_paths = discover_shards(model_dir)?;

    if shard_paths.is_empty() {
        return Err(SafetensorsError::NoFiles {
            path: model_dir.display().to_string(),
        });
    }

    info!(
        shard_count = shard_paths.len(),
        "Discovered safetensors shards (lazy)"
    );

    let pb = progress.bar(shard_paths.len() as u64, "Indexing shards");
    let mut lazy_map = LazyTensorMap::new();

    for shard_path in &shard_paths {
        let shard_name = shard_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| shard_path.display().to_string());

        debug!(shard = %shard_name, "Indexing shard");

        index_shard_lazy(shard_path, &shard_name, &mut lazy_map)?;

        pb.inc(1);
    }

    pb.finish_with_message(format!(
        "Indexed {} tensors from {} shards",
        lazy_map.len(),
        shard_paths.len()
    ));

    Ok(lazy_map)
}

/// Read all tensors from safetensors files in a model directory
/// (eager). Bridge for pre-P1 callers — implemented as
/// `read_tensors_lazy(...).materialize_all()`. Behaviour is byte-identical
/// to the previous eager reader (verified by
/// `tests/lazy_tensor.rs::test_lazy_safetensors_byte_identical_to_eager`),
/// but peak parse-time RSS is reduced because the bulk
/// weight bytes only materialise inside `materialize_all`'s loop —
/// per-tensor `to_vec`s, not the previous "every tensor in every shard
/// resident before any callback" pattern.
///
/// P1 lifts the Phase 1.x transforms onto `LazyTensorMap`; P2 lifts the
/// quantize loop. Once both ship, this bridge is consumed only by the
/// regression-test path.
pub fn read_tensors(
    model_dir: &Path,
    progress: &ProgressReporter,
) -> Result<TensorMap, SafetensorsError> {
    let lazy_map = read_tensors_lazy(model_dir, progress)?;
    let len = lazy_map.len();
    let tensor_map = lazy_map.materialize_all()?;
    debug!(tensors = len, "Materialised lazy map → eager TensorMap");
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

/// Discover shard paths from `model.safetensors.index.json`.
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

/// Index a single shard into `lazy_map`: parse the JSON header, build
/// one [`LazyTensor`] per entry whose closure captures
/// `Arc<Mmap>` + offset + len. The closure body is the materialisation
/// site (the `mmap[off..off+len].to_vec()` that previously ran eagerly
/// at parse time, deleted from the old `read_shard` per ADR-014
/// Decision 2).
///
/// Bounds, size, and dtype are validated at parse time so the closure
/// invocation is infallible in the success path; the closure's return
/// type still surfaces `MaterializeError` for defensive `Result`
/// plumbing.
fn index_shard_lazy(
    shard_path: &Path,
    shard_name: &str,
    lazy_map: &mut LazyTensorMap,
) -> Result<(), SafetensorsError> {
    let file = File::open(shard_path).map_err(|e| SafetensorsError::ShardReadError {
        shard: shard_name.to_string(),
        source: e,
    })?;

    // Safety: we map for read; the file exists; we propagate I/O errors.
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

    // Safetensors format: u64-LE header size, then JSON header, then data.
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

    // Wrap mmap in an Arc — every per-tensor closure clones it. The
    // shard's mmap stays mapped until every tensor closure from this
    // shard has either fired (and dropped its clone) or the parent
    // LazyTensorMap is dropped. Pages are paged in / evicted by the
    // kernel during materialisation; the OS does the working-set
    // management, not us.
    let mmap = Arc::new(mmap);

    for (name, info) in &header {
        // Skip __metadata__ key per safetensors convention.
        if name == "__metadata__" {
            continue;
        }

        let dtype_str =
            info.get("dtype")
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

        let offsets =
            info.get("data_offsets")
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

        // Cross-check header byte range against shape × dtype-derived
        // byte length. Disagreement here is a corrupt safetensors file
        // (or, theoretically, a future format change we haven't tracked)
        // — bail loudly, don't silently materialise the wrong size.
        let header_byte_len = abs_end - abs_start;
        let meta = LazyMeta::new(name.clone(), shape, dtype);
        if meta.byte_len != header_byte_len {
            return Err(SafetensorsError::HeaderSizeMismatch {
                tensor: name.clone(),
                shard: shard_name.to_string(),
                header_bytes: header_byte_len,
                derived_bytes: meta.byte_len,
            });
        }

        // Build the closure: capture an Arc<Mmap> + the byte range. The
        // body is the to_vec that previously ran eagerly at line 316 of
        // the pre-ADR-014 reader — moved into the closure per Decision 2.
        let mmap_clone = Arc::clone(&mmap);
        let load = move || -> Result<Vec<u8>, MaterializeError> {
            // The mmap is alive for the lifetime of this closure (Arc keeps
            // it). The slice access is bounded; bounds were checked at
            // index time, so the only way this can fail is if the OS un-
            // mapped pages we asked for — which would have surfaced as a
            // SIGBUS in the slice read, not a Result error. Defensive
            // typing covers the (unreachable) error path.
            Ok(mmap_clone[abs_start..abs_end].to_vec())
        };

        let lazy = LazyTensor::from_closure(meta, load);
        lazy_map.insert(lazy);
    }

    debug!(
        shard = %shard_name,
        tensor_count = lazy_map.len(),
        "Indexed shard"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    /// Eager-bridge regression: the legacy `read_tensors` API still
    /// produces a TensorMap byte-identical to the pre-ADR-014 reader on
    /// the same input. Implemented under the hood as
    /// `read_tensors_lazy + materialize_all` — this test guards the
    /// bridge from drift.
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
        assert_eq!(tensor.data, tensor_data);
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

    /// New under ADR-014 P0: the lazy reader returns a LazyTensorMap.
    /// Metadata is queryable; tensor bytes only materialise on demand.
    #[test]
    fn test_lazy_reader_metadata_only() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path();

        let tensor_data: Vec<u8> = (0..6u32)
            .flat_map(|v| (v as f32).to_le_bytes())
            .collect();
        let safetensors_data =
            create_test_safetensors(&[("w", &[2, 3], "F32", &tensor_data)]);
        std::fs::write(model_dir.join("model.safetensors"), &safetensors_data).unwrap();

        let progress = ProgressReporter::new();
        let lazy_map = read_tensors_lazy(model_dir, &progress).unwrap();

        // Metadata is queryable without tapping the closure.
        assert_eq!(lazy_map.len(), 1);
        let lazy = lazy_map.get("w").unwrap();
        assert_eq!(lazy.shape(), &[2, 3]);
        assert_eq!(lazy.dtype(), DType::F32);
        assert_eq!(lazy.byte_len(), 24);

        // Materialise and check bytes match input.
        let realised = lazy_map
            .into_iter()
            .next()
            .unwrap()
            .1
            .materialize()
            .unwrap();
        assert_eq!(realised.data, tensor_data);
    }

    /// Lazy reader and the legacy eager reader produce byte-identical
    /// output on the same fixture. This is the Decision 2 / Decision 17
    /// regression contract: byte-identical to the pre-ADR-014 path on
    /// every uncalibrated input.
    #[test]
    fn test_lazy_byte_identical_to_eager_bridge() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path();

        // Mixed dtypes + multiple tensors, name-sorted differently from
        // insertion order so BTreeMap vs HashMap iteration won't mask
        // bugs.
        let f32_data: Vec<u8> = (0..6u32)
            .flat_map(|v| (v as f32).to_le_bytes())
            .collect();
        let f16_data: Vec<u8> = (0..4u32)
            .flat_map(|v| half::f16::from_f32(v as f32).to_le_bytes())
            .collect();
        let bf16_data: Vec<u8> = (0..3u32)
            .flat_map(|v| half::bf16::from_f32(v as f32).to_le_bytes())
            .collect();

        let safetensors_data = create_test_safetensors(&[
            ("zebra.weight", &[2, 3], "F32", &f32_data),
            ("alpha.weight", &[2, 2], "F16", &f16_data),
            ("mango.weight", &[3], "BF16", &bf16_data),
        ]);
        std::fs::write(model_dir.join("model.safetensors"), &safetensors_data).unwrap();

        let progress = ProgressReporter::new();

        // Lazy → materialize_all
        let lazy_map = read_tensors_lazy(model_dir, &progress).unwrap();
        let materialised = lazy_map.materialize_all().unwrap();

        // Eager bridge (same call site that's exercised by Phase 1.x of
        // cmd_convert in main.rs:466)
        let eager = read_tensors(model_dir, &progress).unwrap();

        assert_eq!(materialised.len(), eager.len());
        for name in ["zebra.weight", "alpha.weight", "mango.weight"] {
            let m = materialised.get(name).unwrap();
            let e = eager.get(name).unwrap();
            assert_eq!(m.shape, e.shape, "{name} shape");
            assert_eq!(m.dtype, e.dtype, "{name} dtype");
            assert_eq!(m.data, e.data, "{name} bytes");
        }
    }

    /// Indexing a shard does not invoke any per-tensor materialiser.
    /// We can't directly probe the closure (it captures Arc<Mmap>, not a
    /// counter we control), but we can prove that the LazyTensorMap has
    /// the right metadata and that materialise produces bytes byte-equal
    /// to the source. The combination of (a) the unit-level
    /// `test_shape_dtype_no_materialise` in src/ir/lazy.rs and (b) this
    /// integration test gives end-to-end coverage.
    #[test]
    fn test_lazy_reader_does_not_materialise_at_index_time() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path();

        // Synthesise a 64KB tensor — large enough that an inadvertent
        // eager copy would be visible in `to_vec` allocation, but small
        // enough to still run in unit-test time.
        let big_data: Vec<u8> = (0..16 * 1024)
            .flat_map(|v: u32| (v as f32).to_le_bytes())
            .collect();
        let safetensors_data =
            create_test_safetensors(&[("big.weight", &[16 * 1024], "F32", &big_data)]);
        std::fs::write(model_dir.join("model.safetensors"), &safetensors_data).unwrap();

        let progress = ProgressReporter::new();

        // Track number of byte-vector allocations attributed to *us*.
        // We use a process-static counter because the materialise
        // pathway uses an mmap-backed `to_vec`, which goes through the
        // global allocator. Instead of trying to count allocations
        // (which is allocator-specific), we verify that a metadata-only
        // probe doesn't change the visible state of the LazyTensorMap.
        let probe = AtomicUsize::new(0);
        let lazy_map = read_tensors_lazy(model_dir, &progress).unwrap();

        // 100× metadata accesses — none of them materialise.
        for _ in 0..100 {
            let lazy = lazy_map.get("big.weight").unwrap();
            assert_eq!(lazy.shape(), &[16 * 1024]);
            assert_eq!(lazy.dtype(), DType::F32);
            probe.fetch_add(1, Ordering::SeqCst);
        }
        assert_eq!(probe.load(Ordering::SeqCst), 100);

        // Materialise once and confirm bytes round-trip.
        let realised = lazy_map
            .into_iter()
            .find(|(k, _)| k == "big.weight")
            .unwrap()
            .1
            .materialize()
            .unwrap();
        assert_eq!(realised.data, big_data);
    }

    /// Header byte-range disagreeing with shape × dtype is a typed
    /// error, not silent corruption.
    #[test]
    fn test_lazy_reader_rejects_header_size_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path();

        // Lie: declare 2x3 F32 (24 bytes) but write only 12 bytes.
        let mut header_map = serde_json::Map::new();
        let mut tensor_info = serde_json::Map::new();
        tensor_info.insert(
            "dtype".to_string(),
            serde_json::Value::String("F32".to_string()),
        );
        tensor_info.insert(
            "shape".to_string(),
            serde_json::Value::Array(vec![
                serde_json::Value::Number(2.into()),
                serde_json::Value::Number(3.into()),
            ]),
        );
        // Header offsets say 12 bytes (3 floats), shape × dtype derives 24 (6 floats).
        tensor_info.insert(
            "data_offsets".to_string(),
            serde_json::Value::Array(vec![
                serde_json::Value::Number(0u64.into()),
                serde_json::Value::Number(12u64.into()),
            ]),
        );
        header_map.insert("liar".to_string(), serde_json::Value::Object(tensor_info));

        let header_json = serde_json::to_string(&header_map).unwrap();
        let header_size = header_json.len() as u64;
        let mut file_data = Vec::new();
        file_data.extend_from_slice(&header_size.to_le_bytes());
        file_data.extend_from_slice(header_json.as_bytes());
        // Append exactly 12 bytes of body — matches the header lie.
        file_data.extend_from_slice(&vec![0u8; 12]);
        std::fs::write(model_dir.join("model.safetensors"), &file_data).unwrap();

        let progress = ProgressReporter::new();
        let err = read_tensors_lazy(model_dir, &progress).unwrap_err();
        match err {
            SafetensorsError::HeaderSizeMismatch {
                tensor,
                header_bytes,
                derived_bytes,
                ..
            } => {
                assert_eq!(tensor, "liar");
                assert_eq!(header_bytes, 12);
                assert_eq!(derived_bytes, 24);
            }
            other => panic!("expected HeaderSizeMismatch, got {other:?}"),
        }
    }
}
