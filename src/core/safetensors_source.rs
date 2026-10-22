//! Generic safetensors source-input helpers.
//!
//! Extracted behavior-preserving from `core::mlx_safetensors_loader`
//! (AFFINE-RETIREMENT R1, 2026-10-22) so the generic HF-source reading
//! surface conversion depends on survives separately from the retiring
//! affine/DWQ loader module. Exact dtype, ordering, error, sharded-index
//! and single-file sentinel semantics are unchanged; see the original
//! history in `mlx_safetensors_loader.rs` for provenance.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;
use safetensors::tensor::Dtype;

/// Read raw safetensors bytes of `Dtype::{F32, F16, BF16}` into a
/// `Vec<f32>`.  All inputs are little-endian per the safetensors
/// spec.
pub fn read_floats_to_f32(bytes: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    // Parallel BF16/F16/F32 → F32 conversion. Profile of Qwen 3.5 35B
    // convert (2026-05-21, `sample(1)` PID 80984) showed this function
    // consumes ~40% of the main-thread wall-time (1803 / 4500 main-thread
    // samples) — bigger contributor than F16 round-trip or BakeOps. The
    // per-element transform is pure (no shared state, no FP ordering
    // hazards), so par_chunks_exact preserves byte-identity vs the
    // serial loop. par_chunks_exact's default work-stealing chunk size
    // (≥1) means small tensors stay effectively serial; only the multi-
    // GB safetensors chunks pay any rayon overhead.
    //
    // `with_min_len(1024)` batching was tested 2026-05-21 → only -2.4%
    // wall (within ~6.6% noise band); reverted. The 14915 worker samples
    // in `rayon::bridge_producer_consumer` reported by the post-opt
    // profile turned out NOT to be reducible by simple batching at this
    // call-site; further investigation would need controlled multi-run
    // statistics not a single bench.
    match dtype {
        Dtype::F32 => {
            if bytes.len() % 4 != 0 {
                return Err(anyhow!(
                    "read_floats_to_f32: F32 byte len {} not a multiple of 4",
                    bytes.len()
                ));
            }
            Ok(bytes
                .par_chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        Dtype::F16 => {
            if bytes.len() % 2 != 0 {
                return Err(anyhow!(
                    "read_floats_to_f32: F16 byte len {} not a multiple of 2",
                    bytes.len()
                ));
            }
            Ok(bytes
                .par_chunks_exact(2)
                .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
                .collect())
        }
        Dtype::BF16 => {
            if bytes.len() % 2 != 0 {
                return Err(anyhow!(
                    "read_floats_to_f32: BF16 byte len {} not a multiple of 2",
                    bytes.len()
                ));
            }
            Ok(bytes
                .par_chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
                .collect())
        }
        other => Err(anyhow!(
            "read_floats_to_f32: dtype {other:?} not supported (expected F32/F16/BF16)"
        )),
    }
}

/// Discover the model.safetensors shard files in a directory.
/// Returns the index map (tensor_name → shard_filename) if a sharded
/// `model.safetensors.index.json` is present, or a single `[(*, "model.safetensors")]`
/// equivalent if the model is a single-file save.  Mirrors mlx-lm's
/// `save_model` output convention (utils.py:728-771).
pub fn discover_shards(model_dir: &Path) -> Result<BTreeMap<String, PathBuf>> {
    let single = model_dir.join("model.safetensors");
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path)
            .with_context(|| format!("read {}", index_path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", index_path.display()))?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| anyhow!("index has no 'weight_map' object"))?;
        let mut out = BTreeMap::new();
        for (k, shard) in map {
            let shard_name = shard
                .as_str()
                .ok_or_else(|| anyhow!("weight_map[{k}] not a string"))?;
            out.insert(k.clone(), model_dir.join(shard_name));
        }
        Ok(out)
    } else if single.exists() {
        // Single-file: caller will deserialize once + iterate; we
        // signal "single file" with an empty map sentinel via an
        // explicit marker below.  Since we don't have the tensor
        // names without parsing the file, just return an empty map
        // and let the caller use the single-file fast path.
        let mut out = BTreeMap::new();
        out.insert("__single__".to_string(), single);
        Ok(out)
    } else {
        Err(anyhow!(
            "discover_shards: neither model.safetensors.index.json nor model.safetensors found in {}",
            model_dir.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_floats_to_f32_handles_all_three_dtypes() {
        // F32 round-trip
        let f32_in: Vec<f32> = vec![1.5, -0.25, 3.14];
        let f32_bytes: Vec<u8> = f32_in.iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(read_floats_to_f32(&f32_bytes, Dtype::F32).unwrap(), f32_in);

        // F16 round-trip (within f16 precision)
        let f16_vals: Vec<f32> = vec![1.0, -0.5, 2.5];
        let f16_bytes: Vec<u8> = f16_vals
            .iter()
            .flat_map(|f| half::f16::from_f32(*f).to_le_bytes())
            .collect();
        let got = read_floats_to_f32(&f16_bytes, Dtype::F16).unwrap();
        for (a, b) in got.iter().zip(f16_vals.iter()) {
            assert!((a - b).abs() < 1e-3);
        }

        // BF16 round-trip
        let bf16_vals: Vec<f32> = vec![0.125, -16.0, 100.0];
        let bf16_bytes: Vec<u8> = bf16_vals
            .iter()
            .flat_map(|f| half::bf16::from_f32(*f).to_le_bytes())
            .collect();
        let got = read_floats_to_f32(&bf16_bytes, Dtype::BF16).unwrap();
        for (a, b) in got.iter().zip(bf16_vals.iter()) {
            assert!((a - b).abs() < 0.5 * b.abs().max(1.0));
        }
    }

    #[test]
    fn discover_shards_handles_index_json() {
        let tmp = tempfile::tempdir().unwrap();
        let idx = serde_json::json!({
            "metadata": {"total_size": 12345},
            "weight_map": {
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors"
            }
        });
        std::fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&idx).unwrap(),
        )
        .unwrap();
        let map = discover_shards(tmp.path()).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(
            map["model.layers.0.weight"].file_name().unwrap(),
            "model-00001-of-00002.safetensors"
        );
    }

    #[test]
    fn discover_shards_handles_single_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("model.safetensors"), [0u8; 8]).unwrap();
        let map = discover_shards(tmp.path()).unwrap();
        assert_eq!(map.len(), 1);
        assert!(map.contains_key("__single__"));
    }
}
