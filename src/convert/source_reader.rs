//! `HfModelSource` — read a HuggingFace `model.safetensors` directory
//! (single file or sharded) + its `config.json` into F32-dequantized
//! tensors ready for the convert pipeline.
//!
//! Per ADR-033 §P0: this is the load-side of the new convert path. It
//! sits between the disk layout and the `ConvertOrchestrator`, doing
//! exactly two jobs:
//!
//! 1. Discover `model.safetensors` (or `model-NNNNN-of-MMMMM.safetensors`
//!    shards via `model.safetensors.index.json`) under a model dir,
//!    deserialize each, and enumerate every tensor.
//! 2. Dequantize F16 / BF16 source bytes to F32 (the orchestrator's
//!    expected input dtype).
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: no fallback for
//! older safetensors layouts, no aliasing.
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]]: every unsupported
//! source dtype (U8 / I32 / FP8 / …) returns a typed error. F32 is
//! emitted only for the orchestrator's downstream consumer; never
//! silently demoted on the read side.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::{tensor::Dtype, SafeTensors};

use crate::core::mlx_safetensors_loader::{discover_shards, read_floats_to_f32};
use crate::quantize::ggml_quants::SourceDtype;

/// One safetensors-side tensor, dequantized to F32 row-major.
///
/// `shape` is in PyTorch order (`[out_dim, in_dim]` for a Linear weight).
/// Callers feeding `ConvertOrchestrator` must reverse to GGUF order
/// (`[in_dim, out_dim]`) at the boundary — the orchestrator does NOT
/// re-reverse internally.
#[derive(Debug, Clone)]
pub struct HfTensor {
    /// HuggingFace tensor name as it appears in the safetensors file.
    pub name: String,
    /// PyTorch-order dimensions (outermost-first).
    pub shape: Vec<usize>,
    /// On-disk source dtype (the format we dequantized FROM).
    pub source_dtype: SourceDtype,
    /// F32 row-major data, `shape.iter().product()` elements.
    pub data: Vec<f32>,
}

/// Whole-model bundle: every tensor + the raw `config.json`. Tensors
/// are returned in safetensors-file order; callers iterate and feed
/// the orchestrator via the per-arch tensor-name mapper.
#[derive(Debug)]
pub struct HfModelSource {
    pub tensors: Vec<HfTensor>,
    pub config: serde_json::Value,
}

/// Errors raised by [`HfModelSource::load`].
#[derive(Debug)]
pub enum SourceError {
    /// Filesystem I/O failed (missing dir, unreadable file, etc.).
    Io(std::io::Error),
    /// `config.json` couldn't be parsed as JSON.
    ConfigParse(serde_json::Error),
    /// `discover_shards` failed (missing `model.safetensors` AND
    /// missing `model.safetensors.index.json`, or malformed index).
    Discover(String),
    /// `safetensors::deserialize` rejected a shard file.
    Safetensors(String),
    /// Source-side tensor dtype not supported by the read path. Per
    /// ADR-033 Decision §"FP8 source-dtype auto-detect" the FP8 path is
    /// out of v1 scope here (will land alongside MiniMax-M2.7 support);
    /// quantized U8 / I32 / I64 tensors are out of scope entirely.
    UnsupportedSourceDtype {
        tensor: String,
        dtype: String,
    },
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceError::Io(e) => write!(f, "source/io: {e}"),
            SourceError::ConfigParse(e) => write!(f, "source/config.json: {e}"),
            SourceError::Discover(s) => write!(f, "source/discover: {s}"),
            SourceError::Safetensors(s) => write!(f, "source/safetensors: {s}"),
            SourceError::UnsupportedSourceDtype { tensor, dtype } => {
                write!(
                    f,
                    "source/unsupported dtype `{dtype}` on tensor `{tensor}` \
                     (supported: F32, F16, BF16)"
                )
            }
        }
    }
}

impl std::error::Error for SourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SourceError::Io(e) => Some(e),
            SourceError::ConfigParse(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SourceError {
    fn from(e: std::io::Error) -> Self {
        SourceError::Io(e)
    }
}

impl From<serde_json::Error> for SourceError {
    fn from(e: serde_json::Error) -> Self {
        SourceError::ConfigParse(e)
    }
}

impl HfModelSource {
    /// Load every safetensors tensor + `config.json` under `model_dir`.
    ///
    /// The directory layout follows HuggingFace's convention:
    /// - `<model_dir>/config.json` — model config (mandatory).
    /// - `<model_dir>/model.safetensors` — single-shard weights, OR
    /// - `<model_dir>/model.safetensors.index.json` — shard index
    ///   pointing to `model-NNNNN-of-MMMMM.safetensors` siblings.
    ///
    /// Tensor data is dequantized to F32 in-memory; the resulting
    /// `HfTensor::data` carries `shape.iter().product()` f32s.
    pub fn load(model_dir: &Path) -> Result<Self, SourceError> {
        // ----- config.json --------------------------------------------------
        let config_path = model_dir.join("config.json");
        let config_raw = fs::read_to_string(&config_path).map_err(|e| {
            SourceError::Io(std::io::Error::new(
                e.kind(),
                format!("read {}: {e}", config_path.display()),
            ))
        })?;
        let config: serde_json::Value = serde_json::from_str(&config_raw)?;

        // ----- shard discovery ---------------------------------------------
        let shards: BTreeMap<String, PathBuf> = discover_shards(model_dir)
            .map_err(|e| SourceError::Discover(format!("{e:#}")))?;

        // `discover_shards` returns either:
        //   * a single-file sentinel { "__single__" → <path>/model.safetensors },
        //   * or a per-tensor map { tensor_name → shard_path }.
        // We collapse both to the set of unique shard files we need to
        // read, then iterate every tensor inside each.
        let mut shard_paths: Vec<PathBuf> = shards.values().cloned().collect();
        shard_paths.sort();
        shard_paths.dedup();

        let mut tensors: Vec<HfTensor> = Vec::new();
        for shard in &shard_paths {
            let bytes = fs::read(shard).map_err(|e| {
                SourceError::Io(std::io::Error::new(
                    e.kind(),
                    format!("read {}: {e}", shard.display()),
                ))
            })?;
            let st = SafeTensors::deserialize(&bytes)
                .map_err(|e| SourceError::Safetensors(format!("{}: {e}", shard.display())))?;
            for name in st.names() {
                let view = st.tensor(name).map_err(|e| {
                    SourceError::Safetensors(format!("tensor `{name}` in {}: {e}", shard.display()))
                })?;
                let shape = view.shape().to_vec();
                let (source_dtype, data) = match view.dtype() {
                    Dtype::F32 => (
                        SourceDtype::F32,
                        read_floats_to_f32(view.data(), Dtype::F32)
                            .map_err(|e| SourceError::Safetensors(format!("F32 dequant {name}: {e:#}")))?,
                    ),
                    Dtype::F16 => (
                        SourceDtype::F16,
                        read_floats_to_f32(view.data(), Dtype::F16)
                            .map_err(|e| SourceError::Safetensors(format!("F16 dequant {name}: {e:#}")))?,
                    ),
                    Dtype::BF16 => (
                        SourceDtype::BF16,
                        read_floats_to_f32(view.data(), Dtype::BF16)
                            .map_err(|e| SourceError::Safetensors(format!("BF16 dequant {name}: {e:#}")))?,
                    ),
                    other => {
                        return Err(SourceError::UnsupportedSourceDtype {
                            tensor: name.to_string(),
                            dtype: format!("{other:?}"),
                        })
                    }
                };
                // Sanity: dequantized element count must match shape product.
                let expect = shape.iter().product::<usize>();
                if data.len() != expect {
                    return Err(SourceError::Safetensors(format!(
                        "tensor `{name}` in {}: dequant produced {} f32s, shape product {}",
                        shard.display(),
                        data.len(),
                        expect
                    )));
                }
                tensors.push(HfTensor {
                    name: name.to_string(),
                    shape,
                    source_dtype,
                    data,
                });
            }
        }

        Ok(HfModelSource { tensors, config })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;

    fn write_minimal_single_file(
        dir: &Path,
        tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)],
        config: &serde_json::Value,
    ) {
        let views: Vec<(String, TensorView<'_>)> = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                let v = TensorView::new(*dtype, shape.clone(), bytes).expect("TensorView");
                (name.to_string(), v)
            })
            .collect();
        let view_refs: Vec<(String, &TensorView<'_>)> =
            views.iter().map(|(n, v)| (n.clone(), v)).collect();
        let bytes = safetensors::tensor::serialize(view_refs, None).expect("serialize");
        fs::write(dir.join("model.safetensors"), bytes).expect("write safetensors");
        fs::write(
            dir.join("config.json"),
            serde_json::to_string_pretty(config).unwrap(),
        )
        .expect("write config.json");
    }

    #[test]
    fn load_single_file_f32_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let f32_bytes: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
        write_minimal_single_file(
            dir.path(),
            &[("model.norm.weight", Dtype::F32, vec![6], f32_bytes)],
            &serde_json::json!({ "model_type": "llama" }),
        );

        let src = HfModelSource::load(dir.path()).expect("load");
        assert_eq!(src.tensors.len(), 1);
        let t = &src.tensors[0];
        assert_eq!(t.name, "model.norm.weight");
        assert_eq!(t.shape, vec![6]);
        assert_eq!(t.source_dtype, SourceDtype::F32);
        assert_eq!(t.data, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(src.config["model_type"], "llama");
    }

    #[test]
    fn load_dequantizes_f16_and_bf16() {
        let dir = tempfile::tempdir().unwrap();
        let f16_vals: Vec<f32> = vec![1.5, -0.5, 2.0, 0.25];
        let f16_bytes: Vec<u8> = f16_vals
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        let bf16_vals: Vec<f32> = vec![1.0, -2.0, 0.5];
        let bf16_bytes: Vec<u8> = bf16_vals
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        write_minimal_single_file(
            dir.path(),
            &[
                ("a.weight", Dtype::F16, vec![4], f16_bytes),
                ("b.weight", Dtype::BF16, vec![3], bf16_bytes),
            ],
            &serde_json::json!({}),
        );

        let src = HfModelSource::load(dir.path()).expect("load");
        let by_name: std::collections::HashMap<&str, &HfTensor> =
            src.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

        let a = by_name["a.weight"];
        assert_eq!(a.source_dtype, SourceDtype::F16);
        assert_eq!(a.shape, vec![4]);
        for (got, want) in a.data.iter().zip(f16_vals.iter()) {
            assert!((got - want).abs() < 1e-3, "F16 round-trip drift");
        }

        let b = by_name["b.weight"];
        assert_eq!(b.source_dtype, SourceDtype::BF16);
        assert_eq!(b.shape, vec![3]);
        for (got, want) in b.data.iter().zip(bf16_vals.iter()) {
            assert!((got - want).abs() < 0.05 * want.abs().max(1.0), "BF16 drift");
        }
    }

    #[test]
    fn load_unsupported_dtype_errors_typed() {
        // U8 is a valid safetensors dtype but out of source-reader scope —
        // must surface as `UnsupportedSourceDtype`, never silently cast.
        let dir = tempfile::tempdir().unwrap();
        let u8_bytes: Vec<u8> = vec![1, 2, 3, 4];
        write_minimal_single_file(
            dir.path(),
            &[("weird.weight", Dtype::U8, vec![4], u8_bytes)],
            &serde_json::json!({}),
        );

        let err = HfModelSource::load(dir.path()).expect_err("must error");
        match err {
            SourceError::UnsupportedSourceDtype { tensor, .. } => {
                assert_eq!(tensor, "weird.weight");
            }
            other => panic!("expected UnsupportedSourceDtype, got {other:?}"),
        }
    }

    #[test]
    fn load_missing_config_errors() {
        let dir = tempfile::tempdir().unwrap();
        let f32_bytes: Vec<u8> = (0..4).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, vec![4], &f32_bytes).unwrap();
        let bytes =
            safetensors::tensor::serialize(vec![("a.weight".to_string(), &view)], None).unwrap();
        fs::write(dir.path().join("model.safetensors"), bytes).unwrap();
        // No config.json written.
        let err = HfModelSource::load(dir.path()).expect_err("must error");
        match err {
            SourceError::Io(_) => {}
            other => panic!("expected Io error for missing config.json, got {other:?}"),
        }
    }
}
