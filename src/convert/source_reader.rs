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

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::{tensor::Dtype, SafeTensors};

use crate::convert::source_dtype::fp8;
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
    /// Source-side tensor dtype not supported by the read path.
    /// Quantized U8 / I32 / I64 tensors are out of scope. FP8 IS
    /// supported when `config.json::quantization_config.quant_method
    /// == "fp8"` (auto-detect — see [`Fp8Config`]); a raw FP8 tensor
    /// without that opt-in surfaces here.
    UnsupportedSourceDtype {
        tensor: String,
        dtype: String,
    },
    /// FP8 tensor's sibling `<name>.weight_scale_inv` is missing.
    /// Required when `quant_method == "fp8"`.
    MissingFp8Scales { tensor: String },
    /// FP8 dequant failed for length / shape reasons.
    Fp8Dequant {
        tensor: String,
        error: fp8::Fp8Error,
    },
    /// FP8 config field is malformed (e.g. `weight_block_size` not a
    /// `[i, j]` pair).
    InvalidFp8Config(String),
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
                     (supported: F32, F16, BF16; FP8 only with quantization_config.quant_method=fp8)"
                )
            }
            SourceError::MissingFp8Scales { tensor } => write!(
                f,
                "source/fp8: tensor `{tensor}` is FP8 but its sibling \
                 `{tensor}_scale_inv` is missing from the safetensors shards"
            ),
            SourceError::Fp8Dequant { tensor, error } => {
                write!(f, "source/fp8: dequant `{tensor}`: {error}")
            }
            SourceError::InvalidFp8Config(s) => write!(f, "source/fp8 config: {s}"),
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

/// Parsed `quantization_config` block from `config.json` when
/// `quant_method == "fp8"`. Other quant_methods (e.g. "awq", "gptq")
/// are unsupported by the convert pipeline and surface as
/// [`SourceError::InvalidFp8Config`] only if their `quant_method`
/// string is present. Missing `quantization_config` means dense F16/BF16
/// load (no FP8 dispatch).
///
/// Per ADR-033 Decision §"FP8 source-dtype auto-detect": detection is
/// SILENT — the loader switches to the FP8 path automatically when the
/// config field is present.
#[derive(Debug, Clone)]
pub struct Fp8Config {
    /// `(block_rows, block_cols)` — HF `weight_block_size`. Common
    /// values: `[128, 128]` (DeepSeek-V3 / MiniMax-M2.7), `[1, 128]`.
    pub block_size: fp8::BlockSize,
    /// Tensor name suffixes listed in `modules_to_not_convert`.
    /// Tensors whose name CONTAINS any of these substrings are read in
    /// their native dtype (F32/BF16) WITHOUT FP8 dispatch. Convention
    /// in HF FP8 releases: `lm_head`, `embed_tokens`, layer-norm
    /// weights, biases.
    pub modules_to_not_convert: Vec<String>,
}

impl Fp8Config {
    /// Parse `config.json::quantization_config` if it declares FP8.
    /// Returns `Ok(None)` when there is no quantization_config or its
    /// `quant_method` is absent / not `"fp8"`. Returns
    /// `Err(InvalidFp8Config)` only for FP8-declared configs that are
    /// missing / malformed required fields.
    pub fn from_config(config: &serde_json::Value) -> Result<Option<Self>, SourceError> {
        let qc = match config.get("quantization_config") {
            Some(v) if v.is_object() => v,
            _ => return Ok(None),
        };
        let qm = match qc.get("quant_method").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Ok(None),
        };
        if qm != "fp8" {
            // Not an FP8 config — leave dispatch to the caller / future
            // codepaths. Don't error here; the per-tensor loader will
            // surface unsupported dtypes if they actually appear.
            return Ok(None);
        }

        // weight_block_size: [rows, cols]
        let wbs = qc.get("weight_block_size").ok_or_else(|| {
            SourceError::InvalidFp8Config(
                "quant_method=fp8 but `weight_block_size` is missing".into(),
            )
        })?;
        let arr = wbs.as_array().ok_or_else(|| {
            SourceError::InvalidFp8Config(format!(
                "`weight_block_size` must be a 2-element array, got {wbs:?}"
            ))
        })?;
        if arr.len() != 2 {
            return Err(SourceError::InvalidFp8Config(format!(
                "`weight_block_size` must be a 2-element array, got len {}",
                arr.len()
            )));
        }
        let parse_dim = |v: &serde_json::Value| -> Result<usize, SourceError> {
            v.as_u64().map(|x| x as usize).ok_or_else(|| {
                SourceError::InvalidFp8Config(format!(
                    "`weight_block_size` entries must be non-negative ints, got {v:?}"
                ))
            })
        };
        let block_size = (parse_dim(&arr[0])?, parse_dim(&arr[1])?);

        let modules_to_not_convert = qc
            .get("modules_to_not_convert")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(Some(Fp8Config {
            block_size,
            modules_to_not_convert,
        }))
    }

    /// True iff `tensor_name` matches one of the
    /// `modules_to_not_convert` substrings. HF convention is to list
    /// module-PATH substrings (e.g. `"lm_head"`) rather than full
    /// tensor names; we mirror Python's `any(m in name for m in ...)`.
    pub fn is_not_converted(&self, tensor_name: &str) -> bool {
        self.modules_to_not_convert
            .iter()
            .any(|m| tensor_name.contains(m.as_str()))
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

        // ----- FP8 auto-detect ---------------------------------------------
        // Per ADR-033 Decision §"FP8 source-dtype auto-detect": presence
        // of `quantization_config.quant_method == "fp8"` silently
        // switches dispatch.
        let fp8_cfg = Fp8Config::from_config(&config)?;

        // ----- shard discovery ---------------------------------------------
        let shards: BTreeMap<String, PathBuf> = discover_shards(model_dir)
            .map_err(|e| SourceError::Discover(format!("{e:#}")))?;

        let mut shard_paths: Vec<PathBuf> = shards.values().cloned().collect();
        shard_paths.sort();
        shard_paths.dedup();

        // ----- First pass: read every shard and index tensors by name -----
        // FP8 dispatch needs sibling lookup of `<name>.weight_scale_inv`,
        // which may live in ANOTHER shard. We buffer raw views per tensor
        // name (dtype, shape, byte-payload) so the dequant pass can
        // resolve cross-shard pairs uniformly.
        struct RawTensor {
            shape: Vec<usize>,
            dtype: Dtype,
            data: Vec<u8>,
        }
        let mut raw: HashMap<String, RawTensor> = HashMap::new();
        // Preserve safetensors-file order across all shards for the
        // emitted tensor sequence.
        let mut name_order: Vec<String> = Vec::new();

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
                let entry = RawTensor {
                    shape: view.shape().to_vec(),
                    dtype: view.dtype(),
                    data: view.data().to_vec(),
                };
                if !raw.contains_key(name) {
                    name_order.push(name.to_string());
                }
                raw.insert(name.to_string(), entry);
            }
        }

        // ----- Second pass: per-tensor dispatch ----------------------------
        // FP8 tensors emit ONE `HfTensor` per `<name>.weight` and SKIP
        // their `<name>.weight_scale_inv` siblings (the scales are
        // consumed during dequant, never surfaced to the orchestrator).
        // Non-FP8 (F32/F16/BF16) tensors take the elementwise read path.
        let mut tensors: Vec<HfTensor> = Vec::new();
        for name in &name_order {
            // Skip sibling scale tensors when we're in FP8 mode — they're
            // metadata consumed below.
            if fp8_cfg.is_some() && name.ends_with(".weight_scale_inv") {
                continue;
            }
            let rt = &raw[name];

            let (source_dtype, data) = match rt.dtype {
                Dtype::F32 => (
                    SourceDtype::F32,
                    read_floats_to_f32(&rt.data, Dtype::F32).map_err(|e| {
                        SourceError::Safetensors(format!("F32 dequant {name}: {e:#}"))
                    })?,
                ),
                Dtype::F16 => (
                    SourceDtype::F16,
                    read_floats_to_f32(&rt.data, Dtype::F16).map_err(|e| {
                        SourceError::Safetensors(format!("F16 dequant {name}: {e:#}"))
                    })?,
                ),
                Dtype::BF16 => (
                    SourceDtype::BF16,
                    read_floats_to_f32(&rt.data, Dtype::BF16).map_err(|e| {
                        SourceError::Safetensors(format!("BF16 dequant {name}: {e:#}"))
                    })?,
                ),
                Dtype::F8_E4M3 => {
                    // Per ADR-033: FP8 read path REQUIRES the JSON opt-in.
                    // A raw FP8 tensor without `quant_method=fp8` is an
                    // unsupported source dtype — typed error per
                    // [[feedback-no-loop-suppression-2026-05-17]].
                    let cfg = fp8_cfg.as_ref().ok_or_else(|| {
                        SourceError::UnsupportedSourceDtype {
                            tensor: name.clone(),
                            dtype: "F8_E4M3 without quantization_config.quant_method=fp8".into(),
                        }
                    })?;
                    if cfg.is_not_converted(name) {
                        // HF lists these but they're typically stored in
                        // F32/BF16 anyway — the check happens at the
                        // dtype branch above. Hitting THIS arm means a
                        // module was listed as "not converted" but is
                        // still on disk as FP8: surface as a config
                        // inconsistency rather than silently dequant.
                        return Err(SourceError::InvalidFp8Config(format!(
                            "tensor `{name}` matches modules_to_not_convert \
                             but is on disk as FP8 — config and weights disagree"
                        )));
                    }
                    let scale_name = format!("{}_scale_inv", name);
                    let scale_rt = raw
                        .get(&scale_name)
                        .ok_or_else(|| SourceError::MissingFp8Scales {
                            tensor: name.clone(),
                        })?;
                    if scale_rt.dtype != Dtype::F32 {
                        return Err(SourceError::InvalidFp8Config(format!(
                            "scale `{scale_name}`: expected F32, got {:?}",
                            scale_rt.dtype
                        )));
                    }
                    let scale_inv = read_floats_to_f32(&scale_rt.data, Dtype::F32)
                        .map_err(|e| SourceError::Safetensors(format!(
                            "scale dequant {scale_name}: {e:#}"
                        )))?;
                    let f32_data = fp8::dequantize_fp8_block(
                        &rt.data,
                        &scale_inv,
                        &rt.shape,
                        cfg.block_size,
                    )
                    .map_err(|e| SourceError::Fp8Dequant {
                        tensor: name.clone(),
                        error: e,
                    })?;
                    (SourceDtype::Fp8E4M3, f32_data)
                }
                other => {
                    return Err(SourceError::UnsupportedSourceDtype {
                        tensor: name.clone(),
                        dtype: format!("{other:?}"),
                    })
                }
            };

            let expect = rt.shape.iter().product::<usize>();
            if data.len() != expect {
                return Err(SourceError::Safetensors(format!(
                    "tensor `{name}`: dequant produced {} f32s, shape product {}",
                    data.len(),
                    expect
                )));
            }
            tensors.push(HfTensor {
                name: name.clone(),
                shape: rt.shape.clone(),
                source_dtype,
                data,
            });
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

    /// FP8 detection acceptance — config with `quant_method=fp8` triggers
    /// the FP8 dispatch path (FP8 tensor + sibling `weight_scale_inv` →
    /// one `HfTensor` with `source_dtype=Fp8E4M3`). The scale tensor is
    /// consumed silently and does NOT appear in the output list.
    #[test]
    fn fp8_config_detection() {
        let dir = tempfile::tempdir().unwrap();
        // 4x4 FP8 of +1.0 (0x38 = +1.0 in e4m3fn) — single 4x4 block.
        let fp8_bytes: Vec<u8> = vec![0x38; 16];
        // 1x1 inverse-scale grid for a 4x4 block.
        let scale_bytes: Vec<u8> = 2.0_f32.to_le_bytes().to_vec();
        write_minimal_single_file(
            dir.path(),
            &[
                ("model.layers.0.mlp.gate_proj.weight", Dtype::F8_E4M3, vec![4, 4], fp8_bytes),
                (
                    "model.layers.0.mlp.gate_proj.weight_scale_inv",
                    Dtype::F32,
                    vec![1, 1],
                    scale_bytes,
                ),
            ],
            &serde_json::json!({
                "model_type": "minimax_m2",
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": [4, 4],
                }
            }),
        );

        let src = HfModelSource::load(dir.path()).expect("load");
        assert_eq!(src.tensors.len(), 1, "scale tensor must be hidden");
        let t = &src.tensors[0];
        assert_eq!(t.name, "model.layers.0.mlp.gate_proj.weight");
        assert_eq!(t.source_dtype, SourceDtype::Fp8E4M3);
        assert_eq!(t.shape, vec![4, 4]);
        // Every byte was +1.0; scale = 2.0; expect all 2.0.
        for v in &t.data {
            assert_eq!(*v, 2.0);
        }
    }

    /// FP8 + modules_to_not_convert — a tensor whose name matches a
    /// "not converted" substring is read in its native dtype (here BF16)
    /// instead of going through the FP8 dispatch.
    #[test]
    fn fp8_modules_to_not_convert() {
        let dir = tempfile::tempdir().unwrap();
        let bf16_vals = [1.0_f32, -2.0, 0.5, 0.25];
        let bf16_bytes: Vec<u8> = bf16_vals
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        let fp8_bytes: Vec<u8> = vec![0x38; 4]; // 2x2 of +1.0
        let scale_bytes: Vec<u8> = 1.0_f32.to_le_bytes().to_vec();
        write_minimal_single_file(
            dir.path(),
            &[
                // lm_head listed as "not converted" → read as BF16
                ("lm_head.weight", Dtype::BF16, vec![2, 2], bf16_bytes),
                // gate_proj quantized as FP8 → goes through dequant
                (
                    "model.layers.0.mlp.gate_proj.weight",
                    Dtype::F8_E4M3,
                    vec![2, 2],
                    fp8_bytes,
                ),
                (
                    "model.layers.0.mlp.gate_proj.weight_scale_inv",
                    Dtype::F32,
                    vec![1, 1],
                    scale_bytes,
                ),
            ],
            &serde_json::json!({
                "model_type": "minimax_m2",
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": [2, 2],
                    "modules_to_not_convert": ["lm_head", "embed_tokens"],
                }
            }),
        );

        let src = HfModelSource::load(dir.path()).expect("load");
        let by_name: std::collections::HashMap<&str, &HfTensor> =
            src.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        // 2 outputs: lm_head (BF16) + gate_proj (FP8 dequant).
        assert_eq!(src.tensors.len(), 2);

        let head = by_name["lm_head.weight"];
        assert_eq!(head.source_dtype, SourceDtype::BF16);
        for (got, want) in head.data.iter().zip(bf16_vals.iter()) {
            assert!((got - want).abs() < 0.05 * want.abs().max(1.0));
        }

        let gate = by_name["model.layers.0.mlp.gate_proj.weight"];
        assert_eq!(gate.source_dtype, SourceDtype::Fp8E4M3);
        assert_eq!(gate.data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    /// FP8 config rejects a missing `weight_scale_inv` sibling.
    #[test]
    fn fp8_missing_scale_errors() {
        let dir = tempfile::tempdir().unwrap();
        let fp8_bytes: Vec<u8> = vec![0x38; 4];
        write_minimal_single_file(
            dir.path(),
            &[(
                "model.layers.0.mlp.gate_proj.weight",
                Dtype::F8_E4M3,
                vec![2, 2],
                fp8_bytes,
            )],
            &serde_json::json!({
                "model_type": "minimax_m2",
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": [2, 2],
                }
            }),
        );
        let err = HfModelSource::load(dir.path()).expect_err("must error");
        match err {
            SourceError::MissingFp8Scales { tensor } => {
                assert_eq!(tensor, "model.layers.0.mlp.gate_proj.weight");
            }
            other => panic!("expected MissingFp8Scales, got {other:?}"),
        }
    }

    /// FP8 dtype WITHOUT `quant_method=fp8` opt-in is an unsupported
    /// source dtype (no silent dequant).
    #[test]
    fn fp8_without_config_unsupported() {
        let dir = tempfile::tempdir().unwrap();
        let fp8_bytes: Vec<u8> = vec![0x38; 4];
        write_minimal_single_file(
            dir.path(),
            &[(
                "raw_fp8.weight",
                Dtype::F8_E4M3,
                vec![2, 2],
                fp8_bytes,
            )],
            &serde_json::json!({
                "model_type": "llama" // no quantization_config
            }),
        );
        let err = HfModelSource::load(dir.path()).expect_err("must error");
        match err {
            SourceError::UnsupportedSourceDtype { tensor, .. } => {
                assert_eq!(tensor, "raw_fp8.weight");
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
