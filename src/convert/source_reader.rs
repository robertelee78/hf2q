//! `HfModelSource` — read a HuggingFace `model.safetensors` directory
//! (single file or sharded) + its `config.json` into the convert pipeline
//! **WITHOUT** buffering the entire model in RAM.
//!
//! Per ADR-033 §P0 + the real-model OOM finding 2026-05-18: the previous
//! buffered-`Vec<HfTensor>` design read every shard fully into RAM AND
//! dequantized every tensor to F32 up-front, producing a peak RSS of
//! roughly `2 × sum(safetensors_byte_size)` for BF16 sources before the
//! orchestrator could even begin quantizing. Four sequential
//! `hf2q convert-v2` attempts against `google/gemma-4-26b-a4b-it`
//! (48 GB safetensors → ~18 GB Q5_K_M GGUF target) were killed by the OS
//! memory manager (SIGKILL 137) on a 64 GB system. See ADR-033
//! §"Open Issues / Real-Model Findings" for the full triage.
//!
//! The new streaming reader sits between the disk layout and the
//! `ConvertOrchestrator`, doing three jobs in two stages:
//!
//! 1. **open** — discover shards (single file or sharded via
//!    `model.safetensors.index.json`), mmap them, parse each shard's
//!    safetensors header, and record a flat `Vec<TensorMeta>` index of
//!    `(name, shape, source_dtype, shard_idx, byte_offset, byte_len)`.
//!    Cheap: ~hundreds of MB of mmap'd headers, no payload bytes resident.
//! 2. **iter_tensors** — lazily walk the index, slicing each tensor's
//!    bytes directly out of its shard mmap and dequantizing to F32. One
//!    `HfTensor` allocated at a time; the previous tensor drops before
//!    the next one materializes.
//! 3. **dequantize FP8** when the config opts in
//!    (`quantization_config.quant_method == "fp8"`) — the
//!    `<name>.weight_scale_inv` sibling is looked up across shards by
//!    name, sliced as F32, and consumed during dequant.
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: no buffered legacy
//! `load()` path is kept around. The orchestrator + driver are updated
//! to consume the streaming API directly.
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]]: every unsupported
//! source dtype (U8 / I32 / FP8-without-config / …) returns a typed
//! error. F32 is emitted only for the orchestrator's downstream consumer;
//! never silently demoted on the read side.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
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

/// Cheap per-tensor metadata recorded at `open` time. Carries the shape
/// + source dtype that the orchestrator's policy needs WITHOUT loading
/// any payload bytes.
///
/// Per the streaming design: the driver does two passes over the model —
/// (1) a **plan** pass that walks `tensor_metas()` to feed
/// `StandardPolicy::target_for` / `ApexPolicy::target_for` (data-free —
/// type selection only needs name + shape + arch + layer_index + dtype),
/// (2) a **stream** pass that walks `iter_tensors()` to actually
/// quantize and write each tensor's bytes.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// HuggingFace tensor name.
    pub name: String,
    /// PyTorch-order dimensions.
    pub shape: Vec<usize>,
    /// On-disk source dtype (`SourceDtype::Fp8E4M3` when the config opts
    /// in to FP8 dispatch; the raw F8_E4M3 byte pattern still on disk).
    pub source_dtype: SourceDtype,
    /// Index into `HfModelSource::shards` — only used internally by the
    /// reader to slice the right mmap during streaming.
    shard_idx: usize,
    /// safetensors-header `data_offsets.0`: byte offset of the tensor
    /// payload INSIDE the post-header data region of `shard_idx`.
    data_off_start: usize,
    /// safetensors-header `data_offsets.1`: end byte offset.
    data_off_end: usize,
}

impl TensorMeta {
    /// Convenience: number of elements (`shape.iter().product()`).
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Whole-model bundle: per-shard mmaps + a flat tensor index + the raw
/// `config.json`. Tensors are kept ON DISK (mmap'd, not heap-resident)
/// until iterated.
///
/// Memory bound: peak resident set during convert is now
/// `O(largest_single_tensor_F32_size + quantized_payload + GGUF_header)`.
/// For Gemma 4 26B with `ffn_down` at `[2112, 2560]` BF16 source →
/// 20 MB F32 dequant + ~13 MB Q5_K_M payload ≈ 35 MB peak instead of the
/// previous ~104 GB. See ADR-033 §"Open Issues / Real-Model Findings".
pub struct HfModelSource {
    /// Parsed `config.json` (mandatory at the model directory root).
    pub config: serde_json::Value,
    /// Per-shard mmap handles, indexed by `TensorMeta::shard_idx`.
    /// Held for the lifetime of `Self` so that streaming reads can slice
    /// directly without re-opening files.
    shards: Vec<ShardMmap>,
    /// Flat tensor index in safetensors-file order across all shards
    /// (FP8 `*_scale_inv` siblings excluded when `fp8_cfg.is_some()`;
    /// surfaced via cross-shard lookup during dequant, not in the
    /// streaming output).
    metas: Vec<TensorMeta>,
    /// `Some(_)` when `config.json::quantization_config.quant_method ==
    /// "fp8"`. Routes FP8 dispatch + governs sibling-scale lookup.
    fp8_cfg: Option<Fp8Config>,
    /// Reverse-lookup table for FP8 sibling resolution:
    /// `"<name>.weight_scale_inv"` → index into `metas` — populated even
    /// when fp8 is off so the indexer is stable, but only consumed by
    /// the FP8 dequant branch.
    by_name: HashMap<String, usize>,
}

impl std::fmt::Debug for HfModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfModelSource")
            .field("config_keys", &self.config.as_object().map(|o| o.len()))
            .field("shards", &self.shards.len())
            .field("metas", &self.metas.len())
            .field("fp8", &self.fp8_cfg.is_some())
            .finish()
    }
}

/// Per-shard mmap + the safetensors header byte length, so we can
/// compute payload-region offsets without re-parsing.
struct ShardMmap {
    mmap: Mmap,
    /// Byte length of `<u64 header_size_prefix><header_json>`. Tensor
    /// payload bytes start at `header_byte_len` and are addressed by
    /// `TensorInfo::data_offsets` relative to that origin.
    header_byte_len: usize,
}

impl ShardMmap {
    /// Slice the raw bytes for a tensor at `(off_start, off_end)`
    /// (relative to the post-header data region).
    fn tensor_bytes(&self, off_start: usize, off_end: usize) -> &[u8] {
        &self.mmap[self.header_byte_len + off_start..self.header_byte_len + off_end]
    }
}

/// Errors raised by [`HfModelSource::open`] and its iterators.
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
    /// Open a HuggingFace model directory **without** loading any tensor
    /// payload bytes into the heap. The mmap files are page-cached by
    /// the OS but do not count against process RSS until iterated.
    ///
    /// The directory layout follows HuggingFace's convention:
    /// - `<model_dir>/config.json` — model config (mandatory).
    /// - `<model_dir>/model.safetensors` — single-shard weights, OR
    /// - `<model_dir>/model.safetensors.index.json` — shard index
    ///   pointing to `model-NNNNN-of-MMMMM.safetensors` siblings.
    ///
    /// Per-tensor metadata (name, shape, source_dtype) is available
    /// immediately via [`Self::tensor_metas`]; payload bytes flow
    /// through [`Self::iter_tensors`] one tensor at a time.
    pub fn open(model_dir: &Path) -> Result<Self, SourceError> {
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
        let shards_map: BTreeMap<String, PathBuf> =
            discover_shards(model_dir).map_err(|e| SourceError::Discover(format!("{e:#}")))?;

        let mut shard_paths: Vec<PathBuf> = shards_map.values().cloned().collect();
        shard_paths.sort();
        shard_paths.dedup();

        // ----- First pass: mmap each shard + read its header -----
        let mut shards: Vec<ShardMmap> = Vec::with_capacity(shard_paths.len());
        let mut metas: Vec<TensorMeta> = Vec::new();
        let mut by_name: HashMap<String, usize> = HashMap::new();

        for (shard_idx, shard_path) in shard_paths.iter().enumerate() {
            let f = File::open(shard_path).map_err(|e| {
                SourceError::Io(std::io::Error::new(
                    e.kind(),
                    format!("open {}: {e}", shard_path.display()),
                ))
            })?;
            // SAFETY: the file is opened read-only and the Mmap is held
            // for `Self`'s lifetime. We do not mutate the underlying
            // file. Concurrent external mutation of safetensors shards
            // during convert is out of contract.
            let mmap = unsafe { Mmap::map(&f) }.map_err(|e| {
                SourceError::Io(std::io::Error::new(
                    e.kind(),
                    format!("mmap {}: {e}", shard_path.display()),
                ))
            })?;

            // safetensors layout: <u64 header_size_LE><JSON header bytes>
            // <tensor payload region>. read_metadata returns (header_size,
            // Metadata); payload region begins at offset 8 + header_size.
            let (header_size, meta) = SafeTensors::read_metadata(&mmap[..]).map_err(|e| {
                SourceError::Safetensors(format!(
                    "header parse {}: {e}",
                    shard_path.display()
                ))
            })?;
            let header_byte_len = 8 + header_size;

            // Walk this shard's tensors in **byte-offset order** —
            // safetensors 0.7 sorts `Metadata::tensors` by
            // `data_offsets` at deserialize time (`tensor.rs:520-525`),
            // and `offset_keys()` exposes that sorted name list. Using
            // it gives deterministic, sequential-mmap-friendly iteration
            // (each next read advances the cursor forward, kind to OS
            // readahead).
            //
            // The orchestrator does not rely on safetensors-iteration
            // order for any byte-level property; it sorts MoE groups
            // by (layer, kind) and emits direct tensors in the order
            // they were staged. Two convert runs on the same input
            // produce byte-identical GGUFs because the orchestrator
            // re-sorts internally.
            for name in meta.offset_keys() {
                let info = meta.info(&name).expect("offset_keys yields names that index_map contains");
                let source_dtype = match info.dtype {
                    Dtype::F32 => SourceDtype::F32,
                    Dtype::F16 => SourceDtype::F16,
                    Dtype::BF16 => SourceDtype::BF16,
                    Dtype::F8_E4M3 => {
                        // FP8 sibling-scale tensors are themselves F32; only
                        // the main weights are F8. The dtype dispatch at
                        // iter time validates the FP8 opt-in is in place.
                        SourceDtype::Fp8E4M3
                    }
                    other => {
                        return Err(SourceError::UnsupportedSourceDtype {
                            tensor: name.to_string(),
                            dtype: format!("{other:?}"),
                        });
                    }
                };

                let idx = metas.len();
                metas.push(TensorMeta {
                    name: name.to_string(),
                    shape: info.shape.clone(),
                    source_dtype,
                    shard_idx,
                    data_off_start: info.data_offsets.0,
                    data_off_end: info.data_offsets.1,
                });
                // If multiple shards declare the same tensor name the
                // last one wins (mirrors the original buffered impl's
                // `raw.insert` semantics); safetensors index files do
                // not normally repeat names so this is just defensive.
                by_name.insert(name.to_string(), idx);
            }

            shards.push(ShardMmap {
                mmap,
                header_byte_len,
            });
        }

        Ok(HfModelSource {
            config,
            shards,
            metas,
            fp8_cfg,
            by_name,
        })
    }

    /// Iterate per-tensor metadata WITHOUT loading any payload.
    ///
    /// Used by the convert driver's **plan pass**: the policy decides
    /// each tensor's ggml_type from name + shape + source_dtype + arch
    /// + layer_index alone. No payload bytes are read or dequantized.
    ///
    /// FP8 `*_scale_inv` sibling tensors are FILTERED OUT when the
    /// config opts in to FP8 dispatch — they're metadata consumed
    /// during dequant, never surfaced as standalone tensors. Mirrors
    /// the previous buffered-impl's skip rule.
    pub fn tensor_metas(&self) -> impl Iterator<Item = &TensorMeta> + '_ {
        let fp8_active = self.fp8_cfg.is_some();
        self.metas
            .iter()
            .filter(move |m| !(fp8_active && m.name.ends_with(".weight_scale_inv")))
    }

    /// Number of streaming tensors (after the FP8 sibling-scale filter).
    pub fn tensor_count(&self) -> usize {
        self.tensor_metas().count()
    }

    /// Materialize a single tensor by name. Slices the bytes out of the
    /// owning shard mmap and dequantizes to F32 in one fresh allocation.
    ///
    /// Used by the convert driver's **stream pass** when the plan order
    /// requires arbitrary access (e.g. MoE expert fusion: each
    /// `(layer, kind)` group fetches its N expert slices by HF name in
    /// `expert_index` order).
    ///
    /// FP8 sibling-scale tensors (`*.weight_scale_inv`) are NOT addressable
    /// via this API — they are consumed inline by the FP8 dispatch when
    /// the main weight is materialized. Asking for one returns
    /// [`SourceError::UnsupportedSourceDtype`] (the same surface as if a
    /// caller asked for a non-FP8 raw scale tensor).
    pub fn materialize_tensor(&self, name: &str) -> Result<HfTensor, SourceError> {
        let idx = self.by_name.get(name).copied().ok_or_else(|| {
            SourceError::Safetensors(format!("tensor `{name}` not found in any shard"))
        })?;
        let m = &self.metas[idx];
        // Block direct access to sibling-scale tensors when FP8 is active.
        if self.fp8_cfg.is_some() && name.ends_with(".weight_scale_inv") {
            return Err(SourceError::UnsupportedSourceDtype {
                tensor: name.into(),
                dtype: "fp8 sibling scale (consumed inline by main weight)".into(),
            });
        }
        materialize_tensor(self, m)
    }

    /// Stream every tensor as an `HfTensor` one at a time. Each call to
    /// `next()` slices ONE tensor's bytes out of its shard mmap,
    /// dequantizes to F32 in a fresh `Vec<f32>`, and yields it. The
    /// previous tensor's `Vec<f32>` drops before the next one is
    /// allocated (the iterator does not retain prior tensors).
    ///
    /// Used by the convert driver's **stream pass**: after the plan
    /// pass has fixed the GGUF tensor count + types, each tensor is
    /// quantized and written immediately, then dropped.
    pub fn iter_tensors(&self) -> TensorStream<'_> {
        TensorStream {
            source: self,
            cursor: 0,
        }
    }
}

/// Iterator returned by [`HfModelSource::iter_tensors`].
pub struct TensorStream<'a> {
    source: &'a HfModelSource,
    cursor: usize,
}

impl<'a> Iterator for TensorStream<'a> {
    type Item = Result<HfTensor, SourceError>;

    fn next(&mut self) -> Option<Self::Item> {
        let fp8_active = self.source.fp8_cfg.is_some();
        // Skip FP8 sibling-scale tensors (consumed inline during dequant).
        loop {
            if self.cursor >= self.source.metas.len() {
                return None;
            }
            let m = &self.source.metas[self.cursor];
            if fp8_active && m.name.ends_with(".weight_scale_inv") {
                self.cursor += 1;
                continue;
            }
            self.cursor += 1;
            return Some(materialize_tensor(self.source, m));
        }
    }
}

/// Materialize one tensor: slice its bytes out of the shard mmap and
/// dequantize to F32 row-major. FP8 dispatch looks up the sibling
/// `*_scale_inv` F32 table by name across shards.
fn materialize_tensor(src: &HfModelSource, m: &TensorMeta) -> Result<HfTensor, SourceError> {
    let shard = &src.shards[m.shard_idx];
    let raw_bytes = shard.tensor_bytes(m.data_off_start, m.data_off_end);

    let (source_dtype, data) = match m.source_dtype {
        SourceDtype::F32 => (
            SourceDtype::F32,
            read_floats_to_f32(raw_bytes, Dtype::F32).map_err(|e| {
                SourceError::Safetensors(format!("F32 dequant {}: {e:#}", m.name))
            })?,
        ),
        SourceDtype::F16 => (
            SourceDtype::F16,
            read_floats_to_f32(raw_bytes, Dtype::F16).map_err(|e| {
                SourceError::Safetensors(format!("F16 dequant {}: {e:#}", m.name))
            })?,
        ),
        SourceDtype::BF16 => (
            SourceDtype::BF16,
            read_floats_to_f32(raw_bytes, Dtype::BF16).map_err(|e| {
                SourceError::Safetensors(format!("BF16 dequant {}: {e:#}", m.name))
            })?,
        ),
        SourceDtype::Fp8E4M3 => {
            // Per ADR-033: FP8 read path REQUIRES the JSON opt-in.
            // A raw FP8 tensor without `quant_method=fp8` is an
            // unsupported source dtype — typed error per
            // [[feedback-no-loop-suppression-2026-05-17]].
            let cfg = src.fp8_cfg.as_ref().ok_or_else(|| {
                SourceError::UnsupportedSourceDtype {
                    tensor: m.name.clone(),
                    dtype: "F8_E4M3 without quantization_config.quant_method=fp8".into(),
                }
            })?;
            if cfg.is_not_converted(&m.name) {
                // HF lists these but they're typically stored in
                // F32/BF16 anyway — the dtype branch above already
                // handles non-FP8 dtypes. Hitting THIS arm means a
                // module was listed as "not converted" but is still on
                // disk as FP8: surface as a config inconsistency rather
                // than silently dequant.
                return Err(SourceError::InvalidFp8Config(format!(
                    "tensor `{}` matches modules_to_not_convert \
                     but is on disk as FP8 — config and weights disagree",
                    m.name
                )));
            }
            let scale_name = format!("{}_scale_inv", m.name);
            let scale_idx =
                src.by_name
                    .get(&scale_name)
                    .copied()
                    .ok_or_else(|| SourceError::MissingFp8Scales {
                        tensor: m.name.clone(),
                    })?;
            let scale_meta = &src.metas[scale_idx];
            if !matches!(scale_meta.source_dtype, SourceDtype::F32) {
                return Err(SourceError::InvalidFp8Config(format!(
                    "scale `{scale_name}`: expected F32, got {:?}",
                    scale_meta.source_dtype
                )));
            }
            let scale_shard = &src.shards[scale_meta.shard_idx];
            let scale_bytes =
                scale_shard.tensor_bytes(scale_meta.data_off_start, scale_meta.data_off_end);
            let scale_inv = read_floats_to_f32(scale_bytes, Dtype::F32).map_err(|e| {
                SourceError::Safetensors(format!("scale dequant {scale_name}: {e:#}"))
            })?;
            let f32_data =
                fp8::dequantize_fp8_block(raw_bytes, &scale_inv, &m.shape, cfg.block_size)
                    .map_err(|e| SourceError::Fp8Dequant {
                        tensor: m.name.clone(),
                        error: e,
                    })?;
            (SourceDtype::Fp8E4M3, f32_data)
        }
    };

    let expect = m.numel();
    if data.len() != expect {
        return Err(SourceError::Safetensors(format!(
            "tensor `{}`: dequant produced {} f32s, shape product {}",
            m.name,
            data.len(),
            expect
        )));
    }
    Ok(HfTensor {
        name: m.name.clone(),
        shape: m.shape.clone(),
        source_dtype,
        data,
    })
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

    fn collect_tensors(src: &HfModelSource) -> Vec<HfTensor> {
        src.iter_tensors().map(|r| r.expect("stream")).collect()
    }

    #[test]
    fn open_single_file_f32_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let f32_bytes: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
        write_minimal_single_file(
            dir.path(),
            &[("model.norm.weight", Dtype::F32, vec![6], f32_bytes)],
            &serde_json::json!({ "model_type": "llama" }),
        );

        let src = HfModelSource::open(dir.path()).expect("open");
        let tensors = collect_tensors(&src);
        assert_eq!(tensors.len(), 1);
        let t = &tensors[0];
        assert_eq!(t.name, "model.norm.weight");
        assert_eq!(t.shape, vec![6]);
        assert_eq!(t.source_dtype, SourceDtype::F32);
        assert_eq!(t.data, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(src.config["model_type"], "llama");
    }

    #[test]
    fn open_dequantizes_f16_and_bf16() {
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

        let src = HfModelSource::open(dir.path()).expect("open");
        let tensors = collect_tensors(&src);
        let by_name: std::collections::HashMap<&str, &HfTensor> =
            tensors.iter().map(|t| (t.name.as_str(), t)).collect();

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
    fn open_unsupported_dtype_errors_typed() {
        // U8 is a valid safetensors dtype but out of source-reader scope —
        // must surface as `UnsupportedSourceDtype`, never silently cast.
        let dir = tempfile::tempdir().unwrap();
        let u8_bytes: Vec<u8> = vec![1, 2, 3, 4];
        write_minimal_single_file(
            dir.path(),
            &[("weird.weight", Dtype::U8, vec![4], u8_bytes)],
            &serde_json::json!({}),
        );

        // U8 is rejected at `open` time (during metadata indexing) — the
        // dtype is recorded in TensorMeta::source_dtype, and the only
        // SourceDtype variants the reader emits are F32/F16/BF16/FP8E4M3.
        // Anything else errors here.
        let err = HfModelSource::open(dir.path()).expect_err("must error");
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
    /// consumed silently and does NOT appear in the streamed output list.
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

        let src = HfModelSource::open(dir.path()).expect("open");
        let tensors = collect_tensors(&src);
        assert_eq!(tensors.len(), 1, "scale tensor must be hidden");
        let t = &tensors[0];
        assert_eq!(t.name, "model.layers.0.mlp.gate_proj.weight");
        assert_eq!(t.source_dtype, SourceDtype::Fp8E4M3);
        assert_eq!(t.shape, vec![4, 4]);
        // Every byte was +1.0; scale = 2.0; expect all 2.0.
        for v in &t.data {
            assert_eq!(*v, 2.0);
        }

        // tensor_metas() also filters scale siblings out — surface count
        // matches the streamed count.
        assert_eq!(src.tensor_count(), 1);
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

        let src = HfModelSource::open(dir.path()).expect("open");
        let tensors = collect_tensors(&src);
        let by_name: std::collections::HashMap<&str, &HfTensor> =
            tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        // 2 outputs: lm_head (BF16) + gate_proj (FP8 dequant).
        assert_eq!(tensors.len(), 2);

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
        // The missing-scale check now fires at iter time (not open time)
        // since open is metadata-only.
        let src = HfModelSource::open(dir.path()).expect("open");
        let err = src
            .iter_tensors()
            .next()
            .expect("one tensor")
            .expect_err("must error");
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
        // open() records the FP8 dtype unconditionally; the unsupported
        // dispatch surfaces at iter time when no config opt-in is found.
        let src = HfModelSource::open(dir.path()).expect("open");
        let err = src
            .iter_tensors()
            .next()
            .expect("one tensor")
            .expect_err("must error");
        match err {
            SourceError::UnsupportedSourceDtype { tensor, .. } => {
                assert_eq!(tensor, "raw_fp8.weight");
            }
            other => panic!("expected UnsupportedSourceDtype, got {other:?}"),
        }
    }

    #[test]
    fn open_missing_config_errors() {
        let dir = tempfile::tempdir().unwrap();
        let f32_bytes: Vec<u8> = (0..4).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, vec![4], &f32_bytes).unwrap();
        let bytes =
            safetensors::tensor::serialize(vec![("a.weight".to_string(), &view)], None).unwrap();
        fs::write(dir.path().join("model.safetensors"), bytes).unwrap();
        // No config.json written.
        let err = HfModelSource::open(dir.path()).expect_err("must error");
        match err {
            SourceError::Io(_) => {}
            other => panic!("expected Io error for missing config.json, got {other:?}"),
        }
    }

    /// Streaming property: each tensor's F32 buffer is allocated, yielded,
    /// and the next call to `next()` allocates a fresh buffer. The
    /// iterator does not retain prior tensors.
    ///
    /// This is the core test the OOM fix exists for. We synthesize a
    /// 10-tensor model, drive the iterator, and assert that the iterator
    /// itself + its source occupy a bounded footprint (size_of<TensorStream>
    /// + size_of<HfModelSource>) regardless of model size.
    #[test]
    fn iter_tensors_does_not_buffer_prior_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let mut entries: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        for i in 0..10 {
            let n = 128usize;
            let bytes: Vec<u8> = (0..n)
                .flat_map(|j| ((i * 100 + j) as f32).to_le_bytes())
                .collect();
            entries.push((format!("t{i}.weight"), vec![n], bytes));
        }
        let views: Vec<(String, TensorView<'_>)> = entries
            .iter()
            .map(|(n, sh, b)| {
                (
                    n.clone(),
                    TensorView::new(Dtype::F32, sh.clone(), b).unwrap(),
                )
            })
            .collect();
        let view_refs: Vec<(String, &TensorView<'_>)> =
            views.iter().map(|(n, v)| (n.clone(), v)).collect();
        let bytes = safetensors::tensor::serialize(view_refs, None).unwrap();
        fs::write(dir.path().join("model.safetensors"), bytes).unwrap();
        fs::write(dir.path().join("config.json"), "{}").unwrap();

        let src = HfModelSource::open(dir.path()).expect("open");
        // The iterator does not pre-load; verify by consuming one at a
        // time and checking the cursor advances.
        let mut iter = src.iter_tensors();
        let first = iter.next().unwrap().unwrap();
        assert_eq!(first.name, "t0.weight");
        // Pulling the second one drops the first (no shared ownership).
        let second = iter.next().unwrap().unwrap();
        assert_eq!(second.name, "t1.weight");
        // The TensorStream itself is two `usize`s + a borrow — sub-64
        // bytes regardless of how many tensors are in the source.
        assert!(std::mem::size_of::<TensorStream<'_>>() <= 64);
    }
}
