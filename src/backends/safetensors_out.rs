//! Safetensors output backend — writes quantized models in the
//! safetensors format with the mlx-lm-style directory layout.
//!
//! Target consumers: mlx-lm, inferrs, Candle, vLLM, hf2q's own serve
//! loader.
//!
//! ## ADR-014 P9 iter-1 §S2/S3 — directory layout
//!
//! For quantized variants (everything except `--quant f16` / `--quant
//! bf16`) the backend emits a **directory** at `<output>/`:
//!
//! ```text
//! <output>/
//!   ├── config.json                              # injected (mlx-lm convention)
//!   ├── tokenizer.json + tokenizer_config.json   # copied by main::copy_sidecars
//!   ├── chat_template.jinja                      # copied (when present)
//!   ├── generation_config.json                   # copied (when present)
//!   ├── special_tokens_map.json                  # copied (when present)
//!   ├── quantization_config.json                 # legacy hf2q sidecar
//!   ├── model-NNNNN-of-MMMMM.safetensors         # OR model.safetensors (1 shard)
//!   └── model.safetensors.index.json             # only when MMMMM > 1
//! ```
//!
//! ### `config.json` injection (mlx-lm `save_config` schema)
//!
//! When a `config.json` is found in the **input** HF repo, it is read,
//! mutated to inject the mlx-lm `quantization` + mirrored
//! `quantization_config` keys (per `mlx_lm.utils.save_config:912-913`),
//! sorted, and written out.  When the input has no `config.json` the
//! injection step is silently skipped — the standalone `copy_sidecars`
//! step in `cmd_convert` (Phase 4.7) still byte-copies it later when
//! present.  This duplicate-handling is intentional: the backend owns
//! the *mutation* (mlx-lm schema), the sidecar copier owns the
//! *passthrough* of every other HF file.
//!
//! ### Shard naming (mlx-lm `save_model` convention)
//!
//! - `shards_count == 1` → `model.safetensors` (no index file)
//! - `shards_count > 1`  → `model-NNNNN-of-MMMMM.safetensors`
//!                          (5-digit zero-padded) + index file
//!
//! Match `mlx_lm.utils.save_model:727`. Existing single-shard unit
//! tests assert `model.safetensors` so deviating from the convention
//! would break Decision-17-adjacent byte-identity (AC12 in the iter
//! spec).
//!
//! ### DWQ tensor schema
//!
//! Per weight tensor `<name>.weight`: packed bits live in the
//! `<name>.weight` slot as opaque U8; companion `<name>.scales` (F16)
//! and `<name>.biases` (F16) tensors carry the unpacked codebook,
//! mirroring `mlx_lm.utils:154-155`.  `bits` + `group_size` live in
//! the top-level `config.json#quantization` block per mlx-lm
//! convention; per-tensor headers carry only the dtype tag.
//!
//! ### K-quant tensor schema
//!
//! K-quant variants (`q4_k_m` / `q5_k_m` / `q6_k` / `imatrix-*`) emit
//! **opaque GGUF block bytes** in the `<name>.weight` U8 slot. Per-
//! shard `__metadata__` carries a `quant_method = k_quant_q4_k_m | …`
//! discriminator + `block_size` so a downstream loader can route to
//! the right dequantizer. mlx-lm cannot load these natively (it would
//! TypeError on the U8 dtype mismatch); hf2q's own serve loader is
//! the consumer.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use tracing::{debug, info, warn};

use crate::backends::{BackendError, OutputBackend};
use crate::ir::{
    DType, FormatWarning, OutputFile, OutputManifest, QuantizedModel, WarningSeverity,
};
use crate::progress::ProgressReporter;

/// Default target shard size (5 GB — mlx-lm / HuggingFace community
/// convention).
const DEFAULT_SHARD_SIZE_BYTES: u64 = 5 * 1024 * 1024 * 1024;

/// Safetensors output backend.
pub struct SafetensorsBackend {
    /// Target shard size in bytes (configurable via
    /// `SafetensorsBackend::with_shard_size_gb`).
    shard_size_bytes: u64,
}

impl SafetensorsBackend {
    /// Construct with the default 5 GB shard target.
    pub fn new() -> Self {
        Self {
            shard_size_bytes: DEFAULT_SHARD_SIZE_BYTES,
        }
    }

    /// Construct with an explicit shard size in GB (ADR-014 P9 iter-1
    /// §S5). The CLI flag `--shard-size-gb` validates the
    /// `0.5..=50.0` range up front; any out-of-range or non-finite
    /// input here is clamped to the valid window so a buggy caller
    /// never produces a degenerate `shard_size_bytes = 0`.
    pub fn with_shard_size_gb(shard_size_gb: f64) -> Self {
        let clamped = if shard_size_gb.is_finite() {
            shard_size_gb.clamp(0.5, 50.0)
        } else {
            DEFAULT_SHARD_SIZE_BYTES as f64 / (1024.0 * 1024.0 * 1024.0)
        };
        let bytes = (clamped * (1024.0 * 1024.0 * 1024.0)) as u64;
        Self {
            shard_size_bytes: bytes,
        }
    }

    /// Map our DType to safetensors dtype string.
    fn dtype_to_safetensors(dtype: DType) -> &'static str {
        match dtype {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::U8 => "U8",
            DType::U16 => "U16",
            DType::U32 => "U32",
            DType::Bool => "BOOL",
        }
    }

    /// Predicate: does this `QuantizedModel`'s `quant_method` look like
    /// a DWQ variant? Used by `S3` schema dispatch — DWQ tensors get
    /// `<name>.scales` + `<name>.biases` companion tensors per the
    /// mlx-lm convention; K-quant tensors do not.
    ///
    /// Matches the on-disk strings produced at
    /// `src/calibrate/dwq.rs:370` (`format!("dwq-mixed-{base}-{sens}")`)
    /// and the kebab-form CLI strings.
    fn is_dwq_method(quant_method: &str) -> bool {
        let lc = quant_method.to_ascii_lowercase();
        lc.starts_with("dwq")
    }

    /// Predicate: K-quant block-byte family (uncalibrated +
    /// imatrix-calibrated). The on-disk method strings here are the
    /// `Quantizer::name()` returns from `KQuantCodecQuantizer` /
    /// `VariantKQuantizer` — per `cli::QuantMethod::Display` they're
    /// `q4_k_m`, `q5_k_m`, `q6_k`, `imatrix-q4_k_m`,
    /// `imatrix-q5_k_m`, `imatrix-q6_k`, `imatrix-adaptive`.
    fn is_kquant_method(quant_method: &str) -> bool {
        let lc = quant_method.to_ascii_lowercase();
        lc.starts_with("imatrix-q")
            || lc == "imatrix-adaptive"
            || lc == "q4_k_m"
            || lc == "q5_k_m"
            || lc == "q6_k"
    }

    /// Predicate: passthrough float (f16/bf16) — single-file emit per
    /// Decision 17 byte-identity.
    ///
    /// We classify by the *aggregate* model dtype (`bits == 16` AND
    /// every tensor preserved or marked f16). The
    /// `QuantizedModel::bits` field carries the global bit width set
    /// by the StaticQuantizer at `src/quantize/static_quant.rs:144`.
    fn is_unquantized_float(model: &QuantizedModel) -> bool {
        if model.bits != 16 {
            return false;
        }
        // Every tensor must be either preserved (norms etc.) or
        // marked as f16 / bf16 passthrough. Any K-quant or DWQ tensor
        // (preserved=false + ggml_type or scales) flips the predicate.
        model.tensors.values().all(|t| {
            t.quant_info.preserved
                || t.quant_info.method == "f16"
                || t.quant_info.method == "bf16"
                || t.quant_info.method == "passthrough"
        })
    }

    /// Build the legacy `quantization_config.json` content. Retained
    /// alongside the new `config.json` injection so existing
    /// downstream consumers (and the pre-P9 unit tests) keep working
    /// — Chesterton's fence on the AC12 surface.
    fn build_quant_config(model: &QuantizedModel) -> serde_json::Value {
        let mut per_layer_bits = serde_json::Map::new();
        let mut quant_info_map = serde_json::Map::new();

        for (name, tensor) in &model.tensors {
            per_layer_bits.insert(
                name.clone(),
                serde_json::Value::Number(tensor.quant_info.bits.into()),
            );
            quant_info_map.insert(
                name.clone(),
                serde_json::json!({
                    "method": tensor.quant_info.method,
                    "bits": tensor.quant_info.bits,
                    "group_size": tensor.quant_info.group_size,
                    "preserved": tensor.quant_info.preserved,
                }),
            );
        }

        serde_json::json!({
            "quant_method": model.quant_method,
            "bits": model.bits,
            "group_size": model.group_size,
            "per_layer_bits": per_layer_bits,
            "quantization_info": quant_info_map,
        })
    }

    /// Build the mlx-lm-style `quantization` block injected into the
    /// top-level `config.json` (per `mlx_lm.utils.save_config:912-913`
    /// and `quantize_model:812-819`).
    ///
    /// For DWQ: `{group_size, bits, mode: "affine"}` matching the
    /// `defaults_for_mode("affine") = (64, 4)` defaults.
    /// For K-quant: a single `quant_method` discriminator key (mlx-lm
    /// can't load these but the discriminator surfaces the right
    /// loader-routing tag for hf2q's serve path).
    fn build_mlx_lm_quantization_block(model: &QuantizedModel) -> serde_json::Value {
        if Self::is_dwq_method(&model.quant_method) {
            serde_json::json!({
                "group_size": model.group_size,
                "bits": model.bits,
                "mode": "affine",
            })
        } else if Self::is_kquant_method(&model.quant_method) {
            serde_json::json!({
                "quant_method": format!("k_quant_{}", model.quant_method),
            })
        } else {
            // Flat quant variants (q2/q4/q8) — emit a generic block so
            // the field is present and discoverable by the loader.
            serde_json::json!({
                "group_size": model.group_size,
                "bits": model.bits,
                "mode": "affine",
            })
        }
    }

    /// Read the input HF `config.json`, mutate it to inject the
    /// mlx-lm `quantization` + mirrored `quantization_config` keys
    /// (per `mlx_lm.utils.save_config:912-913`), and write to
    /// `<output>/config.json`. Sorted top-level keys for byte-stable
    /// diffs across runs.
    ///
    /// Silent no-op when the input has no `config.json` (small
    /// synthetic test fixtures do not always ship one). Real HF repos
    /// always do.
    fn write_injected_config_json(
        model: &QuantizedModel,
        input_dir: &Path,
        output_dir: &Path,
    ) -> Result<Option<OutputFile>, BackendError> {
        let input_config = input_dir.join("config.json");
        if !input_config.exists() {
            debug!(
                input_dir = %input_dir.display(),
                "No config.json in input dir — skipping mlx-lm injection \
                 (fixture or non-HF-shaped input)"
            );
            return Ok(None);
        }

        let raw = fs::read_to_string(&input_config)?;
        let mut parsed: serde_json::Value = serde_json::from_str(&raw)?;
        let obj = match parsed.as_object_mut() {
            Some(o) => o,
            None => {
                warn!(
                    "Input config.json at {} is not a JSON object — skipping injection",
                    input_config.display()
                );
                return Ok(None);
            }
        };

        // mlx-lm cleans these unused keys before save (utils.py:910-911).
        obj.remove("_name_or_path");

        let quant_block = Self::build_mlx_lm_quantization_block(model);
        obj.insert("quantization".to_string(), quant_block.clone());
        // utils.py:843 / utils.py:912-913 — mirror to quantization_config
        // so HF Transformers tooling sees the same payload.
        obj.insert("quantization_config".to_string(), quant_block);

        // Sorted output for byte-stable, diff-friendly artifacts (mlx-lm
        // `save_config` does the same sort at utils.py:916).
        let sorted: BTreeMap<String, serde_json::Value> = obj.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let pretty = serde_json::to_string_pretty(&sorted)
            .map_err(BackendError::Serialization)?;

        let dst = output_dir.join("config.json");
        fs::write(&dst, &pretty)?;

        Ok(Some(OutputFile {
            filename: "config.json".to_string(),
            size_bytes: pretty.len() as u64,
        }))
    }

    /// Build the per-shard `__metadata__` map for the safetensors
    /// header.
    ///
    /// For K-quant variants the metadata carries the `quant_method`
    /// discriminator + canonical block sizes so a downstream loader
    /// can route to the right dequantizer without sniffing the
    /// payload bytes.
    fn build_header_metadata(model: &QuantizedModel) -> BTreeMap<String, String> {
        let mut meta = BTreeMap::new();
        meta.insert("format".to_string(), "hf2q".to_string());
        meta.insert("quant_method".to_string(), model.quant_method.clone());
        meta.insert("bits".to_string(), model.bits.to_string());
        meta.insert("group_size".to_string(), model.group_size.to_string());
        meta.insert(
            "architecture".to_string(),
            model.metadata.architecture.clone(),
        );
        meta.insert(
            "tensor_count".to_string(),
            model.tensors.len().to_string(),
        );

        // ADR-014 P9 iter-1 §S3 — K-quant discriminator. The block
        // sizes are fixed by the GGUF spec (Q4_K = 144 B, Q5_K = 176 B,
        // Q6_K = 210 B, Q3_K = 110 B, Q8_0 = 34 B); recording them up
        // front means the loader does not need to sniff each tensor.
        if Self::is_kquant_method(&model.quant_method) {
            meta.insert(
                "k_quant_method".to_string(),
                format!("k_quant_{}", model.quant_method),
            );
        }

        meta
    }

    /// Collect tensor entries with the **mlx-lm DWQ schema** for the
    /// directory-layout path: each non-preserved DWQ tensor produces
    /// `<name>.weight` (U8 packed bits, 1-D byte vector), `<name>.scales`
    /// (F16, shape derived from `data.len()/2`), `<name>.biases` (F16,
    /// shape derived from `data.len()/2`).
    ///
    /// K-quant tensors emit the opaque packed bytes as `<name>.weight`
    /// only (the `.weight` is treated as U8 by mlx-lm — they cannot
    /// load it, but the bytes round-trip through a vanilla safetensors
    /// reader so hf2q's own serve loader can decode them later).
    ///
    /// Preserved tensors keep their original dtype.
    ///
    /// Returned tuples own their per-tensor byte buffers (`Vec<u8>`)
    /// rather than borrowing from `model` — when DWQ scales/biases need
    /// synthesised zero-buffers (symmetric DWQ → mlx-lm
    /// affine-schema-required `<name>.biases` of all zeros) the
    /// shape-promotion step needs the entries to outlive the per-loop
    /// borrow. The owned-bytes type alias is the simplest way to keep
    /// both code paths uniform without a lifetime gymnastics dance.
    fn collect_tensor_entries(
        model: &QuantizedModel,
    ) -> Vec<(String, &'static str, Vec<usize>, Vec<u8>)> {
        let mut entries: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();

        let mut names: Vec<&String> = model.tensors.keys().collect();
        names.sort();

        let dwq_model = Self::is_dwq_method(&model.quant_method);

        for name in names {
            let tensor = &model.tensors[name];

            let dtype_str = if tensor.quant_info.preserved {
                Self::dtype_to_safetensors(tensor.original_dtype)
            } else {
                "U8"
            };

            let shape = if tensor.quant_info.preserved {
                tensor.shape.clone()
            } else {
                vec![tensor.data.len()]
            };

            entries.push((name.clone(), dtype_str, shape, tensor.data.clone()));

            // For preserved tensors there is nothing more to emit.
            if tensor.quant_info.preserved {
                continue;
            }

            // Companion `<name>.scales` tensor (F16) — DWQ + legacy
            // q2/q4/q8. K-quant variants pack scales inline so they
            // never set `quant_info.scales`.
            //
            // `.len() / 2` because each scale is a half::f16
            // (2 bytes). 1-D shape — mlx-lm reshapes from
            // shape[-1]//group_size; preserving the flat layout
            // avoids the safetensors reader needing to know group
            // shape at decode time.
            let scales_present = tensor.quant_info.scales.is_some();
            if let Some(ref scales) = tensor.quant_info.scales {
                let scale_name = format!("{}.scales", name);
                let scale_shape = vec![scales.len() / 2];
                entries.push((scale_name, "F16", scale_shape, scales.clone()));
            }

            // Companion `<name>.biases` tensor (F16). The mlx-lm
            // affine-quantization schema requires both `.scales` AND
            // `.biases` per `mlx_lm.utils:154-155`. hf2q's current DWQ
            // path does **symmetric** weight-space quantization (no
            // asymmetric bias offset) so `quant_info.biases` is None,
            // but the on-disk layout still needs the `.biases` slot
            // for downstream loaders. Emit zero-filled F16 biases
            // matching the scales shape — this is the canonical
            // mlx-lm-compatible representation of a symmetric quant.
            if let Some(ref biases) = tensor.quant_info.biases {
                let bias_name = format!("{}.biases", name);
                let bias_shape = vec![biases.len() / 2];
                entries.push((bias_name, "F16", bias_shape, biases.clone()));
            } else if dwq_model && scales_present {
                let bias_name = format!("{}.biases", name);
                let n_groups = tensor
                    .quant_info
                    .scales
                    .as_ref()
                    .map(|s| s.len() / 2)
                    .unwrap_or(0);
                // F16 zero-bytes (= +0.0 in IEEE-754 half) — 2 bytes per
                // bias, n_groups biases.
                let zeros = vec![0u8; n_groups * 2];
                entries.push((bias_name, "F16", vec![n_groups], zeros));
            }
        }

        entries
    }

    /// Serialize a set of tensor entries into safetensors format bytes.
    /// Uses manual serialization compatible with the safetensors spec:
    /// [header_size: u64 LE] [header: JSON] [tensor data...]
    fn serialize_safetensors(
        entries: &[(String, &str, Vec<usize>, Vec<u8>)],
        metadata: &BTreeMap<String, String>,
    ) -> Result<Vec<u8>, BackendError> {
        let mut header = serde_json::Map::new();

        let meta_value = serde_json::to_value(metadata)
            .map_err(BackendError::Serialization)?;
        header.insert("__metadata__".to_string(), meta_value);

        let mut offset: usize = 0;
        let mut data_ranges: Vec<(usize, usize)> = Vec::with_capacity(entries.len());

        for (_, _, _, data) in entries {
            let start = offset;
            let end = start + data.len();
            data_ranges.push((start, end));
            offset = end;
        }

        for (i, (name, dtype, shape, _)) in entries.iter().enumerate() {
            let (start, end) = data_ranges[i];
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }

        let header_json =
            serde_json::to_string(&header).map_err(BackendError::Serialization)?;

        let header_bytes = header_json.as_bytes();
        let padding = (8 - (header_bytes.len() % 8)) % 8;
        let padded_header_len = header_bytes.len() + padding;

        let total_data_size: usize = entries.iter().map(|(_, _, _, d)| d.len()).sum();
        let total_size = 8 + padded_header_len + total_data_size;

        let mut buffer = Vec::with_capacity(total_size);

        buffer.extend_from_slice(&(padded_header_len as u64).to_le_bytes());

        buffer.extend_from_slice(header_bytes);
        buffer.extend(std::iter::repeat(b' ').take(padding));

        for (_, _, _, data) in entries {
            buffer.extend_from_slice(data);
        }

        Ok(buffer)
    }
}

impl Default for SafetensorsBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputBackend for SafetensorsBackend {
    fn name(&self) -> &str {
        "safetensors"
    }

    fn validate(&self, model: &QuantizedModel) -> Result<Vec<FormatWarning>, BackendError> {
        let mut warnings = Vec::new();

        for (name, tensor) in &model.tensors {
            if tensor.shape.is_empty() {
                warnings.push(FormatWarning {
                    message: format!("Tensor '{}' has empty shape", name),
                    severity: WarningSeverity::Warning,
                });
            }

            if tensor.quant_info.preserved && tensor.data.is_empty() {
                warnings.push(FormatWarning {
                    message: format!("Preserved tensor '{}' has no data", name),
                    severity: WarningSeverity::Warning,
                });
            }

            if !tensor.quant_info.preserved && tensor.data.is_empty() {
                warnings.push(FormatWarning {
                    message: format!("Quantized tensor '{}' has no data", name),
                    severity: WarningSeverity::Warning,
                });
            }
        }

        if model.tensors.is_empty() {
            warnings.push(FormatWarning {
                message: "Model has no tensors".to_string(),
                severity: WarningSeverity::Warning,
            });
        }

        Ok(warnings)
    }

    fn write(
        &self,
        model: &QuantizedModel,
        input_dir: &Path,
        output_dir: &Path,
        progress: &ProgressReporter,
    ) -> Result<OutputManifest, BackendError> {
        fs::create_dir_all(output_dir)?;

        let entries = Self::collect_tensor_entries(model);
        let metadata = Self::build_header_metadata(model);

        let total_data_size: u64 = entries.iter().map(|(_, _, _, d)| d.len() as u64).sum();
        let unquantized = Self::is_unquantized_float(model);
        info!(
            total_size_mb = total_data_size / (1024 * 1024),
            tensor_count = entries.len(),
            shard_size_mb = self.shard_size_bytes / (1024 * 1024),
            unquantized,
            quant_method = %model.quant_method,
            "Writing safetensors output"
        );

        let pb = progress.bar(entries.len() as u64, "Writing safetensors");

        let mut output_files = Vec::new();

        // mlx-lm convention (`save_model:727`): single shard ⇒
        // `model.safetensors` (no index file). Multi-shard ⇒
        // `model-NNNNN-of-MMMMM.safetensors` + index file.
        if total_data_size > self.shard_size_bytes {
            type ShardEntry = (String, &'static str, Vec<usize>, Vec<u8>);
            type Shard = Vec<ShardEntry>;

            let mut shards: Vec<Shard> = Vec::new();
            let mut current_shard: Shard = Vec::new();
            let mut current_size: u64 = 0;

            for entry in entries.into_iter() {
                let entry_size = entry.3.len() as u64;
                if !current_shard.is_empty()
                    && current_size + entry_size > self.shard_size_bytes
                {
                    shards.push(std::mem::take(&mut current_shard));
                    current_size = 0;
                }
                current_shard.push(entry);
                current_size += entry_size;
            }
            if !current_shard.is_empty() {
                shards.push(current_shard);
            }

            let total_shards = shards.len();
            let mut weight_map: BTreeMap<String, String> = BTreeMap::new();

            for (shard_idx, shard_entries) in shards.iter().enumerate() {
                let shard_num = shard_idx + 1;
                let filename =
                    format!("model-{:05}-of-{:05}.safetensors", shard_num, total_shards);

                debug!(shard = shard_num, tensors = shard_entries.len(), "Writing shard");

                for (name, _, _, _) in shard_entries {
                    weight_map.insert(name.clone(), filename.clone());
                }

                let shard_bytes =
                    Self::serialize_safetensors(shard_entries, &metadata)?;

                let shard_path = output_dir.join(&filename);
                fs::write(&shard_path, &shard_bytes)?;

                output_files.push(OutputFile {
                    filename: filename.clone(),
                    size_bytes: shard_bytes.len() as u64,
                });

                pb.inc(shard_entries.len() as u64);
            }

            let index = serde_json::json!({
                "metadata": {
                    "total_size": total_data_size,
                },
                "weight_map": weight_map,
            });
            let index_json = serde_json::to_string_pretty(&index)
                .map_err(BackendError::Serialization)?;
            let index_path = output_dir.join("model.safetensors.index.json");
            fs::write(&index_path, &index_json)?;

            output_files.push(OutputFile {
                filename: "model.safetensors.index.json".to_string(),
                size_bytes: index_json.len() as u64,
            });

            info!(shards = total_shards, "Multi-shard safetensors output written");
        } else {
            // Single-file output. Decision 17 byte-identity gate:
            // `--quant f16 --format safetensors` produces a
            // `model.safetensors` byte-equal to the pre-P9 single-file
            // emission. The header layout (`__metadata__` first,
            // tensors sorted by name, padded to 8-byte alignment, raw
            // tensor data appended in declaration order) is
            // unchanged from the pre-P9 path.
            let shard_bytes = Self::serialize_safetensors(&entries, &metadata)?;
            let output_path = output_dir.join("model.safetensors");
            fs::write(&output_path, &shard_bytes)?;

            output_files.push(OutputFile {
                filename: "model.safetensors".to_string(),
                size_bytes: shard_bytes.len() as u64,
            });

            pb.inc(entries.len() as u64);
            info!("Single-file safetensors output written");
        }

        // ADR-014 P9 iter-1 §S2 — mlx-lm-style `config.json` injection.
        // Only for quantized variants; f16/bf16 keeps the
        // copy_sidecars-only path so the config.json byte-identity
        // chain stays intact.
        if !unquantized {
            if let Some(of) = Self::write_injected_config_json(model, input_dir, output_dir)? {
                output_files.push(of);
            }
        }

        // Legacy hf2q `quantization_config.json` sidecar — retained for
        // existing downstream consumers (and AC12 unit-test surface).
        // Silent skip for the unquantized float path so f16/bf16
        // matches the mlx-lm "no quant config when unquantized"
        // convention.
        if !unquantized {
            let quant_config = Self::build_quant_config(model);
            let config_json = serde_json::to_string_pretty(&quant_config)
                .map_err(BackendError::Serialization)?;
            let config_path = output_dir.join("quantization_config.json");
            fs::write(&config_path, &config_json)?;

            output_files.push(OutputFile {
                filename: "quantization_config.json".to_string(),
                size_bytes: config_json.len() as u64,
            });
        }

        pb.finish_with_message("Safetensors output complete");

        let total_size_bytes: u64 = output_files.iter().map(|f| f.size_bytes).sum();
        let shard_count = output_files
            .iter()
            .filter(|f| {
                f.filename.ends_with(".safetensors")
                    && !f.filename.ends_with(".safetensors.index.json")
            })
            .count();

        Ok(OutputManifest {
            output_dir: output_dir.to_string_lossy().to_string(),
            files: output_files,
            total_size_bytes,
            shard_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::ir::{ModelMetadata, QuantizedModel, QuantizedTensor, TensorQuantInfo};

    fn make_metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: "TestArch".to_string(),
            model_type: "test".to_string(),
            param_count: 1000,
            hidden_size: 64,
            num_layers: 2,
            layer_types: vec!["attention".to_string()],
            num_attention_heads: 4,
            num_kv_heads: Some(4),
            vocab_size: 256,
            dtype: "float16".to_string(),
            shard_count: 1,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: Some(128),
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

    fn make_tensor(name: &str, data: Vec<u8>, preserved: bool) -> QuantizedTensor {
        QuantizedTensor {
            name: name.to_string(),
            shape: vec![4, 4],
            original_dtype: DType::F16,
            data,
            quant_info: TensorQuantInfo {
                method: if preserved {
                    "passthrough".to_string()
                } else {
                    "q4".to_string()
                },
                bits: if preserved { 16 } else { 4 },
                group_size: 32,
                preserved,
                scales: if preserved {
                    None
                } else {
                    Some(vec![0u8; 8])
                },
                biases: None,
                ggml_type: None,
            },
        }
    }

    fn make_model(tensors: Vec<QuantizedTensor>) -> QuantizedModel {
        let mut tensor_map = HashMap::new();
        for t in tensors {
            tensor_map.insert(t.name.clone(), t);
        }
        QuantizedModel {
            metadata: make_metadata(),
            tensors: tensor_map,
            quant_method: "q4_k".to_string(),
            group_size: 32,
            bits: 4,
        }
    }

    #[test]
    fn test_validate_well_formed_model() {
        let backend = SafetensorsBackend::new();
        let model = make_model(vec![
            make_tensor("weight_a", vec![1u8; 16], false),
            make_tensor("norm_b", vec![2u8; 32], true),
        ]);
        let warnings = backend.validate(&model).unwrap();
        assert!(warnings.is_empty(), "Well-formed model should produce no warnings");
    }

    #[test]
    fn test_validate_empty_model_warns() {
        let backend = SafetensorsBackend::new();
        let model = make_model(vec![]);
        let warnings = backend.validate(&model).unwrap();
        assert!(
            warnings.iter().any(|w| w.message.contains("no tensors")),
            "Empty model should produce a warning"
        );
    }

    #[test]
    fn test_write_single_file() {
        let backend = SafetensorsBackend::new();
        let model = make_model(vec![
            make_tensor("layer.0.weight", vec![0xAB; 32], false),
            make_tensor("layer.1.norm", vec![0xCD; 16], true),
        ]);

        let tmp = tempfile::tempdir().unwrap();
        let progress = ProgressReporter::new();
        let manifest = backend
            .write(&model, tmp.path(), tmp.path(), &progress)
            .unwrap();

        // Should produce model.safetensors + quantization_config.json
        assert_eq!(manifest.shard_count, 1);
        assert!(
            manifest.files.iter().any(|f| f.filename == "model.safetensors"),
            "Should write model.safetensors"
        );
        assert!(
            manifest
                .files
                .iter()
                .any(|f| f.filename == "quantization_config.json"),
            "Should write quantization_config.json"
        );

        // Verify the safetensors file is readable
        let st_path = tmp.path().join("model.safetensors");
        let st_bytes = fs::read(&st_path).unwrap();
        assert!(st_bytes.len() > 8, "Safetensors file should have header + data");

        // Verify header size is valid
        let header_size =
            u64::from_le_bytes(st_bytes[..8].try_into().unwrap()) as usize;
        assert!(header_size > 0, "Header size should be positive");
        assert!(
            8 + header_size <= st_bytes.len(),
            "Header size should not exceed file size"
        );

        // Verify header is valid JSON
        let header_str =
            std::str::from_utf8(&st_bytes[8..8 + header_size]).unwrap();
        let header: serde_json::Value =
            serde_json::from_str(header_str.trim()).unwrap();
        assert!(header.get("__metadata__").is_some(), "Should have __metadata__");
        assert!(
            header.get("layer.0.weight").is_some(),
            "Should have weight tensor in header"
        );

        // Verify quantization_config.json
        let config_path = tmp.path().join("quantization_config.json");
        let config: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(config["quant_method"], "q4_k");
        assert_eq!(config["bits"], 4);
        assert_eq!(config["group_size"], 32);
    }

    #[test]
    fn test_validate_empty_data_warns() {
        let backend = SafetensorsBackend::new();
        let model = make_model(vec![make_tensor("bad_tensor", vec![], false)]);
        let warnings = backend.validate(&model).unwrap();
        assert!(
            warnings
                .iter()
                .any(|w| w.message.contains("no data")),
            "Empty quantized data should warn"
        );
    }

    #[test]
    fn test_scales_written_as_separate_tensor() {
        let backend = SafetensorsBackend::new();
        let model = make_model(vec![make_tensor("proj.weight", vec![0xFF; 64], false)]);

        let tmp = tempfile::tempdir().unwrap();
        let progress = ProgressReporter::new();
        let manifest = backend
            .write(&model, tmp.path(), tmp.path(), &progress)
            .unwrap();

        let st_path = tmp.path().join("model.safetensors");
        let st_bytes = fs::read(&st_path).unwrap();
        let header_size =
            u64::from_le_bytes(st_bytes[..8].try_into().unwrap()) as usize;
        let header_str =
            std::str::from_utf8(&st_bytes[8..8 + header_size]).unwrap();
        let header: serde_json::Value =
            serde_json::from_str(header_str.trim()).unwrap();

        assert!(
            header.get("proj.weight.scales").is_some(),
            "Scales should be stored as separate tensor entry"
        );
        assert_eq!(manifest.shard_count, 1);
    }

    #[test]
    fn test_header_metadata_content() {
        let model = make_model(vec![make_tensor("w", vec![1; 8], false)]);
        let meta = SafetensorsBackend::build_header_metadata(&model);
        assert_eq!(meta.get("format").unwrap(), "hf2q");
        assert_eq!(meta.get("quant_method").unwrap(), "q4_k");
        assert_eq!(meta.get("bits").unwrap(), "4");
        assert_eq!(meta.get("architecture").unwrap(), "TestArch");
    }

    // ADR-014 P9 iter-1 §S3 — DWQ schema gate. A DWQ-tagged model
    // emits `<name>.weight` (U8) + `<name>.scales` (F16) + `<name>.biases`
    // (F16) triples per the mlx-lm convention.
    #[test]
    fn dwq_metadata_matches_mlx_lm_schema() {
        let backend = SafetensorsBackend::new();
        let mut t = make_tensor("model.layers.0.attn.q_proj.weight", vec![0xAA; 64], false);
        t.quant_info.method = "dwq-mixed-4-6".to_string();
        t.quant_info.scales = Some(vec![0u8; 8]);
        t.quant_info.biases = Some(vec![0u8; 8]);
        let mut tensor_map = HashMap::new();
        tensor_map.insert(t.name.clone(), t);
        let model = QuantizedModel {
            metadata: make_metadata(),
            tensors: tensor_map,
            quant_method: "dwq-mixed-4-6".to_string(),
            group_size: 64,
            bits: 4,
        };

        let tmp = tempfile::tempdir().unwrap();
        let progress = ProgressReporter::new();
        backend
            .write(&model, tmp.path(), tmp.path(), &progress)
            .unwrap();

        let st_path = tmp.path().join("model.safetensors");
        let st_bytes = fs::read(&st_path).unwrap();
        let header_size =
            u64::from_le_bytes(st_bytes[..8].try_into().unwrap()) as usize;
        let header_str =
            std::str::from_utf8(&st_bytes[8..8 + header_size]).unwrap();
        let header: serde_json::Value =
            serde_json::from_str(header_str.trim()).unwrap();

        // The triple must be present.
        let weight = header
            .get("model.layers.0.attn.q_proj.weight")
            .expect("weight tensor in DWQ header");
        let scales = header
            .get("model.layers.0.attn.q_proj.weight.scales")
            .expect("scales tensor in DWQ header");
        let biases = header
            .get("model.layers.0.attn.q_proj.weight.biases")
            .expect("biases tensor in DWQ header");

        // weight is U8 (packed), scales+biases are F16 per mlx-lm.
        assert_eq!(weight["dtype"], "U8");
        assert_eq!(scales["dtype"], "F16");
        assert_eq!(biases["dtype"], "F16");
    }

    // ADR-014 P9 iter-1 §S3 — K-quant blob schema gate. K-quant
    // tensors emit `<name>.weight` (U8) ONLY (no scales/biases triple);
    // per-shard `__metadata__` carries the `k_quant_method`
    // discriminator so the loader can pick the right dequantizer.
    #[test]
    fn kquant_metadata_uses_opaque_blob_with_discriminator() {
        let backend = SafetensorsBackend::new();
        let mut t = make_tensor("model.layers.0.attn.q_proj.weight", vec![0xBB; 144], false);
        t.quant_info.method = "k_quant_codec_direct".to_string();
        t.quant_info.scales = None; // K-quant packs scales inline.
        t.quant_info.biases = None;
        t.quant_info.bits = 0;
        t.quant_info.group_size = 0;
        t.quant_info.ggml_type = Some("Q4_K".to_string());
        let mut tensor_map = HashMap::new();
        tensor_map.insert(t.name.clone(), t);
        let model = QuantizedModel {
            metadata: make_metadata(),
            tensors: tensor_map,
            quant_method: "q4_k_m".to_string(),
            group_size: 0,
            bits: 0,
        };

        let tmp = tempfile::tempdir().unwrap();
        let progress = ProgressReporter::new();
        backend
            .write(&model, tmp.path(), tmp.path(), &progress)
            .unwrap();

        let st_path = tmp.path().join("model.safetensors");
        let st_bytes = fs::read(&st_path).unwrap();
        let header_size =
            u64::from_le_bytes(st_bytes[..8].try_into().unwrap()) as usize;
        let header_str =
            std::str::from_utf8(&st_bytes[8..8 + header_size]).unwrap();
        let header: serde_json::Value =
            serde_json::from_str(header_str.trim()).unwrap();

        let weight = header
            .get("model.layers.0.attn.q_proj.weight")
            .expect("weight tensor in K-quant header");
        assert_eq!(weight["dtype"], "U8");

        // No companion scales/biases — K-quant packs everything inline.
        assert!(
            header.get("model.layers.0.attn.q_proj.weight.scales").is_none(),
            "K-quant must not emit `<name>.scales` companion (scales are \
             inline in the GGUF block bytes)"
        );
        assert!(
            header.get("model.layers.0.attn.q_proj.weight.biases").is_none(),
            "K-quant must not emit `<name>.biases` companion"
        );

        // Per-shard __metadata__ carries the `k_quant_method` discriminator.
        let meta = header.get("__metadata__").expect("__metadata__ map");
        assert_eq!(
            meta["k_quant_method"], "k_quant_q4_k_m",
            "K-quant shard metadata must include the loader-routing discriminator"
        );
    }

    // ADR-014 P9 iter-1 §S5 — `with_shard_size_gb` builder propagates
    // the configured shard target. The clamp guards against degenerate
    // inputs.
    #[test]
    fn shard_size_gb_builder_propagates_and_clamps() {
        let s5 = SafetensorsBackend::with_shard_size_gb(5.0);
        assert_eq!(s5.shard_size_bytes, 5 * 1024 * 1024 * 1024);

        let s1 = SafetensorsBackend::with_shard_size_gb(1.0);
        assert_eq!(s1.shard_size_bytes, 1024 * 1024 * 1024);

        // Below the 0.5 GB floor → clamped up.
        let s_tiny = SafetensorsBackend::with_shard_size_gb(0.1);
        assert_eq!(s_tiny.shard_size_bytes, (0.5 * (1024.0 * 1024.0 * 1024.0)) as u64);

        // Above the 50 GB ceiling → clamped down.
        let s_huge = SafetensorsBackend::with_shard_size_gb(100.0);
        assert_eq!(s_huge.shard_size_bytes, 50 * 1024 * 1024 * 1024);

        // Default constructor.
        let default = SafetensorsBackend::new();
        assert_eq!(default.shard_size_bytes, 5 * 1024 * 1024 * 1024);
    }
}
