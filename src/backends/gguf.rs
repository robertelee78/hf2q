//! GGUF output backend — produces `.gguf` files compatible with llama.cpp and friends.
//!
//! Implements the GGUF v3 binary format directly using standard Rust I/O.
//! Handles HF-to-GGUF tensor name mapping, GGML dtype selection, and metadata encoding.

use std::fs::File;
use std::io::{BufWriter, Seek, Write as IoWrite};
use std::path::Path;

use tracing::{debug, info, warn};

use crate::backends::{BackendError, OutputBackend};
use crate::ir::{
    FormatWarning, OutputFile, OutputManifest, QuantizedModel,
    TensorQuantInfo, WarningSeverity,
};
use crate::progress::ProgressReporter;

// ---------------------------------------------------------------------------
// GGUF constants
// ---------------------------------------------------------------------------

const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF"
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: u64 = 32;

// GGUF metadata value types
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_STRING: u32 = 8;

// GGML dtype identifiers
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q2_K: u32 = 10;

/// Maximum single-file GGUF size before we warn (20 GB).
const LARGE_MODEL_BYTES: u64 = 20 * 1024 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Backend struct
// ---------------------------------------------------------------------------

/// GGUF output backend — writes a single `.gguf` file from quantized IR.
pub struct GgufBackend;

impl GgufBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GgufBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// OutputBackend impl
// ---------------------------------------------------------------------------

impl OutputBackend for GgufBackend {
    fn name(&self) -> &str {
        "GGUF"
    }

    fn validate(&self, model: &QuantizedModel) -> Result<Vec<FormatWarning>, BackendError> {
        let mut warnings = Vec::new();

        // Check for unsupported bit widths
        for (name, tensor) in &model.tensors {
            let bits = tensor.quant_info.bits;
            if !tensor.quant_info.preserved && !matches!(bits, 2 | 4 | 8 | 16) {
                warnings.push(FormatWarning {
                    message: format!(
                        "Tensor '{}' has {}-bit quantization which has no standard GGML type; \
                         will fall back to F16",
                        name, bits
                    ),
                    severity: WarningSeverity::Warning,
                });
            }
        }

        // Warn if the total data is very large for a single file
        let total_bytes: u64 = model
            .tensors
            .values()
            .map(|t| t.data.len() as u64)
            .sum();
        if total_bytes > LARGE_MODEL_BYTES {
            warnings.push(FormatWarning {
                message: format!(
                    "Model data is {:.1} GB; GGUF output will be a single large file. \
                     Some runtimes may struggle with files this large.",
                    total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                ),
                severity: WarningSeverity::Warning,
            });
        }

        Ok(warnings)
    }

    fn write(
        &self,
        model: &QuantizedModel,
        _input_dir: &Path,
        output_dir: &Path,
        progress: &ProgressReporter,
    ) -> Result<OutputManifest, BackendError> {
        std::fs::create_dir_all(output_dir)?;

        let filename = format!(
            "{}-Q{}_{}.gguf",
            sanitize_model_type(&model.metadata.model_type),
            model.bits,
            model.group_size,
        );
        let out_path = output_dir.join(&filename);
        info!("Writing GGUF to {}", out_path.display());

        // Collect tensors in deterministic order
        let mut tensor_names: Vec<&String> = model.tensors.keys().collect();
        tensor_names.sort();

        let pb = progress.bar(tensor_names.len() as u64, "Writing GGUF tensors");

        // Build metadata key-value pairs
        let metadata = build_metadata(model);
        let tensor_count = tensor_names.len() as u64;
        let kv_count = metadata.len() as u64;

        let file = File::create(&out_path).map_err(|e| BackendError::WriteFailed {
            reason: format!("Failed to create {}: {}", out_path.display(), e),
        })?;
        let mut w = BufWriter::new(file);

        // --- Header ---
        w.write_all(&GGUF_MAGIC)?;
        w.write_all(&GGUF_VERSION.to_le_bytes())?;
        w.write_all(&tensor_count.to_le_bytes())?;
        w.write_all(&kv_count.to_le_bytes())?;

        // --- Metadata KV pairs ---
        for (key, value) in &metadata {
            write_metadata_kv(&mut w, key, value)?;
        }

        // --- Tensor info headers ---
        // We need to compute offsets relative to the start of the tensor data block.
        // First pass: compute all tensor info sizes to find data block start.
        let mut tensor_data_offset: u64 = 0;
        let mut tensor_infos: Vec<TensorWriteInfo> = Vec::with_capacity(tensor_names.len());

        for name in &tensor_names {
            let qt = &model.tensors[*name];
            let gguf_name = hf_name_to_gguf(name);
            let ggml_type = quant_info_to_ggml_type(&qt.quant_info);

            // Align offset
            tensor_data_offset = align_up(tensor_data_offset, ALIGNMENT);

            tensor_infos.push(TensorWriteInfo {
                gguf_name,
                shape: qt.shape.clone(),
                ggml_type,
                data_offset: tensor_data_offset,
                data_len: qt.data.len(),
            });

            tensor_data_offset += qt.data.len() as u64;
        }

        // Write tensor info entries
        for info in &tensor_infos {
            write_tensor_info(&mut w, info)?;
        }

        // --- Padding to alignment before tensor data ---
        let header_end = w.stream_position()?;
        let padding_needed = align_up(header_end, ALIGNMENT) - header_end;
        if padding_needed > 0 {
            w.write_all(&vec![0u8; padding_needed as usize])?;
        }

        let data_block_start = w.stream_position()?;
        debug!("Tensor data block starts at offset {}", data_block_start);

        // --- Tensor data ---
        for (i, name) in tensor_names.iter().enumerate() {
            let qt = &model.tensors[*name];
            let info = &tensor_infos[i];

            // Pad to alignment
            let current = w.stream_position()?;
            let target = data_block_start + info.data_offset;
            if current < target {
                w.write_all(&vec![0u8; (target - current) as usize])?;
            }

            w.write_all(&qt.data)?;
            pb.inc(1);
        }

        w.flush()?;
        pb.finish_with_message("GGUF tensors written");

        let file_size = std::fs::metadata(&out_path)?.len();
        info!(
            "GGUF file written: {} ({:.2} MB)",
            out_path.display(),
            file_size as f64 / (1024.0 * 1024.0)
        );

        Ok(OutputManifest {
            output_dir: output_dir.to_string_lossy().into_owned(),
            files: vec![OutputFile {
                filename,
                size_bytes: file_size,
            }],
            total_size_bytes: file_size,
            shard_count: 1,
        })
    }
}

// ---------------------------------------------------------------------------
// GGML dtype mapping
// ---------------------------------------------------------------------------

/// Map IR quantization metadata to a GGML type code.
fn quant_info_to_ggml_type(info: &TensorQuantInfo) -> u32 {
    if info.preserved {
        return GGML_TYPE_F16;
    }
    match info.bits {
        16 => GGML_TYPE_F16,
        8 => GGML_TYPE_Q8_0,
        4 => GGML_TYPE_Q4_0,
        2 => GGML_TYPE_Q2_K,
        _ => {
            warn!(
                "No standard GGML type for {}-bit; falling back to F16",
                info.bits
            );
            GGML_TYPE_F16
        }
    }
}

// ---------------------------------------------------------------------------
// HF → GGUF tensor name mapping
// ---------------------------------------------------------------------------

/// Convert a HuggingFace tensor name to its GGUF equivalent.
fn hf_name_to_gguf(hf_name: &str) -> String {
    // Static patterns (no layer number)
    let static_map: &[(&str, &str)] = &[
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.norm.weight", "output_norm.weight"),
        ("lm_head.weight", "output.weight"),
    ];

    for &(hf, gguf) in static_map {
        if hf_name == hf {
            return gguf.to_string();
        }
    }

    // Layer-indexed patterns: model.layers.N.<suffix> → blk.N.<gguf_suffix>
    let layer_map: &[(&str, &str)] = &[
        ("self_attn.q_proj.weight", "attn_q.weight"),
        ("self_attn.k_proj.weight", "attn_k.weight"),
        ("self_attn.v_proj.weight", "attn_v.weight"),
        ("self_attn.o_proj.weight", "attn_output.weight"),
        ("mlp.gate_proj.weight", "ffn_gate.weight"),
        ("mlp.up_proj.weight", "ffn_up.weight"),
        ("mlp.down_proj.weight", "ffn_down.weight"),
        ("input_layernorm.weight", "attn_norm.weight"),
        ("post_attention_layernorm.weight", "ffn_norm.weight"),
    ];

    // Parse "model.layers.N.<suffix>" without regex
    const LAYER_PREFIX: &str = "model.layers.";
    if let Some(rest) = hf_name.strip_prefix(LAYER_PREFIX) {
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            // Verify it is actually a number
            if layer_num.chars().all(|c| c.is_ascii_digit()) {
                let suffix = &rest[dot_pos + 1..];
                for &(hf_suffix, gguf_suffix) in layer_map {
                    if suffix == hf_suffix {
                        return format!("blk.{}.{}", layer_num, gguf_suffix);
                    }
                }
                // Pass through unknown layer suffixes with best-effort mapping
                return format!("blk.{}.{}", layer_num, suffix);
            }
        }
    }

    // Fallback: return the name unchanged
    debug!("No GGUF name mapping for '{}', keeping as-is", hf_name);
    hf_name.to_string()
}

// ---------------------------------------------------------------------------
// Metadata construction
// ---------------------------------------------------------------------------

/// Metadata value — either a string or a u32.
enum MetaValue {
    String(String),
    Uint32(u32),
}

/// Build the GGUF metadata key-value list from model metadata.
fn build_metadata(model: &QuantizedModel) -> Vec<(String, MetaValue)> {
    let meta = &model.metadata;
    let arch = &meta.model_type; // e.g. "llama", "gemma4"

    let mut kv: Vec<(String, MetaValue)> = Vec::new();

    kv.push((
        "general.architecture".into(),
        MetaValue::String(arch.clone()),
    ));
    kv.push((
        "general.name".into(),
        MetaValue::String(meta.architecture.clone()),
    ));
    kv.push((
        "general.quantization_version".into(),
        MetaValue::Uint32(2),
    ));
    kv.push((
        format!("{}.block_count", arch),
        MetaValue::Uint32(meta.num_layers),
    ));
    kv.push((
        format!("{}.embedding_length", arch),
        MetaValue::Uint32(meta.hidden_size as u32),
    ));
    kv.push((
        format!("{}.attention.head_count", arch),
        MetaValue::Uint32(meta.num_attention_heads),
    ));

    if let Some(kv_heads) = meta.num_kv_heads {
        kv.push((
            format!("{}.attention.head_count_kv", arch),
            MetaValue::Uint32(kv_heads),
        ));
    }

    if let Some(ff_size) = meta.intermediate_size {
        kv.push((
            format!("{}.feed_forward_length", arch),
            MetaValue::Uint32(ff_size as u32),
        ));
    }

    kv.push((
        "general.file_type".into(),
        MetaValue::Uint32(ggml_ftype_from_bits(model.bits)),
    ));

    kv
}

/// Map global bit width to a GGML file type code.
fn ggml_ftype_from_bits(bits: u8) -> u32 {
    match bits {
        16 => 1,  // MOSTLY_F16
        8 => 7,   // MOSTLY_Q8_0
        4 => 2,   // MOSTLY_Q4_0
        2 => 10,  // MOSTLY_Q2_K
        _ => 1,   // default to F16
    }
}

// ---------------------------------------------------------------------------
// GGUF binary encoding helpers
// ---------------------------------------------------------------------------

/// Write a GGUF string: u64 length followed by raw bytes (no null terminator).
fn write_gguf_string<W: IoWrite>(w: &mut W, s: &str) -> std::io::Result<()> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(bytes)?;
    Ok(())
}

/// Write a single GGUF metadata key-value pair.
fn write_metadata_kv<W: IoWrite>(w: &mut W, key: &str, value: &MetaValue) -> std::io::Result<()> {
    write_gguf_string(w, key)?;
    match value {
        MetaValue::String(s) => {
            w.write_all(&GGUF_TYPE_STRING.to_le_bytes())?;
            write_gguf_string(w, s)?;
        }
        MetaValue::Uint32(v) => {
            w.write_all(&GGUF_TYPE_UINT32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Info needed to write a single tensor's header and data.
struct TensorWriteInfo {
    gguf_name: String,
    shape: Vec<usize>,
    ggml_type: u32,
    data_offset: u64,
    data_len: usize,
}

/// Write a tensor info entry in the GGUF header.
fn write_tensor_info<W: IoWrite>(w: &mut W, info: &TensorWriteInfo) -> std::io::Result<()> {
    // Name
    write_gguf_string(w, &info.gguf_name)?;
    // Number of dimensions
    w.write_all(&(info.shape.len() as u32).to_le_bytes())?;
    // Dimensions (as u64, reversed — GGUF uses row-major / C order dimensions)
    for &dim in &info.shape {
        w.write_all(&(dim as u64).to_le_bytes())?;
    }
    // Type
    w.write_all(&info.ggml_type.to_le_bytes())?;
    // Offset (relative to start of data block)
    w.write_all(&info.data_offset.to_le_bytes())?;
    Ok(())
}

/// Round `offset` up to the next multiple of `alignment`.
fn align_up(offset: u64, alignment: u64) -> u64 {
    let remainder = offset % alignment;
    if remainder == 0 {
        offset
    } else {
        offset + (alignment - remainder)
    }
}

/// Sanitize model type for use in filenames.
fn sanitize_model_type(model_type: &str) -> String {
    model_type
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::ir::{DType, ModelMetadata, QuantizedModel, QuantizedTensor, TensorQuantInfo};

    fn meta() -> ModelMetadata {
        ModelMetadata {
            architecture: "LlamaForCausalLM".into(), model_type: "llama".into(),
            param_count: 7_000_000_000, hidden_size: 4096, num_layers: 32,
            layer_types: vec!["attention".into()], num_attention_heads: 32,
            num_kv_heads: Some(8), vocab_size: 32000, dtype: "float16".into(),
            shard_count: 1, num_experts: None, top_k_experts: None,
            intermediate_size: Some(11008), raw_config: serde_json::Value::Null,
        }
    }

    fn tensor(name: &str, bits: u8, preserved: bool) -> QuantizedTensor {
        QuantizedTensor {
            name: name.into(), shape: vec![32, 32], original_dtype: DType::F16,
            data: vec![0u8; 32 * 32 * 2],
            quant_info: TensorQuantInfo {
                method: if preserved { "passthrough".into() } else { format!("q{}", bits) },
                bits, group_size: 64, preserved, scales: None, biases: None,
            },
        }
    }

    fn model(tensors: Vec<(&str, u8, bool)>, bits: u8) -> QuantizedModel {
        let map = tensors.into_iter().map(|(n, b, p)| (n.into(), tensor(n, b, p))).collect();
        QuantizedModel { metadata: meta(), tensors: map, quant_method: format!("q{}", bits), group_size: 64, bits }
    }

    #[test]
    fn test_name_mapping() {
        // Static mappings
        assert_eq!(hf_name_to_gguf("model.embed_tokens.weight"), "token_embd.weight");
        assert_eq!(hf_name_to_gguf("model.norm.weight"), "output_norm.weight");
        assert_eq!(hf_name_to_gguf("lm_head.weight"), "output.weight");
        // Layer mappings
        let cases = [
            ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
            ("model.layers.15.self_attn.k_proj.weight", "blk.15.attn_k.weight"),
            ("model.layers.31.self_attn.v_proj.weight", "blk.31.attn_v.weight"),
            ("model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight"),
            ("model.layers.3.mlp.gate_proj.weight", "blk.3.ffn_gate.weight"),
            ("model.layers.3.mlp.up_proj.weight", "blk.3.ffn_up.weight"),
            ("model.layers.3.mlp.down_proj.weight", "blk.3.ffn_down.weight"),
            ("model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight"),
            ("model.layers.0.post_attention_layernorm.weight", "blk.0.ffn_norm.weight"),
        ];
        for (hf, gguf) in cases { assert_eq!(hf_name_to_gguf(hf), gguf, "mapping failed for {}", hf); }
        // Unknown passthrough
        assert_eq!(hf_name_to_gguf("model.layers.5.some_new.weight"), "blk.5.some_new.weight");
        assert_eq!(hf_name_to_gguf("decoder.block.0.weight"), "decoder.block.0.weight");
    }

    #[test]
    fn test_dtype_mapping() {
        let qi = |bits, preserved| TensorQuantInfo {
            method: "t".into(), bits, group_size: 64, preserved, scales: None, biases: None,
        };
        assert_eq!(quant_info_to_ggml_type(&qi(16, false)), GGML_TYPE_F16);
        assert_eq!(quant_info_to_ggml_type(&qi(8, false)), GGML_TYPE_Q8_0);
        assert_eq!(quant_info_to_ggml_type(&qi(4, false)), GGML_TYPE_Q4_0);
        assert_eq!(quant_info_to_ggml_type(&qi(2, false)), GGML_TYPE_Q2_K);
        assert_eq!(quant_info_to_ggml_type(&qi(4, true)), GGML_TYPE_F16); // preserved
        assert_eq!(quant_info_to_ggml_type(&qi(6, false)), GGML_TYPE_F16); // unknown fallback
    }

    #[test]
    fn test_validate_clean_and_unsupported() {
        let backend = GgufBackend::new();
        // Clean model: no warnings
        let w = backend.validate(&model(vec![("t1", 4, false), ("t2", 16, true)], 4)).unwrap();
        assert!(w.is_empty(), "Expected no warnings: {:?}", w);
        // Unsupported bit width: 1 warning
        let w = backend.validate(&model(vec![("odd", 6, false)], 6)).unwrap();
        assert_eq!(w.len(), 1);
        assert!(w[0].message.contains("6-bit"));
        assert_eq!(w[0].severity, WarningSeverity::Warning);
    }

    #[test]
    fn test_write_gguf_header() {
        let backend = GgufBackend::new();
        let m = model(vec![
            ("model.layers.0.self_attn.q_proj.weight", 4, false),
            ("model.embed_tokens.weight", 16, true),
        ], 4);
        let tmp = tempfile::tempdir().unwrap();
        let manifest = backend.write(&m, tmp.path(), tmp.path(), &ProgressReporter::new()).unwrap();
        assert_eq!(manifest.shard_count, 1);
        assert_eq!(manifest.files.len(), 1);
        assert!(manifest.files[0].filename.ends_with(".gguf"));
        // Verify binary header
        let data = std::fs::read(tmp.path().join(&manifest.files[0].filename)).unwrap();
        assert_eq!(&data[0..4], &GGUF_MAGIC);
        assert_eq!(u32::from_le_bytes(data[4..8].try_into().unwrap()), GGUF_VERSION);
        assert_eq!(u64::from_le_bytes(data[8..16].try_into().unwrap()), 2);
    }

    #[test]
    fn test_align_up() {
        for (input, expected) in [(0, 0), (1, 32), (32, 32), (33, 64), (63, 64), (64, 64)] {
            assert_eq!(align_up(input, 32), expected);
        }
    }

    #[test]
    fn test_metadata_keys() {
        let kv = build_metadata(&model(vec![], 4));
        let keys: Vec<&str> = kv.iter().map(|(k, _)| k.as_str()).collect();
        for expected in ["general.architecture", "general.name", "llama.block_count",
            "llama.embedding_length", "llama.attention.head_count",
            "llama.attention.head_count_kv", "llama.feed_forward_length", "general.file_type"]
        {
            assert!(keys.contains(&expected), "Missing metadata key: {}", expected);
        }
    }
}
