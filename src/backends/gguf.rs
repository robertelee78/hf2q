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
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;

// GGML dtype identifiers (from llama.cpp ggml.h)
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q4_1: u32 = 3;
const GGML_TYPE_Q5_0: u32 = 6;
const GGML_TYPE_Q5_1: u32 = 7;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q8_1: u32 = 9;
const GGML_TYPE_Q2_K: u32 = 10;
const GGML_TYPE_Q3_K_S: u32 = 11;
const GGML_TYPE_Q3_K_M: u32 = 12;
const GGML_TYPE_Q3_K_L: u32 = 13;
const GGML_TYPE_Q4_K_S: u32 = 14;
const GGML_TYPE_Q4_K_M: u32 = 15;
const GGML_TYPE_Q5_K_S: u32 = 16;
const GGML_TYPE_Q5_K_M: u32 = 17;
const GGML_TYPE_Q6_K: u32 = 18;
const GGML_TYPE_IQ2_XXS: u32 = 19;
const GGML_TYPE_IQ2_XS: u32 = 20;

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
        // Estimate using ggml block sizes (repacked data is larger than raw packed data)
        let total_bytes: u64 = model
            .tensors
            .values()
            .map(|t| {
                let ggml_type = quant_info_to_ggml_type(&t.quant_info);
                let n_elem: usize = t.shape.iter().product();
                ggml_tensor_size(n_elem, ggml_type) as u64
            })
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
        input_dir: &Path,
        output_dir: &Path,
        progress: &ProgressReporter,
    ) -> Result<OutputManifest, BackendError> {
        // If output_dir is a .gguf file path, use it directly; otherwise treat as directory
        let (out_path, filename) = if output_dir.extension().and_then(|e| e.to_str()) == Some("gguf") {
            if let Some(parent) = output_dir.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            let fname = output_dir
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            (output_dir.to_path_buf(), fname)
        } else {
            std::fs::create_dir_all(output_dir)?;
            let fname = format!(
                "{}-Q{}_{}.gguf",
                sanitize_model_type(&model.metadata.model_type),
                model.bits,
                model.group_size,
            );
            (output_dir.join(&fname), fname)
        };
        info!("Writing GGUF to {}", out_path.display());

        // Collect tensors in deterministic order
        let mut tensor_names: Vec<&String> = model.tensors.keys().collect();
        tensor_names.sort();

        let pb = progress.bar(tensor_names.len() as u64, "Writing GGUF tensors");

        // Build metadata key-value pairs
        let metadata = build_metadata(model, input_dir);
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

        // --- Pass 1: Compute tensor sizes and offsets (no allocation) ---
        // We need to know the repacked ggml block size for each tensor to write
        // correct offsets in the header, but we do NOT allocate the repacked data
        // yet to avoid doubling memory usage for a 26B+ model.
        let mut tensor_infos: Vec<TensorWriteInfo> = Vec::with_capacity(tensor_names.len());
        let mut tensor_data_offset: u64 = 0;

        for name in &tensor_names {
            let qt = &model.tensors[*name];
            let gguf_name = hf_name_to_gguf(name, &model.metadata.model_type);
            let ggml_type = quant_info_to_ggml_type(&qt.quant_info);

            // Compute the repacked size without allocating
            let total_elements: usize = qt.shape.iter().product();
            let repacked_size = if qt.quant_info.preserved || ggml_type == GGML_TYPE_F16 || ggml_type == GGML_TYPE_F32 {
                qt.data.len() // preserved/f16/f32 pass through unchanged
            } else {
                ggml_tensor_size(total_elements, ggml_type)
            };

            // Align offset
            tensor_data_offset = align_up(tensor_data_offset, ALIGNMENT);

            tensor_infos.push(TensorWriteInfo {
                gguf_name,
                shape: qt.shape.clone(),
                ggml_type,
                data_offset: tensor_data_offset,
                data_len: repacked_size,
            });

            tensor_data_offset += repacked_size as u64;
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

        // --- Pass 2: Repack and write one tensor at a time ---
        // Each tensor is repacked into ggml block format, written, then the
        // repacked buffer is dropped before processing the next tensor.
        for (i, name) in tensor_names.iter().enumerate() {
            let qt = &model.tensors[*name];
            let info = &tensor_infos[i];

            // Pad to alignment
            let current = w.stream_position()?;
            let target = data_block_start + info.data_offset;
            if current < target {
                w.write_all(&vec![0u8; (target - current) as usize])?;
            }

            // Repack this single tensor and write immediately
            let data = repack_to_ggml_blocks(qt, info.ggml_type).map_err(|e| {
                BackendError::WriteFailed {
                    reason: format!("Failed to repack tensor '{}': {}", name, e),
                }
            })?;
            w.write_all(&data)?;
            // `data` is dropped here — no accumulation
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

        let manifest_dir = if output_dir.extension().and_then(|e| e.to_str()) == Some("gguf") {
            output_dir.parent().unwrap_or(output_dir).to_string_lossy().into_owned()
        } else {
            output_dir.to_string_lossy().into_owned()
        };
        Ok(OutputManifest {
            output_dir: manifest_dir,
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

/// Map a GGML type name string to its numeric type code.
///
/// Supports the full K-quant family plus standard types.
/// Names are matched case-insensitively with optional "GGML_TYPE_" prefix stripped.
fn ggml_type_from_name(name: &str) -> Option<u32> {
    // Normalize: uppercase, strip optional prefix
    let upper = name.trim().to_uppercase();
    let key = upper
        .strip_prefix("GGML_TYPE_")
        .unwrap_or(&upper);
    match key {
        "F32"      => Some(GGML_TYPE_F32),
        "F16"      => Some(GGML_TYPE_F16),
        "Q4_0"     => Some(GGML_TYPE_Q4_0),
        "Q4_1"     => Some(GGML_TYPE_Q4_1),
        "Q5_0"     => Some(GGML_TYPE_Q5_0),
        "Q5_1"     => Some(GGML_TYPE_Q5_1),
        "Q8_0"     => Some(GGML_TYPE_Q8_0),
        "Q8_1"     => Some(GGML_TYPE_Q8_1),
        "Q2_K"     => Some(GGML_TYPE_Q2_K),
        "Q3_K_S"   => Some(GGML_TYPE_Q3_K_S),
        "Q3_K_M" | "Q3_K" => Some(GGML_TYPE_Q3_K_M),
        "Q3_K_L"   => Some(GGML_TYPE_Q3_K_L),
        "Q4_K_S"   => Some(GGML_TYPE_Q4_K_S),
        "Q4_K_M" | "Q4_K" => Some(GGML_TYPE_Q4_K_M),
        "Q5_K_S"   => Some(GGML_TYPE_Q5_K_S),
        "Q5_K_M" | "Q5_K" => Some(GGML_TYPE_Q5_K_M),
        "Q6_K"     => Some(GGML_TYPE_Q6_K),
        "IQ2_XXS"  => Some(GGML_TYPE_IQ2_XXS),
        "IQ2_XS"   => Some(GGML_TYPE_IQ2_XS),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// GGML block repacking — converts hf2q internal format to ggml block layout
// ---------------------------------------------------------------------------

/// GGML Q4_0 block: 32 elements, 18 bytes (2-byte f16 scale + 16 packed nibbles).
const QK4_0: usize = 32;
/// GGML Q4_0 block size in bytes.
const BLOCK_Q4_0_BYTES: usize = 2 + QK4_0 / 2; // 18

/// GGML Q8_0 block: 32 elements, 34 bytes (2-byte f16 scale + 32 int8 values).
const QK8_0: usize = 32;
/// GGML Q8_0 block size in bytes.
const BLOCK_Q8_0_BYTES: usize = 2 + QK8_0; // 34

/// Repack a QuantizedTensor's data from hf2q's internal format into proper ggml
/// block format. Returns the repacked bytes ready for writing to GGUF.
///
/// hf2q internal format:
///   - `qt.data`: packed nibbles (consecutive pairs), signed values [-7,7] for 4-bit
///   - `qt.quant_info.scales`: separate Vec<u8> of f16 scale bytes (2 bytes per group)
///   - `qt.quant_info.group_size`: may be != 32
///
/// For Q4_0 target:
///   - Must produce 18-byte blocks of 32 elements each
///   - Block layout: [d: f16] [qs[0..15]: packed unsigned nibbles]
///   - Nibble packing: byte[i] = qs_low[i] | (qs_high[i] << 4)
///     where qs_low = first 16 values, qs_high = second 16 values
///   - Q4_0 scale: d = max(abs(block)) / -8  (so d is negative when max element is positive)
///   - Unsigned encoding: q = trunc(val/d + 8.5), clipped [0,15]
///   - Dequant: x = (q - 8) * d
///
/// For Q8_0 target:
///   - 34-byte blocks of 32 elements each
///   - Block layout: [d: f16] [qs[0..31]: int8]
///   - Scale: d = absmax / 127
///   - Quantized: q = round(val / d) as int8
fn repack_to_ggml_blocks(
    qt: &crate::ir::QuantizedTensor,
    ggml_type: u32,
) -> Result<Vec<u8>, BackendError> {
    let info = &qt.quant_info;

    // Preserved or f16 tensors: data is already raw element bytes, no repacking needed
    if info.preserved || info.bits == 16 || info.method == "f16" {
        return Ok(qt.data.clone());
    }

    // Only repack if we have scales (quantized tensors)
    let scales_bytes = match &info.scales {
        Some(s) if !s.is_empty() => s,
        _ => {
            // No scales means data is not in our internal quantized format.
            // This shouldn't happen for properly quantized tensors, but return as-is.
            warn!(
                "Tensor '{}' has no scales but is not preserved/f16; writing raw data",
                qt.name
            );
            return Ok(qt.data.clone());
        }
    };

    let total_elements: usize = qt.shape.iter().product();

    match ggml_type {
        GGML_TYPE_Q4_0 => repack_q4_0(qt, scales_bytes, total_elements),
        GGML_TYPE_Q8_0 => repack_q8_0(qt, scales_bytes, total_elements),
        GGML_TYPE_F16 | GGML_TYPE_F32 => {
            // Should not reach here for F16/F32 (caught above), but handle gracefully
            Ok(qt.data.clone())
        }
        _ => Err(BackendError::WriteFailed {
            reason: format!(
                "Cannot repack tensor '{}': unsupported target GGML type {}",
                qt.name, ggml_type
            ),
        }),
    }
}

/// Repack hf2q internal 4-bit quantized data into Q4_0 block format.
///
/// Steps:
/// 1. Decode f16 scales from quant_info.scales
/// 2. Unpack signed nibbles from qt.data
/// 3. Reconstruct approximate f32 values: val = signed_q * scale
/// 4. Re-quantize each 32-element block into Q4_0 format:
///    - Compute d = max(abs(block)) / -8 (matching ggml convention)
///    - q = trunc(val / d + 8.5), clipped to [0, 15]
///    - Pack: byte[i] = q[i] | (q[i+16] << 4) for i in 0..16
fn repack_q4_0(
    qt: &crate::ir::QuantizedTensor,
    scales_bytes: &[u8],
    total_elements: usize,
) -> Result<Vec<u8>, BackendError> {
    let info = &qt.quant_info;
    let group_size = if info.group_size == 0 { 32 } else { info.group_size };

    // Decode f16 scales: 2 bytes each
    let scales_f32: Vec<f32> = scales_bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect();

    // Unpack signed i4 values from hf2q's packed data.
    // hf2q packs consecutive pairs: byte = (pair[0] & 0x0F) | ((pair[1] & 0x0F) << 4)
    // Values are signed [-7, 7] stored as signed nibbles (two's complement in 4 bits).
    let mut signed_values: Vec<i8> = Vec::with_capacity(total_elements);
    for &byte in &qt.data {
        let lo = (byte & 0x0F) as i8;
        let hi = ((byte >> 4) & 0x0F) as i8;
        // Convert from unsigned 4-bit to signed: if >= 8, subtract 16
        let lo_signed = if lo >= 8 { lo - 16 } else { lo };
        let hi_signed = if hi >= 8 { hi - 16 } else { hi };
        signed_values.push(lo_signed);
        signed_values.push(hi_signed);
    }
    // Truncate to actual element count (in case of padding)
    signed_values.truncate(total_elements);

    // Reconstruct approximate f32 values using the original scales
    let mut f32_values: Vec<f32> = Vec::with_capacity(total_elements);
    for (g, &scale) in scales_f32.iter().enumerate() {
        let start = g * group_size;
        let end = (start + group_size).min(total_elements);
        for i in start..end {
            let q = if i < signed_values.len() { signed_values[i] } else { 0 };
            f32_values.push(q as f32 * scale);
        }
    }

    // If scales didn't cover all elements (shouldn't happen), pad with zeros
    while f32_values.len() < total_elements {
        f32_values.push(0.0);
    }

    // Now re-quantize into Q4_0 blocks of 32 elements each
    let num_blocks = total_elements.div_ceil(QK4_0);
    let mut output = Vec::with_capacity(num_blocks * BLOCK_Q4_0_BYTES);

    for block_idx in 0..num_blocks {
        let start = block_idx * QK4_0;
        let end = (start + QK4_0).min(total_elements);

        // Get block values, padding with zeros if needed
        let mut block = [0.0f32; QK4_0];
        for (i, idx) in (start..end).enumerate() {
            block[i] = f32_values[idx];
        }

        // Compute Q4_0 scale: d = max_by_abs / -8
        // Find the element with maximum absolute value, preserving its sign
        let mut max_abs_val = 0.0f32;
        let mut max_abs_idx = 0;
        for (i, &v) in block.iter().enumerate() {
            if v.abs() > max_abs_val {
                max_abs_val = v.abs();
                max_abs_idx = i;
            }
        }
        let max_val = block[max_abs_idx];
        let d = max_val / -8.0;
        let id = if d == 0.0 { 0.0f32 } else { 1.0 / d };

        // Write scale as f16
        let d_f16 = half::f16::from_f32(d);
        output.extend_from_slice(&d_f16.to_le_bytes());

        // Quantize to unsigned [0, 15]: q = trunc(val * id + 8.5), clipped [0, 15]
        let mut qs = [0u8; QK4_0];
        for (i, &val) in block.iter().enumerate() {
            let q = (val * id + 8.5).floor() as i32;
            qs[i] = q.clamp(0, 15) as u8;
        }

        // Pack nibbles in Q4_0 order:
        // byte[i] = qs[i] | (qs[i + 16] << 4) for i in 0..16
        for i in 0..(QK4_0 / 2) {
            let lo = qs[i];
            let hi = qs[i + QK4_0 / 2];
            output.push(lo | (hi << 4));
        }
    }

    debug!(
        "Repacked '{}': {} elements -> {} Q4_0 blocks ({} bytes, was {} bytes)",
        qt.name,
        total_elements,
        num_blocks,
        output.len(),
        qt.data.len()
    );

    Ok(output)
}

/// Repack hf2q internal 8-bit quantized data into Q8_0 block format.
///
/// Steps:
/// 1. Decode f16 scales from quant_info.scales
/// 2. Read int8 values from qt.data
/// 3. Reconstruct approximate f32 values: val = q * scale
/// 4. Re-quantize each 32-element block into Q8_0 format:
///    - d = absmax / 127
///    - q = round(val / d) as int8
fn repack_q8_0(
    qt: &crate::ir::QuantizedTensor,
    scales_bytes: &[u8],
    total_elements: usize,
) -> Result<Vec<u8>, BackendError> {
    let info = &qt.quant_info;
    let group_size = if info.group_size == 0 { 32 } else { info.group_size };

    // Decode f16 scales
    let scales_f32: Vec<f32> = scales_bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect();

    // Read int8 values from qt.data (8-bit: 1 byte per element, stored as u8 cast of i8)
    let mut signed_values: Vec<i8> = Vec::with_capacity(total_elements);
    for &byte in &qt.data {
        signed_values.push(byte as i8);
    }
    signed_values.truncate(total_elements);

    // Reconstruct approximate f32 values
    let mut f32_values: Vec<f32> = Vec::with_capacity(total_elements);
    for (g, &scale) in scales_f32.iter().enumerate() {
        let start = g * group_size;
        let end = (start + group_size).min(total_elements);
        for i in start..end {
            let q = if i < signed_values.len() { signed_values[i] } else { 0 };
            f32_values.push(q as f32 * scale);
        }
    }

    while f32_values.len() < total_elements {
        f32_values.push(0.0);
    }

    // Re-quantize into Q8_0 blocks of 32 elements each
    let num_blocks = total_elements.div_ceil(QK8_0);
    let mut output = Vec::with_capacity(num_blocks * BLOCK_Q8_0_BYTES);

    for block_idx in 0..num_blocks {
        let start = block_idx * QK8_0;
        let end = (start + QK8_0).min(total_elements);

        let mut block = [0.0f32; QK8_0];
        for (i, idx) in (start..end).enumerate() {
            block[i] = f32_values[idx];
        }

        // Q8_0 scale: d = absmax / 127
        let absmax = block.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let d = absmax / 127.0;
        let id = if d == 0.0 { 0.0f32 } else { 1.0 / d };

        // Write scale as f16
        let d_f16 = half::f16::from_f32(d);
        output.extend_from_slice(&d_f16.to_le_bytes());

        // Quantize to int8
        for &val in &block {
            let q = (val * id).round() as i32;
            output.push(q.clamp(-128, 127) as u8);
        }
    }

    debug!(
        "Repacked '{}': {} elements -> {} Q8_0 blocks ({} bytes, was {} bytes)",
        qt.name,
        total_elements,
        num_blocks,
        output.len(),
        qt.data.len()
    );

    Ok(output)
}

/// Compute the expected byte size of a tensor in ggml block format.
fn ggml_tensor_size(total_elements: usize, ggml_type: u32) -> usize {
    match ggml_type {
        GGML_TYPE_F32 => total_elements * 4,
        GGML_TYPE_F16 => total_elements * 2,
        GGML_TYPE_Q4_0 => {
            let n_blocks = total_elements.div_ceil(QK4_0);
            n_blocks * BLOCK_Q4_0_BYTES
        }
        GGML_TYPE_Q8_0 => {
            let n_blocks = total_elements.div_ceil(QK8_0);
            n_blocks * BLOCK_Q8_0_BYTES
        }
        // For any other type, we cannot compute the size; caller should ensure
        // we never reach here for types we don't support.
        _ => total_elements * 2, // fallback to f16 sizing
    }
}

/// Map IR quantization metadata to a GGML type code.
///
/// The GGML type must match the actual block format written to disk. Since hf2q's
/// quantizer produces simple per-group scales + packed values (not K-quant super-block
/// format), we can only write Q4_0 / Q8_0 / F16 / F32 block formats. Apex K-quant
/// type overrides are mapped back to the corresponding simple type.
fn quant_info_to_ggml_type(info: &TensorQuantInfo) -> u32 {
    if info.preserved {
        return GGML_TYPE_F16;
    }

    // Map based on the actual bit width we can produce proper block format for.
    // K-quant types from Apex cannot be honored because hf2q does not produce
    // K-quant super-block data; we map to the closest simple block type instead.
    if let Some(ref type_name) = info.ggml_type {
        let upper = type_name.trim().to_uppercase();
        // K-quant types all get mapped to simple types based on ir_bits
        if upper.starts_with("Q2_K")
            || upper.starts_with("Q3_K")
            || upper.starts_with("Q4_K")
            || upper.starts_with("Q5_K")
            || upper.starts_with("Q6_K")
        {
            debug!(
                "Apex requested '{}', mapping to simple block type for bits={}",
                type_name, info.bits
            );
            // Fall through to bits-based mapping below
        } else if let Some(code) = ggml_type_from_name(type_name) {
            return code;
        } else {
            warn!(
                "Unknown GGML type name '{}'; falling back to bits-based mapping",
                type_name
            );
        }
    }

    // Generic bits-based mapping — only types we can produce proper block format for
    match info.bits {
        16 => GGML_TYPE_F16,
        8 => GGML_TYPE_Q8_0,
        4 => GGML_TYPE_Q4_0,
        2 => GGML_TYPE_Q4_0, // 2-bit is rare; pack as Q4_0 with values in [0,3] range
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
// HF → GGUF tensor name mapping (architecture-aware)
// ---------------------------------------------------------------------------

/// Build the per-layer mapping table for a given architecture.
///
/// The same HF tensor suffix can map to different GGUF names depending on the
/// model architecture. For example `post_attention_layernorm.weight` maps to
/// `ffn_norm.weight` for LLaMA-family models (where it is the only post-attention
/// norm and acts as the FFN pre-norm), but to `post_attention_norm.weight` for
/// Gemma4 (which has a separate `pre_feedforward_layernorm` for the FFN pre-norm).
fn layer_map_for_arch(arch: &str) -> Vec<(&'static str, &'static str)> {
    // Shared entries — identical across all architectures
    let shared: &[(&str, &str)] = &[
        ("self_attn.q_proj.weight", "attn_q.weight"),
        ("self_attn.k_proj.weight", "attn_k.weight"),
        ("self_attn.v_proj.weight", "attn_v.weight"),
        ("self_attn.o_proj.weight", "attn_output.weight"),
        ("mlp.gate_proj.weight", "ffn_gate.weight"),
        ("mlp.up_proj.weight", "ffn_up.weight"),
        ("mlp.down_proj.weight", "ffn_down.weight"),
        ("input_layernorm.weight", "attn_norm.weight"),
        ("self_attn.q_norm.weight", "attn_q_norm.weight"),
        ("self_attn.k_norm.weight", "attn_k_norm.weight"),
    ];

    let mut map = Vec::with_capacity(shared.len() + 12);
    map.extend_from_slice(shared);

    match arch {
        // Gemma family: post_attention_layernorm is a distinct post-attention norm,
        // NOT the FFN pre-norm. The FFN pre-norm is pre_feedforward_layernorm.
        "gemma4" | "gemma3" | "gemma2" => {
            map.extend_from_slice(&[
                ("post_attention_layernorm.weight", "post_attention_norm.weight"),
                // pre_feedforward_layernorm is FFN_PRE_NORM (alias of FFN_NORM)
                ("pre_feedforward_layernorm.weight", "ffn_norm.weight"),
                // MoE norms
                ("post_feedforward_layernorm_1.weight", "post_ffw_norm_1.weight"),
                ("post_feedforward_layernorm_2.weight", "post_ffw_norm_2.weight"),
                ("pre_feedforward_layernorm_2.weight", "pre_ffw_norm_2.weight"),
                ("post_feedforward_layernorm.weight", "post_ffw_norm.weight"),
                // MoE routing
                ("router.proj.weight", "ffn_gate_inp.weight"),
                ("router.scale", "ffn_gate_inp.scale"),
                ("experts.gate_up_proj", "ffn_gate_up_exps.weight"),
                ("experts.down_proj", "ffn_down_exps.weight"),
                ("router.per_expert_scale", "ffn_down_exps.scale"),
                // Layer scalar
                ("layer_scalar", "layer_output_scale.weight"),
            ]);
        }
        // LLaMA-like default: covers llama, mistral, qwen2, qwen3, phi, etc.
        // post_attention_layernorm IS the FFN pre-norm (there is no separate
        // pre_feedforward_layernorm in these architectures).
        _ => {
            map.extend_from_slice(&[
                ("post_attention_layernorm.weight", "ffn_norm.weight"),
            ]);
        }
    }

    map
}

/// Convert a HuggingFace tensor name to its GGUF equivalent.
///
/// `arch` is the model architecture string (e.g. "llama", "gemma4", "qwen3")
/// from `model.metadata.model_type`. Different architectures use different
/// GGUF names for the same HF tensor suffixes.
fn hf_name_to_gguf(hf_name: &str, arch: &str) -> String {
    // Strip language_model. prefix (Gemma4 conditional-generation models)
    let hf_name = hf_name.replace("language_model.", "");
    let hf_name = hf_name.as_str();

    // Static patterns (no layer number) — consistent across all architectures
    let static_map: &[(&str, &str)] = &[
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.norm.weight", "output_norm.weight"),
        ("lm_head.weight", "output.weight"),
        // Vision static tensors
        ("model.vision_tower.patch_embedder.input_proj.weight", "v.patch_embd.weight"),
        ("model.vision_tower.patch_embedder.position_embedding_table", "v.position_embd.weight"),
        ("model.vision_tower.std_bias", "v.std_bias"),
        ("model.vision_tower.std_scale", "v.std_scale"),
        ("model.embed_vision.embedding_projection.weight", "mm.0.weight"),
    ];

    for &(hf, gguf) in static_map {
        if hf_name == hf {
            return gguf.to_string();
        }
    }

    // Layer-indexed patterns: model.layers.N.<suffix> → blk.N.<gguf_suffix>
    // Architecture-aware: the same HF suffix maps to different GGUF names depending on arch.
    let layer_map = layer_map_for_arch(arch);

    // Vision encoder layer patterns: model.vision_tower.encoder.layers.N.<suffix> → v.blk.N.<gguf_suffix>
    let vision_layer_map: &[(&str, &str)] = &[
        ("self_attn.q_proj.linear.weight", "attn_q.weight"),
        ("self_attn.k_proj.linear.weight", "attn_k.weight"),
        ("self_attn.v_proj.linear.weight", "attn_v.weight"),
        ("self_attn.o_proj.linear.weight", "attn_output.weight"),
        ("self_attn.q_norm.weight", "attn_q_norm.weight"),
        ("self_attn.k_norm.weight", "attn_k_norm.weight"),
        ("input_layernorm.weight", "ln1.weight"),
        ("post_attention_layernorm.weight", "ln2.weight"),
        ("pre_feedforward_layernorm.weight", "ffn_norm.weight"),
        ("post_feedforward_layernorm.weight", "post_ffw_norm.weight"),
        ("mlp.gate_proj.linear.weight", "ffn_gate.weight"),
        ("mlp.up_proj.linear.weight", "ffn_up.weight"),
        ("mlp.down_proj.linear.weight", "ffn_down.weight"),
    ];

    const VISION_LAYER_PREFIX: &str = "model.vision_tower.encoder.layers.";
    if let Some(rest) = hf_name.strip_prefix(VISION_LAYER_PREFIX) {
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            if layer_num.chars().all(|c| c.is_ascii_digit()) {
                let suffix = &rest[dot_pos + 1..];
                for &(hf_suffix, gguf_suffix) in vision_layer_map {
                    if suffix == hf_suffix {
                        return format!("v.blk.{}.{}", layer_num, gguf_suffix);
                    }
                }
                // Pass through unknown vision layer suffixes with best-effort mapping
                return format!("v.blk.{}.{}", layer_num, suffix);
            }
        }
    }

    // Parse "model.layers.N.<suffix>" without regex
    const LAYER_PREFIX: &str = "model.layers.";
    if let Some(rest) = hf_name.strip_prefix(LAYER_PREFIX) {
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            // Verify it is actually a number
            if layer_num.chars().all(|c| c.is_ascii_digit()) {
                let suffix = &rest[dot_pos + 1..];
                for &(hf_suffix, gguf_suffix) in &layer_map {
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

/// Metadata value for GGUF key-value pairs.
enum MetaValue {
    String(String),
    Uint32(u32),
    Float32(f32),
    Bool(bool),
    ArrayBool(Vec<bool>),
    ArrayUint32(Vec<u32>),
    ArrayString(Vec<String>),
    ArrayFloat32(Vec<f32>),
    ArrayInt32(Vec<i32>),
}

// ---------------------------------------------------------------------------
// Tokenizer metadata
// ---------------------------------------------------------------------------

// Token type constants matching llama.cpp's TokenType enum
const TOKEN_TYPE_NORMAL: i32 = 1;
// TOKEN_TYPE_UNKNOWN = 2 — not used; <unk> is in added_tokens with special=true,
// so it gets TOKEN_TYPE_CONTROL like in llama.cpp's LlamaHfVocab.get_token_type().
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;
const TOKEN_TYPE_BYTE: i32 = 6;

/// Load tokenizer metadata from the model input directory and return GGUF
/// metadata key-value pairs for embedding into the GGUF file.
///
/// Parses `tokenizer.json` and `tokenizer_config.json` from `input_dir`.
/// Returns `None` if `tokenizer.json` is missing (graceful skip).
fn load_tokenizer_metadata(input_dir: &Path, arch: &str) -> Option<Vec<(String, MetaValue)>> {
    let tokenizer_path = input_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        warn!(
            "No tokenizer.json found in {}; skipping tokenizer embedding",
            input_dir.display()
        );
        return None;
    }

    let tokenizer_json: serde_json::Value = match std::fs::read_to_string(&tokenizer_path) {
        Ok(contents) => match serde_json::from_str(&contents) {
            Ok(v) => v,
            Err(e) => {
                warn!("Failed to parse tokenizer.json: {}; skipping tokenizer embedding", e);
                return None;
            }
        },
        Err(e) => {
            warn!("Failed to read tokenizer.json: {}; skipping tokenizer embedding", e);
            return None;
        }
    };

    // Parse tokenizer_config.json for special token definitions
    let config_path = input_dir.join("tokenizer_config.json");
    let tokenizer_config: Option<serde_json::Value> = std::fs::read_to_string(&config_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());

    let model_section = tokenizer_json.get("model")?;

    // Determine tokenizer model name
    let tokenizer_model_name = determine_tokenizer_model_name(model_section, arch);
    info!("Tokenizer model type: {}", tokenizer_model_name);

    // Extract vocab: HashMap<String, u32> -> sorted Vec<String> by ID
    let vocab_obj = model_section.get("vocab")?.as_object()?;
    let vocab_size = vocab_obj.len();
    let mut vocab_entries: Vec<(String, u32)> = vocab_obj
        .iter()
        .filter_map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32)))
        .collect();
    vocab_entries.sort_by_key(|(_, id)| *id);

    // Build ordered token string list
    // Fill gaps with empty strings (shouldn't happen for well-formed vocabs)
    let max_id = vocab_entries.last().map(|(_, id)| *id as usize).unwrap_or(0);
    let total_tokens = max_id + 1;
    let mut tokens: Vec<String> = vec![String::new(); total_tokens];
    for (token, id) in &vocab_entries {
        tokens[*id as usize] = token.clone();
    }
    info!("Loaded {} vocab tokens (max ID: {})", vocab_size, max_id);

    // Extract merges: may be Vec<Vec<String>> (new format) or Vec<String> (old format)
    let merges = extract_merges(model_section);
    info!("Loaded {} merges", merges.len());

    // Build set of added token IDs and special token IDs
    let added_tokens_arr = tokenizer_json.get("added_tokens").and_then(|v| v.as_array());
    let mut special_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
    if let Some(added) = added_tokens_arr {
        for entry in added {
            let is_special = entry.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
            if is_special {
                if let Some(id) = entry.get("id").and_then(|v| v.as_u64()) {
                    special_ids.insert(id as u32);
                }
            }
        }
    }

    // Gemma4 set_vocab marks certain tokens as USER_DEFINED for chat parser visibility
    let visible_tokens: std::collections::HashSet<&str> = [
        "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>",
        "<|tool_response>", "<tool_response|>", "<|\"|>",
    ]
    .iter()
    .copied()
    .collect();

    // Compute scores and token types for each token
    // LlamaHfVocab.get_token_score() returns -1000.0 for all tokens
    let scores: Vec<f32> = vec![-1000.0; total_tokens];

    let token_types: Vec<i32> = tokens
        .iter()
        .enumerate()
        .map(|(id, token)| {
            let id_u32 = id as u32;
            // Gemma4 overrides: certain visible tokens → USER_DEFINED
            if arch == "gemma4" && visible_tokens.contains(token.as_str()) {
                return TOKEN_TYPE_USER_DEFINED;
            }
            // Byte fallback tokens like <0x00>, <0xAB>
            if is_byte_token(token) {
                return TOKEN_TYPE_BYTE;
            }
            // Special/control tokens
            if special_ids.contains(&id_u32) {
                return TOKEN_TYPE_CONTROL;
            }
            TOKEN_TYPE_NORMAL
        })
        .collect();

    // Look up special token IDs from tokenizer_config.json
    let bos_id = resolve_special_token_id("bos_token", &tokenizer_config, &vocab_entries);
    let eos_id = resolve_special_token_id("eos_token", &tokenizer_config, &vocab_entries);
    let unk_id = resolve_special_token_id("unk_token", &tokenizer_config, &vocab_entries);
    let pad_id = resolve_special_token_id("pad_token", &tokenizer_config, &vocab_entries);

    // Build metadata KV pairs
    let mut kv: Vec<(String, MetaValue)> = Vec::new();

    // Required: tokenizer model name
    kv.push((
        "tokenizer.ggml.model".into(),
        MetaValue::String(tokenizer_model_name),
    ));

    // Token strings
    kv.push((
        "tokenizer.ggml.tokens".into(),
        MetaValue::ArrayString(tokens),
    ));

    // Scores
    kv.push((
        "tokenizer.ggml.scores".into(),
        MetaValue::ArrayFloat32(scores),
    ));

    // Token types
    kv.push((
        "tokenizer.ggml.token_type".into(),
        MetaValue::ArrayInt32(token_types),
    ));

    // Merges
    if !merges.is_empty() {
        kv.push((
            "tokenizer.ggml.merges".into(),
            MetaValue::ArrayString(merges),
        ));
    }

    // Special token IDs
    if let Some(id) = bos_id {
        kv.push((
            "tokenizer.ggml.bos_token_id".into(),
            MetaValue::Uint32(id),
        ));
    }
    if let Some(id) = eos_id {
        kv.push((
            "tokenizer.ggml.eos_token_id".into(),
            MetaValue::Uint32(id),
        ));
    }
    if let Some(id) = unk_id {
        kv.push((
            "tokenizer.ggml.unknown_token_id".into(),
            MetaValue::Uint32(id),
        ));
    }
    if let Some(id) = pad_id {
        kv.push((
            "tokenizer.ggml.padding_token_id".into(),
            MetaValue::Uint32(id),
        ));
    }

    // Bool flags
    kv.push((
        "tokenizer.ggml.add_bos_token".into(),
        MetaValue::Bool(true),
    ));
    kv.push((
        "tokenizer.ggml.add_space_prefix".into(),
        MetaValue::Bool(false),
    ));

    info!(
        "Tokenizer metadata: {} tokens, {} merges, bos={:?}, eos={:?}, unk={:?}, pad={:?}",
        total_tokens,
        kv.iter()
            .find(|(k, _)| k == "tokenizer.ggml.merges")
            .map(|(_, v)| match v {
                MetaValue::ArrayString(a) => a.len(),
                _ => 0,
            })
            .unwrap_or(0),
        bos_id,
        eos_id,
        unk_id,
        pad_id,
    );

    Some(kv)
}

/// Check if a token string matches the byte fallback pattern `<0xNN>` where NN
/// is exactly two hexadecimal digits (case insensitive).
fn is_byte_token(token: &str) -> bool {
    let bytes = token.as_bytes();
    bytes.len() == 6
        && bytes[0] == b'<'
        && bytes[1] == b'0'
        && bytes[2] == b'x'
        && bytes[3].is_ascii_hexdigit()
        && bytes[4].is_ascii_hexdigit()
        && bytes[5] == b'>'
}

/// Determine the GGUF tokenizer model name based on tokenizer.json contents and arch.
///
/// Mapping follows llama.cpp's convert_hf_to_gguf.py:
/// - Gemma4 with BPE + byte_fallback → "gemma4"
/// - Other BPE + byte_fallback + Sequence decoder → "llama" (SentencePiece-style)
/// - BPE without byte_fallback (ByteLevel decoder) → "gpt2"
fn determine_tokenizer_model_name(model_section: &serde_json::Value, arch: &str) -> String {
    let model_type = model_section
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let byte_fallback = model_section
        .get("byte_fallback")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if arch == "gemma4" {
        return "gemma4".into();
    }

    if model_type == "BPE" {
        if byte_fallback {
            // BPE with byte_fallback = SentencePiece-style → "llama"
            "llama".into()
        } else {
            // BPE without byte_fallback → GPT-2 style
            "gpt2".into()
        }
    } else {
        // Default to "llama" for SentencePiece/Unigram models
        "llama".into()
    }
}

/// Extract merges from the tokenizer.json model section.
///
/// Handles both old format (`Vec<String>`: `["a b", ...]`) and new format
/// (`Vec<Vec<String>>`: `[["a","b"], ...]`). In the new format, spaces within
/// merge parts are encoded as chr(288) per llama.cpp convention.
fn extract_merges(model_section: &serde_json::Value) -> Vec<String> {
    let merges_val = match model_section.get("merges") {
        Some(v) => v,
        None => return Vec::new(),
    };

    let merges_arr = match merges_val.as_array() {
        Some(a) if !a.is_empty() => a,
        _ => return Vec::new(),
    };

    // Detect format from first element
    if merges_arr[0].is_string() {
        // Old format: Vec<String> with "a b" format
        merges_arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    } else if merges_arr[0].is_array() {
        // New format: Vec<Vec<String>> — each element is a 2-element array
        // Per llama.cpp SpecialVocab: spaces in merge parts are encoded as chr(288)
        let space_replacement = '\u{0120}'; // chr(ord(' ') + 256) = chr(288) = 'Ġ'

        merges_arr
            .iter()
            .filter_map(|pair| {
                let arr = pair.as_array()?;
                if arr.len() != 2 {
                    return None;
                }
                let left = arr[0].as_str()?;
                let right = arr[1].as_str()?;

                // Check if any part contains spaces and encode them
                let left_encoded: String = left
                    .chars()
                    .map(|c| if c == ' ' { space_replacement } else { c })
                    .collect();
                let right_encoded: String = right
                    .chars()
                    .map(|c| if c == ' ' { space_replacement } else { c })
                    .collect();

                Some(format!("{} {}", left_encoded, right_encoded))
            })
            .collect()
    } else {
        warn!("Unknown merges format in tokenizer.json");
        Vec::new()
    }
}

/// Resolve a special token name (e.g., "bos_token") from tokenizer_config.json
/// to its vocabulary ID.
///
/// The config value may be a plain string (e.g., `"<bos>"`) or an object with
/// a `content` field (e.g., `{"content": "<bos>"}`).
fn resolve_special_token_id(
    token_key: &str,
    config: &Option<serde_json::Value>,
    vocab_entries: &[(String, u32)],
) -> Option<u32> {
    let config = config.as_ref()?;
    let token_val = config.get(token_key)?;

    // Extract the token string — it can be a string or an object with "content"
    let token_str = if let Some(s) = token_val.as_str() {
        s
    } else if let Some(content) = token_val.get("content").and_then(|v| v.as_str()) {
        content
    } else {
        return None;
    };

    // Look up token string in vocab to find its ID
    vocab_entries
        .iter()
        .find(|(tok, _)| tok == token_str)
        .map(|(_, id)| *id)
}

/// Build the GGUF metadata key-value list from model metadata.
fn build_metadata(model: &QuantizedModel, input_dir: &Path) -> Vec<(String, MetaValue)> {
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

    // For Gemma4, head_count_kv is a per-layer array (added below).
    // For other models, write as a scalar.
    if let Some(kv_heads) = meta.num_kv_heads {
        if arch != "gemma4" {
            kv.push((
                format!("{}.attention.head_count_kv", arch),
                MetaValue::Uint32(kv_heads),
            ));
        }
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

    // Context length — required by llama.cpp as {arch}.context_length
    let tc = meta.raw_config.get("text_config").cloned().unwrap_or_default();
    let ctx_len = tc.get("max_position_embeddings")
        .or_else(|| meta.raw_config.get("max_position_embeddings"))
        .and_then(|v| v.as_u64())
        .unwrap_or(131072) as u32;
    kv.push((
        format!("{}.context_length", arch),
        MetaValue::Uint32(ctx_len),
    ));

    // Gemma4-specific metadata required by llama.cpp
    if arch == "gemma4" {
        // RMS norm epsilon
        let rms_eps = tc.get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32;
        kv.push((
            format!("{}.attention.layer_norm_rms_epsilon", arch),
            MetaValue::Float32(rms_eps),
        ));

        // Sliding window size
        let swa = tc.get("sliding_window")
            .and_then(|v| v.as_u64())
            .unwrap_or(1024) as u32;
        kv.push((
            format!("{}.attention.sliding_window", arch),
            MetaValue::Uint32(swa),
        ));

        // Sliding window pattern (bool array: true = sliding, false = full)
        if let Some(layer_types) = tc.get("layer_types").and_then(|v| v.as_array()) {
            let swa_pattern: Vec<bool> = layer_types.iter()
                .map(|v| v.as_str() == Some("sliding_attention"))
                .collect();
            kv.push((
                format!("{}.attention.sliding_window_pattern", arch),
                MetaValue::ArrayBool(swa_pattern),
            ));
        }

        // Head dimensions: key_length (global), value_length (global)
        let global_head_dim = tc.get("global_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as u32;
        kv.push((
            format!("{}.attention.key_length", arch),
            MetaValue::Uint32(global_head_dim),
        ));
        kv.push((
            format!("{}.attention.value_length", arch),
            MetaValue::Uint32(global_head_dim),
        ));

        // Head dimensions: key_length_swa, value_length_swa (sliding)
        let swa_head_dim = tc.get("head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as u32;
        kv.push((
            format!("{}.attention.key_length_swa", arch),
            MetaValue::Uint32(swa_head_dim),
        ));
        kv.push((
            format!("{}.attention.value_length_swa", arch),
            MetaValue::Uint32(swa_head_dim),
        ));

        // Per-layer embedding (hidden_size_per_layer_input, 0 if absent)
        let n_pl_embd = tc.get("hidden_size_per_layer_input")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        kv.push((
            format!("{}.embedding_length_per_layer_input", arch),
            MetaValue::Uint32(n_pl_embd),
        ));

        // Expert feed-forward length (moe_intermediate_size)
        if let Some(moe_ff) = tc.get("moe_intermediate_size")
            .or_else(|| tc.get("expert_intermediate_size"))
            .and_then(|v| v.as_u64())
        {
            kv.push((
                format!("{}.expert_feed_forward_length", arch),
                MetaValue::Uint32(moe_ff as u32),
            ));
        }

        // Rope freq base (global attention)
        let rope_theta = tc.get("rope_parameters")
            .and_then(|rp| rp.get("full_attention"))
            .and_then(|fa| fa.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1_000_000.0) as f32;
        kv.push((
            format!("{}.rope.freq_base", arch),
            MetaValue::Float32(rope_theta),
        ));

        // Rope freq base for SWA (sliding attention)
        let rope_theta_swa = tc.get("rope_parameters")
            .and_then(|rp| rp.get("sliding_attention"))
            .and_then(|sa| sa.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(10_000.0) as f32;
        kv.push((
            format!("{}.rope.freq_base_swa", arch),
            MetaValue::Float32(rope_theta_swa),
        ));

        // Rope dimension counts
        let rope_dim_full = global_head_dim;
        let partial_rotary_factor_swa = tc.get("rope_parameters")
            .and_then(|rp| rp.get("sliding_attention"))
            .and_then(|sa| sa.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let rope_dim_swa = (swa_head_dim as f64 * partial_rotary_factor_swa) as u32;
        kv.push((
            format!("{}.rope.dimension_count", arch),
            MetaValue::Uint32(rope_dim_full),
        ));
        kv.push((
            format!("{}.rope.dimension_count_swa", arch),
            MetaValue::Uint32(rope_dim_swa),
        ));

        // KV head counts per layer (array: different for sliding vs global)
        let num_kv_heads_swa = tc.get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as u32;
        let num_kv_heads_full = tc.get("num_global_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as u32;
        if let Some(layer_types) = tc.get("layer_types").and_then(|v| v.as_array()) {
            let kv_heads_arr: Vec<u32> = layer_types.iter()
                .map(|v| if v.as_str() == Some("sliding_attention") { num_kv_heads_swa } else { num_kv_heads_full })
                .collect();
            kv.push((
                format!("{}.attention.head_count_kv", arch),
                MetaValue::ArrayUint32(kv_heads_arr),
            ));
        }

        // Softcapping
        if let Some(sc) = tc.get("final_logit_softcapping").and_then(|v| v.as_f64()) {
            kv.push((
                format!("{}.final_logit_softcapping", arch),
                MetaValue::Float32(sc as f32),
            ));
        }
    }

    // Load and embed tokenizer metadata from input directory
    if let Some(tok_kv) = load_tokenizer_metadata(input_dir, arch) {
        kv.extend(tok_kv);
    }

    kv
}

/// Map global bit width to a GGML file type code.
///
/// These correspond to llama.cpp's `llama_ftype` enum values.
/// For K-quant models the ftype is selected based on the dominant bit width.
fn ggml_ftype_from_bits(bits: u8) -> u32 {
    match bits {
        16 => 1,  // MOSTLY_F16
        8 => 7,   // MOSTLY_Q8_0
        4 => 15,  // MOSTLY_Q4_K_M (prefer K-quant over plain Q4_0)
        3 => 12,  // MOSTLY_Q3_K_M
        2 => 10,  // MOSTLY_Q2_K
        5 => 17,  // MOSTLY_Q5_K_M
        6 => 18,  // MOSTLY_Q6_K
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
        MetaValue::Float32(v) => {
            w.write_all(&GGUF_TYPE_FLOAT32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetaValue::ArrayBool(arr) => {
            w.write_all(&GGUF_TYPE_ARRAY.to_le_bytes())?;
            w.write_all(&GGUF_TYPE_BOOL.to_le_bytes())?;
            w.write_all(&(arr.len() as u64).to_le_bytes())?;
            for &b in arr {
                w.write_all(&[b as u8])?;
            }
        }
        MetaValue::Bool(v) => {
            w.write_all(&GGUF_TYPE_BOOL.to_le_bytes())?;
            w.write_all(&[*v as u8])?;
        }
        MetaValue::ArrayUint32(arr) => {
            w.write_all(&GGUF_TYPE_ARRAY.to_le_bytes())?;
            w.write_all(&GGUF_TYPE_UINT32.to_le_bytes())?;
            w.write_all(&(arr.len() as u64).to_le_bytes())?;
            for &v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        MetaValue::ArrayString(arr) => {
            w.write_all(&GGUF_TYPE_ARRAY.to_le_bytes())?;
            w.write_all(&GGUF_TYPE_STRING.to_le_bytes())?;
            w.write_all(&(arr.len() as u64).to_le_bytes())?;
            for s in arr {
                write_gguf_string(w, s)?;
            }
        }
        MetaValue::ArrayFloat32(arr) => {
            w.write_all(&GGUF_TYPE_ARRAY.to_le_bytes())?;
            w.write_all(&GGUF_TYPE_FLOAT32.to_le_bytes())?;
            w.write_all(&(arr.len() as u64).to_le_bytes())?;
            for &v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        MetaValue::ArrayInt32(arr) => {
            w.write_all(&GGUF_TYPE_ARRAY.to_le_bytes())?;
            w.write_all(&GGUF_TYPE_INT32.to_le_bytes())?;
            w.write_all(&(arr.len() as u64).to_le_bytes())?;
            for &v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
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
    #[allow(dead_code)]
    data_len: usize,
}

/// Write a tensor info entry in the GGUF header.
fn write_tensor_info<W: IoWrite>(w: &mut W, info: &TensorWriteInfo) -> std::io::Result<()> {
    // Name
    write_gguf_string(w, &info.gguf_name)?;
    // Number of dimensions
    w.write_all(&(info.shape.len() as u32).to_le_bytes())?;
    // Dimensions in ggml order (innermost/row first, reversed from PyTorch/safetensors).
    // e.g. PyTorch [128, 1408, 2816] → GGUF ne = [2816, 1408, 128]
    for &dim in info.shape.iter().rev() {
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
                ggml_type: None,
            },
        }
    }

    fn model(tensors: Vec<(&str, u8, bool)>, bits: u8) -> QuantizedModel {
        let map = tensors.into_iter().map(|(n, b, p)| (n.into(), tensor(n, b, p))).collect();
        QuantizedModel { metadata: meta(), tensors: map, quant_method: format!("q{}", bits), group_size: 64, bits }
    }

    #[test]
    fn test_name_mapping_llama() {
        // Static mappings (arch-independent)
        assert_eq!(hf_name_to_gguf("model.embed_tokens.weight", "llama"), "token_embd.weight");
        assert_eq!(hf_name_to_gguf("model.norm.weight", "llama"), "output_norm.weight");
        assert_eq!(hf_name_to_gguf("lm_head.weight", "llama"), "output.weight");
        // Layer mappings for LLaMA-family
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
        for (hf, gguf) in cases { assert_eq!(hf_name_to_gguf(hf, "llama"), gguf, "mapping failed for {}", hf); }
        // Unknown passthrough
        assert_eq!(hf_name_to_gguf("model.layers.5.some_new.weight", "llama"), "blk.5.some_new.weight");
        assert_eq!(hf_name_to_gguf("decoder.block.0.weight", "llama"), "decoder.block.0.weight");
        // Verify LLaMA-like archs all behave the same
        for arch in &["mistral", "qwen3", "qwen2", "phi"] {
            assert_eq!(
                hf_name_to_gguf("model.layers.0.post_attention_layernorm.weight", arch),
                "blk.0.ffn_norm.weight",
                "LLaMA-like arch '{}' should map post_attention_layernorm to ffn_norm", arch,
            );
        }
    }

    #[test]
    fn test_name_mapping_gemma4() {
        // Gemma4: post_attention_layernorm is a distinct norm, NOT ffn_norm
        assert_eq!(
            hf_name_to_gguf("model.layers.0.post_attention_layernorm.weight", "gemma4"),
            "blk.0.post_attention_norm.weight",
        );
        // Gemma4: pre_feedforward_layernorm IS ffn_norm
        assert_eq!(
            hf_name_to_gguf("model.layers.0.pre_feedforward_layernorm.weight", "gemma4"),
            "blk.0.ffn_norm.weight",
        );
        // MoE norms
        assert_eq!(
            hf_name_to_gguf("model.layers.5.post_feedforward_layernorm_1.weight", "gemma4"),
            "blk.5.post_ffw_norm_1.weight",
        );
        assert_eq!(
            hf_name_to_gguf("model.layers.5.post_feedforward_layernorm_2.weight", "gemma4"),
            "blk.5.post_ffw_norm_2.weight",
        );
        assert_eq!(
            hf_name_to_gguf("model.layers.5.pre_feedforward_layernorm_2.weight", "gemma4"),
            "blk.5.pre_ffw_norm_2.weight",
        );
        assert_eq!(
            hf_name_to_gguf("model.layers.5.post_feedforward_layernorm.weight", "gemma4"),
            "blk.5.post_ffw_norm.weight",
        );
        // MoE routing
        assert_eq!(
            hf_name_to_gguf("model.layers.3.router.proj.weight", "gemma4"),
            "blk.3.ffn_gate_inp.weight",
        );
        assert_eq!(
            hf_name_to_gguf("model.layers.3.router.scale", "gemma4"),
            "blk.3.ffn_gate_inp.scale",
        );
        assert_eq!(
            hf_name_to_gguf("model.layers.3.experts.gate_up_proj", "gemma4"),
            "blk.3.ffn_gate_up_exps.weight",
        );
        assert_eq!(
            hf_name_to_gguf("model.layers.3.experts.down_proj", "gemma4"),
            "blk.3.ffn_down_exps.weight",
        );
        assert_eq!(
            hf_name_to_gguf("model.layers.3.router.per_expert_scale", "gemma4"),
            "blk.3.ffn_down_exps.scale",
        );
        // Layer scalar
        assert_eq!(
            hf_name_to_gguf("model.layers.0.layer_scalar", "gemma4"),
            "blk.0.layer_output_scale.weight",
        );
        // Shared entries still work for Gemma4
        assert_eq!(
            hf_name_to_gguf("model.layers.0.self_attn.q_proj.weight", "gemma4"),
            "blk.0.attn_q.weight",
        );
        assert_eq!(
            hf_name_to_gguf("model.layers.0.input_layernorm.weight", "gemma4"),
            "blk.0.attn_norm.weight",
        );
        // Gemma3/Gemma2 behave the same as Gemma4
        assert_eq!(
            hf_name_to_gguf("model.layers.0.post_attention_layernorm.weight", "gemma3"),
            "blk.0.post_attention_norm.weight",
        );
        assert_eq!(
            hf_name_to_gguf("model.layers.0.post_attention_layernorm.weight", "gemma2"),
            "blk.0.post_attention_norm.weight",
        );
        // language_model. prefix stripping still works
        assert_eq!(
            hf_name_to_gguf("language_model.model.layers.0.pre_feedforward_layernorm.weight", "gemma4"),
            "blk.0.ffn_norm.weight",
        );
    }

    #[test]
    fn test_dtype_mapping() {
        let qi = |bits, preserved| TensorQuantInfo {
            method: "t".into(), bits, group_size: 64, preserved, scales: None, biases: None,
            ggml_type: None,
        };
        assert_eq!(quant_info_to_ggml_type(&qi(16, false)), GGML_TYPE_F16);
        assert_eq!(quant_info_to_ggml_type(&qi(8, false)), GGML_TYPE_Q8_0);
        assert_eq!(quant_info_to_ggml_type(&qi(4, false)), GGML_TYPE_Q4_0);
        // 2-bit maps to Q4_0 (we can't produce Q2_K super-block format)
        assert_eq!(quant_info_to_ggml_type(&qi(2, false)), GGML_TYPE_Q4_0);
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
        let tmp = tempfile::tempdir().unwrap();
        let kv = build_metadata(&model(vec![], 4), tmp.path());
        let keys: Vec<&str> = kv.iter().map(|(k, _)| k.as_str()).collect();
        for expected in ["general.architecture", "general.name", "llama.block_count",
            "llama.embedding_length", "llama.attention.head_count",
            "llama.attention.head_count_kv", "llama.feed_forward_length", "general.file_type"]
        {
            assert!(keys.contains(&expected), "Missing metadata key: {}", expected);
        }
    }

    #[test]
    fn test_ggml_type_from_name_all_kquants() {
        // Standard types
        assert_eq!(ggml_type_from_name("F32"), Some(GGML_TYPE_F32));
        assert_eq!(ggml_type_from_name("F16"), Some(GGML_TYPE_F16));
        assert_eq!(ggml_type_from_name("Q4_0"), Some(GGML_TYPE_Q4_0));
        assert_eq!(ggml_type_from_name("Q4_1"), Some(GGML_TYPE_Q4_1));
        assert_eq!(ggml_type_from_name("Q5_0"), Some(GGML_TYPE_Q5_0));
        assert_eq!(ggml_type_from_name("Q5_1"), Some(GGML_TYPE_Q5_1));
        assert_eq!(ggml_type_from_name("Q8_0"), Some(GGML_TYPE_Q8_0));
        assert_eq!(ggml_type_from_name("Q8_1"), Some(GGML_TYPE_Q8_1));

        // K-quant types
        assert_eq!(ggml_type_from_name("Q2_K"), Some(GGML_TYPE_Q2_K));
        assert_eq!(ggml_type_from_name("Q3_K_S"), Some(GGML_TYPE_Q3_K_S));
        assert_eq!(ggml_type_from_name("Q3_K_M"), Some(GGML_TYPE_Q3_K_M));
        assert_eq!(ggml_type_from_name("Q3_K_L"), Some(GGML_TYPE_Q3_K_L));
        assert_eq!(ggml_type_from_name("Q4_K_S"), Some(GGML_TYPE_Q4_K_S));
        assert_eq!(ggml_type_from_name("Q4_K_M"), Some(GGML_TYPE_Q4_K_M));
        assert_eq!(ggml_type_from_name("Q5_K_S"), Some(GGML_TYPE_Q5_K_S));
        assert_eq!(ggml_type_from_name("Q5_K_M"), Some(GGML_TYPE_Q5_K_M));
        assert_eq!(ggml_type_from_name("Q6_K"), Some(GGML_TYPE_Q6_K));
        assert_eq!(ggml_type_from_name("IQ2_XXS"), Some(GGML_TYPE_IQ2_XXS));
        assert_eq!(ggml_type_from_name("IQ2_XS"), Some(GGML_TYPE_IQ2_XS));

        // Aliases: Q3_K → Q3_K_M, Q4_K → Q4_K_M, Q5_K → Q5_K_M
        assert_eq!(ggml_type_from_name("Q3_K"), Some(GGML_TYPE_Q3_K_M));
        assert_eq!(ggml_type_from_name("Q4_K"), Some(GGML_TYPE_Q4_K_M));
        assert_eq!(ggml_type_from_name("Q5_K"), Some(GGML_TYPE_Q5_K_M));

        // Case insensitivity
        assert_eq!(ggml_type_from_name("q4_k_m"), Some(GGML_TYPE_Q4_K_M));
        assert_eq!(ggml_type_from_name("q6_k"), Some(GGML_TYPE_Q6_K));

        // With GGML_TYPE_ prefix
        assert_eq!(ggml_type_from_name("GGML_TYPE_Q4_K_M"), Some(GGML_TYPE_Q4_K_M));

        // Unknown returns None
        assert_eq!(ggml_type_from_name("Q99_Z"), None);
        assert_eq!(ggml_type_from_name(""), None);
    }

    #[test]
    fn test_ggml_type_override_in_quant_info() {
        // K-quant types from Apex are mapped to the corresponding simple block type
        // because hf2q doesn't produce K-quant super-block data.
        let qi_override = TensorQuantInfo {
            method: "apex".into(), bits: 4, group_size: 64, preserved: false,
            scales: None, biases: None, ggml_type: Some("Q4_K_M".into()),
        };
        assert_eq!(quant_info_to_ggml_type(&qi_override), GGML_TYPE_Q4_0);

        // Q6_K override on an 8-bit tensor maps to Q8_0
        let qi_q6k = TensorQuantInfo {
            method: "apex".into(), bits: 8, group_size: 64, preserved: false,
            scales: None, biases: None, ggml_type: Some("Q6_K".into()),
        };
        assert_eq!(quant_info_to_ggml_type(&qi_q6k), GGML_TYPE_Q8_0);

        // Non-K-quant explicit type is still honored
        let qi_q4_0 = TensorQuantInfo {
            method: "custom".into(), bits: 4, group_size: 32, preserved: false,
            scales: None, biases: None, ggml_type: Some("Q4_0".into()),
        };
        assert_eq!(quant_info_to_ggml_type(&qi_q4_0), GGML_TYPE_Q4_0);

        // Unknown ggml_type falls back to bits-based mapping
        let qi_unknown = TensorQuantInfo {
            method: "apex".into(), bits: 4, group_size: 64, preserved: false,
            scales: None, biases: None, ggml_type: Some("Q99_Z".into()),
        };
        assert_eq!(quant_info_to_ggml_type(&qi_unknown), GGML_TYPE_Q4_0);

        // preserved=true always returns F16 regardless of ggml_type
        let qi_preserved = TensorQuantInfo {
            method: "passthrough".into(), bits: 16, group_size: 0, preserved: true,
            scales: None, biases: None, ggml_type: Some("Q4_K_M".into()),
        };
        assert_eq!(quant_info_to_ggml_type(&qi_preserved), GGML_TYPE_F16);
    }

    #[test]
    fn test_ggml_type_none_falls_back_to_bits() {
        // Confirms backward compatibility: ggml_type=None uses bits mapping
        let qi = |bits| TensorQuantInfo {
            method: "t".into(), bits, group_size: 64, preserved: false,
            scales: None, biases: None, ggml_type: None,
        };
        assert_eq!(quant_info_to_ggml_type(&qi(16)), GGML_TYPE_F16);
        assert_eq!(quant_info_to_ggml_type(&qi(8)), GGML_TYPE_Q8_0);
        assert_eq!(quant_info_to_ggml_type(&qi(4)), GGML_TYPE_Q4_0);
        // 2-bit maps to Q4_0 (we can't produce Q2_K super-block format)
        assert_eq!(quant_info_to_ggml_type(&qi(2)), GGML_TYPE_Q4_0);
    }

    #[test]
    fn test_ftype_kquant_bits() {
        assert_eq!(ggml_ftype_from_bits(16), 1);  // MOSTLY_F16
        assert_eq!(ggml_ftype_from_bits(8), 7);   // MOSTLY_Q8_0
        assert_eq!(ggml_ftype_from_bits(4), 15);  // MOSTLY_Q4_K_M
        assert_eq!(ggml_ftype_from_bits(3), 12);  // MOSTLY_Q3_K_M
        assert_eq!(ggml_ftype_from_bits(2), 10);  // MOSTLY_Q2_K
        assert_eq!(ggml_ftype_from_bits(5), 17);  // MOSTLY_Q5_K_M
        assert_eq!(ggml_ftype_from_bits(6), 18);  // MOSTLY_Q6_K
        assert_eq!(ggml_ftype_from_bits(1), 1);   // unknown → F16
    }

    #[test]
    fn test_repack_q4_0_block_size() {
        // Create a quantized tensor with 64 elements (group_size=64),
        // verify repacking produces correct Q4_0 block count and size.
        let total_elements = 64usize;
        let group_size = 64usize;

        // Simulate hf2q's quantization: create scales and packed nibbles
        // 64 elements / group_size=64 = 1 scale
        let scale_f16 = half::f16::from_f32(0.5);
        let scales = scale_f16.to_le_bytes().to_vec();

        // Pack 64 signed i4 values (all zeros for simplicity)
        let packed_data = vec![0u8; total_elements / 2]; // 32 bytes

        let qt = QuantizedTensor {
            name: "test.weight".into(),
            shape: vec![8, 8],
            original_dtype: DType::F16,
            data: packed_data,
            quant_info: TensorQuantInfo {
                method: "q4".into(),
                bits: 4,
                group_size,
                preserved: false,
                scales: Some(scales),
                biases: None,
                ggml_type: None,
            },
        };

        let repacked = repack_to_ggml_blocks(&qt, GGML_TYPE_Q4_0).unwrap();

        // 64 elements / 32 per block = 2 blocks * 18 bytes = 36 bytes
        let expected_blocks = total_elements.div_ceil(QK4_0);
        let expected_size = expected_blocks * BLOCK_Q4_0_BYTES;
        assert_eq!(expected_blocks, 2);
        assert_eq!(expected_size, 36);
        assert_eq!(repacked.len(), expected_size,
            "Repacked Q4_0 size should be {} bytes (2 blocks * 18), got {}",
            expected_size, repacked.len()
        );

        // Verify each block starts with a 2-byte f16 scale followed by 16 bytes of nibbles
        // Block 0: bytes [0..2] = scale, [2..18] = nibbles
        // Block 1: bytes [18..20] = scale, [20..36] = nibbles
        assert_eq!(repacked.len(), 36);
    }

    #[test]
    fn test_repack_q4_0_roundtrip_values() {
        // Create a tensor with known values, repack to Q4_0, then verify
        // the dequantized values are approximately correct.
        let total_elements = 32usize;

        // 32 elements at group_size=32: 1 group, 1 scale
        // Values: [1.0, -1.0, 0.5, -0.5, ...] (repeat)
        let f32_vals: Vec<f32> = (0..32).map(|i| {
            if i % 4 == 0 { 1.0 }
            else if i % 4 == 1 { -1.0 }
            else if i % 4 == 2 { 0.5 }
            else { -0.5 }
        }).collect();

        // Quantize like hf2q does: symmetric, scale = absmax / 7
        let absmax = 1.0f32;
        let scale = absmax / 7.0;
        let scale_f16 = half::f16::from_f32(scale);
        let scales = scale_f16.to_le_bytes().to_vec();

        // Quantize to signed i4
        let signed_qs: Vec<i8> = f32_vals.iter().map(|&v| {
            let q = (v / scale).round() as i8;
            q.clamp(-7, 7)
        }).collect();

        // Pack as hf2q does: consecutive pairs, lo nibble first
        let mut packed = Vec::with_capacity(total_elements / 2);
        for pair in signed_qs.chunks(2) {
            let lo = (pair[0] & 0x0F) as u8;
            let hi = if pair.len() > 1 { ((pair[1] & 0x0F) as u8) << 4 } else { 0 };
            packed.push(lo | hi);
        }

        let qt = QuantizedTensor {
            name: "test.weight".into(),
            shape: vec![4, 8],
            original_dtype: DType::F16,
            data: packed,
            quant_info: TensorQuantInfo {
                method: "q4".into(),
                bits: 4,
                group_size: 32,
                preserved: false,
                scales: Some(scales),
                biases: None,
                ggml_type: None,
            },
        };

        let repacked = repack_to_ggml_blocks(&qt, GGML_TYPE_Q4_0).unwrap();

        // Should be exactly 1 Q4_0 block = 18 bytes
        assert_eq!(repacked.len(), BLOCK_Q4_0_BYTES);

        // Extract scale from repacked block
        let d_bits = u16::from_le_bytes([repacked[0], repacked[1]]);
        let d = half::f16::from_bits(d_bits).to_f32();

        // Dequantize Q4_0 block and check approximate values
        // Nibble packing: byte[i] = q_lo[i] | (q_hi[i] << 4)
        // where q_lo = first 16 elements, q_hi = second 16 elements
        let mut dequantized = vec![0.0f32; 32];
        for i in 0..16 {
            let byte = repacked[2 + i];
            let q_lo = (byte & 0x0F) as i32 - 8;
            let q_hi = ((byte >> 4) & 0x0F) as i32 - 8;
            dequantized[i] = q_lo as f32 * d;
            dequantized[i + 16] = q_hi as f32 * d;
        }

        // Check that dequantized values are close to originals (within quantization error)
        for (i, (&orig, &deq)) in f32_vals.iter().zip(dequantized.iter()).enumerate() {
            let err = (orig - deq).abs();
            assert!(err < 0.2, "Element {}: orig={}, deq={}, err={}", i, orig, deq, err);
        }
    }

    #[test]
    fn test_repack_q8_0_block_size() {
        let total_elements = 64usize;

        // 2 groups of 32 with 2 scales
        let s0 = half::f16::from_f32(0.5);
        let s1 = half::f16::from_f32(0.3);
        let mut scales = Vec::new();
        scales.extend_from_slice(&s0.to_le_bytes());
        scales.extend_from_slice(&s1.to_le_bytes());

        // 64 int8 values (all zeros)
        let data = vec![0u8; total_elements];

        let qt = QuantizedTensor {
            name: "test.weight".into(),
            shape: vec![8, 8],
            original_dtype: DType::F16,
            data,
            quant_info: TensorQuantInfo {
                method: "q8".into(),
                bits: 8,
                group_size: 32,
                preserved: false,
                scales: Some(scales),
                biases: None,
                ggml_type: None,
            },
        };

        let repacked = repack_to_ggml_blocks(&qt, GGML_TYPE_Q8_0).unwrap();

        // 64 elements / 32 per block = 2 blocks * 34 bytes = 68 bytes
        assert_eq!(repacked.len(), 2 * BLOCK_Q8_0_BYTES);
    }

    #[test]
    fn test_repack_preserved_tensor_passthrough() {
        // Preserved tensors should pass through unchanged
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let qt = QuantizedTensor {
            name: "norm.weight".into(),
            shape: vec![4],
            original_dtype: DType::F16,
            data: data.clone(),
            quant_info: TensorQuantInfo {
                method: "passthrough".into(),
                bits: 16,
                group_size: 0,
                preserved: true,
                scales: None,
                biases: None,
                ggml_type: None,
            },
        };

        let repacked = repack_to_ggml_blocks(&qt, GGML_TYPE_F16).unwrap();
        assert_eq!(repacked, data);
    }

    #[test]
    fn test_ggml_tensor_size_q4_0() {
        // 1024 elements: 1024/32 = 32 blocks * 18 bytes = 576 bytes
        assert_eq!(ggml_tensor_size(1024, GGML_TYPE_Q4_0), 576);
        // 32 elements: 1 block * 18 bytes
        assert_eq!(ggml_tensor_size(32, GGML_TYPE_Q4_0), 18);
        // 64 elements: 2 blocks * 18 bytes = 36
        assert_eq!(ggml_tensor_size(64, GGML_TYPE_Q4_0), 36);
    }

    #[test]
    fn test_ggml_tensor_size_q8_0() {
        // 1024 elements: 1024/32 = 32 blocks * 34 bytes = 1088 bytes
        assert_eq!(ggml_tensor_size(1024, GGML_TYPE_Q8_0), 1088);
        // 32 elements: 1 block * 34 bytes
        assert_eq!(ggml_tensor_size(32, GGML_TYPE_Q8_0), 34);
    }

    #[test]
    fn test_ggml_tensor_size_f16() {
        assert_eq!(ggml_tensor_size(1024, GGML_TYPE_F16), 2048);
    }
}
