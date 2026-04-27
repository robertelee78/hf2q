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
    FormatWarning, OutputFile, OutputManifest, QuantizedModel, TensorQuantInfo,
    WarningSeverity,
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
// K-quant type codes (match ggml.h exactly)
const GGML_TYPE_Q2_K: u32 = 10;
const GGML_TYPE_Q3_K: u32 = 11;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;
#[allow(dead_code)]
const GGML_TYPE_Q8_K: u32 = 15;
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
            if !tensor.quant_info.preserved && !matches!(bits, 2 | 4 | 6 | 8 | 16) {
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
            // Use the output directory name as the base filename
            let dir_name = output_dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| sanitize_model_type(&model.metadata.model_type));
            let fname = format!("{}.gguf", dir_name);
            (output_dir.join(&fname), fname)
        };
        info!("Writing GGUF to {}", out_path.display());

        // Collect tensors in deterministic order.
        // Filter out vision/audio tensors — llama.cpp expects those in a separate
        // mmproj GGUF file, not in the text model GGUF. If they're included, the
        // text model loader rejects the file with "wrong number of tensors".
        let mut tensor_names: Vec<&String> = model.tensors.keys()
            .filter(|name| {
                // Skip vision/audio tensors — they belong in a separate mmproj GGUF.
                // Three namespace conventions seen in the wild:
                //   - Gemma:    model.vision_tower.* + model.embed_vision.*
                //   - Qwen3.6:  model.visual.*   (real-model finding 2026-04-25)
                //   - Audio:    model.audio_tower.*
                let n = name.as_str();
                !(n.contains("vision_tower")
                    || n.contains("embed_vision")
                    || n.contains("audio_tower")
                    || n.contains("model.visual."))
            })
            .collect();
        tensor_names.sort();

        // Generate synthetic tensors (e.g., rope_freqs for Gemma4 partial RoPE)
        let synthetic_tensors = generate_synthetic_tensors(&model.metadata);

        // For Gemma4 full-attention layers, V is tied to K (no separate v_proj in safetensors).
        // llama.cpp expects attn_v.weight to exist. Duplicate K data as V for these layers.
        let mut v_duplicates: Vec<(String, &String)> = Vec::new(); // (gguf_v_name, source_k_hf_name)
        if model.metadata.model_type == "gemma4" {
            let tc = model.metadata.raw_config.get("text_config").cloned().unwrap_or_default();
            let k_eq_v = tc.get("attention_k_eq_v").and_then(|v| v.as_bool()).unwrap_or(true);
            if k_eq_v {
                let layer_types = tc.get("layer_types").and_then(|v| v.as_array());
                if let Some(lt) = layer_types {
                    for (i, lt_val) in lt.iter().enumerate() {
                        if lt_val.as_str() == Some("full_attention") {
                            // Find the K tensor for this layer
                            let k_hf_suffix = format!("layers.{}.self_attn.k_proj.weight", i);
                            if let Some(k_name) = tensor_names.iter().find(|n| n.ends_with(&k_hf_suffix)) {
                                let v_gguf_name = format!("blk.{}.attn_v.weight", i);
                                v_duplicates.push((v_gguf_name, k_name));
                            }
                        }
                    }
                }
            }
        }

        let pb = progress.bar(
            (tensor_names.len() + synthetic_tensors.len() + v_duplicates.len()) as u64,
            "Writing GGUF tensors",
        );

        // Build metadata key-value pairs
        let metadata = build_metadata(model, input_dir);
        let tensor_count = (tensor_names.len() + synthetic_tensors.len() + v_duplicates.len()) as u64;
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

        // Resolve GGUF arch once for tensor name mapping (ADR-012 Decision 1)
        let resolved_arch_for_tensors = arch_gguf_name(&model.metadata);

        for name in &tensor_names {
            let qt = &model.tensors[*name];
            let gguf_name = hf_name_to_gguf(
                name,
                &resolved_arch_for_tensors,
                model.metadata.num_layers,
            );
            let mut ggml_type = quant_info_to_ggml_type(&qt.quant_info);

            // 1D scale/scalar tensors must be F32 — llama.cpp's Metal kernels
            // assume F32 for element-wise operations (router.scale, per_expert_scale,
            // layer_scalar, norms). Override F16→F32 for these.
            let mut needs_f32 = qt.quant_info.preserved && qt.shape.len() <= 1;

            // ADR-012 Decision 19 follow-up (2026-04-25 cron-iter):
            // qwen35 / qwen35moe ssm_conv1d.weight MUST be F32 — not F16.
            // The Metal SSM_CONV kernel (`ggml_metal_op_ssm_conv` in
            // ggml-metal-device.cpp:490) asserts
            //   GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32)
            // and aborts the process at the first generation step if the
            // conv1d weight is F16.  ssm_conv1d has shape [n_v_heads, K=4]
            // (small inner dim → preserved by `is_weight()`'s ≥32-row guard,
            // so it would otherwise stay at F16).  Promote to F32 here.
            //
            // Detection is by GGUF tensor name (post-rename), so the same
            // guard fires for both qwen35 and qwen35moe without an arch
            // string check (no other arch emits a tensor named
            // `*.ssm_conv1d.weight`).
            if !needs_f32 && gguf_name.ends_with(".ssm_conv1d.weight") {
                needs_f32 = true;
            }

            if needs_f32 && ggml_type == GGML_TYPE_F16 {
                ggml_type = GGML_TYPE_F32;
            }

            // Block-quantized K-quant types (block_size=256) require ne[0] % 256 == 0.
            // ne[0] is the last PyTorch dim (innermost, becomes first after GGUF reversal).
            // When incompatible, use llama.cpp's standard fallback chain from
            // tensor_type_fallback() in llama-quant.cpp:362-408:
            //   Q6_K → Q8_0, Q5_K → Q5_1, Q4_K → Q5_0, Q3_K/Q2_K → Q4_0
            //
            // ADR-012 P9b real-model finding (2026-04-25): the original chain
            // only handled K-quant types when row_dim wasn't divisible by 256.
            // But Q4_0 / Q5_0 / Q8_0 (block_size = 32 = QK4_0) ALSO require
            // row_dim divisible by their block size. Tensors with very small
            // row_dim (e.g. ssm_conv1d.weight with K=4 conv kernel) aren't
            // representable in any block-quant type — must fall back to F16.
            // Without this fix, the GGUF emits with row_dim=4 Q4_0 which
            // llama.cpp's loader rejects:
            //   "blk.0.ssm_conv1d.weight of type 2 (q4_0) has 4 elements per
            //    row, not a multiple of block size (32)"
            if !qt.shape.is_empty() {
                let row_dim = *qt.shape.last().unwrap();
                if row_dim % QK6_K != 0 {
                    let fallback = match ggml_type {
                        GGML_TYPE_Q6_K => Some(GGML_TYPE_Q8_0),
                        GGML_TYPE_Q5_K => Some(GGML_TYPE_Q5_1),
                        GGML_TYPE_Q4_K => Some(GGML_TYPE_Q5_0),
                        GGML_TYPE_Q3_K | GGML_TYPE_Q2_K => Some(GGML_TYPE_Q4_0),
                        _ => None,
                    };
                    if let Some(fb) = fallback {
                        if row_dim % QK4_0 == 0 {
                            debug!(
                                "Tensor '{}': ne[0]={} not K-quant compatible, {} → {}",
                                gguf_name, row_dim,
                                ggml_type_name(ggml_type), ggml_type_name(fb)
                            );
                            ggml_type = fb;
                        } else {
                            debug!(
                                "Tensor '{}': ne[0]={} not block-aligned at all, using F16",
                                gguf_name, row_dim
                            );
                            ggml_type = GGML_TYPE_F16;
                        }
                    }
                }

                // Independent of the K-quant chain above: any Q4_0 / Q5_0 /
                // Q8_0 (block 32) tensor with row_dim < 32 OR row_dim not
                // divisible by 32 → fall back to F16. Catches ssm_conv1d
                // (K=4) and any other small-row-dim weight that slipped
                // past the K-quant fallback.
                let block_32_quant = matches!(
                    ggml_type,
                    GGML_TYPE_Q4_0 | GGML_TYPE_Q5_0 | GGML_TYPE_Q8_0
                );
                if block_32_quant && row_dim % QK4_0 != 0 {
                    debug!(
                        "Tensor '{}': ne[0]={} not block-32 aligned, {} → F16",
                        gguf_name, row_dim, ggml_type_name(ggml_type)
                    );
                    ggml_type = GGML_TYPE_F16;
                }
            }

            // Compute the repacked size without allocating
            let total_elements: usize = qt.shape.iter().product();
            let repacked_size = if needs_f32 {
                total_elements * 4 // F32 = 4 bytes per element
            } else if qt.quant_info.preserved || ggml_type == GGML_TYPE_F16 || ggml_type == GGML_TYPE_F32 {
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

        // Add synthetic tensor infos
        for (name, data, shape, ggml_type) in &synthetic_tensors {
            tensor_data_offset = align_up(tensor_data_offset, ALIGNMENT);
            tensor_infos.push(TensorWriteInfo {
                gguf_name: name.clone(),
                shape: shape.clone(),
                ggml_type: *ggml_type,
                data_offset: tensor_data_offset,
                data_len: data.len(),
            });
            tensor_data_offset += data.len() as u64;
        }

        // Add V=K duplicate tensor infos (for Gemma4 full-attention layers)
        for (v_name, k_hf_name) in &v_duplicates {
            let k_qt = &model.tensors[*k_hf_name];
            let k_ggml_type = quant_info_to_ggml_type(&k_qt.quant_info);
            let total_elements: usize = k_qt.shape.iter().product();
            let needs_f32 = k_qt.quant_info.preserved && k_qt.shape.len() <= 1;
            let effective_type = if needs_f32 && k_ggml_type == GGML_TYPE_F16 { GGML_TYPE_F32 } else { k_ggml_type };
            let repacked_size = if needs_f32 {
                total_elements * 4
            } else if k_qt.quant_info.preserved || effective_type == GGML_TYPE_F16 || effective_type == GGML_TYPE_F32 {
                k_qt.data.len()
            } else {
                ggml_tensor_size(total_elements, effective_type)
            };
            tensor_data_offset = align_up(tensor_data_offset, ALIGNMENT);
            tensor_infos.push(TensorWriteInfo {
                gguf_name: v_name.clone(),
                shape: k_qt.shape.clone(),
                ggml_type: effective_type,
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

        // --- Write synthetic tensors ---
        for (name, data, _shape, _ggml_type) in &synthetic_tensors {
            let current = w.stream_position()?;
            let info = tensor_infos.iter().find(|i| &i.gguf_name == name).unwrap();
            let target = data_block_start + info.data_offset;
            if current < target {
                w.write_all(&vec![0u8; (target - current) as usize])?;
            }
            w.write_all(data)?;
            pb.inc(1);
        }

        // --- Write V=K duplicate tensors (Gemma4 full-attention layers) ---
        for (v_name, k_hf_name) in &v_duplicates {
            let k_qt = &model.tensors[*k_hf_name];
            let info = tensor_infos.iter().find(|i| i.gguf_name == *v_name).unwrap();

            // Pad to alignment
            let current = w.stream_position()?;
            let target = data_block_start + info.data_offset;
            if current < target {
                w.write_all(&vec![0u8; (target - current) as usize])?;
            }

            // Repack K tensor data (same repack as the K tensor gets) and write as V
            let data = repack_to_ggml_blocks(k_qt, info.ggml_type).map_err(|e| {
                BackendError::WriteFailed {
                    reason: format!(
                        "Failed to repack V=K duplicate tensor '{}' from '{}': {}",
                        v_name, k_hf_name, e
                    ),
                }
            })?;
            w.write_all(&data)?;
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

        let mut output_files = vec![OutputFile {
            filename,
            size_bytes: file_size,
        }];
        let mut total_size = file_size;

        // Check for vision tensors and write mmproj GGUF if present
        let has_vision = model.tensors.keys().any(|name| {
            let n = name.as_str();
            n.contains("vision_tower") || n.contains("embed_vision")
        });

        if has_vision {
            let mmproj_result = write_mmproj_gguf(model, &out_path, progress)?;
            total_size += mmproj_result.size_bytes;
            output_files.push(mmproj_result);
        }

        Ok(OutputManifest {
            output_dir: manifest_dir,
            files: output_files,
            total_size_bytes: total_size,
            shard_count: 1,
        })
    }
}

// ---------------------------------------------------------------------------
// Mmproj (vision) GGUF writer
// ---------------------------------------------------------------------------

/// Derive the mmproj output path from the text GGUF path.
///
/// Inserts `-mmproj` before the `.gguf` extension:
///   `model-text.gguf` -> `model-text-mmproj.gguf`
fn mmproj_path_from_text(text_path: &Path) -> std::path::PathBuf {
    let stem = text_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let new_name = format!("{}-mmproj.gguf", stem);
    text_path.with_file_name(new_name)
}

/// Build mmproj-specific GGUF metadata from the model's vision config.
///
/// The mmproj GGUF uses `general.architecture = "clip"` and `general.type = "mmproj"`,
/// with all vision parameters under the `clip.vision.*` namespace. This matches the
/// format expected by llama.cpp's clip.cpp loader.
fn build_mmproj_metadata(model: &QuantizedModel) -> Vec<(String, MetaValue)> {
    let meta = &model.metadata;
    let vc = meta.raw_config.get("vision_config").cloned().unwrap_or_default();
    let tc = meta.raw_config.get("text_config").cloned().unwrap_or_default();

    // Projector type — Gemma4 uses "gemma4v"
    let projector_type = match meta.model_type.as_str() {
        "gemma4" => "gemma4v",
        "gemma3" => "gemma3",
        _ => "mlp",
    };

    // Vision geometry parameters from vision_config.
    let image_size = vc.get("image_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(224) as u32;
    let patch_size = vc.get("patch_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(14) as u32;
    let vision_hidden_size = vc.get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(1024) as u32;
    let vision_intermediate_size = vc.get("intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as u32;
    let vision_block_count = vc.get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(24) as u32;
    let vision_head_count = vc.get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as u32;
    let layer_norm_eps = vc.get("layer_norm_eps")
        .or_else(|| vc.get("rms_norm_eps"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;

    // Projection dimension = text model hidden size (the target dim for the projection).
    let text_hidden_size = tc.get("hidden_size")
        .and_then(|v| v.as_u64())
        .or_else(|| meta.raw_config.get("hidden_size").and_then(|v| v.as_u64()))
        .unwrap_or(meta.hidden_size) as u32;

    vec![
        // Core identification — mmproj uses "clip" architecture.
        ("general.architecture".into(), MetaValue::String("clip".into())),
        ("general.type".into(), MetaValue::String("mmproj".into())),
        ("general.file_type".into(), MetaValue::Uint32(ggml_ftype_from_bits(model.bits))),
        ("clip.has_vision_encoder".into(), MetaValue::Bool(true)),
        // NOTE: projector_type is un-namespaced in llama.cpp — the only top-level `clip.*`
        // key in this metadata block. See vendored clip-impl.h:23 (KEY_PROJ_TYPE).
        // Loader counterpart: src/inference/vision/mmproj.rs:148.
        ("clip.projector_type".into(), MetaValue::String(projector_type.into())),
        // Vision geometry.
        ("clip.vision.image_size".into(), MetaValue::Uint32(image_size)),
        ("clip.vision.patch_size".into(), MetaValue::Uint32(patch_size)),
        ("clip.vision.embedding_length".into(), MetaValue::Uint32(vision_hidden_size)),
        ("clip.vision.feed_forward_length".into(), MetaValue::Uint32(vision_intermediate_size)),
        ("clip.vision.block_count".into(), MetaValue::Uint32(vision_block_count)),
        ("clip.vision.attention.head_count".into(), MetaValue::Uint32(vision_head_count)),
        ("clip.vision.attention.layer_norm_epsilon".into(), MetaValue::Float32(layer_norm_eps)),
        ("clip.vision.projection_dim".into(), MetaValue::Uint32(text_hidden_size)),
        // Image mean/std — Gemma4 uses standardization via tensors (std_bias/std_scale),
        // but llama.cpp still expects these metadata fields. Use [0.5, 0.5, 0.5] as the
        // default for Gemma4 (ImageNet-style normalization when no preprocessor_config).
        ("clip.vision.image_mean".into(), MetaValue::ArrayFloat32(vec![0.5, 0.5, 0.5])),
        ("clip.vision.image_std".into(), MetaValue::ArrayFloat32(vec![0.5, 0.5, 0.5])),
    ]
}

/// Write a separate mmproj GGUF file containing vision encoder and multimodal
/// projection tensors. Returns the `OutputFile` entry for the manifest.
///
/// The mmproj file contains only tensors with vision-related HF names
/// (`vision_tower.*`, `embed_vision.*`) — exactly the tensors filtered OUT of
/// the text GGUF. Tensor names are mapped using the same `hf_name_to_gguf()`
/// function which already handles `v.blk.N.*` and `mm.input_projection.*`
/// patterns (was `mm.0.*` pre-iter-116f, before the gemma4v projector
/// rename).
fn write_mmproj_gguf(
    model: &QuantizedModel,
    text_path: &Path,
    progress: &ProgressReporter,
) -> Result<OutputFile, BackendError> {
    let mmproj_path = mmproj_path_from_text(text_path);
    let mmproj_filename = mmproj_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();

    info!("Writing mmproj GGUF to {}", mmproj_path.display());

    // Collect vision tensors in deterministic order
    let mut vision_tensor_names: Vec<&String> = model.tensors.keys()
        .filter(|name| {
            let n = name.as_str();
            n.contains("vision_tower") || n.contains("embed_vision")
        })
        .collect();
    vision_tensor_names.sort();

    if vision_tensor_names.is_empty() {
        return Err(BackendError::WriteFailed {
            reason: "No vision tensors found for mmproj GGUF".into(),
        });
    }

    info!(
        "Mmproj: {} vision tensors to write",
        vision_tensor_names.len()
    );

    let pb = progress.bar(
        vision_tensor_names.len() as u64,
        "Writing mmproj GGUF tensors",
    );

    // Build mmproj metadata
    let metadata = build_mmproj_metadata(model);
    let tensor_count = vision_tensor_names.len() as u64;
    let kv_count = metadata.len() as u64;

    let file = File::create(&mmproj_path).map_err(|e| BackendError::WriteFailed {
        reason: format!("Failed to create {}: {}", mmproj_path.display(), e),
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

    // --- Pass 1: Compute tensor sizes and offsets ---
    let mut tensor_infos: Vec<TensorWriteInfo> = Vec::with_capacity(vision_tensor_names.len());
    let mut tensor_data_offset: u64 = 0;

    for name in &vision_tensor_names {
        let qt = &model.tensors[*name];
        let gguf_name = hf_name_to_gguf(
            name,
            &model.metadata.model_type,
            model.metadata.num_layers,
        );
        let mut ggml_type = quant_info_to_ggml_type(&qt.quant_info);

        // ADR-005 Phase 2c iter-116a (renamed iter-116f): clamp-scalar
        // emit for Gemma4 ClippableLinear bounds
        // (`mm.input_projection.{input_min,input_max,output_min,output_max}`
        // — was `mm.0.*` pre-iter-116f, see hf_name_to_gguf static_map +
        // test_gemma4v_clippable_linear_scalar_bounds_mapping for the
        // canonical name + llama.cpp citation).
        // HF safetensors stores these as 0-d F32 scalars;
        // llama.cpp's CLIP loader expects 1-D F32 tensors — see
        // `convert_hf_to_gguf.py:7851-7879` (`Gemma4VisionAudioModel.modify_tensors`,
        // `data_torch.unsqueeze(0)` + `tensor_force_quant` returning F32).
        //
        // Promote the shape to `[1]` and the ggml_type to F32 here so the
        // emitted mmproj round-trips through both hf2q's `mmproj_weights`
        // loader (W25) and llama-mtmd-cli's CLIP loader.  The shape
        // promotion is unconditional for 0-d tensors (defensive — only
        // clamp scalars hit this path today, but any future 0-d tensor
        // would need the same fix).  The F32 promotion is gated on the
        // tensor name's suffix so we don't change behavior for any other
        // 0-d tensor that might want to stay F16.
        let is_clamp_scalar = name.ends_with(".input_min")
            || name.ends_with(".input_max")
            || name.ends_with(".output_min")
            || name.ends_with(".output_max");
        let mut emit_shape: Vec<usize> = if qt.shape.is_empty() {
            vec![1]
        } else {
            qt.shape.clone()
        };
        if is_clamp_scalar {
            ggml_type = GGML_TYPE_F32;
        }

        // ADR-005 Phase 2c iter-116g: F32-promote 1-D tensors, *_norm.weight,
        // and the gemma4v position-embd table to match llama.cpp's CLIP
        // convention. The convert_hf_to_gguf.py base writer at
        // `/opt/llama.cpp/convert_hf_to_gguf.py:807-809` forces:
        //
        //     if n_dims <= 1 or new_name.endswith("_norm.weight"):
        //         data_qtype = gguf.GGMLQuantizationType.F32
        //
        // and gemma4v's `tensor_force_quant` override at
        // `/opt/llama.cpp/convert_hf_to_gguf.py:7841-7842` adds:
        //
        //     if "position_embedding_table" in name:
        //         return gguf.GGMLQuantizationType.F32
        //
        // The runtime symptom of NOT promoting these: clip's CPU graph
        // crashes inside `ggml_compute_forward_div` with
        // `binary_op: unsupported types: dst: f32, src0: f32, src1: f16`
        // because RMS-norm internally divides an F32 sum by an F32
        // intermediate, then multiplies by the F16 norm weight. The
        // ggml-cpu binary-op kernel does not accept the resulting
        // F32/F32/F16 triple. Promoting the norm weight + 1-D tensors +
        // position-embd lookup table to F32 is the upstream-mandated fix.
        let is_one_d = qt.shape.len() <= 1;
        let is_norm_weight = gguf_name.ends_with("_norm.weight");
        let is_position_embd = gguf_name == "v.position_embd.weight";
        if (is_one_d || is_norm_weight || is_position_embd)
            && (ggml_type == GGML_TYPE_F16)
        {
            ggml_type = GGML_TYPE_F32;
        }

        // ADR-005 Phase 2c iter-116g: gemma4v patch-embd Conv2d weight reshape.
        //
        // HF safetensors stores `model.vision_tower.patch_embedder.input_proj.weight`
        // as 2-D `[n_embd, patch_size² · 3]` (e.g. `[1152, 768]` with
        // patch_size=16, in_channels=3). llama.cpp's clip-graph builds
        // `ggml_conv_2d(model.patch_embeddings_0, inp_raw, ...)` where
        // `inp_raw` is 3-D `[nx, ny, channels=3]` — see
        // `/opt/llama.cpp/tools/mtmd/models/gemma4v.cpp:12`. The im2col
        // kernel asserts `a->ne[2] == b->ne[2]` (3 in-channels match)
        // at `/opt/llama.cpp/ggml/src/ggml.c:4412`. With a 2-D weight
        // `a->ne[2] = 1` ≠ `b->ne[2] = 3` and the convert assert fires.
        //
        // The reshape contract — verbatim from
        // `/opt/llama.cpp/convert_hf_to_gguf.py:7873-7877`
        // (`Gemma4VisionAudioModel.modify_tensors`):
        //
        //     n_embd, ksize_sq_c = data_torch.shape
        //     patch_size = int((ksize_sq_c // 3) ** 0.5)
        //     data_torch = data_torch.reshape(n_embd, patch_size, patch_size, 3)
        //     data_torch = data_torch.permute(0, 3, 1, 2).contiguous()
        //
        // The reshape (`[n_embd, hwc] → [n_embd, h, w, c]`) is a metadata
        // operation on contiguous row-major data — no byte movement. The
        // permute (`[n_embd, h, w, c] → [n_embd, c, h, w]`) is a physical
        // transpose: every output element moves. We do both in
        // `transpose_patch_embd_hwc_to_chw` and the writer (Pass 2 below)
        // picks up the transposed bytes via the same name-keyed branch.
        //
        // Output shape `[n_embd, c, h, w]` is in PyTorch convention; the
        // GGUF writer reverses dims on emit (see `write_tensor_info`),
        // so the on-disk `ne[]` becomes `[w, h, c, n_embd]` = `[16, 16, 3, 1152]`,
        // which matches `a->ne[2] = c = 3` for the im2col assert.
        let is_patch_embd_conv =
            gguf_name == "v.patch_embd.weight" && emit_shape.len() == 2;
        if is_patch_embd_conv {
            let n_embd = emit_shape[0];
            let hwc = emit_shape[1];
            // Default in_channels = 3 (RGB). Gemma4v's vision_config has
            // no `num_channels` — llama.cpp hardcodes `channels=3` via
            // `build_inp_raw()`'s default arg (clip.cpp:518).
            const IN_CHANNELS: usize = 3;
            if hwc % IN_CHANNELS == 0 {
                let patch_area = hwc / IN_CHANNELS;
                let patch_size = (patch_area as f64).sqrt() as usize;
                if patch_size * patch_size == patch_area && patch_size > 0 {
                    emit_shape = vec![n_embd, IN_CHANNELS, patch_size, patch_size];
                } else {
                    warn!(
                        "v.patch_embd.weight: hwc={} not a square*3, skipping reshape",
                        hwc
                    );
                }
            }
        }

        // Vision tensors are preserved as F16 — no repacking needed, data passes through
        let total_elements: usize = emit_shape.iter().product();
        let repacked_size = if qt.quant_info.preserved || ggml_type == GGML_TYPE_F16 || ggml_type == GGML_TYPE_F32 {
            // F16→F32 upcast: each input element becomes 4 bytes instead
            // of 2, so the on-disk size for an F16 source promoted to
            // F32 is 2× the input byte count.  This also covers F32→F32
            // passthrough (data.len() == total_elements * 4 already).
            if ggml_type == GGML_TYPE_F32 && qt.data.len() == total_elements * 2 {
                total_elements * 4
            } else {
                qt.data.len()
            }
        } else {
            ggml_tensor_size(total_elements, ggml_type)
        };

        tensor_data_offset = align_up(tensor_data_offset, ALIGNMENT);

        tensor_infos.push(TensorWriteInfo {
            gguf_name,
            shape: emit_shape,
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
    debug!("Mmproj tensor data block starts at offset {}", data_block_start);

    // --- Pass 2: Repack and write one tensor at a time ---
    for (i, name) in vision_tensor_names.iter().enumerate() {
        let qt = &model.tensors[*name];
        let info = &tensor_infos[i];

        // Pad to alignment
        let current = w.stream_position()?;
        let target = data_block_start + info.data_offset;
        if current < target {
            w.write_all(&vec![0u8; (target - current) as usize])?;
        }

        // Repack and write (vision tensors are typically preserved F16, so this
        // is usually a passthrough copy)
        let mut data = repack_to_ggml_blocks(qt, info.ggml_type).map_err(|e| {
            BackendError::WriteFailed {
                reason: format!("Failed to repack vision tensor '{}': {}", name, e),
            }
        })?;

        // ADR-005 Phase 2c iter-116g: if this tensor is the gemma4v
        // patch-embd Conv2d weight (we know because Pass 1 promoted its
        // emit shape from 2-D to 4-D), perform the HWC→CHW byte transpose
        // that the upstream `modify_tensors` does via `permute(0,3,1,2)`.
        // The reshape from `[n_embd, hwc]` to `[n_embd, h, w, c]` is a
        // no-op on row-major bytes; only the permute moves elements.
        if info.gguf_name == "v.patch_embd.weight" && info.shape.len() == 4 {
            let n_embd = info.shape[0];
            let c = info.shape[1];
            let h = info.shape[2];
            let w_dim = info.shape[3];
            let elem_size = match info.ggml_type {
                GGML_TYPE_F32 => 4,
                GGML_TYPE_F16 => 2,
                _ => 0,
            };
            if elem_size > 0 && data.len() == n_embd * c * h * w_dim * elem_size {
                data = transpose_patch_embd_hwc_to_chw(
                    &data, n_embd, h, w_dim, c, elem_size,
                );
            } else {
                warn!(
                    "v.patch_embd.weight: cannot transpose (ggml_type={} data_len={} elems_expected={})",
                    info.ggml_type,
                    data.len(),
                    n_embd * c * h * w_dim * elem_size
                );
            }
        }

        w.write_all(&data)?;
        pb.inc(1);
    }

    w.flush()?;
    pb.finish_with_message("Mmproj GGUF tensors written");

    let file_size = std::fs::metadata(&mmproj_path)?.len();
    info!(
        "Mmproj GGUF file written: {} ({:.2} MB)",
        mmproj_path.display(),
        file_size as f64 / (1024.0 * 1024.0)
    );

    Ok(OutputFile {
        filename: mmproj_filename,
        size_bytes: file_size,
    })
}

/// HWC→CHW byte-level permute for the gemma4v patch-embd Conv2d weight.
///
/// Source layout (row-major contiguous): `[n_embd, h, w, c]` —
/// `src[((o*h + i)*w + j)*c + k]` is the (output=o, height=i, width=j,
/// channel=k) element.
///
/// Target layout: `[n_embd, c, h, w]` —
/// `dst[((o*c + k)*h + i)*w + j]`.
///
/// Operates on `elem_size`-byte elements (F32=4, F16=2). PyTorch's
/// `permute(0, 3, 1, 2).contiguous()` (see `convert_hf_to_gguf.py:7877`)
/// produces this exact layout.
fn transpose_patch_embd_hwc_to_chw(
    src: &[u8],
    n_embd: usize,
    h: usize,
    w: usize,
    c: usize,
    elem_size: usize,
) -> Vec<u8> {
    let total = n_embd * c * h * w * elem_size;
    let mut dst = vec![0u8; total];
    for o in 0..n_embd {
        for k in 0..c {
            for i in 0..h {
                for j in 0..w {
                    let src_idx = ((o * h + i) * w + j) * c + k;
                    let dst_idx = ((o * c + k) * h + i) * w + j;
                    let s = src_idx * elem_size;
                    let d = dst_idx * elem_size;
                    dst[d..d + elem_size].copy_from_slice(&src[s..s + elem_size]);
                }
            }
        }
    }
    dst
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
        "Q3_K_S" | "Q3_K_M" | "Q3_K_L" | "Q3_K" => Some(GGML_TYPE_Q3_K),
        "Q4_K_S" | "Q4_K_M" | "Q4_K" => Some(GGML_TYPE_Q4_K),
        "Q5_K_S" | "Q5_K_M" | "Q5_K" => Some(GGML_TYPE_Q5_K),
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

/// GGML Q6_K super-block: 256 elements, 210 bytes.
/// Layout: ql[128] + qh[64] + scales[16] + d[2 bytes f16] = 210 bytes.
const QK6_K: usize = 256;
/// GGML Q6_K block size in bytes.
const BLOCK_Q6_K_BYTES: usize = 128 + 64 + 16 + 2; // 210

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

    // Preserved or f16 tensors: data is already raw element bytes
    if info.preserved || info.bits == 16 || info.method == "f16" {
        // If target is F32 but data is F16, convert
        if ggml_type == GGML_TYPE_F32 && qt.data.len() == qt.shape.iter().product::<usize>() * 2 {
            // F16→F32 conversion
            let mut f32_data = Vec::with_capacity(qt.data.len() * 2);
            for chunk in qt.data.chunks_exact(2) {
                let f16_val = half::f16::from_le_bytes([chunk[0], chunk[1]]);
                let f32_val: f32 = f16_val.to_f32();
                f32_data.extend_from_slice(&f32_val.to_le_bytes());
            }
            return Ok(f32_data);
        }
        return Ok(qt.data.clone());
    }

    // ADR-014 P11-prereq Iter A (2026-04-27): codec-direct fast-path. Tensors
    // produced by `KQuantCodecQuantizer` / `VariantKQuantizer` carry
    // `method == METHOD_K_QUANT_CODEC_DIRECT`; their `data` is ALREADY in the
    // target GGUF block format (see `k_quant_codec::quantize_tensor_2d_to_bytes`).
    // Return unchanged with NO warn (this is the intended path now, not a
    // code smell — pre-Iter A this fell through to the warn-and-return-raw
    // branch below by accident). Validate block-size alignment so a future
    // codec or shape regression surfaces here as a typed `Err` rather than a
    // silent half-block corruption downstream.
    if info.method == crate::quantize::k_quant_codec_quantizer::METHOD_K_QUANT_CODEC_DIRECT {
        let expected_block_size = match ggml_type {
            GGML_TYPE_Q4_K => crate::quantize::k_quant::BLOCK_Q4_K_SIZE,
            GGML_TYPE_Q5_K => crate::quantize::k_quant::BLOCK_Q5_K_SIZE,
            GGML_TYPE_Q6_K => BLOCK_Q6_K_BYTES,
            GGML_TYPE_Q3_K => crate::quantize::k_quant::BLOCK_Q3_K_SIZE,
            GGML_TYPE_Q4_0 => BLOCK_Q4_0_BYTES,
            GGML_TYPE_Q8_0 => BLOCK_Q8_0_BYTES,
            _ => {
                return Err(BackendError::WriteFailed {
                    reason: format!(
                        "codec-direct tensor '{}' has unsupported target ggml_type {} \
                         (no known block size); refusing to write potentially-corrupt bytes",
                        qt.name, ggml_type
                    ),
                });
            }
        };
        if qt.data.len() % expected_block_size != 0 {
            return Err(BackendError::WriteFailed {
                reason: format!(
                    "codec-direct tensor '{}': data length {} is not a multiple of \
                     block size {} for ggml_type {} — bytes are misaligned and would \
                     corrupt the output",
                    qt.name,
                    qt.data.len(),
                    expected_block_size,
                    ggml_type
                ),
            });
        }
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

    // 6-bit quantized tensors need Q6_K block format
    if ggml_type == GGML_TYPE_Q6_K {
        return repack_q6_k(qt, scales_bytes, total_elements);
    }

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

/// Repack hf2q internal 6-bit quantized data into Q6_K super-block format.
///
/// Q6_K block: 256 elements, 210 bytes.
/// Layout: ql[128] (lower 4 bits) + qh[64] (upper 2 bits) + scales[16] (int8) + d (f16)
///
/// Steps:
/// 1. Decode f16 scales from quant_info.scales
/// 2. Read signed i8 values from qt.data (6-bit values stored as 1 byte each)
/// 3. Reconstruct approximate f32 values: val = signed_q * f16_scale
/// 4. Re-quantize each 256-element super-block into Q6_K format following
///    the exact algorithm from llama.cpp's quantize_row_q6_K_ref.
///
/// The inner sub-block loops (`for ib in 0..16`, `for i in sub_start..sub_end`)
/// are deliberate line-for-line mirrors of the C reference — rewriting them
/// with iterator adapters (`chunks_exact(16)` + `.iter().enumerate()`) breaks
/// the visual parallel that makes diff-against-upstream tractable. Chesterton's
/// fence: keep the shape, silence the lint.
#[allow(clippy::needless_range_loop)]
fn repack_q6_k(
    qt: &crate::ir::QuantizedTensor,
    scales_bytes: &[u8],
    total_elements: usize,
) -> Result<Vec<u8>, BackendError> {
    let info = &qt.quant_info;
    let group_size = if info.group_size == 0 { 64 } else { info.group_size };

    // Decode f16 scales: 2 bytes each
    let scales_f32: Vec<f32> = scales_bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect();

    // Read signed i8 values from qt.data (6-bit: 1 byte per element, stored as u8 cast of i8)
    let mut signed_values: Vec<i8> = Vec::with_capacity(total_elements);
    for &byte in &qt.data {
        signed_values.push(byte as i8);
    }
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

    // If scales didn't cover all elements, pad with zeros
    while f32_values.len() < total_elements {
        f32_values.push(0.0);
    }

    // Re-quantize into Q6_K super-blocks of 256 elements each.
    // Follows llama.cpp quantize_row_q6_K_ref exactly.
    let num_blocks = total_elements.div_ceil(QK6_K);
    let mut output = Vec::with_capacity(num_blocks * BLOCK_Q6_K_BYTES);

    // Threshold below which a sub-block is considered all-zero
    const GROUP_MAX_EPS: f32 = 1e-15;

    for block_idx in 0..num_blocks {
        let start = block_idx * QK6_K;
        let end = (start + QK6_K).min(total_elements);

        // Get block values, padding with zeros if needed
        let mut x = [0.0f32; QK6_K];
        for (i, idx) in (start..end).enumerate() {
            x[i] = f32_values[idx];
        }

        // Step 1: Compute per-sub-block scales (16 sub-blocks of 16 elements)
        let mut sub_scales = [0.0f32; 16];
        let mut l_values = [0i8; QK6_K]; // quantized values in [-32, 31] range

        let mut max_scale: f32 = 0.0;
        let mut max_abs_scale: f32 = 0.0;

        for ib in 0..16 {
            let sub_start = ib * 16;
            let sub_end = sub_start + 16;

            // Find the element with maximum absolute value in this sub-block
            let mut amax: f32 = 0.0;
            let mut max_val: f32 = 0.0;
            for i in sub_start..sub_end {
                let ax = x[i].abs();
                if ax > amax {
                    amax = ax;
                    max_val = x[i];
                }
            }

            if amax < GROUP_MAX_EPS {
                sub_scales[ib] = 0.0;
                for i in sub_start..sub_end {
                    l_values[i] = 0;
                }
                continue;
            }

            // Compute initial scale using make_qx_quants logic (rmse_type=1):
            // iscale = -nmax / max, then weighted least squares refinement.
            // For re-quantization from already-quantized data, the simple approach
            // (scale = amax / 32) matches well. But to exactly match llama.cpp's
            // quantize_row_q6_K_ref which uses make_qx_quants(16, 32, x, L, 1, NULL),
            // we implement the rmse_type=1 weighted optimization.
            let iscale = -32.0 / max_val;
            let mut sumlx: f64 = 0.0;
            let mut suml2: f64 = 0.0;
            for i in sub_start..sub_end {
                let l = (iscale * x[i]).round() as i32;
                let l = l.clamp(-32, 31);
                l_values[i] = l as i8;
                let lu = (l + 32) as i64; // unsigned offset form for weighted computation
                // rmse_type=1: weight = x[i]^2
                let w = (x[i] as f64) * (x[i] as f64);
                sumlx += w * (x[i] as f64) * ((lu - 32) as f64);
                suml2 += w * ((lu - 32) as f64) * ((lu - 32) as f64);
            }
            let scale = if suml2 > 0.0 { (sumlx / suml2) as f32 } else { 1.0 / iscale };

            sub_scales[ib] = scale;

            let abs_scale = scale.abs();
            if abs_scale > max_abs_scale {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }

        // Step 2: Handle all-zero block
        if max_abs_scale < GROUP_MAX_EPS {
            output.extend_from_slice(&[0u8; BLOCK_Q6_K_BYTES]);
            continue;
        }

        // Step 3: Compute super-block scale d and quantize sub-block scales to int8
        let iscale = -128.0f32 / max_scale;
        let d = 1.0f32 / iscale;
        let d_f16 = half::f16::from_f32(d);

        let mut int8_scales = [0i8; 16];
        for ib in 0..16 {
            let qs = (iscale * sub_scales[ib]).round() as i32;
            int8_scales[ib] = qs.clamp(-128, 127) as i8;
        }

        // Step 4: Re-quantize each element using the final d and int8 scales
        // to get unsigned L values in [0, 63]
        let mut big_l = [0u8; QK6_K];
        for j in 0..16 {
            let scale_f = half::f16::from_bits(d_f16.to_bits()).to_f32() * (int8_scales[j] as f32);
            if scale_f == 0.0 {
                // Leave L values as 0 (which means quantized value = 0 - 32 = -32, dequant = 0)
                for ii in 0..16 {
                    big_l[16 * j + ii] = 32; // 0 + 32 unsigned offset
                }
                continue;
            }
            for ii in 0..16 {
                let l = (x[16 * j + ii] / scale_f).round() as i32;
                let l = l.clamp(-32, 31);
                big_l[16 * j + ii] = (l + 32) as u8;
            }
        }

        // Step 5: Pack into ql[128] + qh[64] + scales[16] + d[2]
        // Following llama.cpp's exact packing from quantize_row_q6_K_ref:
        // Process in two halves of 128 elements (j=0 and j=128)
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];

        let mut ql_idx = 0usize;
        let mut qh_idx = 0usize;
        for j in (0..QK6_K).step_by(128) {
            for l in 0..32 {
                let q1 = big_l[j + l] & 0xF;
                let q2 = big_l[j + l + 32] & 0xF;
                let q3 = big_l[j + l + 64] & 0xF;
                let q4 = big_l[j + l + 96] & 0xF;
                ql[ql_idx + l] = q1 | (q3 << 4);
                ql[ql_idx + l + 32] = q2 | (q4 << 4);
                qh[qh_idx + l] = (big_l[j + l] >> 4)
                    | ((big_l[j + l + 32] >> 4) << 2)
                    | ((big_l[j + l + 64] >> 4) << 4)
                    | ((big_l[j + l + 96] >> 4) << 6);
            }
            ql_idx += 64;
            qh_idx += 32;
        }

        // Write block: ql[128] + qh[64] + scales[16] + d[2]
        output.extend_from_slice(&ql);
        output.extend_from_slice(&qh);
        for &s in &int8_scales {
            output.push(s as u8);
        }
        output.extend_from_slice(&d_f16.to_le_bytes());
    }

    debug!(
        "Repacked '{}': {} elements -> {} Q6_K blocks ({} bytes, was {} bytes)",
        qt.name,
        total_elements,
        num_blocks,
        output.len(),
        qt.data.len()
    );

    Ok(output)
}

/// Human-readable name for a GGML type code (for debug logging).
fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1",
        6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0", 9 => "Q8_1",
        10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K",
        14 => "Q6_K", 15 => "Q8_K",
        _ => "unknown",
    }
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
        GGML_TYPE_Q6_K => {
            let n_blocks = total_elements.div_ceil(QK6_K);
            n_blocks * BLOCK_Q6_K_BYTES
        }
        // For any other type, we cannot compute the size; caller should ensure
        // we never reach here for types we don't support.
        _ => total_elements * 2, // fallback to f16 sizing
    }
}

/// Generate synthetic tensors that aren't in the safetensors but are required by llama.cpp.
///
/// For Gemma4: generates `rope_freqs.weight` — frequency scaling factors for
/// partial RoPE on the first full-attention layer. llama.cpp loads it per-layer:
///   layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), ...)
/// The first non-SWA layer gets the tensor with flag=0 (required), subsequent ones
/// get TENSOR_DUPLICATED (shared data). So we only need one tensor named for the
/// first full-attention layer index.
fn generate_synthetic_tensors(
    metadata: &crate::ir::ModelMetadata,
) -> Vec<(String, Vec<u8>, Vec<usize>, u32)> {
    let mut result = Vec::new();
    let arch = &metadata.model_type;

    if arch == "gemma4" {
        // Extract partial_rotary_factor and global_head_dim from raw_config
        let tc = metadata.raw_config.get("text_config").cloned().unwrap_or_default();
        let global_head_dim = tc.get("global_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;
        let partial_rotary_factor = tc.get("rope_parameters")
            .and_then(|rp| rp.get("full_attention"))
            .and_then(|fa| fa.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25);

        // Find the first full-attention layer index from layer_types
        let first_full_layer = tc.get("layer_types")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                arr.iter().position(|v| v.as_str() == Some("full_attention"))
            });

        if first_full_layer.is_some() {
            // n_rot = floor(global_head_dim * partial_rotary_factor / 2)
            // n_unrot = global_head_dim / 2 - n_rot
            let half_dim = global_head_dim / 2;
            let n_rot = (global_head_dim as f64 * partial_rotary_factor / 2.0).floor() as usize;
            let n_unrot = half_dim - n_rot;

            // Build frequency factors: [1.0]*n_rot + [1e30]*n_unrot
            let mut values: Vec<f32> = Vec::with_capacity(half_dim);
            values.extend(std::iter::repeat(1.0f32).take(n_rot));
            values.extend(std::iter::repeat(1e30f32).take(n_unrot));

            // Serialize as raw f32 bytes
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

            // llama.cpp expects global "rope_freqs.weight" (no blk.N prefix).
            // The tensor name pattern has no %d placeholder — it's shared across layers.
            result.push((
                "rope_freqs.weight".to_string(),
                data,
                vec![half_dim],
                GGML_TYPE_F32,
            ));
        }
    }

    result
}

/// Map IR quantization metadata to a GGML type code.
///
/// The GGML type must match the actual block format written to disk. Since hf2q's
/// quantizer produces simple per-group scales + packed values (not K-quant super-block
/// format), we can only write Q4_0 / Q8_0 / F16 / F32 block formats. Apex K-quant
/// type overrides are mapped back to the corresponding simple type.
///
/// **Codec-direct fast-path** (ADR-014 P11-prereq Iter A, 2026-04-27): when
/// `info.method == METHOD_K_QUANT_CODEC_DIRECT` the tensor data is already in
/// the target K-quant block format and the IR has set `bits = 0` (sentinel).
/// Route via `info.ggml_type` BEFORE the bits-fallback fires; otherwise the
/// `bits = 0` sentinel hits the `_ =>` arm and silently emits an `F16`-header
/// GGUF atop K-quant bytes (the LIVE `--quant q4_k_m` malformed-GGUF defect
/// recorded in the ADR-014 audit revision dated 2026-04-27).
fn quant_info_to_ggml_type(info: &TensorQuantInfo) -> u32 {
    if info.preserved {
        return GGML_TYPE_F16;
    }

    // ADR-014 P11-prereq Iter A: codec-direct fast-path. `KQuantCodecQuantizer`
    // (`src/quantize/k_quant_codec_quantizer.rs:114,222`) and `VariantKQuantizer`
    // (`src/quantize/variant_quantizer.rs:42,225`) both set
    // `method = METHOD_K_QUANT_CODEC_DIRECT` + `bits = 0` + `ggml_type = Some(<canonical>)`
    // when their `data` is already in target GGUF block format. Honor that
    // contract here so the type-code in the GGUF header matches the on-disk
    // bytes; on unknown name return the bits-fallback's `F16`-shaped error
    // is wrong because the bytes are NOT F16 — fail loud via warn, then route
    // to F16 only because the fallback contract is that we must always return
    // *some* type code from this fn signature. (A future iter can change
    // `quant_info_to_ggml_type` to `Result<u32, _>`.)
    if info.method == crate::quantize::k_quant_codec_quantizer::METHOD_K_QUANT_CODEC_DIRECT {
        if let Some(ref type_name) = info.ggml_type {
            if let Some(code) = ggml_type_from_name(type_name) {
                return code;
            }
            warn!(
                "codec-direct sentinel set on tensor with unknown ggml_type '{}'; \
                 the on-disk bytes are NOT F16 — this is a quantizer bug",
                type_name
            );
            return GGML_TYPE_F16;
        }
        warn!(
            "codec-direct sentinel set without ggml_type field; \
             the on-disk bytes are NOT F16 — this is a quantizer bug"
        );
        return GGML_TYPE_F16;
    }

    // ADR-014 P11-prereq Iter C (2026-04-27): the Q2_K/Q3_K/Q4_K/Q5_K
    // "fall through to bits-based mapping" swallow that lived here was
    // removed.  Audit (Iter C, file fence: `quant_info_to_ggml_type`):
    //   * The only production writers of `ggml_type = Some("Q[2-5]_K"…)`
    //     are `KQuantCodecQuantizer` (`k_quant_codec_quantizer.rs:229`)
    //     and `VariantKQuantizer` (`variant_quantizer.rs:231`) — both
    //     unconditionally set `method = METHOD_K_QUANT_CODEC_DIRECT`,
    //     which Iter A's fast-path above already routes via
    //     `ggml_type_from_name` BEFORE this block fires.  After Iter C
    //     wires DWQ through `DwqKQuantizer`, the same property holds
    //     for every DWQ tensor that targets a K-quant block format
    //     (`DwqKQuantizer` constructs a `KQuantCodecQuantizer` per-call;
    //     see `dwq_k_quantizer.rs:280`).
    //   * The legacy `--quant apex` writer that previously asked this
    //     branch to "map K-quant override → bits-fallback Q4_0" was
    //     deleted alongside the variant in Decision 12.  No remaining
    //     production caller asks `Q2_K`/`Q3_K`/`Q4_K`/`Q5_K` to fall
    //     through to a different block type.
    //   * The `bits = 0` sentinel that codec-direct quantizers set is
    //     unsafe input for the bits-based table below (it returns F16
    //     for any unknown bit width) — preserving the swallow would
    //     silently re-introduce the malformed-GGUF defect Iter A closed.
    //
    // Q6_K override (no codec-direct sentinel) is still honored
    // directly — it survives because `apex` was the *historical* writer
    // of bare-`Q6_K`-named overrides, and the codec-direct fast-path
    // above subsumes it for codec-direct callers; this branch keeps the
    // bare-name path live for the remaining test surface
    // (`gguf.rs::tests::test_ggml_type_override_in_quant_info` at
    // `src/backends/gguf.rs:3473-3477`) without re-introducing the
    // swallow on `Q[2-5]_K`.
    if let Some(ref type_name) = info.ggml_type {
        let upper = type_name.trim().to_uppercase();
        if upper.starts_with("Q6_K") {
            return GGML_TYPE_Q6_K;
        }
        if upper.starts_with("Q2_K")
            || upper.starts_with("Q3_K")
            || upper.starts_with("Q4_K")
            || upper.starts_with("Q5_K")
        {
            // Reachable iff some non-codec-direct path sets the K-quant
            // ggml_type without the codec-direct sentinel.  No such
            // path exists in production today (audit above); this is a
            // defence-in-depth guard against a future regression.
            // Surface loudly — bytes are NOT necessarily K-quant block
            // bytes here, so we cannot return the K-quant type code
            // safely; F16 is the only response that cannot mis-frame
            // the bytes (the same fall-back contract Iter A keeps for
            // unknown codec-direct ggml_type names).
            debug_assert!(
                false,
                "K-quant ggml_type '{}' set without METHOD_K_QUANT_CODEC_DIRECT \
                 sentinel — non-codec-direct K-quant path is dead code; \
                 if you hit this, file a bug (ADR-014 P11-prereq Iter C)",
                type_name
            );
            warn!(
                "K-quant ggml_type '{}' set without codec-direct sentinel \
                 (method = '{}'); cannot route to K-quant type code without \
                 confirming on-disk byte format — falling back to F16",
                type_name, info.method
            );
            return GGML_TYPE_F16;
        }
        if let Some(code) = ggml_type_from_name(type_name) {
            return code;
        }
        warn!(
            "Unknown GGML type name '{}'; falling back to bits-based mapping",
            type_name
        );
    }

    // Generic bits-based mapping — only types we can produce proper block format for
    match info.bits {
        16 => GGML_TYPE_F16,
        8 => GGML_TYPE_Q8_0,
        6 => GGML_TYPE_Q6_K,
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
/// Qwen3.5-family linear-attention (Gated DeltaNet) layer-map entries.
///
/// MTP wrapper-tensor mappings for qwen35 / qwen35moe.
///
/// After `models::qwen35::rename_mtp_tensors_to_layer_form` rewrites HF
/// `mtp.fc.weight` etc. to `model.layers.{n_layer}.eh_proj.weight` etc.,
/// the standard layer mapper sees them and translates to the
/// `blk.{N}.nextn.<suffix>` form llama-arch.cpp:447-450 expects.
///
/// Each entry cites the corresponding `LLM_TENSOR_NEXTN_*` line.
fn qwen35_nextn_wrapper_layer_map() -> &'static [(&'static str, &'static str)] {
    &[
        // llama-arch.cpp:447 LLM_TENSOR_NEXTN_EH_PROJ → "blk.%d.nextn.eh_proj"
        // (rename source: mtp.fc.weight per convert_hf_to_gguf.py:10536)
        ("eh_proj.weight", "nextn.eh_proj.weight"),
        // llama-arch.cpp:449 LLM_TENSOR_NEXTN_ENORM → "blk.%d.nextn.enorm"
        // (rename source: mtp.pre_fc_norm_embedding.weight per :10537)
        ("enorm.weight", "nextn.enorm.weight"),
        // llama-arch.cpp:450 LLM_TENSOR_NEXTN_HNORM → "blk.%d.nextn.hnorm"
        // (rename source: mtp.pre_fc_norm_hidden.weight per :10538)
        ("hnorm.weight", "nextn.hnorm.weight"),
        // llama-arch.cpp:452 LLM_TENSOR_NEXTN_SHARED_HEAD_NORM → "blk.%d.nextn.shared_head_norm"
        // (rename source: mtp.norm.weight per :10539; py uses "shared_head.norm")
        ("shared_head.norm.weight", "nextn.shared_head_norm.weight"),
        ("shared_head_norm.weight", "nextn.shared_head_norm.weight"),
        // llama-arch.cpp:451 LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD → "blk.%d.nextn.shared_head_head"
        // (no Qwen3.5 HF source observed yet; mapped for forward compatibility)
        ("shared_head.head.weight", "nextn.shared_head_head.weight"),
        // llama-arch.cpp:448 LLM_TENSOR_NEXTN_EMBED_TOKENS → "blk.%d.nextn.embed_tokens"
        // (some HF variants ship the wrapper-level embed_tokens at mtp.embed_tokens.weight)
        ("embed_tokens.weight", "nextn.embed_tokens.weight"),
    ]
}

/// Shared between the qwen35 (dense) and qwen35moe (MoE) arches because the
/// hybrid 3:1 DeltaNet:Full attention pattern is identical between variants
/// (only the FFN differs). Citations transcribed from llama-arch.cpp via
/// `src/models/qwen35/dense.rs::map_linear_attn_suffix`.
fn qwen35_linear_attn_layer_map() -> &'static [(&'static str, &'static str)] {
    &[
        // llama-arch.cpp:367 LLM_TENSOR_ATTN_POST_NORM → "blk.%d.post_attention_norm"
        ("post_attention_layernorm.weight", "post_attention_norm.weight"),
        // llama-arch.cpp:382 LLM_TENSOR_ATTN_QKV → "blk.%d.attn_qkv"
        // (in_proj_qkvz is split → in_proj_qkv + in_proj_z by Phase 1.7)
        ("linear_attn.in_proj_qkv.weight", "attn_qkv.weight"),
        // llama-arch.cpp:370 LLM_TENSOR_ATTN_GATE → "blk.%d.attn_gate"
        ("linear_attn.in_proj_z.weight", "attn_gate.weight"),
        // llama-arch.cpp:400 LLM_TENSOR_SSM_ALPHA → "blk.%d.ssm_alpha"
        ("linear_attn.in_proj_a.weight", "ssm_alpha.weight"),
        // llama-arch.cpp:416 LLM_TENSOR_SSM_BETA → "blk.%d.ssm_beta"
        ("linear_attn.in_proj_b.weight", "ssm_beta.weight"),
        // llama-arch.cpp:402 LLM_TENSOR_SSM_OUT → "blk.%d.ssm_out"
        ("linear_attn.out_proj.weight", "ssm_out.weight"),
        // llama-arch.cpp:395 LLM_TENSOR_SSM_A_NOSCAN → "blk.%d.ssm_a"
        // (no .weight suffix — the ssm_a tensor is just `blk.N.ssm_a`)
        ("linear_attn.A_log", "ssm_a"),
        // llama-arch.cpp:397 LLM_TENSOR_SSM_DT → "blk.%d.ssm_dt"
        // py:4791 renames .dt_bias → .dt_proj.bias before mapping; we accept
        // either form so the convert pipeline stays robust to that rename.
        ("linear_attn.dt_bias", "ssm_dt.bias"),
        ("linear_attn.dt_proj.bias", "ssm_dt.bias"),
        ("linear_attn.dt_proj.weight", "ssm_dt.weight"),
        // llama-arch.cpp:396 LLM_TENSOR_SSM_CONV1D → "blk.%d.ssm_conv1d"
        ("linear_attn.conv1d.weight", "ssm_conv1d.weight"),
        ("linear_attn.conv1d.bias", "ssm_conv1d.bias"),
        // llama-arch.cpp:401 LLM_TENSOR_SSM_NORM → "blk.%d.ssm_norm"
        ("linear_attn.norm.weight", "ssm_norm.weight"),
    ]
}

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
        // Qwen3.5 family (dense + MoE): post_attention_layernorm is `attn_post_norm`
        // in the model struct (llama-model.cpp:7628/7565), which maps to
        // LLM_TENSOR_ATTN_POST_NORM → "blk.%d.post_attention_norm" (llama-arch.cpp:367).
        // There is NO separate ffn_norm tensor — the post-attn norm gates the FFN.
        // ADR-012 Decision 8: post_attention_layernorm verdict.
        //
        // Plus the qwen35-specific linear-attention (Gated DeltaNet) tensor
        // family. Mappings transcribed from src/models/qwen35/dense.rs::
        // map_linear_attn_suffix (which had the canonical citations to
        // llama-arch.cpp but was never wired into the GGUF backend's
        // layer_map). Each entry cites the matching llama-arch.cpp line.
        // Note: `in_proj_qkvz.weight` is split into `in_proj_qkv` +
        // `in_proj_z` upstream by `transform_in_proj_qkvz` in Phase 1.7,
        // so the layer-map only sees the post-split names.
        "qwen35" => {
            map.extend_from_slice(&qwen35_linear_attn_layer_map());
            map.extend_from_slice(qwen35_nextn_wrapper_layer_map());
        }
        // ADR-012 Decision 11 + P5: qwen35moe adds the MoE-specific surface:
        // router (mlp.gate), per-expert merged stacks (mlp.experts.* post-merge
        // named mlp.experts.{proj}.weight by merge_moe_experts_in_place), and
        // shared-expert + shared-expert-gate. Each entry cites the
        // llama-arch.cpp / src/models/qwen35/moe.rs line that pinned it.
        // Previously this branch only mapped post_attention_layernorm,
        // silently passing MoE tensors through as `blk.N.mlp.*.weight`
        // (wrong GGUF names; llama.cpp loader would reject the file).
        // P11 round-trip gate caught it — fix lands here.
        //
        // qwen35moe ALSO has the qwen35 linear-attention family (the hybrid
        // 3:1 DeltaNet:Full pattern is identical between dense + MoE
        // variants); only the FFN differs. Same mapping table is reused
        // for both arches.
        "qwen35moe" => {
            map.extend_from_slice(&qwen35_linear_attn_layer_map());
            map.extend_from_slice(qwen35_nextn_wrapper_layer_map());
            map.extend_from_slice(&[
                // llama-arch.cpp:393 LLM_TENSOR_FFN_GATE_INP → "blk.%d.ffn_gate_inp"
                // src/models/qwen35/moe.rs:83 — MoE router.
                ("mlp.gate.weight", "ffn_gate_inp.weight"),
                // llama-arch.cpp:394 LLM_TENSOR_FFN_GATE_INP_SHEXP → "blk.%d.ffn_gate_inp_shexp"
                // src/models/qwen35/moe.rs:90 — shared-expert scalar gate.
                ("mlp.shared_expert_gate.weight", "ffn_gate_inp_shexp.weight"),
                // Shared-expert per-projection tensors (src/models/qwen35/moe.rs:138-140).
                ("mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp.weight"),
                ("mlp.shared_expert.up_proj.weight", "ffn_up_shexp.weight"),
                ("mlp.shared_expert.down_proj.weight", "ffn_down_shexp.weight"),
                // Merged expert tensors — post-merge names produced by
                // merge_moe_experts_in_place at src/models/qwen35/moe.rs:494-497.
                // Shape [N_experts, out_features, in_features].
                ("mlp.experts.gate_proj.weight", "ffn_gate_exps.weight"),
                ("mlp.experts.up_proj.weight", "ffn_up_exps.weight"),
                ("mlp.experts.down_proj.weight", "ffn_down_exps.weight"),
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

/// Resolve the `blk.mtp{idx}.…` placeholder emitted by [`hf_name_to_gguf`]
/// for qwen35/qwen35moe MTP tensors into the real block index
/// `blk.{num_hidden_layers + idx}.…` (ADR-012 Decision 11 + 19).
///
/// Returns the input unchanged when no MTP placeholder is present, so this
/// is safe to call for every tensor in the production write path.
///
/// # Why not inline this in `hf_name_to_gguf`
///
/// `hf_name_to_gguf(hf_name, arch)` doesn't have `num_hidden_layers` in
/// scope and threading it through every test caller would balloon the diff.
/// Instead the placeholder is emitted there, and resolved here from the
/// `ModelMetadata` the production caller already holds.
pub(crate) fn resolve_mtp_block_index(gguf_name: &str, num_hidden_layers: u32) -> String {
    if let Some(rest) = gguf_name.strip_prefix("blk.mtp") {
        if let Some(dot_pos) = rest.find('.') {
            let idx_str = &rest[..dot_pos];
            if let Ok(idx) = idx_str.parse::<u32>() {
                let suffix = &rest[dot_pos + 1..];
                return format!("blk.{}.{}", num_hidden_layers + idx, suffix);
            }
        }
    }
    gguf_name.to_string()
}

/// Convert a HuggingFace tensor name to its GGUF equivalent.
///
/// `arch` is the model architecture string (e.g. "llama", "gemma4", "qwen3")
/// from `model.metadata.model_type`. Different architectures use different
/// GGUF names for the same HF tensor suffixes.
///
/// `num_hidden_layers` is required for MTP tensor resolution (ADR-012
/// Decision 11). Non-MTP callers may pass 0.
/// Convenience wrapper for non-MTP callers (tests, one-off probes). Uses
/// `num_hidden_layers = 0` — MTP paths hit the `mtp.layers.` branch which
/// requires the real value, so this is only correct for non-MTP inputs.
#[cfg(test)]
fn hf_name_to_gguf_no_mtp(hf_name: &str, arch: &str) -> String {
    hf_name_to_gguf(hf_name, arch, 0)
}

fn hf_name_to_gguf(hf_name: &str, arch: &str, num_hidden_layers: u32) -> String {
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
        // ADR-005 Phase 2c iter-116f: gemma4v projector base name must be
        // `mm.input_projection.weight` (TN_MM_INP_PROJ at
        // /opt/llama.cpp/tools/mtmd/clip-impl.h:110), not `mm.0.weight`.
        // PROJECTOR_TYPE_GEMMA4V at clip.cpp:1937 hard-requires this exact
        // name via `get_tensor(TN_MM_INP_PROJ)` (no fallback). See also
        // /opt/llama.cpp/gguf-py/gguf/tensor_mapping.py:135 +
        // /opt/llama.cpp/gguf-py/gguf/constants.py:1226 (V_MM_INP_PROJ ->
        // "mm.input_projection"). Note: hf2q's INFERENCE-side reader
        // (`mmproj_weights::mm_0_weight`) still looks up `mm.0.weight` —
        // self-load drift is tracked separately; this iter's gate is
        // llama-mtmd-cli's CLIP loader (`HF2Q_LLAMA_MMPROJ_COMPAT_MODEL_LOAD`).
        ("model.embed_vision.embedding_projection.weight", "mm.input_projection.weight"),
        // Gemma4ClippableLinear scalar bounds for the multimodal projection.
        // Each is a single-f32 tensor in HF (PyTorch 0-D scalar); the
        // llama.cpp converter `unsqueeze(0)`s them to GGUF 1-D `[1]`
        // tensors. The clamp-scalar names key off the projector base
        // (clip.cpp:1941-1959 substitutes `.weight` -> `.input_min` etc on
        // the same base name), so they MUST share the `mm.input_projection`
        // prefix. Reference: `/opt/llama.cpp/tools/mtmd/clip.cpp:1935-1959`.
        ("model.embed_vision.embedding_projection.input_min", "mm.input_projection.input_min"),
        ("model.embed_vision.embedding_projection.input_max", "mm.input_projection.input_max"),
        ("model.embed_vision.embedding_projection.output_min", "mm.input_projection.output_min"),
        ("model.embed_vision.embedding_projection.output_max", "mm.input_projection.output_max"),
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
    //
    // ADR-005 Phase 2c iter 116f: vision-namespace tensor names must match
    // llama.cpp's CLIP loader exactly. Per-mapping citations below — every
    // name was cross-checked against the canonical
    // /opt/llama.cpp/gguf-py/gguf/tensor_mapping.py + constants.py table
    // (the same table the upstream `convert_hf_to_gguf.py:Gemma4VisionAudioModel`
    // uses for HF→GGUF emit) and the live get_tensor calls in
    // /opt/llama.cpp/tools/mtmd/clip.cpp:1665-1694.
    //
    // Three norm mappings differ between text-decoder Gemma 4 and
    // vision-encoder Gemma 4, because gemma4's vision uses a DIFFERENT
    // norm ordering than gemma4's text decoder:
    //   - `pre_feedforward_layernorm` is the pre-FFN norm in HF, which
    //     llama.cpp's vision loader treats as V_ENC_POST_ATTN_NORM (i.e.
    //     `ln2`, the second pre-norm in build_vit). See tensor_mapping.py:1575.
    //   - `post_attention_layernorm` is a SEPARATE ATTN-OUTPUT post-norm
    //     in gemma4 vision (V_ENC_ATTN_POST_NORM = `attn_post_norm`).
    //     See tensor_mapping.py:1630 + constants.py:1218.
    //   - `post_feedforward_layernorm` is V_ENC_FFN_POST_NORM
    //     (`ffn_post_norm`). See tensor_mapping.py:1634 + constants.py:1219.
    // The text decoder mappings (gemma4 arch in `layer_map_for_arch`) are
    // unchanged — they live in a separate codepath above.
    //
    // Vision-namespace `attn_out` (W34's iter-116e fix) is preserved per
    // TN_ATTN_OUTPUT at /opt/llama.cpp/tools/mtmd/clip-impl.h:82
    // (`"%s.blk.%d.attn_out.%s"`) — text decoder's full `attn_output`
    // form remains in `shared` at line ~1663 above, unrelated.
    let vision_layer_map: &[(&str, &str)] = &[
        // Q/K/V/O projections — clip.cpp:1665-1668, prefix=`v`.
        ("self_attn.q_proj.linear.weight", "attn_q.weight"),
        ("self_attn.k_proj.linear.weight", "attn_k.weight"),
        ("self_attn.v_proj.linear.weight", "attn_v.weight"),
        // TN_ATTN_OUTPUT short form (clip-impl.h:82 / W34 iter-116e).
        ("self_attn.o_proj.linear.weight", "attn_out.weight"),
        // Optional Q/K norms — clip.cpp:1670-1671.
        ("self_attn.q_norm.weight", "attn_q_norm.weight"),
        ("self_attn.k_norm.weight", "attn_k_norm.weight"),
        // ln1 = pre-attention norm. tensor_mapping.py:1526
        // (V_ENC_INPUT_NORM line for gemma4) + constants.py:1211.
        ("input_layernorm.weight", "ln1.weight"),
        // ln2 = pre-FFN norm. gemma4's HF `pre_feedforward_layernorm` is
        // what build_vit's `layer.ln_2_w` consumes (clip.cpp:451).
        // tensor_mapping.py:1575 maps gemma4's pre_feedforward_layernorm to
        // V_ENC_POST_ATTN_NORM = "v.blk.{bid}.ln2" (constants.py:1214).
        ("pre_feedforward_layernorm.weight", "ln2.weight"),
        // attn_post_norm = post-attention residual norm. Gemma4-only
        // (vision V_ENC_ATTN_POST_NORM). tensor_mapping.py:1630 + constants.py:1218.
        // build_vit consumes via `layer.attn_post_norm_w` at clip.cpp:439.
        ("post_attention_layernorm.weight", "attn_post_norm.weight"),
        // ffn_post_norm = post-FFN residual norm. tensor_mapping.py:1634
        // + constants.py:1219 (V_ENC_FFN_POST_NORM = `v.blk.{bid}.ffn_post_norm`).
        // build_vit consumes via `layer.ff_post_norm_w` at clip.cpp:464.
        ("post_feedforward_layernorm.weight", "ffn_post_norm.weight"),
        // FFN — gate/up/down. tensor_mapping.py:1597,1605,1625.
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

    // ADR-012 Decision 11 + 19: MTP (Multi-Token Prediction) tensor mapping for qwen35/qwen35moe.
    //
    // HF naming: "mtp.layers.{mtp_idx}.{suffix}"
    // GGUF naming: "blk.{num_hidden_layers + mtp_idx}.nextn.{gguf_suffix}"
    //
    // llama-arch.cpp:447-450 (LLM_TENSOR_NEXTN_*):
    //   LLM_TENSOR_NEXTN_EH_PROJ       → "blk.%d.nextn.eh_proj"
    //   LLM_TENSOR_NEXTN_EMBED_TOKENS  → "blk.%d.nextn.embed_tokens"
    //   LLM_TENSOR_NEXTN_ENORM         → "blk.%d.nextn.enorm"
    //   LLM_TENSOR_NEXTN_HNORM         → "blk.%d.nextn.hnorm"
    //
    // ADR-012 P11 fix (2026-04-24): resolved block index = num_hidden_layers + mtp_idx.
    // Previously used sentinel "blk.mtp{idx}.nextn.*" placeholder which P4 left as a
    // "resolve at write time" TODO. P11's MTP round-trip gate caught the stub —
    // llama.cpp's loader + ADR-013's load_mtp_weights_if_present BOTH expect the
    // resolved name `blk.{num_hidden_layers}.nextn.*`, not the placeholder.
    if (arch == "qwen35" || arch == "qwen35moe") && hf_name.starts_with("mtp.layers.") {
        if let Some(rest) = hf_name.strip_prefix("mtp.layers.") {
            if let Some(dot_pos) = rest.find('.') {
                let mtp_idx: u32 = rest[..dot_pos].parse().unwrap_or(0);
                let suffix = &rest[dot_pos + 1..];
                let nextn_suffix = match suffix {
                    "enorm.weight"        => Some("nextn.enorm.weight"),
                    "hnorm.weight"        => Some("nextn.hnorm.weight"),
                    "embed_tokens.weight" => Some("nextn.embed_tokens.weight"),
                    "eh_proj.weight"      => Some("nextn.eh_proj.weight"),
                    _ => None,
                };
                if let Some(ns) = nextn_suffix {
                    let block_idx = num_hidden_layers.saturating_add(mtp_idx);
                    return format!("blk.{}.{}", block_idx, ns);
                }
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

    // Build metadata KV pairs. The four always-present entries go in via the
    // vec![] literal; conditional ones (merges, special token ids, chat template)
    // are appended below.
    let mut kv: Vec<(String, MetaValue)> = vec![
        // Required: tokenizer model name.
        ("tokenizer.ggml.model".into(), MetaValue::String(tokenizer_model_name.clone())),
        // Token strings.
        ("tokenizer.ggml.tokens".into(), MetaValue::ArrayString(tokens)),
        // Scores.
        ("tokenizer.ggml.scores".into(), MetaValue::ArrayFloat32(scores)),
        // Token types.
        ("tokenizer.ggml.token_type".into(), MetaValue::ArrayInt32(token_types)),
    ];

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

    // Pre-tokenizer type — used by llama.cpp to select the right pre-tokenizer.
    //
    // ADR-012 P9b real-model finding (2026-04-25): we previously emitted
    // `tokenizer.ggml.pre = tokenizer.ggml.model` (e.g. "gpt2"), which
    // llama.cpp's vocab loader rejects:
    //   "unknown pre-tokenizer type: 'gpt2'"
    // The two keys serve different purposes:
    //   tokenizer.ggml.model = BPE/SP family (llama, gpt2, gemma4, ...)
    //   tokenizer.ggml.pre   = pre-tokenizer regex bucket
    //                          (qwen35, qwen2, llama-bpe, gemma4, ...)
    // The `pre` field is enumerated in /opt/llama.cpp/src/llama-vocab.cpp
    // around line 1948-2061. Here we map by GGUF arch.
    let pre_tokenizer = determine_pre_tokenizer_type(arch);
    kv.push((
        "tokenizer.ggml.pre".into(),
        MetaValue::String(pre_tokenizer),
    ));

    // Chat template — read from chat_template.jinja or tokenizer_config.json
    let chat_template_path = input_dir.join("chat_template.jinja");
    let template_str: Option<String> = if chat_template_path.exists() {
        std::fs::read_to_string(&chat_template_path).ok()
    } else {
        tokenizer_config.as_ref()
            .and_then(|c: &serde_json::Value| c.get("chat_template"))
            .and_then(|v: &serde_json::Value| v.as_str())
            .map(|s| s.to_string())
    };
    if let Some(tmpl) = template_str {
        kv.push((
            "tokenizer.chat_template".into(),
            MetaValue::String(tmpl),
        ));
    }

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

/// Determine the GGUF `tokenizer.ggml.pre` (pre-tokenizer type) per arch.
///
/// Enumeration source: `/opt/llama.cpp/src/llama-vocab.cpp` around line
/// 1948-2061. The `pre` field selects the regex-bucket pre-tokenizer that
/// llama.cpp uses to split text before BPE encoding. Different model
/// families use different regex rules.
///
/// ADR-012 P9b real-model finding 2026-04-25 — bug #10: previously we
/// emitted the BPE-family tag (e.g. "gpt2") here, which llama.cpp
/// rejects with "unknown pre-tokenizer type: 'gpt2'".
fn determine_pre_tokenizer_type(arch: &str) -> String {
    match arch {
        // llama-vocab.cpp:2029 — Qwen3.5 / Qwen3.6 family.
        "qwen35" | "qwen35moe" => "qwen35".into(),
        // llama-vocab.cpp:2022 — Qwen2 family.
        "qwen2" | "qwen3" => "qwen2".into(),
        // llama-vocab.cpp:2005 — Gemma4.
        "gemma4" | "gemma3" => "gemma4".into(),
        // llama-vocab.cpp:1951-1959 — LLaMA3 BPE family.
        "llama" | "mistral" => "llama-bpe".into(),
        // Fallback for unknown arch — "default" (LLAMA_VOCAB_PRE_TYPE_DEFAULT).
        _ => "default".into(),
    }
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
    // ADR-012 Decision 1: resolve GGUF arch from architectures[0] for qwen35/qwen35moe.
    // For all other arches this returns model_type unchanged.
    let arch_owned = arch_gguf_name(meta);
    let arch = &arch_owned; // e.g. "llama", "gemma4", "qwen35", "qwen35moe"

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
    // ADR-013 P14: qwen35 / qwen35moe block_count includes appended MTP
    // blocks when present. Other arches keep their historical layer count.
    //
    // Temporary escape hatch mirrors src/main.rs Phase 1.42:
    //   HF2Q_QWEN35_DROP_MTP=1
    //
    // REMOVE when llama.cpp adds qwen35 MTP loading OR by 2026-Q4 if upstream
    // lags. With the hatch enabled, block_count remains the raw layer count so
    // a deliberately stripped file is internally consistent.
    let drop_mtp_escape = std::env::var("HF2Q_QWEN35_DROP_MTP").as_deref() == Ok("1");
    let block_count = match (arch.as_str(), drop_mtp_escape) {
        ("qwen35" | "qwen35moe", false) => {
            meta.num_layers + meta.mtp_num_hidden_layers.unwrap_or(0)
        }
        _ => meta.num_layers,
    };
    kv.push((
        format!("{}.block_count", arch),
        MetaValue::Uint32(block_count),
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

    // Expert count and used count (MoE models)
    if let Some(n_expert) = meta.num_experts {
        kv.push((
            format!("{}.expert_count", arch),
            MetaValue::Uint32(n_expert),
        ));
    }
    if let Some(n_used) = meta.top_k_experts {
        kv.push((
            format!("{}.expert_used_count", arch),
            MetaValue::Uint32(n_used),
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

    // ADR-012 Decision 7: qwen35 / qwen35moe metadata emission.
    // Both variants share the majority of keys; MoE adds expert_* keys.
    if arch == "qwen35" || arch == "qwen35moe" {
        emit_qwen35_metadata(meta, arch, &mut kv);
    }

    // Load and embed tokenizer metadata from input directory
    if let Some(tok_kv) = load_tokenizer_metadata(input_dir, arch) {
        kv.extend(tok_kv);
    }

    kv
}

/// Emit Qwen3.5-family (dense + MoE) GGUF metadata keys.
///
/// # Key catalog with citations (ADR-012 Decision 7)
///
/// All keys are prefixed `{arch}.*` where arch is `"qwen35"` or `"qwen35moe"`.
/// Every key cited below is mandatory unless marked (optional).
///
/// ## SSM / linear-attention hparams — llama-model.cpp:2811-2815 (qwen35)
///                                      llama-model.cpp:2842-2846 (qwen35moe)
/// - `{arch}.ssm.conv_kernel`   = linear_conv_kernel_dim   (llama-arch.cpp:268, LLM_KV_SSM_CONV_KERNEL)
/// - `{arch}.ssm.inner_size`    = linear_value_head_dim * linear_num_value_heads (llama-arch.cpp:269)
/// - `{arch}.ssm.state_size`    = linear_key_head_dim       (llama-arch.cpp:270, LLM_KV_SSM_STATE_SIZE)
/// - `{arch}.ssm.time_step_rank = linear_num_value_heads    (llama-arch.cpp:271, LLM_KV_SSM_TIME_STEP_RANK)
/// - `{arch}.ssm.group_count`   = linear_num_key_heads      (llama-arch.cpp:272, LLM_KV_SSM_GROUP_COUNT)
///
/// ## Full-attention hparams
/// - `{arch}.attention.layer_norm_rms_epsilon`  = rms_norm_eps  (llama-arch.cpp:220, llama-model.cpp:2807)
/// - `{arch}.attention.key_length`              = head_dim       (llama-arch.cpp:217)
/// - `{arch}.attention.value_length`            = head_dim       (llama-arch.cpp:218)
///
/// ## RoPE hparams — llama-model.cpp:2808 reads rope_dimension_sections (mandatory)
/// - `{arch}.rope.freq_base`          = rope_theta             (llama-arch.cpp:249)
/// - `{arch}.rope.dimension_count`    = floor(head_dim * partial_rotary_factor) * 2 (llama-arch.cpp:246)
/// - `{arch}.rope.dimension_sections` = [mrope_section[0..4]]  (llama-arch.cpp:248)
///
/// ## Layer-type hparams
/// - `{arch}.full_attention_interval` = full_attention_interval (llama-arch.cpp:211, optional default 4)
///
/// ## MTP
/// - `{arch}.nextn_predict_layers` = mtp_num_hidden_layers (llama-arch.cpp:194, optional default 0)
///
/// ## MoE-only keys
/// - `{arch}.expert_count`                      (llama-arch.cpp:182, llama-model.cpp:2835 optional)
/// - `{arch}.expert_used_count`                 (llama-arch.cpp:183, llama-model.cpp:2836 optional)
/// - `{arch}.expert_feed_forward_length`        (llama-arch.cpp:175, llama-model.cpp:2835 optional)
/// - `{arch}.expert_shared_feed_forward_length` (llama-arch.cpp:176, llama-model.cpp:2836 optional)
fn emit_qwen35_metadata(
    meta: &crate::ir::ModelMetadata,
    arch: &str,
    kv: &mut Vec<(String, MetaValue)>,
) {
    // SSM inner_size = linear_value_head_dim * linear_num_value_heads
    // convert_hf_to_gguf.py:4779  add_ssm_inner_size(hparams["linear_value_head_dim"] * hparams["linear_num_value_heads"])
    if let (Some(vdim), Some(vheads)) = (meta.linear_value_head_dim, meta.linear_num_value_heads) {
        // llama-arch.cpp:268 LLM_KV_SSM_CONV_KERNEL → "%s.ssm.conv_kernel"
        // llama-model.cpp:2811/2842 mandatory read
        if let Some(conv_k) = meta.linear_conv_kernel_dim {
            kv.push((format!("{}.ssm.conv_kernel", arch), MetaValue::Uint32(conv_k)));
        }
        // llama-arch.cpp:269 LLM_KV_SSM_INNER_SIZE → "%s.ssm.inner_size"
        // llama-model.cpp:2812/2843 mandatory read
        kv.push((format!("{}.ssm.inner_size", arch), MetaValue::Uint32(vdim * vheads)));
    }
    // llama-arch.cpp:270 LLM_KV_SSM_STATE_SIZE → "%s.ssm.state_size"
    // llama-model.cpp:2813/2844 mandatory read = ssm_d_state = linear_key_head_dim
    if let Some(kd) = meta.linear_key_head_dim {
        kv.push((format!("{}.ssm.state_size", arch), MetaValue::Uint32(kd)));
    }
    // llama-arch.cpp:271 LLM_KV_SSM_TIME_STEP_RANK → "%s.ssm.time_step_rank"
    // llama-model.cpp:2814/2845 mandatory read = ssm_dt_rank = linear_num_value_heads
    if let Some(vheads) = meta.linear_num_value_heads {
        kv.push((format!("{}.ssm.time_step_rank", arch), MetaValue::Uint32(vheads)));
    }
    // llama-arch.cpp:272 LLM_KV_SSM_GROUP_COUNT → "%s.ssm.group_count"
    // llama-model.cpp:2815/2846 mandatory read = ssm_n_group = linear_num_key_heads
    if let Some(kheads) = meta.linear_num_key_heads {
        kv.push((format!("{}.ssm.group_count", arch), MetaValue::Uint32(kheads)));
    }

    // RMS norm epsilon — llama-arch.cpp:220 LLM_KV_ATTENTION_LAYERNORM_RMS_EPS
    // llama-model.cpp:2807/2837 mandatory read
    let rms_eps = meta.raw_config
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    kv.push((format!("{}.attention.layer_norm_rms_epsilon", arch), MetaValue::Float32(rms_eps)));

    // head_dim: key_length and value_length — llama-arch.cpp:217-218
    // convert_hf_to_gguf.py:1192-1193  add_key_length / add_value_length from hparams["head_dim"]
    if let Some(hd) = meta.head_dim {
        kv.push((format!("{}.attention.key_length", arch), MetaValue::Uint32(hd)));
        kv.push((format!("{}.attention.value_length", arch), MetaValue::Uint32(hd)));
    }

    // RoPE freq_base — llama-arch.cpp:249 LLM_KV_ROPE_FREQ_BASE
    // convert_hf_to_gguf.py:1157-1159  add_rope_freq_base from rope_parameters.rope_theta
    if let Some(ref rp) = meta.rope_parameters {
        // rope_theta = 10_000_000 for Qwen3.5-MoE apex (rope_parameters.rope_theta)
        let rope_theta = if rp.rope_theta > 0.0 { rp.rope_theta } else { 10_000_000.0 };
        kv.push((format!("{}.rope.freq_base", arch), MetaValue::Float32(rope_theta as f32)));

        // RoPE dimension_sections — llama-arch.cpp:248 LLM_KV_ROPE_DIMENSION_SECTIONS
        // llama-model.cpp:2808/2839 mandatory read via get_key_or_arr
        // convert_hf_to_gguf.py:1149-1154 emits mrope_section padded to 4 elements
        let mut sections = rp.mrope_section.clone();
        while sections.len() < 4 {
            sections.push(0);
        }
        let sections_u32: Vec<u32> = sections.iter().take(4).copied().collect();
        kv.push((format!("{}.rope.dimension_sections", arch), MetaValue::ArrayUint32(sections_u32)));

        // RoPE dimension_count — llama-arch.cpp:246 LLM_KV_ROPE_DIMENSION_COUNT
        // convert_hf_to_gguf.py:4783  rope_dim = int(head_dim * partial_rotary_factor)
        // partial_rotary_factor: prefer rp.partial_rotary_factor; fallback to meta field; then 0.25
        if let Some(hd) = meta.head_dim {
            let prf = if rp.partial_rotary_factor > 0.0 {
                rp.partial_rotary_factor
            } else {
                meta.partial_rotary_factor.unwrap_or(0.25)
            };
            let rope_dim = (hd as f32 * prf) as u32;
            kv.push((format!("{}.rope.dimension_count", arch), MetaValue::Uint32(rope_dim)));
        }
    }

    // full_attention_interval — llama-arch.cpp:211 LLM_KV_FULL_ATTENTION_INTERVAL
    // llama-model.cpp:2820/2851 optional read, default 4
    let fai = meta.full_attention_interval.unwrap_or(4);
    kv.push((format!("{}.full_attention_interval", arch), MetaValue::Uint32(fai)));

    // nextn_predict_layers — llama-arch.cpp:194 LLM_KV_NEXTN_PREDICT_LAYERS
    // optional, default 0; MTP block count.  mtp_num_hidden_layers = 1 for the apex model.
    // Suppressed only by the temporary HF2Q_QWEN35_DROP_MTP=1 conversion hatch.
    if std::env::var("HF2Q_QWEN35_DROP_MTP").as_deref() != Ok("1") {
        if let Some(mtp) = meta.mtp_num_hidden_layers {
            if mtp > 0 {
                kv.push((format!("{}.nextn_predict_layers", arch), MetaValue::Uint32(mtp)));
            }
        }
    }

    // MoE-only keys
    if arch == "qwen35moe" {
        // llama-arch.cpp:175 LLM_KV_EXPERT_FEED_FORWARD_LENGTH → "%s.expert_feed_forward_length"
        // llama-model.cpp:2835 optional read
        if let Some(ff_exp) = meta.moe_intermediate_size {
            kv.push((format!("{}.expert_feed_forward_length", arch), MetaValue::Uint32(ff_exp)));
        }
        // llama-arch.cpp:176 LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH → "%s.expert_shared_feed_forward_length"
        // llama-model.cpp:2836 optional read
        if let Some(ff_shexp) = meta.shared_expert_intermediate_size {
            kv.push((format!("{}.expert_shared_feed_forward_length", arch), MetaValue::Uint32(ff_shexp)));
        }
        // Note: expert_count and expert_used_count are already emitted by the common
        // path in build_metadata (lines 2004-2015) using meta.num_experts / meta.top_k_experts.
    }
}

/// Map global bit width to a GGML file type code.
///
/// These correspond to llama.cpp's `llama_ftype` enum values.
/// For K-quant models the ftype is selected based on the dominant bit width.
fn ggml_ftype_from_bits(bits: u8) -> u32 {
    match bits {
        16 => 1,  // MOSTLY_F16
        8 => 7,   // MOSTLY_Q8_0
        4 => 15,  // MOSTLY_Q4_K_M
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

/// Resolve the GGUF architecture string for a model.
///
/// # Decision 1 (ADR-012 P4): arch routing from `config.architectures[0]`
///
/// llama.cpp uses `general.architecture` in the GGUF file to select the model
/// builder.  For Qwen3.5-family models the HF `model_type` field is not stable
/// (e.g. `"qwen3_5_moe_text"` instead of the GGUF arch `"qwen35moe"`), so we
/// normalize from `architectures[0]` (stored in `metadata.architecture`).
///
/// Mapping (llama-arch.cpp:42-43):
///   `"Qwen3_5ForCausalLM"`    → `"qwen35"`      (LLM_ARCH_QWEN35)
///   `"Qwen3_5MoeForCausalLM"` → `"qwen35moe"`   (LLM_ARCH_QWEN35MOE)
///
/// All other architectures fall back to `metadata.model_type` unchanged — this
/// preserves the gemma4 path and every other existing arch.
pub(crate) fn arch_gguf_name(metadata: &crate::ir::ModelMetadata) -> String {
    match metadata.architecture.as_str() {
        // llama-arch.cpp:42  { LLM_ARCH_QWEN35,    "qwen35"    }
        "Qwen3_5ForCausalLM" | "Qwen3_5ForConditionalGeneration" => "qwen35".to_string(),
        // llama-arch.cpp:43  { LLM_ARCH_QWEN35MOE, "qwen35moe" }
        "Qwen3_5MoeForCausalLM" | "Qwen3_5MoeForConditionalGeneration" => "qwen35moe".to_string(),
        _ => metadata.model_type.clone(),
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
            explicit_layer_types: None, full_attention_interval: None,
            attn_output_gate: None, head_dim: None, partial_rotary_factor: None,
            rope_parameters: None, linear_conv_kernel_dim: None,
            linear_key_head_dim: None, linear_num_key_heads: None,
            linear_value_head_dim: None, linear_num_value_heads: None,
            mamba_ssm_dtype: None, moe_intermediate_size: None,
            shared_expert_intermediate_size: None, mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None, output_router_logits: None,
            router_aux_loss_coef: None,
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

    fn model_with_metadata(metadata: ModelMetadata) -> QuantizedModel {
        QuantizedModel {
            metadata,
            tensors: Default::default(),
            quant_method: "f16".into(),
            group_size: 64,
            bits: 16,
        }
    }

    fn metadata_u32(kv: &[(String, MetaValue)], key: &str) -> Option<u32> {
        kv.iter().find_map(|(k, v)| {
            if k == key {
                if let MetaValue::Uint32(n) = v {
                    Some(*n)
                } else {
                    None
                }
            } else {
                None
            }
        })
    }

    #[test]
    fn test_name_mapping_llama() {
        // Static mappings (arch-independent)
        assert_eq!(hf_name_to_gguf_no_mtp("model.embed_tokens.weight", "llama"), "token_embd.weight");
        assert_eq!(hf_name_to_gguf_no_mtp("model.norm.weight", "llama"), "output_norm.weight");
        assert_eq!(hf_name_to_gguf_no_mtp("lm_head.weight", "llama"), "output.weight");
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
        for (hf, gguf) in cases { assert_eq!(hf_name_to_gguf_no_mtp(hf, "llama"), gguf, "mapping failed for {}", hf); }
        // Unknown passthrough
        assert_eq!(hf_name_to_gguf_no_mtp("model.layers.5.some_new.weight", "llama"), "blk.5.some_new.weight");
        assert_eq!(hf_name_to_gguf_no_mtp("decoder.block.0.weight", "llama"), "decoder.block.0.weight");
        // Verify LLaMA-like archs all behave the same
        for arch in &["mistral", "qwen3", "qwen2", "phi"] {
            assert_eq!(
                hf_name_to_gguf_no_mtp("model.layers.0.post_attention_layernorm.weight", arch),
                "blk.0.ffn_norm.weight",
                "LLaMA-like arch '{}' should map post_attention_layernorm to ffn_norm", arch,
            );
        }
    }

    #[test]
    fn test_name_mapping_gemma4() {
        // Gemma4: post_attention_layernorm is a distinct norm, NOT ffn_norm
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.0.post_attention_layernorm.weight", "gemma4"),
            "blk.0.post_attention_norm.weight",
        );
        // Gemma4: pre_feedforward_layernorm IS ffn_norm
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.0.pre_feedforward_layernorm.weight", "gemma4"),
            "blk.0.ffn_norm.weight",
        );
        // MoE norms
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.5.post_feedforward_layernorm_1.weight", "gemma4"),
            "blk.5.post_ffw_norm_1.weight",
        );
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.5.post_feedforward_layernorm_2.weight", "gemma4"),
            "blk.5.post_ffw_norm_2.weight",
        );
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.5.pre_feedforward_layernorm_2.weight", "gemma4"),
            "blk.5.pre_ffw_norm_2.weight",
        );
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.5.post_feedforward_layernorm.weight", "gemma4"),
            "blk.5.post_ffw_norm.weight",
        );
        // MoE routing
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.3.router.proj.weight", "gemma4"),
            "blk.3.ffn_gate_inp.weight",
        );
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.3.router.scale", "gemma4"),
            "blk.3.ffn_gate_inp.scale",
        );
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.3.experts.gate_up_proj", "gemma4"),
            "blk.3.ffn_gate_up_exps.weight",
        );
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.3.experts.down_proj", "gemma4"),
            "blk.3.ffn_down_exps.weight",
        );
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.3.router.per_expert_scale", "gemma4"),
            "blk.3.ffn_down_exps.scale",
        );
        // Layer scalar
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.0.layer_scalar", "gemma4"),
            "blk.0.layer_output_scale.weight",
        );
        // Shared entries still work for Gemma4
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.0.self_attn.q_proj.weight", "gemma4"),
            "blk.0.attn_q.weight",
        );
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.0.input_layernorm.weight", "gemma4"),
            "blk.0.attn_norm.weight",
        );
        // Gemma3/Gemma2 behave the same as Gemma4
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.0.post_attention_layernorm.weight", "gemma3"),
            "blk.0.post_attention_norm.weight",
        );
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.0.post_attention_layernorm.weight", "gemma2"),
            "blk.0.post_attention_norm.weight",
        );
        // language_model. prefix stripping still works
        assert_eq!(
            hf_name_to_gguf_no_mtp("language_model.model.layers.0.pre_feedforward_layernorm.weight", "gemma4"),
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
        assert_eq!(quant_info_to_ggml_type(&qi(6, false)), GGML_TYPE_Q6_K); // 6-bit uses Q6_K super-block
    }

    #[test]
    fn test_validate_clean_and_unsupported() {
        let backend = GgufBackend::new();
        // Clean model: no warnings
        let w = backend.validate(&model(vec![("t1", 4, false), ("t2", 16, true)], 4)).unwrap();
        assert!(w.is_empty(), "Expected no warnings: {:?}", w);
        // 6-bit is now supported (Q6_K) — no warnings
        let w = backend.validate(&model(vec![("odd", 6, false)], 6)).unwrap();
        assert!(w.is_empty(), "6-bit should produce no warnings: {:?}", w);
        // Truly unsupported bit width (e.g. 3-bit): 1 warning
        let w = backend.validate(&model(vec![("odd", 3, false)], 3)).unwrap();
        assert_eq!(w.len(), 1);
        assert!(w[0].message.contains("3-bit"));
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

        // K-quant types (all S/M/L variants map to same unified type)
        assert_eq!(ggml_type_from_name("Q2_K"), Some(GGML_TYPE_Q2_K));
        assert_eq!(ggml_type_from_name("Q3_K_S"), Some(GGML_TYPE_Q3_K));
        assert_eq!(ggml_type_from_name("Q3_K_M"), Some(GGML_TYPE_Q3_K));
        assert_eq!(ggml_type_from_name("Q3_K_L"), Some(GGML_TYPE_Q3_K));
        assert_eq!(ggml_type_from_name("Q3_K"), Some(GGML_TYPE_Q3_K));
        assert_eq!(ggml_type_from_name("Q4_K_S"), Some(GGML_TYPE_Q4_K));
        assert_eq!(ggml_type_from_name("Q4_K_M"), Some(GGML_TYPE_Q4_K));
        assert_eq!(ggml_type_from_name("Q4_K"), Some(GGML_TYPE_Q4_K));
        assert_eq!(ggml_type_from_name("Q5_K_S"), Some(GGML_TYPE_Q5_K));
        assert_eq!(ggml_type_from_name("Q5_K_M"), Some(GGML_TYPE_Q5_K));
        assert_eq!(ggml_type_from_name("Q5_K"), Some(GGML_TYPE_Q5_K));
        assert_eq!(ggml_type_from_name("Q6_K"), Some(GGML_TYPE_Q6_K));
        assert_eq!(ggml_type_from_name("IQ2_XXS"), Some(GGML_TYPE_IQ2_XXS));
        assert_eq!(ggml_type_from_name("IQ2_XS"), Some(GGML_TYPE_IQ2_XS));

        // Case insensitivity
        assert_eq!(ggml_type_from_name("q4_k_m"), Some(GGML_TYPE_Q4_K));
        assert_eq!(ggml_type_from_name("q6_k"), Some(GGML_TYPE_Q6_K));

        // With GGML_TYPE_ prefix
        assert_eq!(ggml_type_from_name("GGML_TYPE_Q4_K_M"), Some(GGML_TYPE_Q4_K));

        // Unknown returns None
        assert_eq!(ggml_type_from_name("Q99_Z"), None);
        assert_eq!(ggml_type_from_name(""), None);
    }

    #[test]
    fn test_ggml_type_override_in_quant_info() {
        // ADR-014 P11-prereq Iter C (2026-04-27): the legacy `apex`
        // `Q4_K_M` → `GGML_TYPE_Q4_0` swallow was REMOVED.  Apex itself
        // was already deleted by Decision 12; the only remaining
        // production writer of `ggml_type = Some("Q[2-5]_K"…)` is the
        // codec-direct sentinel path (locked below by
        // `codec_direct_*_returns_*`), which now routes via Iter A's
        // fast-path BEFORE this branch fires.  See
        // `quant_info_to_ggml_type`'s Iter C audit comment.
        //
        // The `qi_override` case below now exercises the
        // defence-in-depth guard: a K-quant `ggml_type` set WITHOUT
        // `METHOD_K_QUANT_CODEC_DIRECT` is unreachable in production
        // (audit lives at the function); if a future regression ever
        // re-introduces the path it surfaces loudly via
        // `debug_assert!(false, ...)` in debug builds + `warn!` →
        // `GGML_TYPE_F16` in release.  We test the release behaviour
        // by walking the function body explicitly: the assertion
        // panics in `cfg(debug_assertions)` so we route the test
        // around the call when debug_assertions is on, locking only
        // the release branch.  In release builds the function returns
        // `GGML_TYPE_F16` without panicking.
        #[cfg(not(debug_assertions))]
        {
            let qi_override = TensorQuantInfo {
                method: "apex".into(), bits: 4, group_size: 64, preserved: false,
                scales: None, biases: None, ggml_type: Some("Q4_K_M".into()),
            };
            assert_eq!(
                quant_info_to_ggml_type(&qi_override),
                GGML_TYPE_F16,
                "Iter C: K-quant ggml_type without codec-direct sentinel falls \
                 back to F16 in release (defence-in-depth, was Q4_0 pre-Iter C)"
            );
        }

        // Q6_K override (no codec-direct sentinel) is still honored
        // directly — this branch was preserved by Iter C because the
        // bare-name path is exercised by this test and the Q6_K leg
        // does not need the codec-direct sentinel to disambiguate
        // bytes (Q6_K kernel + repack arms exist).
        let qi_q6k = TensorQuantInfo {
            method: "apex".into(), bits: 8, group_size: 64, preserved: false,
            scales: None, biases: None, ggml_type: Some("Q6_K".into()),
        };
        assert_eq!(quant_info_to_ggml_type(&qi_q6k), GGML_TYPE_Q6_K);

        // Non-K-quant explicit type is still honored.
        let qi_q4_0 = TensorQuantInfo {
            method: "custom".into(), bits: 4, group_size: 32, preserved: false,
            scales: None, biases: None, ggml_type: Some("Q4_0".into()),
        };
        assert_eq!(quant_info_to_ggml_type(&qi_q4_0), GGML_TYPE_Q4_0);

        // Unknown ggml_type falls back to bits-based mapping.  The
        // `Q99_Z` name does NOT start with `Q[2-5]_K`, so it bypasses
        // the Iter C defence-in-depth guard and lands in the
        // bits-based table — same observable as pre-Iter C.
        let qi_unknown = TensorQuantInfo {
            method: "apex".into(), bits: 4, group_size: 64, preserved: false,
            scales: None, biases: None, ggml_type: Some("Q99_Z".into()),
        };
        assert_eq!(quant_info_to_ggml_type(&qi_unknown), GGML_TYPE_Q4_0);

        // preserved=true always returns F16 regardless of ggml_type
        // — short-circuits before the K-quant guard.
        let qi_preserved = TensorQuantInfo {
            method: "passthrough".into(), bits: 16, group_size: 0, preserved: true,
            scales: None, biases: None, ggml_type: Some("Q4_K_M".into()),
        };
        assert_eq!(quant_info_to_ggml_type(&qi_preserved), GGML_TYPE_F16);
    }

    /// ADR-014 P11-prereq Iter C: lock the defence-in-depth guard.  A
    /// K-quant `ggml_type` set WITHOUT `METHOD_K_QUANT_CODEC_DIRECT`
    /// must surface loudly in debug builds (`debug_assert!(false, ...)`
    /// panics) — proves the swallow removal is real and a future
    /// regression that re-introduces the apex-style path trips this
    /// gate immediately.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "non-codec-direct K-quant path is dead code")]
    fn iter_c_k_quant_without_codec_direct_sentinel_panics_in_debug() {
        let qi = TensorQuantInfo {
            method: "apex".into(),
            bits: 4,
            group_size: 64,
            preserved: false,
            scales: None,
            biases: None,
            ggml_type: Some("Q4_K_M".into()),
        };
        let _ = quant_info_to_ggml_type(&qi);
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

    // ---------------------------------------------------------------------
    // ADR-014 P11-prereq Iter A — codec-direct fast-path regression tests.
    // ---------------------------------------------------------------------
    //
    // Pre-Iter A: `KQuantCodecQuantizer` and `VariantKQuantizer` set
    // `bits = 0` + `method = "k_quant_codec_direct"` + `ggml_type =
    // Some("Q4_K"|"Q5_K"|"Q6_K"|...)`. The K-quant fall-through arm at
    // `gguf.rs:1645-1649` matched `Q4_K`/`Q5_K`/`Q3_K`/`Q2_K` but did
    // NOT return; control fell through to the bits-fallback `_ =>` arm
    // at `gguf.rs:1672-1677` which returned `GGML_TYPE_F16` (1) for
    // `bits = 0`. So GGUFs from `--quant q4_k_m` / `--quant
    // imatrix-q4_k_m` / `--quant imatrix-adaptive` shipped with header
    // type **F16** atop on-disk **Q4_K** bytes — MALFORMED for
    // llama.cpp.
    //
    // Post-Iter A: `quant_info_to_ggml_type`'s codec-direct fast-path
    // recognises the sentinel + routes via `ggml_type_from_name` BEFORE
    // the bits-fallback fires. `repack_to_ggml_blocks`'s matching
    // codec-direct fast-path returns the raw bytes unchanged with
    // block-size validation (so a future codec or shape regression
    // surfaces as a typed `Err` rather than silent half-block
    // corruption).

    fn qi_codec_direct(ggml_name: &str) -> TensorQuantInfo {
        TensorQuantInfo {
            method: crate::quantize::k_quant_codec_quantizer::METHOD_K_QUANT_CODEC_DIRECT
                .to_string(),
            bits: 0,
            group_size: 0,
            preserved: false,
            scales: None,
            biases: None,
            ggml_type: Some(ggml_name.to_string()),
        }
    }

    #[test]
    fn codec_direct_q4_k_returns_q4_k_type_code() {
        let qi = qi_codec_direct("Q4_K");
        assert_eq!(
            quant_info_to_ggml_type(&qi),
            GGML_TYPE_Q4_K,
            "codec-direct Q4_K must return GGML_TYPE_Q4_K (12), not GGML_TYPE_F16 (1)"
        );
    }

    #[test]
    fn codec_direct_q5_k_returns_q5_k_type_code() {
        let qi = qi_codec_direct("Q5_K");
        assert_eq!(quant_info_to_ggml_type(&qi), GGML_TYPE_Q5_K);
    }

    #[test]
    fn codec_direct_q6_k_returns_q6_k_type_code() {
        let qi = qi_codec_direct("Q6_K");
        assert_eq!(quant_info_to_ggml_type(&qi), GGML_TYPE_Q6_K);
    }

    #[test]
    fn codec_direct_q4_0_legacy_returns_q4_0_type_code() {
        // KQuantTarget::Q4Legacy maps to ggml_name "Q4_0" per
        // `KQuantCodecQuantizer::target_to_ggml_name` — locks the
        // legacy-target leg of the codec-direct sentinel routing.
        let qi = qi_codec_direct("Q4_0");
        assert_eq!(quant_info_to_ggml_type(&qi), GGML_TYPE_Q4_0);
    }

    #[test]
    fn codec_direct_q8_0_legacy_returns_q8_0_type_code() {
        let qi = qi_codec_direct("Q8_0");
        assert_eq!(quant_info_to_ggml_type(&qi), GGML_TYPE_Q8_0);
    }

    #[test]
    fn codec_direct_unknown_ggml_name_returns_f16_with_warn() {
        // Defensive: a codec-direct sentinel with an UNKNOWN ggml_type
        // name is a quantizer bug. We warn and return F16 (the only
        // safe response within the current `fn -> u32` signature). If
        // the signature is ever changed to `Result`, this should
        // become an `Err`.
        let qi = qi_codec_direct("Q99_NOPE");
        assert_eq!(quant_info_to_ggml_type(&qi), GGML_TYPE_F16);
    }

    #[test]
    fn codec_direct_repack_q4_k_returns_raw_bytes_unchanged() {
        // Synthetic "Q4_K block bytes": the contents are arbitrary —
        // we're locking the no-repack fast-path. Length must be a
        // multiple of BLOCK_Q4_K_SIZE (144) so the block-size
        // alignment check passes.
        let block_size = crate::quantize::k_quant::BLOCK_Q4_K_SIZE;
        let bytes: Vec<u8> = (0..block_size * 4).map(|i| (i & 0xFF) as u8).collect();
        let qt = QuantizedTensor {
            name: "blk.0.attn_q.weight".into(),
            shape: vec![4, 256],
            original_dtype: DType::F16,
            data: bytes.clone(),
            quant_info: qi_codec_direct("Q4_K"),
        };
        let out = repack_to_ggml_blocks(&qt, GGML_TYPE_Q4_K).expect("repack must succeed");
        assert_eq!(
            out, bytes,
            "codec-direct repack must return input bytes byte-identically (no repack)"
        );
    }

    #[test]
    fn codec_direct_repack_q5_k_returns_raw_bytes_unchanged() {
        let block_size = crate::quantize::k_quant::BLOCK_Q5_K_SIZE;
        let bytes: Vec<u8> = (0..block_size * 2).map(|i| ((i * 7) & 0xFF) as u8).collect();
        let qt = QuantizedTensor {
            name: "blk.0.attn_v.weight".into(),
            shape: vec![2, 256],
            original_dtype: DType::F16,
            data: bytes.clone(),
            quant_info: qi_codec_direct("Q5_K"),
        };
        let out = repack_to_ggml_blocks(&qt, GGML_TYPE_Q5_K).expect("repack must succeed");
        assert_eq!(out, bytes);
    }

    #[test]
    fn codec_direct_repack_q6_k_returns_raw_bytes_unchanged() {
        let bytes: Vec<u8> = (0..BLOCK_Q6_K_BYTES).map(|i| (i & 0xFF) as u8).collect();
        let qt = QuantizedTensor {
            name: "output.weight".into(),
            shape: vec![1, 256],
            original_dtype: DType::F16,
            data: bytes.clone(),
            quant_info: qi_codec_direct("Q6_K"),
        };
        let out = repack_to_ggml_blocks(&qt, GGML_TYPE_Q6_K).expect("repack must succeed");
        assert_eq!(out, bytes);
    }

    #[test]
    fn codec_direct_repack_rejects_misaligned_bytes() {
        // 144 + 1 bytes is NOT a multiple of BLOCK_Q4_K_SIZE = 144;
        // must fail loud (typed Err) rather than silently writing
        // half-block-aligned corruption to disk.
        let block_size = crate::quantize::k_quant::BLOCK_Q4_K_SIZE;
        let bytes = vec![0u8; block_size + 1];
        let qt = QuantizedTensor {
            name: "blk.0.attn_q.weight".into(),
            shape: vec![1, 257],
            original_dtype: DType::F16,
            data: bytes,
            quant_info: qi_codec_direct("Q4_K"),
        };
        let err = repack_to_ggml_blocks(&qt, GGML_TYPE_Q4_K).expect_err(
            "misaligned codec-direct bytes must surface as Err (not panic, not silent corruption)",
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("not a multiple of") && msg.contains("misaligned"),
            "Err message should name the misalignment; got: {msg}"
        );
    }

    #[test]
    fn codec_direct_repack_unsupported_target_type_errors() {
        // Codec-direct sentinel paired with a target type the codec
        // never produces (e.g. F32) is a routing bug. Refuse to write
        // potentially-corrupt bytes.
        let bytes = vec![0u8; 144];
        let qt = QuantizedTensor {
            name: "blk.0.weird.weight".into(),
            shape: vec![1, 256],
            original_dtype: DType::F16,
            data: bytes,
            quant_info: qi_codec_direct("Q4_K"),
        };
        let err = repack_to_ggml_blocks(&qt, GGML_TYPE_F32).expect_err(
            "codec-direct + unsupported target ggml_type must surface as Err",
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("unsupported target ggml_type") && msg.contains("refusing"),
            "Err message should name the unsupported routing; got: {msg}"
        );
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
        let mut dequantized = [0.0f32; 32];
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

    // ---------------------------------------------------------------------------
    // ADR-012 Decision 1: arch routing tests
    // ---------------------------------------------------------------------------

    fn meta_qwen35_dense() -> ModelMetadata {
        ModelMetadata {
            architecture: "Qwen3_5ForCausalLM".into(),
            model_type: "qwen3_5_text".into(), // HF model_type is NOT the GGUF arch
            param_count: 27_000_000_000,
            hidden_size: 5120, num_layers: 64, layer_types: vec![],
            num_attention_heads: 40, num_kv_heads: Some(8), vocab_size: 248320,
            dtype: "bfloat16".into(), shard_count: 1, num_experts: None,
            top_k_experts: None, intermediate_size: Some(13824),
            raw_config: serde_json::Value::Null,
            explicit_layer_types: None, full_attention_interval: Some(4),
            attn_output_gate: Some(true), head_dim: Some(256),
            partial_rotary_factor: Some(0.25), rope_parameters: None,
            linear_conv_kernel_dim: Some(4), linear_key_head_dim: Some(128),
            linear_num_key_heads: Some(16), linear_value_head_dim: Some(128),
            linear_num_value_heads: Some(32), mamba_ssm_dtype: None,
            moe_intermediate_size: None, shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: Some(1), mtp_use_dedicated_embeddings: Some(false),
            output_router_logits: None, router_aux_loss_coef: None,
        }
    }

    fn meta_qwen35_moe() -> ModelMetadata {
        ModelMetadata {
            architecture: "Qwen3_5MoeForCausalLM".into(),
            model_type: "qwen3_5_moe_text".into(), // HF model_type is NOT the GGUF arch
            param_count: 35_000_000_000,
            hidden_size: 2048, num_layers: 40, layer_types: vec![],
            num_attention_heads: 16, num_kv_heads: Some(2), vocab_size: 248320,
            dtype: "bfloat16".into(), shard_count: 1, num_experts: Some(256),
            top_k_experts: Some(8), intermediate_size: None,
            raw_config: serde_json::json!({"rms_norm_eps": 1e-6}),
            explicit_layer_types: None, full_attention_interval: Some(4),
            attn_output_gate: Some(true), head_dim: Some(256),
            partial_rotary_factor: Some(0.25),
            rope_parameters: Some(crate::ir::RopeParameters {
                mrope_interleaved: true,
                mrope_section: vec![11, 11, 10],
                rope_theta: 10_000_000.0,
                rope_type: "default".into(),
                partial_rotary_factor: 0.25,
            }),
            linear_conv_kernel_dim: Some(4), linear_key_head_dim: Some(128),
            linear_num_key_heads: Some(16), linear_value_head_dim: Some(128),
            linear_num_value_heads: Some(32), mamba_ssm_dtype: None,
            moe_intermediate_size: Some(512),
            shared_expert_intermediate_size: Some(512),
            mtp_num_hidden_layers: Some(1), mtp_use_dedicated_embeddings: Some(false),
            output_router_logits: None, router_aux_loss_coef: None,
        }
    }

    /// Decision 1: architectures[0] == "Qwen3_5ForCausalLM" → arch string "qwen35".
    /// llama-arch.cpp:42  { LLM_ARCH_QWEN35, "qwen35" }
    #[test]
    fn arch_routing_qwen35_dense() {
        let m = meta_qwen35_dense();
        assert_eq!(
            arch_gguf_name(&m),
            "qwen35",
            "Qwen3_5ForCausalLM must route to GGUF arch 'qwen35' (llama-arch.cpp:42)"
        );
    }

    /// Decision 1: architectures[0] == "Qwen3_5MoeForCausalLM" → arch string "qwen35moe".
    /// llama-arch.cpp:43  { LLM_ARCH_QWEN35MOE, "qwen35moe" }
    #[test]
    fn arch_routing_qwen35_moe() {
        let m = meta_qwen35_moe();
        assert_eq!(
            arch_gguf_name(&m),
            "qwen35moe",
            "Qwen3_5MoeForCausalLM must route to GGUF arch 'qwen35moe' (llama-arch.cpp:43)"
        );
    }

    /// Decision 1: Non-qwen35 architectures pass model_type through unchanged.
    /// Chesterton's fence: gemma4 must continue to emit "gemma4".
    #[test]
    fn arch_routing_gemma4_unchanged() {
        let mut m = meta();
        m.architecture = "Gemma4ForConditionalGeneration".into();
        m.model_type = "gemma4".into();
        assert_eq!(arch_gguf_name(&m), "gemma4");
    }

    /// Decision 1: Qwen3.5 ships two HF architecture aliases per variant —
    /// `*ForCausalLM` (pure language model) and `*ForConditionalGeneration`
    /// (multimodal / vision-tower shipping models). Both must route to the
    /// same llama.cpp arch string. Dropping either alias in a "simpler"
    /// refactor would silently misroute multimodal checkpoints.
    #[test]
    fn arch_routing_qwen35_conditional_generation() {
        let mut m = meta_qwen35_dense();
        m.architecture = "Qwen3_5ForConditionalGeneration".into();
        assert_eq!(
            arch_gguf_name(&m),
            "qwen35",
            "Qwen3_5ForConditionalGeneration must route to 'qwen35' same as *ForCausalLM"
        );

        let mut m = meta_qwen35_moe();
        m.architecture = "Qwen3_5MoeForConditionalGeneration".into();
        assert_eq!(
            arch_gguf_name(&m),
            "qwen35moe",
            "Qwen3_5MoeForConditionalGeneration must route to 'qwen35moe' same as *ForCausalLM"
        );
    }

    /// Decision 1: llama architecture passes through unchanged.
    #[test]
    fn arch_routing_llama_unchanged() {
        let m = meta();
        assert_eq!(arch_gguf_name(&m), "llama");
    }

    #[test]
    fn qwen35_block_count_includes_mtp_layers() {
        let tmp = tempfile::tempdir().unwrap();

        let dense = model_with_metadata(meta_qwen35_dense());
        let dense_kv = build_metadata(&dense, tmp.path());
        assert_eq!(
            metadata_u32(&dense_kv, "qwen35.block_count"),
            Some(65),
            "qwen35 block_count must include 64 main layers + 1 MTP layer"
        );

        let moe = model_with_metadata(meta_qwen35_moe());
        let moe_kv = build_metadata(&moe, tmp.path());
        assert_eq!(
            metadata_u32(&moe_kv, "qwen35moe.block_count"),
            Some(41),
            "qwen35moe block_count must include 40 main layers + 1 MTP layer"
        );
    }

    #[test]
    fn non_qwen_block_count_does_not_include_mtp_layers() {
        let tmp = tempfile::tempdir().unwrap();
        let mut metadata = meta();
        metadata.mtp_num_hidden_layers = Some(1);

        let model = model_with_metadata(metadata);
        let kv = build_metadata(&model, tmp.path());
        assert_eq!(
            metadata_u32(&kv, "llama.block_count"),
            Some(32),
            "non-qwen arches must keep historical block_count semantics"
        );
    }

    // ---------------------------------------------------------------------------
    // ADR-012 Decision 8: qwen35 tensor name mapping tests (via hf_name_to_gguf)
    // ---------------------------------------------------------------------------

    /// Decision 8: post_attention_layernorm → post_attention_norm for qwen35.
    /// llama-arch.cpp:367, llama-model.cpp:7628.
    #[test]
    fn test_qwen35_post_attention_norm_mapping() {
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.0.post_attention_layernorm.weight", "qwen35"),
            "blk.0.post_attention_norm.weight",
            "qwen35: post_attention_layernorm must map to post_attention_norm, not ffn_norm"
        );
        assert_eq!(
            hf_name_to_gguf_no_mtp("model.layers.5.post_attention_layernorm.weight", "qwen35moe"),
            "blk.5.post_attention_norm.weight",
            "qwen35moe: post_attention_layernorm must map to post_attention_norm"
        );
    }

    /// Regression: LLaMA-like archs still map post_attention_layernorm → ffn_norm.
    #[test]
    fn test_llama_post_attention_norm_unchanged() {
        for arch in &["llama", "qwen3", "mistral"] {
            assert_eq!(
                hf_name_to_gguf_no_mtp("model.layers.0.post_attention_layernorm.weight", arch),
                "blk.0.ffn_norm.weight",
                "LLaMA-like arch '{}' must still map to ffn_norm", arch
            );
        }
    }

    /// Decision 11 + P11 fix: MTP tensors map to blk.{num_hidden_layers + mtp_idx}.nextn.*
    /// for qwen35/qwen35moe — the resolved block index, NOT a "blk.mtpN" placeholder.
    /// llama-arch.cpp:447-450 LLM_TENSOR_NEXTN_*.
    ///
    /// The earlier "blk.mtp{idx}" form was a P4 stub that ADR-012 P11 (2026-04-24)
    /// caught via the MTP round-trip gate: llama.cpp's loader + ADR-013's
    /// load_mtp_weights_if_present both look for `blk.{num_hidden_layers}.nextn.*`.
    /// The placeholder was silently dropping MTP tensors from converted GGUFs.
    #[test]
    fn test_qwen35_mtp_tensor_mapping() {
        // With num_hidden_layers=40 (Qwen3.5-MoE / Qwen3.6-35B-A3B), mtp_idx=0 →
        // block index 40. The resolved names are what llama.cpp's loader expects.
        assert_eq!(
            hf_name_to_gguf("mtp.layers.0.enorm.weight", "qwen35moe", 40),
            "blk.40.nextn.enorm.weight",
            "MTP enorm must map to nextn.enorm at resolved block (llama-arch.cpp:449)"
        );
        assert_eq!(
            hf_name_to_gguf("mtp.layers.0.hnorm.weight", "qwen35", 64),
            "blk.64.nextn.hnorm.weight",
            "MTP hnorm must map to nextn.hnorm at resolved block (llama-arch.cpp:450)"
        );
        assert_eq!(
            hf_name_to_gguf("mtp.layers.0.embed_tokens.weight", "qwen35moe", 40),
            "blk.40.nextn.embed_tokens.weight",
            "MTP embed_tokens at resolved block (llama-arch.cpp:448)"
        );
        assert_eq!(
            hf_name_to_gguf("mtp.layers.0.eh_proj.weight", "qwen35moe", 40),
            "blk.40.nextn.eh_proj.weight",
            "MTP eh_proj at resolved block (llama-arch.cpp:447)"
        );
        // Second MTP block (mtp_idx=1) with 40 layers → block 41.
        assert_eq!(
            hf_name_to_gguf("mtp.layers.1.enorm.weight", "qwen35moe", 40),
            "blk.41.nextn.enorm.weight",
            "mtp_idx=1 with num_layers=40 → block 41"
        );
    }

    /// Decision 19: the production write path resolves the placeholder using
    /// `num_hidden_layers`.  This test pins the resolver in isolation; the
    /// end-to-end roundtrip is in `tests/convert_qwen35_mtp_roundtrip.rs`.
    #[test]
    fn test_qwen35_mtp_resolved_block_index() {
        // qwen35moe apex: num_layers=40 → MTP block lands at blk.40.
        assert_eq!(
            resolve_mtp_block_index("blk.mtp0.nextn.enorm.weight", 40),
            "blk.40.nextn.enorm.weight",
        );
        // qwen35 dense 27B: num_layers=64 → MTP block lands at blk.64.
        assert_eq!(
            resolve_mtp_block_index("blk.mtp0.nextn.embed_tokens.weight", 64),
            "blk.64.nextn.embed_tokens.weight",
        );
        // mtp_num_hidden_layers > 1: each MTP block gets its own offset.
        assert_eq!(
            resolve_mtp_block_index("blk.mtp1.nextn.eh_proj.weight", 40),
            "blk.41.nextn.eh_proj.weight",
        );
        // Non-MTP names pass through unchanged (no false positives).
        for unaffected in &[
            "blk.0.attn_norm.weight",
            "blk.39.ssm_a",
            "blk.mtpfoo.weight",  // not parseable as u32 → unchanged
            "token_embd.weight",
        ] {
            assert_eq!(resolve_mtp_block_index(unaffected, 40), *unaffected);
        }
    }

    /// iter-116f (was iter-115): gemma4v Gemma4ClippableLinear scalar
    /// bounds map to GGUF `mm.input_projection.{input_min, input_max,
    /// output_min, output_max}` (TN_MM_INP_PROJ at
    /// `/opt/llama.cpp/tools/mtmd/clip-impl.h:110` + clip.cpp:1937-1959
    /// scalar lookup substituting `.weight` -> `.input_min` etc on the
    /// SAME projector base name). iter-115 originally used `mm.0.*` which
    /// llama.cpp's CLIP loader rejects (V_MMPROJ "mm.{bid}" is
    /// projector_type=mlp/llava — gemma4v's PROJECTOR_TYPE_GEMMA4V uses
    /// V_MM_INP_PROJ "mm.input_projection" exclusively).
    #[test]
    fn test_gemma4v_clippable_linear_scalar_bounds_mapping() {
        for arch in &["gemma4", "gemma4_multimodal", "llama"] {
            assert_eq!(
                hf_name_to_gguf("model.embed_vision.embedding_projection.input_min", arch, 0),
                "mm.input_projection.input_min",
                "[{arch}] input_min mapping"
            );
            assert_eq!(
                hf_name_to_gguf("model.embed_vision.embedding_projection.input_max", arch, 0),
                "mm.input_projection.input_max",
                "[{arch}] input_max mapping"
            );
            assert_eq!(
                hf_name_to_gguf("model.embed_vision.embedding_projection.output_min", arch, 0),
                "mm.input_projection.output_min",
                "[{arch}] output_min mapping"
            );
            assert_eq!(
                hf_name_to_gguf("model.embed_vision.embedding_projection.output_max", arch, 0),
                "mm.input_projection.output_max",
                "[{arch}] output_max mapping"
            );
            // Sanity: weight maps to the same projector base.
            assert_eq!(
                hf_name_to_gguf("model.embed_vision.embedding_projection.weight", arch, 0),
                "mm.input_projection.weight"
            );
        }
    }

    /// ADR-005 Phase 2c iter-116g: HWC→CHW transpose for the gemma4v
    /// patch-embd Conv2d weight matches PyTorch
    /// `t.reshape(n_embd, h, w, c).permute(0, 3, 1, 2).contiguous()`
    /// at byte level.
    ///
    /// Tiny fixture (n_embd=2, h=2, w=2, c=3, F32 elem_size=4):
    /// the source is a fully-distinct sequence; we hand-compute the
    /// expected dst index for every element and assert byte equality.
    #[test]
    fn test_patch_embd_hwc_to_chw_transpose() {
        let n_embd = 2usize;
        let h = 2usize;
        let w = 2usize;
        let c = 3usize;
        let elem_size = 4usize;
        let total = n_embd * h * w * c;

        // Source: src[((o*h+i)*w+j)*c+k] = (o, i, j, k) packed into a
        // distinct f32 so we can verify positional correctness.
        let mut src = Vec::<u8>::with_capacity(total * elem_size);
        for o in 0..n_embd {
            for i in 0..h {
                for j in 0..w {
                    for k in 0..c {
                        let v: f32 = (o * 1000 + i * 100 + j * 10 + k) as f32;
                        src.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }

        let dst = transpose_patch_embd_hwc_to_chw(&src, n_embd, h, w, c, elem_size);
        assert_eq!(dst.len(), total * elem_size);

        // dst[((o*c+k)*h+i)*w+j] must equal source's (o,i,j,k) value.
        for o in 0..n_embd {
            for k in 0..c {
                for i in 0..h {
                    for j in 0..w {
                        let dst_off = (((o * c + k) * h + i) * w + j) * elem_size;
                        let bytes: [u8; 4] = [
                            dst[dst_off],
                            dst[dst_off + 1],
                            dst[dst_off + 2],
                            dst[dst_off + 3],
                        ];
                        let got = f32::from_le_bytes(bytes);
                        let expected = (o * 1000 + i * 100 + j * 10 + k) as f32;
                        assert_eq!(
                            got, expected,
                            "dst[o={},c={},h={},w={}] expected {} got {}",
                            o, k, i, j, expected, got
                        );
                    }
                }
            }
        }
    }
}
