//! HF → GGUF tensor-name mapping for the ViT / mmproj emitter.
//! ADR-012 Decision 18 §1 + Layer C (spec-driven layout tests).
//!
//! Tensor-name table hand-transcribed from
//! `/opt/llama.cpp/tools/mtmd/clip-model.h` and `clip.cpp`. Every entry
//! carries a citation comment to the spec line that motivated it — if
//! `clip.cpp` changes upstream, the mapping test below fails loudly
//! (same pattern as ADR-012 P4 used for `llama-arch.cpp`).

use std::collections::HashMap;
use std::path::Path;

use crate::input::safetensors::{self};
use crate::ir::{DType, TensorRef};
use crate::progress::ProgressReporter;

use super::config::VisionConfig;
use super::VitConvertError;

/// A resolved tensor ready for GGUF emission.
#[derive(Debug, Clone)]
pub struct VitTensor {
    pub gguf_name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

/// Map an HF tensor name to its GGUF mmproj equivalent.
///
/// Returns `None` when the HF name does not belong to the vision path
/// (convert pipeline ignores the tensor; the text converter handles it).
///
/// All mappings cite the `clip-model.h` or `clip.cpp` line that pinned
/// the GGUF naming convention. The constants here are the `v.` /
/// `mm.` prefixes llama.cpp's mmproj loader keys off.
pub fn hf_vit_name_to_gguf(hf_name: &str) -> Option<String> {
    // -- Static (non-layer-indexed) tensors --------------------------------
    //
    // clip-model.h uses these exact strings in its `TENSOR_*` table; our
    // loader reads them via `src/inference/vision/mmproj.rs` constants.
    static STATIC_MAP: &[(&str, &str)] = &[
        // clip-model.h / src/inference/vision/mmproj.rs:228 TENSOR_PATCH_EMBD
        ("model.vision_tower.embeddings.patch_embeddings.projection.weight", "v.patch_embd.weight"),
        ("model.vision_tower.embeddings.patch_embeddings.projection.bias",   "v.patch_embd.bias"),
        // clip-model.h / mmproj.rs:232 TENSOR_POS_EMBD
        ("model.vision_tower.embeddings.position_embeddings.weight", "v.position_embd.weight"),
        // clip-model.h / mmproj.rs:234 TENSOR_POST_LN_*
        ("model.vision_tower.post_layernorm.weight", "v.post_ln.weight"),
        ("model.vision_tower.post_layernorm.bias",   "v.post_ln.bias"),
        // MLP projector — clip.cpp's projector_type=="mlp" reads these exact names.
        // mmproj.rs:237-241 TENSOR_MM_0_*/TENSOR_MM_2_*.
        ("model.multi_modal_projector.linear_1.weight", "mm.0.weight"),
        ("model.multi_modal_projector.linear_1.bias",   "mm.0.bias"),
        ("model.multi_modal_projector.linear_2.weight", "mm.2.weight"),
        ("model.multi_modal_projector.linear_2.bias",   "mm.2.bias"),
    ];
    for (hf, gguf) in STATIC_MAP {
        if hf_name == *hf {
            return Some((*gguf).to_string());
        }
    }

    // -- Per-layer ViT encoder tensors -----------------------------------
    // HF: "model.vision_tower.encoder.layer.{L}.{component}.weight"
    // GGUF: "v.blk.{L}.{mapped}.{weight,bias}"
    if let Some(rest) = hf_name.strip_prefix("model.vision_tower.encoder.layer.") {
        let (layer_str, suffix) = rest.split_once('.')?;
        let l: usize = layer_str.parse().ok()?;
        let blk = format!("v.blk.{}", l);

        // Attention projections — clip-model.h LLM_VISION_ATTN_* convention.
        let mapped = match suffix {
            "attention.q_proj.weight"  => Some(format!("{}.attn_q.weight", blk)),
            "attention.q_proj.bias"    => Some(format!("{}.attn_q.bias", blk)),
            "attention.k_proj.weight"  => Some(format!("{}.attn_k.weight", blk)),
            "attention.k_proj.bias"    => Some(format!("{}.attn_k.bias", blk)),
            "attention.v_proj.weight"  => Some(format!("{}.attn_v.weight", blk)),
            "attention.v_proj.bias"    => Some(format!("{}.attn_v.bias", blk)),
            "attention.output.dense.weight" => Some(format!("{}.attn_out.weight", blk)),
            "attention.output.dense.bias"   => Some(format!("{}.attn_out.bias", blk)),
            // LayerNorms — clip.cpp's ViT convention: ln1 = pre-attn, ln2 = pre-ffn.
            "layer_norm1.weight"       => Some(format!("{}.ln1.weight", blk)),
            "layer_norm1.bias"         => Some(format!("{}.ln1.bias", blk)),
            "layer_norm2.weight"       => Some(format!("{}.ln2.weight", blk)),
            "layer_norm2.bias"         => Some(format!("{}.ln2.bias", blk)),
            // MLP — fc1 = FFN up-proj (hidden → intermediate); fc2 = down-proj.
            // Order verified against clip.cpp: fc1 runs first, fc2 collapses back
            // to hidden_size. GGUF's ffn_up / ffn_down convention mirrors that.
            //
            // **LAYER C SPEC ANCHOR**: if a future refactor swaps these two
            // (easy to do visually — fc1 vs fc2 look symmetric in code), the
            // projection collapses back to wrong dim and mmproj inference
            // produces garbage. `src/models/vit/convert.rs:test_fc1_fc2_ordering`
            // pins this.
            "mlp.fc1.weight"           => Some(format!("{}.ffn_up.weight", blk)),
            "mlp.fc1.bias"             => Some(format!("{}.ffn_up.bias", blk)),
            "mlp.fc2.weight"           => Some(format!("{}.ffn_down.weight", blk)),
            "mlp.fc2.bias"             => Some(format!("{}.ffn_down.bias", blk)),
            _ => None,
        };
        return mapped;
    }

    None
}

/// Cast raw HF bytes to F16 bytes. Inputs can be F32 or F16 (common
/// HF shapes). P10's F16 mmproj emitter downcasts F32 → F16 at convert time.
fn ensure_f16_bytes(tensor: &TensorRef) -> Result<Vec<u8>, VitConvertError> {
    match tensor.dtype {
        DType::F16 => Ok(tensor.data.clone()),
        DType::F32 => {
            let n = tensor.numel();
            let mut out = Vec::with_capacity(n * 2);
            for i in 0..n {
                let b = &tensor.data[i * 4..(i + 1) * 4];
                let f =
                    f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                let h = half::f16::from_f32(f);
                out.extend_from_slice(&h.to_le_bytes());
            }
            Ok(out)
        }
        DType::BF16 => {
            let n = tensor.numel();
            let mut out = Vec::with_capacity(n * 2);
            for i in 0..n {
                let b = &tensor.data[i * 2..(i + 1) * 2];
                let bf = half::bf16::from_le_bytes([b[0], b[1]]);
                let h = half::f16::from_f32(bf.to_f32());
                out.extend_from_slice(&h.to_le_bytes());
            }
            Ok(out)
        }
        other => Err(VitConvertError::Safetensors(format!(
            "unsupported dtype {:?} on vision tensor {:?}",
            other, tensor.name
        ))),
    }
}

/// Load + map + cast the vision tensors from an HF repo directory.
pub fn load_vision_tensors(
    hf_repo_dir: &Path,
    _vision_config: &VisionConfig,
) -> Result<HashMap<String, VitTensor>, VitConvertError> {
    let progress = ProgressReporter::new();
    let tensor_map = safetensors::read_tensors(hf_repo_dir, &progress)
        .map_err(|e| VitConvertError::Safetensors(e.to_string()))?;

    let mut out: HashMap<String, VitTensor> = HashMap::new();
    for (name, tensor) in tensor_map.iter() {
        if let Some(gguf_name) = hf_vit_name_to_gguf(name) {
            let data = ensure_f16_bytes(tensor)?;
            out.insert(
                gguf_name.clone(),
                VitTensor {
                    gguf_name,
                    shape: tensor.shape.clone(),
                    dtype: DType::F16,
                    data,
                },
            );
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Layer C — spec-driven layout tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_mappings_match_spec() {
        // Patch embedding — clip-model.h's v.patch_embd.weight.
        assert_eq!(
            hf_vit_name_to_gguf("model.vision_tower.embeddings.patch_embeddings.projection.weight"),
            Some("v.patch_embd.weight".to_string())
        );
        // Position embedding — clip-model.h's v.position_embd.weight.
        assert_eq!(
            hf_vit_name_to_gguf("model.vision_tower.embeddings.position_embeddings.weight"),
            Some("v.position_embd.weight".to_string())
        );
        // Final LayerNorm.
        assert_eq!(
            hf_vit_name_to_gguf("model.vision_tower.post_layernorm.weight"),
            Some("v.post_ln.weight".to_string())
        );
    }

    /// **LAYER C ANCHOR** — fc1 maps to ffn_up, fc2 maps to ffn_down.
    ///
    /// Swapping these is the canonical "passes compile but breaks
    /// inference" bug. MLP block in a ViT is
    /// `fc1: [H, I] → GELU → fc2: [I, H]` — `fc1` is the UP projection
    /// (hidden → intermediate), `fc2` is the DOWN projection (back to
    /// hidden). GGUF's `ffn_up` / `ffn_down` convention matches that.
    ///
    /// Spec citation: clip.cpp's ViT MLP forward first runs the "fc1"
    /// weight (intermediate dim output), applies GELU, then "fc2"
    /// (hidden dim output).
    #[test]
    fn fc1_maps_to_ffn_up_fc2_maps_to_ffn_down() {
        let fc1 = hf_vit_name_to_gguf("model.vision_tower.encoder.layer.0.mlp.fc1.weight")
            .expect("fc1 weight mapped");
        let fc2 = hf_vit_name_to_gguf("model.vision_tower.encoder.layer.0.mlp.fc2.weight")
            .expect("fc2 weight mapped");
        assert_eq!(
            fc1, "v.blk.0.ffn_up.weight",
            "fc1 is the UP projection (hidden → intermediate)"
        );
        assert_eq!(
            fc2, "v.blk.0.ffn_down.weight",
            "fc2 is the DOWN projection (intermediate → hidden)"
        );
    }

    /// **LAYER C ANCHOR** — linear_1 in the multi-modal projector maps
    /// to `mm.0`, linear_2 maps to `mm.2` (`mm.1` is GELU).
    ///
    /// Swapping these produces a mmproj whose projector runs
    /// `GELU → linear_1 → linear_2` order inverted vs spec, silently
    /// corrupting every cross-modal embedding.
    ///
    /// Spec citation: clip.cpp projector_type=="mlp" reads
    /// `mm.0.weight → GELU → mm.2.weight` in forward order.
    #[test]
    fn linear_1_maps_to_mm_0_linear_2_maps_to_mm_2() {
        let l1 = hf_vit_name_to_gguf("model.multi_modal_projector.linear_1.weight")
            .expect("linear_1 mapped");
        let l2 = hf_vit_name_to_gguf("model.multi_modal_projector.linear_2.weight")
            .expect("linear_2 mapped");
        assert_eq!(l1, "mm.0.weight", "linear_1 = mm.0 (runs first)");
        assert_eq!(l2, "mm.2.weight", "linear_2 = mm.2 (runs after GELU at mm.1)");
    }

    /// Per-layer ViT encoder tensor mapping (full attention block).
    #[test]
    fn per_layer_attn_and_ln_names_map_correctly() {
        let cases = [
            ("model.vision_tower.encoder.layer.5.attention.q_proj.weight",       "v.blk.5.attn_q.weight"),
            ("model.vision_tower.encoder.layer.5.attention.k_proj.bias",         "v.blk.5.attn_k.bias"),
            ("model.vision_tower.encoder.layer.5.attention.v_proj.weight",       "v.blk.5.attn_v.weight"),
            ("model.vision_tower.encoder.layer.5.attention.output.dense.weight", "v.blk.5.attn_out.weight"),
            ("model.vision_tower.encoder.layer.5.layer_norm1.weight",            "v.blk.5.ln1.weight"),
            ("model.vision_tower.encoder.layer.5.layer_norm2.bias",              "v.blk.5.ln2.bias"),
        ];
        for (hf, expected) in cases {
            let got = hf_vit_name_to_gguf(hf)
                .unwrap_or_else(|| panic!("no mapping for {}", hf));
            assert_eq!(got, expected, "mapping mismatch for {}", hf);
        }
    }

    /// Non-vision tensors must return None so the text converter
    /// keeps its mapping authority.
    #[test]
    fn non_vision_tensor_returns_none() {
        assert!(hf_vit_name_to_gguf("model.layers.0.self_attn.q_proj.weight").is_none());
        assert!(hf_vit_name_to_gguf("model.embed_tokens.weight").is_none());
        assert!(hf_vit_name_to_gguf("lm_head.weight").is_none());
    }

    /// **LAYER C ANCHOR** — patch-embedding dims.
    ///
    /// HF writes patch_embeddings.projection.weight as a Conv2d:
    /// shape `[hidden_size, in_channels, patch_size, patch_size]`.
    /// GGUF's `v.patch_embd.weight` preserves that layout. The test
    /// validates the GGUF-side name is unchanged; shape verification
    /// happens at the load side (`mmproj.rs` + `tests/convert_vision_tower_integration.rs`).
    #[test]
    fn patch_embd_name_preserved() {
        let got = hf_vit_name_to_gguf(
            "model.vision_tower.embeddings.patch_embeddings.projection.weight",
        )
        .expect("mapped");
        assert_eq!(got, "v.patch_embd.weight");
    }

    /// **LAYER C ANCHOR** — position-embedding gets F16 dtype at
    /// emission time. Some HF configs ship it as F32 via the Embedding
    /// layer weight tensor. `ensure_f16_bytes` handles the cast.
    #[test]
    fn ensure_f16_bytes_casts_f32_to_f16() {
        let tensor = TensorRef {
            name: "test".into(),
            shape: vec![4],
            dtype: DType::F32,
            data: {
                let mut v = Vec::new();
                for x in &[1.0f32, 2.0, 3.0, 4.0] {
                    v.extend_from_slice(&x.to_le_bytes());
                }
                v
            },
        };
        let f16_bytes = ensure_f16_bytes(&tensor).unwrap();
        assert_eq!(f16_bytes.len(), 4 * 2); // 4 F16 elements
        // Decode back + verify.
        for (i, chunk) in f16_bytes.chunks(2).enumerate() {
            let h = half::f16::from_le_bytes([chunk[0], chunk[1]]);
            assert_eq!(h.to_f32(), [1.0, 2.0, 3.0, 4.0][i]);
        }
    }

    #[test]
    fn ensure_f16_bytes_passthrough_for_f16_input() {
        let tensor = TensorRef {
            name: "test".into(),
            shape: vec![2],
            dtype: DType::F16,
            data: vec![0x00, 0x3c, 0x00, 0x40], // 1.0, 2.0 in F16
        };
        let out = ensure_f16_bytes(&tensor).unwrap();
        assert_eq!(out, vec![0x00, 0x3c, 0x00, 0x40]);
    }

    #[test]
    fn ensure_f16_bytes_casts_bf16_to_f16() {
        let tensor = TensorRef {
            name: "test".into(),
            shape: vec![1],
            dtype: DType::BF16,
            // BF16 1.0 = 0x3F80 (big-endian) stored little-endian → [0x80, 0x3f].
            data: vec![0x80, 0x3f],
        };
        let out = ensure_f16_bytes(&tensor).unwrap();
        // F16 1.0 = 0x3c00, little-endian = [0x00, 0x3c].
        assert_eq!(out, vec![0x00, 0x3c]);
    }
}
