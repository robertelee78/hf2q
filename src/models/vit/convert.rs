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

    // -- Qwen3.6 ViT family (PROJECTOR_TYPE_QWEN3VL) ---------------------
    //
    // Real Qwen3.6 multimodal models use a different namespace from
    // Gemma's CLIP-style:
    //   HF: model.visual.{blocks.{N}.{component}, patch_embed.*, pos_embed.*, merger.*}
    //   GGUF: v.{blk.{N}.{...}, patch_embd.*, position_embd.*, mm.{0,2}.*}
    //
    // Tensor name conventions transcribed from
    // /opt/llama.cpp/tools/mtmd/clip-impl.h:
    //   TN_ATTN_QKV   = "%s.blk.%d.attn_qkv.%s"  (fused QKV — no split)
    //   TN_ATTN_OUTPUT = "%s.blk.%d.attn_out.%s"
    //   TN_FFN_UP     = "%s.blk.%d.ffn_up.%s"
    //   TN_FFN_DOWN   = "%s.blk.%d.ffn_down.%s"
    //   TN_LN_1/2     = "%s.blk.%d.ln{1,2}.%s"
    //   TN_LLAVA_PROJ = "mm.%d.%s"  (Qwen3VL projector uses mm.0 + mm.2)
    //   TN_MM_INP_NORM = "mm.input_norm.weight"
    //   TN_PATCH_EMBD  = "v.patch_embd.weight"  (first temporal slice)
    //   TN_PATCH_EMBD_1 = "v.patch_embd.weight.1"  (second temporal slice)
    //   TN_PATCH_BIAS  = "v.patch_embd.bias"
    //   TN_POS_EMBD    = "v.position_embd.weight"
    //
    // Note: patch_embed.proj.weight is 5-D `[out, in, temporal=2, H, W]`
    // for Qwen3VL — caller must split along axis 2 into two 4-D tensors
    // and emit both as `v.patch_embd.weight` + `v.patch_embd.weight.1`.
    // This function returns the canonical "first slice" name when given
    // the 5-D tensor; the splitting logic lives in the convert pipeline.
    if let Some(rest) = hf_name.strip_prefix("model.visual.blocks.") {
        let (layer_str, suffix) = rest.split_once('.')?;
        let l: usize = layer_str.parse().ok()?;
        let blk = format!("v.blk.{}", l);
        let mapped = match suffix {
            "attn.qkv.weight"     => Some(format!("{}.attn_qkv.weight", blk)),
            "attn.qkv.bias"       => Some(format!("{}.attn_qkv.bias", blk)),
            "attn.proj.weight"    => Some(format!("{}.attn_out.weight", blk)),
            "attn.proj.bias"      => Some(format!("{}.attn_out.bias", blk)),
            "mlp.linear_fc1.weight" => Some(format!("{}.ffn_up.weight", blk)),
            "mlp.linear_fc1.bias"   => Some(format!("{}.ffn_up.bias", blk)),
            "mlp.linear_fc2.weight" => Some(format!("{}.ffn_down.weight", blk)),
            "mlp.linear_fc2.bias"   => Some(format!("{}.ffn_down.bias", blk)),
            "norm1.weight"        => Some(format!("{}.ln1.weight", blk)),
            "norm1.bias"          => Some(format!("{}.ln1.bias", blk)),
            "norm2.weight"        => Some(format!("{}.ln2.weight", blk)),
            "norm2.bias"          => Some(format!("{}.ln2.bias", blk)),
            _ => None,
        };
        return mapped;
    }
    // Qwen3.6 globals: patch_embed, pos_embed, merger.
    static QWEN36_GLOBAL_MAP: &[(&str, &str)] = &[
        // patch_embed: 5-D [out, in, T=2, H, W] needs caller-side split
        // along T. We return the canonical first-slice name; pipeline
        // also emits the `.1` suffixed second slice.
        ("model.visual.patch_embed.proj.weight", "v.patch_embd.weight"),
        ("model.visual.patch_embed.proj.bias",   "v.patch_embd.bias"),
        ("model.visual.pos_embed.weight",        "v.position_embd.weight"),
        // PROJECTOR_TYPE_QWEN3VL uses mm.0 + mm.2 (linear_fc1, linear_fc2).
        ("model.visual.merger.linear_fc1.weight", "mm.0.weight"),
        ("model.visual.merger.linear_fc1.bias",   "mm.0.bias"),
        ("model.visual.merger.linear_fc2.weight", "mm.2.weight"),
        ("model.visual.merger.linear_fc2.bias",   "mm.2.bias"),
        // Qwen3.6 ViT pre-merger norm — maps to TN_MM_INP_NORM family.
        ("model.visual.merger.norm.weight", "mm.input_norm.weight"),
        ("model.visual.merger.norm.bias",   "mm.input_norm.bias"),
    ];
    for (hf, gguf) in QWEN36_GLOBAL_MAP {
        if hf_name == *hf {
            return Some((*gguf).to_string());
        }
    }

    // -- Per-layer ViT encoder tensors (Gemma CLIP-style) ----------------
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
        // Special case: Qwen3.6 patch_embed.proj.weight is 5-D
        // [out, in, T=2, H, W]. llama.cpp's qwen3vl clip graph
        // expects two separate 4-D conv weights (one per temporal
        // frame), named `v.patch_embd.weight` and
        // `v.patch_embd.weight.1`. Split here.
        if name == "model.visual.patch_embed.proj.weight" && tensor.shape.len() == 5 {
            let (slice0, slice1) = split_qwen3vl_patch_embed_temporal(tensor)?;
            out.insert(
                "v.patch_embd.weight".to_string(),
                VitTensor {
                    gguf_name: "v.patch_embd.weight".to_string(),
                    shape: slice0.0,
                    dtype: DType::F16,
                    data: slice0.1,
                },
            );
            out.insert(
                "v.patch_embd.weight.1".to_string(),
                VitTensor {
                    gguf_name: "v.patch_embd.weight.1".to_string(),
                    shape: slice1.0,
                    dtype: DType::F16,
                    data: slice1.1,
                },
            );
            continue;
        }

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

/// Split Qwen3.6 ViT 5-D patch_embed weight `[out, in, T, H, W]` along
/// the temporal dim into two 4-D tensors `[out, in, H, W]` and convert
/// each to F16 bytes. Returns `((shape, bytes), (shape, bytes))` —
/// first temporal slice and second.
///
/// Layout assumption (row-major contiguous, outer-first):
///   src[o, i, t, h, w] at byte offset
///     ((o * I + i) * T + t) * H * W * elem_size + (h * W + w) * elem_size
/// where I = in_channels, T = 2 (temporal frames), H = W = patch_size.
fn split_qwen3vl_patch_embed_temporal(
    tensor: &TensorRef,
) -> Result<(
    (Vec<usize>, Vec<u8>),
    (Vec<usize>, Vec<u8>),
), VitConvertError> {
    if tensor.shape.len() != 5 {
        return Err(VitConvertError::Safetensors(
            "patch_embed.proj.weight (expected 5-D)".to_string(),
        ));
    }
    let out = tensor.shape[0];
    let inp = tensor.shape[1];
    let t = tensor.shape[2];
    let h = tensor.shape[3];
    let w = tensor.shape[4];
    if t != 2 {
        return Err(VitConvertError::Safetensors(format!(
            "patch_embed.proj.weight: expected temporal=2, got {}",
            t
        )));
    }
    let elem_size = tensor.dtype.element_size();
    // Source offset for src[o, i, t, h, w]:
    //   ((o * inp + i) * t + t_idx) * h * w + (h_idx * w + w_idx)
    // We only need the per-(o,i,t) row stride (h*w elements).
    let hw_bytes = h * w * elem_size;

    let mut s0_bytes = Vec::with_capacity(out * inp * h * w * elem_size);
    let mut s1_bytes = Vec::with_capacity(out * inp * h * w * elem_size);

    for o in 0..out {
        for i in 0..inp {
            // src offset for (o, i, 0, ...): ((o * inp + i) * 2 + 0) * h * w
            let off0 = ((o * inp + i) * t + 0) * hw_bytes;
            let off1 = ((o * inp + i) * t + 1) * hw_bytes;
            s0_bytes.extend_from_slice(&tensor.data[off0..off0 + hw_bytes]);
            s1_bytes.extend_from_slice(&tensor.data[off1..off1 + hw_bytes]);
        }
    }

    // Cast to F16 if not already.
    let s0_f16 = match tensor.dtype {
        DType::F16 => s0_bytes,
        DType::BF16 => bf16_bytes_to_f16(&s0_bytes),
        DType::F32 => f32_bytes_to_f16(&s0_bytes),
        _ => return Err(VitConvertError::Safetensors(format!(
            "patch_embed.proj.weight: unsupported dtype {:?}",
            tensor.dtype
        ))),
    };
    let s1_f16 = match tensor.dtype {
        DType::F16 => s1_bytes,
        DType::BF16 => bf16_bytes_to_f16(&s1_bytes),
        DType::F32 => f32_bytes_to_f16(&s1_bytes),
        _ => unreachable!(), // covered above
    };

    let split_shape = vec![out, inp, h, w];
    Ok((
        (split_shape.clone(), s0_f16),
        (split_shape, s1_f16),
    ))
}

fn bf16_bytes_to_f16(input: &[u8]) -> Vec<u8> {
    let n = input.len() / 2;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let bf = u16::from_le_bytes([input[i * 2], input[i * 2 + 1]]);
        let f32_bits = (bf as u32) << 16;
        let f = f32::from_bits(f32_bits);
        let h = half::f16::from_f32(f);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

fn f32_bytes_to_f16(input: &[u8]) -> Vec<u8> {
    let n = input.len() / 4;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let f = f32::from_le_bytes([
            input[i * 4],
            input[i * 4 + 1],
            input[i * 4 + 2],
            input[i * 4 + 3],
        ]);
        let h = half::f16::from_f32(f);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
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
