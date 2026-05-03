//! ViT `vision_config` parser + validator — ADR-012 Decision 18 §1.
//!
//! Parses `config.json::vision_config` into a typed `VisionConfig`.
//! Required fields for Qwen3.6-27B are validated explicitly with
//! named errors (same pattern as ADR-012 P1's config.json parser).
//!
//! References (read-only spec sources per sovereignty):
//!   - `/opt/llama.cpp/tools/mtmd/clip-model.h` — GGUF metadata key conventions
//!   - `/opt/llama.cpp/tools/mtmd/clip.cpp` — projector type string table
//!   - HF `transformers/src/transformers/models/clip/configuration_clip.py`

use serde_json::Value;

/// Errors produced when parsing `config.json::vision_config`.
#[derive(Debug, thiserror::Error)]
pub enum VisionConfigError {
    #[error("config.json not found")]
    NoConfigJson,

    #[error("config.json i/o error: {0}")]
    Io(String),

    #[error("config.json is not valid JSON: {0}")]
    BadJson(String),

    #[error("vision_config is not a JSON object")]
    VisionConfigNotObject,

    #[error("vision_config.{field} missing or not a {expected_type}")]
    MissingField {
        field: &'static str,
        expected_type: &'static str,
    },

    #[error("vision_config.{field}: invalid value {value}")]
    InvalidField { field: &'static str, value: String },
}

/// Parsed ViT configuration.
///
/// All fields from `clip-model.h`'s `clip.vision.*` metadata. Extensible
/// via `Option<T>` for known-optional keys (e.g. layer_norm_eps has a
/// documented default in llama.cpp).
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    /// ViT hidden (embedding) dim — GGUF `clip.vision.embedding_length`.
    pub hidden_size: u32,
    /// ViT encoder layer count — GGUF `clip.vision.block_count`.
    pub num_hidden_layers: u32,
    /// ViT attention head count — GGUF `clip.vision.attention.head_count`.
    pub num_attention_heads: u32,
    /// Patch edge in pixels — GGUF `clip.vision.patch_size`.
    pub patch_size: u32,
    /// Square input image edge in pixels — GGUF `clip.vision.image_size`.
    pub image_size: u32,
    /// FFN intermediate size — GGUF `clip.vision.feed_forward_length`.
    pub intermediate_size: u32,
    /// LayerNorm epsilon — GGUF `clip.vision.attention.layer_norm_epsilon`.
    pub layer_norm_eps: f32,
    /// Projector type string — GGUF `clip.projector_type`. Usually `"mlp"`.
    pub projector_type: String,
    /// Cross-modal projector output dim (matches text hidden_size).
    /// Optional in HF configs; when absent, derived from the text
    /// hidden_size by the caller.
    pub projection_dim: Option<u32>,
    /// Image normalization mean, `[R, G, B]`. Default `[0.5, 0.5, 0.5]`.
    pub image_mean: [f32; 3],
    /// Image normalization std, `[R, G, B]`. Default `[0.5, 0.5, 0.5]`.
    pub image_std: [f32; 3],
    // ----- Qwen3-VL extensions (iter-224 Wedge-4f) ---------------------
    /// `clip.vision.spatial_merge_size` — Qwen3-VL spatial-merger
    /// degree (typically `2`, giving 2×2 patch fold + 4× token
    /// reduction). `None` for non-Qwen3-VL profiles. Source HF key:
    /// `vision_config.spatial_merge_size`. Source GGUF key:
    /// `clip.vision.spatial_merge_size` (Keys.ClipVision.SPATIAL_MERGE_SIZE
    /// at `/opt/llama.cpp/gguf-py/gguf/constants.py:315`). Writer ref:
    /// `add_vision_spatial_merge_size` at gguf_writer.py:1178-1179.
    pub spatial_merge_size: Option<u32>,
    /// `vision_config.deepstack_visual_indexes` — sorted ascending list
    /// of layer indexes (0-based) whose ViT hidden state is fed into
    /// the LM as DeepStack augmentation. Qwen3-VL-2B-Instruct uses
    /// `[5, 11, 17]`. `None` when the HF config has no
    /// `deepstack_visual_indexes`. Empty `Vec` when the key is present
    /// but no layer is flagged. Emitted to GGUF as a length-`block_count`
    /// `Bool[]` array under `clip.vision.is_deepstack_layers`
    /// (Keys.ClipVision.IS_DEEPSTACK_LAYERS at constants.py:320), the
    /// SAME format llama.cpp's `Qwen3VLVisionModel.set_gguf_parameters`
    /// emits at convert_hf_to_gguf.py:4895-4896:
    ///
    /// ```python
    /// if self.is_deepstack_layers:
    ///     self.gguf_writer.add_vision_is_deepstack_layers(self.is_deepstack_layers)
    /// ```
    ///
    /// where `self.is_deepstack_layers` is a `[False] * num_hidden_layers`
    /// list with `True` set at every index in
    /// `vision_config.deepstack_visual_indexes`.
    pub deepstack_visual_indexes: Option<Vec<u32>>,
    /// `vision_config.temporal_patch_size` — Qwen3-VL's dual-conv
    /// patch stem produces TWO separate patch_embd weights (one per
    /// temporal frame) when `temporal_patch_size == 2`. Defaults to
    /// `2` for Qwen3-VL family per
    /// `/opt/llama.cpp/convert_hf_to_gguf.py:4959-4960`. `None` for
    /// CLIP-classic / Gemma 4 (single-frame patch stem).
    pub temporal_patch_size: Option<u32>,
}

impl VisionConfig {
    /// Parse from a loaded `config.json` root `Value`. Reads
    /// `vision_config` sub-object; top-level `_name_or_path` etc. are
    /// ignored here (handled by `super::compute_slug`).
    pub fn from_hf_config(root: &Value) -> Result<Self, VisionConfigError> {
        let vc = root
            .get("vision_config")
            .ok_or(VisionConfigError::MissingField {
                field: "vision_config",
                expected_type: "object",
            })?;
        if !vc.is_object() {
            return Err(VisionConfigError::VisionConfigNotObject);
        }

        let u32_req = |k: &'static str| -> Result<u32, VisionConfigError> {
            vc.get(k)
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .ok_or(VisionConfigError::MissingField {
                    field: k,
                    expected_type: "u32",
                })
        };
        let f32_def = |k: &str, default: f32| -> f32 {
            vc.get(k).and_then(|v| v.as_f64()).map(|x| x as f32).unwrap_or(default)
        };
        let str_def = |k: &str, default: &str| -> String {
            vc.get(k)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| default.to_string())
        };
        let read_triple = |k: &str, default: [f32; 3]| -> [f32; 3] {
            vc.get(k)
                .and_then(|v| v.as_array())
                .and_then(|arr| {
                    if arr.len() != 3 {
                        return None;
                    }
                    let mut out = [0f32; 3];
                    for (i, v) in arr.iter().enumerate() {
                        out[i] = v.as_f64()? as f32;
                    }
                    Some(out)
                })
                .unwrap_or(default)
        };

        // ADR-012 P9b real-model finding: Qwen3.6 vision_config uses
        // different field names than the Gemma-style schema this parser
        // was originally written for. Accept both forms via fallback:
        //   Gemma            Qwen3.6
        //   num_hidden_layers depth
        //   num_attention_heads  num_heads
        //   image_size       (derived from num_position_embeddings)
        let u32_req_alt = |primary: &'static str, fallback: &'static str|
            -> Result<u32, VisionConfigError>
        {
            if let Some(v) = vc.get(primary).and_then(|v| v.as_u64()) {
                return Ok(v as u32);
            }
            if let Some(v) = vc.get(fallback).and_then(|v| v.as_u64()) {
                return Ok(v as u32);
            }
            Err(VisionConfigError::MissingField {
                field: primary,
                expected_type: "u32",
            })
        };

        let hidden_size = u32_req("hidden_size")?;
        let num_hidden_layers = u32_req_alt("num_hidden_layers", "depth")?;
        let num_attention_heads = u32_req_alt("num_attention_heads", "num_heads")?;
        let patch_size = u32_req("patch_size")?;
        let intermediate_size = u32_req("intermediate_size")?;
        // image_size: prefer explicit; otherwise derive from
        // num_position_embeddings = (image_size/patch_size)^2 for ViT.
        let image_size = if let Some(v) = vc.get("image_size").and_then(|v| v.as_u64()) {
            v as u32
        } else if let Some(npe) = vc.get("num_position_embeddings").and_then(|v| v.as_u64()) {
            let patches_per_side = (npe as f64).sqrt() as u32;
            patches_per_side * patch_size
        } else {
            return Err(VisionConfigError::MissingField {
                field: "image_size",
                expected_type: "u32 (or num_position_embeddings to derive)",
            });
        };

        if patch_size == 0 {
            return Err(VisionConfigError::InvalidField {
                field: "patch_size",
                value: "0".into(),
            });
        }
        if image_size % patch_size != 0 {
            return Err(VisionConfigError::InvalidField {
                field: "image_size",
                value: format!(
                    "{} (not divisible by patch_size {})",
                    image_size, patch_size
                ),
            });
        }

        // ----- Qwen3-VL extension fields -----
        // `vision_config.deepstack_visual_indexes` is a list of layer
        // indexes (0-based). Validate that every entry fits in
        // num_hidden_layers BEFORE we accept the config — a mis-sized
        // index would silently produce a length-mismatched
        // `is_deepstack_layers` GGUF array that the loader rejects at
        // load time anyway, but failing here surfaces the producer bug
        // earlier with the offending index named.
        let deepstack_visual_indexes: Option<Vec<u32>> = match vc.get("deepstack_visual_indexes") {
            None => None,
            Some(v) => match v.as_array() {
                None => {
                    return Err(VisionConfigError::InvalidField {
                        field: "deepstack_visual_indexes",
                        value: format!("expected array, got {}", v),
                    });
                }
                Some(arr) => {
                    let mut indexes: Vec<u32> = Vec::with_capacity(arr.len());
                    for entry in arr {
                        let idx = entry.as_u64().ok_or_else(|| {
                            VisionConfigError::InvalidField {
                                field: "deepstack_visual_indexes",
                                value: format!("non-u64 entry {}", entry),
                            }
                        })? as u32;
                        if idx >= num_hidden_layers {
                            return Err(VisionConfigError::InvalidField {
                                field: "deepstack_visual_indexes",
                                value: format!(
                                    "entry {} >= num_hidden_layers {} (out of range)",
                                    idx, num_hidden_layers
                                ),
                            });
                        }
                        indexes.push(idx);
                    }
                    indexes.sort_unstable();
                    Some(indexes)
                }
            },
        };

        let spatial_merge_size: Option<u32> =
            vc.get("spatial_merge_size").and_then(|v| v.as_u64()).map(|n| n as u32);
        let temporal_patch_size: Option<u32> =
            vc.get("temporal_patch_size").and_then(|v| v.as_u64()).map(|n| n as u32);

        Ok(VisionConfig {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            patch_size,
            image_size,
            intermediate_size,
            layer_norm_eps: f32_def("layer_norm_eps", 1e-6),
            projector_type: str_def("projector_type", "mlp"),
            projection_dim: vc.get("projection_dim").and_then(|v| v.as_u64()).map(|n| n as u32),
            image_mean: read_triple("image_mean", [0.5, 0.5, 0.5]),
            image_std: read_triple("image_std", [0.5, 0.5, 0.5]),
            spatial_merge_size,
            deepstack_visual_indexes,
            temporal_patch_size,
        })
    }

    /// True when this `VisionConfig` describes a Qwen3-VL family vision
    /// tower. Detected via either:
    ///   - `projector_type == "qwen3vl_merger"` (HF native marker), or
    ///   - presence of `deepstack_visual_indexes` (Qwen3-VL is the only
    ///     family that ships this key).
    /// The `OR` accommodates HF configs that omit `projector_type` but
    /// carry the deepstack marker (verified for the 2026-04 snapshot of
    /// `Qwen/Qwen3-VL-2B-Instruct/config.json`'s `vision_config`).
    pub fn is_qwen3vl(&self) -> bool {
        self.projector_type == "qwen3vl_merger"
            || self.deepstack_visual_indexes.is_some()
    }

    /// Build the length-`num_hidden_layers` `Vec<bool>` for emission as
    /// `clip.vision.is_deepstack_layers`. Each entry is `true` iff the
    /// matching layer index appears in `deepstack_visual_indexes`.
    /// Returns `None` when `deepstack_visual_indexes` is `None` (the
    /// non-Qwen3-VL case — the GGUF key is then OMITTED entirely, NOT
    /// emitted as all-false, matching llama.cpp's
    /// `Qwen3VLVisionModel.set_gguf_parameters` at
    /// `/opt/llama.cpp/convert_hf_to_gguf.py:4895-4896`:
    ///
    ///     if self.is_deepstack_layers:
    ///         self.gguf_writer.add_vision_is_deepstack_layers(...)
    pub fn build_is_deepstack_layers(&self) -> Option<Vec<bool>> {
        let indexes = self.deepstack_visual_indexes.as_ref()?;
        let mut bools = vec![false; self.num_hidden_layers as usize];
        for &idx in indexes {
            if (idx as usize) < bools.len() {
                bools[idx as usize] = true;
            }
        }
        Some(bools)
    }

    /// Patches per side (image_size / patch_size).
    pub fn num_patches_side(&self) -> u32 {
        self.image_size / self.patch_size
    }

    /// Total patches per image.
    pub fn num_patches(&self) -> u32 {
        let s = self.num_patches_side();
        s * s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> Value {
        serde_json::json!({
            "vision_config": {
                "hidden_size": 384,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "patch_size": 4,
                "image_size": 32,
                "intermediate_size": 1536,
                "layer_norm_eps": 1e-5,
                "projector_type": "mlp"
            }
        })
    }

    #[test]
    fn parses_valid_vision_config() {
        let vc = VisionConfig::from_hf_config(&valid_config()).unwrap();
        assert_eq!(vc.hidden_size, 384);
        assert_eq!(vc.num_hidden_layers, 4);
        assert_eq!(vc.num_attention_heads, 8);
        assert_eq!(vc.patch_size, 4);
        assert_eq!(vc.image_size, 32);
        assert_eq!(vc.intermediate_size, 1536);
        assert_eq!(vc.projector_type, "mlp");
        assert_eq!(vc.num_patches_side(), 8);
        assert_eq!(vc.num_patches(), 64);
    }

    #[test]
    fn missing_vision_config_is_missing_field() {
        let root = serde_json::json!({});
        let err = VisionConfig::from_hf_config(&root).unwrap_err();
        assert!(matches!(
            err,
            VisionConfigError::MissingField { field: "vision_config", .. }
        ));
    }

    #[test]
    fn non_object_vision_config_rejected() {
        let root = serde_json::json!({"vision_config": "invalid"});
        let err = VisionConfig::from_hf_config(&root).unwrap_err();
        assert!(matches!(err, VisionConfigError::VisionConfigNotObject));
    }

    #[test]
    fn missing_required_field_named_in_error() {
        let mut cfg = valid_config();
        cfg["vision_config"]
            .as_object_mut()
            .unwrap()
            .remove("hidden_size");
        let err = VisionConfig::from_hf_config(&cfg).unwrap_err();
        match err {
            VisionConfigError::MissingField { field, .. } => {
                assert_eq!(field, "hidden_size");
            }
            other => panic!("expected MissingField(hidden_size), got {:?}", other),
        }
    }

    #[test]
    fn patch_size_zero_rejected() {
        let mut cfg = valid_config();
        cfg["vision_config"]
            .as_object_mut()
            .unwrap()
            .insert("patch_size".into(), serde_json::json!(0));
        let err = VisionConfig::from_hf_config(&cfg).unwrap_err();
        assert!(matches!(
            err,
            VisionConfigError::InvalidField { field: "patch_size", .. }
        ));
    }

    #[test]
    fn image_size_not_divisible_by_patch_rejected() {
        let mut cfg = valid_config();
        cfg["vision_config"]
            .as_object_mut()
            .unwrap()
            .insert("image_size".into(), serde_json::json!(31));
        let err = VisionConfig::from_hf_config(&cfg).unwrap_err();
        match err {
            VisionConfigError::InvalidField { field, value } => {
                assert_eq!(field, "image_size");
                assert!(value.contains("31") && value.contains("4"));
            }
            other => panic!("expected InvalidField, got {:?}", other),
        }
    }

    #[test]
    fn defaults_layer_norm_eps_projector_and_mean_std() {
        let mut cfg = valid_config();
        let vc_obj = cfg["vision_config"].as_object_mut().unwrap();
        vc_obj.remove("layer_norm_eps");
        vc_obj.remove("projector_type");
        let parsed = VisionConfig::from_hf_config(&cfg).unwrap();
        assert_eq!(parsed.projector_type, "mlp");
        assert!((parsed.layer_norm_eps - 1e-6).abs() < 1e-9);
        assert_eq!(parsed.image_mean, [0.5, 0.5, 0.5]);
        assert_eq!(parsed.image_std, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn custom_mean_std_triples_honored() {
        let mut cfg = valid_config();
        let vc_obj = cfg["vision_config"].as_object_mut().unwrap();
        vc_obj.insert("image_mean".into(), serde_json::json!([0.48, 0.45, 0.40]));
        vc_obj.insert("image_std".into(), serde_json::json!([0.26, 0.26, 0.27]));
        let parsed = VisionConfig::from_hf_config(&cfg).unwrap();
        assert_eq!(parsed.image_mean, [0.48, 0.45, 0.40]);
        assert_eq!(parsed.image_std, [0.26, 0.26, 0.27]);
    }

    #[test]
    fn projection_dim_optional() {
        let cfg = valid_config();
        let parsed = VisionConfig::from_hf_config(&cfg).unwrap();
        assert_eq!(parsed.projection_dim, None);

        let mut cfg_with = cfg.clone();
        cfg_with["vision_config"]
            .as_object_mut()
            .unwrap()
            .insert("projection_dim".into(), serde_json::json!(2048));
        let parsed2 = VisionConfig::from_hf_config(&cfg_with).unwrap();
        assert_eq!(parsed2.projection_dim, Some(2048));
    }

    // ----- Wedge-4f (iter-224 row 6) — Qwen3-VL extension fields -----

    #[test]
    fn parses_qwen3vl_extension_fields() {
        // Mirrors a real Qwen3-VL-2B-Instruct vision_config snippet.
        let root = serde_json::json!({
            "vision_config": {
                "hidden_size": 384,
                "depth": 24,
                "num_heads": 8,
                "patch_size": 16,
                "image_size": 224,
                "intermediate_size": 1536,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "deepstack_visual_indexes": [5, 11, 17],
                "projector_type": "qwen3vl_merger"
            }
        });
        let vc = VisionConfig::from_hf_config(&root).unwrap();
        assert_eq!(vc.spatial_merge_size, Some(2));
        assert_eq!(vc.temporal_patch_size, Some(2));
        assert_eq!(vc.deepstack_visual_indexes, Some(vec![5, 11, 17]));
        assert_eq!(vc.projector_type, "qwen3vl_merger");
        assert!(vc.is_qwen3vl());
    }

    #[test]
    fn deepstack_indexes_sorted_ascending_when_unsorted_input() {
        // Sovereignty: even if HF source ships indexes out of order
        // (e.g. `[17, 5, 11]`), we sort them so downstream code can
        // rely on `deepstack_visual_indexes[0]` being the earliest
        // flagged layer. Mirrors the loader's `read_deepstack_indexes`
        // which iterates the GGUF Bool[] array in ascending order.
        let root = serde_json::json!({
            "vision_config": {
                "hidden_size": 64,
                "num_hidden_layers": 24,
                "num_attention_heads": 8,
                "patch_size": 16,
                "image_size": 224,
                "intermediate_size": 256,
                "deepstack_visual_indexes": [17, 5, 11]
            }
        });
        let vc = VisionConfig::from_hf_config(&root).unwrap();
        assert_eq!(vc.deepstack_visual_indexes, Some(vec![5, 11, 17]));
    }

    #[test]
    fn deepstack_index_out_of_range_rejected() {
        // Defense in depth — reject invalid index now so we don't
        // emit a length-mismatched is_deepstack_layers GGUF array
        // that the loader would reject anyway.
        let root = serde_json::json!({
            "vision_config": {
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "patch_size": 16,
                "image_size": 224,
                "intermediate_size": 256,
                "deepstack_visual_indexes": [2, 5]   // 5 >= 4
            }
        });
        let err = VisionConfig::from_hf_config(&root).unwrap_err();
        match err {
            VisionConfigError::InvalidField { field, value } => {
                assert_eq!(field, "deepstack_visual_indexes");
                assert!(value.contains("5") && value.contains("4"));
            }
            other => panic!("expected InvalidField, got {:?}", other),
        }
    }

    #[test]
    fn build_is_deepstack_layers_emits_correct_bool_array() {
        // [5, 11, 17] in a 24-layer ViT → length-24 array with `true`
        // at exactly those three positions.
        let vc = VisionConfig {
            hidden_size: 64,
            num_hidden_layers: 24,
            num_attention_heads: 8,
            patch_size: 16,
            image_size: 224,
            intermediate_size: 256,
            layer_norm_eps: 1e-6,
            projector_type: "qwen3vl_merger".into(),
            projection_dim: None,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            spatial_merge_size: Some(2),
            deepstack_visual_indexes: Some(vec![5, 11, 17]),
            temporal_patch_size: Some(2),
        };
        let bools = vc.build_is_deepstack_layers().expect("Some(bools)");
        assert_eq!(bools.len(), 24);
        let true_positions: Vec<usize> = bools.iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();
        assert_eq!(true_positions, vec![5, 11, 17]);
    }

    #[test]
    fn build_is_deepstack_layers_returns_none_when_indexes_none() {
        let vc = VisionConfig {
            hidden_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            patch_size: 16,
            image_size: 224,
            intermediate_size: 256,
            layer_norm_eps: 1e-6,
            projector_type: "mlp".into(),
            projection_dim: None,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            spatial_merge_size: None,
            deepstack_visual_indexes: None,
            temporal_patch_size: None,
        };
        assert!(vc.build_is_deepstack_layers().is_none());
        assert!(!vc.is_qwen3vl());
    }

    #[test]
    fn is_qwen3vl_detects_via_projector_type() {
        // Family detection via projector_type alone (no deepstack).
        let root = serde_json::json!({
            "vision_config": {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "patch_size": 16,
                "image_size": 224,
                "intermediate_size": 256,
                "projector_type": "qwen3vl_merger"
            }
        });
        let vc = VisionConfig::from_hf_config(&root).unwrap();
        assert!(vc.is_qwen3vl());
    }

    #[test]
    fn is_qwen3vl_detects_via_deepstack_indexes_alone() {
        // A vision_config that omits projector_type but ships
        // deepstack_visual_indexes is unambiguously Qwen3-VL.
        let root = serde_json::json!({
            "vision_config": {
                "hidden_size": 64,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "patch_size": 16,
                "image_size": 224,
                "intermediate_size": 256,
                "deepstack_visual_indexes": [3, 5]
            }
        });
        let vc = VisionConfig::from_hf_config(&root).unwrap();
        assert!(vc.is_qwen3vl());
    }
}
