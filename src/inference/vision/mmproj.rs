//! mmproj (multimodal projector) GGUF metadata probe — ADR-005 Phase 2c,
//! Task #14.
//!
//! The mmproj GGUF file holds the vision-tower weights (ViT + projector
//! from ViT hidden dim → text decoder hidden dim). hf2q produces mmproj
//! files from safetensors (quantizing the vision tower) AND accepts
//! user-provided mmproj files. This module parses the GGUF metadata
//! header into a typed `MmprojConfig` so the forward-pass iter can
//! build the ViT graph from validated config.
//!
//! # GGUF metadata keys
//!
//! llama.cpp's mmproj GGUF writer uses the `clip.*` namespace (mmproj
//! inherits CLIP's vision-tower semantics). Key conventions:
//!
//!   - `general.architecture = "clip"`  (enforced by this parser)
//!   - `clip.vision.image_size`         → 224, 336, 518, 896, ...
//!   - `clip.vision.patch_size`         → ViT patch dim (typical 14/16)
//!   - `clip.vision.embedding_length`   → ViT hidden size
//!   - `clip.vision.feed_forward_length`→ ViT FFN intermediate
//!   - `clip.vision.attention.head_count` → ViT heads
//!   - `clip.vision.block_count`        → ViT layers
//!   - `clip.vision.attention.layer_norm_epsilon`
//!   - `clip.projector_type`            → "mlp" | "resampler" | ...
//!   - `clip.vision.mean[R,G,B]` + `clip.vision.std[R,G,B]` (arrays)
//!
//! # What this iter does NOT do
//!
//!   - Loading weight tensors into mlx-native buffers (Phase 2c ViT
//!     forward iter). The parser validates the HEADER only.
//!   - Quantizing a safetensors vision tower → mmproj GGUF (hf2q convert
//!     path; separate iter).

#![allow(dead_code)]

use anyhow::{anyhow, Result};
use mlx_native::gguf::{GgufFile, MetadataValue};

/// GGUF architecture identifier for mmproj files.
pub const ARCH_CLIP: &str = "clip";

/// Which projector shape the mmproj uses. Determines the final dim of
/// the ViT hidden → text-embed transform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectorType {
    /// Dense 2-layer MLP (Gemma 4 default).
    Mlp,
    /// Perceiver-style resampler (uncommon; present in some VLMs).
    Resampler,
    /// Other / unknown. Forward pass can only handle MLP today; this
    /// variant exists so probes/`/v1/models` don't fail on unfamiliar
    /// mmproj files — they can be listed but not served.
    Other(String),
}

impl ProjectorType {
    pub fn from_str_gguf(s: &str) -> Self {
        match s {
            "mlp" => ProjectorType::Mlp,
            "resampler" => ProjectorType::Resampler,
            other => ProjectorType::Other(other.to_string()),
        }
    }
    pub fn as_str(&self) -> &str {
        match self {
            ProjectorType::Mlp => "mlp",
            ProjectorType::Resampler => "resampler",
            ProjectorType::Other(s) => s.as_str(),
        }
    }
    pub fn is_supported(&self) -> bool {
        matches!(self, ProjectorType::Mlp)
    }
}

/// Parsed mmproj vision-tower configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct MmprojConfig {
    /// Input image resolution (square; `image_size × image_size`).
    pub image_size: u32,
    /// ViT patch edge (patches are `patch × patch` squares).
    pub patch_size: u32,
    /// Number of patches per side = image_size / patch_size.
    pub num_patches_side: u32,
    /// ViT hidden dim.
    pub hidden_size: u32,
    /// ViT FFN intermediate dim.
    pub intermediate_size: u32,
    /// ViT attention head count.
    pub num_attention_heads: u32,
    /// ViT encoder layers.
    pub num_hidden_layers: u32,
    /// LayerNorm epsilon (applied inside the ViT).
    pub layer_norm_eps: f32,
    /// Projector shape. Only `Mlp` is supported for forward pass today;
    /// other variants flag the file as listable-but-not-servable.
    pub projector: ProjectorType,
    /// Per-channel normalization mean `[R, G, B]` (pixel / 255 scale).
    /// Falls back to `[0.5, 0.5, 0.5]` (Gemma 4 default) when absent.
    pub image_mean: [f32; 3],
    /// Per-channel std `[R, G, B]`.
    pub image_std: [f32; 3],
}

impl MmprojConfig {
    /// Parse from a GGUF file's metadata header. Strict: fails loud when
    /// any required field is missing rather than defaulting to potentially-
    /// wrong values. Image mean/std fall back to Gemma 4's `[0.5, 0.5, 0.5]`
    /// when absent since that's the only mmproj convention hf2q writes;
    /// consumer mmproj files override via `clip.vision.{mean,std}` arrays.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| anyhow!("mmproj GGUF missing general.architecture"))?;
        if arch != ARCH_CLIP {
            return Err(anyhow!(
                "mmproj GGUF architecture is '{}', expected 'clip'",
                arch
            ));
        }
        let u32_key = |k: &str| -> Result<u32> {
            gguf.metadata_u32(k)
                .ok_or_else(|| anyhow!("mmproj GGUF missing u32 '{}'", k))
        };
        let f32_key_default = |k: &str, default: f32| -> f32 {
            gguf.metadata_f32(k).unwrap_or(default)
        };

        let image_size = u32_key("clip.vision.image_size")?;
        let patch_size = u32_key("clip.vision.patch_size")?;
        if patch_size == 0 {
            return Err(anyhow!("clip.vision.patch_size = 0"));
        }
        if image_size % patch_size != 0 {
            return Err(anyhow!(
                "image_size {} must be divisible by patch_size {}",
                image_size,
                patch_size
            ));
        }
        let num_patches_side = image_size / patch_size;
        let hidden_size = u32_key("clip.vision.embedding_length")?;
        let intermediate_size = u32_key("clip.vision.feed_forward_length")?;
        let num_attention_heads = u32_key("clip.vision.attention.head_count")?;
        let num_hidden_layers = u32_key("clip.vision.block_count")?;
        let layer_norm_eps = f32_key_default("clip.vision.attention.layer_norm_epsilon", 1e-6);
        let projector_str = gguf
            .metadata_string("clip.projector_type")
            .unwrap_or("mlp");
        let projector = ProjectorType::from_str_gguf(projector_str);

        // Optional mean/std arrays. llama.cpp writes each as `Array(Float32
        // × 3)`. Fall back to Gemma 4's [0.5, 0.5, 0.5] when absent.
        let read_triple = |key: &str, default: [f32; 3]| -> [f32; 3] {
            match gguf.metadata(key) {
                Some(MetadataValue::Array(arr)) if arr.len() == 3 => {
                    let mut out = [0f32; 3];
                    for (i, v) in arr.iter().enumerate() {
                        if let Some(f) = v.as_f32() {
                            out[i] = f;
                        } else {
                            return default;
                        }
                    }
                    out
                }
                _ => default,
            }
        };
        let image_mean = read_triple("clip.vision.image_mean", [0.5, 0.5, 0.5]);
        let image_std = read_triple("clip.vision.image_std", [0.5, 0.5, 0.5]);

        Ok(MmprojConfig {
            image_size,
            patch_size,
            num_patches_side,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_hidden_layers,
            layer_norm_eps,
            projector,
            image_mean,
            image_std,
        })
    }

    /// Patches-per-image = `num_patches_side ** 2`. The ViT forward pass
    /// produces one hidden-state vector per patch (plus optionally a
    /// [CLS] token — Gemma 4's vision tower does not use one).
    pub fn num_patches(&self) -> u32 {
        self.num_patches_side * self.num_patches_side
    }

    /// Convert to a `PreprocessConfig` for the CPU preprocessor
    /// (`crate::inference::vision::preprocess`). Produces the exact
    /// mean/std/target_size the ViT expects.
    pub fn preprocess_config(&self) -> super::preprocess::PreprocessConfig {
        super::preprocess::PreprocessConfig {
            target_size: self.image_size,
            mean: self.image_mean,
            std: self.image_std,
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor-name table (llama.cpp mmproj convention)
// ---------------------------------------------------------------------------
//
// Standard llama.cpp mmproj tensor naming uses a `v.` prefix:
//   - v.patch_embd.weight          — Conv2d patch stem weight [hidden, 3, patch, patch]
//   - v.patch_embd.bias            — optional
//   - v.position_embd.weight       — [num_patches (+1 cls), hidden]
//   - v.blk.{N}.attn_q.{weight,bias}
//   - v.blk.{N}.attn_k.{weight,bias}
//   - v.blk.{N}.attn_v.{weight,bias}
//   - v.blk.{N}.attn_output.{weight,bias}
//   - v.blk.{N}.attn_norm.{weight,bias}
//   - v.blk.{N}.ffn_up.{weight,bias}
//   - v.blk.{N}.ffn_down.{weight,bias}
//   - v.blk.{N}.ffn_norm.{weight,bias}
//   - v.post_ln.{weight,bias}      — final layer norm
//   - mm.0.{weight,bias}           — MLP projector layer 0
//   - mm.2.{weight,bias}           — MLP projector layer 2 (layer 1 is GELU)

/// Patch-embedding convolution (turn pixels into patch embeddings).
pub const TENSOR_PATCH_EMBD: &str = "v.patch_embd.weight";
/// Optional patch-embedding bias.
pub const TENSOR_PATCH_EMBD_BIAS: &str = "v.patch_embd.bias";
/// Learned positional embeddings over patches.
pub const TENSOR_POS_EMBD: &str = "v.position_embd.weight";
/// Final LayerNorm after the last ViT block.
pub const TENSOR_POST_LN_WEIGHT: &str = "v.post_ln.weight";
pub const TENSOR_POST_LN_BIAS: &str = "v.post_ln.bias";
/// MLP projector first linear.
pub const TENSOR_MM_0_WEIGHT: &str = "mm.0.weight";
pub const TENSOR_MM_0_BIAS: &str = "mm.0.bias";
/// MLP projector second linear.
pub const TENSOR_MM_2_WEIGHT: &str = "mm.2.weight";
pub const TENSOR_MM_2_BIAS: &str = "mm.2.bias";

/// Per-layer tensor name helper.
pub fn vit_layer_tensor(layer_idx: usize, suffix: &str) -> String {
    format!("v.blk.{}.{}", layer_idx, suffix)
}

/// Per-block suffixes every ViT transformer layer must carry. Biases
/// are NOT in this list because Gemma 4's SigLIP-derived tower omits
/// them; bias tensors are accepted-if-present but never required.
const BLOCK_REQUIRED_SUFFIXES: &[&str] = &[
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "attn_norm.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "ffn_norm.weight",
];

/// Build the list of tensor names a fully-featured mmproj GGUF MUST
/// contain given its parsed `MmprojConfig`. Missing any of these is a
/// producer bug or a truncated file — the server must fail-fast rather
/// than hitting a NotFound mid-forward-pass.
///
/// Covers the ViT stem (`v.patch_embd.weight`, `v.position_embd.weight`),
/// every ViT block's required weights (see `BLOCK_REQUIRED_SUFFIXES`),
/// the ViT post-LN (`v.post_ln.weight`), and the projector head.
/// Projector tensors vary by type:
///
///   - `Mlp`       → `mm.0.weight`, `mm.2.weight`.
///   - `Resampler` → no required-tensor list (server refuses to run —
///     unsupported). Empty projector section.
///   - `Other(_)`  → same as Resampler.
///
/// Bias tensors (`v.patch_embd.bias`, `v.post_ln.bias`, per-block
/// `*.bias`, `mm.0.bias`) are intentionally NOT listed as required —
/// Gemma 4's SigLIP vision tower omits all biases.
pub fn expected_tensor_names(cfg: &MmprojConfig) -> Vec<String> {
    let mut out = Vec::with_capacity(
        4 + (cfg.num_hidden_layers as usize) * BLOCK_REQUIRED_SUFFIXES.len() + 2,
    );
    // --- ViT stem ---
    out.push(TENSOR_PATCH_EMBD.to_string());
    out.push(TENSOR_POS_EMBD.to_string());
    out.push(TENSOR_POST_LN_WEIGHT.to_string());
    // --- ViT transformer blocks ---
    for layer_idx in 0..cfg.num_hidden_layers as usize {
        for suffix in BLOCK_REQUIRED_SUFFIXES {
            out.push(vit_layer_tensor(layer_idx, suffix));
        }
    }
    // --- Projector head ---
    match &cfg.projector {
        ProjectorType::Mlp => {
            out.push(TENSOR_MM_0_WEIGHT.to_string());
            out.push(TENSOR_MM_2_WEIGHT.to_string());
        }
        ProjectorType::Resampler | ProjectorType::Other(_) => {
            // No required-tensor list for unsupported projectors;
            // validate_tensor_set flags the unsupported case explicitly
            // so we never advertise a vision capability we can't back.
        }
    }
    out
}

/// Validate that `actual_names` (from `GgufFile::tensor_names()`)
/// contains every tensor in `expected_tensor_names(cfg)`. Unsupported
/// projector types fail with a dedicated error before the tensor-name
/// walk so the operator sees the actual remedy.
///
/// Missing tensors are batched into one error listing up to 10 names —
/// prevents one-at-a-time whack-a-mole on restart when a producer bug
/// drops a whole block's weights.
pub fn validate_tensor_set(cfg: &MmprojConfig, actual_names: &[&str]) -> Result<()> {
    if !cfg.projector.is_supported() {
        return Err(anyhow!(
            "mmproj projector type '{}' is not yet supported by hf2q's ViT \
             forward pass (only 'mlp' is). No forward path will succeed for \
             this file.",
            cfg.projector.as_str()
        ));
    }

    let expected = expected_tensor_names(cfg);
    let actual_set: std::collections::HashSet<&str> = actual_names.iter().copied().collect();
    let mut missing: Vec<String> = expected
        .iter()
        .filter(|name| !actual_set.contains(name.as_str()))
        .cloned()
        .collect();

    if missing.is_empty() {
        return Ok(());
    }

    let total_missing = missing.len();
    missing.truncate(10);
    let more = if total_missing > 10 {
        format!(" (+ {} more)", total_missing - 10)
    } else {
        String::new()
    };
    Err(anyhow!(
        "mmproj GGUF is missing {} required tensor(s): {}{}",
        total_missing,
        missing.join(", "),
        more
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projector_type_parse_and_round_trip() {
        assert_eq!(ProjectorType::from_str_gguf("mlp"), ProjectorType::Mlp);
        assert_eq!(
            ProjectorType::from_str_gguf("resampler"),
            ProjectorType::Resampler
        );
        match ProjectorType::from_str_gguf("q-former") {
            ProjectorType::Other(s) => assert_eq!(s, "q-former"),
            other => panic!("expected Other, got {:?}", other),
        }
    }

    #[test]
    fn projector_supported_only_for_mlp() {
        assert!(ProjectorType::Mlp.is_supported());
        assert!(!ProjectorType::Resampler.is_supported());
        assert!(!ProjectorType::Other("x".into()).is_supported());
    }

    #[test]
    fn projector_as_str_matches_input() {
        assert_eq!(ProjectorType::Mlp.as_str(), "mlp");
        assert_eq!(ProjectorType::Resampler.as_str(), "resampler");
        assert_eq!(
            ProjectorType::Other("q-former".into()).as_str(),
            "q-former"
        );
    }

    #[test]
    fn vit_layer_tensor_formats_blk_prefix() {
        assert_eq!(
            vit_layer_tensor(0, "attn_q.weight"),
            "v.blk.0.attn_q.weight"
        );
        assert_eq!(
            vit_layer_tensor(23, "ffn_down.bias"),
            "v.blk.23.ffn_down.bias"
        );
    }

    #[test]
    fn tensor_constants_lock_llama_cpp_convention() {
        // Changes here are silent compat breaks with llama.cpp mmproj
        // files. Update only in lockstep with a writer change.
        assert_eq!(TENSOR_PATCH_EMBD, "v.patch_embd.weight");
        assert_eq!(TENSOR_POS_EMBD, "v.position_embd.weight");
        assert_eq!(TENSOR_POST_LN_WEIGHT, "v.post_ln.weight");
        assert_eq!(TENSOR_MM_0_WEIGHT, "mm.0.weight");
        assert_eq!(TENSOR_MM_2_WEIGHT, "mm.2.weight");
    }

    #[test]
    fn preprocess_config_conversion() {
        let mm = MmprojConfig {
            image_size: 896,
            patch_size: 14,
            num_patches_side: 64,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            layer_norm_eps: 1e-6,
            projector: ProjectorType::Mlp,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        };
        let p = mm.preprocess_config();
        assert_eq!(p.target_size, 896);
        assert_eq!(p.mean, [0.5, 0.5, 0.5]);
        assert_eq!(p.std, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn num_patches_squares_side_count() {
        let mm = MmprojConfig {
            image_size: 224,
            patch_size: 14,
            num_patches_side: 16,
            hidden_size: 768,
            intermediate_size: 3072,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            layer_norm_eps: 1e-6,
            projector: ProjectorType::Mlp,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        };
        assert_eq!(mm.num_patches(), 256);
    }

    // -----------------------------------------------------------------------
    // expected_tensor_names + validate_tensor_set (iter 30)
    // -----------------------------------------------------------------------

    fn mlp_cfg(num_layers: u32) -> MmprojConfig {
        MmprojConfig {
            image_size: 224,
            patch_size: 14,
            num_patches_side: 16,
            hidden_size: 768,
            intermediate_size: 3072,
            num_attention_heads: 12,
            num_hidden_layers: num_layers,
            layer_norm_eps: 1e-6,
            projector: ProjectorType::Mlp,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        }
    }

    #[test]
    fn expected_tensor_names_mlp_small_is_well_formed() {
        // 2-layer MLP mmproj: 3 stem + 2*8 block + 2 projector = 21 tensors.
        let cfg = mlp_cfg(2);
        let names = expected_tensor_names(&cfg);
        assert_eq!(names.len(), 3 + 2 * 8 + 2);
        // Stem (first 3).
        assert!(names.contains(&TENSOR_PATCH_EMBD.to_string()));
        assert!(names.contains(&TENSOR_POS_EMBD.to_string()));
        assert!(names.contains(&TENSOR_POST_LN_WEIGHT.to_string()));
        // Every block has every required suffix.
        for layer_idx in 0..2 {
            for suffix in BLOCK_REQUIRED_SUFFIXES {
                assert!(
                    names.contains(&vit_layer_tensor(layer_idx, suffix)),
                    "missing v.blk.{}.{}",
                    layer_idx,
                    suffix
                );
            }
        }
        // MLP projector.
        assert!(names.contains(&TENSOR_MM_0_WEIGHT.to_string()));
        assert!(names.contains(&TENSOR_MM_2_WEIGHT.to_string()));
    }

    #[test]
    fn expected_tensor_names_resampler_has_empty_projector_section() {
        // Resampler is unsupported — projector tensors aren't required;
        // validate_tensor_set rejects the config outright.
        let mut cfg = mlp_cfg(1);
        cfg.projector = ProjectorType::Resampler;
        let names = expected_tensor_names(&cfg);
        // Stem (3) + block suffixes (8) = 11, no mm.*.
        assert_eq!(names.len(), 3 + 8);
        assert!(!names.iter().any(|n| n.starts_with("mm.")));
    }

    #[test]
    fn expected_tensor_names_gemma4_production_shape() {
        // Gemma 4 vision tower: 27 layers MLP.
        // Total = 3 stem + 27*8 block + 2 projector = 221 tensors.
        let cfg = mlp_cfg(27);
        let names = expected_tensor_names(&cfg);
        assert_eq!(names.len(), 3 + 27 * 8 + 2);
    }

    #[test]
    fn validate_tensor_set_ok_when_all_present() {
        let cfg = mlp_cfg(1);
        let expected = expected_tensor_names(&cfg);
        let actual: Vec<&str> = expected.iter().map(|s| s.as_str()).collect();
        validate_tensor_set(&cfg, &actual).expect("all present → Ok");
    }

    #[test]
    fn validate_tensor_set_ok_with_extra_tensors() {
        // Producers may add biases + extras; we tolerate unknowns.
        let cfg = mlp_cfg(1);
        let expected = expected_tensor_names(&cfg);
        let mut actual: Vec<&str> = expected.iter().map(|s| s.as_str()).collect();
        actual.push("v.patch_embd.bias"); // optional bias — not required but present
        actual.push("mm.0.bias");
        actual.push("some.future.tensor"); // unknown producer extension
        validate_tensor_set(&cfg, &actual).expect("extras are tolerated");
    }

    #[test]
    fn validate_tensor_set_flags_missing_single_tensor() {
        let cfg = mlp_cfg(1);
        let expected = expected_tensor_names(&cfg);
        // Drop the last one (mm.2.weight) to simulate a truncated producer.
        let actual: Vec<&str> = expected[..expected.len() - 1]
            .iter()
            .map(|s| s.as_str())
            .collect();
        let err = validate_tensor_set(&cfg, &actual).expect_err("should fail");
        let msg = format!("{err}");
        assert!(msg.contains("missing 1 required tensor"), "got: {msg}");
        assert!(msg.contains("mm.2.weight"), "got: {msg}");
    }

    #[test]
    fn validate_tensor_set_batches_many_missing_tensors() {
        // Simulate a producer bug that drops all block tensors.
        let cfg = mlp_cfg(2); // 2 layers × 8 = 16 missing block tensors.
        let stem_and_projector: Vec<String> = vec![
            TENSOR_PATCH_EMBD.to_string(),
            TENSOR_POS_EMBD.to_string(),
            TENSOR_POST_LN_WEIGHT.to_string(),
            TENSOR_MM_0_WEIGHT.to_string(),
            TENSOR_MM_2_WEIGHT.to_string(),
        ];
        let actual: Vec<&str> = stem_and_projector.iter().map(|s| s.as_str()).collect();
        let err = validate_tensor_set(&cfg, &actual).expect_err("should fail");
        let msg = format!("{err}");
        assert!(msg.contains("missing 16 required tensor"), "got: {msg}");
        // 10 shown + overflow marker.
        assert!(msg.contains("+ 6 more"), "got: {msg}");
    }

    #[test]
    fn validate_tensor_set_rejects_unsupported_projector_before_tensor_walk() {
        let mut cfg = mlp_cfg(1);
        cfg.projector = ProjectorType::Resampler;
        // Even with EVERY possible tensor name, unsupported projector
        // short-circuits before the walk so operators see the actual remedy.
        let actual: Vec<&str> = vec![];
        let err = validate_tensor_set(&cfg, &actual).expect_err("unsupported projector");
        let msg = format!("{err}");
        assert!(msg.contains("'resampler' is not yet supported"), "got: {msg}");
    }

    #[test]
    fn validate_tensor_set_rejects_other_projector_with_name_echoed() {
        let mut cfg = mlp_cfg(1);
        cfg.projector = ProjectorType::Other("q-former".into());
        let err = validate_tensor_set(&cfg, &[]).expect_err("unsupported projector");
        let msg = format!("{err}");
        assert!(msg.contains("'q-former'"), "got: {msg}");
    }
}
