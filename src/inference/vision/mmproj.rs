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

/// Supported architectural profile for an mmproj GGUF. Different
/// producers (SigLIP, CLIP, Gemma 4 vision tower) carry different
/// tensor-name conventions. The profile is auto-detected from the
/// actual tensor set at startup; forward-pass dispatch branches on it.
///
/// Detection rule (order matters — first match wins):
///   - `Gemma4Siglip` — per-block has `ln1.weight`, `ln2.weight`, and
///     `post_ffw_norm.weight` (Gemma 4's dual-LN SigLIP variant).
///   - `ClipClassic`  — per-block has `attn_norm.weight` (llama.cpp's
///     default CLIP-style writer).
///   - `Unknown`      — neither pattern found. Forward pass not supported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchProfile {
    Gemma4Siglip,
    ClipClassic,
    Unknown,
}

impl ArchProfile {
    pub fn as_str(&self) -> &'static str {
        match self {
            ArchProfile::Gemma4Siglip => "gemma4_siglip",
            ArchProfile::ClipClassic => "clip_classic",
            ArchProfile::Unknown => "unknown",
        }
    }
    pub fn is_supported(&self) -> bool {
        !matches!(self, ArchProfile::Unknown)
    }
}

/// Detect the mmproj's tensor-naming profile from a tensor-name list.
/// Used by startup validation + forward-pass dispatch.
///
/// This is cheap: a ~10-element probe over the block-0 tensor set.
/// Returns `Unknown` when the file has neither marker, which the
/// validator maps to a 400 `no_mmproj_loaded`-style error at request time.
pub fn detect_arch_profile(actual_names: &[&str]) -> ArchProfile {
    let set: std::collections::HashSet<&str> = actual_names.iter().copied().collect();
    if set.contains("v.blk.0.ln1.weight")
        && set.contains("v.blk.0.ln2.weight")
        && set.contains("v.blk.0.post_ffw_norm.weight")
    {
        return ArchProfile::Gemma4Siglip;
    }
    if set.contains("v.blk.0.attn_norm.weight") {
        return ArchProfile::ClipClassic;
    }
    ArchProfile::Unknown
}

/// Minimal, arch-agnostic tensor-set sanity check for an mmproj GGUF.
///
/// Verifies only the tensors EVERY ViT mmproj must carry regardless of
/// the underlying arch:
///   - `v.patch_embd.weight` (pixel → patch-embedding)
///   - `v.position_embd.weight` (learned position encoding)
///   - `v.blk.0.attn_q.weight` + `attn_k.weight` + `attn_v.weight` +
///     `attn_output.weight` (block 0's QKV + output projection)
///   - at least one of `mm.0.weight` / `mm.2.weight` (the projector head)
///   - per `MmprojConfig.num_hidden_layers`, the same QKV+output set for
///     every block (catches truncated files)
///
/// Does NOT require arch-specific tensors like `attn_norm.weight` or
/// `ln1.weight` — those are detected via `ArchProfile` separately and
/// drive forward-pass dispatch, not boot validation.
///
/// Unsupported projector types (`Resampler`, `Other(_)`) still fail
/// the check outright — forward pass can't run regardless of tensor
/// completeness.
///
/// Missing tensors batched into one error listing up to 10 names so a
/// producer bug doesn't become one-at-a-time whack-a-mole on restart.
pub fn validate_tensor_set(cfg: &MmprojConfig, actual_names: &[&str]) -> Result<()> {
    if !cfg.projector.is_supported() {
        return Err(anyhow!(
            "mmproj projector type '{}' is not yet supported by hf2q's ViT \
             forward pass (only 'mlp' is). No forward path will succeed for \
             this file.",
            cfg.projector.as_str()
        ));
    }

    let actual_set: std::collections::HashSet<&str> = actual_names.iter().copied().collect();

    // Universal tensors (arch-agnostic).
    let mut required: Vec<String> = vec![
        TENSOR_PATCH_EMBD.to_string(),
        TENSOR_POS_EMBD.to_string(),
    ];
    // Per-block QKV + output (present in both CLIP + Gemma 4).
    for layer_idx in 0..cfg.num_hidden_layers as usize {
        for suffix in ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"] {
            required.push(vit_layer_tensor(layer_idx, suffix));
        }
    }

    let mut missing: Vec<String> = required
        .iter()
        .filter(|name| !actual_set.contains(name.as_str()))
        .cloned()
        .collect();

    // Projector: accept EITHER mm.0.weight alone (Gemma 4 single-linear)
    // OR the CLIP 2-layer MLP pair. Absence of both is a missing projector.
    if !actual_set.contains(TENSOR_MM_0_WEIGHT) && !actual_set.contains(TENSOR_MM_2_WEIGHT) {
        missing.push(format!(
            "{} (or {})",
            TENSOR_MM_0_WEIGHT, TENSOR_MM_2_WEIGHT
        ));
    }

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
    // validate_tensor_set + detect_arch_profile (iter 30 + iter 31 rewrite
    // against real Gemma 4 mmproj data)
    // -----------------------------------------------------------------------

    fn mlp_cfg(num_layers: u32) -> MmprojConfig {
        MmprojConfig {
            image_size: 224,
            patch_size: 16,
            num_patches_side: 14,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: num_layers,
            layer_norm_eps: 1e-6,
            projector: ProjectorType::Mlp,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        }
    }

    /// Build the arch-agnostic minimum tensor set for a given num_layers:
    /// patch_embd + pos_embd + per-block QKV+output + mm.0.weight.
    fn minimum_tensor_names(num_layers: u32) -> Vec<String> {
        let mut names = vec![
            TENSOR_PATCH_EMBD.to_string(),
            TENSOR_POS_EMBD.to_string(),
            TENSOR_MM_0_WEIGHT.to_string(),
        ];
        for layer_idx in 0..num_layers as usize {
            for suffix in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
            ] {
                names.push(vit_layer_tensor(layer_idx, suffix));
            }
        }
        names
    }

    #[test]
    fn validate_tensor_set_ok_when_minimum_present() {
        let cfg = mlp_cfg(2);
        let names = minimum_tensor_names(2);
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        validate_tensor_set(&cfg, &actual).expect("minimum set should pass");
    }

    #[test]
    fn validate_tensor_set_ok_with_mm_2_instead_of_mm_0() {
        // CLIP classic mmproj has mm.0 AND mm.2. If only mm.2 is present,
        // we still accept — some producers might only ship the 2nd layer.
        let cfg = mlp_cfg(1);
        let mut names = minimum_tensor_names(1);
        // Remove mm.0.weight; add mm.2.weight.
        names.retain(|n| n != TENSOR_MM_0_WEIGHT);
        names.push(TENSOR_MM_2_WEIGHT.to_string());
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        validate_tensor_set(&cfg, &actual).expect("mm.2 alone should pass");
    }

    #[test]
    fn validate_tensor_set_ok_with_arch_specific_extras() {
        // Real Gemma 4 mmproj has 13 tensors per block (our minimum has 4).
        // All 9 extras (ln1, ln2, post_ffw_norm, attn_{q,k}_norm, ffn_up,
        // ffn_down, ffn_gate, ffn_norm) are arch-specific and tolerated.
        let cfg = mlp_cfg(1);
        let mut names = minimum_tensor_names(1);
        for suffix in [
            "ln1.weight",
            "ln2.weight",
            "post_ffw_norm.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "ffn_gate.weight",
            "ffn_norm.weight",
        ] {
            names.push(vit_layer_tensor(0, suffix));
        }
        names.push("v.std_bias".into());
        names.push("v.std_scale".into());
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        validate_tensor_set(&cfg, &actual).expect("gemma-style extras should pass");
    }

    #[test]
    fn validate_tensor_set_flags_missing_patch_embd() {
        let cfg = mlp_cfg(1);
        let mut names = minimum_tensor_names(1);
        names.retain(|n| n != TENSOR_PATCH_EMBD);
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let err = validate_tensor_set(&cfg, &actual).expect_err("should fail");
        let msg = format!("{err}");
        assert!(msg.contains("v.patch_embd.weight"), "got: {msg}");
    }

    #[test]
    fn validate_tensor_set_flags_missing_projector() {
        // Neither mm.0.weight nor mm.2.weight present.
        let cfg = mlp_cfg(1);
        let mut names = minimum_tensor_names(1);
        names.retain(|n| n != TENSOR_MM_0_WEIGHT);
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let err = validate_tensor_set(&cfg, &actual).expect_err("should fail");
        let msg = format!("{err}");
        assert!(msg.contains("mm.0.weight"), "got: {msg}");
    }

    #[test]
    fn validate_tensor_set_flags_missing_whole_block() {
        // 2-layer cfg, only block 0's tensors present.
        let cfg = mlp_cfg(2);
        let names = minimum_tensor_names(1); // only 1 layer's tensors
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let err = validate_tensor_set(&cfg, &actual).expect_err("should fail");
        let msg = format!("{err}");
        // 4 missing: v.blk.1.{attn_q, attn_k, attn_v, attn_output}.weight
        assert!(msg.contains("missing 4 required tensor"), "got: {msg}");
        assert!(msg.contains("v.blk.1.attn_q.weight"), "got: {msg}");
    }

    #[test]
    fn validate_tensor_set_rejects_unsupported_projector() {
        let mut cfg = mlp_cfg(1);
        cfg.projector = ProjectorType::Resampler;
        let err = validate_tensor_set(&cfg, &[]).expect_err("unsupported projector");
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

    #[test]
    fn detect_arch_profile_gemma4_siglip_from_ln1_ln2_post_ffw() {
        let names: Vec<&str> = vec![
            "v.patch_embd.weight",
            "v.blk.0.ln1.weight",
            "v.blk.0.ln2.weight",
            "v.blk.0.post_ffw_norm.weight",
            "v.blk.0.attn_q.weight",
        ];
        assert_eq!(detect_arch_profile(&names), ArchProfile::Gemma4Siglip);
    }

    #[test]
    fn detect_arch_profile_clip_classic_from_attn_norm() {
        let names: Vec<&str> = vec![
            "v.patch_embd.weight",
            "v.blk.0.attn_norm.weight",
            "v.blk.0.attn_q.weight",
        ];
        assert_eq!(detect_arch_profile(&names), ArchProfile::ClipClassic);
    }

    #[test]
    fn detect_arch_profile_unknown_when_neither_marker_present() {
        let names: Vec<&str> = vec!["v.patch_embd.weight", "v.blk.0.attn_q.weight"];
        assert_eq!(detect_arch_profile(&names), ArchProfile::Unknown);
    }

    #[test]
    fn detect_arch_profile_prefers_gemma4_when_both_markers_present() {
        // Pathological case: producer emits both. Gemma 4 dispatch wins
        // (more specific — 3-tensor match beats 1-tensor match).
        let names: Vec<&str> = vec![
            "v.patch_embd.weight",
            "v.blk.0.ln1.weight",
            "v.blk.0.ln2.weight",
            "v.blk.0.post_ffw_norm.weight",
            "v.blk.0.attn_norm.weight",
        ];
        assert_eq!(detect_arch_profile(&names), ArchProfile::Gemma4Siglip);
    }

    #[test]
    fn arch_profile_is_supported_rejects_unknown_only() {
        assert!(ArchProfile::Gemma4Siglip.is_supported());
        assert!(ArchProfile::ClipClassic.is_supported());
        assert!(!ArchProfile::Unknown.is_supported());
    }

    #[test]
    fn arch_profile_as_str_is_snake_case() {
        assert_eq!(ArchProfile::Gemma4Siglip.as_str(), "gemma4_siglip");
        assert_eq!(ArchProfile::ClipClassic.as_str(), "clip_classic");
        assert_eq!(ArchProfile::Unknown.as_str(), "unknown");
    }
}
