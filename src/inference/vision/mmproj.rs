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
//! ## Qwen3-VL extensions (iter-224 Wedge-4b, ADR-005)
//!
//! Qwen3-VL ships its mmproj using the `clip.*` namespace too but adds:
//!
//!   - `clip.projector_type = "qwen3vl_merger"`           — selects
//!     `ProjectorType::Qwen3VlMerger` (`is_supported()=false` until
//!     Wedge-4c lands the ViT forward).
//!   - `clip.vision.spatial_merge_size`                   — typically
//!     `2` (2×2 patch merger → 4× token reduction).
//!   - `clip.vision.projection_dim`                       — the LM
//!     hidden size the projector targets (e.g. `2048` for Qwen3-VL-2B).
//!   - `clip.vision.is_deepstack_layers`                  — `Bool[N]`
//!     of length `block_count`; `true` for layers whose ViT hidden
//!     state is fed into the LM as DeepStack augmentation. Qwen3-VL-2B
//!     sets `[5, 11, 17]`. The convert script encodes the per-layer
//!     bool array; we present it as a sorted `Vec<u32>` of true-indexes
//!     for consumer-friendliness.
//!   - Per-flagged-layer `v.deepstack.{N}.{norm,fc1,fc2}.{weight,bias}`
//!     tensors (TN_DEEPSTACK_NORM/FC1/FC2; clip-impl.h:117-119). The
//!     primary projector is still the CLIP-classic two-layer MLP at
//!     `mm.0.weight` + `mm.2.weight` (clip.cpp:1844-1850, identical
//!     load shape to PROJECTOR_TYPE_MLP).
//!
//! Detection rule (`detect_arch_profile`): for tensor-only callers, the
//! presence of `v.deepstack.0.fc1.weight` is the canonical Qwen3-VL
//! marker (DeepStack tensors are unique to the qwen3vl projector
//! family). For projector-aware callers,
//! `detect_arch_profile_with_projector` short-circuits on
//! `ProjectorType::Qwen3VlMerger` regardless of tensor enumeration.
//!
//! Detection uses `v.deepstack.0.fc1.weight` (a tensor name) rather
//! than a metadata flag because llama.cpp does NOT define a dedicated
//! `clip.has_qwen3vl_merger` GGUF key (verified in
//! `/opt/llama.cpp/tools/mtmd/clip-impl.h` 2026-05-01) — only the
//! projector-type string and the per-layer DeepStack tensors uniquely
//! identify a Qwen3-VL mmproj. The Worker W audit
//! (`wedge4-qwen35-vision-plan.md`) initially named a
//! `clip.has_qwen3vl_merger` flag; this implementation drops that
//! assumption in favor of the actual upstream signal set.
//!
//! # What this iter does NOT do
//!
//!   - Loading weight tensors into mlx-native buffers (Phase 2c ViT
//!     forward iter). The parser validates the HEADER only.
//!   - Quantizing a safetensors vision tower → mmproj GGUF (hf2q convert
//!     path; separate iter).
//!   - Running the Qwen3-VL ViT forward pass — Wedge-4c's job; this
//!     module's `ProjectorType::Qwen3VlMerger.is_supported()` returns
//!     `false` so listing-without-serving works but `serve --mmproj`
//!     rejects the file at startup. Wedge-4c flips that bit when the
//!     `compute_vision_embeddings_gpu_qwen3vl` path lands.

#![allow(dead_code)]

use anyhow::{anyhow, Result};
use mlx_native::gguf::{GgufFile, MetadataValue};

/// GGUF architecture identifier for mmproj files.
pub const ARCH_CLIP: &str = "clip";

/// Which projector shape the mmproj uses. Determines the final dim of
/// the ViT hidden → text-embed transform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectorType {
    /// Dense 2-layer MLP (CLIP-classic / SigLIP-49 default).
    Mlp,
    /// Perceiver-style resampler (uncommon; present in some VLMs).
    Resampler,
    /// Gemma-4 vision projector — single `Gemma4ClippableLinear` head
    /// (clamp-then-linear-then-clamp; see
    /// `/opt/llama.cpp/tools/mtmd/clip-impl.h:323` ↦ "gemma4v").
    /// hf2q runtime path lives in
    /// `vit_gpu::gemma4v_apply_full_forward_gpu` (W25 iter-115 +
    /// W26 iter-116a + iter-117/118 kernel fixes); the
    /// arch-profile dispatch in
    /// `compute_vision_embeddings_gpu_dispatch` routes here when the
    /// loaded mmproj is `ArchProfile::Gemma4Siglip`.
    Gemma4v,
    /// Qwen3-VL spatial-merger projector — 2×2 patch merge → 2-layer
    /// MLP (`mm.0`/`mm.2`) → optional per-layer DeepStack heads at
    /// `v.deepstack.{N}.{fc1,fc2,norm}`. llama.cpp constant
    /// `PROJECTOR_TYPE_QWEN3VL` writes string `"qwen3vl_merger"`
    /// (`/opt/llama.cpp/tools/mtmd/clip-impl.h:318`).
    ///
    /// **Wedge-4b note (iter-224, this iter):** `is_supported()` returns
    /// `false` while the runtime ViT path is still queued under
    /// Wedge-4c. The variant exists so that `MmprojConfig::from_gguf`
    /// + `validate_tensor_set` can already PARSE a Qwen3-VL mmproj
    /// (listing-without-serving works; tensor-set validation can run
    /// against the deepstack tensor expectations) without prematurely
    /// claiming a forward path. Wedge-4c flips
    /// `is_supported()` true when `compute_vision_embeddings_gpu_qwen3vl`
    /// lands at `vit_gpu.rs:2412`.
    Qwen3VlMerger,
    /// Other / unknown projector shape — forward pass cannot run.
    /// Listable via `/v1/models` but rejected by `validate_tensor_set`
    /// at serve startup so the server never advertises a forward path
    /// it can't back.
    Other(String),
}

impl ProjectorType {
    pub fn from_str_gguf(s: &str) -> Self {
        match s {
            "mlp" => ProjectorType::Mlp,
            "resampler" => ProjectorType::Resampler,
            // llama.cpp's `PROJECTOR_TYPE_GEMMA4V` writes the literal
            // string "gemma4v" via PROJECTOR_TYPE_NAMES (clip-impl.h:323).
            // hf2q's writer matches via `build_mmproj_metadata`
            // (`backends/gguf.rs:606-610`).
            "gemma4v" => ProjectorType::Gemma4v,
            // llama.cpp's `PROJECTOR_TYPE_QWEN3VL` writes the literal
            // string "qwen3vl_merger" via PROJECTOR_TYPE_NAMES
            // (`/opt/llama.cpp/tools/mtmd/clip-impl.h:318`). iter-224
            // Wedge-4b: parse-only; the runtime forward path is still
            // queued under Wedge-4c.
            "qwen3vl_merger" => ProjectorType::Qwen3VlMerger,
            other => ProjectorType::Other(other.to_string()),
        }
    }
    pub fn as_str(&self) -> &str {
        match self {
            ProjectorType::Mlp => "mlp",
            ProjectorType::Resampler => "resampler",
            ProjectorType::Gemma4v => "gemma4v",
            ProjectorType::Qwen3VlMerger => "qwen3vl_merger",
            ProjectorType::Other(s) => s.as_str(),
        }
    }
    /// Whether hf2q has a runtime forward path for this projector.
    ///
    /// `Mlp` — yes (SigLIP-49 / classic-CLIP path; production since
    /// Phase 2c iter-50s).
    ///
    /// `Gemma4v` — yes (W25 iter-115 +  W26 iter-116a runtime
    /// dispatch; W41 iter-116i wires the writer ↔ runtime
    /// `is_supported()` gate so `hf2q serve --mmproj` accepts the
    /// hf2q-emitted gemma4v fixture). The arch-profile dispatch
    /// in `compute_vision_embeddings_gpu_dispatch` routes Gemma4v
    /// inputs through `gemma4v_apply_full_forward_gpu`; the
    /// projector head `Gemma4ClippableLinear` is in
    /// `vit_gpu.rs::gemma4v_apply_full_forward_gpu` (Stage 8).
    ///
    /// `Qwen3VlMerger` — **YES** (iter-224 Wedge-4c.5 LANDED). The full
    /// Qwen3-VL ViT forward (`compute_vision_embeddings_gpu_qwen3vl` at
    /// `vit_gpu_qwen3vl.rs`), the LM-side split-and-add hooks
    /// (`Qwen35Model::forward_gpu_*_with_soft_tokens`), the
    /// `image_token_residual_add` GPU dispatch, and the mmproj loader's
    /// fused-`attn_qkv` slice-view extension all ship together. The
    /// `is_supported()` flip is the gate that lets `validate_tensor_set`
    /// accept Qwen3-VL mmproj files at `serve --mmproj` startup.
    ///
    /// `Resampler` / `Other(_)` — no (no kernel path).
    pub fn is_supported(&self) -> bool {
        // iter-224 Wedge-4c.5: `Qwen3VlMerger` LANDED — every leg of
        // the Qwen3-VL forward path (ViT forward in vit_gpu_qwen3vl.rs,
        // LM-side hooks at forward_gpu.rs, image_token_residual_add
        // GPU dispatch, fused-attn_qkv loader) is wired before this
        // gate flipped. The matching assertion at
        // `projector_supported_for_qwen3vl_merger_after_wedge_4c5`
        // (tests below) regression-guards the flip.
        matches!(
            self,
            ProjectorType::Mlp | ProjectorType::Gemma4v | ProjectorType::Qwen3VlMerger
        )
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
    // ---- Qwen3-VL extensions (iter-224 Wedge-4b) -----------------------
    /// `clip.vision.spatial_merge_size` — Qwen3-VL spatial-merger
    /// degree (typically `2`, giving 2×2 patch fold + 4× token
    /// reduction). `None` for non-Qwen3-VL profiles. Source key:
    /// `KEY_SPATIAL_MERGE_SIZE` at clip-impl.h:49.
    pub spatial_merge_size: Option<u32>,
    /// `clip.vision.projection_dim` — the LM hidden size the projector
    /// targets. For Qwen3-VL-2B this is `2048` (matches LM
    /// `hidden_size`). `None` for profiles that don't write a
    /// projection-dim metadata key (Gemma 4 / CLIP-classic do NOT).
    /// Source key: `KEY_PROJ_DIM` at clip-impl.h:32 (formatted with
    /// the `vision` prefix → `clip.vision.projection_dim`).
    pub projection_dim: Option<u32>,
    /// Sorted ascending list of layer indexes (0-based) whose ViT
    /// hidden state is fed into the LM as DeepStack augmentation in
    /// addition to the post-LN output. Qwen3-VL-2B uses
    /// `[5, 11, 17]`. `None` when the GGUF lacks
    /// `clip.vision.is_deepstack_layers`. Empty `Vec` (length 0) when
    /// the key is present but no layer is flagged.
    ///
    /// **Source format**: GGUF stores this as a **boolean array of
    /// length `block_count`** under
    /// `KEY_IS_DEEPSTACK_LAYERS = "clip.vision.is_deepstack_layers"`
    /// (clip-impl.h:50). Writer reference:
    /// `gguf_writer.add_array(IS_DEEPSTACK_LAYERS, layers)` at
    /// `/opt/llama.cpp/gguf-py/gguf/gguf_writer.py:1219-1220`.
    /// Conversion: indexes where `bool[i] == true` are pushed in
    /// ascending order.
    pub deepstack_indexes: Option<Vec<u32>>,
}

/// Read `clip.vision.is_deepstack_layers` (a `Bool[block_count]` array
/// per `/opt/llama.cpp/tools/mtmd/clip-impl.h:50`) and convert to a
/// sorted ascending `Vec<u32>` of true-flagged layer indexes.
///
/// Returns `Ok(None)` when the key is absent (non-Qwen3-VL mmproj).
/// Returns `Ok(Some(vec))` when present, where `vec` may be empty.
/// Returns `Err` when:
///   - the value is present but not an `Array`,
///   - any array element is not a `Bool`,
///   - the array length disagrees with `block_count` (loudly: this
///     means the writer mis-encoded the bool array, and silently
///     consuming a length-mismatched array could mask off-by-one
///     deepstack-layer routing in Wedge-4c).
fn read_deepstack_indexes(
    gguf: &GgufFile,
    num_hidden_layers: u32,
) -> Result<Option<Vec<u32>>> {
    let raw = match gguf.metadata("clip.vision.is_deepstack_layers") {
        Some(v) => v,
        None => return Ok(None),
    };
    let arr = match raw {
        MetadataValue::Array(a) => a,
        _ => {
            return Err(anyhow!(
                "mmproj GGUF 'clip.vision.is_deepstack_layers' is not an \
                 Array(Bool); refusing to silently coerce"
            ));
        }
    };
    if arr.len() as u32 != num_hidden_layers {
        return Err(anyhow!(
            "mmproj GGUF 'clip.vision.is_deepstack_layers' length {} \
             disagrees with block_count {} — writer mis-encoded the \
             per-layer bool array; refusing to silently truncate",
            arr.len(),
            num_hidden_layers
        ));
    }
    let mut out: Vec<u32> = Vec::new();
    for (idx, v) in arr.iter().enumerate() {
        match v {
            MetadataValue::Bool(true) => out.push(idx as u32),
            MetadataValue::Bool(false) => {}
            _ => {
                return Err(anyhow!(
                    "mmproj GGUF 'clip.vision.is_deepstack_layers'[{}] is \
                     not a Bool element",
                    idx
                ));
            }
        }
    }
    // Inserted in ascending iteration order, so already sorted.
    Ok(Some(out))
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

        // ----- Qwen3-VL extensions (iter-224 Wedge-4b) -----
        // All three fields are Optional — non-Qwen3-VL mmproj GGUFs
        // (Gemma 4 / CLIP-classic) do not write these keys, so absence
        // is the universal default and presence opts the file into the
        // Qwen3-VL family. We do NOT set these from the projector_type
        // string alone — a malformed Qwen3-VL mmproj that ships
        // `projector_type=qwen3vl_merger` but omits these keys should
        // surface as `None` here so the Wedge-4c forward path can
        // fail loud rather than guess.
        let spatial_merge_size = gguf.metadata_u32("clip.vision.spatial_merge_size");
        let projection_dim = gguf.metadata_u32("clip.vision.projection_dim");
        let deepstack_indexes = read_deepstack_indexes(gguf, num_hidden_layers)?;

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
            spatial_merge_size,
            projection_dim,
            deepstack_indexes,
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
//   - v.blk.{N}.attn_out.{weight,bias}    (TN_ATTN_OUTPUT short form,
//                                           clip-impl.h:82)
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
/// gemma4v projector head — `TN_MM_INP_PROJ` at
/// `/opt/llama.cpp/tools/mtmd/clip-impl.h:110`. Distinct base name
/// from the CLIP-classic `mm.0.weight` because gemma4v's projector is
/// a `Gemma4ClippableLinear`, not an MLP. clip.cpp:1937 hard-requires
/// this exact name when `proj_type == PROJECTOR_TYPE_GEMMA4V`.
pub const TENSOR_MM_INPUT_PROJECTION_WEIGHT: &str = "mm.input_projection.weight";

// ---------------------------------------------------------------------------
// Qwen3-VL DeepStack tensor names (iter-224 Wedge-4b)
// ---------------------------------------------------------------------------
//
// Qwen3-VL augments the standard CLIP-classic per-block tensor set with
// a per-flagged-layer DeepStack head fed back into the LM. Naming
// matches llama.cpp's `TN_DEEPSTACK_*` family
// (`/opt/llama.cpp/tools/mtmd/clip-impl.h:117-119`):
//
//   - v.deepstack.{N}.norm.{weight,bias}    (LayerNorm before the
//                                            DeepStack MLP)
//   - v.deepstack.{N}.fc1.{weight,bias}     (Linear → GELU → Linear)
//   - v.deepstack.{N}.fc2.{weight,bias}
//
// Per clip.cpp:1697-1705, these are loaded as OPTIONAL on every block —
// the model treats a layer as "deepstack" when the trio is present
// (`has_deepstack()` returns true). Validator behavior: when
// `MmprojConfig.deepstack_indexes` is `Some(vec)`, require the trio
// for every flagged index. When `None` (key absent), skip the
// deepstack tensor check entirely (lenient — mirrors llama.cpp's
// "load if present" semantics for the per-layer fields).

/// Per-DeepStack-layer tensor name. `suffix` is e.g. `"fc1.weight"` or
/// `"norm.bias"`.
///
/// Source format string: `TN_DEEPSTACK_FC1 = "v.deepstack.%d.fc1.%s"`
/// at `/opt/llama.cpp/tools/mtmd/clip-impl.h:118`. Matching constants
/// for `norm` (line 117) and `fc2` (line 119).
pub fn vit_deepstack_tensor(layer_idx: usize, suffix: &str) -> String {
    format!("v.deepstack.{}.{}", layer_idx, suffix)
}

/// Required DeepStack-tensor suffixes for a flagged Qwen3-VL layer.
/// Bias tensors are present in the upstream definition but llama.cpp's
/// loader treats them as optional (`get_tensor(..., false)`); the
/// validator enforces only the weights, matching the
/// `has_deepstack()` predicate at clip-model.h:227-229 which keys on
/// `deepstack_fc1_w != nullptr`.
pub const DEEPSTACK_REQUIRED_SUFFIXES: &[&str] = &[
    "norm.weight",
    "fc1.weight",
    "fc2.weight",
];

/// Per-layer tensor name helper.
pub fn vit_layer_tensor(layer_idx: usize, suffix: &str) -> String {
    format!("v.blk.{}.{}", layer_idx, suffix)
}

/// Supported architectural profile for an mmproj GGUF. Different
/// producers (SigLIP, CLIP, Gemma 4 vision tower, Qwen3-VL) carry
/// different tensor-name conventions. The profile is auto-detected
/// from the actual tensor set at startup; forward-pass dispatch
/// branches on it.
///
/// Detection rule (order matters — first match wins):
///   - `Qwen3VlSiglip` — file ships `v.deepstack.0.fc1.weight` (the
///     unique Qwen3-VL DeepStack head; clip-impl.h:118). This wins
///     over `Gemma4Siglip` because Qwen3-VL inherits Gemma 4's
///     `ln1`/`ln2` per-block layout but adds DeepStack on top.
///   - `Gemma4Siglip` — per-block has `ln1.weight`, `ln2.weight`, and
///     `ffn_post_norm.weight` (Gemma 4's dual-LN SigLIP variant —
///     llama.cpp's `TN_FFN_POST_NORM` short form, clip-impl.h:95).
///   - `ClipClassic`  — per-block has `attn_norm.weight` (llama.cpp's
///     default CLIP-style writer).
///   - `Unknown`      — none matched. Forward pass not supported.
///
/// **iter-224 Wedge-4b**: `Qwen3VlSiglip` is added at parse-time but
/// `is_supported()` returns `false` until Wedge-4c lands the
/// runtime forward path at `vit_gpu.rs:2412`. This keeps
/// listing-without-serving working on Qwen3-VL mmproj while
/// `serve --mmproj` rejects it cleanly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchProfile {
    Gemma4Siglip,
    ClipClassic,
    /// Qwen3-VL SigLIP variant — inherits CLIP-classic per-block
    /// `attn_norm` + `ln1`/`ln2` layout but adds per-flagged-layer
    /// `v.deepstack.{N}.{norm,fc1,fc2}.{weight,bias}` heads on top
    /// of the standard `mm.0`/`mm.2` 2-layer MLP projector. Selected
    /// via the DeepStack tensor marker (see `detect_arch_profile`)
    /// or via `ProjectorType::Qwen3VlMerger` (see
    /// `detect_arch_profile_with_projector`).
    Qwen3VlSiglip,
    Unknown,
}

impl ArchProfile {
    pub fn as_str(&self) -> &'static str {
        match self {
            ArchProfile::Gemma4Siglip => "gemma4_siglip",
            ArchProfile::ClipClassic => "clip_classic",
            ArchProfile::Qwen3VlSiglip => "qwen3vl_siglip",
            ArchProfile::Unknown => "unknown",
        }
    }
    /// Whether hf2q has a runtime forward path for this arch.
    ///
    /// **iter-224 Wedge-4c.5**: `Qwen3VlSiglip` returns `true` —
    /// `compute_vision_embeddings_gpu_qwen3vl` (vit_gpu_qwen3vl.rs)
    /// supplies the ViT forward; the LM-side split-and-add hooks at
    /// `Qwen35Model::forward_gpu_*_with_soft_tokens` consume the
    /// augmented embed; mmproj loader handles fused `attn_qkv`. Note
    /// that the chat-handler-side preprocess routing for Qwen3-VL
    /// (variable-resolution patch grid + 3D-mRoPE positions +
    /// `<|vision_start|><|image_pad|><|vision_end|>` placeholder
    /// expansion) is Wedge-4d territory — when an image-bearing
    /// chat request lands, `process_multimodal_content` fails loud
    /// with a Wedge-4d-pointing message until that lands.
    pub fn is_supported(&self) -> bool {
        matches!(
            self,
            ArchProfile::Gemma4Siglip | ArchProfile::ClipClassic | ArchProfile::Qwen3VlSiglip
        )
    }
}

/// Detect the mmproj's tensor-naming profile from a tensor-name list.
/// Used by startup validation + forward-pass dispatch.
///
/// This is cheap: a ~10-element probe over the block-0 tensor set.
/// Returns `Unknown` when the file has neither marker, which the
/// validator maps to a 400 `no_mmproj_loaded`-style error at request time.
///
/// W41 iter-116i: gemma4 marker tightened to llama.cpp's actual
/// `TN_FFN_POST_NORM = "%s.blk.%d.ffn_post_norm.%s"`
/// (`/opt/llama.cpp/tools/mtmd/clip-impl.h:95`); the prior
/// `post_ffw_norm.weight` literal predates that header convention
/// and never appeared in real llama.cpp-emitted gemma4 mmproj files.
/// The legacy spelling is still accepted as a fallback so any older
/// fixtures don't regress.
///
/// **iter-224 Wedge-4b**: Qwen3-VL detection probes for
/// `v.deepstack.0.fc1.weight` — the canonical DeepStack head tensor
/// at clip-impl.h:118. This is unique to the qwen3vl projector
/// family (no other clip projector emits a `v.deepstack.*` tensor),
/// so it's a sound first-match for the order. Callers that have
/// the parsed `ProjectorType` available should prefer
/// `detect_arch_profile_with_projector` — that path catches a
/// hypothetical Qwen3-VL mmproj where every flagged-DeepStack-layer
/// happens to be at index > 0 (none observed in the wild, but the
/// projector_type string is the more decisive signal).
pub fn detect_arch_profile(actual_names: &[&str]) -> ArchProfile {
    let set: std::collections::HashSet<&str> = actual_names.iter().copied().collect();

    // Qwen3-VL canonical marker: any `v.deepstack.{N}.fc1.weight`
    // tensor uniquely identifies the qwen3vl projector. Probing
    // index 0 specifically because that's the lowest layer index
    // that could possibly carry a DeepStack head; if a producer
    // ever ships Qwen3-VL with no flagged layer at index 0,
    // detect_arch_profile_with_projector covers that edge.
    if set.contains("v.deepstack.0.fc1.weight") {
        return ArchProfile::Qwen3VlSiglip;
    }

    let has_ln_pair = set.contains("v.blk.0.ln1.weight")
        && set.contains("v.blk.0.ln2.weight");
    let has_gemma4_post_norm = set.contains("v.blk.0.ffn_post_norm.weight")
        || set.contains("v.blk.0.post_ffw_norm.weight");
    if has_ln_pair && has_gemma4_post_norm {
        return ArchProfile::Gemma4Siglip;
    }
    if set.contains("v.blk.0.attn_norm.weight") {
        return ArchProfile::ClipClassic;
    }
    ArchProfile::Unknown
}

/// Projector-aware variant of `detect_arch_profile`. Use this when
/// the parsed `MmprojConfig` is available (e.g. inside the serve-side
/// startup pipeline that just ran `MmprojConfig::from_gguf`). When
/// `projector == Qwen3VlMerger` the profile is `Qwen3VlSiglip`
/// regardless of which tensors are present — the projector_type
/// string is the most decisive upstream signal because llama.cpp
/// gates its `clip_graph_qwen3vl` builder on it
/// (`/opt/llama.cpp/tools/mtmd/clip.cpp:865-867`).
///
/// Falls through to `detect_arch_profile` for the non-Qwen3-VL cases.
pub fn detect_arch_profile_with_projector(
    projector: &ProjectorType,
    actual_names: &[&str],
) -> ArchProfile {
    if matches!(projector, ProjectorType::Qwen3VlMerger) {
        return ArchProfile::Qwen3VlSiglip;
    }
    detect_arch_profile(actual_names)
}

/// Minimal, arch-agnostic tensor-set sanity check for an mmproj GGUF.
///
/// Verifies only the tensors EVERY ViT mmproj must carry regardless of
/// the underlying arch:
///   - `v.patch_embd.weight` (pixel → patch-embedding)
///   - `v.position_embd.weight` (learned position encoding)
///   - `v.blk.0.attn_q.weight` + `attn_k.weight` + `attn_v.weight` +
///     `attn_out.weight` (block 0's QKV + output projection — the
///     vision-namespace short form per llama.cpp's
///     `TN_ATTN_OUTPUT = "%s.blk.%d.attn_out.%s"`,
///     `/opt/llama.cpp/tools/mtmd/clip-impl.h:82`).
///   - at least one of `mm.0.weight` / `mm.2.weight` (the projector head)
///   - per `MmprojConfig.num_hidden_layers`, the same QKV+output set for
///     every block (catches truncated files)
///   - per Qwen3-VL `MmprojConfig.deepstack_indexes` (when `Some`), the
///     `v.deepstack.{N}.{norm,fc1,fc2}.weight` trio for every flagged
///     index. iter-224 Wedge-4b.
///
/// Does NOT require arch-specific tensors like `attn_norm.weight` or
/// `ln1.weight` — those are detected via `ArchProfile` separately and
/// drive forward-pass dispatch, not boot validation.
///
/// Unsupported projector types (`Resampler`, `Other(_)`, and currently
/// `Qwen3VlMerger` until Wedge-4c) still fail the check outright —
/// forward pass can't run regardless of tensor completeness.
///
/// Missing tensors batched into one error listing up to 10 names so a
/// producer bug doesn't become one-at-a-time whack-a-mole on restart.
///
/// W41 iter-116i note: pre-iter-116i this validator required
/// `attn_output.weight` (the text-decoder name).  W34 iter-116e
/// fixed the writer to emit the vision-namespace `attn_out.weight`
/// short form per `TN_ATTN_OUTPUT`, but the validator was missed —
/// causing `hf2q serve --mmproj <gemma4v>` to bail with "missing 28
/// required tensor(s)" once the projector-type guard was unblocked.
///
/// **iter-224 Wedge-4b**: `validate_tensor_set` was restructured so
/// the universal/per-block/projector/deepstack tensor checks run
/// BEFORE the projector-supported gate. This means even an
/// unsupported-projector mmproj file gets a coherent missing-tensor
/// diagnostic when the writer dropped a universal tensor — more
/// actionable than a bare "projector type not supported" message
/// that hides the producer bug. The projector-supported gate runs
/// LAST.
///
/// **iter-224 Wedge-4c.5**: per-block QKV check now accepts EITHER
/// the fused `v.blk.{N}.attn_qkv.weight` (Qwen3-VL HF-source canonical,
/// emitted by /opt/llama.cpp/convert_hf_to_gguf.py:4853-4972) OR the
/// classic split `attn_q/k/v.weight` trio. Mixing is rejected loud
/// (producer bug). `Qwen3VlMerger.is_supported()` returns true after
/// this iter, so a tensor-complete Qwen3-VL mmproj passes the gate
/// rather than surfacing the old "not yet supported" reject.
pub fn validate_tensor_set(cfg: &MmprojConfig, actual_names: &[&str]) -> Result<()> {
    let actual_set: std::collections::HashSet<&str> = actual_names.iter().copied().collect();
    let mut missing: Vec<String> = Vec::new();

    // Universal tensors (arch-agnostic). After Wedge-4c.5 the per-block
    // QKV + attn_out checks moved into the layer loop below, so this
    // list now carries only the patch-embed + position-embed pair —
    // no longer needs to grow inside the loop.
    let required: Vec<String> = vec![
        TENSOR_PATCH_EMBD.to_string(),
        TENSOR_POS_EMBD.to_string(),
    ];
    // Per-block QKV + output (present in CLIP, Gemma 4, AND Qwen3-VL).
    // The output projection's vision-namespace short-form is `attn_out`
    // per `TN_ATTN_OUTPUT = "%s.blk.%d.attn_out.%s"`
    // (`/opt/llama.cpp/tools/mtmd/clip-impl.h:82`); this is distinct
    // from the text-decoder `attn_output.weight` long-form.
    //
    // **Wedge-4c.5**: Qwen3-VL converters (HF source) emit a FUSED
    // `attn_qkv.{weight,bias}` per block (clip-impl.h:78
    // `TN_ATTN_QKV = "%s.blk.%d.attn_qkv.%s"`,
    // /opt/llama.cpp/tools/mtmd/clip.cpp:1669) instead of split
    // `attn_q/k/v`. Both forms back the same logical Q/K/V projection;
    // the runtime mmproj loader (`LoadedMmprojWeights::load`) installs
    // canonical-name slice views so the consumer code (`block_tensor`)
    // remains unchanged. The validator accepts EITHER convention
    // per-block but rejects MIXED (some blocks fused, others split) —
    // mixing means a producer bug worth surfacing loud.
    for layer_idx in 0..cfg.num_hidden_layers as usize {
        // attn_out is universal — neither convention fuses it.
        let attn_out_name = vit_layer_tensor(layer_idx, "attn_out.weight");
        if !actual_set.contains(attn_out_name.as_str()) {
            missing.push(attn_out_name);
        }

        // QKV: accept fused `attn_qkv.weight` OR the split
        // `attn_q.weight + attn_k.weight + attn_v.weight` trio.
        let qkv_fused_w = vit_layer_tensor(layer_idx, "attn_qkv.weight");
        let q_w = vit_layer_tensor(layer_idx, "attn_q.weight");
        let k_w = vit_layer_tensor(layer_idx, "attn_k.weight");
        let v_w = vit_layer_tensor(layer_idx, "attn_v.weight");
        let has_fused = actual_set.contains(qkv_fused_w.as_str());
        let split_present = [&q_w, &k_w, &v_w]
            .iter()
            .filter(|n| actual_set.contains(n.as_str()))
            .count();
        let has_full_split = split_present == 3;
        match (has_fused, has_full_split, split_present) {
            (true, false, 0) => { /* fused-only — accepted */ }
            (false, true, _) => { /* split-only — accepted */ }
            (true, true, _) => {
                missing.push(format!(
                    "block {layer_idx}: BOTH fused '{qkv_fused_w}' AND split \
                     attn_q/k/v are present — producer must emit one form, not both"
                ));
            }
            (true, false, _) => {
                missing.push(format!(
                    "block {layer_idx}: fused '{qkv_fused_w}' is present but \
                     PARTIAL split tensors leaked through (only {split_present}/3 \
                     of attn_q/k/v.weight present) — refusing to mix conventions"
                ));
            }
            (false, false, _) => {
                if split_present == 0 {
                    missing.push(format!(
                        "block {layer_idx}: missing QKV — neither fused '{qkv_fused_w}' \
                         nor split attn_q/k/v.weight trio is present"
                    ));
                } else {
                    // partial split, no fused — list each missing leg
                    if !actual_set.contains(q_w.as_str()) { missing.push(q_w); }
                    if !actual_set.contains(k_w.as_str()) { missing.push(k_w); }
                    if !actual_set.contains(v_w.as_str()) { missing.push(v_w); }
                }
            }
        }
    }
    for name in &required {
        if !actual_set.contains(name.as_str()) {
            missing.push(name.clone());
        }
    }

    // Projector head: accept any of three names depending on projector
    // family:
    //   - `mm.0.weight`              — CLIP-classic MLP layer 0 (also
    //                                   Qwen3-VL's primary projector;
    //                                   clip.cpp:1846-1850)
    //   - `mm.2.weight`              — CLIP-classic / Qwen3-VL MLP
    //                                   layer 2 (clip.cpp:1848-1849)
    //   - `mm.input_projection.weight` — gemma4v's `Gemma4ClippableLinear`
    //                                   (TN_MM_INP_PROJ; clip-impl.h:110)
    // Absence of all three = no projector head.
    if !actual_set.contains(TENSOR_MM_0_WEIGHT)
        && !actual_set.contains(TENSOR_MM_2_WEIGHT)
        && !actual_set.contains(TENSOR_MM_INPUT_PROJECTION_WEIGHT)
    {
        missing.push(format!(
            "{} (or {}, or gemma4v's {})",
            TENSOR_MM_0_WEIGHT,
            TENSOR_MM_2_WEIGHT,
            TENSOR_MM_INPUT_PROJECTION_WEIGHT,
        ));
    }

    // Qwen3-VL DeepStack heads — for every flagged layer
    // (per `cfg.deepstack_indexes`), require the trio
    // `v.deepstack.{N}.{norm,fc1,fc2}.weight`. iter-224 Wedge-4b: this
    // check runs whenever `deepstack_indexes` is `Some(_)`, regardless
    // of whether the projector is `Qwen3VlMerger` — a Gemma 4 mmproj
    // that somehow shipped a deepstack-indexes key would produce a
    // sensible error here too. When `deepstack_indexes` is `None`
    // (no GGUF metadata key, the universal case), no DeepStack
    // tensors are required.
    if let Some(indexes) = &cfg.deepstack_indexes {
        for &flagged_idx in indexes {
            if flagged_idx >= cfg.num_hidden_layers {
                missing.push(format!(
                    "deepstack_indexes entry {} exceeds block_count {} \
                     (writer mis-encoded clip.vision.is_deepstack_layers)",
                    flagged_idx, cfg.num_hidden_layers
                ));
                continue;
            }
            for suffix in DEEPSTACK_REQUIRED_SUFFIXES {
                let name = vit_deepstack_tensor(flagged_idx as usize, suffix);
                if !actual_set.contains(name.as_str()) {
                    missing.push(name);
                }
            }
        }
    }

    if !missing.is_empty() {
        let total_missing = missing.len();
        missing.truncate(10);
        let more = if total_missing > 10 {
            format!(" (+ {} more)", total_missing - 10)
        } else {
            String::new()
        };
        return Err(anyhow!(
            "mmproj GGUF is missing {} required tensor(s): {}{}",
            total_missing,
            missing.join(", "),
            more
        ));
    }

    // Tensor-set is complete — final gate is the projector-supported
    // check. iter-224 Wedge-4b: `Qwen3VlMerger` lands here with a
    // tensor-complete file but `is_supported() = false`, producing the
    // canonical "not yet supported" error so the validator's overall
    // verdict stays "reject at serve startup". Wedge-4c flips
    // `is_supported()` true and this branch becomes the success path.
    if !cfg.projector.is_supported() {
        return Err(anyhow!(
            "mmproj projector type '{}' is not yet supported by hf2q's ViT \
             forward pass. No forward path will succeed for this file.",
            cfg.projector.as_str()
        ));
    }

    Ok(())
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
        // W41 iter-116i: gemma4v parses to its own variant (was
        // `Other("gemma4v")`, which made `is_supported() = false`
        // and blocked `hf2q serve --mmproj <gemma4v>`).
        assert_eq!(
            ProjectorType::from_str_gguf("gemma4v"),
            ProjectorType::Gemma4v
        );
        // iter-224 Wedge-4b: qwen3vl_merger parses to its own variant.
        // Until Wedge-4c lands the runtime forward, `is_supported()` on
        // this variant remains false (regression-guarded by
        // `qwen3vl_merger_is_supported_returns_false_until_wedge_4c`
        // in the dedicated tests/mmproj_qwen3vl.rs file).
        assert_eq!(
            ProjectorType::from_str_gguf("qwen3vl_merger"),
            ProjectorType::Qwen3VlMerger
        );
        match ProjectorType::from_str_gguf("q-former") {
            ProjectorType::Other(s) => assert_eq!(s, "q-former"),
            other => panic!("expected Other, got {:?}", other),
        }
    }

    #[test]
    fn projector_supported_for_mlp_and_gemma4v() {
        // All three projector heads have a runtime forward path:
        //   - Mlp            → compute_vision_embeddings_gpu (SigLIP-49)
        //   - Gemma4v        → gemma4v_apply_full_forward_gpu via dispatch
        //   - Qwen3VlMerger  → compute_vision_embeddings_gpu_qwen3vl via
        //                      dispatch (Wedge-4c.5 LANDED)
        assert!(ProjectorType::Mlp.is_supported());
        assert!(ProjectorType::Gemma4v.is_supported());
        // iter-224 Wedge-4c.5 LANDED: Qwen3VlMerger is now supported —
        // the runtime ViT forward + LM-side hooks + fused-attn_qkv
        // loader all shipped in the same iter that flipped this bit.
        assert!(ProjectorType::Qwen3VlMerger.is_supported());
        assert!(!ProjectorType::Resampler.is_supported());
        assert!(!ProjectorType::Other("x".into()).is_supported());
    }

    #[test]
    fn projector_as_str_matches_input() {
        assert_eq!(ProjectorType::Mlp.as_str(), "mlp");
        assert_eq!(ProjectorType::Resampler.as_str(), "resampler");
        assert_eq!(ProjectorType::Gemma4v.as_str(), "gemma4v");
        // iter-224 Wedge-4b: round-trip through the GGUF string form
        // matches `PROJECTOR_TYPE_NAMES[QWEN3VL]` at clip-impl.h:318.
        assert_eq!(ProjectorType::Qwen3VlMerger.as_str(), "qwen3vl_merger");
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
            // iter-224 Wedge-4b: Qwen3-VL extension fields default to
            // None for non-Qwen3-VL profiles.
            spatial_merge_size: None,
            projection_dim: None,
            deepstack_indexes: None,
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
            spatial_merge_size: None,
            projection_dim: None,
            deepstack_indexes: None,
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
            // iter-224 Wedge-4b: Qwen3-VL extensions default to None
            // (Mlp profile has no DeepStack / spatial-merger metadata).
            spatial_merge_size: None,
            projection_dim: None,
            deepstack_indexes: None,
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
                // W41 iter-116i: vision-namespace short form per
                // TN_ATTN_OUTPUT (clip-impl.h:82); not the
                // text-decoder `attn_output.weight`.
                "attn_out.weight",
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
        // 2-layer cfg, only block 0's tensors present. Wedge-4c.5 changes
        // the missing-tensor surface for QKV: instead of listing each of
        // attn_q/k/v.weight individually, the validator emits a single
        // "block N: missing QKV — neither fused 'v.blk.N.attn_qkv.weight'
        // nor split attn_q/k/v.weight trio is present" message naming
        // both alternatives. attn_out.weight stays as its own line. Total
        // missing = 2 (1 QKV diagnostic + 1 attn_out).
        let cfg = mlp_cfg(2);
        let names = minimum_tensor_names(1); // only 1 layer's tensors
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let err = validate_tensor_set(&cfg, &actual).expect_err("should fail");
        let msg = format!("{err}");
        assert!(msg.contains("missing 2 required tensor"), "got: {msg}");
        assert!(msg.contains("v.blk.1.attn_out.weight"), "got: {msg}");
        assert!(
            msg.contains("block 1: missing QKV") && msg.contains("attn_qkv"),
            "got: {msg}"
        );
    }

    #[test]
    fn validate_tensor_set_rejects_unsupported_projector() {
        // iter-224 Wedge-4b: validator now runs the universal-tensor
        // arm BEFORE the projector-supported gate, so this test must
        // pass a tensor-complete actual_names so the gate fires last.
        // (Pre-iter-224 the gate was first and `actual_names = &[]`
        // was sufficient.)
        let mut cfg = mlp_cfg(1);
        cfg.projector = ProjectorType::Resampler;
        let names = minimum_tensor_names(1);
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let err = validate_tensor_set(&cfg, &actual).expect_err("unsupported projector");
        let msg = format!("{err}");
        assert!(msg.contains("'resampler' is not yet supported"), "got: {msg}");
    }

    #[test]
    fn validate_tensor_set_rejects_other_projector_with_name_echoed() {
        let mut cfg = mlp_cfg(1);
        cfg.projector = ProjectorType::Other("q-former".into());
        let names = minimum_tensor_names(1);
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let err = validate_tensor_set(&cfg, &actual).expect_err("unsupported projector");
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
    fn arch_profile_is_supported_only_for_runtime_paths() {
        // iter-224 Wedge-4c.5: all three arch profiles with a runtime
        // ViT forward report supported — Gemma4Siglip, ClipClassic, and
        // Qwen3VlSiglip (the latter via compute_vision_embeddings_gpu_qwen3vl).
        // Unknown never does. NOTE: image-bearing chat requests for
        // Qwen3-VL still fail loud at preprocess routing in
        // `process_multimodal_content` (handlers.rs) until Wedge-4d
        // lands the variable-resolution patch + 3D-mRoPE preprocess
        // pipeline; this test asserts the LOAD-TIME validator gate flip
        // only.
        assert!(ArchProfile::Gemma4Siglip.is_supported());
        assert!(ArchProfile::ClipClassic.is_supported());
        assert!(ArchProfile::Qwen3VlSiglip.is_supported());
        assert!(!ArchProfile::Unknown.is_supported());
    }

    #[test]
    fn arch_profile_as_str_is_snake_case() {
        assert_eq!(ArchProfile::Gemma4Siglip.as_str(), "gemma4_siglip");
        assert_eq!(ArchProfile::ClipClassic.as_str(), "clip_classic");
        assert_eq!(ArchProfile::Qwen3VlSiglip.as_str(), "qwen3vl_siglip");
        assert_eq!(ArchProfile::Unknown.as_str(), "unknown");
    }

    // -----------------------------------------------------------------------
    // iter-224 Wedge-4b — Qwen3-VL detect_arch_profile + projector-aware
    // detection inline regression guards (full GGUF round-trip lives in
    // tests/mmproj_qwen3vl.rs).
    // -----------------------------------------------------------------------

    #[test]
    fn detect_arch_profile_qwen3vl_from_deepstack_marker() {
        // Canonical Qwen3-VL detection: presence of any
        // `v.deepstack.{N}.fc1.weight` tensor (clip-impl.h:118).
        let names: Vec<&str> = vec![
            "v.patch_embd.weight",
            "v.blk.0.ln1.weight",
            "v.blk.0.attn_norm.weight",
            "v.blk.0.attn_q.weight",
            "v.deepstack.0.fc1.weight",
            "v.deepstack.0.fc2.weight",
            "v.deepstack.0.norm.weight",
        ];
        assert_eq!(detect_arch_profile(&names), ArchProfile::Qwen3VlSiglip);
    }

    #[test]
    fn detect_arch_profile_qwen3vl_marker_wins_over_gemma4_pattern() {
        // Pathological case: a Qwen3-VL mmproj whose per-block layout
        // overlaps the Gemma 4 ln1/ln2/ffn_post_norm trio. The
        // deepstack tensor is the more specific signal — it's unique
        // to qwen3vl, so it must win the order.
        let names: Vec<&str> = vec![
            "v.patch_embd.weight",
            "v.blk.0.ln1.weight",
            "v.blk.0.ln2.weight",
            "v.blk.0.ffn_post_norm.weight",
            "v.blk.0.attn_q.weight",
            "v.deepstack.0.fc1.weight",
        ];
        assert_eq!(detect_arch_profile(&names), ArchProfile::Qwen3VlSiglip);
    }

    #[test]
    fn detect_arch_profile_with_projector_short_circuits_on_qwen3vl_merger() {
        // The projector-aware variant: when the parsed projector is
        // `Qwen3VlMerger`, the profile is `Qwen3VlSiglip` regardless
        // of which tensors are enumerated. Mirrors llama.cpp's
        // builder-selection at clip.cpp:865-867.
        let names: Vec<&str> = vec!["v.patch_embd.weight"]; // no deepstack marker
        assert_eq!(
            detect_arch_profile_with_projector(&ProjectorType::Qwen3VlMerger, &names),
            ArchProfile::Qwen3VlSiglip
        );
    }

    #[test]
    fn detect_arch_profile_with_projector_falls_through_for_non_qwen3vl() {
        // For non-Qwen3VlMerger projectors the helper defers to the
        // tensor-only detection — proves we don't accidentally
        // mis-route a Gemma4Siglip mmproj just because the helper got
        // touched by Wedge-4b.
        let names: Vec<&str> = vec![
            "v.patch_embd.weight",
            "v.blk.0.ln1.weight",
            "v.blk.0.ln2.weight",
            "v.blk.0.ffn_post_norm.weight",
        ];
        assert_eq!(
            detect_arch_profile_with_projector(&ProjectorType::Gemma4v, &names),
            ArchProfile::Gemma4Siglip
        );
        assert_eq!(
            detect_arch_profile_with_projector(&ProjectorType::Mlp, &names),
            ArchProfile::Gemma4Siglip
        );
    }

    #[test]
    fn vit_deepstack_tensor_formats_deepstack_prefix() {
        // Mirrors TN_DEEPSTACK_FC1 at clip-impl.h:118.
        assert_eq!(
            vit_deepstack_tensor(0, "fc1.weight"),
            "v.deepstack.0.fc1.weight"
        );
        assert_eq!(
            vit_deepstack_tensor(17, "norm.bias"),
            "v.deepstack.17.norm.bias"
        );
    }

    #[test]
    fn deepstack_required_suffixes_match_llama_cpp_load_predicate() {
        // llama.cpp's `has_deepstack()` (clip-model.h:227-229) keys on
        // `deepstack_fc1_w != nullptr`; we additionally require fc2
        // and norm weights (the trio that defines a fully-formed
        // DeepStack head). Bias tensors are optional in llama.cpp's
        // loader, so they're NOT in this list.
        assert_eq!(
            DEEPSTACK_REQUIRED_SUFFIXES,
            &["norm.weight", "fc1.weight", "fc2.weight"]
        );
    }

    fn qwen3vl_cfg(num_layers: u32, deepstack_indexes: Vec<u32>) -> MmprojConfig {
        MmprojConfig {
            image_size: 768,
            patch_size: 16,
            num_patches_side: 48,
            hidden_size: 1024,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: num_layers,
            layer_norm_eps: 1e-6,
            projector: ProjectorType::Qwen3VlMerger,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            spatial_merge_size: Some(2),
            projection_dim: Some(2048),
            deepstack_indexes: Some(deepstack_indexes),
        }
    }

    /// Build a Qwen3-VL minimum tensor set: universal tensors + per-block
    /// QKV+output + mm.0.weight + per-flagged-deepstack-layer trio.
    fn qwen3vl_minimum_tensor_names(num_layers: u32, flagged: &[u32]) -> Vec<String> {
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
                "attn_out.weight",
            ] {
                names.push(vit_layer_tensor(layer_idx, suffix));
            }
        }
        for &idx in flagged {
            for suffix in DEEPSTACK_REQUIRED_SUFFIXES {
                names.push(vit_deepstack_tensor(idx as usize, suffix));
            }
        }
        names
    }

    #[test]
    fn validate_qwen3vl_complete_set_passes_after_wedge_4c5() {
        // iter-224 Wedge-4c.5 LANDED: a tensor-complete Qwen3-VL mmproj
        // must PASS the universal/projector/deepstack checks AND the
        // final projector-supported gate (now true for Qwen3VlMerger).
        // This is the green-flip signal — pre-Wedge-4c.5 this test
        // asserted the gate REJECTED with "not yet supported"; flipping
        // the test polarity here is the regression-guard counterpart
        // to flipping `is_supported()`'s matches!() arm.
        let cfg = qwen3vl_cfg(24, vec![5, 11, 17]);
        let names = qwen3vl_minimum_tensor_names(24, &[5, 11, 17]);
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        validate_tensor_set(&cfg, &actual).expect(
            "Wedge-4c.5: tensor-complete Qwen3-VL with split attn_q/k/v must validate cleanly \
             now that ProjectorType::Qwen3VlMerger.is_supported() returns true",
        );
    }

    #[test]
    fn validate_qwen3vl_flags_missing_deepstack_trio() {
        // Tensor set complete EXCEPT the deepstack tensors for a
        // flagged layer — must surface the specific missing tensor
        // names.
        let cfg = qwen3vl_cfg(24, vec![5, 11, 17]);
        // Drop layer-11's deepstack trio (flagged-but-missing).
        let names = qwen3vl_minimum_tensor_names(24, &[5, 17]);
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let err = validate_tensor_set(&cfg, &actual)
            .expect_err("missing deepstack trio at layer 11 must fail");
        let msg = format!("{err}");
        assert!(msg.contains("v.deepstack.11.norm.weight"), "got: {msg}");
        assert!(msg.contains("v.deepstack.11.fc1.weight"), "got: {msg}");
        assert!(msg.contains("v.deepstack.11.fc2.weight"), "got: {msg}");
    }

    #[test]
    fn validate_qwen3vl_flags_out_of_range_deepstack_index() {
        // A writer that emits is_deepstack_layers with an index >=
        // block_count would panic the loader at runtime; the
        // validator must catch it up front.
        let cfg = qwen3vl_cfg(4, vec![10]); // 10 >= 4 layers
        let names = qwen3vl_minimum_tensor_names(4, &[]);
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let err = validate_tensor_set(&cfg, &actual)
            .expect_err("deepstack index >= block_count must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("deepstack_indexes entry 10 exceeds block_count 4"),
            "got: {msg}"
        );
    }

    #[test]
    fn validate_qwen3vl_no_deepstack_indexes_is_lenient() {
        // When `deepstack_indexes` is None (no GGUF metadata key —
        // the universal case for non-Qwen3-VL files AND a hypothetical
        // older Qwen3-VL fixture that predates the writer), no
        // DeepStack tensors are required.
        let mut cfg = qwen3vl_cfg(4, vec![]);
        cfg.deepstack_indexes = None;
        // Use the Mlp projector to bypass the unsupported-projector
        // gate — this test isolates the deepstack-leniency arm.
        cfg.projector = ProjectorType::Mlp;
        let names = qwen3vl_minimum_tensor_names(4, &[]);
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        validate_tensor_set(&cfg, &actual)
            .expect("None deepstack_indexes + complete universal set must pass");
    }

    #[test]
    fn validate_qwen3vl_empty_deepstack_indexes_requires_no_extras() {
        // Some(vec![]) — the GGUF key is present but no layer is
        // flagged. Validator must accept (no extra tensors required)
        // when the universal set is complete.
        let mut cfg = qwen3vl_cfg(4, vec![]);
        cfg.projector = ProjectorType::Mlp; // bypass unsupported gate
        let names = qwen3vl_minimum_tensor_names(4, &[]);
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        validate_tensor_set(&cfg, &actual)
            .expect("empty deepstack_indexes with complete universal set must pass");
    }

    // -----------------------------------------------------------------------
    // Wedge-4c.5: validator accepts fused `attn_qkv` per-block.
    //
    // /opt/llama.cpp/convert_hf_to_gguf.py:4853-4972's Qwen3VLVisionModel
    // emits the fused name (`V_ENC_ATTN_QKV` mapped to
    // `v.blk.{bid}.attn_qkv` at gguf-py/gguf/constants.py:1205);
    // /opt/llama.cpp/tools/mtmd/clip.cpp:1669 loads it as a single
    // optional tensor. Both shapes back the same logical Q/K/V projection.
    // -----------------------------------------------------------------------

    /// Build a Qwen3-VL minimum tensor set with FUSED `attn_qkv`
    /// instead of split `attn_q/k/v`. Used by the Wedge-4c.5 validator
    /// tests below.
    fn qwen3vl_fused_qkv_tensor_names(num_layers: u32, flagged: &[u32]) -> Vec<String> {
        let mut names = vec![
            TENSOR_PATCH_EMBD.to_string(),
            TENSOR_POS_EMBD.to_string(),
            TENSOR_MM_0_WEIGHT.to_string(),
        ];
        for layer_idx in 0..num_layers as usize {
            // Fused QKV (one tensor) instead of three.
            names.push(vit_layer_tensor(layer_idx, "attn_qkv.weight"));
            names.push(vit_layer_tensor(layer_idx, "attn_out.weight"));
        }
        for &idx in flagged {
            for suffix in DEEPSTACK_REQUIRED_SUFFIXES {
                names.push(vit_deepstack_tensor(idx as usize, suffix));
            }
        }
        names
    }

    #[test]
    fn validate_qwen3vl_fused_attn_qkv_accepted() {
        // Wedge-4c.5: a Qwen3-VL mmproj with fused `attn_qkv.weight`
        // (canonical from llama.cpp's HF converter) must validate
        // cleanly. This is the green-flip companion to the loader's
        // slice-view extension.
        let cfg = qwen3vl_cfg(24, vec![5, 11, 17]);
        let names = qwen3vl_fused_qkv_tensor_names(24, &[5, 11, 17]);
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        validate_tensor_set(&cfg, &actual).expect(
            "Wedge-4c.5: tensor-complete Qwen3-VL with FUSED attn_qkv must validate cleanly",
        );
    }

    #[test]
    fn validate_qwen3vl_split_attn_qkv_still_accepted() {
        // Wedge-4c.5: the split form (legacy, hf2q-converted) must
        // remain accepted — both loader paths are equivalent. This is
        // a regression-guard against accidentally requiring fused.
        let cfg = qwen3vl_cfg(24, vec![5, 11, 17]);
        let names = qwen3vl_minimum_tensor_names(24, &[5, 11, 17]);
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        validate_tensor_set(&cfg, &actual)
            .expect("split attn_q/k/v.weight trio must remain a valid Qwen3-VL form");
    }

    #[test]
    fn validate_qwen3vl_mixed_qkv_form_rejected() {
        // Wedge-4c.5: a producer that emits BOTH fused AND split for
        // the same block is a bug — the loader can't deterministically
        // pick which to use. Validator must surface this loud.
        let cfg = qwen3vl_cfg(4, vec![]);
        let mut names = qwen3vl_minimum_tensor_names(4, &[]);
        // Add the fused tensor at block 0 — now block 0 has BOTH split
        // attn_q/k/v.weight AND fused attn_qkv.weight.
        names.push(vit_layer_tensor(0, "attn_qkv.weight"));
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let err = validate_tensor_set(&cfg, &actual)
            .expect_err("BOTH fused AND split present at the same block must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("BOTH fused") && msg.contains("attn_qkv"),
            "error must call out the both-forms case at block 0; got: {msg}"
        );
    }

    #[test]
    fn validate_qwen3vl_missing_qkv_names_fused_alternative() {
        // Wedge-4c.5: when QKV is entirely missing (neither fused nor
        // split), the error message must NAME the fused alternative
        // so a producer reading the diagnostic knows both forms are
        // accepted (avoids whack-a-mole adding split tensors when the
        // fused tensor would have been simpler).
        let cfg = qwen3vl_cfg(2, vec![]);
        // Only ship the universals + attn_out + projector head (no
        // QKV in any form).
        let mut names = vec![
            TENSOR_PATCH_EMBD.to_string(),
            TENSOR_POS_EMBD.to_string(),
            TENSOR_MM_0_WEIGHT.to_string(),
        ];
        for layer_idx in 0..2usize {
            names.push(vit_layer_tensor(layer_idx, "attn_out.weight"));
        }
        let actual: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let err = validate_tensor_set(&cfg, &actual)
            .expect_err("missing QKV in any form must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("attn_qkv") && msg.contains("attn_q/k/v"),
            "error must name BOTH the fused alternative ('attn_qkv') AND \
             the split form ('attn_q/k/v') so the producer can pick; got: {msg}"
        );
    }
}
