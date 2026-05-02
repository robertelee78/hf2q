//! Qwen3-VL ViT forward (ADR-005 iter-224 row 3 — Wedge-4c).
//!
//! **Status (sub-iter 4c.1)**: scaffold only — public types and a
//! single dispatch entry-point that returns `Err(...)`. The arithmetic
//! sub-iters land separately so that each can be reviewed under its
//! own /cfa cycle.
//!
//! # Sub-iter roadmap (sequential, all blocked on this 4c.1 scaffold)
//!
//! | Sub-iter | Scope                                                                    |
//! |----------|--------------------------------------------------------------------------|
//! | 4c.1     | **THIS** — `Qwen3VlViTConfig` + `compute_vision_embeddings_gpu_qwen3vl`  |
//! |          |   stub, dispatch-arm wired, `is_supported()` stays `false`               |
//! | 4c.2     | Patch embedding (Conv2d 16×16, dual stem) + position-embedding lookup    |
//! |          |   with bilinear interpolation from `num_position_embeddings`             |
//! | 4c.3     | Per-block forward: LN → 2D-RoPE Q,K → attn → +residual → LN → GELU MLP   |
//! |          |   → +residual; reuse `apply_layer_norm_gpu`, `apply_attention_gpu`       |
//! | 4c.4     | Per-flagged-layer DeepStack heads (`v.deepstack.{N}.{norm,fc1,fc2}`),    |
//! |          |   spatial 2×2 merge reshape, main projector (`mm.0/mm.2`), final         |
//! |          |   feature-dim concat producing `[hidden*(1+N_deepstack), n_image_tokens]`|
//! | 4c.5     | LM-side hooks in `Qwen35Model::forward_gpu_*_with_soft_tokens` to split  |
//! |          |   the augmented embedding and inject chunk `(il+1)` at LM layer `il <    |
//! |          |   N_deepstack` (post-FFN-residual `cur += ds`); flips                    |
//! |          |   `ProjectorType::Qwen3VlMerger.is_supported()` to `true`                |
//!
//! # Architecture reference (production-correct, source-side audited)
//!
//! Per the Wedge-4c R1 audit at
//! `/opt/hf2q/docs/research/wedge4c-deepstack-r1-audit-2026-05-01.md`,
//! Qwen3-VL DeepStack is **per-layer LM injection via concatenated-feature
//! transport**, NOT concat-along-sequence. The ViT-side outputs an
//! augmented embedding of shape `[hidden*(1+N_deepstack), n_image_tokens]`
//! which the LM splits at runtime and adds chunk `(il+1)` to the
//! post-FFN-residual at LM layer `il < N_deepstack`. See:
//!
//! - `/opt/llama.cpp/tools/mtmd/models/qwen3vl.cpp:1-193` (ViT graph)
//! - `/opt/llama.cpp/src/models/qwen3vl.cpp:96-100` (LM split-and-add)
//! - `/opt/llama.cpp/tools/mtmd/clip.cpp:3808-3809`
//!   (`embed_dim_per_image_token = mm_1_b->ne[0] * (1 + n_deepstack_layers)`)
//!
//! # Why a stub returns `Err` instead of `unimplemented!()`
//!
//! `unimplemented!()` would panic and abort the serve process if a
//! future code path inadvertently called it. An `Err` propagates up
//! through `compute_vision_embeddings_gpu_dispatch` to the chat handler
//! which converts it to a 500-class JSON error — the user gets a
//! diagnostic instead of an unrecoverable abort. This matches Wedge-4b's
//! existing fail-loud arm semantics in `vit_gpu.rs::dispatch`.

use anyhow::{anyhow, Result};

use super::mmproj::MmprojConfig;
use super::mmproj_weights::LoadedMmprojWeights;
use super::vit_gpu::VisionInput;

/// Static, immutable Qwen3-VL ViT shape configuration extracted at
/// dispatch time from the parsed `MmprojConfig` plus the loaded
/// position-embedding tensor.
///
/// All field values are u32/Vec — no `Option<_>` or string fallbacks —
/// because `from_mmproj` rejects loud (returns `Err`) on missing
/// required fields. Downstream sub-iters (4c.2/3/4) consume this as
/// a no-fallback shape contract.
///
/// # Field provenance
///
/// | Field                     | Source                                                  |
/// |---------------------------|---------------------------------------------------------|
/// | `n_layer`                 | `MmprojConfig.num_hidden_layers`                        |
/// | `n_embd`                  | `MmprojConfig.hidden_size`                              |
/// | `n_head`                  | `MmprojConfig.num_attention_heads`                      |
/// | `intermediate_size`       | `MmprojConfig.intermediate_size`                        |
/// | `patch_size`              | `MmprojConfig.patch_size`                               |
/// | `spatial_merge_size`      | `MmprojConfig.spatial_merge_size` (must be `Some`)      |
/// | `out_hidden_size`         | `MmprojConfig.projection_dim` (must be `Some`)          |
/// | `num_position_embeddings` | `LoadedMmprojWeights.position_embd_weight().shape()[0]` |
/// |                           |   — see note below; passed in explicitly                |
/// | `deepstack_indexes`       | `MmprojConfig.deepstack_indexes` (must be `Some`)       |
/// | `eps`                     | `MmprojConfig.layer_norm_eps`                           |
///
/// # Why `num_position_embeddings` is a separate parameter to `from_mmproj`
///
/// llama.cpp does NOT write `num_position_embeddings` as a metadata
/// key (verified via `grep -rn "num_position_embeddings" clip.cpp
/// clip-impl.h` — only the position embedding *tensor* itself
/// (`v.position_embd.weight`) carries the dimension as `ne[0]` per
/// `clip.cpp:3849`). Source-side that's a tensor-shape derivation,
/// not a config field. To keep `Qwen3VlViTConfig` honest about its
/// data flow, the dispatch site reads
/// `mmproj_weights.position_embd_weight()?.shape()[0]` and passes it
/// in explicitly — pretending to extract it from `MmprojConfig`
/// (which doesn't carry it) would be a Chesterton's-fence violation.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3VlViTConfig {
    /// ViT encoder layer count (e.g. 24 for Qwen3-VL-2B). Sourced
    /// from `MmprojConfig.num_hidden_layers`.
    pub n_layer: u32,
    /// ViT hidden dim (e.g. 1024 for Qwen3-VL-2B). Sourced from
    /// `MmprojConfig.hidden_size`.
    pub n_embd: u32,
    /// ViT attention head count. Sourced from
    /// `MmprojConfig.num_attention_heads`.
    pub n_head: u32,
    /// ViT FFN intermediate dim. Sourced from
    /// `MmprojConfig.intermediate_size`.
    pub intermediate_size: u32,
    /// Conv2d patch edge in pixels (Qwen3-VL: 16). Sourced from
    /// `MmprojConfig.patch_size`.
    pub patch_size: u32,
    /// Spatial-merge degree (Qwen3-VL: 2 → 2×2 merge → 4× token
    /// reduction). `MmprojConfig.spatial_merge_size` MUST be
    /// `Some(_)` — `from_mmproj` rejects loudly if `None`.
    pub spatial_merge_size: u32,
    /// LM hidden size that the projector targets — i.e., the
    /// `mm_1_b->ne[0]` in llama.cpp. For Qwen3-VL-2B: 2048
    /// (matches LM `hidden_size`). `MmprojConfig.projection_dim`
    /// MUST be `Some(_)` — `from_mmproj` rejects loudly if `None`.
    pub out_hidden_size: u32,
    /// Trained-resolution position-table length (e.g. 2304 for
    /// Qwen3-VL-2B at 768×768 / 16×16 = 48² + spec slack). Sourced
    /// from the LOADED `v.position_embd.weight` tensor's first dim
    /// (per `clip.cpp:3849`), not from a metadata key. Passed in
    /// explicitly to `from_mmproj` so the call-site is unambiguous
    /// about where the value came from.
    pub num_position_embeddings: u32,
    /// Sorted ascending list of layer indexes flagged in
    /// `clip.vision.is_deepstack_layers` (e.g. `[5, 11, 17]` for
    /// Qwen3-VL-2B). MUST be `Some(_)` —  `from_mmproj` rejects
    /// loudly if `None`. May be empty (length 0) if a future
    /// Qwen3-VL variant ships without DeepStack heads, but the
    /// metadata key MUST be present (typed `Some(empty Vec)`)
    /// because llama.cpp writes `Bool[block_count]` unconditionally
    /// for the qwen3vl_merger projector.
    pub deepstack_indexes: Vec<u32>,
    /// LayerNorm epsilon (NORM_TYPE_NORMAL in llama.cpp's clip
    /// graph; not RMS). Sourced from `MmprojConfig.layer_norm_eps`.
    pub eps: f32,
}

impl Qwen3VlViTConfig {
    /// Extract a `Qwen3VlViTConfig` from a parsed `MmprojConfig` plus
    /// a separately-sourced `num_position_embeddings`.
    ///
    /// # Fail-loud contract
    ///
    /// All three Qwen3-VL-only `Option<_>` fields on `MmprojConfig`
    /// MUST be `Some(_)`:
    ///
    /// - `spatial_merge_size`
    /// - `projection_dim`
    /// - `deepstack_indexes`
    ///
    /// A Qwen3-VL mmproj GGUF that arrives at this function without
    /// any of these set means the writer mis-encoded the file — we
    /// refuse to silently fall back to a wrong default and instead
    /// return a descriptive `Err` naming the missing field. The
    /// dispatch site catches this and surfaces it as a 500-class
    /// JSON error rather than a kernel-time NaN.
    ///
    /// `num_position_embeddings` MUST be `> 0` — a zero-length
    /// position table can't possibly produce valid bilinear-
    /// interpolation output, so we reject upfront.
    ///
    /// # Errors
    ///
    /// - `MmprojConfig.spatial_merge_size == None`
    /// - `MmprojConfig.projection_dim == None`
    /// - `MmprojConfig.deepstack_indexes == None`
    /// - `num_position_embeddings == 0`
    pub fn from_mmproj(
        cfg: &MmprojConfig,
        num_position_embeddings: u32,
    ) -> Result<Self> {
        let spatial_merge_size = cfg.spatial_merge_size.ok_or_else(|| {
            anyhow!(
                "Qwen3VlViTConfig: MmprojConfig.spatial_merge_size is None — \
                 a Qwen3-VL mmproj GGUF MUST write 'clip.vision.spatial_merge_size'. \
                 Refusing to silently fall back to a default."
            )
        })?;
        let out_hidden_size = cfg.projection_dim.ok_or_else(|| {
            anyhow!(
                "Qwen3VlViTConfig: MmprojConfig.projection_dim is None — \
                 a Qwen3-VL mmproj GGUF MUST write 'clip.vision.projection_dim'. \
                 Refusing to silently fall back to a default."
            )
        })?;
        let deepstack_indexes = cfg.deepstack_indexes.clone().ok_or_else(|| {
            anyhow!(
                "Qwen3VlViTConfig: MmprojConfig.deepstack_indexes is None — \
                 a Qwen3-VL mmproj GGUF MUST write 'clip.vision.is_deepstack_layers' \
                 as Bool[block_count]. Refusing to silently fall back."
            )
        })?;
        if num_position_embeddings == 0 {
            return Err(anyhow!(
                "Qwen3VlViTConfig: num_position_embeddings = 0 — must be > 0 \
                 (sourced from v.position_embd.weight tensor shape[0])"
            ));
        }
        // Sanity: every flagged index must be < n_layer. This is also
        // enforced by `read_deepstack_indexes` at parse time
        // (validates Bool[block_count] length matches block_count),
        // but we re-check here so the kernel-side iter-224 4c.3 path
        // can index `model.layers[idx]` without bounds checks.
        for &idx in &deepstack_indexes {
            if idx >= cfg.num_hidden_layers {
                return Err(anyhow!(
                    "Qwen3VlViTConfig: deepstack_indexes contains {} which is \
                     >= num_hidden_layers {} — this should have been caught at \
                     mmproj parse time",
                    idx,
                    cfg.num_hidden_layers
                ));
            }
        }
        Ok(Self {
            n_layer: cfg.num_hidden_layers,
            n_embd: cfg.hidden_size,
            n_head: cfg.num_attention_heads,
            intermediate_size: cfg.intermediate_size,
            patch_size: cfg.patch_size,
            spatial_merge_size,
            out_hidden_size,
            num_position_embeddings,
            deepstack_indexes,
            eps: cfg.layer_norm_eps,
        })
    }

    /// Per-image post-merge token count = `(image_size / (patch_size *
    /// spatial_merge_size))^2`. Helper for downstream sub-iters; not
    /// used by the 4c.1 stub.
    ///
    /// # Errors
    ///
    /// - `image_size` not divisible by `patch_size * spatial_merge_size`
    ///   (this should be guarded by preprocessing — Wedge-4d will resize
    ///   to a multiple of the merge stride; we still validate here so
    ///   the contract is enforced at the ViT entry point).
    pub fn n_image_tokens(&self, image_size: u32) -> Result<u32> {
        let stride = self.patch_size * self.spatial_merge_size;
        if stride == 0 {
            return Err(anyhow!(
                "Qwen3VlViTConfig::n_image_tokens: stride (patch_size * spatial_merge_size) = 0"
            ));
        }
        if image_size % stride != 0 {
            return Err(anyhow!(
                "Qwen3VlViTConfig::n_image_tokens: image_size {} not divisible by \
                 patch_size * spatial_merge_size = {}",
                image_size,
                stride
            ));
        }
        let side = image_size / stride;
        Ok(side * side)
    }

    /// Augmented per-image-token feature dim =
    /// `out_hidden_size * (1 + deepstack_indexes.len())`. Matches
    /// llama.cpp's `clip.cpp:3808-3809` formula. The LM-side hook
    /// (sub-iter 4c.5) splits this into `(1 + N_deepstack)` chunks
    /// of `out_hidden_size` each.
    pub fn augmented_embed_dim(&self) -> u32 {
        self.out_hidden_size * (1 + self.deepstack_indexes.len() as u32)
    }
}

/// Qwen3-VL ViT end-to-end GPU forward (sibling to
/// `compute_vision_embeddings_gpu_gemma4v` in `vit_gpu.rs`).
///
/// # Status (sub-iter 4c.1)
///
/// **STUB** — returns `Err(...)` naming sub-iters 4c.2..4c.5 which
/// fill in the arithmetic in order. The dispatch arm in
/// `vit_gpu.rs::compute_vision_embeddings_gpu_dispatch` delegates here
/// so subsequent sub-iters can land arithmetic without touching
/// dispatch wiring.
///
/// # Future contract (sub-iter 4c.4 closes this)
///
/// Returns one `Vec<f32>` per image, each with length
/// `n_image_tokens(image_size) * augmented_embed_dim()` row-major,
/// shape `[n_image_tokens, augmented_embed_dim] = [n_image_tokens,
/// out_hidden_size * (1 + N_deepstack)]`. Sub-iter 4c.5's LM hooks
/// then split each row into `(1 + N_deepstack)` chunks of
/// `out_hidden_size` and inject chunk `(il+1)` at LM layer
/// `il < N_deepstack` (post-FFN-residual `cur += ds`).
///
/// # Inputs (deferred — placeholder signature stable across sub-iters)
///
/// - `inputs`: heterogeneous `VisionInput` slice; only the
///   `Siglip49(_)` variant is valid for Qwen3-VL preprocessing
///   today (Wedge-4d may add a `Qwen3Vl(_)` variant if dynamic
///   resolution lands; until then the same square fixed-resolution
///   pipeline used for SigLIP-49 feeds Qwen3-VL).
/// - `mmproj_weights`: loaded ViT tower weights (patch+pos embeds,
///   per-block QKV/FFN, per-flagged-layer DeepStack heads, mm.0/mm.2
///   projector).
/// - `cfg`: shape contract from `Qwen3VlViTConfig::from_mmproj`.
/// - `mmproj_cfg`: still passed for `image_size` / `image_mean/std`
///   bookkeeping that's MmprojConfig-only.
///
/// # Errors
///
/// Sub-iter 4c.1: ALWAYS returns `Err` with a message naming the
/// pending sub-iter coverage. Never panics.
pub fn compute_vision_embeddings_gpu_qwen3vl(
    _inputs: &[VisionInput],
    _mmproj_weights: &LoadedMmprojWeights,
    _cfg: &Qwen3VlViTConfig,
    _mmproj_cfg: &MmprojConfig,
) -> Result<Vec<Vec<f32>>> {
    Err(anyhow!(
        "Wedge-4c.2/3/4: Qwen3-VL ViT forward arithmetic not yet implemented \
         (4c.1 scaffold only). Subsequent sub-iters fill in: \
         4c.2 patch_embedding+position_embedding, \
         4c.3 per-block forward (LN+2D-RoPE+attn+MLP), \
         4c.4 deepstack heads+spatial merger+main projector, \
         4c.5 LM-side split-and-add hooks (then \
         ProjectorType::Qwen3VlMerger.is_supported() flips to true)"
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::vision::mmproj::ProjectorType;

    /// Construct a synthetic Qwen3-VL `MmprojConfig` matching the
    /// shape conventions from `tests/mmproj_qwen3vl.rs` /
    /// `mmproj.rs::tests::qwen3vl_cfg`. We don't import that helper
    /// because it lives behind `#[cfg(test)] mod tests` which is
    /// crate-private — easier to mirror its 13 lines than wire a
    /// `pub(crate)` accessor that doesn't pay rent outside this file.
    fn synth_qwen3vl_mmproj_cfg(
        num_layers: u32,
        deepstack_indexes: Option<Vec<u32>>,
        spatial_merge_size: Option<u32>,
        projection_dim: Option<u32>,
    ) -> MmprojConfig {
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
            spatial_merge_size,
            projection_dim,
            deepstack_indexes,
        }
    }

    #[test]
    fn qwen3vl_vit_config_from_mmproj_round_trip() {
        // Qwen3-VL-2B-style fixture: 24 layers, DeepStack at 5/11/17.
        let cfg = synth_qwen3vl_mmproj_cfg(
            24,
            Some(vec![5, 11, 17]),
            Some(2),
            Some(2048),
        );
        let vit = Qwen3VlViTConfig::from_mmproj(&cfg, 2304)
            .expect("from_mmproj on a fully-populated Qwen3-VL config must succeed");
        assert_eq!(vit.n_layer, 24);
        assert_eq!(vit.n_embd, 1024);
        assert_eq!(vit.n_head, 16);
        assert_eq!(vit.intermediate_size, 4304);
        assert_eq!(vit.patch_size, 16);
        assert_eq!(vit.spatial_merge_size, 2);
        assert_eq!(vit.out_hidden_size, 2048);
        assert_eq!(vit.num_position_embeddings, 2304);
        assert_eq!(vit.deepstack_indexes, vec![5, 11, 17]);
        assert!((vit.eps - 1e-6).abs() < 1e-12);
        // Augmented embed dim = 2048 * (1 + 3) = 8192 (matches
        // /opt/llama.cpp/tools/mtmd/clip.cpp:3808-3809 for Qwen3-VL-2B).
        assert_eq!(vit.augmented_embed_dim(), 8192);
        // n_image_tokens at 768×768 / (16*2)² = (768/32)² = 24² = 576.
        assert_eq!(vit.n_image_tokens(768).unwrap(), 576);
    }

    #[test]
    fn qwen3vl_vit_config_fails_on_missing_deepstack_indexes() {
        // deepstack_indexes = None means the writer omitted
        // `clip.vision.is_deepstack_layers`. Refuse to fall back.
        let cfg = synth_qwen3vl_mmproj_cfg(
            24,
            None, // ← the missing field under test
            Some(2),
            Some(2048),
        );
        let err = Qwen3VlViTConfig::from_mmproj(&cfg, 2304)
            .expect_err("from_mmproj must reject a config with deepstack_indexes=None");
        let msg = format!("{err}");
        assert!(
            msg.contains("deepstack_indexes"),
            "error must name the missing 'deepstack_indexes' field; got: {msg}"
        );
        assert!(
            msg.contains("is_deepstack_layers") || msg.contains("None"),
            "error must hint at the missing GGUF metadata key or the None state; got: {msg}"
        );
    }

    #[test]
    fn qwen3vl_vit_config_fails_on_missing_spatial_merge_size() {
        let cfg = synth_qwen3vl_mmproj_cfg(
            24,
            Some(vec![5, 11, 17]),
            None, // ← the missing field under test
            Some(2048),
        );
        let err = Qwen3VlViTConfig::from_mmproj(&cfg, 2304)
            .expect_err("from_mmproj must reject a config with spatial_merge_size=None");
        assert!(format!("{err}").contains("spatial_merge_size"));
    }

    #[test]
    fn qwen3vl_vit_config_fails_on_missing_projection_dim() {
        let cfg = synth_qwen3vl_mmproj_cfg(
            24,
            Some(vec![5, 11, 17]),
            Some(2),
            None, // ← the missing field under test
        );
        let err = Qwen3VlViTConfig::from_mmproj(&cfg, 2304)
            .expect_err("from_mmproj must reject a config with projection_dim=None");
        assert!(format!("{err}").contains("projection_dim"));
    }

    #[test]
    fn qwen3vl_vit_config_fails_on_zero_num_position_embeddings() {
        let cfg = synth_qwen3vl_mmproj_cfg(
            24,
            Some(vec![5, 11, 17]),
            Some(2),
            Some(2048),
        );
        let err = Qwen3VlViTConfig::from_mmproj(&cfg, 0)
            .expect_err("from_mmproj must reject num_position_embeddings=0");
        assert!(format!("{err}").contains("num_position_embeddings"));
    }

    #[test]
    fn qwen3vl_vit_config_fails_on_out_of_range_deepstack_index() {
        // deepstack_indexes contains 24 but n_layer = 24 → 24 is OOB.
        // Sanity-check that should have been caught at parse time but
        // we belt-and-suspender re-validate here so the kernel-side
        // 4c.3 path can index `model.layers[idx]` without bounds checks.
        let cfg = synth_qwen3vl_mmproj_cfg(
            24,
            Some(vec![5, 11, 24]),
            Some(2),
            Some(2048),
        );
        let err = Qwen3VlViTConfig::from_mmproj(&cfg, 2304)
            .expect_err("from_mmproj must reject deepstack_indexes >= num_hidden_layers");
        let msg = format!("{err}");
        assert!(
            msg.contains("deepstack_indexes") && msg.contains("num_hidden_layers"),
            "error must name both fields; got: {msg}"
        );
    }

    #[test]
    fn qwen3vl_vit_config_accepts_empty_deepstack_indexes() {
        // Some(empty Vec) is legal — the metadata key was present but
        // no layer was flagged. The augmented embed dim collapses to
        // out_hidden_size (1 + 0 deepstack chunks).
        let cfg = synth_qwen3vl_mmproj_cfg(
            24,
            Some(vec![]),
            Some(2),
            Some(2048),
        );
        let vit = Qwen3VlViTConfig::from_mmproj(&cfg, 2304)
            .expect("Some(empty Vec) is a valid deepstack_indexes value");
        assert_eq!(vit.deepstack_indexes, Vec::<u32>::new());
        assert_eq!(vit.augmented_embed_dim(), 2048);
    }

    #[test]
    fn compute_vision_embeddings_gpu_qwen3vl_returns_scaffold_error() {
        // 4c.1 stub contract: ALWAYS Err, message names the pending
        // sub-iters. We construct minimum stub args — the function
        // is supposed to return Err before touching them.
        let cfg = synth_qwen3vl_mmproj_cfg(
            24,
            Some(vec![5, 11, 17]),
            Some(2),
            Some(2048),
        );
        let vit_cfg = Qwen3VlViTConfig::from_mmproj(&cfg, 2304).unwrap();
        // `LoadedMmprojWeights` has no public `Default` / `new()` —
        // its construction goes through the GGUF loader. For a stub
        // that returns Err before reading weights, an empty slice +
        // a dangling reference would still trigger compile-side
        // borrow checks. We use a `Result<()>::Err` round-trip
        // instead: directly call the function via a tiny helper
        // that conjures a no-op `LoadedMmprojWeights` is overkill —
        // simpler to inline-skip the weights arg by exercising the
        // public API indirectly through the dispatch.
        //
        // Since the function is `pub`, we call it directly with a
        // `LoadedMmprojWeights` synthesized via the same loader the
        // test fixtures use. The only requirement on the loader is
        // that it produces *some* valid handle — its contents are
        // never read by the 4c.1 stub. Use the existing
        // `LoadedMmprojWeights::empty_for_test()` helper if it
        // exists; otherwise skip the direct call and assert the Err
        // message via a thin wrapper.
        //
        // Implementation: we rely on the simplest path — construct
        // an empty `Vec<VisionInput>` and verify the function still
        // returns the scaffold Err (it returns Err before iterating
        // inputs). We pass a minimal `LoadedMmprojWeights` via a
        // helper below.
        let weights = make_empty_loaded_mmproj_weights();
        let result =
            compute_vision_embeddings_gpu_qwen3vl(&[], &weights, &vit_cfg, &cfg);
        let err = result.expect_err(
            "4c.1 scaffold must always return Err, regardless of input slice content",
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("4c.1 scaffold only"),
            "error message must self-identify as the 4c.1 scaffold; got: {msg}"
        );
        assert!(
            msg.contains("4c.2") && msg.contains("4c.3") && msg.contains("4c.4") && msg.contains("4c.5"),
            "error message must enumerate the pending sub-iters 4c.2/3/4/5; got: {msg}"
        );
    }

    /// Construct a minimal `LoadedMmprojWeights` for unit tests of
    /// stubs that never read its fields. Uses the existing
    /// `LoadedMmprojWeights::empty(device)` constructor that other
    /// mmproj tests in this crate already use (see
    /// `mmproj_weights.rs:220`). The 4c.1 stub returns `Err` before
    /// touching any tensor, so the zero-tensor map is sufficient.
    ///
    /// Sub-iters 4c.2+ will replace this with a real loader-backed
    /// fixture once arithmetic exists to consume the tensors.
    fn make_empty_loaded_mmproj_weights() -> LoadedMmprojWeights {
        let device =
            mlx_native::MlxDevice::new().expect("MlxDevice::new() for unit test");
        LoadedMmprojWeights::empty(device)
    }
}
