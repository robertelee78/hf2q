//! Shared vision pipeline — ViT forward + family-derived stride bookkeeping.
//!
//! Extracted from `serve::api::handlers` so both the SERVE chat handler
//! (`/v1/chat/completions` with `image_url` content parts) and the CLI
//! `hf2q generate --image` flow exercise one path. The split is deliberate:
//! - The handler decides chat-message rewriting (`<|image|>` placeholder
//!   insertion) and prompt rendering, which is request-shape-specific.
//! - This module owns the request-shape-agnostic ViT forward, family-aware
//!   embedding-stride computation, and Qwen3-VL per-image grid derivation,
//!   given a `Vec<VisionInput>` and the loaded mmproj.
//!
//! The contract is bidirectional: any change to handler-side validation
//! semantics (e.g. arch-profile rejection) must also update the matching
//! site here so SERVE and CLI cannot diverge silently.

use anyhow::{anyhow, Context as _, Result};

use crate::inference::vision::mmproj::VisionFamily;
use crate::inference::vision::vit_gpu::{compute_vision_embeddings_gpu_dispatch, VisionInput};
use crate::serve::api::state::LoadedMmproj;

/// Output of a single ViT GPU forward pass over a batch of preprocessed
/// images, with all the family-derived bookkeeping the downstream
/// soft-token splice needs.
#[derive(Debug, Clone)]
pub struct VisionPipelineOutput {
    /// One projected embedding tensor per input image. For Gemma4-family
    /// arches each tensor has length `n_image_tokens * hidden`. For
    /// Qwen3-VL the per-row stride includes DeepStack residuals so each
    /// tensor has length `n_image_tokens * hidden * (1 + N_deepstack)`.
    pub embeddings: Vec<Vec<f32>>,
    /// Vision family tag (Gemma vs Qwen3Vl vs Unknown) — drives prompt
    /// placeholder syntax + token-id lookup.
    pub family: VisionFamily,
    /// Per-image-token stride in `f32` units. Always a positive divisor
    /// of every entry in `embeddings`. Validated at construction.
    pub per_row_floats: usize,
    /// Per-image post-merge token grid `(n_x_tokens, n_y_tokens)`. Empty
    /// for non-Qwen3-VL families (Gemma routes through patch grid only).
    pub qwen3vl_image_grids: Vec<(u32, u32)>,
    /// Wall-clock ms spent inside the GPU dispatch.
    pub forward_ms: u64,
}

impl VisionPipelineOutput {
    /// Number of soft tokens contributed by image `i`. Equal to
    /// `embeddings[i].len() / per_row_floats`.
    pub fn n_image_tokens(&self, i: usize) -> usize {
        self.embeddings[i].len() / self.per_row_floats
    }

    /// Total soft tokens across all images.
    pub fn total_image_tokens(&self) -> usize {
        (0..self.embeddings.len())
            .map(|i| self.n_image_tokens(i))
            .sum()
    }
}

/// Run the GPU ViT forward over a batch of preprocessed `VisionInput`s
/// and return the projected embeddings + family-derived bookkeeping.
///
/// Errors:
/// - GPU dispatch failure (kernel compile error, OOM, NaN guard).
/// - Validation failure: any embedding tensor whose `len()` is not a
///   positive multiple of `per_row_floats` (would prevent unique
///   inversion to `n_image_tokens`).
///
/// The arch-profile-supported check is the caller's responsibility:
/// `LoadedMmproj` is expected to have been built by a load path that
/// already rejected `ArchProfile::Unknown`.
pub fn run_vit_forward(
    preprocessed_inputs: &[VisionInput],
    mmproj: &LoadedMmproj,
    hidden_size: usize,
) -> Result<VisionPipelineOutput> {
    if preprocessed_inputs.is_empty() {
        return Ok(VisionPipelineOutput {
            embeddings: Vec::new(),
            family: mmproj.arch.vision_family(),
            per_row_floats: hidden_size,
            qwen3vl_image_grids: Vec::new(),
            forward_ms: 0,
        });
    }

    let head_dim_f =
        (mmproj.config.hidden_size / mmproj.config.num_attention_heads) as f32;
    let scale = 1.0f32 / head_dim_f.sqrt();
    let t0 = std::time::Instant::now();
    let embeddings = compute_vision_embeddings_gpu_dispatch(
        preprocessed_inputs,
        mmproj.arch,
        &mmproj.weights,
        &mmproj.config,
        scale,
    )
    .context("ViT GPU forward failed")?;
    let forward_ms = t0.elapsed().as_millis() as u64;

    let family = mmproj.arch.vision_family();
    let n_deepstack = mmproj
        .config
        .deepstack_indexes
        .as_ref()
        .map(|v| v.len())
        .unwrap_or(0);
    let per_row_floats = match family {
        VisionFamily::Gemma => hidden_size,
        VisionFamily::Qwen3Vl => hidden_size.saturating_mul(1 + n_deepstack),
        VisionFamily::Unknown => hidden_size,
    };

    for (i, e) in embeddings.iter().enumerate() {
        if per_row_floats == 0 || e.is_empty() || e.len() % per_row_floats != 0 {
            return Err(anyhow!(
                "vision embedding [{i}] length {} is not a positive multiple \
                 of per_row_floats {per_row_floats} (family={family:?}, \
                 hidden={hidden_size}, n_deepstack={n_deepstack})",
                e.len()
            ));
        }
    }

    let qwen3vl_image_grids: Vec<(u32, u32)> = if matches!(family, VisionFamily::Qwen3Vl) {
        let stride = mmproj
            .config
            .patch_size
            .saturating_mul(mmproj.config.spatial_merge_size.unwrap_or(1));
        preprocessed_inputs
            .iter()
            .map(|input| match input {
                VisionInput::Siglip49(p) => {
                    let (pw, ph) = p.pixel_grid();
                    let nx = if stride > 0 { pw / stride } else { 0 };
                    let ny = if stride > 0 { ph / stride } else { 0 };
                    (nx, ny)
                }
                VisionInput::Gemma4v(_) => (0, 0),
            })
            .collect()
    } else {
        Vec::new()
    };

    let n_images = embeddings.len();
    let embed_dim = embeddings.first().map(|e| e.len()).unwrap_or(0);
    tracing::info!(
        n_images,
        embed_dim,
        forward_ms,
        arch = mmproj.arch.as_str(),
        "Vision embeddings computed via GPU ViT forward"
    );

    Ok(VisionPipelineOutput {
        embeddings,
        family,
        per_row_floats,
        qwen3vl_image_grids,
        forward_ms,
    })
}
