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
use crate::serve::api::engine::SoftTokenData;
use crate::serve::api::state::LoadedMmproj;
use tokenizers::Tokenizer;

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

/// Expand image-placeholder tokens in `prompt_tokens` into per-image
/// runs of placeholder ids, allocate a per-image GPU buffer carrying
/// the BASE-chunk projected vision embeddings, and produce
/// `SoftTokenData` slots so the prefill call can override the
/// per-position embed at exactly those ranges.
///
/// Mirrors `serve::api::handlers::expand_image_placeholders_family`
/// extracted iter-2 of mmproj-on-generate so both SERVE and CLI run
/// one path. Returns:
/// - `prompt_expanded`: `Vec<u32>` with each placeholder token expanded
///   into `n_image_tokens` consecutive copies of the placeholder id.
/// - `soft_tokens`: per-image `SoftTokenData` (owned `MlxBuffer`,
///   half-open range matching the expansion).
/// - `image_token_positions`: per-image absolute positions in
///   `prompt_expanded` (used by Qwen3-VL DeepStack injection).
///
/// Pre-conditions:
/// - `embeddings.len() == n_images`.
/// - Every `embeddings[i].len()` is a positive multiple of
///   `per_row_floats` (caller guaranteed via `run_vit_forward`).
/// - `tokenizer.token_to_id(family.placeholder_token_literal())`
///   resolves to a valid id (otherwise the loaded model doesn't ship
///   the vision placeholder token).
///
/// Errors:
/// - The tokenizer doesn't have the family's placeholder special
///   token (the loaded chat model can't carry vision soft tokens).
/// - The rendered prompt's placeholder count doesn't match `n_images`
///   (chat template dropped or duplicated image markers).
/// - GPU buffer alloc / mut-slice access fails.
pub fn expand_image_placeholders(
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    embeddings: &[Vec<f32>],
    family: VisionFamily,
    per_row_floats: usize,
    hidden_size: usize,
) -> Result<(Vec<u32>, Vec<SoftTokenData>, Vec<Vec<u32>>)> {
    let n_images = embeddings.len();
    if hidden_size == 0 || per_row_floats == 0 {
        return Err(anyhow!(
            "expand_image_placeholders: degenerate hidden ({hidden_size}) or \
             per_row_floats ({per_row_floats})"
        ));
    }
    let placeholder_literal = family
        .placeholder_token_literal()
        .ok_or_else(|| {
            anyhow!(
                "expand_image_placeholders: VisionFamily::{:?} has no placeholder \
                 token literal — caller should have rejected this profile upstream",
                family
            )
        })?;
    let img_token_id: u32 = tokenizer
        .token_to_id(placeholder_literal)
        .ok_or_else(|| {
            anyhow!(
                "tokenizer has no `{placeholder_literal}` special-token id; the \
                 loaded chat model does not support vision input through hf2q's \
                 soft-token path"
            )
        })?;
    let placeholder_positions: Vec<usize> = prompt_tokens
        .iter()
        .enumerate()
        .filter_map(|(p, t)| if *t == img_token_id { Some(p) } else { None })
        .collect();
    if placeholder_positions.len() != n_images {
        return Err(anyhow!(
            "rendered prompt has {} `{placeholder_literal}` placeholder(s) but \
             request carries {} image(s); the chat template likely dropped or \
             duplicated image markers — check `tokenizer_config.json` and the \
             GGUF chat template",
            placeholder_positions.len(),
            n_images
        ));
    }
    for (i, e) in embeddings.iter().enumerate() {
        if e.len() % per_row_floats != 0 || e.is_empty() {
            return Err(anyhow!(
                "vision embedding [{i}] length {} is not a positive multiple \
                 of per_row_floats {per_row_floats} (family={family:?}, \
                 hidden={hidden_size})",
                e.len()
            ));
        }
    }
    let mlx_dev = mlx_native::MlxDevice::new()
        .map_err(|e| anyhow!("MlxDevice init failed: {e}"))?;
    let total_extra: usize = embeddings
        .iter()
        .map(|e| e.len() / per_row_floats)
        .sum::<usize>()
        .saturating_sub(n_images);
    let mut prompt_expanded: Vec<u32> = Vec::with_capacity(prompt_tokens.len() + total_extra);
    let mut soft_tokens: Vec<SoftTokenData> = Vec::with_capacity(n_images);
    let mut image_token_positions: Vec<Vec<u32>> = Vec::with_capacity(n_images);
    let mut last_pos = 0usize;
    for (i, &pos) in placeholder_positions.iter().enumerate() {
        prompt_expanded.extend_from_slice(&prompt_tokens[last_pos..pos]);
        let n_image_tokens = embeddings[i].len() / per_row_floats;
        let start = prompt_expanded.len();
        for _ in 0..n_image_tokens {
            prompt_expanded.push(img_token_id);
        }
        let end = prompt_expanded.len();
        let byte_len = n_image_tokens * hidden_size * std::mem::size_of::<f32>();
        let mut buf = mlx_dev
            .alloc_buffer(
                byte_len,
                mlx_native::DType::F32,
                vec![n_image_tokens, hidden_size],
            )
            .map_err(|e| anyhow!("soft-token buffer alloc failed (image {i}): {e}"))?;
        {
            let dst = buf
                .as_mut_slice::<f32>()
                .map_err(|e| anyhow!("soft-token buffer mut slice failed (image {i}): {e}"))?;
            debug_assert_eq!(dst.len(), n_image_tokens * hidden_size);
            if per_row_floats == hidden_size {
                dst.copy_from_slice(&embeddings[i]);
            } else {
                for row in 0..n_image_tokens {
                    let src_base = row * per_row_floats;
                    let dst_base = row * hidden_size;
                    dst[dst_base..dst_base + hidden_size]
                        .copy_from_slice(&embeddings[i][src_base..src_base + hidden_size]);
                }
            }
        }
        soft_tokens.push(SoftTokenData {
            range: start..end,
            embeddings: buf,
        });
        image_token_positions.push((start..end).map(|p| p as u32).collect());
        last_pos = pos + 1;
    }
    prompt_expanded.extend_from_slice(&prompt_tokens[last_pos..]);
    Ok((prompt_expanded, soft_tokens, image_token_positions))
}
