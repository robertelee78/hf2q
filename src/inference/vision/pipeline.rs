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
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::inference::vision::mmproj::VisionFamily;
use crate::inference::vision::vit_gpu::{compute_vision_embeddings_gpu_dispatch, VisionInput};
use crate::serve::api::engine::SoftTokenData;
use crate::serve::api::state::LoadedMmproj;
use tokenizers::Tokenizer;

pub const DEFAULT_VISION_EMBEDDING_CACHE_BYTES: usize = 512 * 1024 * 1024;

/// Output of a single ViT GPU forward pass over a batch of preprocessed
/// images, with all the family-derived bookkeeping the downstream
/// soft-token splice needs.
#[derive(Debug, Clone)]
pub struct VisionPipelineOutput {
    /// Stable identity of the exact post-preprocessing tensors and geometry
    /// consumed by the vision graph. Unlike projected embeddings, this is
    /// independent of harmless GPU floating-point scheduling differences and
    /// is therefore safe to use for prompt/KV cache affinity.
    pub input_fingerprint: [u8; 32],
    /// One projected embedding tensor per input image. For Gemma4-family
    /// arches each tensor has length `n_image_tokens * hidden`. For
    /// Qwen3-VL the per-row stride includes DeepStack residuals so each
    /// tensor has length `n_image_tokens * hidden * (1 + N_deepstack)`.
    pub embeddings: Arc<Vec<Vec<f32>>>,
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

/// Projector-local, bounded cache entry for one exact preprocessed image
/// batch. `output.forward_ms` records the original computation; cache hits
/// return a clone with `forward_ms = 0` so request telemetry distinguishes
/// reuse from a fresh GPU forward.
#[derive(Debug, Clone)]
pub struct VisionEmbeddingCacheEntry {
    key: [u8; 32],
    output: VisionPipelineOutput,
    resident_bytes: usize,
}

#[derive(Debug, Default)]
struct VisionEmbeddingCacheState {
    entries: VecDeque<VisionEmbeddingCacheEntry>,
    resident_bytes: usize,
    in_flight: Option<[u8; 32]>,
}

/// Projector-local bounded cache and compute gate. Embedding payloads are
/// Arc-owned, cache hits are pointer-cheap, and only one vision forward is
/// live per projector so concurrent large images cannot multiply workspace
/// without a bound.
#[derive(Debug)]
pub struct VisionEmbeddingCache {
    byte_budget: usize,
    state: Mutex<VisionEmbeddingCacheState>,
    wake: Condvar,
}

impl VisionEmbeddingCache {
    pub fn new(byte_budget: usize) -> Self {
        Self {
            byte_budget,
            state: Mutex::new(VisionEmbeddingCacheState::default()),
            wake: Condvar::new(),
        }
    }

    fn reserve_or_hit(
        &self,
        key: [u8; 32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Option<VisionPipelineOutput>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("vision embedding cache lock poisoned"))?;
        loop {
            anyhow::ensure!(
                !cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)),
                "request_cancelled: vision preparation was abandoned"
            );
            if let Some(position) = state.entries.iter().position(|entry| entry.key == key) {
                let hit = state
                    .entries
                    .remove(position)
                    .expect("vision cache hit position must exist");
                let mut output = hit.output.clone();
                output.forward_ms = 0;
                tracing::debug!(
                    resident_bytes = hit.resident_bytes,
                    "Vision cache payload reused"
                );
                state.entries.push_back(hit);
                return Ok(Some(output));
            }
            if state.in_flight.is_none() {
                state.in_flight = Some(key);
                return Ok(None);
            }
            let (next, _) = self
                .wake
                .wait_timeout(state, Duration::from_millis(25))
                .map_err(|_| anyhow!("vision embedding cache lock poisoned while waiting"))?;
            state = next;
        }
    }

    fn complete(&self, key: [u8; 32], output: Option<&VisionPipelineOutput>) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("vision embedding cache lock poisoned"))?;
        anyhow::ensure!(
            state.in_flight == Some(key),
            "vision embedding cache completion key does not own the compute gate"
        );
        if let Some(output) = output {
            let resident_bytes =
                match output
                    .embeddings
                    .iter()
                    .try_fold(0usize, |total, embedding| {
                        embedding
                            .len()
                            .checked_mul(std::mem::size_of::<f32>())
                            .and_then(|bytes| total.checked_add(bytes))
                            .ok_or_else(|| anyhow!("vision embedding resident-byte count overflow"))
                    }) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        state.in_flight = None;
                        self.wake.notify_all();
                        return Err(error);
                    }
                };
            if resident_bytes <= self.byte_budget {
                if let Some(position) = state.entries.iter().position(|entry| entry.key == key) {
                    if let Some(replaced) = state.entries.remove(position) {
                        state.resident_bytes =
                            state.resident_bytes.saturating_sub(replaced.resident_bytes);
                    }
                }
                while state
                    .resident_bytes
                    .checked_add(resident_bytes)
                    .is_none_or(|total| total > self.byte_budget)
                {
                    let Some(evicted) = state.entries.pop_front() else {
                        break;
                    };
                    state.resident_bytes =
                        state.resident_bytes.saturating_sub(evicted.resident_bytes);
                }
                state.entries.push_back(VisionEmbeddingCacheEntry {
                    key,
                    output: output.clone(),
                    resident_bytes,
                });
                state.resident_bytes += resident_bytes;
            }
        }
        state.in_flight = None;
        self.wake.notify_all();
        Ok(())
    }

    #[cfg(test)]
    fn resident_bytes(&self) -> usize {
        self.state
            .lock()
            .expect("vision cache test lock")
            .resident_bytes
    }
}

fn hash_u32(h: &mut Sha256, value: u32) {
    h.update(value.to_le_bytes());
}

/// Hash the exact post-preprocessing tensors and geometry consumed by the
/// vision GPU graph. Debug labels and source URLs are deliberately excluded:
/// they do not affect inference. Variant tags and all shape/position fields
/// are included so differently interpreted byte-identical buffers cannot
/// alias.
fn vision_input_cache_key(inputs: &[VisionInput], hidden_size: usize) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"hf2q-vision-input-v1\0");
    h.update((hidden_size as u64).to_le_bytes());
    h.update((inputs.len() as u64).to_le_bytes());
    for input in inputs {
        match input {
            VisionInput::Siglip49(p) => {
                h.update([0]);
                hash_u32(&mut h, p.target_size);
                hash_u32(&mut h, p.pixel_w.unwrap_or(0));
                hash_u32(&mut h, p.pixel_h.unwrap_or(0));
                h.update((p.pixel_values.len() as u64).to_le_bytes());
                for value in &p.pixel_values {
                    h.update(value.to_bits().to_le_bytes());
                }
            }
            VisionInput::Gemma4v(p) => {
                h.update([1]);
                hash_u32(&mut h, p.n_x);
                hash_u32(&mut h, p.n_y);
                h.update((p.pos_x.len() as u64).to_le_bytes());
                for value in &p.pos_x {
                    hash_u32(&mut h, *value);
                }
                h.update((p.pos_y.len() as u64).to_le_bytes());
                for value in &p.pos_y {
                    hash_u32(&mut h, *value);
                }
                h.update((p.patches.len() as u64).to_le_bytes());
                for value in &p.patches {
                    h.update(value.to_bits().to_le_bytes());
                }
            }
        }
    }
    h.finalize().into()
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
    run_vit_forward_cancellable(preprocessed_inputs, mmproj, hidden_size, None)
}

pub fn run_vit_forward_cancellable(
    preprocessed_inputs: &[VisionInput],
    mmproj: &LoadedMmproj,
    hidden_size: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<VisionPipelineOutput> {
    anyhow::ensure!(
        !cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)),
        "request_cancelled: vision preparation was abandoned"
    );
    if preprocessed_inputs.is_empty() {
        return Ok(VisionPipelineOutput {
            input_fingerprint: vision_input_cache_key(preprocessed_inputs, hidden_size),
            embeddings: Arc::new(Vec::new()),
            family: mmproj.arch.vision_family(),
            per_row_floats: hidden_size,
            qwen3vl_image_grids: Vec::new(),
            forward_ms: 0,
        });
    }

    let cache_key = vision_input_cache_key(preprocessed_inputs, hidden_size);
    if let Some(output) = mmproj.vision_cache.reserve_or_hit(cache_key, cancelled)? {
        tracing::info!(
            n_images = output.embeddings.len(),
            arch = mmproj.arch.as_str(),
            "Vision embedding cache hit"
        );
        return Ok(output);
    }

    let head_dim_f = (mmproj.config.hidden_size / mmproj.config.num_attention_heads) as f32;
    let scale = 1.0f32 / head_dim_f.sqrt();
    let t0 = std::time::Instant::now();
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        mmproj.vision_cache.complete(cache_key, None)?;
        anyhow::bail!("request_cancelled: vision preparation was abandoned");
    }
    let computed = (|| -> Result<VisionPipelineOutput> {
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
            input_fingerprint: cache_key,
            embeddings: Arc::new(embeddings),
            family,
            per_row_floats,
            qwen3vl_image_grids,
            forward_ms,
        })
    })();
    let was_cancelled = cancelled.is_some_and(|flag| flag.load(Ordering::Acquire));
    mmproj.vision_cache.complete(
        cache_key,
        if was_cancelled {
            None
        } else {
            computed.as_ref().ok()
        },
    )?;
    if was_cancelled {
        anyhow::bail!("request_cancelled: vision preparation was abandoned");
    }
    computed
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
    let placeholder_literal = family.placeholder_token_literal().ok_or_else(|| {
        anyhow!(
            "expand_image_placeholders: VisionFamily::{:?} has no placeholder \
                 token literal — caller should have rejected this profile upstream",
            family
        )
    })?;
    let img_token_id: u32 = tokenizer.token_to_id(placeholder_literal).ok_or_else(|| {
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
    let mlx_dev =
        mlx_native::MlxDevice::new().map_err(|e| anyhow!("MlxDevice init failed: {e}"))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::vision::vit_gpu::Gemma4vPreprocessedImage;
    use crate::inference::vision::PreprocessedImage;

    fn siglip(values: Vec<f32>) -> VisionInput {
        VisionInput::Siglip49(PreprocessedImage {
            pixel_values: values,
            target_size: 2,
            pixel_w: Some(2),
            pixel_h: Some(1),
            source_label: "ignored-by-cache-key".into(),
        })
    }

    #[test]
    fn vision_embedding_cache_key_uses_exact_tensor_not_source_label() {
        let a = siglip(vec![0.0, 1.0, 2.0]);
        let mut b = match a.clone() {
            VisionInput::Siglip49(p) => p,
            VisionInput::Gemma4v(_) => unreachable!(),
        };
        b.source_label = "different-url-same-pixels".into();
        assert_eq!(
            vision_input_cache_key(&[a], 16),
            vision_input_cache_key(&[VisionInput::Siglip49(b)], 16)
        );
    }

    #[test]
    fn vision_embedding_cache_key_misses_changed_pixels_shape_and_hidden() {
        let base = siglip(vec![0.0, 1.0, 2.0]);
        let changed = siglip(vec![0.0, 1.0, 2.5]);
        let mut reshaped = match base.clone() {
            VisionInput::Siglip49(p) => p,
            VisionInput::Gemma4v(_) => unreachable!(),
        };
        reshaped.pixel_w = Some(1);
        reshaped.pixel_h = Some(2);
        let base_key = vision_input_cache_key(&[base.clone()], 16);
        assert_ne!(base_key, vision_input_cache_key(&[changed], 16));
        assert_ne!(
            base_key,
            vision_input_cache_key(&[VisionInput::Siglip49(reshaped)], 16)
        );
        assert_ne!(base_key, vision_input_cache_key(&[base], 32));
    }

    #[test]
    fn vision_embedding_cache_key_separates_input_families_and_positions() {
        let siglip_key = vision_input_cache_key(&[siglip(vec![1.0, 2.0])], 8);
        let gemma = |pos_x| {
            VisionInput::Gemma4v(Gemma4vPreprocessedImage {
                patches: vec![1.0, 2.0],
                pos_x,
                pos_y: vec![0],
                n_x: 1,
                n_y: 1,
                source_label: "ignored".into(),
            })
        };
        let gemma_key = vision_input_cache_key(&[gemma(vec![0])], 8);
        assert_ne!(siglip_key, gemma_key);
        assert_ne!(gemma_key, vision_input_cache_key(&[gemma(vec![1])], 8));
    }

    fn cached_output(key: [u8; 32], values: Vec<f32>) -> VisionPipelineOutput {
        VisionPipelineOutput {
            input_fingerprint: key,
            embeddings: Arc::new(vec![values]),
            family: VisionFamily::Gemma,
            per_row_floats: 1,
            qwen3vl_image_grids: Vec::new(),
            forward_ms: 17,
        }
    }

    #[test]
    fn vision_embedding_cache_hit_is_pointer_cheap_and_budgeted() {
        let key = [3; 32];
        let cache = VisionEmbeddingCache::new(16);
        assert!(cache.reserve_or_hit(key, None).expect("reserve").is_none());
        let output = cached_output(key, vec![1.0, 2.0, 3.0, 4.0]);
        cache.complete(key, Some(&output)).expect("complete");
        assert_eq!(cache.resident_bytes(), 16);
        let hit = cache
            .reserve_or_hit(key, None)
            .expect("lookup")
            .expect("cache hit");
        assert!(Arc::ptr_eq(&output.embeddings, &hit.embeddings));
        assert_eq!(hit.forward_ms, 0);
    }

    #[test]
    fn vision_embedding_cache_oversize_payload_is_not_retained() {
        let key = [4; 32];
        let cache = VisionEmbeddingCache::new(4);
        assert!(cache.reserve_or_hit(key, None).expect("reserve").is_none());
        let output = cached_output(key, vec![1.0, 2.0]);
        cache.complete(key, Some(&output)).expect("complete");
        assert_eq!(cache.resident_bytes(), 0);
        assert!(cache
            .reserve_or_hit(key, None)
            .expect("reserve after oversize")
            .is_none());
        cache
            .complete(key, None)
            .expect("release second reservation");
    }

    #[test]
    fn vision_embedding_cache_lru_retains_multiple_images_within_budget() {
        let cache = VisionEmbeddingCache::new(16);
        for key in [[1; 32], [2; 32]] {
            assert!(cache.reserve_or_hit(key, None).unwrap().is_none());
            cache
                .complete(key, Some(&cached_output(key, vec![1.0, 2.0])))
                .unwrap();
        }
        assert_eq!(cache.resident_bytes(), 16);
        assert!(cache.reserve_or_hit([1; 32], None).unwrap().is_some());

        assert!(cache.reserve_or_hit([3; 32], None).unwrap().is_none());
        cache
            .complete([3; 32], Some(&cached_output([3; 32], vec![3.0, 4.0])))
            .unwrap();
        assert!(cache.reserve_or_hit([1; 32], None).unwrap().is_some());
        assert!(cache.reserve_or_hit([2; 32], None).unwrap().is_none());
        cache.complete([2; 32], None).unwrap();
    }

    #[test]
    fn vision_embedding_cache_serializes_same_key_and_wakes_on_failure() {
        use std::sync::mpsc;

        let key = [5; 32];
        let cache = Arc::new(VisionEmbeddingCache::new(64));
        assert!(cache
            .reserve_or_hit(key, None)
            .expect("leader reserve")
            .is_none());
        let (started_tx, started_rx) = mpsc::channel();
        let follower_cache = Arc::clone(&cache);
        let follower = std::thread::spawn(move || {
            started_tx.send(()).expect("started signal");
            follower_cache.reserve_or_hit(key, None)
        });
        started_rx.recv().expect("follower started");
        cache
            .complete(key, None)
            .expect("failed leader releases gate");
        assert!(follower
            .join()
            .expect("follower thread")
            .expect("follower reservation")
            .is_none());
        cache.complete(key, None).expect("follower releases gate");
    }
}
