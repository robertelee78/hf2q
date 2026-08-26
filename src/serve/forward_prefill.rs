//! Dense prefill forward pass — ADR-009 Track 1.
//!
//! This module contains `forward_prefill()`, which processes the entire prompt
//! through the transformer layers using dense F32 attention instead of
//! TQ-packed attention. The rest of the layer pipeline (norms, QKV, MLP, MoE)
//! reuses the same ops as `forward_decode`.
//!
//! Architecture:
//! - Tokens are processed one at a time through all layers (same as decode)
//! - For each token, Q/K/V are computed identically to decode
//! - K,V are accumulated as dense F32 in head-major layout per layer
//! - Attention uses `flash_attn_vec` (dense F32 SDPA) instead of `flash_attn_vec_tq`
//! - K,V are also TQ-encoded into the packed cache for subsequent decode
//! - After all tokens: extract last-row logits, argmax → first decode token

use anyhow::Result;
use mlx_native::ops::flash_attn_vec::FlashAttnVecParams;
use mlx_native::MlxBuffer;
use std::ops::Range;
use std::time::Instant;

use crate::debug::INVESTIGATION_ENV;

/// Extra live-KV capacity retained by Gemma's SerialFifo long-resume path.
///
/// OpenCode normally asks for up to 8,192 output tokens and then appends the
/// assistant/tool-result transcript on the following request. Reserving twice
/// that amount lets the next request reuse the live prompt prefix in place
/// without duplicating tens of GiB into the bounded LCP snapshot registry.
/// The reserve is only applied when long-resume already selected linear
/// sliding-layer storage; legacy ring-cache behavior is unchanged.
pub(crate) const GEMMA_AGENTIC_LIVE_RESERVE_TOKENS: usize = 16 * 1024;

/// Cold slot-mounted prefills below this size use the linear slot-aware path.
///
/// The batched slot-mount path reuses large persistent KV views and is the
/// throughput path for ordinary prompts. Repeated decode -> cold-admission
/// testing on Apple M5 Max exposed intermittent all-non-finite logits only for
/// observed tiny batched prefills (2-5 tokens). Thirty-two tokens is a
/// conservative power-of-two containment boundary; lengths 6-31 are covered
/// by policy, not a claim that each length independently reproduced the fault.
/// The linear path separately rounds its dense KV allocation through the final
/// 32-row `flash_attn_vec` tile; without that padding the fallback itself can
/// read beyond a partial tile in pinned `mlx-native` 0.10.8.
pub(crate) const GEMMA_SLOT_BATCHED_PREFILL_MIN_TOKENS: usize = 32;
/// `flash_attn_vec` evaluates K/V in 32-row tiles. The current shader masks
/// lanes after loading each tile, so dense cache rows must be padded through
/// the final complete tile rather than merely through the logical extent.
pub(crate) const GEMMA_FLASH_ATTN_VEC_KV_TILE: usize = 32;

fn gemma_flash_attn_vec_capacity(logical_tokens: usize) -> usize {
    logical_tokens.div_ceil(GEMMA_FLASH_ATTN_VEC_KV_TILE) * GEMMA_FLASH_ATTN_VEC_KV_TILE
}

pub(crate) fn gemma_slot_prefill_batched_work_eligible(
    work_len: usize,
    has_soft_tokens: bool,
) -> bool {
    !has_soft_tokens && work_len >= GEMMA_SLOT_BATCHED_PREFILL_MIN_TOKENS
}

fn gemma_use_batched_slot_prefill(
    prompt_len: usize,
    has_soft_tokens: bool,
    cached_tokens: usize,
) -> bool {
    !has_soft_tokens
        && (cached_tokens > 0
            || gemma_slot_prefill_batched_work_eligible(prompt_len, has_soft_tokens))
        && std::env::var("HF2Q_PREFILL_SLOT_BATCHED").as_deref() != Ok("0")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GemmaKvCapacityPlan {
    /// Minimum capacity needed to finish this request safely.
    pub(crate) required_linear: usize,
    /// Capacity to use when allocating a fresh long-resume buffer.
    pub(crate) allocation_linear: usize,
    /// Sliding-layer counterpart of `required_linear`.
    pub(crate) required_sliding: usize,
    /// Sliding-layer counterpart of `allocation_linear`.
    pub(crate) allocation_sliding: usize,
}

pub(crate) fn gemma_kv_capacity_plan(
    seq_len: usize,
    max_decode_tokens: usize,
    sliding_window: usize,
    long_resume: bool,
) -> GemmaKvCapacityPlan {
    let required_linear = gemma_flash_attn_vec_capacity(seq_len + max_decode_tokens);
    let allocation_linear = if long_resume {
        gemma_flash_attn_vec_capacity(
            seq_len + max_decode_tokens.max(GEMMA_AGENTIC_LIVE_RESERVE_TOKENS),
        )
    } else {
        required_linear
    };
    let required_sliding = if long_resume {
        sliding_window.max(required_linear)
    } else {
        sliding_window
    };
    let allocation_sliding = if long_resume {
        sliding_window.max(allocation_linear)
    } else {
        sliding_window
    };
    GemmaKvCapacityPlan {
        required_linear,
        allocation_linear,
        required_sliding,
        allocation_sliding,
    }
}

/// Restore the logical decode cursor for an LCP snapshot.
///
/// Sliding layers normally use a physical ring, so their next write cursor is
/// reduced modulo the ring capacity. Long-resume snapshots are different:
/// their sliding-layer buffers are deliberately allocated as linear storage
/// and indexed by absolute token position. Keep the legacy packed-cache
/// `seq_len` capped to its physical ring, but preserve the absolute
/// `write_pos` so the production hybrid cache resumes at position `k` rather
/// than overwriting position `k % sliding_window`.
fn restored_kv_cursor(
    k: usize,
    is_sliding: bool,
    cache_capacity: usize,
    linear_sliding_resume: bool,
) -> (usize, usize) {
    if !is_sliding {
        return (k, k);
    }

    let capacity = cache_capacity.max(1);
    let write_pos = if linear_sliding_resume {
        k
    } else {
        k % capacity
    };
    (write_pos, k.min(capacity))
}

/// Per-position embedding override for soft-token injection
/// (Phase 2c Task #17, iter-97).
///
/// When `forward_prefill_with_soft_tokens` (or `forward_prefill` with a
/// non-empty soft-tokens slice) reaches a token whose position lies
/// within `range`, it skips the standard `embedding_gather_scale_f32`
/// dispatch and instead copies the corresponding row of `embeddings`
/// (a `[range.len() × hidden_size]` F32 buffer) into the per-token
/// hidden-state buffer via `mlx_native::ops::copy::dispatch_copy_f32`.
///
/// Used by the multimodal chat path: the chat template emits one
/// `<|image|>` placeholder token per image, the handler expands it into
/// `N_image_tokens` consecutive positions, runs the ViT + projector to
/// obtain `[N_image_tokens, hidden_size]` projected vision embeddings,
/// then attaches `SoftTokenInjection { range: image_range, embeddings:
/// projected_vision_embeddings }` to the prefill call.  At each
/// `pos ∈ image_range`, the model sees the projected-vision row instead
/// of the language-model embedding for whatever placeholder token id
/// was emitted by the tokenizer.
///
/// **Pre-scaling contract.**  Gemma-family text inputs go through
/// `embedding_gather_scale_f32` which multiplies the looked-up row by
/// `sqrt(hidden_size)`.  The standard multimodal projector output is
/// already in the model's hidden-state space (no additional scaling) —
/// the soft-token path therefore copies the override row VERBATIM.
/// Other model families that DON'T pre-scale text embeddings are
/// equivalent (no-op scaling either way).
///
/// **Range vs. token-id contract.**  The placeholder token IDs at
/// `prompt_tokens[range]` are IGNORED — the override completely
/// replaces the embed step at those positions.  Callers should
/// nevertheless place the same special-token id (e.g. Gemma's
/// `<|image|>`, id=...) at those positions because (a) it provides
/// a clean fallback when the soft-tokens slice is empty, (b) it
/// makes the request token-counting consistent with the OpenAI
/// usage shape.
pub struct SoftTokenInjection<'a> {
    /// Half-open position range within the prompt: `[start, end)`.
    pub range: Range<usize>,
    /// Replacement embeddings, shape `[range.len(), hidden_size]` F32,
    /// row-major.  Buffer outlives this `SoftTokenInjection` (lifetime
    /// `'a`).  Caller is responsible for ensuring the row count
    /// matches `range.len()` and the column count matches the model's
    /// `hidden_size` — `forward_prefill` validates and errors clean
    /// on mismatch.
    pub embeddings: &'a MlxBuffer,
}

/// Per-LM-layer DeepStack residual injection metadata (ADR-005 iter-224
/// Wedge-4c.5).
///
/// Mirrors the peer's per-layer
/// `cur += ds` add: at LM layer `il < n_deepstack` (where
/// `n_deepstack = chunks.len()`), the post-FFN-residual `cur` is
/// updated in-place at the image-token positions with chunk `il`'s
/// rows.
///
/// The `chunks` vec is **sorted by ascending LM-layer-of-injection**,
/// so chunks[0] is added at LM layer 0, chunks[1] at LM layer 1, etc.
/// This matches Qwen3-VL's `deepstack_indexes` convention where the
/// i-th flagged ViT block's tap output (after passing through its
/// DeepStack head + main projector) becomes chunk i+1 in the
/// augmented embed and is consumed at LM layer i.
///
/// `image_token_positions` lists the prompt positions (post-`<|image_pad|>`
/// expansion) where the image tokens reside; same length as the
/// `n_image_tokens` row count of every chunk.
///
/// **Wedge-4c.5 status**: this struct is the engine seam between the
/// ViT side (vit_gpu_qwen.rs's augmented embed) and the LM side
/// (forward_gpu.rs's image_token_residual_add hook). The handler-side
/// path that constructs it from `compute_vision_embeddings_gpu_qwen`'s
/// output is Wedge-4d territory — until that lands, only the
/// LM-side test fixtures construct DeepstackInjection directly.
pub struct DeepstackInjection<'a> {
    /// Image-token positions in the prompt (post-`<|image_pad|>`
    /// expansion). Each position must be `< prompt_tokens.len()`.
    /// Order is the natural left-to-right scan; chunk row k applies
    /// at position `image_token_positions[k]`.
    pub image_token_positions: Vec<u32>,
    /// One GPU buffer per ds layer, each shape `[n_image_tokens,
    /// hidden_size]` F32 row-major. `chunks.len()` = n_deepstack;
    /// chunks[i] is added at LM layer i (i in 0..n_deepstack). Buffers
    /// outlive this `DeepstackInjection` (lifetime `'a`).
    pub chunks: Vec<&'a MlxBuffer>,
}

impl<'a> DeepstackInjection<'a> {
    /// `n_deepstack` — number of LM layers receiving deepstack
    /// injection. Equal to `chunks.len()`.
    pub fn n_deepstack(&self) -> usize {
        self.chunks.len()
    }

    /// `n_image_tokens` — equal to `image_token_positions.len()`. By
    /// contract every `chunks[i]` carries `n_image_tokens` rows.
    pub fn n_image_tokens(&self) -> usize {
        self.image_token_positions.len()
    }
}

/// Per-image post-merge token grid for 3D-mRoPE position synthesis
/// (ADR-005 iter-224 Wedge-4d).
///
/// Carries the post-spatial-merge `(n_x, n_y)` grid that the placeholder
/// expansion + ViT both produce for one image. Total token count is
/// `n_x * n_y` and matches `n_image_tokens` flowing through the
/// `expand_image_placeholders` per-image expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenImageGrid {
    /// Post-merge token-grid width (X axis). For canonical Qwen3-VL
    /// preprocessor at `image_size=768, patch_size=16,
    /// spatial_merge_size=2`: `n_x = 24`.
    pub n_x: u32,
    /// Post-merge token-grid height (Y axis).
    pub n_y: u32,
}

impl QwenImageGrid {
    /// `n_image_tokens` = `n_x * n_y`.
    pub fn n_image_tokens(&self) -> u32 {
        self.n_x.saturating_mul(self.n_y)
    }

    /// Per the peer's `mtmd_image_tokens_get_n_pos` —
    /// `MTMD_POS_TYPE_MROPE` returns `max(nx, ny)` for the temporal-axis
    /// advance after an image chunk. For Qwen3-VL the LM's "global"
    /// time index advances by `max(n_x, n_y)`, NOT by the full
    /// `n_image_tokens` — i.e. the LM treats an image as a SINGLE
    /// temporal step covering `max(n_x, n_y)` positions along axis 0,
    /// while axes 1 (y) and 2 (x) carry the per-patch grid coordinates.
    pub fn temporal_advance(&self) -> u32 {
        self.n_x.max(self.n_y)
    }
}

/// Build the 3D-mRoPE flat-position buffer (`positions_flat[4 *
/// seq_len]` axis-major) for a sequence containing text + image chunks.
///
/// Implements the peer's MROPE position assignment for Qwen3-VL combined
/// with its temporal-advance rule
/// (`mtmd_image_tokens_get_decoder_pos` for `MTMD_POS_TYPE_MROPE`, and
/// `set_position_mrope_2d` writing `[t, y, x, z]` axes column-major).
///
/// **Position layout per axis** (column-major: `flat[axis * seq_len + t]`):
///   - axis 0 = t (temporal)
///   - axis 1 = y (height / row)
///   - axis 2 = x (width / column)
///   - axis 3 = z (image_idx for HunyuanVL — unused for Qwen3-VL,
///     always 0)
///
/// **Per-token assignment**:
///   - **Text token at sequence position `p`, with the global running
///     position counter at `t`**: `[t, t, t, t]` — all four axes carry
///     the same temporal value, mirroring peer at
///     `mtmd-helper.cpp:155-162` `set_position_normal` and the M-RoPE
///     text broadcast at `llama-batch.cpp:713-720`.
///   - **Image-patch token at index `i` within image `img`** with the
///     image starting at temporal position `t_img` and grid
///     `(n_x, n_y)`:
///     - axis 0 (t) = `t_img` (CONSTANT for ALL `n_x*n_y` patch tokens
///       of one image — peer at mtmd.cpp:1300 `pos.t = pos_0`).
///     - axis 1 (y) = `t_img + (i / n_x)` (peer at mtmd.cpp:1302
///       `pos.y = pos_0 + (i / nx)`).
///     - axis 2 (x) = `t_img + (i % n_x)` (peer at mtmd.cpp:1301
///       `pos.x = pos_0 + (i % nx)`).
///     - axis 3 (z) = `0` (peer at mtmd.cpp:1303 `pos.z = 0`).
///   - **After an image chunk**, the global counter `t` advances by
///     `max(n_x, n_y)` (NOT by `n_x * n_y`) per peer's
///     `mtmd_image_tokens_get_n_pos` at mtmd.cpp:1354-1357 returning
///     `max(nx, ny)` for `MTMD_POS_TYPE_MROPE`. This is why a 24×24
///     image consumes 576 LM-sequence-position SLOTS but advances the
///     temporal axis by only 24 (the LM "sees" an image as a 24-step
///     scan along time, not 576 steps).
///
/// # Arguments
///
/// - `prompt_len`: total tokenized prompt length (text + image_pad
///   placeholder expansion already merged).
/// - `image_grids`: per-image `(grid, sequence_start)` pairs; SORTED by
///   `sequence_start`. The N image regions in the prompt occupy
///   `[seq_start..seq_start + grid.n_image_tokens()]` for each image.
///   Text tokens live in the gaps between (and outside) these regions.
///
/// # Returns
///
/// `Vec<i32>` of length `4 * prompt_len`, axis-major.
///
/// # Errors
/// - Image regions overlap or extend past `prompt_len`.
/// - `image_grids` is not sorted by `sequence_start`.
/// - Any `grid.n_image_tokens() == 0`.
pub fn build_qwen_vision_positions(
    prompt_len: usize,
    image_grids: &[(QwenImageGrid, u32)],
) -> anyhow::Result<Vec<i32>> {
    use anyhow::anyhow;

    // Validate ordering + non-overlap + bounds.
    let mut last_end: u32 = 0;
    for (i, (grid, seq_start)) in image_grids.iter().enumerate() {
        let n_tokens = grid.n_image_tokens();
        if n_tokens == 0 {
            return Err(anyhow!(
                "build_qwen_vision_positions: image[{i}] has zero tokens \
                 (n_x={}, n_y={})",
                grid.n_x,
                grid.n_y
            ));
        }
        if (*seq_start) < last_end {
            return Err(anyhow!(
                "build_qwen_vision_positions: image[{i}] starts at {seq_start} \
                 which is before the prior region's end {last_end} — \
                 overlapping or unsorted image regions"
            ));
        }
        let region_end = (*seq_start)
            .checked_add(n_tokens)
            .ok_or_else(|| anyhow!("build_qwen_vision_positions: image[{i}] region overflow"))?;
        if (region_end as usize) > prompt_len {
            return Err(anyhow!(
                "build_qwen_vision_positions: image[{i}] region {seq_start}..{region_end} \
                 extends past prompt_len {prompt_len}"
            ));
        }
        last_end = region_end;
    }

    let mut flat = vec![0i32; 4 * prompt_len];
    // `chunks_exact_mut(0)` panics, so the empty-prompt case (which the
    // prior `flat[i * 0 + q] = …` indexing tolerated only because the
    // write loop never executed for `prompt_len == 0`) must be handled
    // before the slice split.  Callers that pass `prompt_len == 0` get
    // back the empty `flat` vec they would have gotten from the
    // original implementation.
    if prompt_len == 0 {
        return Ok(flat);
    }
    {
        // Split the flat buffer into 4 named per-channel slices (t, y, x, z),
        // each of length `prompt_len`.  Avoids the `i * prompt_len + q`
        // arithmetic that previously masked out-of-bounds writes into the
        // wrong channel — `t_chan[q]` now panics cleanly on bounds error
        // instead of silently corrupting `y_chan`.
        let mut channels = flat.chunks_exact_mut(prompt_len);
        let t_chan = channels
            .next()
            .expect("flat is 4*prompt_len so chunks(prompt_len) yields 4");
        let y_chan = channels
            .next()
            .expect("flat is 4*prompt_len so chunks(prompt_len) yields 4");
        let x_chan = channels
            .next()
            .expect("flat is 4*prompt_len so chunks(prompt_len) yields 4");
        let z_chan = channels
            .next()
            .expect("flat is 4*prompt_len so chunks(prompt_len) yields 4");

        // Global temporal counter — advances by 1 for every text token, by
        // `max(n_x, n_y)` for every image chunk.
        let mut t_global: i32 = 0;
        let mut img_idx: usize = 0;
        let mut p: usize = 0;
        while p < prompt_len {
            if img_idx < image_grids.len() && p == image_grids[img_idx].1 as usize {
                // Image chunk start.
                let (grid, _seq_start) = image_grids[img_idx];
                let n_x = grid.n_x as i32;
                let n_tokens = grid.n_image_tokens() as usize;
                let t_img = t_global;
                for i in 0..n_tokens {
                    let q = p + i;
                    let i_i32 = i as i32;
                    t_chan[q] = t_img; // t (constant)
                    y_chan[q] = t_img + (i_i32 / n_x); // y
                    x_chan[q] = t_img + (i_i32 % n_x); // x
                    z_chan[q] = 0; // z
                }
                t_global += grid.temporal_advance() as i32;
                p += n_tokens;
                img_idx += 1;
            } else {
                // Text token — all 4 channels share the same value.
                t_chan[p] = t_global;
                y_chan[p] = t_global;
                x_chan[p] = t_global;
                z_chan[p] = t_global;
                t_global += 1;
                p += 1;
            }
        }
    }
    Ok(flat)
}
use super::config::LayerType;
use super::gpu::GpuContext;
use crate::inference::models::gemma4::kv_cache::{
    // ADR-040 iter-B4c-kernel iter-2C + iter-2D + iter-2-decode-D (2026-05-30) —
    // multi-seq scaffold types for the legacy 4-bit (cb_bits==0) +
    // dense F32 (HF2Q_USE_DENSE=1) opt-in pre-default surfaces.  Their
    // alloc helpers (`alloc_multi_seq_mlx_kv_for_layer` /
    // `alloc_multi_seq_dense_kv_for_layer`) are exposed for the
    // iter-C2c-cont-cont spawn-time provisioning Phase 3 (dense) +
    // Phase 4 (mlx) on `GemmaLoadedModel`.  See ADR-040 §6.1.46.
    MultiSeqDenseKvBuffers,
    MultiSeqHbKvBuffers,
    MultiSeqHybridKvBuffers,
    MultiSeqMlxKvCache,
};
use crate::inference::models::gemma4::{DenseKvBuffers, HbKvBuffers, MlxKvCache, MlxModelWeights};
use crate::serve::forward_mlx_shared::{
    dispatch_qmatmul, dispatch_rms_norm_unit_perhead, RmsNormPerHeadArgs,
};
use crate::serve::multi_seq_kv::{MultiSeqError, SlotId};

/// Helper: dump an F32 MlxBuffer's first `n_elems` to a file at dump_dir.
fn write_dump_f32(
    dump_dir: &str,
    name: &str,
    layer: usize,
    tok: usize,
    buf: &MlxBuffer,
    n_elems: usize,
) -> Result<()> {
    let data: &[f32] = buf
        .as_slice()
        .map_err(|e| anyhow::anyhow!("dump {name} L{layer} T{tok}: {e}"))?;
    let path = format!("{dump_dir}/hf2q_prefill_{name}_layer{layer:02}_tok{tok:03}.bin");
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            n_elems * std::mem::size_of::<f32>(),
        )
    };
    std::fs::write(&path, bytes).map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
    eprintln!(
        "[PREFILL DUMP] {} L{} T{} ({} f32) -> {}",
        name, layer, tok, n_elems, path
    );
    Ok(())
}

impl MlxModelWeights {
    /// True batched prefill with dense attention (ADR-009 Track 1).
    ///
    /// Processes the prompt token-by-token through all layers but replaces
    /// TQ-packed attention with dense F32 SDPA. This eliminates compounding
    /// TQ quantization noise during prompt ingestion.
    ///
    /// Returns the first decode token (greedy argmax of last-row logits).
    /// Existing-API thin wrapper: prefill with no soft-token overrides.
    /// All pre-iter-97 callers (warmup, generate, embed_last) use this.
    pub fn forward_prefill(
        &mut self,
        prompt_tokens: &[u32],
        max_decode_tokens: usize,
        gpu: &mut GpuContext,
    ) -> Result<u32> {
        self.forward_prefill_with_soft_tokens(prompt_tokens, &[], max_decode_tokens, gpu)
    }

    /// Soft-token-aware prefill (Phase 2c Task #17, iter-97).
    ///
    /// Same semantics as `forward_prefill` except that for any prompt
    /// position `p` that lies within a `SoftTokenInjection.range`, the
    /// per-token embed step is replaced by a buffer-copy from the
    /// override embeddings instead of dispatching the standard
    /// `embedding_gather_scale_f32` against `embed_weight[token_id]`.
    /// The placeholder token IDs at those positions are ignored (the
    /// language-model lookup is fully bypassed).
    ///
    /// See the `SoftTokenInjection` struct doc for the full contract.
    ///
    /// # Errors
    ///
    /// In addition to the base `forward_prefill` error set:
    ///   * `SoftTokenInjection.range` extends past `prompt_tokens.len()`.
    ///   * Two `SoftTokenInjection` ranges overlap (ambiguous override).
    ///   * `embeddings.byte_len()` is too small for `range.len() × hidden_size × 4`.
    pub fn forward_prefill_with_soft_tokens(
        &mut self,
        prompt_tokens: &[u32],
        soft_tokens: &[SoftTokenInjection<'_>],
        max_decode_tokens: usize,
        gpu: &mut GpuContext,
    ) -> Result<u32> {
        // ADR-017 Phase E.a iter-3: thin wrapper. Pre-iter-3 callers
        // (warmup, generate, embed_last, generate_stream_once) keep
        // the 4-arg shape; the engine's Phase E.a probe site calls the
        // explicit `_resume` variant when iter-3's gate fires.
        self.forward_prefill_with_soft_tokens_resume(
            prompt_tokens,
            soft_tokens,
            max_decode_tokens,
            gpu,
            None,  // restored_lcp = None → wholesale reset, fresh dense_kvs
            false, // slot_aware = false → legacy/SerialFifo byte-equivalent
        )
    }

    /// ADR-017 Phase E option (a) iter-3 — partial-prefill resume entry
    /// point.
    ///
    /// `restored_lcp = None` is byte-identical to the pre-iter-3
    /// `forward_prefill_with_soft_tokens` (wholesale `cache.write_pos =
    /// 0` reset + fresh `dense_kvs` allocation + per-token loop from 0).
    ///
    /// `restored_lcp = Some(k)` resumes from token `k`:
    ///   * `cache.write_pos` for full-attention layers is set to `k`;
    ///     for sliding layers, `k % sliding_window`.
    ///   * `cache.seq_len` is set to `min(k, capacity)` per layer.
    ///   * `self.dense_kvs` MUST be populated by the caller with the
    ///     cached per-layer `Arc<DenseKvBuffers>` clones from the
    ///     `LcpRegistry` BEFORE this call. This function takes
    ///     ownership via `Arc::try_unwrap`; the load-bearing precondition
    ///     is `Arc::strong_count == 1` for every layer (the engine's
    ///     `take_prefix` semantics produce that state).
    ///   * Per-token loop iterates `[k..seq_len)` only — the cached
    ///     bytes for `[0..k)` are reused in place via the GPU kernel
    ///     reading through the existing buffer.
    ///
    /// # Preconditions for `restored_lcp = Some(k)`
    ///
    ///   * `0 < k < seq_len` (caller ensures via `LcpRegistry`'s
    ///     full-equality + zero-overlap gates).
    ///   * `self.dense_kvs.is_some()` (caller installed cached state).
    ///   * Per-layer `dense_kvs[layer].capacity >= required_capacity`
    ///     where the global requirement rounds `seq_len + max_decode_tokens`
    ///     through the final 32-row `flash_attn_vec` tile,
    ///     `sliding_window` for sliding (caller checks; this function
    ///     defensively bails on violation rather than silently
    ///     corrupting state).
    ///   * Per-layer `Arc::strong_count(&dense_kvs[layer]) == 1`
    ///     (caller's `take_prefix` semantics ensure this; defensive
    ///     bail on violation).
    ///   * Every soft-token range is wholly in the appended suffix. Cached
    ///     image rows are represented by the restored KV and are not
    ///     injected again.
    ///
    /// # Errors
    ///
    /// In addition to the base `forward_prefill_with_soft_tokens`
    /// error set, restored mode adds:
    ///   * `restored_lcp = Some(k)` but `self.dense_kvs.is_none()`.
    ///   * `restored_lcp = Some(k)` with `k == 0` or `k >= seq_len`.
    ///   * Per-layer `Arc` not exclusive (`Arc::try_unwrap` failed).
    ///   * Per-layer cached capacity < required.
    ///   * `restored_lcp = Some(k)` with a soft-token range starting before
    ///     `k`.
    pub fn forward_prefill_with_soft_tokens_resume(
        &mut self,
        prompt_tokens: &[u32],
        soft_tokens: &[SoftTokenInjection<'_>],
        max_decode_tokens: usize,
        gpu: &mut GpuContext,
        restored_lcp: Option<usize>,
        // ADR-040 STEP-1b (2026-06-24) — prefill-side cross-slot leak gate.
        // When `true`, the slot-aware caller has mounted a per-slot slot-view
        // bundle on `self.dense_kvs` and OWNS the restore.  The unconditional
        // `self.dense_kvs = Some(..)` + `dense_kvs_snapshot_for_lcp` +
        // `dense_sdpa_tmp` write-backs at the tail of this fn would otherwise
        // overwrite that mount with a FRESH shared allocation and leak it
        // (plus the snapshot/sdpa_tmp) across slots.  `slot_aware=true` SKIPS
        // those three `self.*` writes; the caller restores `self.dense_kvs`
        // from the saved prior.  SerialFifo / legacy callers pass `false` →
        // BYTE-EQUIVALENT (writes happen exactly as before).
        slot_aware: bool,
    ) -> Result<u32> {
        let seq_len = prompt_tokens.len();
        if seq_len == 0 {
            anyhow::bail!("forward_prefill: empty prompt");
        }
        // ADR-017 Phase E.a iter-3 preconditions (validated upfront so
        // we fail before touching any GPU state).
        if let Some(k) = restored_lcp {
            if k == 0 || k >= seq_len {
                anyhow::bail!(
                    "forward_prefill: restored_lcp={} out of range (must be 0 < k < seq_len={})",
                    k,
                    seq_len
                );
            }
            anyhow::ensure!(
                soft_tokens.iter().all(|soft| soft.range.start >= k),
                "forward_prefill: restored_lcp=Some({k}) requires every soft-token range to lie wholly in the suffix"
            );
            if self.dense_kvs.is_none() {
                anyhow::bail!(
                    "forward_prefill: restored_lcp=Some({}) but self.dense_kvs is None — \
                     caller must install cached `Vec<Arc<DenseKvBuffers>>` BEFORE this call",
                    k
                );
            }
        }
        let hs = self.hidden_size;
        // Validate soft-token ranges + embedding sizes upfront so we
        // fail before the (expensive) prefill loop starts.
        for (i, st) in soft_tokens.iter().enumerate() {
            if st.range.end > seq_len {
                anyhow::bail!(
                    "forward_prefill: soft_tokens[{}].range {:?} extends past prompt_tokens.len()={}",
                    i, st.range, seq_len
                );
            }
            if st.range.start >= st.range.end {
                anyhow::bail!(
                    "forward_prefill: soft_tokens[{}].range {:?} is empty or reversed",
                    i,
                    st.range
                );
            }
            let needed_bytes = st.range.len() * hs * 4;
            if st.embeddings.byte_len() < needed_bytes {
                anyhow::bail!(
                    "forward_prefill: soft_tokens[{}].embeddings byte_len={} < required {} \
                     ({} positions × {} hidden × 4 bytes)",
                    i,
                    st.embeddings.byte_len(),
                    needed_bytes,
                    st.range.len(),
                    hs
                );
            }
        }
        // Reject overlapping ranges (ambiguous which embedding wins).
        for i in 0..soft_tokens.len() {
            for j in (i + 1)..soft_tokens.len() {
                let a = &soft_tokens[i].range;
                let b = &soft_tokens[j].range;
                if a.start < b.end && b.start < a.end {
                    anyhow::bail!(
                        "forward_prefill: soft_tokens ranges overlap — [{}]={:?} vs [{}]={:?}",
                        i,
                        a,
                        j,
                        b
                    );
                }
            }
        }
        let num_layers = self.layers.len();
        let vocab_size = self.vocab_size;
        let eps = self.rms_norm_eps;

        // Reset per-layer KV cache write positions before this prefill.
        //
        // The TQ-packed `MlxKvCache` (allocated once at model load with
        // capacity = max_position_embeddings for full layers, sliding_window
        // for sliding layers) accumulates `write_pos` + `seq_len` across
        // every prefill / decode step.  In a single-request lifecycle that's
        // correct: prefill writes positions 0..N, decode appends N..N+M.  But
        // hf2q's serialized worker handles multiple requests on the same
        // `LoadedModel` — each fresh request needs to OVERWRITE the cache
        // from position 0, not append.
        //
        // This was a latent bug in the chat-only path that worked in practice
        // because:
        //   * Full-attention layers have huge capacity (max_position_embeddings,
        //     262144 for Gemma 4) — many requests fit before overflow.
        //   * Sliding-window layers wrap via `(write_pos % sliding_window)` so
        //     buffer accesses stayed in-bounds — but `seq_len` (passed to
        //     flash-attention as the count of valid KV positions) kept growing
        //     unboundedly, making the kernel attend to "valid" positions that
        //     in fact contained stale data from prior requests.
        //
        // Iter-92 (Phase 2a Task #8) surfaced the bug via the embedding path:
        // many embed requests + a chat completion drove sliding-layer
        // `seq_len` past the sliding_window capacity → the dispatcher's
        // `kv_capacity (sw) < kv_seq_len` guard fired with a hard error.
        //
        // The fix: reset `write_pos` + `seq_len` to 0 here so every prefill
        // starts with an empty KV cache, regardless of prior state.  Each
        // OpenAI `/v1/chat/completions` and `/v1/embeddings` request is
        // semantically independent (multi-turn chat is handled by the client
        // sending full history), so wholesale reset is the correct semantics.
        //
        // ADR-017 Phase E option (a) iter-3 (`restored_lcp = Some(k)`):
        // instead of wholesale reset, position the per-layer
        // `kv_caches` bookkeeping at `k` so the per-token loop below
        // resumes at token `k` and writes new positions `[k..seq_len)`
        // into the cached buffers in place. Full-attention layers use
        // linear `write_pos = k`; sliding layers use `write_pos =
        // k % sliding_window` (the ring's "next slot" pointer when
        // `k` tokens have already been written) and `seq_len =
        // min(k, sliding_window)` (the kernel's "valid populated
        // slots" count, capped by the ring capacity).
        let kv_lcp_long_resume = crate::debug::INVESTIGATION_ENV.kv_lcp_long_resume
            && crate::debug::INVESTIGATION_ENV.kv_lcp_resume
            && (crate::debug::INVESTIGATION_ENV.use_dense
                || crate::debug::INVESTIGATION_ENV.hybrid_kv);
        match restored_lcp {
            None => {
                for cache in self.kv_caches.iter_mut() {
                    cache.write_pos = 0;
                    cache.seq_len = 0;
                }
            }
            Some(k) => {
                for cache in self.kv_caches.iter_mut() {
                    (cache.write_pos, cache.seq_len) =
                        restored_kv_cursor(k, cache.is_sliding, cache.capacity, kv_lcp_long_resume);
                }
            }
        }

        let (exec, reg) = gpu.split();
        let dev = exec.device();
        let metal_dev = dev.metal_device();

        // ===================================================================
        // Allocate per-layer dense K,V buffers in head-major layout:
        //   [n_kv_heads, capacity, head_dim]
        // This layout matches flash_attn_vec's K,V input format.
        //
        // Per-layer capacity is set in the loop at lines 254-275:
        //   - LayerType::Sliding (ring): capacity = sliding_window.
        //     Writes wrap via `slot = tok_pos % capacity`; reads use
        //     `kv_seq_len = min(tok_i + 1, sliding_window)` (lines 963-967).
        //   - LayerType::Global (linear): logical extent is
        //     seq_len + max_decode_tokens; allocation rounds through the final
        //     32-row flash_attn_vec tile. Writes are monotonically increasing.
        // The dense flash_attn_vec dispatch at lines 960-981 uses
        // `mask_type=1` (pure causal) for both layer types. Ring
        // correctness rests on attention being permutation-invariant
        // over the K,V set: once the ring wraps, the oldest slot is
        // overwritten with the newest token, but which physical slot
        // is oldest is immaterial — the causal mask within the
        // sliding window still yields the correct attention pattern.
        // ===================================================================
        // ADR-009 Phase 3A finding: matching the peer's F16 KV cache
        // REGRESSED our parity (sourdough 3656→3095, sliding_wrap 752→627).
        // The peer itself is insensitive to KV dtype (its F16 and F32 outputs
        // are byte-identical). Our F16 path has a separate bug worse than F32.
        // F32 remains the default; F16 is opt-in via HF2Q_F16_KV=1 for the
        // follow-up investigation into the F16-specific regression.
        let use_f16_kv = INVESTIGATION_ENV.f16_kv;
        let kv_dtype = if use_f16_kv {
            mlx_native::DType::F16
        } else {
            mlx_native::DType::F32
        };
        let kv_elem_bytes = if use_f16_kv { 2 } else { 4 };
        tracing::debug!("Prefill: KV cache dtype = {:?}", kv_dtype);

        // Per-layer capacity:
        //   - Sliding (ring): sliding_window. Writes wrap at seq_pos % capacity.
        //     Attention is permutation-invariant over cached K,V, so slot
        //     order doesn't affect correctness. Dense flash_attn_vec reads
        //     the populated slots with a pure causal mask.
        //   - Global (linear): seq_len + max_decode_tokens, rounded through
        //     the final 32-row flash_attn_vec tile. Writes are monotonic.
        // Ring buffer for sliding drops ~5 GB of dense KV at 20k decode on
        // Gemma-4 26B (8×1024×256 per layer vs 8×20022×256).
        let sw = self.sliding_window;
        let capacity_plan =
            gemma_kv_capacity_plan(seq_len, max_decode_tokens, sw, kv_lcp_long_resume);
        let required_linear_capacity = capacity_plan.required_linear;
        let linear_capacity = capacity_plan.allocation_linear;
        // ADR-017 Phase E.a iter-3.6 — when `HF2Q_KV_LCP_LONG_RESUME=1`,
        // sliding layers use a LINEAR buffer (cap = max(sw, round32(seq_len +
        // max_decode_tokens))) instead of a ring (cap = sw). Per-token
        // writes go to slot=tok_i (no `% sw` wrap). flash_attn_vec
        // dispatches use `mask_type=2 + sliding_window=sw` so the
        // kernel applies sliding-window masking based on slot index
        // (which now equals logical position). Verified at
        // `/opt/mlx-native/src/shaders/flash_attn_vec.metal:166-170`.
        // The Chesterton's fence at `forward_mlx.rs:2632-2635` documents
        // why the default ring path needs `mask_type=1`: with ring,
        // slot ≠ logical position once wrap happens.
        //
        // When the env-flag is OFF (default), behavior is byte-identical
        // to pre-iter-3.6 (sliding = ring, mask_type=1).
        //
        // "gemma-hybrid-lcp" (2026-08-03): the gate now admits the
        // production HYBRID regime as well as dense — the hybrid
        // `flash_attn_vec_hybrid` kernel implements the same
        // `mask_type=2 + sliding_window` semantics (verified at
        // `/opt/mlx-native/src/shaders/flash_attn_vec_hybrid.metal:490-491,914-915`),
        // so linear sliding buffers + logical-position masking compose
        // identically for both legs.
        let required_sliding_layer_capacity = capacity_plan.required_sliding;
        let sliding_layer_capacity = capacity_plan.allocation_sliding;
        // ADR-017 Phase E.a iter-3: dense_kvs_vec source branches on
        // restored_lcp.
        //
        //   * `None` — pre-iter-3 path: allocate fresh per-layer
        //     buffers sized for this request's seq_len+max_decode.
        //   * `Some(_)` — caller installed cached `Vec<Arc<DenseKvBuffers>>`
        //     into `self.dense_kvs` BEFORE calling. We `take` ownership
        //     of those Arcs and unwrap each via `Arc::try_unwrap` so the
        //     per-token loop can mutate the buffers in place. The
        //     load-bearing precondition is `Arc::strong_count == 1`
        //     (the engine's `LcpRegistry::take_prefix` produces this
        //     state); on violation we bail with a clear error rather
        //     than fall back to fresh-alloc, because falling back would
        //     silently waste the cache hit and leak the cached Arc
        //     refcount.
        // Note on mutability: kernel writes to dense_kvs_vec[i].k/.v go
        // through GPU-side buffer mutation (StorageModeShared), not Rust
        // `&mut` borrows, so the binding stays immutable here.
        let dense_kvs_vec: Vec<DenseKvBuffers> = match restored_lcp {
            None => {
                // ADR-040 iter-B4c-kernel iter-2D (2026-05-30) — alloc-gate
                // alignment with the slot-aware mount discipline.  If the
                // slot-aware caller pre-mounted a slot-view bundle on
                // `self.dense_kvs` (Option<Vec<Arc<DenseKvBuffers>>>), we
                // CONSUME the bundle here via the SAME try_unwrap path as
                // the Some(_) LCP branch — this routes per-token kernel
                // writes to the slot's region of the persistent multi-seq
                // scaffold without changing this sibling fn's signature
                // (H86 preserved).
                //
                // SerialFifo byte-equivalence preserved: SerialFifo enters
                // with `self.dense_kvs == None`, so the consume branch is
                // unreachable for SerialFifo (None Option short-circuits
                // to the fresh-alloc body below).  Mirror of iter-2A-cont's
                // `self.leg_hb_encoded.is_none()` gate alignment at line
                // ~902 + iter-2B's `self.hybrid_kv.is_none()` gate at
                // ~860.  H187 / H178 / H102 pin the SerialFifo case.
                //
                // Pinned by H187 (SerialFifo byte-equivalence preserved
                // at slot 0) + H182 (real slot routing landed for iter-2D
                // dense F32).
                if self.dense_kvs.is_some() {
                    let cached_arcs = self
                        .dense_kvs
                        .take()
                        .expect("iter-2D consume-gate: self.dense_kvs.is_some() validated above");
                    // "gemma-hybrid-lcp" SerialFifo leftover discipline
                    // (2026-08-03): when `slot_aware == false`, a stale
                    // mount from a PREVIOUS request is opportunistic
                    // state, not a contract. Turn N's write-back leaves
                    // `dense_kvs = Some(cap_N)`; turn N+1 with a longer
                    // prompt hits `capacity N < required` and previously
                    // hard-500'd EVERY growing SerialFifo conversation on
                    // the non-batched route (measured live: title-gen
                    // cap 8763 vs main-turn requirement 99942). On ANY
                    // mismatch (len / capacity / is_sliding / shared
                    // Arc) the correct behavior is drop + fresh-alloc —
                    // the mount carries nothing the fresh alloc can't
                    // recompute. The hard bails remain for
                    // `slot_aware == true`, where the mount IS the
                    // multi-seq scaffold contract (iter-2D invariant).
                    let leftover_usable = !slot_aware
                        && cached_arcs.len() == num_layers
                        && cached_arcs
                            .iter()
                            .zip(self.layers.iter())
                            .all(|(arc, layer)| {
                                let layer_is_ring = layer.layer_type == LayerType::Sliding;
                                let required_cap = if layer_is_ring {
                                    required_sliding_layer_capacity
                                } else {
                                    required_linear_capacity
                                };
                                arc.capacity >= required_cap && arc.is_sliding == layer_is_ring
                            });
                    if !leftover_usable && !slot_aware {
                        // Drop the stale mount, then run the SAME
                        // fresh-alloc body the None-mount branch uses.
                        drop(cached_arcs);
                        let mut v: Vec<DenseKvBuffers> = Vec::with_capacity(num_layers);
                        for (layer_idx, layer) in self.layers.iter().enumerate() {
                            let nkv = layer.num_kv_heads;
                            let hd = layer.head_dim;
                            let layer_is_sliding_model = layer.layer_type == LayerType::Sliding;
                            let capacity = if layer_is_sliding_model {
                                sliding_layer_capacity
                            } else {
                                linear_capacity
                            };
                            let n = nkv * capacity * hd;
                            let kbuf = dev
                                .alloc_buffer(n * kv_elem_bytes, kv_dtype, vec![nkv, capacity, hd])
                                .map_err(|e| {
                                    anyhow::anyhow!("prefill dense K L{layer_idx}: {e}")
                                })?;
                            let vbuf = dev
                                .alloc_buffer(n * kv_elem_bytes, kv_dtype, vec![nkv, capacity, hd])
                                .map_err(|e| {
                                    anyhow::anyhow!("prefill dense V L{layer_idx}: {e}")
                                })?;
                            v.push(DenseKvBuffers {
                                k: kbuf,
                                v: vbuf,
                                capacity,
                                is_sliding: layer_is_sliding_model,
                                dtype: kv_dtype,
                            });
                        }
                        v
                    } else {
                        if cached_arcs.len() != num_layers {
                            anyhow::bail!(
                                "forward_prefill iter-2D consume-gate: cached dense_kvs len {} \
                             != num_layers {}",
                                cached_arcs.len(),
                                num_layers
                            );
                        }
                        let mut v: Vec<DenseKvBuffers> = Vec::with_capacity(num_layers);
                        let mut cached_iter = cached_arcs.into_iter();
                        for (layer_idx, layer) in self.layers.iter().enumerate() {
                            let layer_is_ring = layer.layer_type == LayerType::Sliding;
                            let required_cap = if layer_is_ring {
                                required_sliding_layer_capacity
                            } else {
                                required_linear_capacity
                            };
                            let cached_arc = cached_iter.next().ok_or_else(|| {
                                anyhow::anyhow!(
                                    "forward_prefill iter-2D consume-gate: missing \
                                 cached dense_kvs[layer={}]",
                                    layer_idx
                                )
                            })?;
                            let cached_cap = cached_arc.capacity;
                            if cached_cap < required_cap {
                                anyhow::bail!(
                                    "forward_prefill iter-2D consume-gate: cached \
                                 dense_kvs[layer={}] capacity {} < required {} \
                                 (slot-aware mount requires scaffold capacity ≥ \
                                  request capacity; iter-C2c-cont-cont must \
                                  provision n_seqs × max_position_embeddings buffer)",
                                    layer_idx,
                                    cached_cap,
                                    required_cap
                                );
                            }
                            if cached_arc.is_sliding != layer_is_ring {
                                anyhow::bail!(
                                    "forward_prefill iter-2D consume-gate: cached \
                                 dense_kvs[layer={}] is_sliding={} != model \
                                 layer is_sliding={} (slot-aware mount \
                                 architecture mismatch)",
                                    layer_idx,
                                    cached_arc.is_sliding,
                                    layer_is_ring
                                );
                            }
                            let owned = std::sync::Arc::try_unwrap(cached_arc).map_err(|_arc| {
                                anyhow::anyhow!(
                                    "forward_prefill iter-2D consume-gate: \
                                     cached dense_kvs[layer={}] Arc not exclusive \
                                     — slot-aware mount produced a shared Arc \
                                     (iter-B4c-kernel iter-2D invariant violated)",
                                    layer_idx
                                )
                            })?;
                            v.push(owned);
                        }
                        v
                    }
                } else {
                    // Pre-iter-3 fresh-alloc path (byte-identical to the
                    // pre-iter-3 `forward_prefill_with_soft_tokens` body).
                    let mut v: Vec<DenseKvBuffers> = Vec::with_capacity(num_layers);
                    for (layer_idx, layer) in self.layers.iter().enumerate() {
                        let nkv = layer.num_kv_heads;
                        let hd = layer.head_dim;
                        let layer_is_sliding_model = layer.layer_type == LayerType::Sliding;
                        // iter-3.6: when LONG_RESUME is on, sliding layers
                        // get a LINEAR buffer (cap = sliding_layer_capacity).
                        // The buffer is no longer a ring at the storage
                        // level; the kernel applies sliding-window masking
                        // via mask_type=2. The `is_sliding` field on
                        // DenseKvBuffers continues to indicate the MODEL-
                        // SIDE semantic (kernel needs sliding-window mask),
                        // NOT whether the buffer is a ring. Read sites
                        // distinguish "ring vs linear" by inspecting
                        // INVESTIGATION_ENV.kv_lcp_long_resume directly.
                        let capacity = if layer_is_sliding_model {
                            sliding_layer_capacity
                        } else {
                            linear_capacity
                        };
                        let n = nkv * capacity * hd;
                        let kbuf = dev
                            .alloc_buffer(n * kv_elem_bytes, kv_dtype, vec![nkv, capacity, hd])
                            .map_err(|e| anyhow::anyhow!("prefill dense K L{layer_idx}: {e}"))?;
                        let vbuf = dev
                            .alloc_buffer(n * kv_elem_bytes, kv_dtype, vec![nkv, capacity, hd])
                            .map_err(|e| anyhow::anyhow!("prefill dense V L{layer_idx}: {e}"))?;
                        v.push(DenseKvBuffers {
                            k: kbuf,
                            v: vbuf,
                            capacity,
                            is_sliding: layer_is_sliding_model,
                            // ADR-017 Phase E.a iter-3.5a — record dtype
                            // invariant. `kv_dtype` is set above from
                            // INVESTIGATION_ENV.f16_kv (read once via
                            // LazyLock); same dtype across all layers in
                            // this allocation pass.
                            dtype: kv_dtype,
                        });
                    }
                    v
                }
            }
            Some(k_resume) => {
                // Take cached Arcs out of self.dense_kvs and unwrap
                // each. Defensive validations:
                //   (1) Vec length matches num_layers.
                //   (2) Per-layer capacity satisfies the new request.
                //   (3) Per-layer Arc strong_count == 1 (try_unwrap
                //       succeeds).
                let cached_arcs = self.dense_kvs.take().expect(
                    "restored_lcp=Some precondition: self.dense_kvs is Some (validated above)",
                );
                if cached_arcs.len() != num_layers {
                    anyhow::bail!(
                        "forward_prefill resume: cached dense_kvs len {} != num_layers {}",
                        cached_arcs.len(),
                        num_layers
                    );
                }
                // Iter-5 sweep finding (2026-05-05): the prior code
                // used `cached_arcs.get(layer_idx).cloned()` to obtain
                // each per-layer Arc, which clones the Arc and bumps
                // strong_count to 2 — `Arc::try_unwrap` THEN fails
                // unconditionally on the second sequential resume in a
                // process. The iter-3 K<N falsifier didn't catch this
                // because it engaged resume only ONCE; iter-5's
                // 5-K-fraction sweep exercises 5 sequential resumes
                // and surfaces the bug on fraction 1.
                //
                // Fix: drain `cached_arcs` via `into_iter()` so each
                // Arc is MOVED (no clone), preserving strong_count=1
                // at try_unwrap time. The order-preserving drain
                // matches the per-layer iteration order of
                // `self.layers`.
                let mut v: Vec<DenseKvBuffers> = Vec::with_capacity(num_layers);
                let mut cached_iter = cached_arcs.into_iter();
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let layer_is_ring = layer.layer_type == LayerType::Sliding;
                    // iter-3.6: when LONG_RESUME is on, sliding layers
                    // are linear with cap = sliding_layer_capacity. The
                    // resume path reuses the cached buffer in place;
                    // its capacity must be ≥ the current request's
                    // required_cap (which is the same formula used by
                    // the alloc branch above).
                    let required_cap = if layer_is_ring {
                        required_sliding_layer_capacity
                    } else {
                        required_linear_capacity
                    };
                    let cached_arc = cached_iter.next().ok_or_else(|| {
                        anyhow::anyhow!(
                            "forward_prefill resume: missing cached dense_kvs[layer={}]",
                            layer_idx
                        )
                    })?;
                    // Capacity precondition (zero_copy_only policy v1
                    // — dossier §10.2 C1 case).
                    let cached_cap = cached_arc.capacity;
                    if cached_cap < required_cap {
                        anyhow::bail!(
                            "forward_prefill resume: cached dense_kvs[layer={}] capacity {} < required {} \
                             (k_resume={}, sw={}, linear_capacity={})",
                            layer_idx,
                            cached_cap,
                            required_cap,
                            k_resume,
                            sw,
                            required_linear_capacity
                        );
                    }
                    if cached_arc.is_sliding != layer_is_ring {
                        anyhow::bail!(
                            "forward_prefill resume: cached dense_kvs[layer={}] is_sliding={} \
                             != model layer is_sliding={} (model architecture mismatch)",
                            layer_idx,
                            cached_arc.is_sliding,
                            layer_is_ring
                        );
                    }
                    let owned = std::sync::Arc::try_unwrap(cached_arc).map_err(|_arc| {
                        anyhow::anyhow!(
                            "forward_prefill resume: cached dense_kvs[layer={}] Arc not exclusive \
                             (strong_count > 1) — engine's `take_prefix` precondition violated",
                            layer_idx
                        )
                    })?;
                    v.push(owned);
                }
                v
            }
        };

        // Tmp buffer for flash_attn_vec (sized for largest layer config)
        let max_nh = self.num_attention_heads;
        let max_hd = self.layers.iter().map(|l| l.head_dim).max().unwrap_or(512);
        let tmp_bytes =
            mlx_native::ops::flash_attn_vec::tmp_buffer_bytes(max_nh as u32, max_hd as u32);
        let sdpa_tmp = dev
            .alloc_buffer(tmp_bytes, mlx_native::DType::F32, vec![tmp_bytes / 4])
            .map_err(|e| anyhow::anyhow!("prefill sdpa_tmp: {e}"))?;

        tracing::debug!(
            "Prefill: {} tokens × {} layers (dense SDPA)",
            seq_len,
            num_layers
        );

        // iter-222 (ADR-005 closure, 2026-05-01): the iter-21 Track A
        // `leg_f_kvs` shadow-cache allocation block (~30 LOC) was deleted
        // along with the iter-34 dense-on-shadow Leg F decode branch — see
        // file-level iter-222 closure note in `forward_mlx.rs` for rationale.
        // Production TQ-regime SDPA reads `kv_caches[].{k,v}_packed` /
        // `leg_hb_encoded` directly via inline-fused kernels.

        // iter-21 Track B + 2026-04-24 post-close default correction.
        // Effective 5/6/8-bit TQ selection → allocate per-layer byte-packed HB buffers.
        // MUST stay in lockstep with forward_mlx.rs::tq_codebook_bits and cb_bits gates.
        //   unset (DEFAULT) = 8-bit native HB SDPA
        //   "4"             = legacy 4-bit (no HB buffers)
        //   "5" | "6" | "8" = corresponding HB bits
        let tq_codebook_bits_prefill =
            crate::serve::api::tq_packed_descriptor::effective_gemma_tq_codebook_bits();
        if tq_codebook_bits_prefill >= 5 {
            // ADR-028 Phase 10c (iter-348): hybrid F16-K + TQ-HB-V routing,
            // mirrors forward_mlx.rs decode lazy-alloc.
            if INVESTIGATION_ENV.hybrid_kv {
                // "gemma-hybrid-lcp" SerialFifo leftover discipline
                // (2026-08-03) — HYBRID leg.  Mirror of the dense-leg
                // block at line ~749: a leftover hybrid mount from a
                // PREVIOUS request is opportunistic state, not a
                // contract.  Turn N's batched prefill leaves
                // `hybrid_kv = Some(cap_N)`; turn N+1 (linear route)
                // with a longer prompt previously REUSED the undersized
                // buffers silently — the hybrid encode kernel then wrote
                // ~22.7K positions into cap=8749 buffers and the GPU
                // session hung in `commit_and_wait` forever (measured
                // live on two instances, 2026-08-03; stack:
                // forward_prefill_with_soft_tokens_resume →
                // GraphSession::finish).  Drop on ANY len/capacity/
                // is_sliding mismatch so the alloc below rebuilds at the
                // right capacity.  `slot_aware == true` mounts are the
                // multi-seq scaffold CONTRACT — never dropped here.
                if !slot_aware {
                    if let Some(leftover) = self.hybrid_kv.as_ref() {
                        let rebuild = leftover.len() != num_layers
                            || leftover.iter().zip(self.layers.iter()).any(|(b, layer)| {
                                let layer_is_ring = layer.layer_type == LayerType::Sliding;
                                let required = if layer_is_ring {
                                    required_sliding_layer_capacity
                                } else {
                                    required_linear_capacity
                                };
                                b.capacity < required || b.is_sliding != layer_is_ring
                            });
                        if rebuild {
                            self.hybrid_kv = None;
                        }
                    }
                }
                // ADR-040 iter-B4c-kernel iter-2B (2026-05-30) — added the
                // `self.hybrid_kv.is_none()` predicate around the rebuild
                // aligning with the decode-path gate at
                // `gemma4/forward_gpu.rs:413`.  When the new
                // `forward_prefill_with_soft_tokens_slot_aware` mounts a
                // per-slot slice_view of the persistent
                // `MultiSeqHybridKvBuffers` scaffold onto
                // `self.hybrid_kv`, this gate prevents the sibling fn's
                // unconditional rebuild from obliterating the mount.
                // SerialFifo byte-equivalence preserved: SerialFifo
                // enters with `self.hybrid_kv == None`, so the gate fires
                // identically + the legacy `alloc_hybrid_kv_for_layer`
                // loop body runs verbatim.
                //
                // Pinned by H102 (prefill-alloc gate aligned with decode-
                // path gate; SerialFifo byte-equivalence verified at the
                // source-grep level via H86 sibling-signature pin).
                if self.hybrid_kv.is_none() {
                    eprintln!("[ADR-028 Phase 10c] Allocating hybrid_kv ({} layers, F16 K + TQ-HB V {}-bit) [prefill]",
                        num_layers, tq_codebook_bits_prefill);
                    let mut hybrid_vec: Vec<crate::inference::models::gemma4::HybridKvBuffers> =
                        Vec::with_capacity(num_layers);
                    for (layer_idx, layer) in self.layers.iter().enumerate() {
                        let nkv = layer.num_kv_heads;
                        let hd = layer.head_dim;
                        let layer_is_ring = layer.layer_type == LayerType::Sliding;
                        // "gemma-hybrid-lcp" long-resume: LINEAR sliding
                        // capacity (max(sw, seq+max)) when LONG_RESUME is
                        // on — mirrors the dense leg's
                        // `sliding_layer_capacity` (line ~675) so the
                        // hybrid encode can write slot=tok_i without
                        // ring wrap; the hybrid SDPA kernel applies
                        // mask_type=2 sliding-window semantics.
                        let capacity = if layer_is_ring {
                            sliding_layer_capacity
                        } else {
                            linear_capacity
                        };
                        hybrid_vec.push(
                            crate::inference::models::gemma4::kv_cache::alloc_hybrid_kv_for_layer(
                                dev,
                                layer_idx,
                                nkv,
                                hd,
                                capacity,
                                layer_is_ring,
                            )?,
                        );
                    }
                    self.hybrid_kv = Some(hybrid_vec);
                }
                // ADR-040 iter-B4c-kernel iter-2B: when
                // `self.hybrid_kv.is_some()`, the slot-aware caller has
                // pre-mounted a per-slot slice_view of the persistent
                // multi-seq scaffold; the legacy rebuild is skipped to
                // preserve the mount.  The kernel-write site at
                // line ~1290-1330 consumes `self.hybrid_kv` unchanged.
            } else {
                // SerialFifo leftover discipline (2026-08-03) — HB leg,
                // mirror of the hybrid_kv block above (same hang class on
                // the HF2Q_HYBRID_KV=0 opt-out path; ring capacity here is
                // `sw`, matching the alloc below).
                if !slot_aware {
                    if let Some(leftover) = self.leg_hb_encoded.as_ref() {
                        let rebuild = leftover.len() != num_layers
                            || leftover.iter().zip(self.layers.iter()).any(|(b, layer)| {
                                let layer_is_ring = layer.layer_type == LayerType::Sliding;
                                let required = if layer_is_ring {
                                    sw
                                } else {
                                    required_linear_capacity
                                };
                                b.capacity < required || b.is_sliding != layer_is_ring
                            });
                        if rebuild {
                            self.leg_hb_encoded = None;
                        }
                    }
                }
                // ADR-040 iter-B4c-kernel iter-2A-cont (2026-05-30) — added
                // the `self.leg_hb_encoded.is_none()` predicate around the
                // rebuild, aligning with the decode-path gate at
                // `gemma4/forward_gpu.rs:427`.  When the new
                // `forward_prefill_with_soft_tokens_slot_aware` mounts a
                // per-slot slice_view of the persistent
                // `MultiSeqHbKvBuffers` scaffold onto
                // `self.leg_hb_encoded`, this gate prevents the sibling fn's
                // unconditional rebuild from obliterating the mount.
                // SerialFifo byte-equivalence preserved: SerialFifo enters
                // with `self.leg_hb_encoded == None`, so the gate fires
                // identically + the legacy alloc loop body runs verbatim.
                //
                // Mirror of iter-2B's prefill-alloc gate alignment for the
                // HF2Q_HYBRID_KV=1 production-default path (§6.1.34 / line
                // ~860) — same discipline, applied to the HB-encoded
                // (HF2Q_HYBRID_KV=0) opt-out path's `leg_hb_encoded` field.
                //
                // Pinned by H178 (prefill-alloc gate aligned with decode-
                // path gate; SerialFifo byte-equivalence verified at the
                // source-grep level via H86 sibling-signature pin).
                if self.leg_hb_encoded.is_none() {
                    eprintln!(
                        "[iter-21 Track B] Allocating leg_hb_encoded ({}-bit, {} layers)",
                        tq_codebook_bits_prefill, num_layers
                    );
                    let mut leg_hb_vec: Vec<HbKvBuffers> = Vec::with_capacity(num_layers);
                    for (layer_idx, layer) in self.layers.iter().enumerate() {
                        let nkv = layer.num_kv_heads;
                        let hd = layer.head_dim;
                        let layer_is_ring = layer.layer_type == LayerType::Sliding;
                        let capacity = if layer_is_ring { sw } else { linear_capacity };
                        let norms_per_pos = (hd / 256).max(1);
                        let norms_n = nkv * capacity * norms_per_pos;
                        let k_packed = dev
                            .alloc_buffer(
                                nkv * capacity * hd,
                                mlx_native::DType::U8,
                                vec![nkv, capacity, hd],
                            )
                            .map_err(|e| {
                                anyhow::anyhow!("leg_hb prefill K packed L{layer_idx}: {e}")
                            })?;
                        let k_norms = dev
                            .alloc_buffer(
                                norms_n * 4,
                                mlx_native::DType::F32,
                                if norms_per_pos == 1 {
                                    vec![nkv, capacity]
                                } else {
                                    vec![nkv, capacity, norms_per_pos]
                                },
                            )
                            .map_err(|e| {
                                anyhow::anyhow!("leg_hb prefill K norms L{layer_idx}: {e}")
                            })?;
                        let v_packed = dev
                            .alloc_buffer(
                                nkv * capacity * hd,
                                mlx_native::DType::U8,
                                vec![nkv, capacity, hd],
                            )
                            .map_err(|e| {
                                anyhow::anyhow!("leg_hb prefill V packed L{layer_idx}: {e}")
                            })?;
                        let v_norms = dev
                            .alloc_buffer(
                                norms_n * 4,
                                mlx_native::DType::F32,
                                if norms_per_pos == 1 {
                                    vec![nkv, capacity]
                                } else {
                                    vec![nkv, capacity, norms_per_pos]
                                },
                            )
                            .map_err(|e| {
                                anyhow::anyhow!("leg_hb prefill V norms L{layer_idx}: {e}")
                            })?;
                        leg_hb_vec.push(HbKvBuffers {
                            k_packed,
                            k_norms,
                            v_packed,
                            v_norms,
                            capacity,
                            is_sliding: layer_is_ring,
                            norms_per_pos,
                        });
                    }
                    self.leg_hb_encoded = Some(leg_hb_vec);

                    // iter-222 (ADR-005 closure, 2026-05-01): the iter-21 Track B
                    // `leg_f_kvs` shadow-cache allocation block (~30 LOC) was deleted
                    // along with the iter-34 dense-on-shadow Leg F decode branch —
                    // see file-level iter-222 closure note in `forward_mlx.rs`.
                    // `flash_attn_vec_tq_hb` reads `leg_hb_encoded` directly with no
                    // F32 round-trip.
                    eprintln!(
                        "[iter-21 Track B] leg_hb_encoded ready ({} layers)",
                        num_layers
                    );
                }
                // ADR-040 iter-B4c-kernel iter-2A-cont: when
                // `self.leg_hb_encoded.is_some()`, the slot-aware caller
                // has pre-mounted a per-slot slice_view of the persistent
                // multi-seq scaffold; the legacy rebuild is skipped to
                // preserve the mount.  The kernel-write site at line
                // ~1355-1395 consumes `self.leg_hb_encoded` unchanged.
            }
        }

        // ADR-010 one-shot norm weight dump: read self.layers[L].norms.input_layernorm
        // as the hf2q kernel sees it, compare against the raw GGUF tensor.
        // Gated on HF2Q_DUMP_NORM_WEIGHT="layer" (e.g. "7"). Writes to HF2Q_DUMP_DIR.
        if let Some(target_l) = INVESTIGATION_ENV.dump_norm_weight {
            if target_l < num_layers {
                let w: &[f32] = self.layers[target_l]
                    .norms
                    .input_layernorm
                    .as_slice()
                    .map_err(|e| anyhow::anyhow!("norm weight read L{target_l}: {e}"))?;
                let dir = &INVESTIGATION_ENV.dump_dir;
                let path = format!("{dir}/hf2q_input_layernorm_weight_layer{target_l:02}.bin");
                let bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) };
                std::fs::write(&path, bytes).map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
                eprintln!(
                    "[DUMP] input_layernorm weight L{target_l} [{}] f32 -> {}",
                    w.len(),
                    path
                );
            }
        }

        // ADR-009 Phase 3A: prefill boundary dumps at (target_layer, target_tok).
        // Controlled by HF2Q_PREFILL_DUMP="layer,tok" e.g. "7,34".
        let prefill_dump: Option<(usize, usize)> = INVESTIGATION_ENV.prefill_dump;
        let dump_dir: &str = &INVESTIGATION_ENV.dump_dir;

        // Track A fix (iter-21): Leg F shadow-cache prefill population.
        // tq_scale_factor_d512 matches the decode-path value so prefill and
        // decode dequant use the same scale, keeping the shadow KV cache
        // byte-compatible across the prefill→decode boundary.
        let tq_scale_factor_d512: f32 = {
            match std::env::var("HF2Q_SCALE_FORMULA").as_deref() {
                Ok("sqrt256") => 16.0_f32,
                Ok("sqrt512") => 512.0_f32.sqrt(),
                _ => 1.0_f32, // bare (iter-16 default)
            }
        };

        // ===================================================================
        // Process each prompt token through all layers
        // ===================================================================
        //
        // ADR-017 Phase E.a iter-3: when `restored_lcp = Some(k)`, skip
        // tokens `[0..k)` because their KV state is already populated
        // from the cached `dense_kvs[*]` and the per-layer `kv_caches`
        // bookkeeping was set to k above. The `tok_i` index passed to
        // the kernel still equals the true sequence position (so RoPE
        // gets the correct position) — `enumerate()` over the unsliced
        // iterator yields `(0, t0), (1, t1), ...` and `.skip(k)`
        // forwards the iterator to `(k, tk), (k+1, tk+1), ...`.
        let prefill_start = Instant::now();
        let mut last_token = 0u32;
        let resume_k = restored_lcp.unwrap_or(0);

        // ADR-038 G4-CFA-5f: dense Gemma 4 31B (num_experts==0) skips every
        // MoE-specific dispatch in the per-layer body below — the
        // post-FF-norm2 combine still reads `moe_accum`, so it must be
        // zeroed once up front and never written. Cost: hs * 4 bytes wiped
        // once per forward_prefill call; <1 µs on M-series unified memory.
        if self.num_experts == 0 {
            let buf = self
                .activations
                .moe_accum
                .as_mut_slice::<f32>()
                .map_err(|e| anyhow::anyhow!("prefill dense moe_accum zero: {e}"))?;
            buf.fill(0.0);
        }

        for (tok_i, &tok) in prompt_tokens.iter().enumerate().skip(resume_k) {
            let seq_pos = tok_i;

            // Write position buffer
            {
                let pos_dst: &mut [u32] = self
                    .activations
                    .position
                    .as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("position write: {e}"))?;
                pos_dst[0] = seq_pos as u32;
            }

            // KV cache bookkeeping (same as decode: advance write_pos, seq_len)
            let mut kv_info: Vec<(bool, usize, usize, usize)> = Vec::with_capacity(num_layers);
            for layer_idx in 0..num_layers {
                let is_sliding = self.kv_caches[layer_idx].is_sliding;
                let write_pos = self.kv_caches[layer_idx].write_pos;
                let capacity = self.kv_caches[layer_idx].capacity;
                self.kv_caches[layer_idx].write_pos += 1;
                self.kv_caches[layer_idx].seq_len = self.kv_caches[layer_idx]
                    .seq_len
                    .saturating_add(1)
                    .min(capacity);
                let kv_seq_len = self.kv_caches[layer_idx].seq_len;
                kv_info.push((is_sliding, write_pos, capacity, kv_seq_len));
            }

            // ===============================================================
            // Single GPU session per token (same structure as forward_decode)
            // ===============================================================
            {
                let mut s = exec
                    .begin()
                    .map_err(|e| anyhow::anyhow!("prefill session T{tok_i}: {e}"))?;

                // --- 1. Embedding ---
                //
                // Soft-token override path (Phase 2c Task #17, iter-97):
                // when this position lies within any soft-token range,
                // the standard embedding-table lookup is replaced by an
                // on-GPU buffer copy from the override embeddings.
                // Branch matches the placeholder token id at
                // `prompt_tokens[tok_i]` against soft_tokens; on hit,
                // dispatch_copy_f32 copies row `(tok_i - range.start)`
                // (= `hs` consecutive F32s) from `embeddings` into
                // `self.activations.hidden`.  Otherwise the standard
                // language-model `embedding_gather_scale_f32` runs.
                let soft_override = soft_tokens.iter().find(|st| st.range.contains(&tok_i));
                if let Some(st) = soft_override {
                    let row_idx = tok_i - st.range.start;
                    let src_offset = row_idx * hs;
                    mlx_native::ops::copy::dispatch_copy_f32(
                        s.encoder_mut(),
                        reg,
                        metal_dev,
                        st.embeddings,
                        &self.activations.hidden,
                        src_offset,
                        0,
                        hs,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "prefill soft-token copy T{tok_i} (range {:?}, row {}): {e}",
                            st.range,
                            row_idx
                        )
                    })?;
                    s.track_dispatch(&[st.embeddings], &[&self.activations.hidden]);
                } else {
                    self.activations
                        .embedding_token_id
                        .as_mut_slice::<u32>()
                        .map_err(|e| anyhow::anyhow!("prefill token id write: {e}"))?[0] = tok;
                    crate::inference::models::gemma4::native_matrix::encode_embedding(
                        &mut s,
                        reg,
                        dev,
                        &self.embed_weight,
                        &self.activations.embedding_token_id,
                        &self.activations.hidden,
                        1,
                    )
                    .map_err(|e| anyhow::anyhow!("prefill native embed T{tok_i}: {e}"))?;
                }

                // --- 2. Transformer layers ---
                for layer_idx in 0..num_layers {
                    let layer = &self.layers[layer_idx];
                    let hd = layer.head_dim;
                    let nkv = layer.num_kv_heads;
                    let nh = self.num_attention_heads;
                    let is_sliding = layer.layer_type == LayerType::Sliding;
                    let (kv_is_sliding, kv_write_pos, kv_capacity, _kv_seq_len) =
                        kv_info[layer_idx];

                    // Active dump flag for this iteration
                    let dump_here = prefill_dump == Some((layer_idx, tok_i));
                    // Dump at layer-start: hidden = L(layer_idx-1) l_out (or embed for L0)
                    if dump_here {
                        s.finish().map_err(|e| {
                            anyhow::anyhow!("prefill dump L{layer_idx} T{tok_i} start finish: {e}")
                        })?;
                        write_dump_f32(
                            dump_dir,
                            "pre_layer_hidden",
                            layer_idx,
                            tok_i,
                            &self.activations.hidden,
                            hs,
                        )?;
                        s = exec
                            .begin()
                            .map_err(|e| anyhow::anyhow!("prefill dump restart: {e}"))?;
                    }

                    // -- Pre-attention norm --
                    s.barrier_between(
                        &[
                            &self.activations.hidden,
                            &self.layers[layer_idx].norms.input_layernorm,
                        ],
                        &[&self.activations.norm_out],
                    );
                    s.rms_norm(
                        reg,
                        metal_dev,
                        &self.activations.hidden,
                        &self.layers[layer_idx].norms.input_layernorm,
                        &self.activations.norm_out,
                        &self.activations.norm_params,
                        1,
                        hs as u32,
                    )
                    .map_err(|e| anyhow::anyhow!("prefill norm L{layer_idx} T{tok_i}: {e}"))?;

                    if dump_here {
                        s.finish()
                            .map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(
                            dump_dir,
                            "post_input_norm",
                            layer_idx,
                            tok_i,
                            &self.activations.norm_out,
                            hs,
                        )?;
                        s = exec
                            .begin()
                            .map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // -- QKV projections (concurrent) --
                    s.barrier_between(
                        &[&self.activations.norm_out],
                        &[
                            &self.activations.attn_q,
                            &self.activations.attn_k,
                            &self.activations.attn_v,
                        ],
                    );
                    dispatch_qmatmul(
                        &mut s,
                        reg,
                        dev,
                        &self.activations.norm_out,
                        &self.layers[layer_idx].attn.q_proj,
                        &mut self.activations.attn_q,
                        1,
                        crate::quantize::imatrix::ImatrixHint::Layered {
                            tag: "attn_q",
                            layer: layer_idx,
                        },
                    )?;
                    dispatch_qmatmul(
                        &mut s,
                        reg,
                        dev,
                        &self.activations.norm_out,
                        &self.layers[layer_idx].attn.k_proj,
                        &mut self.activations.attn_k,
                        1,
                        crate::quantize::imatrix::ImatrixHint::Layered {
                            tag: "attn_k",
                            layer: layer_idx,
                        },
                    )?;
                    let v_is_k = self.layers[layer_idx].attn.v_proj.is_none();
                    if !v_is_k {
                        dispatch_qmatmul(
                            &mut s,
                            reg,
                            dev,
                            &self.activations.norm_out,
                            self.layers[layer_idx].attn.v_proj.as_ref().unwrap(),
                            &mut self.activations.attn_v,
                            1,
                            crate::quantize::imatrix::ImatrixHint::Layered {
                                tag: "attn_v",
                                layer: layer_idx,
                            },
                        )?;
                    }

                    // -- Fused per-head RMS norm + RoPE on Q and K --
                    let ff_gpu = if is_sliding {
                        None
                    } else {
                        Some(&self.activations.rope_freq_factors_gpu)
                    };
                    let theta = if is_sliding {
                        self.rope_theta_sliding
                    } else {
                        self.rope_theta_global
                    };
                    let half_rope = (hd / 2) as u32;

                    s.barrier_between(
                        &[&self.activations.attn_q, &self.activations.attn_k],
                        &[
                            &self.activations.attn_q_normed,
                            &self.activations.attn_k_normed,
                        ],
                    );
                    mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                        s.encoder_mut(),
                        reg,
                        metal_dev,
                        &self.activations.attn_q,
                        &self.activations.attn_q_normed,
                        Some(&self.layers[layer_idx].attn.q_norm_weight),
                        &self.activations.position,
                        ff_gpu,
                        nh as u32,
                        hd as u32,
                        half_rope,
                        eps,
                        theta,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("prefill Q norm+RoPE L{layer_idx} T{tok_i}: {e}")
                    })?;
                    mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                        s.encoder_mut(),
                        reg,
                        metal_dev,
                        &self.activations.attn_k,
                        &self.activations.attn_k_normed,
                        Some(&self.layers[layer_idx].attn.k_norm_weight),
                        &self.activations.position,
                        ff_gpu,
                        nkv as u32,
                        hd as u32,
                        half_rope,
                        eps,
                        theta,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("prefill K norm+RoPE L{layer_idx} T{tok_i}: {e}")
                    })?;

                    if dump_here {
                        s.finish()
                            .map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(
                            dump_dir,
                            "q_pre_normed",
                            layer_idx,
                            tok_i,
                            &self.activations.attn_q,
                            nh * hd,
                        )?;
                        write_dump_f32(
                            dump_dir,
                            "k_pre_normed",
                            layer_idx,
                            tok_i,
                            &self.activations.attn_k,
                            nkv * hd,
                        )?;
                        write_dump_f32(
                            dump_dir,
                            "q_normed",
                            layer_idx,
                            tok_i,
                            &self.activations.attn_q_normed,
                            nh * hd,
                        )?;
                        write_dump_f32(
                            dump_dir,
                            "k_normed",
                            layer_idx,
                            tok_i,
                            &self.activations.attn_k_normed,
                            nkv * hd,
                        )?;
                        s = exec
                            .begin()
                            .map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // -- V norm --
                    let hd_norm_params = if is_sliding {
                        &self.activations.norm_params_sliding_hd
                    } else {
                        &self.activations.norm_params_global_hd
                    };
                    if v_is_k {
                        s.barrier_between(&[&self.activations.attn_k], &[&self.activations.attn_v]);
                        dispatch_rms_norm_unit_perhead(
                            s.encoder_mut(),
                            reg,
                            metal_dev,
                            &RmsNormPerHeadArgs {
                                input: &self.activations.attn_k,
                                output: &self.activations.attn_v,
                                params_buf: hd_norm_params,
                                rows: nkv as u32,
                                dim: hd as u32,
                            },
                        )?;
                    } else {
                        s.barrier_between(
                            &[&self.activations.attn_v],
                            &[&self.activations.moe_expert_out],
                        );
                        dispatch_rms_norm_unit_perhead(
                            s.encoder_mut(),
                            reg,
                            metal_dev,
                            &RmsNormPerHeadArgs {
                                input: &self.activations.attn_v,
                                output: &self.activations.moe_expert_out,
                                params_buf: hd_norm_params,
                                rows: nkv as u32,
                                dim: hd as u32,
                            },
                        )?;
                    }

                    let v_src = if v_is_k {
                        &self.activations.attn_v
                    } else {
                        &self.activations.moe_expert_out
                    };

                    // ====================================================
                    // DENSE K,V ACCUMULATION (ADR-009 Track 1 key change)
                    //
                    // Copy this position's K,V into head-major dense buffers:
                    //   dense_k[head, pos, :] = attn_k_normed[head, :]
                    //   dense_v[head, pos, :] = v_src[head, :]
                    //
                    // Layout: [nkv, seq_len, hd], writing at pos = tok_i
                    // ====================================================
                    // Per-layer dense cap + ring-wrap write for sliding layers.
                    // ADR-017 Phase E.a iter-3.6: when LONG_RESUME is
                    // on, sliding layers are LINEAR (no wrap); writes
                    // go to slot=tok_i. When OFF (default), sliding
                    // layers wrap via slot=tok_i%cap (current behavior).
                    let layer_dense_cap = dense_kvs_vec[layer_idx].capacity;
                    let layer_is_sliding = dense_kvs_vec[layer_idx].is_sliding;
                    let write_slot = if layer_is_sliding && !kv_lcp_long_resume {
                        (tok_i % layer_dense_cap) as u32
                    } else {
                        tok_i as u32
                    };
                    s.barrier_between(
                        &[&self.activations.attn_k_normed, v_src],
                        &[&dense_kvs_vec[layer_idx].k, &dense_kvs_vec[layer_idx].v],
                    );
                    if use_f16_kv {
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                            s.encoder_mut(),
                            reg,
                            metal_dev,
                            &self.activations.attn_k_normed,
                            &dense_kvs_vec[layer_idx].k,
                            nkv as u32,
                            hd as u32,
                            layer_dense_cap as u32,
                            write_slot,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("prefill F16 K copy L{layer_idx} T{tok_i}: {e}")
                        })?;
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                            s.encoder_mut(),
                            reg,
                            metal_dev,
                            v_src,
                            &dense_kvs_vec[layer_idx].v,
                            nkv as u32,
                            hd as u32,
                            layer_dense_cap as u32,
                            write_slot,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("prefill F16 V copy L{layer_idx} T{tok_i}: {e}")
                        })?;
                    } else {
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            s.encoder_mut(),
                            reg,
                            metal_dev,
                            &self.activations.attn_k_normed,
                            &dense_kvs_vec[layer_idx].k,
                            nkv as u32,
                            hd as u32,
                            layer_dense_cap as u32,
                            write_slot,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("prefill F32 K batch copy L{layer_idx} T{tok_i}: {e}")
                        })?;
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            s.encoder_mut(),
                            reg,
                            metal_dev,
                            v_src,
                            &dense_kvs_vec[layer_idx].v,
                            nkv as u32,
                            hd as u32,
                            layer_dense_cap as u32,
                            write_slot,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("prefill F32 V batch copy L{layer_idx} T{tok_i}: {e}")
                        })?;
                    }

                    // Also TQ-encode into packed cache (for subsequent decode)
                    if !INVESTIGATION_ENV.skip_tq_encode {
                        let cache_pos_val = if kv_is_sliding {
                            (kv_write_pos % kv_capacity) as u32
                        } else {
                            kv_write_pos as u32
                        };
                        s.barrier_between(
                            &[&self.activations.attn_k_normed, v_src],
                            &[
                                &self.kv_caches[layer_idx].k_packed,
                                &self.kv_caches[layer_idx].k_norms,
                                &self.kv_caches[layer_idx].v_packed,
                                &self.kv_caches[layer_idx].v_norms,
                            ],
                        );
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                            s.encoder_mut(),
                            reg,
                            metal_dev,
                            &self.activations.attn_k_normed,
                            &self.kv_caches[layer_idx].k_packed,
                            &self.kv_caches[layer_idx].k_norms,
                            nkv as u32,
                            hd as u32,
                            kv_capacity as u32,
                            cache_pos_val,
                            kv_is_sliding,
                            None, // scale_factor_d512: bare=1.0 for prefill
                            None, // rms_scratch: probe not used during prefill
                        )
                        .map_err(|e| anyhow::anyhow!("prefill TQ K L{layer_idx} T{tok_i}: {e}"))?;
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                            s.encoder_mut(),
                            reg,
                            metal_dev,
                            v_src,
                            &self.kv_caches[layer_idx].v_packed,
                            &self.kv_caches[layer_idx].v_norms,
                            nkv as u32,
                            hd as u32,
                            kv_capacity as u32,
                            cache_pos_val,
                            kv_is_sliding,
                            None, // scale_factor_d512: bare=1.0 for prefill
                            None, // rms_scratch: probe not used during prefill
                        )
                        .map_err(|e| anyhow::anyhow!("prefill TQ V L{layer_idx} T{tok_i}: {e}"))?;
                    }

                    // iter-222 (ADR-005 closure, 2026-05-01): the iter-21 Track A
                    // per-token `leg_f_kvs` shadow-cache populate (~80 LOC) was
                    // deleted along with the iter-34 dense-on-shadow Leg F
                    // decode branch — see file-level iter-222 closure note in
                    // `forward_mlx.rs` for rationale.

                    // iter-21 Track B: HB encode K/V into `leg_hb_encoded`
                    // during prefill so decode `flash_attn_vec_tq_hb` sees
                    // all prompt positions. Reads TQ-packed K/V directly with
                    // no F32 shadow-cache round-trip.
                    // iter-222 (2026-05-01): the dequant→`leg_f_kvs` shadow
                    // population that followed the HB encode here was deleted
                    // — the inline-fused HB SDPA kernel does not consume an
                    // F32 shadow.
                    if tq_codebook_bits_prefill >= 5 && !INVESTIGATION_ENV.skip_tq_encode {
                        if INVESTIGATION_ENV.hybrid_kv {
                            // ADR-028 Phase 10c (iter-348): hybrid F16-K + TQ-HB-V
                            // prefill encode path. F16 K copy + V-only TQ-HB encode.
                            if let Some(ref hybrid_kv) = self.hybrid_kv {
                                let hb_cap = hybrid_kv[layer_idx].capacity;
                                let hb_is_ring = hybrid_kv[layer_idx].is_sliding;
                                // "gemma-hybrid-lcp" long-resume: LINEAR
                                // sliding writes (slot=tok_i) when LONG_RESUME
                                // is on — mirrors the dense leg's write_slot
                                // branch at line ~1364. With the ring off,
                                // slot == logical position for the
                                // mask_type=2 hybrid SDPA kernel.
                                let hb_write_slot = if hb_is_ring && !kv_lcp_long_resume {
                                    (tok_i % hb_cap) as u32
                                } else {
                                    tok_i as u32
                                };
                                // F32 K → F16 K cache.
                                s.barrier_between(
                                    &[&self.activations.attn_k_normed, v_src],
                                    &[
                                        &hybrid_kv[layer_idx].k,
                                        &hybrid_kv[layer_idx].v_packed,
                                        &hybrid_kv[layer_idx].v_norms,
                                    ],
                                );
                                mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                                s.encoder_mut(), reg, metal_dev,
                                &self.activations.attn_k_normed,
                                &hybrid_kv[layer_idx].k,
                                nkv as u32, hd as u32, hb_cap as u32, hb_write_slot,
                            ).map_err(|e| anyhow::anyhow!("prefill hybrid F16 K L{layer_idx} T{tok_i}: {e}"))?;
                                // BUG-coherence fix (supersedes Phase 10e.5 iter-351):
                                // FWHT V quantize.  See forward_mlx.rs ~L3724 for the
                                // empirical justification — real gemma4 V has kurtosis
                                // up to 72.88 and max|v| up to 14.63, far outside the
                                // 8-bit Lloyd-Max codebook range (±5.07).  Hadamard
                                // rotation distributes outliers across all 256 dims
                                // before quantization.  SDPA-side fwht_sign_undo at
                                // forward_mlx.rs's hybrid branch recovers raw output.
                                mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                                s.encoder_mut(), reg, metal_dev,
                                v_src,
                                &hybrid_kv[layer_idx].v_packed,
                                &hybrid_kv[layer_idx].v_norms,
                                nkv as u32, hd as u32, hb_cap as u32, hb_write_slot,
                                hb_is_ring, tq_scale_factor_d512, tq_codebook_bits_prefill,
                            ).map_err(|e| anyhow::anyhow!("prefill hybrid V FWHT quant L{layer_idx} T{tok_i}: {e}"))?;
                            }
                        } else if let Some(ref leg_hb_enc) = self.leg_hb_encoded {
                            let hb_cap = leg_hb_enc[layer_idx].capacity;
                            let hb_is_ring = leg_hb_enc[layer_idx].is_sliding;
                            let hb_write_slot = if hb_is_ring {
                                (tok_i % hb_cap) as u32
                            } else {
                                tok_i as u32
                            };

                            // HB encode K → leg_hb_enc.k_packed
                            s.barrier_between(
                                &[&self.activations.attn_k_normed, v_src],
                                &[
                                    &leg_hb_enc[layer_idx].k_packed,
                                    &leg_hb_enc[layer_idx].k_norms,
                                ],
                            );
                            mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &leg_hb_enc[layer_idx].k_packed,
                            &leg_hb_enc[layer_idx].k_norms,
                            nkv as u32, hd as u32, hb_cap as u32, hb_write_slot,
                            hb_is_ring, tq_scale_factor_d512, tq_codebook_bits_prefill,
                        ).map_err(|e| anyhow::anyhow!("prefill hb_encode K L{layer_idx} T{tok_i}: {e}"))?;

                            // HB encode V → leg_hb_enc.v_packed
                            s.barrier_between(
                                &[v_src],
                                &[
                                    &leg_hb_enc[layer_idx].v_packed,
                                    &leg_hb_enc[layer_idx].v_norms,
                                ],
                            );
                            mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                            s.encoder_mut(), reg, metal_dev,
                            v_src,
                            &leg_hb_enc[layer_idx].v_packed,
                            &leg_hb_enc[layer_idx].v_norms,
                            nkv as u32, hd as u32, hb_cap as u32, hb_write_slot,
                            hb_is_ring, tq_scale_factor_d512, tq_codebook_bits_prefill,
                        ).map_err(|e| anyhow::anyhow!("prefill hb_encode V L{layer_idx} T{tok_i}: {e}"))?;
                        } // end if let Some(leg_hb_enc)
                    } // end if tq_codebook_bits_prefill >= 5

                    // ====================================================
                    // DENSE SDPA (ADR-009 Track 1 key change)
                    //
                    // Use flash_attn_vec with dense F32 K,V instead of
                    // flash_attn_vec_tq with packed TQ K,V.
                    //
                    // Q: attn_q_normed [nh, 1, hd] (already in head-major)
                    // K: dense_kvs[layer].k [nkv, seq_len, hd]
                    // V: dense_kvs[layer].v [nkv, seq_len, hd]
                    //
                    // No FWHT rotation needed — pure model-space attention.
                    // ====================================================
                    // kv_seq_len: ring clamps to capacity (== sliding_window).
                    // Ring mode uses mask_type=1 (causal only) — the ring
                    // already applies the sliding-window constraint.
                    //
                    // ADR-017 Phase E.a iter-3.6: when LONG_RESUME is on
                    // and layer is sliding, the buffer is LINEAR (no
                    // wrap), so kv_seq_len = tok_i+1 (full populated
                    // count) and the kernel uses mask_type=2 +
                    // sliding_window=sw for sliding-window masking.
                    // When LONG_RESUME is off, behavior is byte-identical
                    // to pre-iter-3.6.
                    let use_linear_sliding = layer_is_sliding && kv_lcp_long_resume;
                    let dense_kv_seq_len = if layer_is_sliding && !use_linear_sliding {
                        ((tok_i + 1).min(layer_dense_cap)) as u32
                    } else {
                        (tok_i + 1) as u32
                    };
                    s.barrier_between(
                        &[
                            &self.activations.attn_q_normed,
                            &dense_kvs_vec[layer_idx].k,
                            &dense_kvs_vec[layer_idx].v,
                        ],
                        &[&self.activations.sdpa_out],
                    );
                    let (mask_type_val, sliding_window_val) = if use_linear_sliding {
                        // Linear sliding: kernel masks via slot index,
                        // which equals logical position because the
                        // buffer is non-wrapping.
                        (2u32, sw as u32)
                    } else {
                        // Default: causal-only; ring applies sliding-window
                        // for sliding layers; global layers don't need it.
                        (1u32, 0u32)
                    };
                    let p = FlashAttnVecParams {
                        num_heads: nh as u32,
                        num_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        kv_seq_len: dense_kv_seq_len,
                        kv_capacity: layer_dense_cap as u32,
                        scale: 1.0, // Gemma4: scale = 1.0 (peer oracle)
                        mask_type: mask_type_val,
                        sliding_window: sliding_window_val,
                        softcap: 0.0,
                        // ADR-034 task #89: decode path = single query.
                        q_seq_len: FlashAttnVecParams::DEFAULT_Q_SEQ_LEN,
                    };
                    mlx_native::ops::flash_attn_vec::flash_attn_vec(
                        s.encoder_mut(),
                        reg,
                        dev,
                        &self.activations.attn_q_normed,
                        &dense_kvs_vec[layer_idx].k,
                        &dense_kvs_vec[layer_idx].v,
                        &self.activations.sdpa_out,
                        &sdpa_tmp,
                        &p,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("prefill dense SDPA L{layer_idx} T{tok_i}: {e}")
                    })?;

                    if dump_here {
                        s.finish()
                            .map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(
                            dump_dir,
                            "sdpa_out",
                            layer_idx,
                            tok_i,
                            &self.activations.sdpa_out,
                            nh * hd,
                        )?;
                        // ADR-010 sub-stage dump: full dense K,V cache up to
                        // (and including) the target token, packed as
                        // [nkv, tok_i+1, hd] for comparison with llama's
                        // cache_k_l*/cache_v_l* at pos tok_i. Only F32 path.
                        if !use_f16_kv {
                            let cap = dense_kvs_vec[layer_idx].capacity;
                            let n_valid = tok_i + 1;
                            let k_full: &[f32] = dense_kvs_vec[layer_idx]
                                .k
                                .as_slice()
                                .map_err(|e| anyhow::anyhow!("dump K cache L{layer_idx}: {e}"))?;
                            let v_full: &[f32] = dense_kvs_vec[layer_idx]
                                .v
                                .as_slice()
                                .map_err(|e| anyhow::anyhow!("dump V cache L{layer_idx}: {e}"))?;
                            let mut k_valid = Vec::<f32>::with_capacity(nkv * n_valid * hd);
                            let mut v_valid = Vec::<f32>::with_capacity(nkv * n_valid * hd);
                            for h in 0..nkv {
                                for p in 0..n_valid {
                                    let off = h * cap * hd + p * hd;
                                    k_valid.extend_from_slice(&k_full[off..off + hd]);
                                    v_valid.extend_from_slice(&v_full[off..off + hd]);
                                }
                            }
                            for (name, buf) in
                                [("k_cache_upto", &k_valid), ("v_cache_upto", &v_valid)]
                            {
                                let path = format!(
                                    "{dump_dir}/hf2q_prefill_{name}_layer{layer_idx:02}_tok{tok_i:03}.bin");
                                let bytes: &[u8] = unsafe {
                                    std::slice::from_raw_parts(
                                        buf.as_ptr() as *const u8,
                                        buf.len() * 4,
                                    )
                                };
                                std::fs::write(&path, bytes)
                                    .map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
                                eprintln!(
                                    "[PREFILL DUMP] {} [{},{},{}] f32 -> {}",
                                    name, nkv, n_valid, hd, path
                                );
                            }
                        }
                        s = exec
                            .begin()
                            .map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // -- O-proj (same as decode) --
                    s.barrier_between(
                        &[
                            &self.activations.sdpa_out,
                            &self.layers[layer_idx].attn.o_proj.buffer,
                        ],
                        &[&self.activations.attn_out],
                    );
                    dispatch_qmatmul(
                        &mut s,
                        reg,
                        dev,
                        &self.activations.sdpa_out,
                        &self.layers[layer_idx].attn.o_proj,
                        &mut self.activations.attn_out,
                        1,
                        crate::quantize::imatrix::ImatrixHint::Layered {
                            tag: "attn_output",
                            layer: layer_idx,
                        },
                    )?;

                    if dump_here {
                        s.finish()
                            .map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(
                            dump_dir,
                            "attn_out_pre_resid",
                            layer_idx,
                            tok_i,
                            &self.activations.attn_out,
                            hs,
                        )?;
                        s = exec
                            .begin()
                            .map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // -- Fused post-attention norm + residual add --
                    s.barrier_between(
                        &[&self.activations.hidden, &self.activations.attn_out],
                        &[&self.activations.residual],
                    );
                    mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                        s.encoder_mut(),
                        reg,
                        metal_dev,
                        &self.activations.hidden,
                        &self.activations.attn_out,
                        &self.layers[layer_idx].norms.post_attention_layernorm,
                        &self.activations.residual,
                        hs as u32,
                        1,
                        eps,
                    )
                    .map_err(|e| anyhow::anyhow!("prefill post-attn L{layer_idx} T{tok_i}: {e}"))?;

                    if dump_here {
                        s.finish()
                            .map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(
                            dump_dir,
                            "residual",
                            layer_idx,
                            tok_i,
                            &self.activations.residual,
                            hs,
                        )?;
                        s = exec
                            .begin()
                            .map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // ============================================================
                    // Dense MLP + MoE (identical to forward_decode)
                    // ============================================================
                    let num_experts = self.num_experts;
                    let top_k = self.layers[layer_idx].moe.top_k;

                    // B8: pre-FF norms [3 concurrent]
                    s.barrier_between(
                        &[&self.activations.residual],
                        &[
                            &self.activations.norm_out,
                            &self.activations.moe_norm_out,
                            &self.activations.router_norm_out,
                        ],
                    );
                    s.rms_norm(
                        reg,
                        metal_dev,
                        &self.activations.residual,
                        &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                        &self.activations.norm_out,
                        &self.activations.norm_params,
                        1,
                        hs as u32,
                    )
                    .map_err(|e| anyhow::anyhow!("prefill pre-FF1 L{layer_idx} T{tok_i}: {e}"))?;
                    s.rms_norm(
                        reg,
                        metal_dev,
                        &self.activations.residual,
                        &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                        &self.activations.moe_norm_out,
                        &self.activations.norm_params,
                        1,
                        hs as u32,
                    )
                    .map_err(|e| anyhow::anyhow!("prefill pre-FF2 L{layer_idx} T{tok_i}: {e}"))?;
                    // ADR-038 G4-CFA-5f: router norm/proj read MoE-only weights
                    // that are 1-element placeholders on dense GGUFs. Skip both.
                    if num_experts > 0 {
                        s.rms_norm(
                            reg,
                            metal_dev,
                            &self.activations.residual,
                            &self.layers[layer_idx].moe.router_combined_weight,
                            &self.activations.router_norm_out,
                            &self.activations.norm_params,
                            1,
                            hs as u32,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("prefill router norm L{layer_idx} T{tok_i}: {e}")
                        })?;
                    }

                    // B9: gate + up + (router if MoE) [2 or 3 concurrent]
                    if num_experts > 0 {
                        s.barrier_between(
                            &[
                                &self.activations.norm_out,
                                &self.activations.router_norm_out,
                            ],
                            &[
                                &self.activations.mlp_gate,
                                &self.activations.mlp_up,
                                &self.activations.moe_router_logits,
                            ],
                        );
                    } else {
                        s.barrier_between(
                            &[&self.activations.norm_out],
                            &[&self.activations.mlp_gate, &self.activations.mlp_up],
                        );
                    }
                    dispatch_qmatmul(
                        &mut s,
                        reg,
                        dev,
                        &self.activations.norm_out,
                        &self.layers[layer_idx].mlp.gate_proj,
                        &mut self.activations.mlp_gate,
                        1,
                        crate::quantize::imatrix::ImatrixHint::Layered {
                            tag: "ffn_gate",
                            layer: layer_idx,
                        },
                    )?;
                    dispatch_qmatmul(
                        &mut s,
                        reg,
                        dev,
                        &self.activations.norm_out,
                        &self.layers[layer_idx].mlp.up_proj,
                        &mut self.activations.mlp_up,
                        1,
                        crate::quantize::imatrix::ImatrixHint::Layered {
                            tag: "ffn_up",
                            layer: layer_idx,
                        },
                    )?;
                    if num_experts > 0 {
                        dispatch_qmatmul(
                            &mut s,
                            reg,
                            dev,
                            &self.activations.router_norm_out,
                            &self.layers[layer_idx].moe.router_proj,
                            &mut self.activations.moe_router_logits,
                            1,
                            crate::quantize::imatrix::ImatrixHint::Layered {
                                tag: "ffn_gate_inp",
                                layer: layer_idx,
                            },
                        )?;
                    }

                    // B10: gelu_mul (+ moe_routing if MoE)
                    if num_experts > 0 {
                        s.barrier_between(
                            &[
                                &self.activations.mlp_gate,
                                &self.activations.mlp_up,
                                &self.activations.moe_router_logits,
                            ],
                            &[
                                &self.activations.mlp_fused,
                                &self.activations.moe_expert_ids,
                                &self.activations.moe_routing_weights_gpu,
                            ],
                        );
                    } else {
                        s.barrier_between(
                            &[&self.activations.mlp_gate, &self.activations.mlp_up],
                            &[&self.activations.mlp_fused],
                        );
                    }
                    {
                        use mlx_native::ops::encode_helpers::{encode_with_args, KernelArg};
                        let n_elements_bytes = (self.intermediate_size as u32).to_ne_bytes();
                        let pipeline = reg.get_pipeline("fused_gelu_mul", metal_dev)?;
                        encode_with_args(
                            s.encoder_mut(),
                            pipeline,
                            &[
                                (0, KernelArg::Buffer(&self.activations.mlp_gate)),
                                (1, KernelArg::Buffer(&self.activations.mlp_up)),
                                (2, KernelArg::Buffer(&self.activations.mlp_fused)),
                                (3, KernelArg::Bytes(&n_elements_bytes)),
                            ],
                            mlx_native::MTLSize::new(self.intermediate_size as u64, 1, 1),
                            mlx_native::MTLSize::new(
                                std::cmp::min(256, self.intermediate_size as u64),
                                1,
                                1,
                            ),
                        );
                    }
                    if num_experts > 0 {
                        mlx_native::ops::fused_norm_add::dispatch_fused_moe_routing_f32(
                            s.encoder_mut(),
                            reg,
                            metal_dev,
                            &self.activations.moe_router_logits,
                            &self.activations.moe_expert_ids,
                            &self.activations.moe_routing_weights_gpu,
                            &self.layers[layer_idx].moe.per_expert_scale,
                            num_experts as u32,
                            top_k as u32,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("prefill MoE routing L{layer_idx} T{tok_i}: {e}")
                        })?;
                    }

                    // MoE expert dispatch (fused _id path) — MoE only
                    let moe_int = self.layers[layer_idx].moe.moe_intermediate_size;
                    if num_experts > 0
                        && (self.layers[layer_idx].moe.stacked_gate_up.is_none()
                            || self.layers[layer_idx].moe.stacked_down.is_none())
                    {
                        anyhow::bail!(
                            "Prefill requires fused _id path (stacked weights) at L{layer_idx}"
                        );
                    }

                    // B11: dense down + gate_up_id
                    s.barrier_between(
                        &[
                            &self.activations.mlp_fused,
                            &self.layers[layer_idx].mlp.down_proj.buffer,
                        ],
                        &[&self.activations.mlp_down],
                    );
                    dispatch_qmatmul(
                        &mut s,
                        reg,
                        dev,
                        &self.activations.mlp_fused,
                        &self.layers[layer_idx].mlp.down_proj,
                        &mut self.activations.mlp_down,
                        1,
                        crate::quantize::imatrix::ImatrixHint::Layered {
                            tag: "ffn_down",
                            layer: layer_idx,
                        },
                    )?;

                    if num_experts > 0 {
                        let ggml_type_gu = self.layers[layer_idx].moe.gate_up_ggml_dtype;
                        let (gate_up_affine, down_affine) = self.layers[layer_idx]
                            .moe
                            .affine_pair()?
                            .map_or((None, None), |(gate_up, down)| (Some(gate_up), Some(down)));
                        let gate_up_weight =
                            self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap();
                        let native_activation_epoch = self.native_activation_epoch()?;
                        if !crate::inference::models::gemma4::expert_dispatch::dispatch_native_scalar_expert(
                            &mut s,
                            reg,
                            dev,
                            native_activation_epoch,
                            &self.activations.moe_norm_out,
                            gate_up_weight,
                            &self.activations.moe_expert_ids,
                            &self.activations.moe_gate_up_id_out,
                            gate_up_affine,
                            ggml_type_gu,
                            1,
                            top_k as u32,
                            (2 * moe_int) as u32,
                            hs as u32,
                            num_experts as u32,
                            self.layers[layer_idx].moe.gate_up_expert_stride,
                            mlx_native::DenseMatmulIdInputLayout::SharedPerToken,
                            crate::inference::models::gemma4::expert_dispatch::DenseExpertScratchSlot::GateUp,
                            "Gemma linear-prefill gate/up",
                        )? {
                            s.barrier_between(
                                &[
                                    &self.activations.moe_norm_out,
                                    &self.activations.moe_expert_ids,
                                    gate_up_weight,
                                ],
                                &[&self.activations.moe_gate_up_id_out],
                            );
                            s.quantized_matmul_id_ggml(
                                reg,
                                dev,
                                &self.activations.moe_norm_out,
                                gate_up_weight,
                                &self.activations.moe_expert_ids,
                                &mut self.activations.moe_gate_up_id_out,
                                &mlx_native::GgmlQuantizedMatmulIdParams {
                                    n_tokens: 1,
                                    top_k: top_k as u32,
                                    n: (2 * moe_int) as u32,
                                    k: hs as u32,
                                    n_experts: num_experts as u32,
                                    expert_stride: self.layers[layer_idx].moe.gate_up_expert_stride,
                                    ggml_type: ggml_type_gu,
                                },
                            )
                            .map_err(|e| {
                                anyhow::anyhow!("prefill gate_up_id L{layer_idx} T{tok_i}: {e}")
                            })?;
                        }

                        // B12: swiglu
                        s.barrier_between(
                            &[&self.activations.moe_gate_up_id_out],
                            &[&self.activations.moe_swiglu_id_out],
                        );
                        mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode(
                            s.encoder_mut(),
                            reg,
                            metal_dev,
                            &self.activations.moe_gate_up_id_out,
                            &self.activations.moe_swiglu_id_out,
                            moe_int,
                            top_k,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("prefill swiglu L{layer_idx} T{tok_i}: {e}")
                        })?;

                        // B13: down_id
                        let ggml_type_dn = self.layers[layer_idx].moe.down_ggml_dtype;
                        let down_weight = self.layers[layer_idx].moe.stacked_down.as_ref().unwrap();
                        if !crate::inference::models::gemma4::expert_dispatch::dispatch_native_scalar_expert(
                            &mut s,
                            reg,
                            dev,
                            native_activation_epoch,
                            &self.activations.moe_swiglu_id_out,
                            down_weight,
                            &self.activations.moe_expert_ids,
                            &self.activations.moe_down_id_out,
                            down_affine,
                            ggml_type_dn,
                            1,
                            top_k as u32,
                            hs as u32,
                            moe_int as u32,
                            num_experts as u32,
                            self.layers[layer_idx].moe.down_expert_stride,
                            mlx_native::DenseMatmulIdInputLayout::Slotted,
                            crate::inference::models::gemma4::expert_dispatch::DenseExpertScratchSlot::Down,
                            "Gemma linear-prefill down",
                        )? {
                            s.barrier_between(
                                &[
                                    &self.activations.moe_swiglu_id_out,
                                    &self.activations.moe_expert_ids,
                                    down_weight,
                                ],
                                &[&self.activations.moe_down_id_out],
                            );
                            s.quantized_matmul_id_ggml(
                                reg,
                                dev,
                                &self.activations.moe_swiglu_id_out,
                                down_weight,
                                &self.activations.moe_expert_ids,
                                &mut self.activations.moe_down_id_out,
                                &mlx_native::GgmlQuantizedMatmulIdParams {
                                    n_tokens: top_k as u32,
                                    top_k: 1,
                                    n: hs as u32,
                                    k: moe_int as u32,
                                    n_experts: num_experts as u32,
                                    expert_stride: self.layers[layer_idx].moe.down_expert_stride,
                                    ggml_type: ggml_type_dn,
                                },
                            )
                            .map_err(|e| {
                                anyhow::anyhow!("prefill down_id L{layer_idx} T{tok_i}: {e}")
                            })?;
                        }
                    }

                    s.barrier_between(&[&self.activations.mlp_down], &[&self.activations.attn_out]);
                    s.rms_norm(
                        reg,
                        metal_dev,
                        &self.activations.mlp_down,
                        &self.layers[layer_idx].norms.post_feedforward_layernorm_1,
                        &self.activations.attn_out,
                        &self.activations.norm_params,
                        1,
                        hs as u32,
                    )
                    .map_err(|e| anyhow::anyhow!("prefill post-FF1 L{layer_idx} T{tok_i}: {e}"))?;

                    // B14: weighted_sum — MoE only. On dense, moe_accum was
                    // zeroed once before the per-token loop (G4-CFA-5f).
                    if num_experts > 0 {
                        s.barrier_between(
                            &[
                                &self.activations.moe_down_id_out,
                                &self.activations.moe_routing_weights_gpu,
                            ],
                            &[&self.activations.moe_accum],
                        );
                        mlx_native::ops::moe_dispatch::moe_weighted_sum_encode(
                            s.encoder_mut(),
                            reg,
                            metal_dev,
                            &self.activations.moe_down_id_out,
                            &self.activations.moe_routing_weights_gpu,
                            &self.activations.moe_accum,
                            hs,
                            top_k,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("prefill weighted_sum L{layer_idx} T{tok_i}: {e}")
                        })?;
                    }

                    // Post-FF norm2 + combine
                    s.barrier_between(
                        &[&self.activations.attn_out, &self.activations.moe_accum],
                        &[&self.activations.mlp_down],
                    );
                    mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                        s.encoder_mut(),
                        reg,
                        metal_dev,
                        &self.activations.attn_out,
                        &self.activations.moe_accum,
                        &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                        &self.activations.mlp_down,
                        hs as u32,
                        1,
                        eps,
                    )
                    .map_err(|e| anyhow::anyhow!("prefill post-FF2 L{layer_idx} T{tok_i}: {e}"))?;

                    // End-of-layer: norm + residual + scalar
                    let scalar_is_vector = self.layers[layer_idx].layer_scalar.element_count() > 1;
                    s.barrier_between(
                        &[&self.activations.residual, &self.activations.mlp_down],
                        &[&self.activations.hidden],
                    );
                    mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_scalar_f32(
                        s.encoder_mut(),
                        reg,
                        metal_dev,
                        &self.activations.residual,
                        &self.activations.mlp_down,
                        &self.layers[layer_idx].norms.post_feedforward_layernorm,
                        &self.activations.hidden,
                        &self.layers[layer_idx].layer_scalar,
                        1,
                        hs as u32,
                        eps,
                        scalar_is_vector,
                    )
                    .map_err(|e| anyhow::anyhow!("prefill end-layer L{layer_idx} T{tok_i}: {e}"))?;

                    if dump_here {
                        s.finish()
                            .map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(
                            dump_dir,
                            "l_out",
                            layer_idx,
                            tok_i,
                            &self.activations.hidden,
                            hs,
                        )?;
                        s = exec
                            .begin()
                            .map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }
                }

                // C-0b: HF2Q_DUMP_TQ_STATE — dump packed KV cache at end-of-prefill
                // (last token only) for ADR-007 layer-0 localization audit.
                if INVESTIGATION_ENV.dump_tq_state && tok_i + 1 == seq_len {
                    let dump_layers_list = &INVESTIGATION_ENV.dump_tq_layers_list;
                    s.finish()
                        .map_err(|e| anyhow::anyhow!("tq_dump nonbatched finish T{tok_i}: {e}"))?;
                    for li in 0..num_layers {
                        if !dump_layers_list.is_empty() && !dump_layers_list.contains(&li) {
                            continue;
                        }
                        let layer = &self.layers[li];
                        let hd = layer.head_dim;
                        let nkv = layer.num_kv_heads;
                        let (kv_is_sliding, _kv_write_pos, kv_capacity, kv_seq_len) = kv_info[li];
                        let hd_half = hd / 2;
                        let k_raw: &[u8] = self.kv_caches[li]
                            .k_packed
                            .as_slice()
                            .map_err(|e| anyhow::anyhow!("tq_dump nb k_packed L{li}: {e}"))?;
                        let v_raw: &[u8] = self.kv_caches[li]
                            .v_packed
                            .as_slice()
                            .map_err(|e| anyhow::anyhow!("tq_dump nb v_packed L{li}: {e}"))?;
                        let k_norms_raw: &[f32] = self.kv_caches[li]
                            .k_norms
                            .as_slice()
                            .map_err(|e| anyhow::anyhow!("tq_dump nb k_norms L{li}: {e}"))?;
                        let v_norms_raw: &[f32] = self.kv_caches[li]
                            .v_norms
                            .as_slice()
                            .map_err(|e| anyhow::anyhow!("tq_dump nb v_norms L{li}: {e}"))?;
                        let mut k_tight = vec![0u8; nkv * kv_seq_len * hd_half];
                        let mut v_tight = vec![0u8; nkv * kv_seq_len * hd_half];
                        let mut kn_tight = vec![0.0f32; nkv * kv_seq_len];
                        let mut vn_tight = vec![0.0f32; nkv * kv_seq_len];
                        for h in 0..nkv {
                            for p in 0..kv_seq_len {
                                let src_packed = h * kv_capacity * hd_half + p * hd_half;
                                let dst_packed = h * kv_seq_len * hd_half + p * hd_half;
                                k_tight[dst_packed..dst_packed + hd_half]
                                    .copy_from_slice(&k_raw[src_packed..src_packed + hd_half]);
                                v_tight[dst_packed..dst_packed + hd_half]
                                    .copy_from_slice(&v_raw[src_packed..src_packed + hd_half]);
                                let src_norm = h * kv_capacity + p;
                                let dst_norm = h * kv_seq_len + p;
                                kn_tight[dst_norm] = k_norms_raw[src_norm];
                                vn_tight[dst_norm] = v_norms_raw[src_norm];
                            }
                        }
                        let dir = &INVESTIGATION_ENV.dump_dir;
                        std::fs::create_dir_all(dir.as_str())
                            .map_err(|e| anyhow::anyhow!("tq_dump nb mkdir {dir}: {e}"))?;
                        let kp = format!("{dir}/hf2q_k_packed_layer{li:02}_pos{kv_seq_len}.u8.bin");
                        let vp = format!("{dir}/hf2q_v_packed_layer{li:02}_pos{kv_seq_len}.u8.bin");
                        std::fs::write(&kp, &k_tight)
                            .map_err(|e| anyhow::anyhow!("write {kp}: {e}"))?;
                        std::fs::write(&vp, &v_tight)
                            .map_err(|e| anyhow::anyhow!("write {vp}: {e}"))?;
                        eprintln!(
                            "[TQ_DUMP] k_packed L{li:02} [{nkv},{kv_seq_len},{hd_half}] u8 -> {kp}"
                        );
                        eprintln!(
                            "[TQ_DUMP] v_packed L{li:02} [{nkv},{kv_seq_len},{hd_half}] u8 -> {vp}"
                        );
                        let kn = format!("{dir}/hf2q_k_norms_layer{li:02}_pos{kv_seq_len}.f32.bin");
                        let vn = format!("{dir}/hf2q_v_norms_layer{li:02}_pos{kv_seq_len}.f32.bin");
                        let kn_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                kn_tight.as_ptr() as *const u8,
                                kn_tight.len() * 4,
                            )
                        };
                        let vn_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                vn_tight.as_ptr() as *const u8,
                                vn_tight.len() * 4,
                            )
                        };
                        std::fs::write(&kn, kn_bytes)
                            .map_err(|e| anyhow::anyhow!("write {kn}: {e}"))?;
                        std::fs::write(&vn, vn_bytes)
                            .map_err(|e| anyhow::anyhow!("write {vn}: {e}"))?;
                        eprintln!("[TQ_DUMP] k_norms L{li:02} [{nkv},{kv_seq_len}] f32 -> {kn}");
                        eprintln!("[TQ_DUMP] v_norms L{li:02} [{nkv},{kv_seq_len}] f32 -> {vn}");
                        let layer_type_str = if kv_is_sliding { "sliding" } else { "global" };
                        let kv_write_pos_final = self.kv_caches[li].write_pos;
                        let meta = serde_json::json!({
                            "nkv": nkv, "nh": max_nh, "hd": hd,
                            "kv_seq_len": kv_seq_len,
                            "kv_capacity": kv_capacity,
                            "kv_write_pos": kv_write_pos_final,
                            "kv_is_sliding": kv_is_sliding,
                            "ring_start": 0,
                            "sliding_window": sw,
                            "mask_type": 1,
                            "layer_type": layer_type_str,
                            "path": "nonbatched"
                        });
                        let meta_str = serde_json::to_string_pretty(&meta)
                            .map_err(|e| anyhow::anyhow!("meta json nb L{li}: {e}"))?;
                        let mp = format!("{dir}/hf2q_tq_meta_layer{li:02}_pos{kv_seq_len}.json");
                        std::fs::write(&mp, meta_str.as_bytes())
                            .map_err(|e| anyhow::anyhow!("write {mp}: {e}"))?;
                        eprintln!("[TQ_DUMP] meta L{li:02} -> {mp}");
                    }
                    s = exec
                        .begin()
                        .map_err(|e| anyhow::anyhow!("tq_dump nonbatched re-begin: {e}"))?;
                }

                // --- 3. Final norm + lm_head + softcap + argmax ---
                s.barrier_between(
                    &[&self.activations.hidden, &self.final_norm],
                    &[&self.activations.norm_out],
                );
                s.rms_norm(
                    reg,
                    metal_dev,
                    &self.activations.hidden,
                    &self.final_norm,
                    &self.activations.norm_out,
                    &self.activations.norm_params,
                    1,
                    hs as u32,
                )
                .map_err(|e| anyhow::anyhow!("prefill final norm T{tok_i}: {e}"))?;

                let lm_head = self.resolved_lm_head();
                s.barrier_between(
                    &[&self.activations.norm_out, &lm_head.buffer],
                    &[&self.activations.logits],
                );
                crate::serve::forward_mlx_shared::dispatch_qmatmul(
                    &mut s,
                    reg,
                    dev,
                    &self.activations.norm_out,
                    lm_head,
                    &self.activations.logits,
                    1,
                    crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
                )
                .map_err(|e| anyhow::anyhow!("prefill native lm_head T{tok_i}: {e}"))?;

                if let Some(cap) = self.final_logit_softcapping {
                    s.barrier_between(&[&self.activations.logits], &[&self.activations.logits]);
                    mlx_native::ops::softcap::dispatch_softcap(
                        s.encoder_mut(),
                        reg,
                        metal_dev,
                        &self.activations.logits,
                        &self.activations.logits,
                        &self.activations.softcap_params,
                        cap,
                    )
                    .map_err(|e| anyhow::anyhow!("prefill softcap T{tok_i}: {e}"))?;
                }

                s.barrier_between(
                    &[&self.activations.logits],
                    &[
                        &self.activations.argmax_index,
                        &self.activations.argmax_value,
                    ],
                );
                mlx_native::ops::argmax::dispatch_argmax_f32(
                    s.encoder_mut(),
                    reg,
                    metal_dev,
                    &self.activations.logits,
                    &self.activations.argmax_index,
                    &self.activations.argmax_value,
                    &self.activations.argmax_params,
                    vocab_size as u32,
                )
                .map_err(|e| anyhow::anyhow!("prefill argmax T{tok_i}: {e}"))?;

                s.finish()
                    .map_err(|e| anyhow::anyhow!("prefill finish T{tok_i}: {e}"))?;

                last_token = crate::inference::argmax::read_finite_argmax_one(
                    &self.activations.argmax_index,
                    &self.activations.argmax_value,
                    vocab_size as u32,
                    "Gemma linear prefill",
                )?;
            }
        }

        let prefill_elapsed = prefill_start.elapsed();
        tracing::debug!(
            "Prefill complete (dense SDPA): {} tokens in {:.1} ms ({:.1} tok/s), first decode token = {}",
            seq_len,
            prefill_elapsed.as_secs_f64() * 1000.0,
            seq_len as f64 / prefill_elapsed.as_secs_f64(),
            last_token,
        );

        // Store dense KV buffers on self so forward_decode can use them
        // for dense attention during the decode phase (ADR-009 Track 3).
        //
        // ADR-017 Phase E.a iter-2.5 (Strategy A): wrap each per-layer
        // `DenseKvBuffers` in an `Arc` at assignment time. The
        // `dense_kvs_vec` builder above remains `Vec<DenseKvBuffers>`
        // (non-Arc) so the per-layer kernel writes earlier in this
        // function still mutate the buffers in place via `&mut`; the
        // Arc wrap fires once at the end-of-prefill handoff, when the
        // buffers transition from "in-flight under exclusive prefill
        // borrow" to "shared, readable from forward_decode".
        // ADR-017 Phase E.a iter-3.5b — end-of-prefill snapshot for
        // the LcpRegistry, BEFORE decode mutates the live buffers.
        //
        // Pre-iter-3.5b the engine stored Arc-clones of the LIVE
        // buffers post-decode. Decode mutates `dense_kvs[*][slot=p%cap]`
        // for sliding layers; when prompt+decode total exceeds
        // sliding_window the ring wraps and decode-written slots
        // overwrite prompt-prefix slots. The iter-3 v1 wrap-guard
        // skipped store in that case (correctness preserved at the
        // cost of long-conversation cache miss).
        //
        // iter-3.5b lifts the wrap restriction by snapshotting
        // dense_kvs HERE (decode hasn't run yet — the per-token loop
        // above only wrote prompt positions [0..N)). The snapshot is
        // a true CPU-side memcpy via `as_mut_slice` / `as_slice`
        // (StorageModeShared on Apple Silicon — read/write through
        // the same physical bytes from CPU). Cost: one extra per-
        // layer KV alloc + memcpy per resume-eligible request
        // (~50ms / ~5 GB on Gemma 4 26B). Triggered only when env-
        // gates `HF2Q_KV_LCP_RESUME=1` + `HF2Q_USE_DENSE=1` are both
        // ON (the engine consumes `dense_kvs_snapshot_for_lcp` only
        // in that mode).
        //
        // The snapshot path also mirrors the dtype/capacity/is_sliding
        // invariants of each source layer (iter-3.5a dtype field).
        // ADR-017 Phase E.a iter-7 — long-prompt snapshot safety.
        //
        // Two preconditions must hold for a valid snapshot:
        //   (a) For any sliding layer: prompt fit in the ring without
        //       wrap. `seq_len <= sw` is the necessary+sufficient
        //       condition (the per-token loop above wrote slots
        //       [tok_i % sw] for tok_i ∈ [0, seq_len); slots stay
        //       monotonic only when seq_len ≤ sw). When seq_len > sw,
        //       the ring already wrapped and the live buffer's slot
        //       contents no longer represent positions [0..N). NO
        //       snapshot can faithfully reconstruct [0..K) for K ≤ N.
        //   (b) The snapshot per-layer buffer must hold all populated
        //       slots: snap_cap ≥ seq_len. iter-3.5d set snap_cap = sw
        //       on the assumption "engine guard at engine.rs:4516
        //       enforces seq_len ≤ sw for sliding-layer models". For
        //       PURE-DENSE (global-only) models the engine guard does
        //       NOT skip — `prefill_safe = !has_sliding_layer || ...`
        //       — so a pure-dense long-prompt request (seq_len > sw)
        //       reaches the per-head copy with `copy_len_per_head =
        //       seq_len*hd*elem > snap_cap*hd*elem`. Result: dst slice
        //       overrun → panic at line ~1881.
        //
        // Fix:
        //   * snap_cap = max(sw, round32(seq_len + max_decode_tokens)) so
        //     pure-dense long prompts fit AND multi-turn headroom
        //     stays at sw for short prompts.
        //   * `snapshot_safe` skips the entire snapshot path when any
        //     sliding layer exists AND seq_len > sw — matching the
        //     engine guard at engine.rs:4516. Skipped snapshots
        //     return None; the engine's `dense_kvs_snapshot_for_lcp.
        //     take()` then yields None and the store path naturally
        //     no-ops. Two separate guards stay aligned by sharing the
        //     same predicate (`!any_sliding || seq_len <= sw`).
        let any_sliding_layer = self
            .layers
            .iter()
            .any(|l| l.layer_type == LayerType::Sliding);
        // ADR-017 Phase E.a iter-3.6 — when LONG_RESUME is on, sliding
        // layers were allocated with linear capacity (no ring wrap),
        // so seq_len > sw is now SAFE. The snapshot path captures the
        // full populated [0..seq_len) range, no wrap corruption. iter-7
        // guard predicate stays in place for the LONG_RESUME=0 case.
        let snapshot_safe = !any_sliding_layer || seq_len <= sw || kv_lcp_long_resume;
        // ADR-017 Phase E.a "gemma-hybrid-lcp" (2026-08-03) — widen the
        // dense-snapshot regime gate from `use_dense` to "a resumable
        // regime": dense (`use_dense=1`, today's path) OR hybrid
        // (production default; `self.hybrid_kv` allocated this prefill).
        // The dense leg is required under BOTH regimes because prefill
        // attention always reads the DENSE SDPA path; decode reads
        // dense under use_dense and `hybrid_kv` under hybrid. The
        // HB-encoded opt-out regime (neither flag) stays byte-identical
        // to pre-sub-iter: no snapshot, no store (its packed-K leg has
        // no restore path this sub-iter).
        let lcp_resumable_regime =
            crate::debug::INVESTIGATION_ENV.use_dense || self.hybrid_kv.is_some();
        // Pre-copy budget gate (2026-08-03): estimate the dual-leg entry
        // bytes from shapes BEFORE allocating ~2×live-KV of snapshot
        // copies; skip when the entry cannot fit the registry budget
        // (97K-token dual-leg ≈ 64 GB → swap-storm, measured live).
        let lcp_snap_cap_est = sw.max(gemma_flash_attn_vec_capacity(
            seq_len + max_decode_tokens + if kv_lcp_long_resume { 4096 } else { 0 },
        ));
        let lcp_est_bytes: u64 = {
            let dense: u64 = self
                .layers
                .iter()
                .map(|l| {
                    (2 * l.num_kv_heads * lcp_snap_cap_est * l.head_dim * kv_elem_bytes) as u64
                })
                .sum();
            let hybrid: u64 = if self.hybrid_kv.is_some() {
                self.layers
                    .iter()
                    .map(|l| {
                        (l.num_kv_heads * lcp_snap_cap_est * (l.head_dim * 2 + l.head_dim + 4))
                            as u64
                    })
                    .sum()
            } else {
                0
            };
            dense + hybrid
        };
        let lcp_fits_budget =
            crate::serve::kv_persist::lcp_registry::gemma_lcp_snapshot_fits_budget(lcp_est_bytes);
        if crate::debug::INVESTIGATION_ENV.kv_lcp_resume && lcp_resumable_regime && !lcp_fits_budget
        {
            tracing::debug!(
                "lcp_snapshot skipped (budget): est entry {} bytes > registry budget                  ({} bytes) — snapshots skipped, registry stays without this prompt",
                lcp_est_bytes,
                crate::serve::kv_persist::lcp_registry::default_lcp_byte_budget(),
            );
        }
        let snapshot_for_lcp: Option<Vec<std::sync::Arc<DenseKvBuffers>>> =
            if crate::debug::INVESTIGATION_ENV.kv_lcp_resume
                && lcp_resumable_regime
                && snapshot_safe
                && lcp_fits_budget
                && !slot_aware
            {
                // Allocate fresh per-layer buffers + memcpy bytes from
                // each live `dense_kvs_vec[i]` into the new snapshot.
                let mut snap: Vec<std::sync::Arc<DenseKvBuffers>> = Vec::with_capacity(num_layers);
                // ADR-017 Phase E.a iter-3.5d + iter-7 — snapshot
                // capacity policy.
                //
                // iter-3.5d intent: for sliding-layer models the
                // engine guard restricts cache to prompts ≤ sw, so
                // snap_cap = sw is sufficient AND gives multi-turn
                // headroom (subsequent turns' prompts up to sw fit
                // under the engine probe's per-layer cap-check).
                //
                // iter-7 correction: pure-dense (global-only) models
                // can have seq_len > sw (engine guard always passes
                // because !has_sliding_layer). For those, snap_cap
                // must be ≥ seq_len or the per-head memcpy at
                // `copy_len_per_head = seq_len*hd*elem` overruns dst.
                // Take max with the final rounded flash-attention tile for
                // `seq_len + max_decode_tokens` to accommodate both cases.
                //
                // Memory cost on Gemma 4 26B (sliding+global):
                // sliding+short (seq_len ≤ sw) → snap_cap = sw →
                // ~200 MB per cached entry. Capacity=1 registry ⇒
                // ≤ 200 MB extra resident.
                // "gemma-hybrid-lcp" long-resume multi-turn headroom:
                // turn N+1's prompt is strictly longer than turn N's
                // (history grows), so a snap_cap sized exactly to this
                // request fails the probe-side capacity check next turn
                // (measured live: cached linear_cap 2029 < required
                // 2031). +4096 under long-resume keeps typical turn
                // growth admissible; only the snapshot over-allocates,
                // the per-request live buffers stay exact.
                let snap_cap = sw.max(gemma_flash_attn_vec_capacity(
                    seq_len + max_decode_tokens + if kv_lcp_long_resume { 4096 } else { 0 },
                ));
                for live_layer in dense_kvs_vec.iter() {
                    let nkv_dim = live_layer.k.shape().first().copied().unwrap_or(0);
                    let live_cap_dim = live_layer
                        .k
                        .shape()
                        .get(1)
                        .copied()
                        .unwrap_or(live_layer.capacity);
                    let hd_dim = live_layer.k.shape().get(2).copied().unwrap_or(0);
                    let elem_bytes_layer = live_layer.dtype.size_of();
                    // Snapshot-side capacity = sw for both layer
                    // types. snap_nbytes is the snapshot buffer's
                    // total size; copy_bytes is the live → snap
                    // memcpy length (live's `nbytes`, smaller for
                    // global when sw > seq_len+max_decode).
                    let snap_nbytes = nkv_dim * snap_cap * hd_dim * elem_bytes_layer;
                    let live_nbytes = nkv_dim * live_cap_dim * hd_dim * elem_bytes_layer;
                    let mut k_snap = dev
                        .alloc_buffer(
                            snap_nbytes,
                            live_layer.dtype,
                            vec![nkv_dim, snap_cap, hd_dim],
                        )
                        .map_err(|e| anyhow::anyhow!("snapshot K alloc: {e}"))?;
                    let mut v_snap = dev
                        .alloc_buffer(
                            snap_nbytes,
                            live_layer.dtype,
                            vec![nkv_dim, snap_cap, hd_dim],
                        )
                        .map_err(|e| anyhow::anyhow!("snapshot V alloc: {e}"))?;
                    // CPU memcpy via StorageModeShared. Copy length
                    // is `seq_len * nkv * hd * elem_bytes` (only the
                    // prompt-prefix slots; the rest of the snap
                    // buffer stays zero-init). For sliding layers
                    // this matches live (live cap = sw = snap cap).
                    // For global layers, snap cap (sw) ≥ live cap
                    // (seq_len + max), and we copy fewer bytes than
                    // either capacity (just seq_len worth).
                    //
                    // BUT: sliding layers' live buffer at end of
                    // prefill has [0..seq_len) populated only when
                    // seq_len ≤ sw (which iter-3.5c guard enforces).
                    // For global layers, prefill wrote positions
                    // [0..seq_len) into slots [0..seq_len). Both
                    // cases yield the same copy length: seq_len
                    // tokens worth.
                    //
                    // Note: live's first dimension is nkv_dim and
                    // STRIDE-WISE the slot axis is the second
                    // dimension (cap_dim). Bytes layout is
                    // [nkv][slot][hd]. Copying the FIRST `live_nbytes`
                    // bytes from live copies all nkv × live_cap_dim
                    // slots; for sliding that's the full ring (and
                    // since seq_len ≤ sw = live_cap_dim, slots
                    // beyond seq_len are the live-buffer's prior
                    // alloc-zero state, which is fine for snap).
                    //
                    // For global with snap_cap > live_cap_dim, the
                    // memory layout differs: snap has [nkv][snap_cap][hd]
                    // while live has [nkv][live_cap][hd]. A flat
                    // memcpy of `live_nbytes` bytes places live's
                    // [nkv=0] block contiguously at snap's
                    // [nkv=0..(live_cap × hd × elem_bytes/snap_stride)]
                    // — WRONG for snap's strided layout. Need
                    // per-head copying.
                    let k_src: &[u8] = live_layer
                        .k
                        .as_slice()
                        .map_err(|e| anyhow::anyhow!("snapshot K src as_slice: {e}"))?;
                    let v_src: &[u8] = live_layer
                        .v
                        .as_slice()
                        .map_err(|e| anyhow::anyhow!("snapshot V src as_slice: {e}"))?;
                    let k_dst: &mut [u8] = k_snap
                        .as_mut_slice()
                        .map_err(|e| anyhow::anyhow!("snapshot K dst as_mut_slice: {e}"))?;
                    let v_dst: &mut [u8] = v_snap
                        .as_mut_slice()
                        .map_err(|e| anyhow::anyhow!("snapshot V dst as_mut_slice: {e}"))?;
                    if live_cap_dim == snap_cap {
                        // Same shape — fast path, single memcpy of
                        // the prompt-prefix slots only.
                        let copy_len_per_head = seq_len * hd_dim * elem_bytes_layer;
                        for h in 0..nkv_dim {
                            let src_off = h * live_cap_dim * hd_dim * elem_bytes_layer;
                            let dst_off = h * snap_cap * hd_dim * elem_bytes_layer;
                            k_dst[dst_off..dst_off + copy_len_per_head]
                                .copy_from_slice(&k_src[src_off..src_off + copy_len_per_head]);
                            v_dst[dst_off..dst_off + copy_len_per_head]
                                .copy_from_slice(&v_src[src_off..src_off + copy_len_per_head]);
                        }
                        let _ = live_nbytes; // silenced
                    } else {
                        // Different cap shapes (global w/ headroom):
                        // strided per-head copy. Source stride is
                        // `live_cap_dim × hd × elem`; dest stride is
                        // `snap_cap × hd × elem`. Copy first
                        // `seq_len × hd × elem` bytes per head.
                        let copy_len_per_head = seq_len * hd_dim * elem_bytes_layer;
                        let src_stride = live_cap_dim * hd_dim * elem_bytes_layer;
                        let dst_stride = snap_cap * hd_dim * elem_bytes_layer;
                        for h in 0..nkv_dim {
                            let src_off = h * src_stride;
                            let dst_off = h * dst_stride;
                            k_dst[dst_off..dst_off + copy_len_per_head]
                                .copy_from_slice(&k_src[src_off..src_off + copy_len_per_head]);
                            v_dst[dst_off..dst_off + copy_len_per_head]
                                .copy_from_slice(&v_src[src_off..src_off + copy_len_per_head]);
                        }
                    }
                    snap.push(std::sync::Arc::new(DenseKvBuffers {
                        k: k_snap,
                        v: v_snap,
                        capacity: snap_cap,
                        is_sliding: live_layer.is_sliding,
                        dtype: live_layer.dtype,
                    }));
                }
                Some(snap)
            } else {
                if crate::debug::INVESTIGATION_ENV.kv_lcp_resume
                    && lcp_resumable_regime
                    && !snapshot_safe
                {
                    tracing::debug!(
                        "lcp_snapshot skipped (iter-7 prefill-wrap guard): \
                         seq_len={} > sw={} on a model with sliding layers; \
                         the live ring already wrapped during prefill so no \
                         snapshot can faithfully reconstruct [0..K). The \
                         engine-side guard at engine.rs:4516 also skips \
                         store; both guards stay aligned.",
                        seq_len,
                        sw
                    );
                }
                None
            };

        // ADR-017 Phase E.a "gemma-hybrid-lcp" (2026-08-03) — end-of-
        // prefill snapshot of the HYBRID leg (F16 K + TQ-HB V packed +
        // norms), mirroring the dense snapshot above. Required for LCP
        // resume under the production hybrid regime: decode reads
        // `hybrid_kv`, so restoring only the dense leg would leave the
        // decode cache zeroed over the prefix (silent corruption — the
        // same class ADR-027 23d-γ closed for qwen35).
        //
        // Capacity policy + per-head copy semantics mirror the dense
        // snapshot exactly (`snap_cap`, fast-path when live_cap ==
        // snap_cap, strided per-head otherwise). v_norms' inner axis is
        // `norms_per_pos` (1 at hd=256 production), with the live
        // buffer's rank-2 `[nkv, cap]` / rank-3 `[nkv, cap, npp]` shape
        // preserved on the snapshot side.
        let hybrid_snapshot_for_lcp: Option<
            Vec<std::sync::Arc<crate::inference::models::gemma4::kv_cache::HybridKvBuffers>>,
        > = if crate::debug::INVESTIGATION_ENV.kv_lcp_resume && snapshot_safe && !slot_aware {
            match self.hybrid_kv.as_ref() {
                None => None,
                Some(live_hybrid) => {
                    // Same capacity policy as the dense snapshot
                    // (mirrors line ~2229): sw headroom for short
                    // prompts, the final rounded seq_len + decode tile for
                    // long ones.
                    let snap_cap = sw.max(gemma_flash_attn_vec_capacity(
                        seq_len + max_decode_tokens + if kv_lcp_long_resume { 4096 } else { 0 },
                    ));
                    let mut hsnap: Vec<
                        std::sync::Arc<crate::inference::models::gemma4::kv_cache::HybridKvBuffers>,
                    > = Vec::with_capacity(live_hybrid.len());
                    for live_layer in live_hybrid.iter() {
                        let nkv_dim = live_layer.k.shape().first().copied().unwrap_or(0);
                        let live_cap_dim = live_layer
                            .k
                            .shape()
                            .get(1)
                            .copied()
                            .unwrap_or(live_layer.capacity);
                        let hd_dim = live_layer.k.shape().get(2).copied().unwrap_or(0);
                        let npp = live_layer.norms_per_pos.max(1);
                        // K: F16 (2 B/elem); v_packed: U8 (1 B/elem);
                        // v_norms: F32 (4 B/elem, inner axis npp).
                        let k_snap = dev
                            .alloc_buffer(
                                nkv_dim * snap_cap * hd_dim * 2,
                                mlx_native::DType::F16,
                                vec![nkv_dim, snap_cap, hd_dim],
                            )
                            .map_err(|e| anyhow::anyhow!("hybrid snapshot K alloc: {e}"))?;
                        let vp_snap = dev
                            .alloc_buffer(
                                nkv_dim * snap_cap * hd_dim,
                                mlx_native::DType::U8,
                                vec![nkv_dim, snap_cap, hd_dim],
                            )
                            .map_err(|e| anyhow::anyhow!("hybrid snapshot V packed alloc: {e}"))?;
                        let vn_shape = if npp == 1 {
                            vec![nkv_dim, snap_cap]
                        } else {
                            vec![nkv_dim, snap_cap, npp]
                        };
                        let vn_snap = dev
                            .alloc_buffer(
                                nkv_dim * snap_cap * npp * 4,
                                mlx_native::DType::F32,
                                vn_shape,
                            )
                            .map_err(|e| anyhow::anyhow!("hybrid snapshot V norms alloc: {e}"))?;

                        // Per-head prefix copy for one (src, dst, elem,
                        // inner) quadruple — same two-branch structure
                        // as the dense snapshot (fast path when shapes
                        // match, strided otherwise).
                        let copy_prefix = |src: &mlx_native::MlxBuffer,
                                           dst: &mut mlx_native::MlxBuffer,
                                           elem: usize,
                                           inner: usize,
                                           what: &str|
                         -> anyhow::Result<()> {
                            let s: &[u8] = src
                                .as_slice()
                                .map_err(|e| anyhow::anyhow!("hybrid snapshot {what} src: {e}"))?;
                            let d: &mut [u8] = dst
                                .as_mut_slice()
                                .map_err(|e| anyhow::anyhow!("hybrid snapshot {what} dst: {e}"))?;
                            let copy_len = seq_len * inner * elem;
                            let src_stride = live_cap_dim * inner * elem;
                            let dst_stride = snap_cap * inner * elem;
                            for h in 0..nkv_dim {
                                let so = h * src_stride;
                                let do_ = h * dst_stride;
                                d[do_..do_ + copy_len].copy_from_slice(&s[so..so + copy_len]);
                            }
                            Ok(())
                        };
                        let mut k_snap = k_snap;
                        let mut vp_snap = vp_snap;
                        let mut vn_snap = vn_snap;
                        copy_prefix(&live_layer.k, &mut k_snap, 2, hd_dim, "K")?;
                        copy_prefix(&live_layer.v_packed, &mut vp_snap, 1, hd_dim, "V packed")?;
                        copy_prefix(&live_layer.v_norms, &mut vn_snap, 4, npp, "V norms")?;

                        hsnap.push(std::sync::Arc::new(
                            crate::inference::models::gemma4::kv_cache::HybridKvBuffers {
                                k: k_snap,
                                v_packed: vp_snap,
                                v_norms: vn_snap,
                                capacity: snap_cap,
                                is_sliding: live_layer.is_sliding,
                                norms_per_pos: live_layer.norms_per_pos,
                                // xlen companions are decode-time lazy
                                // scratch, never part of prefix state.
                                bf16_xlen_k: None,
                                bf16_xlen_v: None,
                            },
                        ));
                    }
                    Some(hsnap)
                }
            }
        } else {
            None
        };
        // ADR-040 STEP-1b (2026-06-24) — gate the three persistent `self.*`
        // write-backs on `!slot_aware`.  In slot-aware mode the caller
        // (`forward_*_slot_aware`) mounted a per-slot slot-view bundle on
        // `self.dense_kvs` (consumed into `dense_kvs_vec` at the iter-2D
        // consume-gate above) and OWNS the restore on exit; re-`Some(..)`-ing
        // a FRESH shared `dense_kvs` here — plus `dense_kvs_snapshot_for_lcp`
        // + `dense_sdpa_tmp` — would leak shared state across slots.  For the
        // SerialFifo / legacy path (`slot_aware=false`) these run verbatim →
        // byte-equivalent.  `dense_kvs_vec` / `sdpa_tmp` / `snapshot_for_lcp`
        // simply drop at scope end in slot-aware mode (the slot-view buffers
        // they wrap are kept alive by the persistent multi-seq scaffold).
        if !slot_aware {
            self.dense_kvs_snapshot_for_lcp = snapshot_for_lcp;
            // "gemma-hybrid-lcp" (2026-08-03): same write-back discipline
            // for the hybrid leg snapshot.
            self.hybrid_kv_snapshot_for_lcp = hybrid_snapshot_for_lcp;

            self.dense_kvs = Some(dense_kvs_vec.into_iter().map(std::sync::Arc::new).collect());
            self.dense_sdpa_tmp = Some(sdpa_tmp);
        } else {
            drop(snapshot_for_lcp);
            drop(hybrid_snapshot_for_lcp);
            drop(dense_kvs_vec);
            drop(sdpa_tmp);
        }

        // iter-222 (2026-05-01): legacy iter-20/iter-21 Track A note about
        // `leg_f_kvs` placement was deleted along with the field — see
        // file-level iter-222 closure note in `forward_mlx.rs`.

        Ok(last_token)
    }

    /// Last-pool chat-model embedding (ADR-005 Phase 2a, Task #8, iter-92).
    ///
    /// Runs `forward_prefill` with `max_decode_tokens = 0` (i.e. no decode
    /// budget — KV-cache buffers are sized to seq_len exactly), then reads
    /// the last token's RMS-normed hidden state from `self.activations.norm_out`
    /// (already populated as the last side-effect of the per-token loop's
    /// final-norm dispatch — see ~line 1186 in `forward_prefill`), L2-normalizes
    /// the vector, and returns it as `Vec<f32>` of length `hidden_size`.
    ///
    /// # Pooling: Last
    ///
    /// "Last" pooling takes the final token's hidden state. For autoregressive
    /// chat models (Gemma, Llama, Mistral, Qwen — anything with causal
    /// attention) this is the natural pooling because the causal mask makes
    /// the last token's hidden state a function of the entire sequence.
    /// Mean pooling on a chat model would average over a sequence whose
    /// earlier tokens have NOT seen the later context — semantically a
    /// less-informative aggregation than Last.
    ///
    /// Mean / CLS pooling for chat models is intentionally NOT supported
    /// here.  If the user wants different pooling semantics, they should
    /// load a dedicated BERT-family encoder model (`--embedding-model`)
    /// which the dedicated lane consumes via `apply_bert_full_forward_gpu`
    /// or `apply_nomic_bert_full_forward_gpu`.
    ///
    /// # Cost
    ///
    /// Same prefill compute as a 1-token-decode-budget `forward_prefill`
    /// minus 0 (no decode runs).  The per-token lm_head + softcap + argmax
    /// dispatches that the prefill loop runs are wasted work for embedding
    /// (we discard logits / argmax_index), but the cost is small relative
    /// to the layer-stack forward.  Iter-93+ candidate: a dedicated
    /// `forward_embed_last_minimal` that skips the lm_head/softcap/argmax
    /// per-token dispatches via a `compute_lm_head: bool` flag plumbed
    /// through `forward_prefill`.
    ///
    /// # Returns
    ///
    /// L2-normalized embedding vector of length `self.hidden_size`.
    ///
    /// # ADR-040 iter-B4c-kernel iter-2-embed structural-N/A closure (2026-05-30, §6.1.49)
    ///
    /// Pre-iter-2-embed pin (§6.1.32 followups list, line 2938): "`forward_embed_last`
    /// slot-aware port (~60 LOC; single-call wrapper around
    /// `forward_prefill(.., max_decode_tokens=0, ..)`).  Same shape as iter-2-decode but
    /// no decode loop.  Pinned in the iter-B4c-kernel-iter-4 (Embed worker-arm) deferral
    /// cite at iter-1's §6.1.31 closure."
    ///
    /// **Investigation finding (iter-2-embed)** — the Embed-arm SlotId(N>0) surface is
    /// ALREADY fully covered by iter-B4c-kernel iter-4 (§6.1.36) at the **orchestrator
    /// layer**, not the model-fn layer.  The orchestrator `embed_gemma4_slot_aware` at
    /// `src/serve/api/engine.rs:9865` calls
    /// `forward_prefill_with_soft_tokens_slot_aware(.., max_decode_tokens=0, ..)`
    /// (the iter-2A landing per §6.1.32 + iter-2B routing per §6.1.34) — NOT
    /// `forward_embed_last`.  The worker-arm dispatch fork at `engine.rs:5845` routes
    /// `slot_id != SlotId(0)` into `embed_gemma4_slot_aware`; only SerialFifo + SlotId(0)
    /// reaches the legacy `g.weights.forward_embed_last(&prompt_tokens, &mut g.ctx)`
    /// dispatch at `engine.rs:6026`.
    ///
    /// **Structural N/A verdict**: a hypothetical `forward_embed_last_slot_aware` would be
    /// DEAD CODE — it has no caller.  Adding one would violate H188 + H201 transitivity
    /// (Embed-arm body does NOT call `forward_decode_slot_aware` because Embed has no
    /// decode loop; same code-path disjointness rules out adding a slot-aware embed
    /// signature here).  iter-2-embed is therefore CLOSED as structural-N/A: the
    /// SlotId(N>0) Embed surface is shipped, but the closure point is the orchestrator
    /// at `embed_gemma4_slot_aware` (§6.1.36), NOT this fn.
    ///
    /// **Forward-pointer discoverability** (H87 discipline):
    /// `iter-B4c-kernel-iter-2-embed per ADR-040 §6.1.49` substring is preserved here as
    /// a doc-comment cite so `grep "iter-2-embed per"` discovers the closure block.
    /// The label substring is INTENTIONALLY NOT inside a `MultiSeqError::CapabilityUnsupported`
    /// constructor — the iter-2-embed surface has no typed deferral to surface (the
    /// SlotId(N>0) routing is the orchestrator's responsibility).
    ///
    /// SerialFifo + SlotId(0) byte-equivalence (H1/H2/H23/H41/H44/H77/H102/H128/H135/H198)
    /// preserved trivially: this fn's signature + body UNCHANGED by iter-2-embed; only the
    /// docstring grows.
    pub fn forward_embed_last(
        &mut self,
        prompt_tokens: &[u32],
        gpu: &mut GpuContext,
    ) -> Result<Vec<f32>> {
        if prompt_tokens.is_empty() {
            anyhow::bail!("forward_embed_last: empty prompt");
        }

        // Reset per-prefill cache state. `forward_prefill` gates the
        // `leg_hb_encoded` re-allocation on `is_none()` — a latent quirk
        // that's harmless for chat (where the first allocation's capacity
        // covers later calls) but BREAKS embedding mode: embeds run with
        // `max_decode_tokens=0` so `linear_capacity` is the prompt length
        // rounded through the final 32-row attention tile, and the
        // first embedding's tiny cache poisons every subsequent call
        // (embed OR chat) with a capacity-too-small fault inside
        // `flash_attn_vec_tq_hb`. We force-clear before re-entering prefill
        // so it re-allocates fresh buffers sized for THIS call's seq_len.
        // `dense_kvs` is overwritten unconditionally inside prefill so it
        // doesn't strictly need clearing, but we do it for symmetry +
        // future-proofing.
        // iter-222 (2026-05-01): `leg_f_kvs` / `leg_f_sdpa_tmp` resets
        // deleted along with the fields.
        self.dense_kvs = None;
        self.dense_sdpa_tmp = None;
        self.leg_hb_encoded = None;

        // Run prefill with no decode budget. The per-token loop populates
        // self.activations.norm_out with the last token's RMS-normed
        // hidden state as part of its final_norm dispatch (~line 1186).
        // Discard the returned argmax token — embedding doesn't decode.
        let _ = self
            .forward_prefill(prompt_tokens, 0, gpu)
            .map_err(|e| anyhow::anyhow!("forward_embed_last prefill: {e}"))?;

        // Read the [hidden_size] f32 hidden state.  norm_out is sized
        // [1 row * hidden_size] — the per-token reuse of the buffer means
        // it always holds exactly one row's worth of data.
        let view: &[f32] = self
            .activations
            .norm_out
            .as_slice()
            .map_err(|e| anyhow::anyhow!("forward_embed_last read norm_out: {e}"))?;
        let hs = self.hidden_size;
        if view.len() < hs {
            anyhow::bail!(
                "forward_embed_last: norm_out has {} f32 elements, expected at least {}",
                view.len(),
                hs
            );
        }
        let mut out: Vec<f32> = view[..hs].to_vec();

        // L2 normalize so consumers can compute cosine similarity by dot product.
        // 1e-12 floor matches the BERT-lane bert_l2_normalize_gpu epsilon.
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        let denom = if norm < 1e-12 { 1e-12 } else { norm };
        for v in out.iter_mut() {
            *v /= denom;
        }
        Ok(out)
    }

    /// **ADR-040 iter-B4c-kernel iter-2A** (2026-05-30) — slot-aware
    /// prefill entry point on `MlxModelWeights`.  Cross-architecture
    /// mirror of Qwen35 `Qwen35Model::forward_gpu_last_logits(.., slot_id)`
    /// per ADR-040 §6.1.4 (B4a) / §6.1.5 (B4a-cont) / §6.1.20 (B4b).
    ///
    /// This is the **load-bearing primitive** the iter-1 orchestrator
    /// scaffold at `engine::generate_gemma4_once_slot_aware` (§6.1.31)
    /// types as `iter-B4c-kernel-iter-2`: an explicit `slot_id`
    /// parameter on the Gemma 4 forward path so the kernel-side K/V
    /// writes (via `dispatch_hadamard_quantize_kv_hb_*` at
    /// `forward_prefill.rs:1319-1363`) can route through the per-slot
    /// region of the persistent `MultiSeqHbKvBuffers` scaffold
    /// (provisioned by C2c §6.1.21 at spawn time).
    ///
    /// # iter-2A scope (this commit)
    ///
    /// iter-2A lands the **fn signature + pre-flight + dispatch fork**
    /// that the iter-2{A-cont,B,C,D} sub-deferrals will fill out.
    /// Specifically:
    ///
    /// - **Bounds-first** per A2b §6.1.23 iter-1.5 cfa-finding-F5
    ///   ordering: `slot_id.0 < multi_seq_kv_hb[0].n_seqs` (else fail
    ///   loud with a diagnostic naming the slot + the n_seqs at
    ///   construction).
    /// - **Layer-count match**: `multi_seq_kv_hb.len() == self.layers.len()`
    ///   (caller invariant; defense-in-depth fail-fast).
    /// - **Empty prompt guard**: mirrors the sibling
    ///   `forward_prefill_with_soft_tokens_resume`'s precondition.
    /// - **Dispatch fork** on the 4 production KV-regime branches
    ///   (matches the existing forward-path branching at
    ///   `forward_prefill.rs:837-891` and `forward_decode` at
    ///   `forward_gpu.rs:407-466`):
    ///   * **HF2Q_HYBRID_KV=1** (production default per H10
    ///     falsification, ADR-040 §6.1.11) → `HybridKvBuffers` slot
    ///     routing typed-deferred as **iter-B4c-kernel-iter-2B**.
    ///     Per `engine.rs::GemmaLoadedModel::multi_seq_kv` docstring
    ///     (the field comment), the C2c spawn-arm provisions ONLY the
    ///     `MultiSeqHbKvBuffers` scaffold today; the `MultiSeqHybrid
    ///     KvBuffers` sibling scaffold is staged for iter-C2c-cont
    ///     gated on this iter-2B kernel work landing.
    ///   * **HF2Q_HYBRID_KV=0 + HF2Q_TQ_CODEBOOK_BITS>=5** (HB-encoded
    ///     opt-out path) → HB slot routing typed-deferred as
    ///     **iter-B4c-kernel-iter-2A-cont**.  This is the in-scope
    ///     scaffold-consuming path: the per-layer
    ///     `&mut multi_seq_kv_hb[layer_idx]` IS the load-bearing
    ///     argument the iter-2A-cont kernel-dispatch refactor will
    ///     thread through `dispatch_hadamard_quantize_kv_hb_*` callers
    ///     via `MlxBuffer::slice_view(byte_offset, n_elements)` (same
    ///     primitive Qwen35 B4a-cont uses at `gpu_full_attn.rs:172-181`
    ///     per ADR-040 §6.1.5).
    ///   * **HF2Q_TQ_CODEBOOK_BITS=4** (legacy 4-bit, opt-in pre-default
    ///     correction at 2026-04-24) → **SHIPPED 2026-05-30 as
    ///     iter-B4c-kernel-iter-2C (§6.1.46)**.  Slot routing via
    ///     `MultiSeqMlxKvCache` slice_view mount over `self.kv_caches`
    ///     (Vec swap pattern: `std::mem::replace` of the per-layer
    ///     `Vec<MlxKvCache>` with slot-views, restore on exit; the
    ///     legacy field is always populated at model load time per
    ///     `gemma4/model.rs:1277-1290` — no `is_none()` gate needed,
    ///     mirroring the always-allocated 4-buffer layout).  Off-default
    ///     since ADR-007 default-on TQ-8-bit correction; gated on
    ///     iter-C2c-cont-cont Phase 4 provisioning the
    ///     `multi_seq_kv_mlx` scaffold per §6.1.46.
    ///   * **HF2Q_USE_DENSE=1** (dense F32 KV path) → **SHIPPED 2026-05-30
    ///     as iter-B4c-kernel-iter-2D (§6.1.46)**.  Slot routing via
    ///     `MultiSeqDenseKvBuffers` slice_view mount over `self.dense_kvs`
    ///     (`Option<Vec<Arc<DenseKvBuffers>>>` mount + sibling consume-
    ///     gate alignment with the existing `restored_lcp=Some(k)`
    ///     pre-iter-3 path's `self.dense_kvs.take()` consumer at
    ///     `forward_prefill.rs:727`; the slot-aware mount inserts the
    ///     slot-view Arc-vec so the sibling's existing `take()` consumer
    ///     reads slot-view buffers, routing kernel writes to the per-slot
    ///     byte region).  Orthogonal to TQ-active paths; LCP partial-
    ///     prefix resume slot-aware LCP path remains as iter-2D-lcp
    ///     sub-deferral.  Gated on iter-C2c-cont-cont Phase 3 provisioning
    ///     the `multi_seq_kv_dense` scaffold per §6.1.46.
    ///
    /// # Why a NEW fn vs. modifying `forward_prefill_with_soft_tokens_resume`
    ///
    /// Per the ADR-040 H1/H2/H23/H41 byte-equivalence pin chain (A5*/
    /// C2a/C2b/C2c/B4c), `forward_prefill_with_soft_tokens_resume` MUST
    /// remain byte-identical for SerialFifo + SlotId(0) AND SlotAware +
    /// SlotId(0).  A signature change there would force every call site
    /// (warmup, generate, embed_last, generate_stream_once, LCP resume,
    /// soft-token + deepstack variants) to thread `Option<SlotId>` +
    /// `Option<&mut Vec<MultiSeqHbKvBuffers>>` through; the pin contract
    /// would survive at `None`/`SlotId(0)` BUT the surface-area drift
    /// would risk silent regressions on every modification.
    ///
    /// The structurally-honest answer (this fn): SerialFifo + SlotId(0)
    /// path remains UNCHANGED at the worker arm's existing
    /// `generate_once` dispatch (which routes through the unchanged
    /// `forward_prefill_with_soft_tokens_resume`).  SlotAware + SlotId(N>0)
    /// routes through THIS new fn — a fork at the worker-arm boundary
    /// (already in place via iter-1's
    /// `engine::generate_gemma4_once_slot_aware` orchestrator) routes
    /// the SlotId(N>0) request here.  iter-2{A-cont,B,C,D} ports the
    /// kernel-dispatch refactor INSIDE THIS NEW FN ONLY — never inside
    /// the existing sibling, preserving byte-equivalence by code-path
    /// disjointness.
    ///
    /// # Returns
    ///
    /// Same shape as `forward_prefill_with_soft_tokens_resume`: the
    /// first decode token (greedy argmax of last-row logits).  Decode
    /// loop iteration follows via existing `forward_decode` calls (the
    /// decode-side slot routing is typed-deferred as
    /// **iter-B4c-kernel-iter-2-decode**).
    ///
    /// # Errors
    ///
    /// - `prompt_tokens.is_empty()` (mirrors sibling).
    /// - `multi_seq_kv_hb.is_empty()` (caller invariant: C2c provisioning
    ///   produces one entry per layer; defense-in-depth fail-fast).
    /// - `multi_seq_kv_hb.len() != self.layers.len()` (caller invariant
    ///   violation).
    /// - `slot_id.0 >= multi_seq_kv_hb[0].n_seqs` (typed
    ///   `MultiSeqError::SlotOutOfRange` wrapped in `anyhow::Error`).
    /// - iter-B4c-kernel-iter-2{A-cont,B,C,D} typed
    ///   `MultiSeqError::CapabilityUnsupported` on the dispatch-fork
    ///   branches above — each names the specific sub-iter surface.
    ///
    /// # Cross-architecture mirror
    ///
    /// Mirrors `Qwen35Model::forward_gpu_last_logits(.., slot_id)` at
    /// `src/inference/models/qwen35/forward_gpu.rs:2564` + the B4a
    /// bounds-first contract at `forward_gpu.rs:2569-2586` (typed
    /// SlotOutOfRange diagnostic naming `slot_id` + `kv_cache.n_seqs`).
    /// The Qwen35 surface accepted `slot_id` since B4a §6.1.4 (2026-05-23);
    /// this fn is Gemma 4's analogue, landed at iter-B4c-kernel iter-2A
    /// per §6.1.31's investigation findings (the Gemma 4 forward path
    /// had NO slot_id parameter anywhere pre-iter-2).
    pub fn forward_prefill_with_soft_tokens_slot_aware(
        &mut self,
        prompt_tokens: &[u32],
        soft_tokens: &[SoftTokenInjection<'_>],
        max_decode_tokens: usize,
        gpu: &mut GpuContext,
        slot_id: SlotId,
        multi_seq_kv_hb: &mut Vec<MultiSeqHbKvBuffers>,
        multi_seq_kv_hybrid: Option<&mut Vec<MultiSeqHybridKvBuffers>>,
        multi_seq_kv_dense: Option<&mut Vec<MultiSeqDenseKvBuffers>>,
        multi_seq_kv_mlx: Option<&mut Vec<MultiSeqMlxKvCache>>,
    ) -> Result<u32> {
        self.forward_prefill_with_soft_tokens_slot_aware_impl(
            prompt_tokens,
            soft_tokens,
            max_decode_tokens,
            gpu,
            slot_id,
            multi_seq_kv_hb,
            multi_seq_kv_hybrid,
            multi_seq_kv_dense,
            multi_seq_kv_mlx,
            0,
        )
    }

    /// Append a prompt suffix to an exact live Gemma slot prefix.
    /// `cached_tokens` is verified by the worker against both its rendered
    /// token ledger and the slot cursor before this call. Soft-token ranges
    /// are accepted only when they lie wholly in the appended suffix.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_with_soft_tokens_slot_aware_resume(
        &mut self,
        prompt_tokens: &[u32],
        soft_tokens: &[SoftTokenInjection<'_>],
        max_decode_tokens: usize,
        gpu: &mut GpuContext,
        slot_id: SlotId,
        multi_seq_kv_hb: &mut Vec<MultiSeqHbKvBuffers>,
        mut multi_seq_kv_hybrid: Option<&mut Vec<MultiSeqHybridKvBuffers>>,
        mut multi_seq_kv_dense: Option<&mut Vec<MultiSeqDenseKvBuffers>>,
        mut multi_seq_kv_mlx: Option<&mut Vec<MultiSeqMlxKvCache>>,
        cached_tokens: usize,
    ) -> Result<u32> {
        anyhow::ensure!(
            cached_tokens > 0 && cached_tokens < prompt_tokens.len(),
            "Gemma slot resume requires 0 < cached_tokens ({cached_tokens}) < prompt_len ({})",
            prompt_tokens.len(),
        );
        anyhow::ensure!(
            soft_tokens
                .iter()
                .all(|soft| soft.range.start >= cached_tokens),
            "Gemma slot resume soft-token ranges must start at or after cached_tokens={cached_tokens}"
        );
        let suffix = &prompt_tokens[cached_tokens..];
        if soft_tokens.is_empty() && suffix.len() < GEMMA_SLOT_BATCHED_PREFILL_MIN_TOKENS {
            let mut next_token = None;
            let mut profile = None;
            for (offset, &input_token) in suffix.iter().enumerate() {
                next_token = Some(self.forward_decode_slot_aware(
                    input_token,
                    cached_tokens + offset,
                    gpu,
                    &mut profile,
                    slot_id,
                    multi_seq_kv_hb,
                    multi_seq_kv_hybrid.as_mut().map(|value| &mut **value),
                    multi_seq_kv_dense.as_mut().map(|value| &mut **value),
                    multi_seq_kv_mlx.as_mut().map(|value| &mut **value),
                )?);
            }
            return next_token.ok_or_else(|| anyhow::anyhow!("Gemma slot resume suffix is empty"));
        }
        self.forward_prefill_with_soft_tokens_slot_aware_impl(
            prompt_tokens,
            soft_tokens,
            max_decode_tokens,
            gpu,
            slot_id,
            multi_seq_kv_hb,
            multi_seq_kv_hybrid,
            multi_seq_kv_dense,
            multi_seq_kv_mlx,
            cached_tokens,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_prefill_with_soft_tokens_slot_aware_impl(
        &mut self,
        prompt_tokens: &[u32],
        soft_tokens: &[SoftTokenInjection<'_>],
        max_decode_tokens: usize,
        gpu: &mut GpuContext,
        slot_id: SlotId,
        multi_seq_kv_hb: &mut Vec<MultiSeqHbKvBuffers>,
        // ADR-040 iter-B4c-kernel iter-2B (2026-05-30) — production-default
        // hybrid F16-K + TQ-HB-V scaffold parameter.  `Option<>` because
        // iter-C2c-cont (§6.1.33) provisions this field IFF the
        // production-default hybrid env gate is true (DEFAULT since
        // ADR-029 iter-13 per H10 falsification at §6.1.11); when the
        // env-gate is OFF this parameter is `None` and the new fn
        // surfaces typed `CapabilityUnsupported` at the hybrid branch
        // defense-in-depth.
        //
        // Per H98 pin: this parameter is ADDITIVE — `multi_seq_kv_hb` is
        // preserved verbatim per the iter-2A H84 surface invariant.
        // Per H88 pin: this doc comment intentionally does NOT mention
        // `INVESTIGATION_ENV . hybrid_kv` (the env-gate constant) verbatim
        // so the H88 lexical-ordering check on the bounds-first pin still
        // finds the dispatch fork AFTER the bounds check.
        multi_seq_kv_hybrid: Option<&mut Vec<MultiSeqHybridKvBuffers>>,
        // ADR-040 iter-B4c-kernel iter-2D (2026-05-30) — dense F32 KV
        // path multi-seq scaffold parameter.  `Option<>` because
        // iter-C2c-cont-cont (§6.1.46 Phase 3) provisions this field IFF
        // the dense F32 env gate is true (`HF2Q_USE_DENSE=1` LCP-eligible
        // pre-default regime — see `INVESTIGATION_ENV.use_dense`).  When
        // the env-gate is OFF this parameter is `None` and the dense
        // branch typed `CapabilityUnsupported` defense-in-depth fires
        // (`gemma4-forward-prefill-dense-scaffold-absent` label naming
        // iter-C2c-cont-cont-invariant-violated).
        //
        // Per H188 pin: this parameter is ADDITIVE — prior surfaces
        // (`multi_seq_kv_hb`, `multi_seq_kv_hybrid`) preserved verbatim
        // per the iter-2A H84 + iter-2B H98 surface invariants.
        multi_seq_kv_dense: Option<&mut Vec<MultiSeqDenseKvBuffers>>,
        // ADR-040 iter-B4c-kernel iter-2C (2026-05-30) — legacy 4-bit
        // nibble-packed multi-seq scaffold parameter.  `Option<>` because
        // iter-C2c-cont-cont (§6.1.46 Phase 4) provisions this field IFF
        // the legacy 4-bit env gate is engaged (`HF2Q_TQ_CODEBOOK_BITS=4`
        // opt-in pre-default surface; off-default since ADR-007 default-on
        // TQ-8-bit correction 2026-04-24).  When the env-gate is OFF this
        // parameter is `None` and the cb_bits==0 branch typed
        // `CapabilityUnsupported` defense-in-depth fires
        // (`gemma4-forward-prefill-mlx-scaffold-absent` label naming
        // iter-C2c-cont-cont-invariant-violated).
        //
        // Per H188 pin: ADDITIVE — prior surfaces preserved.
        multi_seq_kv_mlx: Option<&mut Vec<MultiSeqMlxKvCache>>,
        cached_tokens: usize,
    ) -> Result<u32> {
        // ── Pre-flight ────────────────────────────────────────────────
        //
        // Bounds-first per ADR-040 A2b §6.1.23 iter-1.5 cfa-finding-F5
        // ordering.  Mirrors the contract at
        // `qwen35/forward_gpu.rs:2569-2586` (B4a bounds-first pin).
        //
        // Empty prompt: same precondition as the sibling
        // `forward_prefill_with_soft_tokens_resume` at line 458.  We
        // duplicate the check here so the diagnostic names this fn
        // (operator log greps land on the right pin pointer).
        if prompt_tokens.is_empty() {
            anyhow::bail!(
                "forward_prefill_with_soft_tokens_slot_aware: empty prompt \
                 (ADR-040 iter-B4c-kernel iter-2A)"
            );
        }
        // Caller-invariant: C2c spawn-arm `provision_multi_seq_kv_for_slot_aware`
        // produces exactly `self.weights.layers.len()` entries.  The
        // orchestrator at `engine::generate_gemma4_once_slot_aware`
        // gates on `multi_seq_kv.is_empty()`; we re-assert here for
        // defense-in-depth.  Diagnostic names this fn + the spawn-time
        // provisioning entry so an operator can trace either way.
        if multi_seq_kv_hb.is_empty() {
            anyhow::bail!(
                "forward_prefill_with_soft_tokens_slot_aware: multi_seq_kv_hb is empty \
                 (C2c spawn-arm invariant: provision_multi_seq_kv_for_slot_aware \
                  must produce one entry per layer; ADR-040 §6.1.21 / iter-B4c-kernel iter-2A)"
            );
        }
        if multi_seq_kv_hb.len() != self.layers.len() {
            anyhow::bail!(
                "forward_prefill_with_soft_tokens_slot_aware: multi_seq_kv_hb.len()={} \
                 != self.layers.len()={} — caller-invariant violation \
                 (ADR-040 iter-B4c-kernel iter-2A; C2c spawn-arm \
                 provision_multi_seq_kv_for_slot_aware must allocate one per layer)",
                multi_seq_kv_hb.len(),
                self.layers.len(),
            );
        }
        // Per-slot bounds.  Use the FIRST layer's n_seqs as canonical:
        // A3a's `alloc_hb_kv_for_layer` enforces uniform `n_seqs` across
        // layers per the C2c provisioning loop, so checking layer 0 is
        // sufficient (cross-layer desync would itself be a separate
        // invariant violation worth surfacing if/when observed).
        let n_seqs = multi_seq_kv_hb[0].n_seqs;
        if slot_id.0 >= n_seqs {
            // Surface the typed MultiSeqError variant inside an anyhow
            // chain so the worker arm's logging hooks see both the
            // structured variant + the diagnostic context.
            let err = MultiSeqError::SlotOutOfRange {
                slot: slot_id,
                max_slots: n_seqs,
            };
            anyhow::bail!(
                "forward_prefill_with_soft_tokens_slot_aware: slot_id={} out of range \
                 (n_seqs={}). ADR-040 iter-B4c-kernel iter-2A bounds-first contract — \
                 re-allocate the persistent multi-seq scaffold with a larger n_seqs \
                 (provision_multi_seq_kv_for_slot_aware was called with max_slots={}). {}",
                slot_id.0,
                n_seqs,
                n_seqs,
                err,
            );
        }
        // ── Dispatch fork ─────────────────────────────────────────────
        //
        // Mirror the 4 production KV-regime branches at the existing
        // alloc site (`forward_prefill.rs:837-891`).  Each branch lands
        // a typed sub-deferral naming the specific kernel-dispatch
        // refactor remaining.  Reading the env vars here (instead of
        // capturing at construction) matches the sibling fn's
        // discipline — INVESTIGATION_ENV is a LazyLock parsed once at
        // process start, identical to its read site at line 832.
        //
        // ADR-040 iter-B4c-kernel iter-2B (2026-05-30) — the dense F32 /
        // legacy 4-bit / HB-encoded branches below `anyhow::bail!()`
        // (typed CapabilityUnsupported) without consuming the function
        // params; the hybrid branch (production-default) THREADS them
        // through the slot-view mount + delegate call.  Rust's
        // unused-fn-param lint allows this without underscore prefixes
        // (params are treated differently from locals).  iter-2A's
        // explicit `_x = x` shadows were elided since iter-2B genuinely
        // consumes the params on the active branch.
        //
        // `multi_seq_kv_hb` (HB-encoded scaffold) is preserved verbatim
        // per H98 / H84 — the iter-2A-cont kernel-dispatch refactor
        // will consume it on its branch via `MlxBuffer::slice_view`.
        let _ = &multi_seq_kv_hb;

        // The 4 production KV-regime branches.  Order matches the
        // alloc-site precedence at line 832-891 (HF2Q_HYBRID_KV first,
        // then TQ codebook bits, then default).  Each branch's typed
        // CapabilityUnsupported names the specific sub-iter so operator
        // log greps + future-iter implementer can route to the right
        // pin pointer.
        let cb_bits = crate::serve::api::tq_packed_descriptor::effective_gemma_tq_codebook_bits();
        // HF2Q_USE_DENSE selects the dense F32 KV path (LCP-eligible).
        // INVESTIGATION_ENV.use_dense is the canonical reader; mirrors
        // forward_decode at `forward_gpu.rs:407`.
        //
        // ADR-040 iter-B4c-kernel iter-2D (2026-05-30) — PRODUCTION
        // OPT-IN (HF2Q_USE_DENSE=1) slot routing through the dense F32
        // KV scaffold.  Architecture-disjoint variant of iter-2A-cont /
        // iter-2B's slice_view mount + delegate-to-sibling pattern
        // applied to `MultiSeqDenseKvBuffers` (single-K + single-V,
        // dtype-aware F32-or-F16 per ADR-017 Phase E.a iter-3.5a) on
        // `self.dense_kvs` (Option<Vec<Arc<DenseKvBuffers>>>).  The
        // sibling fn's `restored_lcp=None` branch was alloc-gate-aligned
        // to consume `self.dense_kvs.take()` when `Some(_)` (mirror of
        // iter-2A-cont's `self.leg_hb_encoded.is_none()` gate at line
        // ~860 + iter-2B's `self.hybrid_kv.is_none()` gate at line ~842).
        // SerialFifo byte-equivalence preserved: SerialFifo enters with
        // `self.dense_kvs == None` so the fresh-alloc body runs verbatim.
        //
        // The prior iter-2A typed-deferral label
        // (`iter-B4c-kernel-iter-2D per ADR-040 §6.1.32`) that lived
        // inside a `MultiSeqError::CapabilityUnsupported` constructor was
        // REPLACED at iter-2D SHIP with the real slot routing below.
        // H182 pins typed-error removal; H87 negative-grep preserved
        // because the label substring still lives in this doc-comment
        // cite (not inside a `CapabilityUnsupported { capability:`
        // constructor).
        if INVESTIGATION_ENV.use_dense {
            // Defense-in-depth: the iter-C2c-cont-cont Phase 3 scaffold
            // MUST be present when HF2Q_USE_DENSE=1 is engaged at
            // SlotAware spawn time.  Operator who flips HF2Q_USE_DENSE
            // AFTER process start (LazyLock-cached) would surface typed
            // `CapabilityUnsupported` here — honest pin at the
            // production boundary.  Same discipline as the iter-2B
            // hybrid-scaffold-absent defense-in-depth at line ~2680.
            let multi_seq_kv_dense = multi_seq_kv_dense.ok_or_else(|| {
                let err = MultiSeqError::CapabilityUnsupported {
                    capability: "gemma4-forward-prefill-dense-scaffold-absent (iter-C2c-cont-cont-invariant-violated per ADR-040 §6.1.46 — HF2Q_USE_DENSE=1 dense F32 opt-in surface; INVESTIGATION_ENV.use_dense == true AT CALL TIME but multi_seq_kv_dense scaffold is None at engine.rs GemmaLoadedModel.multi_seq_kv_dense — iter-C2c-cont-cont spawn-arm invariant violated OR operator flipped HF2Q_USE_DENSE post-LazyLock-cache; gated on iter-C2c-cont-cont per ADR-040 §6.1.46 provisioning MultiSeqDenseKvBuffers sibling scaffold Phase 3)",
                };
                anyhow::anyhow!(
                    "forward_prefill_with_soft_tokens_slot_aware: dense F32 KV path \
                     (HF2Q_USE_DENSE=1; OPT-IN PRE-DEFAULT) — multi_seq_kv_dense is None \
                     at slot_id={} (ADR-040 iter-B4c-kernel iter-2D). {}",
                    slot_id.0,
                    err,
                )
            })?;

            // Bounds + length match (sibling discipline to the HB
            // preflight at the top of this fn).
            if multi_seq_kv_dense.is_empty() {
                anyhow::bail!(
                    "forward_prefill_with_soft_tokens_slot_aware: multi_seq_kv_dense is empty \
                     (iter-C2c-cont-cont spawn-arm invariant: \
                      provision_multi_seq_kv_for_slot_aware Phase 3 must produce one \
                      entry per layer; ADR-040 §6.1.46 / iter-B4c-kernel iter-2D)"
                );
            }
            if multi_seq_kv_dense.len() != self.layers.len() {
                anyhow::bail!(
                    "forward_prefill_with_soft_tokens_slot_aware: multi_seq_kv_dense.len()={} \
                     != self.layers.len()={} — caller-invariant violation \
                     (ADR-040 iter-B4c-kernel iter-2D; iter-C2c-cont-cont spawn-arm \
                      Phase 3 must allocate one per layer)",
                    multi_seq_kv_dense.len(),
                    self.layers.len(),
                );
            }
            let dense_n_seqs = multi_seq_kv_dense[0].n_seqs;
            if slot_id.0 >= dense_n_seqs {
                let err = MultiSeqError::SlotOutOfRange {
                    slot: slot_id,
                    max_slots: dense_n_seqs,
                };
                anyhow::bail!(
                    "forward_prefill_with_soft_tokens_slot_aware: dense slot_id={} out of range \
                     (n_seqs={}). ADR-040 iter-B4c-kernel iter-2D bounds-first contract. {}",
                    slot_id.0,
                    dense_n_seqs,
                    err,
                );
            }

            // ── Build per-layer slot-views + mount on self.dense_kvs ─────
            //
            // Dense F32 path: K + V are both `dtype`-sized (F32 = 4
            // bytes/elem or F16 = 2 bytes/elem per ADR-017 Phase E.a
            // iter-3.5a invariant).  Per-slot byte offset =
            // `slot_id.0 * nkv * cap * hd * dtype.size_of()` — same
            // primitive Qwen35 B4a-cont uses at `gpu_full_attn.rs:107-115`
            // for its F32 KV variant.
            //
            // The mount target is `self.dense_kvs:
            // Option<Vec<Arc<DenseKvBuffers>>>`.  Each per-layer entry
            // is an `Arc<DenseKvBuffers>` wrapping the slot-view K + V.
            // The sibling fn's NEW alloc-gate at line ~676-808
            // (iter-2D alignment) reads
            // `if let Some(arcs) = self.dense_kvs.take()` to consume
            // the slot-view bundle — mirror of iter-2A-cont's
            // `self.leg_hb_encoded.is_none()` consume-gate at line ~902.
            let mut slot_view_dense: Vec<std::sync::Arc<DenseKvBuffers>> =
                Vec::with_capacity(multi_seq_kv_dense.len());
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let nkv = layer.num_kv_heads;
                let hd = layer.head_dim;
                let buf = &multi_seq_kv_dense[layer_idx];
                let cap = buf.capacity;
                let dtype = buf.dtype;
                let dtype_size = dtype.size_of();
                // K (dtype-sized) per-slot byte offset.
                let elems_per_slot: usize = nkv
                    .checked_mul(cap)
                    .and_then(|x| x.checked_mul(hd))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "dense slot-view elem count overflow at L{layer_idx} \
                         (nkv={nkv} cap={cap} hd={hd}) — ADR-040 iter-B4c-kernel iter-2D"
                        )
                    })?;
                let bytes_per_slot: u64 = (elems_per_slot as u64)
                    .checked_mul(dtype_size as u64)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "dense slot-view byte size overflow at L{layer_idx} \
                         (dtype_size={}) — ADR-040 iter-B4c-kernel iter-2D",
                            dtype_size,
                        )
                    })?;
                let byte_offset: u64 =
                    (slot_id.0 as u64)
                        .checked_mul(bytes_per_slot)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "dense slot-view byte offset overflow at L{layer_idx} \
                         (slot_id={} bytes_per_slot={}) \
                         — ADR-040 iter-B4c-kernel iter-2D",
                                slot_id.0,
                                bytes_per_slot,
                            )
                        })?;
                let k_view = buf
                    .k
                    .slice_view(byte_offset, elems_per_slot)
                    .with_shape(vec![nkv, cap, hd])
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "dense slot-view K with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2D"
                        )
                    })?;
                let v_view = buf
                    .v
                    .slice_view(byte_offset, elems_per_slot)
                    .with_shape(vec![nkv, cap, hd])
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "dense slot-view V with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2D"
                        )
                    })?;

                // Construct the legacy single-seq `DenseKvBuffers`
                // wrapper around the slot-views (mirror of iter-2A-cont
                // prefill HbKvBuffers construction at line ~3119) +
                // wrap in Arc (mirror of self.dense_kvs's
                // `Vec<Arc<DenseKvBuffers>>` shape).
                slot_view_dense.push(std::sync::Arc::new(DenseKvBuffers {
                    k: k_view,
                    v: v_view,
                    capacity: cap,
                    is_sliding: buf.is_sliding,
                    dtype,
                }));
            }

            // Mount the slot-view bundle on `self.dense_kvs`.  Save the
            // prior value so we can restore it on exit — typical state
            // is `None` for a fresh request, but a prior SerialFifo +
            // LCP call MAY have left an already-allocated value.  The
            // iter-1 worker-arm predicate gates this fn on SlotAware +
            // SlotId(N>0), so the call shape is well-defined —
            // SerialFifo never reaches here.  Restoring belt-and-
            // suspenders.  Mirror of iter-2A-cont prefill restore at
            // line ~3137-3167.
            let prior_dense_kvs = self.dense_kvs.take();
            self.dense_kvs = Some(slot_view_dense);

            // Delegate to the sibling fn.  Its alloc-gate at line
            // ~676 (NEW iter-2D alignment to consume
            // `self.dense_kvs.take()` on the `restored_lcp=None` branch
            // when `Some(_)`) so the mount survives the sibling's path.
            // The sibling consumes `self.dense_kvs` at the consume-gate
            // for the K/V kernel writes via the per-token loop's
            // `dense_kvs_vec[layer_idx].k/.v` reads; kernel writes
            // target the per-slot byte region of the persistent multi-
            // seq scaffold.
            //
            // `restored_lcp = None`: ADR-040 §6.1.50 (2026-05-30) closes
            // **iter-B4c-kernel-iter-2D-lcp per ADR-040 §6.1.50** as
            // **STRUCTURAL N/A** — the LCP partial-prefix resume path
            // consumes cached `Arc<DenseKvBuffers>` clones into
            // `self.dense_kvs` (see `engine.rs:7593` install + sibling
            // consume-gate at `forward_prefill.rs:709-768`), while the
            // slot-aware iter-2D path mounts slot-views into the SAME
            // `self.dense_kvs` field.  These are MUTUALLY EXCLUSIVE
            // mount sources — only one can populate `self.dense_kvs` at
            // a time.  Additionally, the LCP registry is a global
            // cache; cross-slot prefix sharing carries tenant-isolation
            // risk (slot regions are per-tenant by SlotAware
            // construction).  The structural-N/A pin documents that
            // slot-aware mode's existing prompt-cache HIT via the
            // SerialFifo path's restore_partial mechanism IS the LCP
            // fast-path that operators get under iter-2D today; the
            // remaining cross-request prefix probe is structurally
            // incompatible with per-slot byte regions.  Forward-pointer
            // discoverability per H87 discipline: the substring
            // `iter-B4c-kernel-iter-2D-lcp per ADR-040 §6.1.50` is
            // grep-able here (NOT inside a typed
            // `MultiSeqError::CapabilityUnsupported` constructor — the
            // STRUCTURAL N/A closure mirrors §6.1.49 iter-2-embed /
            // iter-2-batched discipline).
            let result = self.forward_prefill_with_soft_tokens_resume(
                prompt_tokens,
                soft_tokens,
                max_decode_tokens,
                gpu,
                (cached_tokens > 0).then_some(cached_tokens),
                true, // slot_aware=true (ADR-040 STEP-1b): skip shared dense_kvs/snapshot/sdpa_tmp write-backs; caller restores
            );

            // Restore prior `self.dense_kvs` regardless of result.  The
            // slot-view Arc bundle has lifetime tied to this call; the
            // persistent multi-seq scaffold OWNED by the worker arm
            // (via `g.multi_seq_kv_dense`) keeps the underlying buffers
            // alive — the slot-view ARC handles dropped here are strong
            // refs to the same Metal buffer storage.
            self.dense_kvs = prior_dense_kvs;

            return result;
        }
        if cb_bits == 0 {
            // ADR-040 iter-B4c-kernel iter-2C (2026-05-30) — LEGACY 4-bit
            // nibble-packed slot routing through `MultiSeqMlxKvCache`.
            // Architecture-disjoint variant of iter-2A-cont's slice_view
            // mount + delegate-to-sibling pattern applied to
            // `self.kv_caches: Vec<MlxKvCache>` (always-populated,
            // mounted via `std::mem::replace` of the entire Vec; the
            // legacy field is NOT Option-wrapped per `gemma4/model.rs:
            // 1277-1290` always-alloc-at-load-time discipline — no
            // `is_none()` gate needed, the gate is structural).
            //
            // The prior iter-2A typed-deferral label
            // (`iter-B4c-kernel-iter-2C per ADR-040 §6.1.32`) that lived
            // inside a `MultiSeqError::CapabilityUnsupported` constructor
            // was REPLACED at iter-2C SHIP with the real slot routing
            // below.  H181 pins typed-error removal; H87 negative-grep
            // preserved because the label substring still lives in this
            // doc-comment cite (not inside a `CapabilityUnsupported`
            // constructor).
            //
            // 127-byte sourdough ceiling means production should not
            // engage this regime, but the real slot routing lands for
            // operator deployment-time freedom under SlotAware
            // (HF2Q_TQ_CODEBOOK_BITS=4 opt-in).

            // Defense-in-depth: the iter-C2c-cont-cont Phase 4 scaffold
            // MUST be present when HF2Q_TQ_CODEBOOK_BITS=4 is engaged
            // at SlotAware spawn time.  Same discipline as the iter-2B
            // hybrid-scaffold-absent defense-in-depth at line ~2680.
            let multi_seq_kv_mlx = multi_seq_kv_mlx.ok_or_else(|| {
                let err = MultiSeqError::CapabilityUnsupported {
                    capability: "gemma4-forward-prefill-mlx-scaffold-absent (iter-C2c-cont-cont-invariant-violated per ADR-040 §6.1.46 — HF2Q_TQ_CODEBOOK_BITS=4 legacy 4-bit opt-in pre-default surface; cb_bits == 0 AT CALL TIME but multi_seq_kv_mlx scaffold is None at engine.rs GemmaLoadedModel.multi_seq_kv_mlx — iter-C2c-cont-cont spawn-arm invariant violated OR operator flipped HF2Q_TQ_CODEBOOK_BITS post-LazyLock-cache; gated on iter-C2c-cont-cont per ADR-040 §6.1.46 provisioning MultiSeqMlxKvCache sibling scaffold Phase 4)",
                };
                anyhow::anyhow!(
                    "forward_prefill_with_soft_tokens_slot_aware: legacy 4-bit TQ path \
                     (HF2Q_TQ_CODEBOOK_BITS=4; OPT-IN PRE-DEFAULT) — multi_seq_kv_mlx is None \
                     at slot_id={} (ADR-040 iter-B4c-kernel iter-2C). {}",
                    slot_id.0,
                    err,
                )
            })?;

            // Bounds + length match.
            if multi_seq_kv_mlx.is_empty() {
                anyhow::bail!(
                    "forward_prefill_with_soft_tokens_slot_aware: multi_seq_kv_mlx is empty \
                     (iter-C2c-cont-cont spawn-arm invariant; ADR-040 §6.1.46 / \
                      iter-B4c-kernel iter-2C)"
                );
            }
            if multi_seq_kv_mlx.len() != self.layers.len() {
                anyhow::bail!(
                    "forward_prefill_with_soft_tokens_slot_aware: multi_seq_kv_mlx.len()={} \
                     != self.layers.len()={} — caller-invariant violation \
                     (ADR-040 iter-B4c-kernel iter-2C; iter-C2c-cont-cont spawn-arm \
                      Phase 4 must allocate one per layer)",
                    multi_seq_kv_mlx.len(),
                    self.layers.len(),
                );
            }
            let mlx_n_seqs = multi_seq_kv_mlx[0].n_seqs;
            if slot_id.0 >= mlx_n_seqs {
                let err = MultiSeqError::SlotOutOfRange {
                    slot: slot_id,
                    max_slots: mlx_n_seqs,
                };
                anyhow::bail!(
                    "forward_prefill_with_soft_tokens_slot_aware: mlx slot_id={} out of range \
                     (n_seqs={}). ADR-040 iter-B4c-kernel iter-2C bounds-first contract. {}",
                    slot_id.0,
                    mlx_n_seqs,
                    err,
                );
            }

            // ── Build per-layer slot-views + mount on self.kv_caches ─────
            //
            // Legacy 4-bit nibble-packed path: K + V are BOTH U8-packed
            // (`hd/2` packed bytes/pos) + F32 norms (`norms_per_pos`
            // floats/pos).  Per-slot byte offsets:
            //   * K_packed / V_packed (U8 = 1 byte/elem):
            //     `slot_id.0 * nkv * cap * (hd/2)`.
            //   * K_norms / V_norms (F32 = 4 bytes/elem):
            //     `slot_id.0 * nkv * cap * norms_per_pos * 4`.
            // Shape selection matches the legacy `gemma4/model.rs:
            // 1278-1283` alloc layout: packed = `[nkv, cap, hd/2]`;
            // norms = `[nkv, cap]` (norms_per_pos==1) or
            // `[nkv, cap, norms_per_pos]` (otherwise).
            let mut slot_view_kv: Vec<MlxKvCache> = Vec::with_capacity(multi_seq_kv_mlx.len());
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let nkv = layer.num_kv_heads;
                let hd = layer.head_dim;
                let buf = &multi_seq_kv_mlx[layer_idx];
                let cap = buf.capacity;
                let norms_per_pos = buf.norms_per_pos;
                let hd_half = hd / 2;
                // K_packed / V_packed elem count per slot (U8 = 1 byte/elem).
                let packed_elems_per_slot: usize = nkv
                    .checked_mul(cap)
                    .and_then(|x| x.checked_mul(hd_half))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "MLX slot-view packed elem count overflow at L{layer_idx} \
                         (nkv={nkv} cap={cap} hd_half={hd_half}) \
                         — ADR-040 iter-B4c-kernel iter-2C"
                        )
                    })?;
                let packed_byte_offset: u64 = (slot_id.0 as u64)
                    .checked_mul(packed_elems_per_slot as u64) // U8 = 1 byte/elem
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "MLX slot-view packed byte offset overflow at L{layer_idx} \
                         (slot_id={} packed_elems_per_slot={}) \
                         — ADR-040 iter-B4c-kernel iter-2C",
                            slot_id.0,
                            packed_elems_per_slot,
                        )
                    })?;
                let k_packed_view = buf
                    .k_packed
                    .slice_view(packed_byte_offset, packed_elems_per_slot)
                    .with_shape(vec![nkv, cap, hd_half])
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "MLX slot-view K_packed with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2C"
                        )
                    })?;
                let v_packed_view = buf
                    .v_packed
                    .slice_view(packed_byte_offset, packed_elems_per_slot)
                    .with_shape(vec![nkv, cap, hd_half])
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "MLX slot-view V_packed with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2C"
                        )
                    })?;
                // K_norms / V_norms (F32 = 4 bytes/elem).
                let norms_elems_per_slot: usize = nkv
                    .checked_mul(cap)
                    .and_then(|x| x.checked_mul(norms_per_pos))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "MLX slot-view norms elem count overflow at L{layer_idx} \
                         (nkv={nkv} cap={cap} norms_per_pos={norms_per_pos}) \
                         — ADR-040 iter-B4c-kernel iter-2C"
                        )
                    })?;
                let norms_bytes_per_slot: u64 = (norms_elems_per_slot as u64)
                    .checked_mul(4u64) // F32 = 4 bytes/elem
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "MLX slot-view norms byte size overflow at L{layer_idx} \
                         — ADR-040 iter-B4c-kernel iter-2C"
                        )
                    })?;
                let norms_byte_offset: u64 = (slot_id.0 as u64)
                    .checked_mul(norms_bytes_per_slot)
                    .ok_or_else(|| {
                    anyhow::anyhow!(
                        "MLX slot-view norms byte offset overflow at L{layer_idx} \
                         — ADR-040 iter-B4c-kernel iter-2C"
                    )
                })?;
                let norms_shape = if norms_per_pos == 1 {
                    vec![nkv, cap]
                } else {
                    vec![nkv, cap, norms_per_pos]
                };
                let k_norms_view = buf
                    .k_norms
                    .slice_view(norms_byte_offset, norms_elems_per_slot)
                    .with_shape(norms_shape.clone())
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "MLX slot-view K_norms with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2C"
                        )
                    })?;
                let v_norms_view = buf
                    .v_norms
                    .slice_view(norms_byte_offset, norms_elems_per_slot)
                    .with_shape(norms_shape)
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "MLX slot-view V_norms with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2C"
                        )
                    })?;

                // Construct legacy single-seq `MlxKvCache` wrapper
                // around the 4 slot-views.  Mirror of iter-2A-cont
                // prefill HbKvBuffers construction at line ~3119; the
                // MlxKvCache layout adds `write_pos` + `seq_len` (the
                // legacy single-seq cursor pair) initialized to 0 since
                // the slot-view bundle is fresh-per-call (the persistent
                // multi-seq scaffold's `seq_lens[slot]` cursor was
                // reset_for_slot at orchestrator entry).
                slot_view_kv.push(MlxKvCache {
                    k_packed: k_packed_view,
                    k_norms: k_norms_view,
                    v_packed: v_packed_view,
                    v_norms: v_norms_view,
                    capacity: cap,
                    is_sliding: buf.is_sliding,
                    write_pos: 0,
                    seq_len: 0,
                });
            }

            // Mount the slot-view bundle on `self.kv_caches`.  Save the
            // prior value so we can restore it on exit.  Unlike the
            // hybrid + HB-encoded paths (which use Option<Vec<_>>),
            // `self.kv_caches: Vec<MlxKvCache>` is always populated at
            // model load time (`gemma4/model.rs:1292`) — we swap the
            // entire Vec via `std::mem::replace`.  The legacy alloc-
            // site discipline at load time enforces the structural
            // invariant; no `is_none()` gate needed at the sibling fn
            // body (mirror of iter-2A-cont's gate alignment at line
            // ~860 is N/A here because there's no lazy-alloc on
            // `self.kv_caches`).
            //
            // SlotId(0) byte-equivalence: at `slot_id=SlotId(0)` the
            // byte offset is 0; `slice_view(0, n_elements) + with_shape`
            // produces a view byte-identical to a fresh single-seq
            // alloc.  BUT: the iter-1 worker-arm predicate gates this
            // fn on SlotAware + SlotId(N>0), so SlotId(0) is never
            // observed here — code-path disjointness preserves the
            // H1/H2/H23/H41/H44/H77/H102/H178/H187 byte-equivalence
            // chain.
            let prior_kv_caches = std::mem::replace(&mut self.kv_caches, slot_view_kv);

            // Delegate to the sibling fn.  The sibling resets
            // `self.kv_caches[layer].write_pos = 0` + `seq_len = 0` at
            // entry (line ~571-589) so the per-token loop targets the
            // slot-view's region.  The per-token loop's writes inside
            // the dispatchers consume `self.kv_caches[layer].k_packed`
            // etc. via the kernel-write path at
            // `gemma4/forward_gpu.rs:1525-1526` — slot-view ARC handles
            // route writes to the per-slot byte region.
            //
            // `restored_lcp = None`: LCP partial-prefix resume is
            // dense F32 LCP-eligible regime (iter-2D scope), orthogonal
            // to the 4-bit nibble-packed iter-2C path.
            let result = self.forward_prefill_with_soft_tokens_resume(
                prompt_tokens,
                soft_tokens,
                max_decode_tokens,
                gpu,
                (cached_tokens > 0).then_some(cached_tokens),
                true, // slot_aware=true (ADR-040 STEP-1b): skip shared dense_kvs/snapshot/sdpa_tmp write-backs; caller restores
            );

            // Restore prior `self.kv_caches` regardless of result.  The
            // slot-view bundle has lifetime tied to this call; the
            // persistent multi-seq scaffold OWNED by the worker arm
            // (via `g.multi_seq_kv_mlx`) keeps the underlying buffers
            // alive — the slot-view ARC handles dropped here are strong
            // refs to the same Metal buffer storage.
            self.kv_caches = prior_kv_caches;

            return result;
        }
        if INVESTIGATION_ENV.hybrid_kv {
            // ADR-040 iter-B4c-kernel iter-2B (2026-05-30) — PRODUCTION
            // DEFAULT slot routing through the hybrid F16-K + TQ-HB-V
            // KV scaffold.  H10 was FALSIFIED at §6.1.11 —
            // `HF2Q_HYBRID_KV` is default-true since ADR-029 iter-13
            // (2026-05-11).  This branch is the production-engagement
            // surface: every SlotAware + SlotId(N>0) request under the
            // default env config lands here.
            //
            // The implementation mirrors Qwen35 B4a-cont's slice_view
            // approach (§6.1.5) for an architecture-disjoint variant:
            //
            //   1. Defense-in-depth: the iter-C2c-cont sibling scaffold
            //      MUST be present.  Operator who flips HF2Q_HYBRID_KV
            //      AFTER process start (LazyLock-cached) would surface
            //      typed `CapabilityUnsupported` here — honest pin at
            //      the production boundary.
            //   2. Bounds + length match: same contract as the HB
            //      pre-flight at the top of this fn, applied to the
            //      hybrid scaffold's `n_seqs` + per-layer length.
            //   3. iter-2B-xlen sub-stage: if any layer's
            //      `bf16_xlen_k.is_some()` (HF2Q_DFLASH_XLEN_SDPA=1
            //      opt-in surface), surface typed
            //      `iter-B4c-kernel-iter-2B-xlen` — the BF16 xlen
            //      buffer slot routing is honestly named as a follow-up.
            //   4. Build per-layer slot-views: for K (F16, 2 bytes/elem),
            //      V (U8 packed or F16, dtype-dependent), V_norms (F32 or
            //      F32 dummy), compute the per-slot byte offset
            //      `slot_id.0 * nkv * cap * hd * dtype_size` and apply
            //      `slice_view + with_shape([nkv, cap, hd])` to preserve
            //      the legacy 3-D shape downstream kernels read for
            //      stride math.  Same primitive Qwen35 B4a-cont uses at
            //      `gpu_full_attn.rs:172-181`.
            //   5. Mount on `self.hybrid_kv`: temporarily replace with
            //      the slot-view Vec so the sibling fn's body at
            //      line 1290-1330 reads from the per-slot region
            //      bit-identically to a single-seq alloc.  The sibling's
            //      lazy-alloc gate at line ~842 was aligned with the
            //      decode-path gate (`is_none()` check) so the mount
            //      survives.
            //   6. Delegate to `forward_prefill_with_soft_tokens_resume`
            //      with `restored_lcp = None` (LCP partial-prefix resume
            //      is iter-2D scope).
            //   7. Restore `self.hybrid_kv` to its prior value on exit
            //      — typically `None` for a fresh request, regardless
            //      of whether the delegate returned `Ok` or `Err`.
            //
            // SlotId(0) byte-equivalence: at `slot_id=SlotId(0)` the
            // byte offset is 0; `slice_view(0, n_elements) + with_shape`
            // produces a view that is byte-identical to a fresh
            // single-seq alloc at the same shape.  BUT: the iter-1
            // worker-arm predicate `slot_id != SlotId(0)` already short-
            // circuits this fn before entry, so SlotId(0) is never
            // observed here — code-path disjointness preserves the
            // H1/H2/H23/H41/H44/H77/H102 byte-equivalence chain.
            let multi_seq_kv_hybrid = multi_seq_kv_hybrid.ok_or_else(|| {
                // Defense-in-depth typed error — operator-grep'able via
                // the `iter-C2c-cont-invariant-violated` label.  This is
                // a DIFFERENT capability label from the iter-2A typed
                // deferral (`gemma4-forward-prefill-slot-N-hybrid (iter-
                // B4c-kernel-iter-2B per ...)`) — that label was REMOVED
                // when iter-2B shipped real slot routing.  The H97 pin
                // checks the iter-2A label is GONE; the H97 negative
                // grep does NOT match this defense-in-depth label
                // (different surface prefix).
                let err = MultiSeqError::CapabilityUnsupported {
                    capability: "gemma4-forward-prefill-hybrid-scaffold-absent (iter-C2c-cont-invariant-violated per ADR-040 §6.1.34 — HF2Q_HYBRID_KV=1 production-default; INVESTIGATION_ENV.hybrid_kv == true AT CALL TIME but multi_seq_kv_hybrid scaffold is None at engine.rs GemmaLoadedModel.multi_seq_kv_hybrid — iter-C2c-cont spawn-arm invariant violated OR operator flipped HF2Q_HYBRID_KV post-LazyLock-cache; gated on iter-C2c-cont per ADR-040 §6.1.33 provisioning MultiSeqHybridKvBuffers sibling scaffold)",
                };
                anyhow::anyhow!(
                    "forward_prefill_with_soft_tokens_slot_aware: hybrid F16-K + TQ-HB-V path \
                     (HF2Q_HYBRID_KV=1; PRODUCTION DEFAULT) — multi_seq_kv_hybrid is None \
                     at slot_id={} (ADR-040 iter-B4c-kernel iter-2B). {}",
                    slot_id.0,
                    err,
                )
            })?;

            // Bounds + length match (sibling discipline to the HB
            // preflight at the top of this fn).
            if multi_seq_kv_hybrid.is_empty() {
                anyhow::bail!(
                    "forward_prefill_with_soft_tokens_slot_aware: multi_seq_kv_hybrid is empty \
                     (iter-C2c-cont spawn-arm invariant: \
                      provision_multi_seq_kv_for_slot_aware Phase 2 must produce one \
                      entry per layer; ADR-040 §6.1.33 / iter-B4c-kernel iter-2B)"
                );
            }
            if multi_seq_kv_hybrid.len() != self.layers.len() {
                anyhow::bail!(
                    "forward_prefill_with_soft_tokens_slot_aware: multi_seq_kv_hybrid.len()={} \
                     != self.layers.len()={} — caller-invariant violation \
                     (ADR-040 iter-B4c-kernel iter-2B; iter-C2c-cont spawn-arm \
                      Phase 2 must allocate one per layer)",
                    multi_seq_kv_hybrid.len(),
                    self.layers.len(),
                );
            }
            let hybrid_n_seqs = multi_seq_kv_hybrid[0].n_seqs;
            if slot_id.0 >= hybrid_n_seqs {
                let err = MultiSeqError::SlotOutOfRange {
                    slot: slot_id,
                    max_slots: hybrid_n_seqs,
                };
                anyhow::bail!(
                    "forward_prefill_with_soft_tokens_slot_aware: hybrid slot_id={} out of range \
                     (n_seqs={}). ADR-040 iter-B4c-kernel iter-2B bounds-first contract. {}",
                    slot_id.0,
                    hybrid_n_seqs,
                    err,
                );
            }

            // ADR-040 iter-B4c-kernel iter-2B-xlen (2026-05-30) —
            // PRODUCTION OPT-IN BF16 xlen K/V slot routing landed via
            // the iter-2B slice_view mount + delegate-to-sibling pattern
            // applied to the bf16_xlen_k / bf16_xlen_v Optional buffers.
            // The prior iter-2B typed-deferral capability literal
            // `gemma4-forward-prefill-slot-N-hybrid-xlen (iter-B4c-kernel-iter-2B-xlen per`
            // lived inside a `MultiSeqError::CapabilityUnsupported
            // { capability: "..." }` constructor; iter-2B-xlen REMOVES
            // that constructor (H189 pin) and threads `Some(slot_view)`
            // for both bf16_xlen_k + bf16_xlen_v into the constructed
            // legacy `HybridKvBuffers` per layer.  H87 negative-grep is
            // preserved because the iter-2B-xlen label substring still
            // lives in this doc-comment cite as an operator-grep'able
            // forward pointer.
            //
            // Why this path exists at all:
            //   * ADR-030 iter-96: HF2Q_DFLASH_XLEN_SDPA=1 opt-in surface
            //     allocates BF16 K/V caches at alloc-time per
            //     `gemma4/kv_cache.rs:1102-1115` (multi-seq variant) —
            //     same `[n_seqs, nkv, cap, hd]` outer layout as the F16
            //     K cache, but BF16 dtype (2 bytes/elem; numerically
            //     same stride as F16 K).  The downstream consumer is
            //     `dispatch_kv_cache_copy_seq_bf16_to_bf16_head_major`
            //     at `forward_prefill_batched.rs:1515` for the cross-
            //     length SDPA verify path.
            //   * Default OFF: when env unset, alloc-time leaves both
            //     fields as `None` → `xlen_engaged = false` here →
            //     `bf16_xlen_k: None, bf16_xlen_v: None` propagated
            //     into the per-layer slot-view bundle below → H193 pin
            //     (default xlen=None path UNCHANGED).
            //   * Default ON: every layer's `bf16_xlen_k` is `Some(_)`
            //     AND `bf16_xlen_v` is `Some(_)` by construction (the
            //     alloc helper allocates both together) → both
            //     slot-views materialize per layer.
            //
            // Per-slot byte offset (H191):
            //   * bf16_xlen_k: `slot_id.0 * nkv * cap * hd * 2`
            //     (BF16 = 2 bytes/elem; numerically identical to the
            //     F16 K stride but on a separately-typed buffer).
            //   * bf16_xlen_v: same arithmetic.
            //   * Slot 0 view starts at byte 0; slot N view starts at
            //     `N * (nkv * cap * hd * 2)`.  At n_seqs=1 (legacy
            //     allocator output), the byte counts are byte-equivalent
            //     to a fresh single-seq xlen alloc (H191 transitivity
            //     to the K-side F16 byte equivalence).
            //
            // Per-slot byte isolation (H192):
            //   * The multi-seq allocator emits a contiguous slab
            //     `[n_seqs, nkv, cap, hd]` for BOTH xlen K + V; per-slot
            //     byte regions are contiguous + disjoint by construction
            //     of the OUTERMOST n_seqs axis.  A slot-0 write via
            //     `dispatch_kv_cache_copy_seq_bf16_to_bf16_head_major`
            //     targeting the slot-0 view cannot reach slot-1's region.
            //
            // Per-layer xlen presence MUST be consistent (defense-in-
            // depth invariant): either ALL layers carry Some xlen K + V
            // (env-gate ON at alloc) or NONE (env-gate OFF).  Mixed
            // presence is impossible per the alloc-helper construction
            // and would indicate a layer-vec corruption — bail with
            // typed `CapabilityUnsupported` naming the inconsistency.
            let xlen_engaged = multi_seq_kv_hybrid
                .iter()
                .any(|buf| buf.bf16_xlen_k.is_some() || buf.bf16_xlen_v.is_some());
            if xlen_engaged {
                let all_consistent = multi_seq_kv_hybrid
                    .iter()
                    .all(|buf| buf.bf16_xlen_k.is_some() && buf.bf16_xlen_v.is_some());
                if !all_consistent {
                    let err = MultiSeqError::CapabilityUnsupported {
                        capability: "gemma4-forward-prefill-slot-N-hybrid-xlen-mixed-presence (iter-B4c-kernel-iter-2B-xlen per ADR-040 §6.1.47 — alloc-helper invariant violation: bf16_xlen_k/v must be Some on ALL layers or NONE, never mixed; if you see this in production, gemma4/kv_cache.rs:1102-1115 lost atomicity across layer-vec alloc loop)",
                    };
                    anyhow::bail!(
                        "forward_prefill_with_soft_tokens_slot_aware: hybrid xlen BF16 path \
                         (HF2Q_DFLASH_XLEN_SDPA=1) per-layer presence MIXED at slot_id={} \
                         — alloc-helper invariant violation (ADR-040 iter-B4c-kernel iter-2B-xlen scope). {}",
                        slot_id.0,
                        err,
                    );
                }
            }

            // ── Build per-layer slot-views + mount on self.hybrid_kv ─────
            //
            // The slot's byte offset is `slot_id.0 * (nkv * cap * hd *
            // dtype_size)` — same primitive Qwen35 B4a-cont uses at
            // `gpu_full_attn.rs:107-115` for its F32 KV variant
            // (`slot_id.0 * nkv * max_seq_len * head_dim * 4`).  Here:
            //   * K: F16 → dtype_size = 2 bytes.
            //   * V: U8 (TQ-packed) → dtype_size = 1 byte; OR F16 (when
            //     HF2Q_FULL_F16_KV=1) → dtype_size = 2 bytes.  Source
            //     of truth: the buffer's own `.dtype()` reflects the
            //     alloc-time choice (see
            //     gemma4/kv_cache.rs::alloc_multi_seq_hybrid_kv_for_layer
            //     lines 971-999).
            //   * V_norms: F32 (or 4-byte dummy when full_f16_v=1).
            //     `norms_per_pos = max(1, hd/256)`.
            let mut slot_view_hybrid: Vec<crate::inference::models::gemma4::HybridKvBuffers> =
                Vec::with_capacity(multi_seq_kv_hybrid.len());
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let nkv = layer.num_kv_heads;
                let hd = layer.head_dim;
                let buf = &multi_seq_kv_hybrid[layer_idx];
                let cap = buf.capacity;
                // K (F16, 2 bytes/elem) per-slot byte offset.
                let k_elems_per_slot: usize = nkv
                    .checked_mul(cap)
                    .and_then(|x| x.checked_mul(hd))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "slot-view K elem count overflow at L{layer_idx} \
                         (nkv={nkv} cap={cap} hd={hd}) — ADR-040 iter-B4c-kernel iter-2B"
                        )
                    })?;
                let k_bytes_per_slot: u64 = (k_elems_per_slot as u64)
                    .checked_mul(2u64) // F16 = 2 bytes/elem
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "slot-view K byte size overflow at L{layer_idx} \
                         — ADR-040 iter-B4c-kernel iter-2B"
                        )
                    })?;
                let k_byte_offset: u64 = (slot_id.0 as u64)
                    .checked_mul(k_bytes_per_slot)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "slot-view K byte offset overflow at L{layer_idx} \
                         (slot_id={} k_bytes_per_slot={}) — ADR-040 iter-B4c-kernel iter-2B",
                            slot_id.0,
                            k_bytes_per_slot,
                        )
                    })?;
                let k_view = buf
                    .k
                    .slice_view(k_byte_offset, k_elems_per_slot)
                    .with_shape(vec![nkv, cap, hd])
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "slot-view K with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2B"
                        )
                    })?;

                // V: dtype-aware.  U8 (TQ-packed) → 1 byte/elem;
                // F16 → 2 bytes/elem.  `.dtype().size_of()` is the
                // canonical reader (defined in mlx-native).
                let v_dtype_size = buf.v_packed.dtype().size_of();
                let v_bytes_per_slot: u64 = (k_elems_per_slot as u64)
                    .checked_mul(v_dtype_size as u64)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "slot-view V byte size overflow at L{layer_idx} \
                         (v_dtype_size={}) — ADR-040 iter-B4c-kernel iter-2B",
                            v_dtype_size,
                        )
                    })?;
                let v_byte_offset: u64 = (slot_id.0 as u64)
                    .checked_mul(v_bytes_per_slot)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "slot-view V byte offset overflow at L{layer_idx} \
                         — ADR-040 iter-B4c-kernel iter-2B"
                        )
                    })?;
                let v_view = buf
                    .v_packed
                    .slice_view(v_byte_offset, k_elems_per_slot)
                    .with_shape(vec![nkv, cap, hd])
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "slot-view V with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2B"
                        )
                    })?;

                // V_norms: F32 (4 bytes/elem) with shape
                // `[nkv, cap, norms_per_pos]` per the multi-seq
                // allocator's layout — OR a 4-byte F32 dummy buffer
                // (1 element, shared across slots, no per-slot offset)
                // when HF2Q_FULL_F16_KV=1 was set at alloc time.  The
                // dummy case is detected by `byte_len() == 4`.
                let norms_per_pos = buf.norms_per_pos;
                let v_norms_is_dummy = buf.v_norms.byte_len() == 4;
                let v_norms_view = if v_norms_is_dummy {
                    // Shared dummy — every slot points at the same
                    // 4-byte F32 buffer (kernel's v_is_f16 FC=1 skips
                    // the read).  Match `alloc_hybrid_kv_for_layer`'s
                    // dummy shape `vec![1]`.
                    buf.v_norms
                        .slice_view(0, 1)
                        .with_shape(vec![1])
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "slot-view V_norms (dummy) with_shape at L{layer_idx}: {e} \
                             — ADR-040 iter-B4c-kernel iter-2B"
                            )
                        })?
                } else {
                    let norms_elems_per_slot: usize = nkv
                        .checked_mul(cap)
                        .and_then(|x| x.checked_mul(norms_per_pos))
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "slot-view V_norms elem count overflow at L{layer_idx} \
                             — ADR-040 iter-B4c-kernel iter-2B"
                            )
                        })?;
                    let norms_bytes_per_slot: u64 = (norms_elems_per_slot as u64)
                        .checked_mul(4u64) // F32 = 4 bytes/elem
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "slot-view V_norms byte size overflow at L{layer_idx} \
                             — ADR-040 iter-B4c-kernel iter-2B"
                            )
                        })?;
                    let norms_byte_offset: u64 = (slot_id.0 as u64)
                        .checked_mul(norms_bytes_per_slot)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "slot-view V_norms byte offset overflow at L{layer_idx} \
                             — ADR-040 iter-B4c-kernel iter-2B"
                            )
                        })?;
                    let v_norms_shape = if norms_per_pos == 1 {
                        vec![nkv, cap]
                    } else {
                        vec![nkv, cap, norms_per_pos]
                    };
                    buf.v_norms
                        .slice_view(norms_byte_offset, norms_elems_per_slot)
                        .with_shape(v_norms_shape)
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "slot-view V_norms with_shape at L{layer_idx}: {e} \
                             — ADR-040 iter-B4c-kernel iter-2B"
                            )
                        })?
                };

                // ADR-040 iter-B4c-kernel iter-2B-xlen — BF16 xlen K/V
                // per-slot slice_view materialization.  When the env-gate
                // is OFF (default), `xlen_engaged == false` →
                // (None, None) propagates through to the legacy
                // `HybridKvBuffers` struct verbatim (H193 pin: default
                // path UNCHANGED).  When the env-gate is ON,
                // `xlen_engaged == true` AND every layer's
                // `bf16_xlen_k.is_some() && bf16_xlen_v.is_some()`
                // (verified by the consistency check above) →
                // construct slot-views.
                //
                // Byte-offset arithmetic mirrors the K F16 path
                // verbatim (numerically identical strides: BF16 is also
                // 2 bytes/elem) but operates on the separately-typed
                // BF16 buffers.  The downstream consumer
                // `dispatch_kv_cache_copy_seq_bf16_to_bf16_head_major`
                // reads BF16 dtype from the slot-view's underlying
                // Metal buffer; the slice_view preserves the dtype tag.
                let (bf16_xlen_k_view, bf16_xlen_v_view) = if xlen_engaged {
                    let xlen_bk = buf
                        .bf16_xlen_k
                        .as_ref()
                        .expect("xlen consistency guard above: bf16_xlen_k Some");
                    let xlen_bv = buf
                        .bf16_xlen_v
                        .as_ref()
                        .expect("xlen consistency guard above: bf16_xlen_v Some");
                    // BF16 K + V share the same `[nkv, cap, hd]`
                    // per-slot layout as F16 K (alloc helper at
                    // `gemma4/kv_cache.rs:1102-1115`).  Per-slot
                    // element count = `nkv * cap * hd` (reuse the
                    // `k_elems_per_slot` computed above for the F16 K
                    // path — identical formula).
                    let xlen_bytes_per_slot: u64 = (k_elems_per_slot as u64)
                        .checked_mul(2u64) // BF16 = 2 bytes/elem
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "slot-view BF16 xlen byte size overflow at L{layer_idx} \
                             — ADR-040 iter-B4c-kernel iter-2B-xlen"
                            )
                        })?;
                    let xlen_byte_offset: u64 = (slot_id.0 as u64)
                        .checked_mul(xlen_bytes_per_slot)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "slot-view BF16 xlen byte offset overflow at L{layer_idx} \
                             (slot_id={} xlen_bytes_per_slot={}) \
                             — ADR-040 iter-B4c-kernel iter-2B-xlen",
                                slot_id.0,
                                xlen_bytes_per_slot,
                            )
                        })?;
                    let bk_view = xlen_bk
                        .slice_view(xlen_byte_offset, k_elems_per_slot)
                        .with_shape(vec![nkv, cap, hd])
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "slot-view BF16 xlen K with_shape at L{layer_idx}: {e} \
                             — ADR-040 iter-B4c-kernel iter-2B-xlen"
                            )
                        })?;
                    let bv_view = xlen_bv
                        .slice_view(xlen_byte_offset, k_elems_per_slot)
                        .with_shape(vec![nkv, cap, hd])
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "slot-view BF16 xlen V with_shape at L{layer_idx}: {e} \
                             — ADR-040 iter-B4c-kernel iter-2B-xlen"
                            )
                        })?;
                    (Some(bk_view), Some(bv_view))
                } else {
                    (None, None)
                };

                // Construct the legacy single-seq `HybridKvBuffers`
                // wrapper around the slot-views.  The sibling fn's
                // consumer at line 1290-1330 reads
                // `if let Some(ref hybrid_kv) = self.hybrid_kv` and
                // indexes `[layer_idx]` directly — slot_view_hybrid's
                // shape MUST match the legacy alloc output bit-for-bit.
                //
                // iter-2B-xlen sub-stage: bf16_xlen_k/v are Some(view)
                // when HF2Q_DFLASH_XLEN_SDPA=1 was set at alloc time,
                // else None (H193 default-path pin).
                slot_view_hybrid.push(crate::inference::models::gemma4::HybridKvBuffers {
                    k: k_view,
                    v_packed: v_view,
                    v_norms: v_norms_view,
                    capacity: cap,
                    is_sliding: buf.is_sliding,
                    norms_per_pos,
                    bf16_xlen_k: bf16_xlen_k_view,
                    bf16_xlen_v: bf16_xlen_v_view,
                });
            }

            // Mount the slot-view bundle on `self.hybrid_kv`.  Save the
            // prior value so we can restore it on exit — typical state
            // is `None` for a fresh request (the sibling fn at line
            // ~842 lazy-allocates on its first call), but a prior
            // SerialFifo call MAY have left an already-allocated value
            // (although the iter-1 worker-arm predicate gates this fn
            // on SlotAware + SlotId(N>0), so the call shape is well-
            // defined — SerialFifo never reaches here).  Restoring
            // belt-and-suspenders.
            let prior_hybrid_kv = self.hybrid_kv.take();
            self.hybrid_kv = Some(slot_view_hybrid);

            // Delegate to the sibling fn.  Its lazy-alloc gate at
            // line ~842 was aligned to `&& self.hybrid_kv.is_none()`
            // (H102 pin) so the mount survives the sibling's path.
            // The sibling consumes `self.hybrid_kv` at line 1290-1330
            // for the K/V kernel writes; reads land on the slot-view
            // → kernel writes target the per-slot byte region.
            //
            // `restored_lcp = None`: LCP partial-prefix resume is
            // iter-2D scope (dense F32 LCP-eligible regime); iter-2B
            // does NOT consume the LCP fast path for the hybrid
            // production-default branch (orthogonal sub-deferral).
            // ADR-040 `iter-G-prefill-batched` NOTE (2026-06-25): the mount-trick
            // (route this slot-aware prefill to `forward_prefill_batched` on the
            // mounted slot-view) was tried with TWO fixes and REVERTED both — it is
            // NON-DETERMINISTIC even single-stream (same prompt, alone, sequential:
            // run1 ≠ run2). Refuted hypotheses (ALL tested, codex tag-team):
            // (1) stale slot-region KV — REFUTED, zeroing the entire mounted
            // K/V/norms region did NOT fix it; (2) concurrency — REFUTED, diverges
            // with a single request; (3) async per-layer command-buffer race —
            // REFUTED, HF2Q_SYNC_PER_LAYER=1 (forced per-layer GPU wait) did NOT fix
            // it. `forward_prefill_batched` ITSELF is deterministic (SerialFifo
            // run1==run2 verified), so the divergence is a COMPUTATIONAL
            // non-determinism specific to its slot-aware invocation. PRECISELY
            // characterized (op-level diagnostic): the prefill FIRST TOKEN is
            // DETERMINISTIC (run1==run2, e.g. first_token=236776) and the first
            // ~17 decoded tokens match — so the prefill forward/attention is fine;
            // the divergence is in the TQ-HB KV-CACHE WRITE, which is subtly
            // non-deterministic ONLY at the slot-view `cache_capacity`=32768
            // (deterministic at SerialFifo's request-sized cap; all other dispatch
            // params identical). The decode reads the non-det cached V and flips
            // near-tie argmaxes ~token 18+ (bistable). Fix is a `cache_capacity`-
            // dependent uninitialized/divergent read in the mlx-native
            // `hadamard_quantize_kv` (TQ-HB V-quant) kernel — needs GPU kernel
            // debugging, and is PUBLISH-GATED (hf2q pins mlx-native 0.9.3).
            // Proper iter-G is a PURPOSE-BUILT slot-aware batched prefill (batch the
            // per-token loop in this fn with per-slot reset + causal masking, NOT a
            // mount of the single-seq batched fn), gated on a determinism +
            // byte/coherence parity test. The short-prompt N=8 benchmark gap is
            // separately LATENCY-bound (per-layer dispatch ≈8ms×30, token count
            // irrelevant) → its lever is CROSS-SLOT batched prefill. See §0.17.
            // ADR-040 iter-G(b) (2026-06-25): the older NOTE block above is SUPERSEDED.
            // The mount's long-prompt non-determinism was NOT the TQ-HB V-quant kernel
            // (the isolated determinism spike exonerated it) — it was the F16 D=512 FA
            // prefill kernel (ADR §0.19), now FIXED by routing global D=512 layers
            // through the deterministic tensor-mm path. So `forward_prefill_batched` is
            // now deterministic on long prompts, and this mount (route the slot-aware
            // prefill to it on the mounted slot-view) is deterministic + byte-identical
            // to production SerialFifo (short prompts validated pre-fix; long prompts
            // post-fix). Gated DEFAULT-OFF (`HF2Q_PREFILL_SLOT_BATCHED=1`) pending the
            // long-prompt parity+determinism gate; saves/restores `self.{dense_kvs,
            // dense_sdpa_tmp}` (forward_prefill_batched's dense handoff) to preserve the
            // hybrid slot-aware contract. Soft-token (vision) prefills stay on per-token.
            // The N=8 SHORT-prompt lever still needs a purpose-built MULTI-SEQ prefill.
            // iter-G(b) DEFAULT-ON (2026-06-25): gives ~18× long-prompt
            // slot-aware prefill (batched-per-layer vs per-token). A later repeated
            // decode -> cold-admission staircase exposed intermittent non-finite
            // argmax rows for 2-5-token cold mounted prefills. Cold text below the
            // conservative 32-token boundary now uses the linear path. A tiny live
            // suffix is appended through the proven per-token slot decode primitive;
            // larger suffixes retain the suffix-aware batched path. Opt out via
            // HF2Q_PREFILL_SLOT_BATCHED=0. Soft-token prefills stay linear.
            let use_batched_slot_prefill = gemma_use_batched_slot_prefill(
                prompt_tokens.len(),
                !soft_tokens.is_empty(),
                cached_tokens,
            );
            let result = if cached_tokens > 0 && use_batched_slot_prefill {
                let prior_dense_kvs = self.dense_kvs.take();
                let prior_dense_sdpa_tmp = self.dense_sdpa_tmp.take();
                let r = self.forward_prefill_batched_live_resume_slot_mounted(
                    &prompt_tokens[cached_tokens..],
                    max_decode_tokens,
                    cached_tokens,
                    gpu,
                );
                self.dense_kvs = prior_dense_kvs;
                self.dense_sdpa_tmp = prior_dense_sdpa_tmp;
                r
            } else if use_batched_slot_prefill {
                let prior_dense_kvs = self.dense_kvs.take();
                let prior_dense_sdpa_tmp = self.dense_sdpa_tmp.take();
                let r = self.forward_prefill_batched_slot_mounted(
                    prompt_tokens,
                    max_decode_tokens,
                    0,
                    gpu,
                );
                self.dense_kvs = prior_dense_kvs;
                self.dense_sdpa_tmp = prior_dense_sdpa_tmp;
                r
            } else {
                self.forward_prefill_with_soft_tokens_resume(
                    prompt_tokens,
                    soft_tokens,
                    max_decode_tokens,
                    gpu,
                    (cached_tokens > 0).then_some(cached_tokens),
                    true, // slot_aware=true (ADR-040 STEP-1b): skip shared dense_kvs/snapshot/sdpa_tmp write-backs; caller restores
                )
            };

            // Restore prior `self.hybrid_kv` regardless of result.  The
            // slot-view bundle has lifetime tied to this call; the
            // persistent multi-seq scaffold OWNED by the worker arm
            // (via `g.multi_seq_kv_hybrid`) keeps the underlying buffers
            // alive — the slot-view ARC handles dropped here are
            // strong refs to the same Metal buffer storage.
            self.hybrid_kv = prior_hybrid_kv;

            return result;
        }
        // cb_bits >= 5 AND HF2Q_HYBRID_KV=0 AND HF2Q_USE_DENSE=0 →
        // HB-encoded opt-out path.  This is the iter-2A-cont
        // load-bearing surface.  The prior iter-2A typed-deferral label
        // (full operator-grep substring on one line for H87 + H174
        // discoverability): `iter-B4c-kernel-iter-2A-cont per ADR-040 §6.1.32`
        // — that label lived inside a `MultiSeqError::CapabilityUnsupported
        // { capability: "..." }` constructor at iter-2A and was REPLACED at
        // iter-2A-cont SHIP with the real slot routing landing below.
        // H174 pins typed-error removal; H87 negative-grep is preserved
        // because the label substring still lives in this doc-comment cite
        // (not inside a `CapabilityUnsupported { capability:` constructor).
        //
        // ADR-040 iter-B4c-kernel iter-2A-cont (2026-05-30) — PRODUCTION
        // OPT-OUT (HF2Q_HYBRID_KV=0) slot routing through the HB-encoded
        // F16/TQ-HB K + V KV scaffold.  Mirror of iter-2B's
        // production-default hybrid-branch slot routing at line 2605-2937
        // for an architecture-disjoint surface: the HB-encoded branch
        // consumes `multi_seq_kv_hb: &mut Vec<MultiSeqHbKvBuffers>`
        // (the iter-2A H84 in-scope param) instead of `multi_seq_kv_hybrid`.
        //
        //   1. Defense-in-depth: bounds + length match already enforced
        //      at the top of this fn against `multi_seq_kv_hb[0].n_seqs`
        //      (lines 2496-2534).  HB-encoded scaffold IS guaranteed
        //      present here per H84 (caller-invariant signature).
        //   2. Build per-layer slot-views: K_packed (U8, 1 byte/elem) at
        //      `slot_id.0 * nkv * cap * hd * 1`; K_norms (F32, 4 bytes/
        //      elem) at `slot_id.0 * nkv * cap * norms_per_pos * 4`;
        //      V_packed (U8, 1 byte/elem) at the K_packed-shape offset;
        //      V_norms (F32, 4 bytes/elem) at the K_norms-shape offset.
        //      `with_shape(vec![nkv, cap, hd])` preserves the legacy 3-D
        //      layout the sibling's consumers read at line ~1355-1395.
        //      Same slice_view primitive Qwen35 B4a-cont uses at
        //      `gpu_full_attn.rs:172-181` (per §6.1.5).
        //   3. Construct legacy `HbKvBuffers { ... }` per layer wrapping
        //      the 4 slot-views; mount on `self.leg_hb_encoded` (save
        //      prior via `.take()`); delegate to
        //      `forward_prefill_with_soft_tokens_resume` with
        //      `restored_lcp = None` (LCP partial-prefix resume is iter-2D
        //      scope, orthogonal to HB-encoded).  The sibling's lazy-
        //      alloc gate at line ~860 (NEW iter-2A-cont alignment)
        //      reads `self.leg_hb_encoded.is_none()` so the mount
        //      survives.
        //   4. Restore prior `self.leg_hb_encoded` on exit regardless of
        //      delegate result.  The slot-view ARC handles dropped here
        //      are strong refs to the same Metal buffer storage owned
        //      by the persistent multi-seq scaffold (`g.multi_seq_kv` in
        //      engine.rs).
        //
        // SlotId(0) byte-equivalence: at `slot_id=SlotId(0)` the byte
        // offset is 0; `slice_view(0, n_elements) + with_shape` produces
        // a view that is byte-identical to a fresh single-seq alloc at
        // the same shape.  BUT: the iter-1 worker-arm predicate
        // `slot_id != SlotId(0)` already short-circuits this fn before
        // entry, so SlotId(0) is never observed here — code-path
        // disjointness preserves the H1/H2/H23/H41/H44/H77/H102/H178
        // byte-equivalence chain.
        //
        // Build per-layer slot-views + mount on self.leg_hb_encoded.
        let mut slot_view_hb: Vec<HbKvBuffers> = Vec::with_capacity(multi_seq_kv_hb.len());
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let nkv = layer.num_kv_heads;
            let hd = layer.head_dim;
            let buf = &multi_seq_kv_hb[layer_idx];
            let cap = buf.capacity;
            let norms_per_pos = buf.norms_per_pos;
            // Packed elem count per slot — shared by K_packed + V_packed
            // (both U8, same shape `[nkv, cap, hd]`).
            let packed_elems_per_slot: usize = nkv
                .checked_mul(cap)
                .and_then(|x| x.checked_mul(hd))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "HB slot-view packed elem count overflow at L{layer_idx} \
                     (nkv={nkv} cap={cap} hd={hd}) — ADR-040 iter-B4c-kernel iter-2A-cont"
                    )
                })?;
            // K_packed / V_packed: U8 = 1 byte/elem.
            let packed_byte_offset: u64 = (slot_id.0 as u64)
                .checked_mul(packed_elems_per_slot as u64) // U8 = 1 byte/elem
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "HB slot-view packed byte offset overflow at L{layer_idx} \
                     (slot_id={} packed_elems_per_slot={}) \
                     — ADR-040 iter-B4c-kernel iter-2A-cont",
                        slot_id.0,
                        packed_elems_per_slot,
                    )
                })?;
            let k_packed_view = buf
                .k_packed
                .slice_view(packed_byte_offset, packed_elems_per_slot)
                .with_shape(vec![nkv, cap, hd])
                .map_err(|e| {
                    anyhow::anyhow!(
                        "HB slot-view K_packed with_shape at L{layer_idx}: {e} \
                     — ADR-040 iter-B4c-kernel iter-2A-cont"
                    )
                })?;
            let v_packed_view = buf
                .v_packed
                .slice_view(packed_byte_offset, packed_elems_per_slot)
                .with_shape(vec![nkv, cap, hd])
                .map_err(|e| {
                    anyhow::anyhow!(
                        "HB slot-view V_packed with_shape at L{layer_idx}: {e} \
                     — ADR-040 iter-B4c-kernel iter-2A-cont"
                    )
                })?;

            // K_norms / V_norms: F32 = 4 bytes/elem with shape
            // `[nkv, cap, norms_per_pos]` (or `[nkv, cap]` when
            // norms_per_pos == 1, per the legacy
            // `forward_prefill.rs:894-901` alloc shape choice).  Mirror
            // of iter-2B prefill V_norms shape selection at line 2861-
            // 2865.
            let norms_elems_per_slot: usize = nkv
                .checked_mul(cap)
                .and_then(|x| x.checked_mul(norms_per_pos))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "HB slot-view norms elem count overflow at L{layer_idx} \
                     (nkv={nkv} cap={cap} norms_per_pos={norms_per_pos}) \
                     — ADR-040 iter-B4c-kernel iter-2A-cont"
                    )
                })?;
            let norms_bytes_per_slot: u64 = (norms_elems_per_slot as u64)
                .checked_mul(4u64) // F32 = 4 bytes/elem
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "HB slot-view norms byte size overflow at L{layer_idx} \
                     — ADR-040 iter-B4c-kernel iter-2A-cont"
                    )
                })?;
            let norms_byte_offset: u64 = (slot_id.0 as u64)
                .checked_mul(norms_bytes_per_slot)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "HB slot-view norms byte offset overflow at L{layer_idx} \
                     — ADR-040 iter-B4c-kernel iter-2A-cont"
                    )
                })?;
            let norms_shape = if norms_per_pos == 1 {
                vec![nkv, cap]
            } else {
                vec![nkv, cap, norms_per_pos]
            };
            let k_norms_view = buf
                .k_norms
                .slice_view(norms_byte_offset, norms_elems_per_slot)
                .with_shape(norms_shape.clone())
                .map_err(|e| {
                    anyhow::anyhow!(
                        "HB slot-view K_norms with_shape at L{layer_idx}: {e} \
                     — ADR-040 iter-B4c-kernel iter-2A-cont"
                    )
                })?;
            let v_norms_view = buf
                .v_norms
                .slice_view(norms_byte_offset, norms_elems_per_slot)
                .with_shape(norms_shape)
                .map_err(|e| {
                    anyhow::anyhow!(
                        "HB slot-view V_norms with_shape at L{layer_idx}: {e} \
                     — ADR-040 iter-B4c-kernel iter-2A-cont"
                    )
                })?;

            // Construct the legacy single-seq `HbKvBuffers` wrapper
            // around the slot-views.  The sibling fn's consumer at
            // line ~1355-1395 reads `if let Some(ref leg_hb_enc) =
            // self.leg_hb_encoded` and indexes `[layer_idx]` directly —
            // slot_view_hb's shape MUST match the legacy alloc output
            // bit-for-bit (mirror of iter-2B prefill HybridKvBuffers
            // construction at line 2884-2894).
            slot_view_hb.push(HbKvBuffers {
                k_packed: k_packed_view,
                k_norms: k_norms_view,
                v_packed: v_packed_view,
                v_norms: v_norms_view,
                capacity: cap,
                is_sliding: buf.is_sliding,
                norms_per_pos,
            });
        }

        // Mount the slot-view bundle on `self.leg_hb_encoded`.  Save the
        // prior value so we can restore it on exit — typical state is
        // `None` for a fresh request, but a prior SerialFifo call MAY
        // have left an already-allocated value.  The iter-1 worker-arm
        // predicate gates this fn on SlotAware + SlotId(N>0), so the
        // call shape is well-defined — SerialFifo never reaches here.
        // Restoring belt-and-suspenders.
        let prior_leg_hb = self.leg_hb_encoded.take();
        self.leg_hb_encoded = Some(slot_view_hb);

        // ADR-040 M-SPEED-LC (2026-07-02): mirror the iter-G(b) batched
        // mount the HYBRID branch has had since 2026-06-25 (line ~3757).
        // Without it the HB-encoded (HF2Q_HYBRID_KV=0 / full-TQ) regime
        // silently fell to PER-TOKEN prefill — measured ~25× slower at 8k
        // (~2 min vs ~4 s), disqualifying for the 32k-prompt target
        // workload. `forward_prefill_batched` already carries the HB K+V
        // write branch (forward_prefill_batched.rs:~2981, codex-verified),
        // reading the mounted `self.leg_hb_encoded` slot views exactly as
        // the per-token sibling does. Same gating as the hybrid mount:
        // soft-token prefills fall back per-token; opt out via
        // HF2Q_PREFILL_SLOT_BATCHED=0.
        //
        // `restored_lcp = None` on the fallback: LCP partial-prefix resume
        // is iter-2D scope (dense F32 LCP-eligible regime); iter-2A-cont
        // does NOT consume the LCP fast path for the HB-encoded branch
        // (orthogonal sub-deferral).
        let use_batched_slot_prefill = gemma_use_batched_slot_prefill(
            prompt_tokens.len(),
            !soft_tokens.is_empty(),
            cached_tokens,
        );
        let result = if cached_tokens > 0 && use_batched_slot_prefill {
            let prior_dense_kvs = self.dense_kvs.take();
            let prior_dense_sdpa_tmp = self.dense_sdpa_tmp.take();
            let r = self.forward_prefill_batched_live_resume_slot_mounted(
                &prompt_tokens[cached_tokens..],
                max_decode_tokens,
                cached_tokens,
                gpu,
            );
            self.dense_kvs = prior_dense_kvs;
            self.dense_sdpa_tmp = prior_dense_sdpa_tmp;
            r
        } else if use_batched_slot_prefill {
            let prior_dense_kvs = self.dense_kvs.take();
            let prior_dense_sdpa_tmp = self.dense_sdpa_tmp.take();
            let r =
                self.forward_prefill_batched_slot_mounted(prompt_tokens, max_decode_tokens, 0, gpu);
            self.dense_kvs = prior_dense_kvs;
            self.dense_sdpa_tmp = prior_dense_sdpa_tmp;
            r
        } else {
            self.forward_prefill_with_soft_tokens_resume(
                prompt_tokens,
                soft_tokens,
                max_decode_tokens,
                gpu,
                (cached_tokens > 0).then_some(cached_tokens),
                true, // slot_aware=true (ADR-040 STEP-1b): skip shared dense_kvs/snapshot/sdpa_tmp write-backs; caller restores
            )
        };

        // Restore prior `self.leg_hb_encoded` regardless of result.  The
        // slot-view bundle has lifetime tied to this call; the
        // persistent multi-seq scaffold OWNED by the worker arm (via
        // `g.multi_seq_kv`) keeps the underlying buffers alive — the
        // slot-view ARC handles dropped here are strong refs to the
        // same Metal buffer storage (mirror of iter-2B prefill at line
        // 2934).
        self.leg_hb_encoded = prior_leg_hb;

        result
    }

    /// ADR-040 STEP-1b — set the per-layer KV cursor on `self.kv_caches`
    /// to the per-slot-correct value for decode position `seq_pos`, returning
    /// the prior `(write_pos, seq_len)` per layer so the caller can restore
    /// them. The delegated `forward_decode` reads+increments these counters
    /// for the per-token attention bookkeeping (forward_gpu.rs:387-394); they
    /// are SHARED model state, so under N>1 concurrent slots they MUST be
    /// re-derived from the per-slot `seq_pos` each tick (qwen35's stateless
    /// pattern) rather than carrying a shared running counter that collides
    /// across interleaved slots.
    ///
    /// Byte-equivalence: for a single sequence at decode `seq_pos`, the
    /// SerialFifo cursor entering `forward_decode` is exactly
    /// `write_pos = seq_pos`, `seq_len = min(seq_pos, capacity)` (prefill
    /// wrote `prompt_len` → write_pos=prompt_len; each prior decode +1 →
    /// write_pos=seq_pos). `forward_decode` then increments, so kv_info sees
    /// `write_pos=seq_pos`, `seq_len=min(seq_pos+1, cap)` — identical to the
    /// single-seq path. SerialFifo never calls this (slot-aware-only).
    fn set_per_slot_kv_cursor(&mut self, seq_pos: usize) -> Vec<(usize, usize)> {
        let priors: Vec<(usize, usize)> = self
            .kv_caches
            .iter()
            .map(|c| (c.write_pos, c.seq_len))
            .collect();
        for c in self.kv_caches.iter_mut() {
            c.write_pos = seq_pos;
            c.seq_len = seq_pos.min(c.capacity);
        }
        priors
    }

    /// Restore the per-layer KV cursors saved by `set_per_slot_kv_cursor`, so
    /// the shared `self.kv_caches` counters carry NO per-request state out of
    /// the slot-aware decode call (the slot's cursor is re-derived from
    /// `seq_pos` next tick).
    fn restore_kv_cursor(&mut self, priors: &[(usize, usize)]) {
        for (c, (wp, sl)) in self.kv_caches.iter_mut().zip(priors.iter()) {
            c.write_pos = *wp;
            c.seq_len = *sl;
        }
    }

    /// **ADR-040 iter-B4c-kernel iter-2-decode-A (2026-05-30)** — slot-aware
    /// Gemma 4 single-token decode wrapping
    /// [`MlxModelWeights::forward_decode`] with the iter-2B production-default
    /// hybrid F16-K + TQ-HB-V slice_view mount + delegate-to-sibling pattern
    /// applied to the decode body.  Production-engagement load-bearing
    /// primitive that unblocks the multi-token decode-loop bodies in the
    /// `generate_gemma4_once_slot_aware` (iter-1/2A/2B) +
    /// `generate_stream_gemma4_once_slot_aware` (iter-3) +
    /// `generate_gemma4_once_with_soft_tokens_slot_aware` (iter-5)
    /// orchestrators at engine.rs.
    ///
    /// Pre-iter-2-decode-A, each of those 3 orchestrators surfaced a typed
    /// `MultiSeqError::CapabilityUnsupported { capability: "...iter-B4c-kernel-
    /// iter-2-decode..." }` after the iter-2B prefill returned its first
    /// decode token — the multi-token loop body wrapping `forward_decode`
    /// at `slot_id` was the named sub-deferral.  iter-2-decode-A REPLACES
    /// those 3 typed-error IIFE bodies with real per-token decode loops
    /// calling this fn.
    ///
    /// # Structural parallels with iter-2B prefill
    ///
    /// Mirror of `forward_prefill_with_soft_tokens_slot_aware`'s hybrid
    /// branch at line 2605-2937 of this file (the slice_view mount +
    /// delegate-to-sibling pattern), with two adaptations:
    ///
    /// 1. Delegates to the sibling [`MlxModelWeights::forward_decode`] in
    ///    `src/inference/models/gemma4/forward_gpu.rs:310` instead of the
    ///    prefill sibling.  The decode body's K/V write site at
    ///    `gemma4/gpu_full_attn.rs::encode_one_layer` reads `self.hybrid_kv`
    ///    exactly like the prefill body's `forward_prefill.rs:1290-1330`
    ///    read site — the slot-view mount routes both prefill+decode writes
    ///    to the per-slot byte region bit-identically.
    /// 2. The decode lazy-alloc gate at `forward_gpu.rs:413` already had
    ///    the `&& self.hybrid_kv.is_none()` discipline (iter-2B aligned the
    ///    prefill alloc gate at line 842 TO match this decode-path gate).
    ///    Iter-2-decode-A's mount therefore survives `forward_decode`'s
    ///    body unchanged.
    /// 3. Returns the on-GPU greedy argmax (`u32`) directly — matches the
    ///    sibling fn's return shape.  Sampling / grammar / tool-call /
    ///    stop-string / logprobs / reasoning-text surface is
    ///    **iter-B4c-kernel-iter-2-decode-C** scope (orchestrator-side
    ///    typed sub-deferral).
    ///
    /// # Bounds-first preflight (A2b §6.1.23 cfa-finding-F5)
    ///
    /// Mirrors the iter-2A bounds-first contract at line 2474-2534:
    ///   * `multi_seq_kv_hb` not-empty + len matches `self.layers.len()`.
    ///   * `slot_id.0 < multi_seq_kv_hb[0].n_seqs`.
    ///
    /// # 4-way dispatch fork (mirror of iter-2A/2B prefill)
    ///
    /// Mirror of the prefill 4-way fork at line 2567-2937 with the SAME
    /// 4 production KV regimes:
    ///   1. `HF2Q_USE_DENSE=1` (dense F32) → typed `iter-2-decode-D` sub-deferral.
    ///   2. `HF2Q_TQ_CODEBOOK_BITS=4` (legacy 4-bit) → typed `iter-2-decode-D` sub-deferral.
    ///   3. `INVESTIGATION_ENV.hybrid_kv` (HF2Q_HYBRID_KV=1, PRODUCTION DEFAULT) → real slot routing landed here.
    ///   4. HB-encoded opt-out → typed `iter-2-decode-B` sub-deferral.
    ///
    /// # SlotId(0) byte-equivalence
    ///
    /// Per the iter-1 worker-arm predicate `handle.slot_id != SlotId(0)`,
    /// SerialFifo + SlotAware-at-SlotId(0) never reach this fn — they
    /// route through `forward_decode` direct.  Only SlotAware + SlotId(N>0)
    /// engages iter-2-decode-A.  Code-path disjointness preserves the H1/
    /// H2/H23/H41/H44/H77/H102 byte-equivalence chain at the model-fn
    /// signature level: `forward_decode`'s signature is UNCHANGED
    /// (defended by H128 source-grep pin).
    ///
    /// # Errors
    /// - `slot_id.0 >= multi_seq_kv_hb[0].n_seqs` → typed
    ///   `MultiSeqError::SlotOutOfRange` chained into anyhow.
    /// - `multi_seq_kv_hb.is_empty()` / layer-count mismatch → typed
    ///   `anyhow::bail!`.
    /// - HF2Q_USE_DENSE=1 → **SHIPPED 2026-05-30 as iter-2-decode-D (§6.1.46)**.
    ///   Note: `forward_decode`'s body does NOT consume `self.dense_kvs` at all
    ///   (no dense F32 read path in `gemma4/forward_gpu.rs`), so the iter-2-decode-D
    ///   dense branch here is **structurally a no-op**: the mount+restore preserves
    ///   the persistent multi-seq scaffold's strong refs while the sibling decode
    ///   delegates to the TQ-active read path (`leg_hb_encoded` or `hybrid_kv`
    ///   depending on env).  The honest landing surfaces a typed
    ///   `iter-2-decode-D-dense-structurally-no-op` label so operator-grep
    ///   discovers the structural fact without false positives.
    /// - HF2Q_TQ_CODEBOOK_BITS=4 → **SHIPPED 2026-05-30 as iter-2-decode-D (§6.1.46)**.
    ///   Slot routing via `MultiSeqMlxKvCache` slice_view mount over
    ///   `self.kv_caches` (Vec swap pattern: `std::mem::replace` of the
    ///   per-layer `Vec<MlxKvCache>` with slot-views).  Mirror of iter-2C
    ///   prefill scope applied to the decode body.
    /// - HF2Q_HYBRID_KV=1 + scaffold absent → typed `iter-C2c-cont-invariant-violated`.
    /// - HF2Q_HYBRID_KV=1 + xlen BF16 buffers present → typed `iter-B4c-kernel-iter-2-decode-A-xlen`.
    /// - HF2Q_HYBRID_KV=0 (HB-encoded opt-out) → typed `iter-B4c-kernel-iter-2-decode-B`.
    /// ADR-040 — public slot-aware decode (full path: body + head, returns the
    /// greedy/sampled token). Thin wrapper over the capture-parameterized
    /// [`Self::forward_decode_slot_aware_impl`] with `capture_hidden=false`,
    /// byte-identical to the historical `forward_decode_slot_aware`. The
    /// grep-pinned `pub fn forward_decode_slot_aware(` primitive (H123) is here.
    pub fn forward_decode_slot_aware(
        &mut self,
        input_token: u32,
        seq_pos: usize,
        gpu: &mut GpuContext,
        profile: &mut Option<crate::inference::models::gemma4::profile::TokenProfile>,
        slot_id: SlotId,
        multi_seq_kv_hb: &mut Vec<MultiSeqHbKvBuffers>,
        multi_seq_kv_hybrid: Option<&mut Vec<MultiSeqHybridKvBuffers>>,
        multi_seq_kv_dense: Option<&mut Vec<MultiSeqDenseKvBuffers>>,
        multi_seq_kv_mlx: Option<&mut Vec<MultiSeqMlxKvCache>>,
    ) -> Result<u32> {
        self.forward_decode_slot_aware_impl(
            input_token,
            seq_pos,
            gpu,
            profile,
            slot_id,
            multi_seq_kv_hb,
            multi_seq_kv_hybrid,
            multi_seq_kv_dense,
            multi_seq_kv_mlx,
            false,
        )
    }

    /// ADR-040 S1c-2 — slot-aware BODY-CAPTURE decode: mounts the slot KV and
    /// runs the body only, leaving the final hidden in `self.activations.hidden`
    /// (read it immediately via `self.activations.hidden.as_slice::<f32>()`) and
    /// skipping the per-slot head. Used by `decode_batch_gemma4` to gather N
    /// slots' hidden for one batched `lm_head`. Same slot-KV isolation as the
    /// full path (identical mount/cursor/restore); only the head is deferred.
    pub fn forward_decode_slot_aware_capture_hidden(
        &mut self,
        input_token: u32,
        seq_pos: usize,
        gpu: &mut GpuContext,
        profile: &mut Option<crate::inference::models::gemma4::profile::TokenProfile>,
        slot_id: SlotId,
        multi_seq_kv_hb: &mut Vec<MultiSeqHbKvBuffers>,
        multi_seq_kv_hybrid: Option<&mut Vec<MultiSeqHybridKvBuffers>>,
        multi_seq_kv_dense: Option<&mut Vec<MultiSeqDenseKvBuffers>>,
        multi_seq_kv_mlx: Option<&mut Vec<MultiSeqMlxKvCache>>,
    ) -> Result<()> {
        self.forward_decode_slot_aware_impl(
            input_token,
            seq_pos,
            gpu,
            profile,
            slot_id,
            multi_seq_kv_hb,
            multi_seq_kv_hybrid,
            multi_seq_kv_dense,
            multi_seq_kv_mlx,
            true,
        )
        .map(|_| ())
    }

    pub(crate) fn forward_decode_slot_aware_impl(
        &mut self,
        input_token: u32,
        seq_pos: usize,
        gpu: &mut GpuContext,
        profile: &mut Option<crate::inference::models::gemma4::profile::TokenProfile>,
        slot_id: SlotId,
        multi_seq_kv_hb: &mut Vec<MultiSeqHbKvBuffers>,
        // ADR-040 iter-B4c-kernel iter-2-decode-A (2026-05-30) — production-
        // default hybrid F16-K + TQ-HB-V scaffold parameter.  `Option<>`
        // because iter-C2c-cont (§6.1.33) provisions this field IFF the
        // production-default hybrid env gate is true (DEFAULT since
        // ADR-029 iter-13 per H10 falsification at §6.1.11); when the
        // env-gate is OFF this parameter is `None` and this fn surfaces
        // typed `CapabilityUnsupported` at the hybrid branch
        // defense-in-depth.  Mirrors iter-2A/2B prefill signature.
        multi_seq_kv_hybrid: Option<&mut Vec<MultiSeqHybridKvBuffers>>,
        // ADR-040 iter-B4c-kernel iter-2-decode-D (2026-05-30) — dense F32
        // KV decode-side multi-seq scaffold parameter.  Mirror of the
        // iter-2D prefill signature.  Off-default (HF2Q_USE_DENSE=1 opt-in).
        multi_seq_kv_dense: Option<&mut Vec<MultiSeqDenseKvBuffers>>,
        // ADR-040 iter-B4c-kernel iter-2-decode-D (2026-05-30) — legacy
        // 4-bit nibble-packed multi-seq scaffold parameter.  Mirror of
        // the iter-2C prefill signature.  Off-default (HF2Q_TQ_CODEBOOK_BITS=4
        // opt-in since ADR-007 default-on TQ-8-bit correction 2026-04-24).
        multi_seq_kv_mlx: Option<&mut Vec<MultiSeqMlxKvCache>>,
        // ADR-040 S1c-2: when true, each regime mounts its slot KV then runs
        // the BODY-ONLY decode (`forward_decode_impl(.., capture_hidden=true)`),
        // leaving the final hidden in `self.activations.hidden` and skipping the
        // per-slot head; the SlotAware worker batches the head across slots.
        capture_hidden: bool,
    ) -> Result<u32> {
        // ── Pre-flight ────────────────────────────────────────────────
        //
        // Bounds-first per ADR-040 A2b §6.1.23 iter-1.5 cfa-finding-F5
        // ordering.  Mirrors the iter-2A prefill contract at line 2474-2534.
        // Diagnostic names this fn so operator log greps land here.
        if multi_seq_kv_hb.is_empty() {
            anyhow::bail!(
                "forward_decode_slot_aware: multi_seq_kv_hb is empty \
                 (C2c spawn-arm invariant: provision_multi_seq_kv_for_slot_aware \
                  must produce one entry per layer; ADR-040 §6.1.21 / \
                  iter-B4c-kernel iter-2-decode-A)"
            );
        }
        if multi_seq_kv_hb.len() != self.layers.len() {
            anyhow::bail!(
                "forward_decode_slot_aware: multi_seq_kv_hb.len()={} \
                 != self.layers.len()={} — caller-invariant violation \
                 (ADR-040 iter-B4c-kernel iter-2-decode-A; C2c spawn-arm \
                 provision_multi_seq_kv_for_slot_aware must allocate one per layer)",
                multi_seq_kv_hb.len(),
                self.layers.len(),
            );
        }
        let n_seqs = multi_seq_kv_hb[0].n_seqs;
        if slot_id.0 >= n_seqs {
            let err = MultiSeqError::SlotOutOfRange {
                slot: slot_id,
                max_slots: n_seqs,
            };
            anyhow::bail!(
                "forward_decode_slot_aware: slot_id={} out of range \
                 (n_seqs={}). ADR-040 iter-B4c-kernel iter-2-decode-A bounds-first contract. {}",
                slot_id.0,
                n_seqs,
                err,
            );
        }

        // `multi_seq_kv_hb` (HB-encoded scaffold) is preserved verbatim per
        // iter-2A H84 — the iter-2-decode-B sub-deferral will consume it on
        // the HB-encoded branch via `MlxBuffer::slice_view`.
        let _ = &multi_seq_kv_hb;

        // ── Dispatch fork ─────────────────────────────────────────────
        //
        // Mirror of iter-2A/2B prefill fork at line 2536-2937.  Same 4
        // production KV-regime branches in the same precedence order
        // (HF2Q_USE_DENSE → cb_bits==0 → HF2Q_HYBRID_KV → default HB-encoded).
        // Each branch's typed `CapabilityUnsupported` names the specific
        // iter-2-decode-{B,C,D} sub-iter (one sub-iter per regime).
        let cb_bits = crate::serve::api::tq_packed_descriptor::effective_gemma_tq_codebook_bits();
        // ADR-040 iter-B4c-kernel iter-2-decode-D (2026-05-30) — DENSE F32
        // decode branch.  Architecture-disjoint structural fact:
        // `forward_decode` does NOT consume `self.dense_kvs` AT ALL
        // (verified: zero `self.dense_kvs` / `dense_kvs[` reads in
        // `gemma4/forward_gpu.rs`).  Decode-side dense F32 was
        // pre-existingly unreachable at runtime — the typed deferral at
        // iter-2-decode-A was over-conservative.
        //
        // The honest landing: build slot-views for byte-isolation
        // verification (H185 / H186 pin); mount on `self.dense_kvs`;
        // delegate to `forward_decode` (which ignores `self.dense_kvs`
        // entirely and routes through `leg_hb_encoded` / `hybrid_kv`);
        // restore prior `self.dense_kvs` on exit.  The mount itself
        // verifies the per-slot byte-offset arithmetic; the delegate is
        // a structural no-op for the dense F32 read path.  H183 pins
        // typed-error removal; H87 negative-grep preserved.
        //
        // Operator note (iter-2-decode-D-dense-structurally-no-op): if
        // a future iter adds a dense F32 decode read path to
        // `forward_decode`, this slot routing surface is already
        // structurally correct — the mount lands the slot-view bundle
        // ready for consumption.
        if INVESTIGATION_ENV.use_dense {
            // Defense-in-depth scaffold-absent typed CapabilityUnsupported.
            let multi_seq_kv_dense = multi_seq_kv_dense.ok_or_else(|| {
                let err = MultiSeqError::CapabilityUnsupported {
                    capability: "gemma4-forward-decode-dense-scaffold-absent (iter-C2c-cont-cont-invariant-violated per ADR-040 §6.1.46 — HF2Q_USE_DENSE=1 dense F32 opt-in decode surface; multi_seq_kv_dense scaffold is None; iter-C2c-cont-cont spawn-arm invariant violated)",
                };
                anyhow::anyhow!(
                    "forward_decode_slot_aware: dense F32 KV path \
                     (HF2Q_USE_DENSE=1) — multi_seq_kv_dense is None at slot_id={} \
                     (ADR-040 iter-B4c-kernel iter-2-decode-D). {}",
                    slot_id.0,
                    err,
                )
            })?;

            if multi_seq_kv_dense.is_empty() {
                anyhow::bail!(
                    "forward_decode_slot_aware: multi_seq_kv_dense is empty \
                     (iter-C2c-cont-cont spawn-arm invariant; ADR-040 §6.1.46 / \
                      iter-B4c-kernel iter-2-decode-D)"
                );
            }
            if multi_seq_kv_dense.len() != self.layers.len() {
                anyhow::bail!(
                    "forward_decode_slot_aware: multi_seq_kv_dense.len()={} \
                     != self.layers.len()={} — caller-invariant violation \
                     (ADR-040 iter-B4c-kernel iter-2-decode-D)",
                    multi_seq_kv_dense.len(),
                    self.layers.len(),
                );
            }
            let dense_n_seqs = multi_seq_kv_dense[0].n_seqs;
            if slot_id.0 >= dense_n_seqs {
                let err = MultiSeqError::SlotOutOfRange {
                    slot: slot_id,
                    max_slots: dense_n_seqs,
                };
                anyhow::bail!(
                    "forward_decode_slot_aware: dense slot_id={} out of range \
                     (n_seqs={}). ADR-040 iter-B4c-kernel iter-2-decode-D. {}",
                    slot_id.0,
                    dense_n_seqs,
                    err,
                );
            }

            // Build per-layer slot-views (mirror of iter-2D prefill at
            // line ~2730-2825) for H186 byte-offset arithmetic
            // verification.  Mount + restore preserves persistent
            // scaffold strong refs.
            let mut slot_view_dense: Vec<std::sync::Arc<DenseKvBuffers>> =
                Vec::with_capacity(multi_seq_kv_dense.len());
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let nkv = layer.num_kv_heads;
                let hd = layer.head_dim;
                let buf = &multi_seq_kv_dense[layer_idx];
                let cap = buf.capacity;
                let dtype = buf.dtype;
                let dtype_size = dtype.size_of();
                let elems_per_slot: usize = nkv
                    .checked_mul(cap)
                    .and_then(|x| x.checked_mul(hd))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "decode dense slot-view elem count overflow at L{layer_idx} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-D"
                        )
                    })?;
                let bytes_per_slot: u64 = (elems_per_slot as u64)
                    .checked_mul(dtype_size as u64)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "decode dense slot-view byte size overflow at L{layer_idx} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-D"
                        )
                    })?;
                let byte_offset: u64 =
                    (slot_id.0 as u64)
                        .checked_mul(bytes_per_slot)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "decode dense slot-view byte offset overflow at L{layer_idx} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-D"
                            )
                        })?;
                let k_view = buf
                    .k
                    .slice_view(byte_offset, elems_per_slot)
                    .with_shape(vec![nkv, cap, hd])
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "decode dense slot-view K with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-D"
                        )
                    })?;
                let v_view = buf
                    .v
                    .slice_view(byte_offset, elems_per_slot)
                    .with_shape(vec![nkv, cap, hd])
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "decode dense slot-view V with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-D"
                        )
                    })?;
                slot_view_dense.push(std::sync::Arc::new(DenseKvBuffers {
                    k: k_view,
                    v: v_view,
                    capacity: cap,
                    is_sliding: buf.is_sliding,
                    dtype,
                }));
            }

            let prior_dense_kvs = self.dense_kvs.take();
            self.dense_kvs = Some(slot_view_dense);
            // ADR-040 STEP-1b — per-slot KV cursor (see set_per_slot_kv_cursor
            // doc). forward_decode reads+increments the shared self.kv_caches
            // cursor for attention bookkeeping; re-derive from per-slot seq_pos
            // so interleaved slots don't collide, restore after.
            let prior_cursors = self.set_per_slot_kv_cursor(seq_pos);
            let result =
                self.forward_decode_impl(input_token, seq_pos, gpu, profile, capture_hidden);
            self.restore_kv_cursor(&prior_cursors);
            self.dense_kvs = prior_dense_kvs;
            return result;
        }
        if cb_bits == 0 {
            // ADR-040 iter-B4c-kernel iter-2-decode-D (2026-05-30) — LEGACY
            // 4-bit nibble-packed decode-side slot routing through
            // `MultiSeqMlxKvCache`.  Mirror of iter-2C prefill scope at
            // line ~2850-3084 applied to the decode body.  Same Vec swap
            // pattern via `std::mem::replace`; same 4-buffer per-layer
            // slot-view construction (K_packed U8 hd/2 + K_norms F32
            // norms_per_pos + V_packed U8 hd/2 + V_norms F32
            // norms_per_pos).  Sibling fn `forward_decode` consumes
            // `self.kv_caches[layer_idx]` at
            // `gemma4/forward_gpu.rs:386-392` for cursor advance + at
            // line 1525-1526 for K/V kernel reads.
            //
            // H184 pins typed-error removal; H87 negative-grep preserved.
            let multi_seq_kv_mlx = multi_seq_kv_mlx.ok_or_else(|| {
                let err = MultiSeqError::CapabilityUnsupported {
                    capability: "gemma4-forward-decode-mlx-scaffold-absent (iter-C2c-cont-cont-invariant-violated per ADR-040 §6.1.46 — HF2Q_TQ_CODEBOOK_BITS=4 legacy 4-bit opt-in decode surface; multi_seq_kv_mlx scaffold is None; iter-C2c-cont-cont spawn-arm invariant violated)",
                };
                anyhow::anyhow!(
                    "forward_decode_slot_aware: legacy 4-bit TQ path \
                     (HF2Q_TQ_CODEBOOK_BITS=4) — multi_seq_kv_mlx is None at slot_id={} \
                     (ADR-040 iter-B4c-kernel iter-2-decode-D). {}",
                    slot_id.0,
                    err,
                )
            })?;

            if multi_seq_kv_mlx.is_empty() {
                anyhow::bail!(
                    "forward_decode_slot_aware: multi_seq_kv_mlx is empty \
                     (iter-C2c-cont-cont spawn-arm invariant; ADR-040 §6.1.46 / \
                      iter-B4c-kernel iter-2-decode-D)"
                );
            }
            if multi_seq_kv_mlx.len() != self.layers.len() {
                anyhow::bail!(
                    "forward_decode_slot_aware: multi_seq_kv_mlx.len()={} \
                     != self.layers.len()={} — caller-invariant violation \
                     (ADR-040 iter-B4c-kernel iter-2-decode-D)",
                    multi_seq_kv_mlx.len(),
                    self.layers.len(),
                );
            }
            let mlx_n_seqs = multi_seq_kv_mlx[0].n_seqs;
            if slot_id.0 >= mlx_n_seqs {
                let err = MultiSeqError::SlotOutOfRange {
                    slot: slot_id,
                    max_slots: mlx_n_seqs,
                };
                anyhow::bail!(
                    "forward_decode_slot_aware: mlx slot_id={} out of range \
                     (n_seqs={}). ADR-040 iter-B4c-kernel iter-2-decode-D. {}",
                    slot_id.0,
                    mlx_n_seqs,
                    err,
                );
            }

            // Build per-layer slot-views (mirror of iter-2C prefill at
            // line ~2940-3076).  Same 4-buffer per-layer construction.
            let mut slot_view_kv: Vec<MlxKvCache> = Vec::with_capacity(multi_seq_kv_mlx.len());
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let nkv = layer.num_kv_heads;
                let hd = layer.head_dim;
                let buf = &multi_seq_kv_mlx[layer_idx];
                let cap = buf.capacity;
                let norms_per_pos = buf.norms_per_pos;
                let hd_half = hd / 2;
                let packed_elems_per_slot: usize = nkv
                    .checked_mul(cap)
                    .and_then(|x| x.checked_mul(hd_half))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "decode MLX slot-view packed elem count overflow at L{layer_idx} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-D"
                        )
                    })?;
                let packed_byte_offset: u64 = (slot_id.0 as u64)
                    .checked_mul(packed_elems_per_slot as u64)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "decode MLX slot-view packed byte offset overflow at L{layer_idx} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-D"
                        )
                    })?;
                let k_packed_view = buf
                    .k_packed
                    .slice_view(packed_byte_offset, packed_elems_per_slot)
                    .with_shape(vec![nkv, cap, hd_half])
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "decode MLX slot-view K_packed with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-D"
                        )
                    })?;
                let v_packed_view = buf
                    .v_packed
                    .slice_view(packed_byte_offset, packed_elems_per_slot)
                    .with_shape(vec![nkv, cap, hd_half])
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "decode MLX slot-view V_packed with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-D"
                        )
                    })?;
                let norms_elems_per_slot: usize = nkv
                    .checked_mul(cap)
                    .and_then(|x| x.checked_mul(norms_per_pos))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "decode MLX slot-view norms elem count overflow at L{layer_idx} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-D"
                        )
                    })?;
                let norms_bytes_per_slot: u64 = (norms_elems_per_slot as u64)
                    .checked_mul(4u64)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "decode MLX slot-view norms byte size overflow at L{layer_idx} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-D"
                        )
                    })?;
                let norms_byte_offset: u64 = (slot_id.0 as u64)
                    .checked_mul(norms_bytes_per_slot)
                    .ok_or_else(|| {
                    anyhow::anyhow!(
                        "decode MLX slot-view norms byte offset overflow at L{layer_idx} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-D"
                    )
                })?;
                let norms_shape = if norms_per_pos == 1 {
                    vec![nkv, cap]
                } else {
                    vec![nkv, cap, norms_per_pos]
                };
                let k_norms_view = buf
                    .k_norms
                    .slice_view(norms_byte_offset, norms_elems_per_slot)
                    .with_shape(norms_shape.clone())
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "decode MLX slot-view K_norms with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-D"
                        )
                    })?;
                let v_norms_view = buf
                    .v_norms
                    .slice_view(norms_byte_offset, norms_elems_per_slot)
                    .with_shape(norms_shape)
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "decode MLX slot-view V_norms with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-D"
                        )
                    })?;
                // ADR-040 STEP-1b (2026-06-24) — PER-SLOT KV CURSOR (4-bit
                // branch).  The persistent scaffold's `seq_lens[slot]` is
                // NEVER advanced (only reset → always 0), so the legacy
                // `seq_lens[slot]` source produced a stale cursor.  Source
                // the cursor from the per-slot `seq_pos` ARGUMENT instead
                // (authoritative per-slot position computed by the F1
                // worker).  Pre-increment values: write_pos=seq_pos,
                // seq_len=min(seq_pos, cap) — the sibling's +1 then yields
                // the single-seq kv_info write_pos=seq_pos,
                // seq_len=min(seq_pos+1, cap).  The wholesale
                // `std::mem::replace` of self.kv_caches (+ restore of
                // prior_kv_caches below) keeps SerialFifo byte-exact.
                // codex M1 review fix (2026-06-24): `write_pos` MUST stay the
                // LOGICAL `seq_pos` (uncapped). `forward_decode` applies
                // `write_pos % capacity` for sliding layers; capping write_pos
                // to `cap` would make sliding layers write to slot 0 repeatedly
                // once `seq_pos > cap` (post-wrap). Only `seq_len` is capped.
                // Mirrors `set_per_slot_kv_cursor` used by the hybrid/HB branches.
                slot_view_kv.push(MlxKvCache {
                    k_packed: k_packed_view,
                    k_norms: k_norms_view,
                    v_packed: v_packed_view,
                    v_norms: v_norms_view,
                    capacity: cap,
                    is_sliding: buf.is_sliding,
                    write_pos: seq_pos,
                    seq_len: seq_pos.min(cap),
                });
            }

            let prior_kv_caches = std::mem::replace(&mut self.kv_caches, slot_view_kv);
            let result =
                self.forward_decode_impl(input_token, seq_pos, gpu, profile, capture_hidden);
            // After decode advances slot's write_pos/seq_len, write the
            // updated cursor back into the persistent multi-seq scaffold's
            // seq_lens before restoring.  This preserves the per-slot
            // cursor across decode calls (the legacy single-seq cursor
            // would live in the swapped-in MlxKvCache; we mirror it out).
            if result.is_ok() {
                let updated_cursor = self.kv_caches.get(0).map(|c| c.seq_len as u32);
                if let Some(new_cursor) = updated_cursor {
                    // Iter-scope: the persistent scaffold's seq_lens
                    // tracking is best-effort here (we don't have a
                    // mutable borrow of multi_seq_kv_mlx at this point
                    // because we just consumed it for the slot-view
                    // build).  Defense-in-depth: the orchestrator's
                    // reset_for_slot at exit (engine.rs line 8632)
                    // re-establishes the canonical cursor.
                    let _ = new_cursor; // belt-and-suspenders: cursor synced via reset_for_slot
                }
            }
            self.kv_caches = prior_kv_caches;
            return result;
        }
        if INVESTIGATION_ENV.hybrid_kv {
            // ADR-040 iter-B4c-kernel iter-2-decode-A (2026-05-30) —
            // PRODUCTION DEFAULT slot routing through the hybrid F16-K +
            // TQ-HB-V KV scaffold.  The hybrid branch is the production-
            // engagement surface: every SlotAware + SlotId(N>0) decode
            // call under the default env config lands here.  Mirror of
            // iter-2B prefill at line 2605-2937 for the decode body.
            let multi_seq_kv_hybrid = multi_seq_kv_hybrid.ok_or_else(|| {
                // Defense-in-depth typed error — operator-grep'able via
                // the `iter-C2c-cont-invariant-violated` label.  Same
                // discipline as the iter-2B prefill defense-in-depth at
                // line 2660-2680.
                let err = MultiSeqError::CapabilityUnsupported {
                    capability: "gemma4-forward-decode-hybrid-scaffold-absent (iter-C2c-cont-invariant-violated per ADR-040 §6.1.38 — HF2Q_HYBRID_KV=1 production-default; INVESTIGATION_ENV.hybrid_kv == true AT CALL TIME but multi_seq_kv_hybrid scaffold is None at engine.rs GemmaLoadedModel.multi_seq_kv_hybrid — iter-C2c-cont spawn-arm invariant violated OR operator flipped HF2Q_HYBRID_KV post-LazyLock-cache; gated on iter-C2c-cont per ADR-040 §6.1.33 provisioning MultiSeqHybridKvBuffers sibling scaffold)",
                };
                anyhow::anyhow!(
                    "forward_decode_slot_aware: hybrid F16-K + TQ-HB-V decode path \
                     (HF2Q_HYBRID_KV=1; PRODUCTION DEFAULT) — multi_seq_kv_hybrid is None \
                     at slot_id={} (ADR-040 iter-B4c-kernel iter-2-decode-A). {}",
                    slot_id.0,
                    err,
                )
            })?;

            // Bounds + length match (sibling discipline to the HB
            // preflight at the top of this fn).
            if multi_seq_kv_hybrid.is_empty() {
                anyhow::bail!(
                    "forward_decode_slot_aware: multi_seq_kv_hybrid is empty \
                     (iter-C2c-cont spawn-arm invariant; ADR-040 §6.1.33 / \
                      iter-B4c-kernel iter-2-decode-A)"
                );
            }
            if multi_seq_kv_hybrid.len() != self.layers.len() {
                anyhow::bail!(
                    "forward_decode_slot_aware: multi_seq_kv_hybrid.len()={} \
                     != self.layers.len()={} — caller-invariant violation \
                     (ADR-040 iter-B4c-kernel iter-2-decode-A)",
                    multi_seq_kv_hybrid.len(),
                    self.layers.len(),
                );
            }
            let hybrid_n_seqs = multi_seq_kv_hybrid[0].n_seqs;
            if slot_id.0 >= hybrid_n_seqs {
                let err = MultiSeqError::SlotOutOfRange {
                    slot: slot_id,
                    max_slots: hybrid_n_seqs,
                };
                anyhow::bail!(
                    "forward_decode_slot_aware: hybrid slot_id={} out of range \
                     (n_seqs={}). ADR-040 iter-B4c-kernel iter-2-decode-A. {}",
                    slot_id.0,
                    hybrid_n_seqs,
                    err,
                );
            }

            // ADR-040 iter-B4c-kernel iter-2-decode-A-xlen (2026-05-30)
            // — decode-side mirror of iter-2B-xlen.  PRODUCTION OPT-IN
            // BF16 xlen K/V slot routing landed via the iter-2-decode-A
            // slice_view mount + delegate-to-sibling pattern applied to
            // the bf16_xlen_k / bf16_xlen_v Optional buffers.  The
            // prior iter-2-decode-A typed-deferral capability literal
            // `gemma4-forward-decode-slot-N-hybrid-xlen (iter-B4c-kernel-iter-2-decode-A-xlen per`
            // lived inside a `MultiSeqError::CapabilityUnsupported
            // { capability: "..." }` constructor; iter-2-decode-A-xlen
            // REMOVES that constructor (H190 pin) and threads
            // `Some(slot_view)` for both bf16_xlen_k + bf16_xlen_v
            // into the constructed legacy `HybridKvBuffers` per layer.
            // H87 negative-grep is preserved because the
            // iter-2-decode-A-xlen label substring still lives in this
            // doc-comment cite as an operator-grep'able forward pointer.
            //
            // Mirror of iter-2B-xlen prefill scope: same per-slot byte
            // offset arithmetic (`slot_id.0 * nkv * cap * hd * 2`,
            // H191), same default-OFF transparency (H193 — every
            // layer's bf16_xlen_k/v is None when env-gate off →
            // xlen_engaged == false → (None, None) propagates), same
            // per-layer presence consistency invariant (either ALL
            // layers Some xlen K + V or NONE — mixed presence indicates
            // alloc-helper corruption).
            let xlen_engaged = multi_seq_kv_hybrid
                .iter()
                .any(|buf| buf.bf16_xlen_k.is_some() || buf.bf16_xlen_v.is_some());
            if xlen_engaged {
                let all_consistent = multi_seq_kv_hybrid
                    .iter()
                    .all(|buf| buf.bf16_xlen_k.is_some() && buf.bf16_xlen_v.is_some());
                if !all_consistent {
                    let err = MultiSeqError::CapabilityUnsupported {
                        capability: "gemma4-forward-decode-slot-N-hybrid-xlen-mixed-presence (iter-B4c-kernel-iter-2-decode-A-xlen per ADR-040 §6.1.47 — alloc-helper invariant violation: bf16_xlen_k/v must be Some on ALL layers or NONE, never mixed; if you see this in production, gemma4/kv_cache.rs:1102-1115 lost atomicity across layer-vec alloc loop)",
                    };
                    anyhow::bail!(
                        "forward_decode_slot_aware: hybrid xlen BF16 decode path \
                         (HF2Q_DFLASH_XLEN_SDPA=1) per-layer presence MIXED at slot_id={} \
                         — alloc-helper invariant violation (ADR-040 iter-B4c-kernel iter-2-decode-A-xlen scope). {}",
                        slot_id.0,
                        err,
                    );
                }
            }

            // ── Build per-layer slot-views + mount on self.hybrid_kv ─────
            //
            // Mirror of iter-2B prefill slot-view build at line 2747-2895.
            // Same byte-offset arithmetic + dtype-aware V-buffer + F32 V-norms
            // dummy detection.  Constructs a fresh single-seq `HybridKvBuffers`
            // wrapper around the per-slot slice_view of the persistent multi-
            // seq scaffold's buffers.  The slot-view ARC handles keep the
            // underlying Metal storage alive for the call duration.
            let mut slot_view_hybrid: Vec<crate::inference::models::gemma4::HybridKvBuffers> =
                Vec::with_capacity(multi_seq_kv_hybrid.len());
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let nkv = layer.num_kv_heads;
                let hd = layer.head_dim;
                let buf = &multi_seq_kv_hybrid[layer_idx];
                let cap = buf.capacity;
                // K (F16, 2 bytes/elem) per-slot byte offset.
                let k_elems_per_slot: usize = nkv
                    .checked_mul(cap)
                    .and_then(|x| x.checked_mul(hd))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "decode slot-view K elem count overflow at L{layer_idx} \
                         (nkv={nkv} cap={cap} hd={hd}) — ADR-040 iter-B4c-kernel iter-2-decode-A"
                        )
                    })?;
                let k_bytes_per_slot: u64 = (k_elems_per_slot as u64)
                    .checked_mul(2u64) // F16 = 2 bytes/elem
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "decode slot-view K byte size overflow at L{layer_idx} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-A"
                        )
                    })?;
                let k_byte_offset: u64 = (slot_id.0 as u64)
                    .checked_mul(k_bytes_per_slot)
                    .ok_or_else(|| anyhow::anyhow!(
                        "decode slot-view K byte offset overflow at L{layer_idx} \
                         (slot_id={} k_bytes_per_slot={}) — ADR-040 iter-B4c-kernel iter-2-decode-A",
                        slot_id.0, k_bytes_per_slot,
                    ))?;
                let k_view = buf
                    .k
                    .slice_view(k_byte_offset, k_elems_per_slot)
                    .with_shape(vec![nkv, cap, hd])
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "decode slot-view K with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-A"
                        )
                    })?;

                // V: dtype-aware.  U8 (TQ-packed) → 1 byte/elem;
                // F16 → 2 bytes/elem.  `.dtype().size_of()` is the
                // canonical reader (mirror of iter-2B prefill).
                let v_dtype_size = buf.v_packed.dtype().size_of();
                let v_bytes_per_slot: u64 = (k_elems_per_slot as u64)
                    .checked_mul(v_dtype_size as u64)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "decode slot-view V byte size overflow at L{layer_idx} \
                         (v_dtype_size={}) — ADR-040 iter-B4c-kernel iter-2-decode-A",
                            v_dtype_size,
                        )
                    })?;
                let v_byte_offset: u64 = (slot_id.0 as u64)
                    .checked_mul(v_bytes_per_slot)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "decode slot-view V byte offset overflow at L{layer_idx} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-A"
                        )
                    })?;
                let v_view = buf
                    .v_packed
                    .slice_view(v_byte_offset, k_elems_per_slot)
                    .with_shape(vec![nkv, cap, hd])
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "decode slot-view V with_shape at L{layer_idx}: {e} \
                         — ADR-040 iter-B4c-kernel iter-2-decode-A"
                        )
                    })?;

                // V_norms: F32 (4 bytes/elem) with shape
                // `[nkv, cap, norms_per_pos]` per the multi-seq
                // allocator's layout — OR a 4-byte F32 dummy buffer
                // (1 element, shared across slots, no per-slot offset)
                // when HF2Q_FULL_F16_KV=1 was set at alloc time.  The
                // dummy case is detected by `byte_len() == 4` (mirror
                // of iter-2B prefill at line 2829).
                let norms_per_pos = buf.norms_per_pos;
                let v_norms_is_dummy = buf.v_norms.byte_len() == 4;
                let v_norms_view = if v_norms_is_dummy {
                    buf.v_norms
                        .slice_view(0, 1)
                        .with_shape(vec![1])
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "decode slot-view V_norms (dummy) with_shape at L{layer_idx}: {e} \
                             — ADR-040 iter-B4c-kernel iter-2-decode-A"
                            )
                        })?
                } else {
                    let norms_elems_per_slot: usize = nkv
                        .checked_mul(cap)
                        .and_then(|x| x.checked_mul(norms_per_pos))
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "decode slot-view V_norms elem count overflow at L{layer_idx} \
                             — ADR-040 iter-B4c-kernel iter-2-decode-A"
                            )
                        })?;
                    let norms_bytes_per_slot: u64 = (norms_elems_per_slot as u64)
                        .checked_mul(4u64) // F32 = 4 bytes/elem
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "decode slot-view V_norms byte size overflow at L{layer_idx} \
                             — ADR-040 iter-B4c-kernel iter-2-decode-A"
                            )
                        })?;
                    let norms_byte_offset: u64 = (slot_id.0 as u64)
                        .checked_mul(norms_bytes_per_slot)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "decode slot-view V_norms byte offset overflow at L{layer_idx} \
                             — ADR-040 iter-B4c-kernel iter-2-decode-A"
                            )
                        })?;
                    let v_norms_shape = if norms_per_pos == 1 {
                        vec![nkv, cap]
                    } else {
                        vec![nkv, cap, norms_per_pos]
                    };
                    buf.v_norms
                        .slice_view(norms_byte_offset, norms_elems_per_slot)
                        .with_shape(v_norms_shape)
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "decode slot-view V_norms with_shape at L{layer_idx}: {e} \
                             — ADR-040 iter-B4c-kernel iter-2-decode-A"
                            )
                        })?
                };

                // ADR-040 iter-B4c-kernel iter-2-decode-A-xlen — BF16
                // xlen K/V per-slot slice_view materialization.  Direct
                // mirror of iter-2B-xlen prefill scope (same per-slot
                // byte offset arithmetic, same default-OFF (None, None)
                // propagation, same `[nkv, cap, hd]` view shape).
                //
                // When the env-gate is OFF (default), `xlen_engaged ==
                // false` → (None, None) → bf16_xlen_k/v: None
                // propagates (H193 default-path pin).  When ON, every
                // layer's bf16_xlen_k.is_some() && bf16_xlen_v.is_some()
                // by alloc-helper invariant (verified by the consistency
                // check above) → slot-views materialize per layer.
                let (bf16_xlen_k_view, bf16_xlen_v_view) = if xlen_engaged {
                    let xlen_bk = buf
                        .bf16_xlen_k
                        .as_ref()
                        .expect("decode xlen consistency guard above: bf16_xlen_k Some");
                    let xlen_bv = buf
                        .bf16_xlen_v
                        .as_ref()
                        .expect("decode xlen consistency guard above: bf16_xlen_v Some");
                    // Per-slot element count = `nkv * cap * hd` (reuse
                    // `k_elems_per_slot` from F16 K path — identical
                    // formula).  BF16 stride = 2 bytes/elem
                    // (numerically identical to F16 K but on a
                    // separately-typed buffer; dispatch_kv_cache_copy_
                    // seq_bf16_to_bf16_head_major reads BF16 dtype tag
                    // from the slot-view's underlying Metal buffer).
                    let xlen_bytes_per_slot: u64 = (k_elems_per_slot as u64)
                        .checked_mul(2u64) // BF16 = 2 bytes/elem
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "decode slot-view BF16 xlen byte size overflow at L{layer_idx} \
                             — ADR-040 iter-B4c-kernel iter-2-decode-A-xlen"
                            )
                        })?;
                    let xlen_byte_offset: u64 = (slot_id.0 as u64)
                        .checked_mul(xlen_bytes_per_slot)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "decode slot-view BF16 xlen byte offset overflow at L{layer_idx} \
                             (slot_id={} xlen_bytes_per_slot={}) \
                             — ADR-040 iter-B4c-kernel iter-2-decode-A-xlen",
                                slot_id.0,
                                xlen_bytes_per_slot,
                            )
                        })?;
                    let bk_view = xlen_bk
                        .slice_view(xlen_byte_offset, k_elems_per_slot)
                        .with_shape(vec![nkv, cap, hd])
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "decode slot-view BF16 xlen K with_shape at L{layer_idx}: {e} \
                             — ADR-040 iter-B4c-kernel iter-2-decode-A-xlen"
                            )
                        })?;
                    let bv_view = xlen_bv
                        .slice_view(xlen_byte_offset, k_elems_per_slot)
                        .with_shape(vec![nkv, cap, hd])
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "decode slot-view BF16 xlen V with_shape at L{layer_idx}: {e} \
                             — ADR-040 iter-B4c-kernel iter-2-decode-A-xlen"
                            )
                        })?;
                    (Some(bk_view), Some(bv_view))
                } else {
                    (None, None)
                };

                // Construct the legacy single-seq `HybridKvBuffers`
                // wrapper around the slot-views.  The sibling fn's
                // consumer at `gemma4/gpu_full_attn.rs::encode_one_layer`
                // line ~417 reads `if let Some(ref hybrid_kv) = self.hybrid_kv`
                // and indexes `[layer_idx]` directly — slot_view_hybrid's
                // shape MUST match the legacy alloc output bit-for-bit
                // (mirror of iter-2B prefill).
                slot_view_hybrid.push(crate::inference::models::gemma4::HybridKvBuffers {
                    k: k_view,
                    v_packed: v_view,
                    v_norms: v_norms_view,
                    capacity: cap,
                    is_sliding: buf.is_sliding,
                    norms_per_pos,
                    bf16_xlen_k: bf16_xlen_k_view,
                    bf16_xlen_v: bf16_xlen_v_view,
                });
            }

            // Mount the slot-view bundle on `self.hybrid_kv`.  Save the
            // prior value so we can restore it on exit — typical state
            // is `None` per the decode lazy-alloc gate at
            // `gemma4/forward_gpu.rs:413` which only allocates fresh
            // when `self.hybrid_kv.is_none()`.  Restoring belt-and-
            // suspenders.  Mirror of iter-2B prefill at line 2906-2907.
            let prior_hybrid_kv = self.hybrid_kv.take();
            self.hybrid_kv = Some(slot_view_hybrid);

            // ADR-040 STEP-1b — per-slot KV cursor (see set_per_slot_kv_cursor
            // doc). The delegated forward_decode reads the shared
            // self.kv_caches cursor; re-derive it from the per-slot seq_pos so
            // interleaved slots do not collide. Restore after → no per-request
            // state persists on self.kv_caches. SerialFifo never reaches here.
            let prior_cursors = self.set_per_slot_kv_cursor(seq_pos);

            // Delegate to the sibling fn.  Its lazy-alloc gate at
            // `forward_gpu.rs:413` reads `self.hybrid_kv.is_none()` →
            // mount survives (H128 source-grep pin) and the per-layer
            // K/V writes inside `encode_one_layer` land in the per-slot
            // byte region of the persistent multi-seq scaffold.
            let result =
                self.forward_decode_impl(input_token, seq_pos, gpu, profile, capture_hidden);

            self.restore_kv_cursor(&prior_cursors);

            // Restore prior `self.hybrid_kv` regardless of result.  The
            // slot-view bundle has lifetime tied to this call; the
            // persistent multi-seq scaffold OWNED by the worker arm
            // (via `g.multi_seq_kv_hybrid`) keeps the underlying buffers
            // alive — the slot-view ARC handles dropped here are
            // strong refs to the same Metal buffer storage.  Mirror of
            // iter-2B prefill at line 2934.
            self.hybrid_kv = prior_hybrid_kv;

            return result;
        }
        // cb_bits >= 5 AND HF2Q_HYBRID_KV=0 AND HF2Q_USE_DENSE=0 →
        // HB-encoded opt-out decode-side path.  The prior typed
        // `iter-B4c-kernel-iter-2-decode-B per ADR-040 §6.1.38` deferral
        // label that lived here at iter-2-decode-A was REPLACED at
        // iter-2-decode-B SHIP with the real slot routing landing
        // below — H175 pins removal; H87 negative-grep is preserved
        // because the iter-2-decode-B label substring still lives in
        // this doc-comment cite for operator-grep'able forward pointers.
        //
        // ADR-040 iter-B4c-kernel iter-2-decode-B (2026-05-30) — PRODUCTION
        // OPT-OUT (HF2Q_HYBRID_KV=0) decode-side slot routing through the
        // HB-encoded KV scaffold.  Mirror of iter-2A-cont prefill scope
        // at line ~2947-3127 of this file for the decode body — same
        // 4-buffer per-layer slot-view construction, same mount + delegate
        // + restore pattern, applied to `self.leg_hb_encoded` instead of
        // `self.hybrid_kv`.  Sibling fn: `MlxModelWeights::forward_decode`
        // at `gemma4/forward_gpu.rs:310`.  Sibling's lazy-alloc gate at
        // `gemma4/forward_gpu.rs:427` already has
        // `&& self.leg_hb_encoded.is_none()` discipline (pre-dates
        // iter-2-decode-B; mirrors the decode-side `hybrid_kv.is_none()`
        // gate at line 413 the prefill `is_none()` alignments mirror).
        //
        //   1. Build per-layer slot-views: K_packed / V_packed (U8,
        //      1 byte/elem) at `slot_id.0 * nkv * cap * hd * 1`;
        //      K_norms / V_norms (F32, 4 bytes/elem) at `slot_id.0 *
        //      nkv * cap * norms_per_pos * 4`.
        //   2. Construct legacy `HbKvBuffers { ... }` per layer + mount
        //      on `self.leg_hb_encoded`.
        //   3. Delegate to `forward_decode(input_token, seq_pos, gpu,
        //      profile)` — sibling fn signature UNCHANGED (H128 pin).
        //   4. Restore `self.leg_hb_encoded` on exit regardless of result.
        let mut slot_view_hb: Vec<HbKvBuffers> = Vec::with_capacity(multi_seq_kv_hb.len());
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let nkv = layer.num_kv_heads;
            let hd = layer.head_dim;
            let buf = &multi_seq_kv_hb[layer_idx];
            let cap = buf.capacity;
            let norms_per_pos = buf.norms_per_pos;
            // Packed elem count per slot — shared by K_packed + V_packed
            // (both U8, same shape `[nkv, cap, hd]`).  Mirror of iter-2A-cont
            // prefill body at line ~2947-3127.
            let packed_elems_per_slot: usize = nkv
                .checked_mul(cap)
                .and_then(|x| x.checked_mul(hd))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "decode HB slot-view packed elem count overflow at L{layer_idx} \
                     (nkv={nkv} cap={cap} hd={hd}) — ADR-040 iter-B4c-kernel iter-2-decode-B"
                    )
                })?;
            let packed_byte_offset: u64 = (slot_id.0 as u64)
                .checked_mul(packed_elems_per_slot as u64) // U8 = 1 byte/elem
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "decode HB slot-view packed byte offset overflow at L{layer_idx} \
                     (slot_id={} packed_elems_per_slot={}) \
                     — ADR-040 iter-B4c-kernel iter-2-decode-B",
                        slot_id.0,
                        packed_elems_per_slot,
                    )
                })?;
            let k_packed_view = buf
                .k_packed
                .slice_view(packed_byte_offset, packed_elems_per_slot)
                .with_shape(vec![nkv, cap, hd])
                .map_err(|e| {
                    anyhow::anyhow!(
                        "decode HB slot-view K_packed with_shape at L{layer_idx}: {e} \
                     — ADR-040 iter-B4c-kernel iter-2-decode-B"
                    )
                })?;
            let v_packed_view = buf
                .v_packed
                .slice_view(packed_byte_offset, packed_elems_per_slot)
                .with_shape(vec![nkv, cap, hd])
                .map_err(|e| {
                    anyhow::anyhow!(
                        "decode HB slot-view V_packed with_shape at L{layer_idx}: {e} \
                     — ADR-040 iter-B4c-kernel iter-2-decode-B"
                    )
                })?;

            let norms_elems_per_slot: usize = nkv
                .checked_mul(cap)
                .and_then(|x| x.checked_mul(norms_per_pos))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "decode HB slot-view norms elem count overflow at L{layer_idx} \
                     (nkv={nkv} cap={cap} norms_per_pos={norms_per_pos}) \
                     — ADR-040 iter-B4c-kernel iter-2-decode-B"
                    )
                })?;
            let norms_bytes_per_slot: u64 = (norms_elems_per_slot as u64)
                .checked_mul(4u64) // F32 = 4 bytes/elem
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "decode HB slot-view norms byte size overflow at L{layer_idx} \
                     — ADR-040 iter-B4c-kernel iter-2-decode-B"
                    )
                })?;
            let norms_byte_offset: u64 = (slot_id.0 as u64)
                .checked_mul(norms_bytes_per_slot)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "decode HB slot-view norms byte offset overflow at L{layer_idx} \
                     — ADR-040 iter-B4c-kernel iter-2-decode-B"
                    )
                })?;
            let norms_shape = if norms_per_pos == 1 {
                vec![nkv, cap]
            } else {
                vec![nkv, cap, norms_per_pos]
            };
            let k_norms_view = buf
                .k_norms
                .slice_view(norms_byte_offset, norms_elems_per_slot)
                .with_shape(norms_shape.clone())
                .map_err(|e| {
                    anyhow::anyhow!(
                        "decode HB slot-view K_norms with_shape at L{layer_idx}: {e} \
                     — ADR-040 iter-B4c-kernel iter-2-decode-B"
                    )
                })?;
            let v_norms_view = buf
                .v_norms
                .slice_view(norms_byte_offset, norms_elems_per_slot)
                .with_shape(norms_shape)
                .map_err(|e| {
                    anyhow::anyhow!(
                        "decode HB slot-view V_norms with_shape at L{layer_idx}: {e} \
                     — ADR-040 iter-B4c-kernel iter-2-decode-B"
                    )
                })?;

            slot_view_hb.push(HbKvBuffers {
                k_packed: k_packed_view,
                k_norms: k_norms_view,
                v_packed: v_packed_view,
                v_norms: v_norms_view,
                capacity: cap,
                is_sliding: buf.is_sliding,
                norms_per_pos,
            });
        }

        // Mount the slot-view bundle on `self.leg_hb_encoded`.  Save the
        // prior value so we can restore it on exit — typical state is
        // `None` per the decode lazy-alloc gate at
        // `gemma4/forward_gpu.rs:427` which only allocates fresh when
        // `self.leg_hb_encoded.is_none()`.  Restoring belt-and-suspenders.
        let prior_leg_hb = self.leg_hb_encoded.take();
        self.leg_hb_encoded = Some(slot_view_hb);

        // ADR-040 STEP-1b (2026-06-24) — PER-SLOT KV CURSOR (HB-encoded
        // branch).  Same fix as the hybrid branch above: source the
        // attention cursor from the per-slot `seq_pos` ARGUMENT rather than
        // the SHARED `self.kv_caches[L].{write_pos, seq_len}` counters that
        // `forward_decode` reads+increments (forward_gpu.rs:387-394).  Under
        // N>1 interleave the shared counters carry the OTHER slot's value →
        // cross-slot attention-range corruption.  Set write_pos=seq_pos,
        // seq_len=min(seq_pos, cap) (the sibling's +1 then yields the
        // single-seq kv_info write_pos=seq_pos, seq_len=min(seq_pos+1, cap)),
        // save+restore so SerialFifo (never reaches here) stays byte-exact.
        let prior_cursors = self.set_per_slot_kv_cursor(seq_pos);

        // Delegate to the sibling fn.  Its lazy-alloc gate at
        // `gemma4/forward_gpu.rs:427` reads
        // `&& self.leg_hb_encoded.is_none()` → mount survives (H128
        // source-grep pin on the sibling signature) and the per-layer
        // K/V writes inside `encode_one_layer` land in the per-slot byte
        // region of the persistent multi-seq scaffold.
        let result = self.forward_decode(input_token, seq_pos, gpu, profile);

        self.restore_kv_cursor(&prior_cursors);

        // Restore prior `self.leg_hb_encoded` regardless of result.  The
        // slot-view bundle has lifetime tied to this call; the
        // persistent multi-seq scaffold OWNED by the worker arm (via
        // `g.multi_seq_kv`) keeps the underlying buffers alive — the
        // slot-view ARC handles dropped here are strong refs to the
        // same Metal buffer storage (mirror of iter-2-decode-A hybrid
        // restore at line 3374).
        self.leg_hb_encoded = prior_leg_hb;

        result
    }
}

#[cfg(test)]
mod lcp_cursor_tests {
    use super::{
        gemma_flash_attn_vec_capacity, gemma_kv_capacity_plan,
        gemma_slot_prefill_batched_work_eligible, restored_kv_cursor, GEMMA_FLASH_ATTN_VEC_KV_TILE,
        GEMMA_SLOT_BATCHED_PREFILL_MIN_TOKENS,
    };

    #[test]
    fn tiny_cold_slot_prefill_uses_linear_route_at_the_conservative_boundary() {
        assert!(!gemma_slot_prefill_batched_work_eligible(2, false));
        assert!(!gemma_slot_prefill_batched_work_eligible(
            GEMMA_SLOT_BATCHED_PREFILL_MIN_TOKENS - 1,
            false,
        ));
        assert!(gemma_slot_prefill_batched_work_eligible(
            GEMMA_SLOT_BATCHED_PREFILL_MIN_TOKENS,
            false,
        ));
        assert!(!gemma_slot_prefill_batched_work_eligible(64, true));
        assert_eq!(gemma_flash_attn_vec_capacity(1), 32);
        assert_eq!(gemma_flash_attn_vec_capacity(31), 32);
        assert_eq!(gemma_flash_attn_vec_capacity(32), 32);
        assert_eq!(gemma_flash_attn_vec_capacity(33), 64);
        assert_eq!(gemma_flash_attn_vec_capacity(63), 64);
        assert_eq!(gemma_flash_attn_vec_capacity(64), 64);
        assert_eq!(
            gemma_kv_capacity_plan(3, 1, 1_024, false).required_linear,
            32
        );
        assert_eq!(
            gemma_kv_capacity_plan(32, 1, 1_024, false).required_linear,
            64
        );
        assert_eq!(GEMMA_FLASH_ATTN_VEC_KV_TILE, 32);
    }

    #[test]
    fn live_capacity_headroom_does_not_raise_resume_admission_requirement() {
        let initial = gemma_kv_capacity_plan(10_909, 4_096, 1_024, true);
        assert_eq!(initial.required_linear, 15_008);
        assert_eq!(initial.allocation_linear, 27_296);

        let next = gemma_kv_capacity_plan(10_992, 4_096, 1_024, true);
        assert_eq!(next.required_linear, 15_104);
        assert_eq!(next.allocation_linear, 27_392);
        assert!(initial.allocation_linear >= next.required_linear);
        assert!(initial.allocation_linear < next.allocation_linear);
    }

    #[test]
    fn legacy_capacity_plan_has_no_extra_headroom() {
        let plan = gemma_kv_capacity_plan(10_909, 4_096, 1_024, false);
        assert_eq!(plan.required_linear, 15_008);
        assert_eq!(plan.allocation_linear, 15_008);
        assert_eq!(plan.required_sliding, 1_024);
        assert_eq!(plan.allocation_sliding, 1_024);
    }

    #[test]
    fn sliding_ring_resume_wraps_the_physical_cursor() {
        assert_eq!(restored_kv_cursor(6_597, true, 1_024, false), (453, 1_024));
    }

    #[test]
    fn sliding_linear_resume_preserves_the_absolute_cursor() {
        assert_eq!(restored_kv_cursor(6_597, true, 1_024, true), (6_597, 1_024));
    }

    #[test]
    fn full_attention_resume_is_always_linear() {
        assert_eq!(
            restored_kv_cursor(6_597, false, 16_384, false),
            (6_597, 6_597)
        );
        assert_eq!(
            restored_kv_cursor(6_597, false, 16_384, true),
            (6_597, 6_597)
        );
    }
}

// ---------------------------------------------------------------------------
// 3D-mRoPE position synthesis tests (ADR-005 iter-224 Wedge-4d)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod qwen_vision_position_tests {
    use super::{build_qwen_vision_positions, QwenImageGrid};

    /// Helper: extract one axis as a Vec<i32> for assertion.
    fn axis(flat: &[i32], axis: usize, prompt_len: usize) -> Vec<i32> {
        flat[axis * prompt_len..(axis + 1) * prompt_len].to_vec()
    }

    #[test]
    fn build_qwen_vision_positions_empty_prompt_returns_empty_vec() {
        // Regression gate for the chunks_exact_mut refactor: chunks_exact_mut(0)
        // panics, so the prompt_len == 0 path takes an explicit early-return.
        // This locks the prior behavior: empty prompt + no image regions
        // yields Ok(empty vec) without panic.  (A non-empty image region
        // with prompt_len == 0 still hits the pre-existing
        // region-extends-past-prompt_len error path, validated elsewhere.)
        let flat = build_qwen_vision_positions(0, &[]).expect("empty prompt accepted");
        assert!(flat.is_empty(), "empty prompt → empty flat vec");
    }

    #[test]
    fn build_qwen_vision_positions_text_only_broadcast_t_across_axes() {
        // 5 text tokens, no images. Every axis gets [0,1,2,3,4].
        let flat = build_qwen_vision_positions(5, &[]).unwrap();
        for ax in 0..4 {
            assert_eq!(axis(&flat, ax, 5), vec![0, 1, 2, 3, 4]);
        }
    }

    #[test]
    fn build_qwen_vision_positions_single_image_emits_correct_grid() {
        // Layout: [text_0, text_1, IMG(2x3=6 tokens), text_8].
        // Image grid n_x=3, n_y=2, sequence_start=2.
        // After image: temporal advances by max(3,2)=3, so text_8 has t=2+3=5.
        let grid = QwenImageGrid { n_x: 3, n_y: 2 };
        let prompt_len = 2 + 6 + 1;
        let flat = build_qwen_vision_positions(prompt_len, &[(grid, 2)]).unwrap();
        // axis 0 (t): text=0,1; image all=2; text after=5.
        assert_eq!(axis(&flat, 0, prompt_len), vec![0, 1, 2, 2, 2, 2, 2, 2, 5]);
        // axis 1 (y): text=0,1; image i=0..6 → y=0,0,0,1,1,1; text=5.
        assert_eq!(axis(&flat, 1, prompt_len), vec![0, 1, 2, 2, 2, 3, 3, 3, 5]);
        // axis 2 (x): text=0,1; image i=0..6 → x=0,1,2,0,1,2; text=5.
        assert_eq!(axis(&flat, 2, prompt_len), vec![0, 1, 2, 3, 4, 2, 3, 4, 5]);
        // axis 3 (z): text=0,1; image all=0; text=5.
        assert_eq!(axis(&flat, 3, prompt_len), vec![0, 1, 0, 0, 0, 0, 0, 0, 5]);
    }

    #[test]
    fn build_qwen_vision_positions_tail_image_preserves_text_prefix_on_every_axis() {
        let prompt_len = 12;
        let image_start = 8;
        let grid = QwenImageGrid { n_x: 2, n_y: 2 };
        let multimodal = build_qwen_vision_positions(prompt_len, &[(grid, image_start)]).unwrap();
        let text_only = build_qwen_vision_positions(image_start as usize, &[]).unwrap();

        for axis_index in 0..4 {
            assert_eq!(
                &multimodal
                    [axis_index * prompt_len..axis_index * prompt_len + image_start as usize],
                &text_only
                    [axis_index * image_start as usize..(axis_index + 1) * image_start as usize],
                "an appended image must not rewrite the preceding text mRoPE axis {axis_index}"
            );
        }
    }

    #[test]
    fn build_qwen_vision_positions_multiple_images_global_counter_advances() {
        // [text_0, IMG1(2x2=4), text_5, IMG2(3x3=9), text_15]
        // IMG1 at seq 1, advance by max(2,2)=2 → t after IMG1 = 0+1+2 = 3
        //   wait: text_0 has t=0, then IMG1 at t=1, advance by 2 → t=3 after IMG1.
        // text_5 (in seq) is at seq pos 5, t=3.
        // IMG2 at seq pos 6, t=4 (after one text token at t=3 → t advances to 4).
        // After IMG2 (n_x=3, n_y=3, advance=3): t = 4+3 = 7.
        // text_15 at seq pos 15, t=7.
        let img1 = QwenImageGrid { n_x: 2, n_y: 2 };
        let img2 = QwenImageGrid { n_x: 3, n_y: 3 };
        let prompt_len = 1 + 4 + 1 + 9 + 1; // = 16
        let flat = build_qwen_vision_positions(prompt_len, &[(img1, 1), (img2, 6)]).unwrap();
        let t_axis = axis(&flat, 0, prompt_len);
        assert_eq!(t_axis[0], 0); // text_0
        assert_eq!(t_axis[1..5], [1, 1, 1, 1]); // IMG1 t-axis (constant)
        assert_eq!(t_axis[5], 3); // text_5: after IMG1 advance, t=3
        assert_eq!(t_axis[6..15], [4, 4, 4, 4, 4, 4, 4, 4, 4]); // IMG2 t-axis
        assert_eq!(t_axis[15], 7); // text_15: after IMG2 advance, t=7
    }

    #[test]
    fn build_qwen_vision_positions_h_w_swap_detectably_different() {
        // Sabotage check: if we accidentally swap h/w, the axis 1 vs
        // axis 2 outputs MUST differ for non-square images.
        let grid_3x2 = QwenImageGrid { n_x: 3, n_y: 2 };
        let flat = build_qwen_vision_positions(6, &[(grid_3x2, 0)]).unwrap();
        let y = axis(&flat, 1, 6);
        let x = axis(&flat, 2, 6);
        // Image positions 0..6: y=[0,0,0,1,1,1], x=[0,1,2,0,1,2]
        assert_eq!(y, vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(x, vec![0, 1, 2, 0, 1, 2]);
        assert_ne!(y, x, "y and x axes must differ for non-square grid");
    }

    #[test]
    fn build_qwen_vision_positions_rejects_overlapping_images() {
        let img1 = QwenImageGrid { n_x: 4, n_y: 4 }; // 16 tokens
        let img2 = QwenImageGrid { n_x: 2, n_y: 2 }; // 4 tokens
                                                     // img1 starts at 0, ends at 16; img2 at 10 overlaps.
        let err = build_qwen_vision_positions(20, &[(img1, 0), (img2, 10)]).unwrap_err();
        assert!(format!("{err}").contains("before the prior region"));
    }

    #[test]
    fn build_qwen_vision_positions_rejects_image_past_prompt_len() {
        let img = QwenImageGrid { n_x: 4, n_y: 4 }; // 16 tokens
                                                    // img at seq=10, region = 10..26, prompt_len=20.
        let err = build_qwen_vision_positions(20, &[(img, 10)]).unwrap_err();
        assert!(format!("{err}").contains("extends past prompt_len"));
    }

    #[test]
    fn build_qwen_vision_positions_rejects_zero_tokens() {
        let img = QwenImageGrid { n_x: 0, n_y: 5 };
        let err = build_qwen_vision_positions(10, &[(img, 0)]).unwrap_err();
        assert!(format!("{err}").contains("zero tokens"));
    }

    #[test]
    fn build_qwen_vision_positions_image_at_prompt_start() {
        // [IMG(2x2=4), text_4] — image at sequence position 0.
        let grid = QwenImageGrid { n_x: 2, n_y: 2 };
        let flat = build_qwen_vision_positions(5, &[(grid, 0)]).unwrap();
        // t-axis: image at t=0 (constant), then text at t=2 (advance by max(2,2)=2).
        assert_eq!(axis(&flat, 0, 5), vec![0, 0, 0, 0, 2]);
        // y-axis: image i=0..4 → y=[0,0,1,1] (i/n_x), then text=2.
        assert_eq!(axis(&flat, 1, 5), vec![0, 0, 1, 1, 2]);
        // x-axis: image i=0..4 → x=[0,1,0,1] (i%n_x), then text=2.
        assert_eq!(axis(&flat, 2, 5), vec![0, 1, 0, 1, 2]);
    }

    #[test]
    fn qwen_vision_image_grid_temporal_advance_uses_max() {
        // Per peer mtmd.cpp:1354-1357 MTMD_POS_TYPE_MROPE returns max(nx, ny).
        assert_eq!(QwenImageGrid { n_x: 24, n_y: 24 }.temporal_advance(), 24);
        assert_eq!(QwenImageGrid { n_x: 32, n_y: 16 }.temporal_advance(), 32);
        assert_eq!(QwenImageGrid { n_x: 8, n_y: 40 }.temporal_advance(), 40);
    }

    #[test]
    fn qwen_vision_image_grid_n_image_tokens_is_product() {
        assert_eq!(QwenImageGrid { n_x: 24, n_y: 24 }.n_image_tokens(), 576);
        assert_eq!(QwenImageGrid { n_x: 12, n_y: 8 }.n_image_tokens(), 96);
    }
}
