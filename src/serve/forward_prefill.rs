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
use mlx_native::MlxBuffer;
use mlx_native::ops::flash_attn_vec::FlashAttnVecParams;
use mlx_native::ops::dense_gemm::DenseGemmF16Params;
use std::ops::Range;
use std::time::Instant;

use crate::debug::INVESTIGATION_ENV;

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
/// Mirrors /opt/llama.cpp/src/models/qwen3vl.cpp:96-100's per-layer
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
/// ViT side (vit_gpu_qwen3vl.rs's augmented embed) and the LM side
/// (forward_gpu.rs's image_token_residual_add hook). The handler-side
/// path that constructs it from `compute_vision_embeddings_gpu_qwen3vl`'s
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
pub struct Qwen3VlImageGrid {
    /// Post-merge token-grid width (X axis). For canonical Qwen3-VL
    /// preprocessor at `image_size=768, patch_size=16,
    /// spatial_merge_size=2`: `n_x = 24`.
    pub n_x: u32,
    /// Post-merge token-grid height (Y axis).
    pub n_y: u32,
}

impl Qwen3VlImageGrid {
    /// `n_image_tokens` = `n_x * n_y`.
    pub fn n_image_tokens(&self) -> u32 {
        self.n_x.saturating_mul(self.n_y)
    }

    /// Per peer `mtmd_image_tokens_get_n_pos` at
    /// `/opt/llama.cpp/tools/mtmd/mtmd.cpp:1354-1357` —
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
/// Implements peer's MROPE position assignment for Qwen3-VL combined
/// with the temporal-advance rule observed at
/// `/opt/llama.cpp/tools/mtmd/mtmd.cpp:1295-1304`
/// (`mtmd_image_tokens_get_decoder_pos` for `MTMD_POS_TYPE_MROPE`) and
/// `/opt/llama.cpp/tools/mtmd/mtmd-helper.cpp:166-181`
/// (`set_position_mrope_2d` writing `[t, y, x, z]` axes column-major).
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
pub fn build_qwen3vl_positions(
    prompt_len: usize,
    image_grids: &[(Qwen3VlImageGrid, u32)],
) -> anyhow::Result<Vec<i32>> {
    use anyhow::anyhow;

    // Validate ordering + non-overlap + bounds.
    let mut last_end: u32 = 0;
    for (i, (grid, seq_start)) in image_grids.iter().enumerate() {
        let n_tokens = grid.n_image_tokens();
        if n_tokens == 0 {
            return Err(anyhow!(
                "build_qwen3vl_positions: image[{i}] has zero tokens \
                 (n_x={}, n_y={})",
                grid.n_x,
                grid.n_y
            ));
        }
        if (*seq_start) < last_end {
            return Err(anyhow!(
                "build_qwen3vl_positions: image[{i}] starts at {seq_start} \
                 which is before the prior region's end {last_end} — \
                 overlapping or unsorted image regions"
            ));
        }
        let region_end = (*seq_start)
            .checked_add(n_tokens)
            .ok_or_else(|| anyhow!("build_qwen3vl_positions: image[{i}] region overflow"))?;
        if (region_end as usize) > prompt_len {
            return Err(anyhow!(
                "build_qwen3vl_positions: image[{i}] region {seq_start}..{region_end} \
                 extends past prompt_len {prompt_len}"
            ));
        }
        last_end = region_end;
    }

    let mut flat = vec![0i32; 4 * prompt_len];
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
                flat[0 * prompt_len + q] = t_img;                 // t (constant)
                flat[1 * prompt_len + q] = t_img + (i_i32 / n_x); // y
                flat[2 * prompt_len + q] = t_img + (i_i32 % n_x); // x
                flat[3 * prompt_len + q] = 0;                     // z
            }
            t_global += grid.temporal_advance() as i32;
            p += n_tokens;
            img_idx += 1;
        } else {
            // Text token.
            flat[0 * prompt_len + p] = t_global;
            flat[1 * prompt_len + p] = t_global;
            flat[2 * prompt_len + p] = t_global;
            flat[3 * prompt_len + p] = t_global;
            t_global += 1;
            p += 1;
        }
    }
    Ok(flat)
}
use super::forward_mlx::{
    MlxModelWeights, DenseKvBuffers, HbKvBuffers, dispatch_qmatmul,
    dispatch_rms_norm_unit_perhead, RmsNormPerHeadArgs,
};
use super::config::LayerType;
use super::gpu::GpuContext;

/// Helper: dump an F32 MlxBuffer's first `n_elems` to a file at dump_dir.
fn write_dump_f32(
    dump_dir: &str,
    name: &str,
    layer: usize,
    tok: usize,
    buf: &MlxBuffer,
    n_elems: usize,
) -> Result<()> {
    let data: &[f32] = buf.as_slice()
        .map_err(|e| anyhow::anyhow!("dump {name} L{layer} T{tok}: {e}"))?;
    let path = format!("{dump_dir}/hf2q_prefill_{name}_layer{layer:02}_tok{tok:03}.bin");
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8,
            n_elems * std::mem::size_of::<f32>())
    };
    std::fs::write(&path, bytes)
        .map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
    eprintln!("[PREFILL DUMP] {} L{} T{} ({} f32) -> {}", name, layer, tok, n_elems, path);
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
        let seq_len = prompt_tokens.len();
        if seq_len == 0 {
            anyhow::bail!("forward_prefill: empty prompt");
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
                    i, st.range
                );
            }
            let needed_bytes = st.range.len() * hs * 4;
            if st.embeddings.byte_len() < needed_bytes {
                anyhow::bail!(
                    "forward_prefill: soft_tokens[{}].embeddings byte_len={} < required {} \
                     ({} positions × {} hidden × 4 bytes)",
                    i, st.embeddings.byte_len(), needed_bytes, st.range.len(), hs
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
                        i, a, j, b
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
        // When prompt-cache lands (Phase 2a Task #7) it'll need to revisit
        // this and reset only positions past the cached prefix length.
        for cache in self.kv_caches.iter_mut() {
            cache.write_pos = 0;
            cache.seq_len = 0;
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
        //   - LayerType::Global (linear): capacity = seq_len + max_decode_tokens.
        //     Writes are monotonically increasing.
        // The dense flash_attn_vec dispatch at lines 960-981 uses
        // `mask_type=1` (pure causal) for both layer types. Ring
        // correctness rests on attention being permutation-invariant
        // over the K,V set: once the ring wraps, the oldest slot is
        // overwritten with the newest token, but which physical slot
        // is oldest is immaterial — the causal mask within the
        // sliding window still yields the correct attention pattern.
        // ===================================================================
        // ADR-009 Phase 3A finding: matching llama.cpp's F16 KV cache
        // REGRESSED our parity (sourdough 3656→3095, sliding_wrap 752→627).
        // llama.cpp itself is insensitive to KV dtype (its F16 and F32 outputs
        // are byte-identical). Our F16 path has a separate bug worse than F32.
        // F32 remains the default; F16 is opt-in via HF2Q_F16_KV=1 for the
        // follow-up investigation into the F16-specific regression.
        let use_f16_kv = INVESTIGATION_ENV.f16_kv;
        let kv_dtype = if use_f16_kv { mlx_native::DType::F16 } else { mlx_native::DType::F32 };
        let kv_elem_bytes = if use_f16_kv { 2 } else { 4 };
        tracing::debug!("Prefill: KV cache dtype = {:?}", kv_dtype);

        // Per-layer capacity:
        //   - Sliding (ring): sliding_window. Writes wrap at seq_pos % capacity.
        //     Attention is permutation-invariant over cached K,V, so slot
        //     order doesn't affect correctness. Dense flash_attn_vec reads
        //     the populated slots with a pure causal mask.
        //   - Global (linear): seq_len + max_decode_tokens. Writes are monotonic.
        // Ring buffer for sliding drops ~5 GB of dense KV at 20k decode on
        // Gemma-4 26B (8×1024×256 per layer vs 8×20022×256).
        let linear_capacity = seq_len + max_decode_tokens;
        let sw = self.sliding_window;
        let mut dense_kvs_vec: Vec<DenseKvBuffers> = Vec::with_capacity(num_layers);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let nkv = layer.num_kv_heads;
            let hd = layer.head_dim;
            let layer_is_ring = layer.layer_type == LayerType::Sliding;
            let capacity = if layer_is_ring { sw } else { linear_capacity };
            let n = nkv * capacity * hd;
            let k = dev.alloc_buffer(n * kv_elem_bytes, kv_dtype, vec![nkv, capacity, hd])
                .map_err(|e| anyhow::anyhow!("prefill dense K L{layer_idx}: {e}"))?;
            let v = dev.alloc_buffer(n * kv_elem_bytes, kv_dtype, vec![nkv, capacity, hd])
                .map_err(|e| anyhow::anyhow!("prefill dense V L{layer_idx}: {e}"))?;
            dense_kvs_vec.push(DenseKvBuffers { k, v, capacity, is_sliding: layer_is_ring });
        }

        // Tmp buffer for flash_attn_vec (sized for largest layer config)
        let max_nh = self.num_attention_heads;
        let max_hd = self.layers.iter().map(|l| l.head_dim).max().unwrap_or(512);
        let tmp_bytes = mlx_native::ops::flash_attn_vec::tmp_buffer_bytes(
            max_nh as u32, max_hd as u32);
        let sdpa_tmp = dev.alloc_buffer(tmp_bytes, mlx_native::DType::F32,
            vec![tmp_bytes / 4])
            .map_err(|e| anyhow::anyhow!("prefill sdpa_tmp: {e}"))?;

        tracing::debug!("Prefill: {} tokens × {} layers (dense SDPA)", seq_len, num_layers);

        // iter-222 (ADR-005 closure, 2026-05-01): the iter-21 Track A
        // `leg_f_kvs` shadow-cache allocation block (~30 LOC) was deleted
        // along with the iter-34 dense-on-shadow Leg F decode branch — see
        // file-level iter-222 closure note in `forward_mlx.rs` for rationale.
        // Production TQ-regime SDPA reads `kv_caches[].{k,v}_packed` /
        // `leg_hb_encoded` directly via inline-fused kernels.

        // iter-21 Track B + 2026-04-24 post-close default correction.
        // HF2Q_TQ_CODEBOOK_BITS=5|6|8 (or unset) → allocate per-layer byte-packed HB buffers.
        // MUST stay in lockstep with forward_mlx.rs::tq_codebook_bits and cb_bits gates.
        //   unset (DEFAULT) = 8-bit native HB SDPA
        //   "4"             = legacy 4-bit (no HB buffers)
        //   "5" | "6" | "8" = corresponding HB bits
        let tq_codebook_bits_prefill: u32 = match std::env::var("HF2Q_TQ_CODEBOOK_BITS").as_deref() {
            Ok("4") => 0,
            Ok("5") => 5, Ok("6") => 6, Ok("8") => 8,
            _ => 8,  // DEFAULT: 8-bit
        };
        if tq_codebook_bits_prefill >= 5 {
            eprintln!("[iter-21 Track B] Allocating leg_hb_encoded ({}-bit, {} layers)",
                      tq_codebook_bits_prefill, num_layers);
            let mut leg_hb_vec: Vec<HbKvBuffers> = Vec::with_capacity(num_layers);
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let nkv = layer.num_kv_heads;
                let hd = layer.head_dim;
                let layer_is_ring = layer.layer_type == LayerType::Sliding;
                let capacity = if layer_is_ring { sw } else { linear_capacity };
                let norms_per_pos = (hd / 256).max(1);
                let norms_n = nkv * capacity * norms_per_pos;
                let k_packed = dev.alloc_buffer(nkv * capacity * hd, mlx_native::DType::U8,
                    vec![nkv, capacity, hd])
                    .map_err(|e| anyhow::anyhow!("leg_hb prefill K packed L{layer_idx}: {e}"))?;
                let k_norms = dev.alloc_buffer(norms_n * 4, mlx_native::DType::F32,
                    if norms_per_pos == 1 { vec![nkv, capacity] } else { vec![nkv, capacity, norms_per_pos] })
                    .map_err(|e| anyhow::anyhow!("leg_hb prefill K norms L{layer_idx}: {e}"))?;
                let v_packed = dev.alloc_buffer(nkv * capacity * hd, mlx_native::DType::U8,
                    vec![nkv, capacity, hd])
                    .map_err(|e| anyhow::anyhow!("leg_hb prefill V packed L{layer_idx}: {e}"))?;
                let v_norms = dev.alloc_buffer(norms_n * 4, mlx_native::DType::F32,
                    if norms_per_pos == 1 { vec![nkv, capacity] } else { vec![nkv, capacity, norms_per_pos] })
                    .map_err(|e| anyhow::anyhow!("leg_hb prefill V norms L{layer_idx}: {e}"))?;
                leg_hb_vec.push(HbKvBuffers {
                    k_packed, k_norms, v_packed, v_norms,
                    capacity, is_sliding: layer_is_ring, norms_per_pos,
                });
            }
            self.leg_hb_encoded = Some(leg_hb_vec);

            // iter-222 (ADR-005 closure, 2026-05-01): the iter-21 Track B
            // `leg_f_kvs` shadow-cache allocation block (~30 LOC) was deleted
            // along with the iter-34 dense-on-shadow Leg F decode branch —
            // see file-level iter-222 closure note in `forward_mlx.rs`.
            // `flash_attn_vec_tq_hb` reads `leg_hb_encoded` directly with no
            // F32 round-trip.
            eprintln!("[iter-21 Track B] leg_hb_encoded ready ({} layers)", num_layers);
        }

        // ADR-010 one-shot norm weight dump: read self.layers[L].norms.input_layernorm
        // as the hf2q kernel sees it, compare against the raw GGUF tensor.
        // Gated on HF2Q_DUMP_NORM_WEIGHT="layer" (e.g. "7"). Writes to HF2Q_DUMP_DIR.
        if let Some(target_l) = INVESTIGATION_ENV.dump_norm_weight {
            if target_l < num_layers {
                let w: &[f32] = self.layers[target_l].norms.input_layernorm.as_slice()
                    .map_err(|e| anyhow::anyhow!("norm weight read L{target_l}: {e}"))?;
                let dir = &INVESTIGATION_ENV.dump_dir;
                let path = format!("{dir}/hf2q_input_layernorm_weight_layer{target_l:02}.bin");
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) };
                std::fs::write(&path, bytes)
                    .map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
                eprintln!("[DUMP] input_layernorm weight L{target_l} [{}] f32 -> {}",
                          w.len(), path);
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
        let prefill_start = Instant::now();
        let mut last_token = 0u32;

        for (tok_i, &tok) in prompt_tokens.iter().enumerate() {
            let seq_pos = tok_i;

            // Write position buffer
            {
                let pos_dst: &mut [u32] = self.activations.position.as_mut_slice()
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
                self.kv_caches[layer_idx].seq_len = self.kv_caches[layer_idx].seq_len
                    .saturating_add(1).min(capacity);
                let kv_seq_len = self.kv_caches[layer_idx].seq_len;
                kv_info.push((is_sliding, write_pos, capacity, kv_seq_len));
            }

            // ===============================================================
            // Single GPU session per token (same structure as forward_decode)
            // ===============================================================
            {
                let mut s = exec.begin()
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
                let soft_override = soft_tokens
                    .iter()
                    .find(|st| st.range.contains(&tok_i));
                if let Some(st) = soft_override {
                    let row_idx = tok_i - st.range.start;
                    let src_offset = row_idx * hs;
                    mlx_native::ops::copy::dispatch_copy_f32(
                        s.encoder_mut(), reg, metal_dev,
                        st.embeddings,
                        &self.activations.hidden,
                        src_offset,
                        0,
                        hs,
                    ).map_err(|e| anyhow::anyhow!(
                        "prefill soft-token copy T{tok_i} (range {:?}, row {}): {e}",
                        st.range, row_idx
                    ))?;
                    s.track_dispatch(&[st.embeddings], &[&self.activations.hidden]);
                } else {
                    mlx_native::ops::elementwise::embedding_gather_scale_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.embed_weight,
                        &self.activations.hidden,
                        tok, hs,
                        (hs as f32).sqrt(),
                    ).map_err(|e| anyhow::anyhow!("prefill embed T{tok_i}: {e}"))?;
                    s.track_dispatch(&[&self.embed_weight], &[&self.activations.hidden]);
                }

                // --- 2. Transformer layers ---
                for layer_idx in 0..num_layers {
                    let layer = &self.layers[layer_idx];
                    let hd = layer.head_dim;
                    let nkv = layer.num_kv_heads;
                    let nh = self.num_attention_heads;
                    let is_sliding = layer.layer_type == LayerType::Sliding;
                    let (kv_is_sliding, kv_write_pos, kv_capacity, _kv_seq_len) = kv_info[layer_idx];

                    // Active dump flag for this iteration
                    let dump_here = prefill_dump == Some((layer_idx, tok_i));
                    // Dump at layer-start: hidden = L(layer_idx-1) l_out (or embed for L0)
                    if dump_here {
                        s.finish()
                            .map_err(|e| anyhow::anyhow!("prefill dump L{layer_idx} T{tok_i} start finish: {e}"))?;
                        write_dump_f32(dump_dir, "pre_layer_hidden", layer_idx, tok_i,
                                        &self.activations.hidden, hs)?;
                        s = exec.begin()
                            .map_err(|e| anyhow::anyhow!("prefill dump restart: {e}"))?;
                    }

                    // -- Pre-attention norm --
                    s.barrier_between(
                        &[&self.activations.hidden, &self.layers[layer_idx].norms.input_layernorm],
                        &[&self.activations.norm_out],
                    );
                    s.rms_norm(
                        reg, metal_dev,
                        &self.activations.hidden,
                        &self.layers[layer_idx].norms.input_layernorm,
                        &self.activations.norm_out,
                        &self.activations.norm_params,
                        1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("prefill norm L{layer_idx} T{tok_i}: {e}"))?;

                    if dump_here {
                        s.finish().map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(dump_dir, "post_input_norm", layer_idx, tok_i,
                                        &self.activations.norm_out, hs)?;
                        s = exec.begin().map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // -- QKV projections (concurrent) --
                    s.barrier_between(
                        &[&self.activations.norm_out],
                        &[&self.activations.attn_q, &self.activations.attn_k, &self.activations.attn_v],
                    );
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].attn.q_proj, &mut self.activations.attn_q, 1)?;
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].attn.k_proj, &mut self.activations.attn_k, 1)?;
                    let v_is_k = self.layers[layer_idx].attn.v_proj.is_none();
                    if !v_is_k {
                        dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                            self.layers[layer_idx].attn.v_proj.as_ref().unwrap(),
                            &mut self.activations.attn_v, 1)?;
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
                        &[&self.activations.attn_q_normed, &self.activations.attn_k_normed],
                    );
                    mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_q,
                        &self.activations.attn_q_normed,
                        Some(&self.layers[layer_idx].attn.q_norm_weight),
                        &self.activations.position,
                        ff_gpu,
                        nh as u32, hd as u32, half_rope,
                        eps, theta,
                    ).map_err(|e| anyhow::anyhow!("prefill Q norm+RoPE L{layer_idx} T{tok_i}: {e}"))?;
                    mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_k,
                        &self.activations.attn_k_normed,
                        Some(&self.layers[layer_idx].attn.k_norm_weight),
                        &self.activations.position,
                        ff_gpu,
                        nkv as u32, hd as u32, half_rope,
                        eps, theta,
                    ).map_err(|e| anyhow::anyhow!("prefill K norm+RoPE L{layer_idx} T{tok_i}: {e}"))?;

                    if dump_here {
                        s.finish().map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(dump_dir, "q_pre_normed", layer_idx, tok_i,
                                        &self.activations.attn_q, nh * hd)?;
                        write_dump_f32(dump_dir, "k_pre_normed", layer_idx, tok_i,
                                        &self.activations.attn_k, nkv * hd)?;
                        write_dump_f32(dump_dir, "q_normed", layer_idx, tok_i,
                                        &self.activations.attn_q_normed, nh * hd)?;
                        write_dump_f32(dump_dir, "k_normed", layer_idx, tok_i,
                                        &self.activations.attn_k_normed, nkv * hd)?;
                        s = exec.begin().map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // -- V norm --
                    let hd_norm_params = if is_sliding {
                        &self.activations.norm_params_sliding_hd
                    } else {
                        &self.activations.norm_params_global_hd
                    };
                    if v_is_k {
                        s.barrier_between(
                            &[&self.activations.attn_k],
                            &[&self.activations.attn_v],
                        );
                        dispatch_rms_norm_unit_perhead(
                            s.encoder_mut(), reg, metal_dev,
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
                            s.encoder_mut(), reg, metal_dev,
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
                    let layer_dense_cap = dense_kvs_vec[layer_idx].capacity;
                    let layer_is_ring = dense_kvs_vec[layer_idx].is_sliding;
                    let write_slot = if layer_is_ring {
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
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &dense_kvs_vec[layer_idx].k,
                            nkv as u32, hd as u32,
                            layer_dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("prefill F16 K copy L{layer_idx} T{tok_i}: {e}"))?;
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16(
                            s.encoder_mut(), reg, metal_dev,
                            v_src,
                            &dense_kvs_vec[layer_idx].v,
                            nkv as u32, hd as u32,
                            layer_dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("prefill F16 V copy L{layer_idx} T{tok_i}: {e}"))?;
                    } else {
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &dense_kvs_vec[layer_idx].k,
                            nkv as u32, hd as u32,
                            layer_dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("prefill F32 K batch copy L{layer_idx} T{tok_i}: {e}"))?;
                        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
                            s.encoder_mut(), reg, metal_dev,
                            v_src,
                            &dense_kvs_vec[layer_idx].v,
                            nkv as u32, hd as u32,
                            layer_dense_cap as u32, write_slot,
                        ).map_err(|e| anyhow::anyhow!("prefill F32 V batch copy L{layer_idx} T{tok_i}: {e}"))?;
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
                            &[&self.kv_caches[layer_idx].k_packed, &self.kv_caches[layer_idx].k_norms,
                              &self.kv_caches[layer_idx].v_packed, &self.kv_caches[layer_idx].v_norms],
                        );
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                            s.encoder_mut(), reg, metal_dev,
                            &self.activations.attn_k_normed,
                            &self.kv_caches[layer_idx].k_packed,
                            &self.kv_caches[layer_idx].k_norms,
                            nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                            kv_is_sliding,
                            None, // scale_factor_d512: bare=1.0 for prefill
                            None, // rms_scratch: probe not used during prefill
                        ).map_err(|e| anyhow::anyhow!("prefill TQ K L{layer_idx} T{tok_i}: {e}"))?;
                        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                            s.encoder_mut(), reg, metal_dev,
                            v_src,
                            &self.kv_caches[layer_idx].v_packed,
                            &self.kv_caches[layer_idx].v_norms,
                            nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                            kv_is_sliding,
                            None, // scale_factor_d512: bare=1.0 for prefill
                            None, // rms_scratch: probe not used during prefill
                        ).map_err(|e| anyhow::anyhow!("prefill TQ V L{layer_idx} T{tok_i}: {e}"))?;
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
                    if let Some(ref leg_hb_enc) = self.leg_hb_encoded {
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
                            &[&leg_hb_enc[layer_idx].k_packed, &leg_hb_enc[layer_idx].k_norms],
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
                            &[&leg_hb_enc[layer_idx].v_packed, &leg_hb_enc[layer_idx].v_norms],
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
                    let dense_kv_seq_len = if layer_is_ring {
                        ((tok_i + 1).min(layer_dense_cap)) as u32
                    } else {
                        (tok_i + 1) as u32
                    };
                    s.barrier_between(
                        &[&self.activations.attn_q_normed,
                          &dense_kvs_vec[layer_idx].k, &dense_kvs_vec[layer_idx].v],
                        &[&self.activations.sdpa_out],
                    );
                    let p = FlashAttnVecParams {
                        num_heads: nh as u32,
                        num_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        kv_seq_len: dense_kv_seq_len,
                        kv_capacity: layer_dense_cap as u32,
                        scale: 1.0, // Gemma4: scale = 1.0 (llama.cpp oracle)
                        mask_type: 1, // causal; ring applies the sliding window
                        sliding_window: 0,
                        softcap: 0.0,
                    };
                    mlx_native::ops::flash_attn_vec::flash_attn_vec(
                        s.encoder_mut(), reg, dev,
                        &self.activations.attn_q_normed,
                        &dense_kvs_vec[layer_idx].k,
                        &dense_kvs_vec[layer_idx].v,
                        &self.activations.sdpa_out,
                        &sdpa_tmp,
                        &p,
                    ).map_err(|e| anyhow::anyhow!("prefill dense SDPA L{layer_idx} T{tok_i}: {e}"))?;

                    if dump_here {
                        s.finish().map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(dump_dir, "sdpa_out", layer_idx, tok_i,
                                        &self.activations.sdpa_out, nh * hd)?;
                        // ADR-010 sub-stage dump: full dense K,V cache up to
                        // (and including) the target token, packed as
                        // [nkv, tok_i+1, hd] for comparison with llama's
                        // cache_k_l*/cache_v_l* at pos tok_i. Only F32 path.
                        if !use_f16_kv {
                            let cap = dense_kvs_vec[layer_idx].capacity;
                            let n_valid = tok_i + 1;
                            let k_full: &[f32] = dense_kvs_vec[layer_idx].k.as_slice()
                                .map_err(|e| anyhow::anyhow!("dump K cache L{layer_idx}: {e}"))?;
                            let v_full: &[f32] = dense_kvs_vec[layer_idx].v.as_slice()
                                .map_err(|e| anyhow::anyhow!("dump V cache L{layer_idx}: {e}"))?;
                            let mut k_valid = Vec::<f32>::with_capacity(nkv * n_valid * hd);
                            let mut v_valid = Vec::<f32>::with_capacity(nkv * n_valid * hd);
                            for h in 0..nkv {
                                for p in 0..n_valid {
                                    let off = h * cap * hd + p * hd;
                                    k_valid.extend_from_slice(&k_full[off..off+hd]);
                                    v_valid.extend_from_slice(&v_full[off..off+hd]);
                                }
                            }
                            for (name, buf) in [("k_cache_upto", &k_valid), ("v_cache_upto", &v_valid)] {
                                let path = format!(
                                    "{dump_dir}/hf2q_prefill_{name}_layer{layer_idx:02}_tok{tok_i:03}.bin");
                                let bytes: &[u8] = unsafe {
                                    std::slice::from_raw_parts(
                                        buf.as_ptr() as *const u8, buf.len() * 4) };
                                std::fs::write(&path, bytes)
                                    .map_err(|e| anyhow::anyhow!("write {path}: {e}"))?;
                                eprintln!(
                                    "[PREFILL DUMP] {} [{},{},{}] f32 -> {}",
                                    name, nkv, n_valid, hd, path);
                            }
                        }
                        s = exec.begin().map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // -- O-proj (same as decode) --
                    s.barrier_between(
                        &[&self.activations.sdpa_out, &self.layers[layer_idx].attn.o_proj.buffer],
                        &[&self.activations.attn_out],
                    );
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.sdpa_out,
                        &self.layers[layer_idx].attn.o_proj, &mut self.activations.attn_out, 1)?;

                    if dump_here {
                        s.finish().map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(dump_dir, "attn_out_pre_resid", layer_idx, tok_i,
                                        &self.activations.attn_out, hs)?;
                        s = exec.begin().map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // -- Fused post-attention norm + residual add --
                    s.barrier_between(
                        &[&self.activations.hidden, &self.activations.attn_out],
                        &[&self.activations.residual],
                    );
                    mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.hidden,
                        &self.activations.attn_out,
                        &self.layers[layer_idx].norms.post_attention_layernorm,
                        &self.activations.residual,
                        hs as u32, 1, eps,
                    ).map_err(|e| anyhow::anyhow!("prefill post-attn L{layer_idx} T{tok_i}: {e}"))?;

                    if dump_here {
                        s.finish().map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(dump_dir, "residual", layer_idx, tok_i,
                                        &self.activations.residual, hs)?;
                        s = exec.begin().map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
                    }

                    // ============================================================
                    // Dense MLP + MoE (identical to forward_decode)
                    // ============================================================
                    let num_experts = self.num_experts;
                    let top_k = self.layers[layer_idx].moe.top_k;

                    // B8: pre-FF norms [3 concurrent]
                    s.barrier_between(
                        &[&self.activations.residual],
                        &[&self.activations.norm_out, &self.activations.moe_norm_out,
                          &self.activations.router_norm_out],
                    );
                    s.rms_norm(reg, metal_dev,
                        &self.activations.residual,
                        &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                        &self.activations.norm_out,
                        &self.activations.norm_params, 1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("prefill pre-FF1 L{layer_idx} T{tok_i}: {e}"))?;
                    s.rms_norm(reg, metal_dev,
                        &self.activations.residual,
                        &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                        &self.activations.moe_norm_out,
                        &self.activations.norm_params, 1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("prefill pre-FF2 L{layer_idx} T{tok_i}: {e}"))?;
                    s.rms_norm(reg, metal_dev,
                        &self.activations.residual,
                        &self.layers[layer_idx].moe.router_combined_weight,
                        &self.activations.router_norm_out,
                        &self.activations.norm_params, 1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("prefill router norm L{layer_idx} T{tok_i}: {e}"))?;

                    // B9: gate + up + router [3 concurrent]
                    s.barrier_between(
                        &[&self.activations.norm_out, &self.activations.router_norm_out],
                        &[&self.activations.mlp_gate, &self.activations.mlp_up,
                          &self.activations.moe_router_logits],
                    );
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].mlp.gate_proj, &mut self.activations.mlp_gate, 1)?;
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                        &self.layers[layer_idx].mlp.up_proj, &mut self.activations.mlp_up, 1)?;
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.router_norm_out,
                        &self.layers[layer_idx].moe.router_proj,
                        &mut self.activations.moe_router_logits, 1)?;

                    // B10: gelu_mul + moe_routing [2 concurrent]
                    s.barrier_between(
                        &[&self.activations.mlp_gate, &self.activations.mlp_up,
                          &self.activations.moe_router_logits],
                        &[&self.activations.mlp_fused,
                          &self.activations.moe_expert_ids, &self.activations.moe_routing_weights_gpu],
                    );
                    {
                        use mlx_native::ops::encode_helpers::{encode_with_args, KernelArg};
                        let n_elements_bytes = (self.intermediate_size as u32).to_ne_bytes();
                        let pipeline = reg.get_pipeline("fused_gelu_mul", metal_dev)?;
                        encode_with_args(
                            s.encoder_mut(), pipeline,
                            &[
                                (0, KernelArg::Buffer(&self.activations.mlp_gate)),
                                (1, KernelArg::Buffer(&self.activations.mlp_up)),
                                (2, KernelArg::Buffer(&self.activations.mlp_fused)),
                                (3, KernelArg::Bytes(&n_elements_bytes)),
                            ],
                            mlx_native::MTLSize::new(self.intermediate_size as u64, 1, 1),
                            mlx_native::MTLSize::new(
                                std::cmp::min(256, self.intermediate_size as u64), 1, 1),
                        );
                    }
                    mlx_native::ops::fused_norm_add::dispatch_fused_moe_routing_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_router_logits,
                        &self.activations.moe_expert_ids,
                        &self.activations.moe_routing_weights_gpu,
                        &self.layers[layer_idx].moe.per_expert_scale,
                        num_experts as u32, top_k as u32,
                    ).map_err(|e| anyhow::anyhow!("prefill MoE routing L{layer_idx} T{tok_i}: {e}"))?;

                    // MoE expert dispatch (fused _id path)
                    let moe_int = self.layers[layer_idx].moe.moe_intermediate_size;
                    if self.layers[layer_idx].moe.stacked_gate_up.is_none()
                        || self.layers[layer_idx].moe.stacked_down.is_none()
                    {
                        anyhow::bail!("Prefill requires fused _id path (stacked weights) at L{layer_idx}");
                    }

                    // B11: dense down + gate_up_id
                    s.barrier_between(
                        &[&self.activations.mlp_fused, &self.layers[layer_idx].mlp.down_proj.buffer],
                        &[&self.activations.mlp_down],
                    );
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.mlp_fused,
                        &self.layers[layer_idx].mlp.down_proj, &mut self.activations.mlp_down, 1)?;

                    let ggml_type_gu = self.layers[layer_idx].moe.gate_up_ggml_dtype;
                    s.barrier_between(
                        &[&self.activations.moe_norm_out, &self.activations.moe_expert_ids,
                          self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap()],
                        &[&self.activations.moe_gate_up_id_out],
                    );
                    s.quantized_matmul_id_ggml(
                        reg, dev,
                        &self.activations.moe_norm_out,
                        self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap(),
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
                    ).map_err(|e| anyhow::anyhow!("prefill gate_up_id L{layer_idx} T{tok_i}: {e}"))?;

                    // B12: swiglu
                    s.barrier_between(
                        &[&self.activations.moe_gate_up_id_out],
                        &[&self.activations.moe_swiglu_id_out],
                    );
                    mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_gate_up_id_out,
                        &self.activations.moe_swiglu_id_out,
                        moe_int, top_k,
                    ).map_err(|e| anyhow::anyhow!("prefill swiglu L{layer_idx} T{tok_i}: {e}"))?;

                    // B13: down_id + post-FF norm1
                    let ggml_type_dn = self.layers[layer_idx].moe.down_ggml_dtype;
                    s.barrier_between(
                        &[&self.activations.moe_swiglu_id_out, &self.activations.moe_expert_ids,
                          self.layers[layer_idx].moe.stacked_down.as_ref().unwrap()],
                        &[&self.activations.moe_down_id_out],
                    );
                    s.quantized_matmul_id_ggml(
                        reg, dev,
                        &self.activations.moe_swiglu_id_out,
                        self.layers[layer_idx].moe.stacked_down.as_ref().unwrap(),
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
                    ).map_err(|e| anyhow::anyhow!("prefill down_id L{layer_idx} T{tok_i}: {e}"))?;

                    s.barrier_between(
                        &[&self.activations.mlp_down],
                        &[&self.activations.attn_out],
                    );
                    s.rms_norm(reg, metal_dev,
                        &self.activations.mlp_down,
                        &self.layers[layer_idx].norms.post_feedforward_layernorm_1,
                        &self.activations.attn_out,
                        &self.activations.norm_params, 1, hs as u32,
                    ).map_err(|e| anyhow::anyhow!("prefill post-FF1 L{layer_idx} T{tok_i}: {e}"))?;

                    // B14: weighted_sum
                    s.barrier_between(
                        &[&self.activations.moe_down_id_out, &self.activations.moe_routing_weights_gpu],
                        &[&self.activations.moe_accum],
                    );
                    mlx_native::ops::moe_dispatch::moe_weighted_sum_encode(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.moe_down_id_out,
                        &self.activations.moe_routing_weights_gpu,
                        &self.activations.moe_accum,
                        hs, top_k,
                    ).map_err(|e| anyhow::anyhow!("prefill weighted_sum L{layer_idx} T{tok_i}: {e}"))?;

                    // Post-FF norm2 + combine
                    s.barrier_between(
                        &[&self.activations.attn_out, &self.activations.moe_accum],
                        &[&self.activations.mlp_down],
                    );
                    mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.attn_out,
                        &self.activations.moe_accum,
                        &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                        &self.activations.mlp_down,
                        hs as u32, 1, eps,
                    ).map_err(|e| anyhow::anyhow!("prefill post-FF2 L{layer_idx} T{tok_i}: {e}"))?;

                    // End-of-layer: norm + residual + scalar
                    let scalar_is_vector = self.layers[layer_idx].layer_scalar.element_count() > 1;
                    s.barrier_between(
                        &[&self.activations.residual, &self.activations.mlp_down],
                        &[&self.activations.hidden],
                    );
                    mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_scalar_f32(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.residual,
                        &self.activations.mlp_down,
                        &self.layers[layer_idx].norms.post_feedforward_layernorm,
                        &self.activations.hidden,
                        &self.layers[layer_idx].layer_scalar,
                        1, hs as u32, eps,
                        scalar_is_vector,
                    ).map_err(|e| anyhow::anyhow!("prefill end-layer L{layer_idx} T{tok_i}: {e}"))?;

                    if dump_here {
                        s.finish().map_err(|e| anyhow::anyhow!("dump finish: {e}"))?;
                        write_dump_f32(dump_dir, "l_out", layer_idx, tok_i,
                                        &self.activations.hidden, hs)?;
                        s = exec.begin().map_err(|e| anyhow::anyhow!("dump restart: {e}"))?;
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
                        let k_raw: &[u8] = self.kv_caches[li].k_packed.as_slice()
                            .map_err(|e| anyhow::anyhow!("tq_dump nb k_packed L{li}: {e}"))?;
                        let v_raw: &[u8] = self.kv_caches[li].v_packed.as_slice()
                            .map_err(|e| anyhow::anyhow!("tq_dump nb v_packed L{li}: {e}"))?;
                        let k_norms_raw: &[f32] = self.kv_caches[li].k_norms.as_slice()
                            .map_err(|e| anyhow::anyhow!("tq_dump nb k_norms L{li}: {e}"))?;
                        let v_norms_raw: &[f32] = self.kv_caches[li].v_norms.as_slice()
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
                        eprintln!("[TQ_DUMP] k_packed L{li:02} [{nkv},{kv_seq_len},{hd_half}] u8 -> {kp}");
                        eprintln!("[TQ_DUMP] v_packed L{li:02} [{nkv},{kv_seq_len},{hd_half}] u8 -> {vp}");
                        let kn = format!("{dir}/hf2q_k_norms_layer{li:02}_pos{kv_seq_len}.f32.bin");
                        let vn = format!("{dir}/hf2q_v_norms_layer{li:02}_pos{kv_seq_len}.f32.bin");
                        let kn_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                kn_tight.as_ptr() as *const u8, kn_tight.len() * 4)
                        };
                        let vn_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                vn_tight.as_ptr() as *const u8, vn_tight.len() * 4)
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
                    s = exec.begin()
                        .map_err(|e| anyhow::anyhow!("tq_dump nonbatched re-begin: {e}"))?;
                }

                // --- 3. Final norm + lm_head + softcap + argmax ---
                s.barrier_between(
                    &[&self.activations.hidden, &self.final_norm],
                    &[&self.activations.norm_out],
                );
                s.rms_norm(reg, metal_dev,
                    &self.activations.hidden,
                    &self.final_norm,
                    &self.activations.norm_out,
                    &self.activations.norm_params, 1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("prefill final norm T{tok_i}: {e}"))?;

                if let Some(ref q8) = self.lm_head_q8 {
                    s.barrier_between(
                        &[&self.activations.norm_out, &q8.buffer],
                        &[&self.activations.logits],
                    );
                    super::forward_mlx::dispatch_qmatmul(
                        &mut s, reg, dev,
                        &self.activations.norm_out,
                        q8,
                        &mut self.activations.logits,
                        1,
                    ).map_err(|e| anyhow::anyhow!("prefill lm_head Q8 T{tok_i}: {e}"))?;
                } else if let Some(ref lm_head_f16) = self.lm_head_f16 {
                    s.barrier_between(
                        &[&self.activations.norm_out, lm_head_f16],
                        &[&self.activations.logits],
                    );
                    mlx_native::ops::dense_gemm::dispatch_dense_matvec_f16w_f32io(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.norm_out,
                        lm_head_f16,
                        &self.activations.logits,
                        &DenseGemmF16Params { m: 1, n: vocab_size as u32, k: hs as u32 },
                    ).map_err(|e| anyhow::anyhow!("prefill lm_head T{tok_i}: {e}"))?;
                } else {
                    anyhow::bail!("Prefill requires GPU lm_head (F16 or Q8 weight)");
                }

                if let Some(cap) = self.final_logit_softcapping {
                    s.barrier_between(
                        &[&self.activations.logits],
                        &[&self.activations.logits],
                    );
                    mlx_native::ops::softcap::dispatch_softcap(
                        s.encoder_mut(), reg, metal_dev,
                        &self.activations.logits,
                        &self.activations.logits,
                        &self.activations.softcap_params,
                        cap,
                    ).map_err(|e| anyhow::anyhow!("prefill softcap T{tok_i}: {e}"))?;
                }

                s.barrier_between(
                    &[&self.activations.logits],
                    &[&self.activations.argmax_index, &self.activations.argmax_value],
                );
                mlx_native::ops::argmax::dispatch_argmax_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.logits,
                    &self.activations.argmax_index,
                    &self.activations.argmax_value,
                    &self.activations.argmax_params,
                    vocab_size as u32,
                ).map_err(|e| anyhow::anyhow!("prefill argmax T{tok_i}: {e}"))?;

                s.finish()
                    .map_err(|e| anyhow::anyhow!("prefill finish T{tok_i}: {e}"))?;

                last_token = {
                    let idx: &[u32] = self.activations.argmax_index.as_slice()
                        .map_err(|e| anyhow::anyhow!("prefill argmax read T{tok_i}: {e}"))?;
                    idx[0]
                };
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
        self.dense_kvs = Some(dense_kvs_vec);
        self.dense_sdpa_tmp = Some(sdpa_tmp);

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
        // `max_decode_tokens=0` so `linear_capacity = prompt_len`, and the
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
}

// ---------------------------------------------------------------------------
// 3D-mRoPE position synthesis tests (ADR-005 iter-224 Wedge-4d)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod qwen3vl_position_tests {
    use super::{build_qwen3vl_positions, Qwen3VlImageGrid};

    /// Helper: extract one axis as a Vec<i32> for assertion.
    fn axis(flat: &[i32], axis: usize, prompt_len: usize) -> Vec<i32> {
        flat[axis * prompt_len..(axis + 1) * prompt_len].to_vec()
    }

    #[test]
    fn build_qwen3vl_positions_text_only_broadcast_t_across_axes() {
        // 5 text tokens, no images. Every axis gets [0,1,2,3,4].
        let flat = build_qwen3vl_positions(5, &[]).unwrap();
        for ax in 0..4 {
            assert_eq!(axis(&flat, ax, 5), vec![0, 1, 2, 3, 4]);
        }
    }

    #[test]
    fn build_qwen3vl_positions_single_image_emits_correct_grid() {
        // Layout: [text_0, text_1, IMG(2x3=6 tokens), text_8].
        // Image grid n_x=3, n_y=2, sequence_start=2.
        // After image: temporal advances by max(3,2)=3, so text_8 has t=2+3=5.
        let grid = Qwen3VlImageGrid { n_x: 3, n_y: 2 };
        let prompt_len = 2 + 6 + 1;
        let flat = build_qwen3vl_positions(prompt_len, &[(grid, 2)]).unwrap();
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
    fn build_qwen3vl_positions_multiple_images_global_counter_advances() {
        // [text_0, IMG1(2x2=4), text_5, IMG2(3x3=9), text_15]
        // IMG1 at seq 1, advance by max(2,2)=2 → t after IMG1 = 0+1+2 = 3
        //   wait: text_0 has t=0, then IMG1 at t=1, advance by 2 → t=3 after IMG1.
        // text_5 (in seq) is at seq pos 5, t=3.
        // IMG2 at seq pos 6, t=4 (after one text token at t=3 → t advances to 4).
        // After IMG2 (n_x=3, n_y=3, advance=3): t = 4+3 = 7.
        // text_15 at seq pos 15, t=7.
        let img1 = Qwen3VlImageGrid { n_x: 2, n_y: 2 };
        let img2 = Qwen3VlImageGrid { n_x: 3, n_y: 3 };
        let prompt_len = 1 + 4 + 1 + 9 + 1; // = 16
        let flat = build_qwen3vl_positions(prompt_len, &[(img1, 1), (img2, 6)]).unwrap();
        let t_axis = axis(&flat, 0, prompt_len);
        assert_eq!(t_axis[0], 0); // text_0
        assert_eq!(t_axis[1..5], [1, 1, 1, 1]); // IMG1 t-axis (constant)
        assert_eq!(t_axis[5], 3); // text_5: after IMG1 advance, t=3
        assert_eq!(t_axis[6..15], [4, 4, 4, 4, 4, 4, 4, 4, 4]); // IMG2 t-axis
        assert_eq!(t_axis[15], 7); // text_15: after IMG2 advance, t=7
    }

    #[test]
    fn build_qwen3vl_positions_h_w_swap_detectably_different() {
        // Sabotage check: if we accidentally swap h/w, the axis 1 vs
        // axis 2 outputs MUST differ for non-square images.
        let grid_3x2 = Qwen3VlImageGrid { n_x: 3, n_y: 2 };
        let flat = build_qwen3vl_positions(6, &[(grid_3x2, 0)]).unwrap();
        let y = axis(&flat, 1, 6);
        let x = axis(&flat, 2, 6);
        // Image positions 0..6: y=[0,0,0,1,1,1], x=[0,1,2,0,1,2]
        assert_eq!(y, vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(x, vec![0, 1, 2, 0, 1, 2]);
        assert_ne!(y, x, "y and x axes must differ for non-square grid");
    }

    #[test]
    fn build_qwen3vl_positions_rejects_overlapping_images() {
        let img1 = Qwen3VlImageGrid { n_x: 4, n_y: 4 }; // 16 tokens
        let img2 = Qwen3VlImageGrid { n_x: 2, n_y: 2 }; // 4 tokens
        // img1 starts at 0, ends at 16; img2 at 10 overlaps.
        let err = build_qwen3vl_positions(20, &[(img1, 0), (img2, 10)]).unwrap_err();
        assert!(format!("{err}").contains("before the prior region"));
    }

    #[test]
    fn build_qwen3vl_positions_rejects_image_past_prompt_len() {
        let img = Qwen3VlImageGrid { n_x: 4, n_y: 4 }; // 16 tokens
        // img at seq=10, region = 10..26, prompt_len=20.
        let err = build_qwen3vl_positions(20, &[(img, 10)]).unwrap_err();
        assert!(format!("{err}").contains("extends past prompt_len"));
    }

    #[test]
    fn build_qwen3vl_positions_rejects_zero_tokens() {
        let img = Qwen3VlImageGrid { n_x: 0, n_y: 5 };
        let err = build_qwen3vl_positions(10, &[(img, 0)]).unwrap_err();
        assert!(format!("{err}").contains("zero tokens"));
    }

    #[test]
    fn build_qwen3vl_positions_image_at_prompt_start() {
        // [IMG(2x2=4), text_4] — image at sequence position 0.
        let grid = Qwen3VlImageGrid { n_x: 2, n_y: 2 };
        let flat = build_qwen3vl_positions(5, &[(grid, 0)]).unwrap();
        // t-axis: image at t=0 (constant), then text at t=2 (advance by max(2,2)=2).
        assert_eq!(axis(&flat, 0, 5), vec![0, 0, 0, 0, 2]);
        // y-axis: image i=0..4 → y=[0,0,1,1] (i/n_x), then text=2.
        assert_eq!(axis(&flat, 1, 5), vec![0, 0, 1, 1, 2]);
        // x-axis: image i=0..4 → x=[0,1,0,1] (i%n_x), then text=2.
        assert_eq!(axis(&flat, 2, 5), vec![0, 1, 0, 1, 2]);
    }

    #[test]
    fn qwen3vl_image_grid_temporal_advance_uses_max() {
        // Per peer mtmd.cpp:1354-1357 MTMD_POS_TYPE_MROPE returns max(nx, ny).
        assert_eq!(Qwen3VlImageGrid { n_x: 24, n_y: 24 }.temporal_advance(), 24);
        assert_eq!(Qwen3VlImageGrid { n_x: 32, n_y: 16 }.temporal_advance(), 32);
        assert_eq!(Qwen3VlImageGrid { n_x: 8, n_y: 40 }.temporal_advance(), 40);
    }

    #[test]
    fn qwen3vl_image_grid_n_image_tokens_is_product() {
        assert_eq!(Qwen3VlImageGrid { n_x: 24, n_y: 24 }.n_image_tokens(), 576);
        assert_eq!(Qwen3VlImageGrid { n_x: 12, n_y: 8 }.n_image_tokens(), 96);
    }
}
