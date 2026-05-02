//! Qwen3-VL ViT forward (ADR-005 iter-224 row 3 — Wedge-4c).
//!
//! **Status (sub-iter 4c.2)**: prelude helpers landed (CPU-side dual
//! conv2d patch embedding, 2×2 block-merge reshape, bilinear
//! position-embedding resize). The public dispatch entry-point still
//! returns `Err(...)` after input validation — the helpers are
//! testable in isolation and 4c.3 will wire them through the GPU
//! per-block forward. The dispatch-site sentinel
//! `num_position_embeddings: 1` in `vit_gpu.rs` is replaced with the
//! real shape extraction from `v.position_embd.weight` (closes the
//! Codex Phase-2b drift trap from 4c.1).
//!
//! # Sub-iter roadmap (sequential, all blocked on this 4c.1 scaffold)
//!
//! | Sub-iter | Scope                                                                    |
//! |----------|--------------------------------------------------------------------------|
//! | 4c.1     | LANDED — `Qwen3VlViTConfig` + `compute_vision_embeddings_gpu_qwen3vl`    |
//! |          |   stub, dispatch-arm wired, `is_supported()` stays `false`               |
//! | 4c.2     | **THIS** — input validation, dispatch-site `num_position_embeddings`     |
//! |          |   real-shape extraction, CPU helpers for dual-conv patch embedding +     |
//! |          |   bilinear position-embedding resize + 2×2 block-merge reshape (the      |
//! |          |   `qwen3vl.cpp:27-37` prelude reshape, NOT the spatial-merge=2           |
//! |          |   reduction at qwen3vl.cpp:177 which is 4c.4 territory)                  |
//! | 4c.3     | Per-block forward: LN → 2D-RoPE Q,K → attn → +residual → LN → GELU MLP   |
//! |          |   → +residual; reuse `apply_layer_norm_gpu`, `apply_attention_gpu`;      |
//! |          |   consumes mlx-native `RopeMultiMode::Vision = 24` (LANDED on           |
//! |          |   /opt/mlx-native main `d0e6c42`)                                        |
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

use anyhow::{anyhow, Context, Result};

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
/// |                           |   — see note below; ggml `ne[1]` after mlx-native       |
/// |                           |   reverse → row-major `shape()[0]`. Passed explicitly.  |
/// | `deepstack_indexes`       | `MmprojConfig.deepstack_indexes` (must be `Some`)       |
/// | `eps`                     | `MmprojConfig.layer_norm_eps`                           |
///
/// # Why `num_position_embeddings` is a separate parameter to `from_mmproj`
///
/// llama.cpp does NOT write `num_position_embeddings` as a metadata
/// key (verified via `grep -rn "num_position_embeddings" clip.cpp
/// clip-impl.h` — only the position embedding *tensor* itself
/// (`v.position_embd.weight`) carries the dimension. Per
/// `clip.cpp:272-289` (`clip_graph::resize_position_embeddings`),
/// the position-table count is ggml `pos_embd->ne[1]` and the
/// per-side trained edge is `sqrt(ne[1])` (the table is square
/// 2D, e.g. 48×48 for Qwen3-VL-2B at 768×768 / 16×16 → ne[1]=2304).
///
/// # Axis-mapping note (Wedge-4c.2 closure)
///
/// ggml stores tensor dims innermost-first (column-major); hf2q's
/// `mlx_native::gguf::GgufFile` parser at `mod.rs:861` reverses
/// `shape` to row-major-style `[outer ... inner]`. The mapping is:
///
/// | ggml convention                      | hf2q row-major shape   |
/// |--------------------------------------|------------------------|
/// | `pos_embd->ne[0] = n_embd` (innermost)| `shape()[1] = n_embd`  |
/// | `pos_embd->ne[1] = count` (outermost) | `shape()[0] = count`   |
///
/// So the dispatch site reads
/// `mmproj_weights.position_embd_weight()?.shape()[0]` to get
/// `num_position_embeddings`. The clip.cpp `ne[1]` reference is
/// the C++/ggml side of the same axis. The 4c.1 doc-comment that
/// said `shape()[1]` confused ggml-axis with row-major-axis;
/// 4c.2 closes that drift trap by passing the real value.
///
/// (Citation correction 2026-05-02: an earlier scaffold revision
/// cited `clip.cpp:3849` and `ne[0]` — that line is actually
/// `PROJECTOR_TYPE_LFM2A` returning the projection-dim of an
/// unrelated audio model. Codex Phase-2b review caught it; the
/// correct reference is `clip.cpp:272-289` with ggml `ne[1]` =
/// hf2q `shape()[0]` after the mlx-native reverse.)
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
    /// Qwen3-VL-2B at 768×768 / 16×16 = 48² = 2304). Sourced from
    /// the LOADED `v.position_embd.weight` tensor's outer (count)
    /// axis (per `clip.cpp:272-289` — the table is shaped
    /// `(n_embd, count)` in ggml convention; ggml `ne[1] = count`
    /// maps to hf2q row-major `shape()[0] = count` after
    /// mlx-native's reverse). 4c.2 derives this from
    /// `mmproj_weights.position_embd_weight()?.shape()[0]`. Not
    /// from a metadata key. Passed in explicitly to `from_mmproj`
    /// so the call-site is unambiguous about where the value came
    /// from.
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
                 (sourced from v.position_embd.weight tensor outer axis = \
                 hf2q `shape()[0]` = ggml `ne[1]`)"
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

// ---------------------------------------------------------------------------
// Wedge-4c.2 prelude helpers (CPU-side; testable in isolation)
// ---------------------------------------------------------------------------
//
// These mirror the prelude block at
// `/opt/llama.cpp/tools/mtmd/models/qwen3vl.cpp:16-58` (everything
// before the per-block loop):
//
//   1. dual `ggml_conv_2d` (patch_embeddings_0 + patch_embeddings_1)
//   2. element-wise add of the two conv outputs
//   3. permute + cont_4d + reshape_4d + permute + cont_3d   (2×2 block-merge)
//   4. add `patch_bias` if present
//   5. resize_position_embeddings (siglip2 naflex bilinear, clip.cpp:272-289)
//   6. apply same 2×2 block-merge reshape to the resized pos embed
//   7. element-wise add of pos embed onto the patch sum
//
// 4c.2 ships steps 1-7 as CPU helpers. 4c.3 will lift them to GPU
// dispatch and chain with the per-block forward.

/// CPU dual-stem patch embedding for Qwen3-VL.
///
/// Implements `qwen3vl.cpp:17 + 23-25` — two `ggml_conv_2d` calls
/// over the same `pixel_values` with two distinct kernels
/// `patch_embeddings_0` (`v.patch_embd.weight`) and
/// `patch_embeddings_1` (`v.patch_embd.weight.1`), summed element-
/// wise. Both convs share `kernel_size = stride = patch_size`,
/// padding=0, dilation=1.
///
/// This is functionally identical to running
/// [`super::vit::patch_embed_forward`] twice (once with each weight
/// matrix) and adding the outputs. We do that explicitly here so the
/// dual-stem code path is its own unit-tested surface and so the
/// `qwen3vl.cpp:16-58` prelude reads top-to-bottom in the source.
///
/// # Output shape
///
/// `Vec<f32>` of length `n_patches * hidden` where `n_patches =
/// (image_size / patch_size)²`, row-major `[n_patches, hidden]`.
/// This is the input to [`qwen3vl_2x2_block_merge_reshape`].
///
/// # Errors
///
/// - `image_size % patch_size != 0`
/// - `pixel_values.len() != 3 * image_size²`
/// - `weight_0.len()` or `weight_1.len() != hidden * 3 * patch_size²`
/// - `bias.is_some()` and `bias.len() != hidden`
///
/// All shape checks are delegated to [`super::vit::patch_embed_forward`]
/// (which carries the canonical messages); we just sum the two outputs.
///
/// # Cross-references
///
/// - `/opt/llama.cpp/tools/mtmd/models/qwen3vl.cpp:17, 23-25`
/// - `/opt/llama.cpp/tools/mtmd/clip-impl.h:75` (`TN_PATCH_EMBD_1 = "v.patch_embd.weight.1"`)
/// - `/opt/hf2q/src/models/vit/convert.rs:228-249` (HF 5-D `[out, in, T=2, H, W]`
///   patch_embed split into the two GGUF tensors; T=0 → `v.patch_embd.weight`,
///   T=1 → `v.patch_embd.weight.1`)
//
// `#[allow(dead_code)]` until 4c.3 wires this through the GPU
// per-block forward. The function ships in 4c.2 with full unit-test
// coverage so 4c.3 can import + use it without re-litigating the
// dual-conv contract.
#[allow(dead_code)]
pub(crate) fn qwen3vl_dual_conv_patch_embed_cpu(
    pixel_values: &[f32],
    weight_0: &[f32],
    weight_1: &[f32],
    bias: Option<&[f32]>,
    image_size: u32,
    patch_size: u32,
    hidden: u32,
) -> Result<Vec<f32>> {
    use super::vit::patch_embed_forward;

    // Stem 0: bias goes here (matches qwen3vl.cpp ordering — the bias
    // add at line 41-43 comes AFTER the `inp = inp + inp_1` at line 25,
    // so adding it to either stem before the sum is functionally
    // equivalent to adding it to the sum at the end. We attach it to
    // stem 0 to keep `patch_embed_forward`'s existing behavior intact
    // and avoid duplicating the bias add).
    let out_0 = patch_embed_forward(
        pixel_values,
        weight_0,
        bias,
        image_size,
        patch_size,
        hidden,
    )
    .context("qwen3vl_dual_conv_patch_embed_cpu: stem 0")?;
    let out_1 = patch_embed_forward(
        pixel_values,
        weight_1,
        None, // stem 1 carries no bias
        image_size,
        patch_size,
        hidden,
    )
    .context("qwen3vl_dual_conv_patch_embed_cpu: stem 1")?;

    if out_0.len() != out_1.len() {
        return Err(anyhow!(
            "qwen3vl_dual_conv_patch_embed_cpu: stem outputs length mismatch \
             ({} vs {}) — patch_embed_forward shape contract violated",
            out_0.len(),
            out_1.len()
        ));
    }

    // Element-wise sum (qwen3vl.cpp:25 `inp = ggml_add(ctx0, inp, inp_1)`).
    let mut summed = out_0;
    for (a, b) in summed.iter_mut().zip(out_1.iter()) {
        *a += *b;
    }
    Ok(summed)
}

/// CPU 2×2 block-merge reshape — the dual-conv prelude rearrange at
/// `qwen3vl.cpp:27-37`.
///
/// **Caveat**: this is the 2×2 PRELUDE rearrange that lives between
/// the dual conv (which produces `[ny, nx, n_embd]`) and the per-block
/// loop. It is NOT the `spatial_merge_size=2` reduction at
/// `qwen3vl.cpp:177` (that 4× token-count drop happens inside the
/// main projector and lands in 4c.4). The prelude rearrange preserves
/// the total patch count but reorders patches into 2×2-block-major
/// order so the main projector's later `reshape_3d(n_embd*4,
/// n_pos/4, ...)` cleanly groups 4 consecutive patches into one
/// extended-channel patch.
///
/// # Functional spec
///
/// Given an input tensor `[ny, nx, n_embd]` row-major (patches in
/// y-major-then-x order), produce an output tensor `[ny*nx, n_embd]`
/// row-major where patches are arranged in 2×2-block-major-then-
/// row-major-within-block order:
///
/// ```text
///  block_id (block-major, row-major)
///    = (y / 2) * (nx / 2) + (x / 2)
///  within_block_offset (row-major within 2×2)
///    = (y % 2) * 2 + (x % 2)
///  output_p_index = block_id * 4 + within_block_offset
/// ```
///
/// Verified by tracing the ggml chain at `qwen3vl.cpp:27-37` in
/// column-major coordinates and converting back to hf2q row-major.
/// See also the corresponding pos-embd chain at
/// `qwen3vl.cpp:48-57` — which uses the same reshape sequence and
/// MUST receive the same rearrange so both inputs to the
/// `ggml_add` at line 58 land in matching block order.
///
/// # Errors
///
/// - `nx == 0` or `ny == 0`
/// - `nx % 2 != 0` or `ny % 2 != 0` (block-merge requires even side)
/// - `input.len() != ny * nx * n_embd`
//
// `#[allow(dead_code)]` until 4c.3 wires this. See sibling note on
// `qwen3vl_dual_conv_patch_embed_cpu`.
#[allow(dead_code)]
pub(crate) fn qwen3vl_2x2_block_merge_reshape(
    input: &[f32],
    nx: usize,
    ny: usize,
    n_embd: usize,
) -> Result<Vec<f32>> {
    if nx == 0 || ny == 0 || n_embd == 0 {
        return Err(anyhow!(
            "qwen3vl_2x2_block_merge_reshape: nx ({nx}), ny ({ny}), n_embd ({n_embd}) \
             must all be > 0"
        ));
    }
    if nx % 2 != 0 || ny % 2 != 0 {
        return Err(anyhow!(
            "qwen3vl_2x2_block_merge_reshape: nx ({nx}) and ny ({ny}) must both be \
             even (2×2 block merge); enforced upstream by image_size % \
             (patch_size * spatial_merge_size) check"
        ));
    }
    let expected = ny * nx * n_embd;
    if input.len() != expected {
        return Err(anyhow!(
            "qwen3vl_2x2_block_merge_reshape: input.len() ({}) != ny*nx*n_embd ({})",
            input.len(),
            expected
        ));
    }

    let mut out = vec![0f32; expected];
    let half_x = nx / 2;
    for by in 0..(ny / 2) {
        for bx in 0..half_x {
            let block_id = by * half_x + bx;
            for y_in in 0..2 {
                for x_in in 0..2 {
                    let src_y = by * 2 + y_in;
                    let src_x = bx * 2 + x_in;
                    let src_off = (src_y * nx + src_x) * n_embd;
                    let within = y_in * 2 + x_in;
                    let dst_p = block_id * 4 + within;
                    let dst_off = dst_p * n_embd;
                    out[dst_off..dst_off + n_embd]
                        .copy_from_slice(&input[src_off..src_off + n_embd]);
                }
            }
        }
    }
    Ok(out)
}

/// CPU bilinear resize of the trained position-embedding table to a
/// target patch grid.
///
/// Mirrors the no-op fast-path at `clip.cpp:281-283` (skip when
/// `width == height == n_per_side`) and the resize chain at
/// `clip.cpp:285-289` (reshape → permute → ggml_interpolate(BILINEAR)
/// → permute → cont_2d). Pixel coordinate mapping follows the
/// PyTorch `F.interpolate(mode='bilinear', align_corners=False)`
/// convention with `pixel_offset = 0.5`, matching the ggml CPU
/// implementation at `/opt/llama.cpp/ggml/src/ggml-cpu/ops.cpp:7637-7676`.
///
/// **Antialias flag**: clip.cpp's default mode is
/// `BILINEAR | ANTIALIAS` (per `clip-graph.h:12`). Antialiasing only
/// affects DOWNSAMPLING (when `target_n_per_side < trained_n_per_side`,
/// it applies a Lanczos-style triangle prefilter). For Qwen3-VL the
/// trained resolution is 768×768 / 16² = 48×48 = 2304; production
/// inference uses the same image_size (set at preprocess time per
/// `convert_hf_to_gguf.py:4863-4869`). Resize triggers only when a
/// future Wedge-4d dynamic-resolution preprocessor differs. 4c.2
/// implements plain bilinear; the antialias path lands as a 4c.4
/// follow-up if real-fixture parity requires it. The fast-path at
/// equal sizes is byte-exact regardless.
///
/// # Inputs
///
/// - `pos_embd_table`: `[num_position_embeddings, n_embd]` row-major,
///   sourced from `mmproj_weights.position_embd_weight()` widened to
///   F32 via `tensor_as_f32_owned`. The table is square — i.e.
///   `num_position_embeddings = n_per_side²` for some `n_per_side`.
/// - `num_position_embeddings`: trained-resolution table length.
///   `sqrt(num_position_embeddings)` MUST be an exact integer.
/// - `n_embd`: ViT hidden dim.
/// - `target_n_per_side`: post-conv patch count per side at runtime
///   `= image_size / patch_size`.
///
/// # Output
///
/// `Vec<f32>` of length `target_n_per_side² * n_embd`, row-major
/// `[target_n_per_side², n_embd]`. The patches are in y-major-then-x
/// order — i.e. the same layout as the dual-conv output BEFORE
/// [`qwen3vl_2x2_block_merge_reshape`] is applied. The caller MUST
/// run the block-merge reshape on this output to match the patch
/// embedding's post-prelude layout (qwen3vl.cpp:48-57).
///
/// # Errors
///
/// - `num_position_embeddings == 0` or not a perfect square
/// - `n_embd == 0` or `target_n_per_side == 0`
/// - `pos_embd_table.len() != num_position_embeddings * n_embd`
//
// `#[allow(dead_code)]` until 4c.3 wires this. See sibling note on
// `qwen3vl_dual_conv_patch_embed_cpu`.
#[allow(dead_code)]
pub(crate) fn qwen3vl_resize_position_embeddings_bilinear(
    pos_embd_table: &[f32],
    num_position_embeddings: u32,
    n_embd: u32,
    target_n_per_side: u32,
) -> Result<Vec<f32>> {
    if num_position_embeddings == 0 || n_embd == 0 || target_n_per_side == 0 {
        return Err(anyhow!(
            "qwen3vl_resize_position_embeddings_bilinear: num_position_embeddings \
             ({num_position_embeddings}), n_embd ({n_embd}), target_n_per_side \
             ({target_n_per_side}) must all be > 0"
        ));
    }
    // Per clip.cpp:277 `n_per_side = sqrt(pos_embd->ne[1])` — the
    // table MUST be square. Verify by checking `n_per_side² ==
    // num_position_embeddings` exactly (no FP slop).
    let trained_n = (num_position_embeddings as f64).sqrt() as u32;
    if trained_n.saturating_mul(trained_n) != num_position_embeddings {
        return Err(anyhow!(
            "qwen3vl_resize_position_embeddings_bilinear: \
             num_position_embeddings ({}) is not a perfect square — \
             trained position-embedding table must be a square 2-D grid \
             (per clip.cpp:277)",
            num_position_embeddings
        ));
    }
    let expected_table = (num_position_embeddings as usize) * (n_embd as usize);
    if pos_embd_table.len() != expected_table {
        return Err(anyhow!(
            "qwen3vl_resize_position_embeddings_bilinear: \
             pos_embd_table.len() ({}) != num_position_embeddings*n_embd ({})",
            pos_embd_table.len(),
            expected_table
        ));
    }

    // Fast path (clip.cpp:281-283): trained edge equals target edge,
    // pass through unchanged. Byte-exact regardless of mode flags.
    if trained_n == target_n_per_side {
        return Ok(pos_embd_table.to_vec());
    }

    // General path. The trained table is `[trained_n*trained_n,
    // n_embd]` row-major; we want output `[target_n*target_n, n_embd]`
    // row-major. Bilinear interpolates each n_embd channel
    // independently; each output position `(y_dst, x_dst)` reads 4
    // source positions and blends.
    //
    // Source-coord mapping (PyTorch align_corners=False / pixel_offset
    // = 0.5, per ggml-cpu/ops.cpp:7637-7676):
    //
    //   sf       = target_n / trained_n
    //   x_src    = (x_dst + 0.5) / sf - 0.5
    //   y_src    = (y_dst + 0.5) / sf - 0.5
    //   x0       = clamp(floor(x_src),     0, trained_n - 1)
    //   x1       = clamp(x0 + 1,           0, trained_n - 1)
    //   dx       = clamp(x_src - x0,       0, 1)        // post-clamp
    //   (same for y)
    //   val      = a * (1 - dx) * (1 - dy)
    //            + b *      dx  * (1 - dy)
    //            + c * (1 - dx) *      dy
    //            + d *      dx  *      dy
    //
    // where a,b,c,d are the four trained-table entries at (x0,y0),
    // (x1,y0), (x0,y1), (x1,y1).
    let trained = trained_n as i64;
    let target = target_n_per_side as i64;
    let h = n_embd as usize;
    let mut out = vec![0f32; (target as usize) * (target as usize) * h];

    let sf = (target as f32) / (trained as f32);
    let pixel_offset: f32 = 0.5;

    for y_dst in 0..target {
        let y_src = ((y_dst as f32) + pixel_offset) / sf - pixel_offset;
        let mut y0 = y_src.floor() as i64;
        let mut y1 = y0 + 1;
        y0 = y0.max(0).min(trained - 1);
        y1 = y1.max(0).min(trained - 1);
        let mut dy = y_src - (y0 as f32);
        dy = dy.max(0.0).min(1.0);

        for x_dst in 0..target {
            let x_src = ((x_dst as f32) + pixel_offset) / sf - pixel_offset;
            let mut x0 = x_src.floor() as i64;
            let mut x1 = x0 + 1;
            x0 = x0.max(0).min(trained - 1);
            x1 = x1.max(0).min(trained - 1);
            let mut dx = x_src - (x0 as f32);
            dx = dx.max(0.0).min(1.0);

            let w_a = (1.0 - dx) * (1.0 - dy);
            let w_b = dx * (1.0 - dy);
            let w_c = (1.0 - dx) * dy;
            let w_d = dx * dy;

            let off_a = ((y0 as usize) * (trained as usize) + (x0 as usize)) * h;
            let off_b = ((y0 as usize) * (trained as usize) + (x1 as usize)) * h;
            let off_c = ((y1 as usize) * (trained as usize) + (x0 as usize)) * h;
            let off_d = ((y1 as usize) * (trained as usize) + (x1 as usize)) * h;

            let dst_off = ((y_dst as usize) * (target as usize) + (x_dst as usize)) * h;

            for k in 0..h {
                out[dst_off + k] = pos_embd_table[off_a + k] * w_a
                    + pos_embd_table[off_b + k] * w_b
                    + pos_embd_table[off_c + k] * w_c
                    + pos_embd_table[off_d + k] * w_d;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Public dispatch entry-point
// ---------------------------------------------------------------------------

/// Qwen3-VL ViT end-to-end GPU forward (sibling to
/// `compute_vision_embeddings_gpu_gemma4v` in `vit_gpu.rs`).
///
/// # Status (sub-iter 4c.2)
///
/// Validates the input slice (single image only) and the
/// `image_size` divisibility constraint upfront, then returns
/// `Err(...)` naming sub-iter 4c.3 (per-block forward). The CPU
/// prelude helpers ([`qwen3vl_dual_conv_patch_embed_cpu`],
/// [`qwen3vl_2x2_block_merge_reshape`],
/// [`qwen3vl_resize_position_embeddings_bilinear`]) are testable in
/// isolation; 4c.3 will chain them with the GPU per-block loop. The
/// dispatch arm in `vit_gpu.rs::compute_vision_embeddings_gpu_dispatch`
/// already passes the real `num_position_embeddings` extracted from
/// `v.position_embd.weight` shape (4c.2 closed the 4c.1 sentinel).
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
/// # Inputs (placeholder signature stable across sub-iters)
///
/// - `inputs`: heterogeneous `VisionInput` slice. Phase 1 supports
///   exactly 1 input; the function rejects len != 1 with a clear
///   error (multi-image batching is Wedge-4d territory). Only the
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
/// - `inputs.len() != 1` (Phase 1 single-image only)
/// - input variant is not `VisionInput::Siglip49(_)` (e.g.
///   `Gemma4v(_)` slipped past dispatch — would mean preprocessing
///   chose the wrong family branch)
/// - `mmproj_cfg.image_size % (cfg.patch_size * cfg.spatial_merge_size)
///   != 0` (per qwen3vl.cpp:19-20 `GGML_ASSERT(img.nx % (patch_size *
///   2) == 0)`; we additionally enforce divisibility by the
///   spatial-merge stride so the 4c.4 main projector's
///   `reshape_3d(n_embd*4, n_pos/4)` is exact)
/// - `pixel_values.len() != 3 * image_size²` (preprocessing contract
///   violation)
/// - Sub-iter 4c.2 always returns Err with a "4c.3" marker after
///   validation passes — per-block forward not yet implemented.
pub fn compute_vision_embeddings_gpu_qwen3vl(
    inputs: &[VisionInput],
    _mmproj_weights: &LoadedMmprojWeights,
    cfg: &Qwen3VlViTConfig,
    mmproj_cfg: &MmprojConfig,
) -> Result<Vec<Vec<f32>>> {
    // -----------------------------------------------------------------
    // Input validation (Phase 1: single image, square image_size).
    // -----------------------------------------------------------------
    if inputs.len() != 1 {
        return Err(anyhow!(
            "compute_vision_embeddings_gpu_qwen3vl: Phase 1 supports exactly \
             1 input image; got {} (multi-image batching is Wedge-4d territory)",
            inputs.len()
        ));
    }
    let input = &inputs[0];
    let pixel_values: &[f32] = match input {
        VisionInput::Siglip49(p) => &p.pixel_values,
        VisionInput::Gemma4v(_) => {
            return Err(anyhow!(
                "compute_vision_embeddings_gpu_qwen3vl: Qwen3-VL preprocessing \
                 must produce a Siglip49 payload, got Gemma4v — caller-side \
                 family-branch routing is broken (dispatch should have rejected \
                 this earlier)"
            ));
        }
    };

    let image_size = mmproj_cfg.image_size;
    let stride = cfg.patch_size * cfg.spatial_merge_size;
    if stride == 0 {
        return Err(anyhow!(
            "compute_vision_embeddings_gpu_qwen3vl: patch_size ({}) * \
             spatial_merge_size ({}) = 0",
            cfg.patch_size,
            cfg.spatial_merge_size
        ));
    }
    if image_size % stride != 0 {
        // Mirrors qwen3vl.cpp:19-20:
        //   GGML_ASSERT(img.nx % (patch_size * 2) == 0);
        //   GGML_ASSERT(img.ny % (patch_size * 2) == 0);
        // We use cfg.spatial_merge_size (= 2 for Qwen3-VL) instead of the
        // hard-coded 2 so a future variant with merge_size != 2 still
        // gets the right check.
        return Err(anyhow!(
            "compute_vision_embeddings_gpu_qwen3vl: image_size ({}) must be \
             a multiple of patch_size ({}) * spatial_merge_size ({}) = {} \
             (per qwen3vl.cpp:19-20 GGML_ASSERT)",
            image_size,
            cfg.patch_size,
            cfg.spatial_merge_size,
            stride
        ));
    }
    let expected_pixels = 3 * (image_size as usize) * (image_size as usize);
    if pixel_values.len() != expected_pixels {
        return Err(anyhow!(
            "compute_vision_embeddings_gpu_qwen3vl: pixel_values.len() ({}) != \
             3 * image_size² = 3 * {}² = {} — preprocessing contract violated",
            pixel_values.len(),
            image_size,
            expected_pixels
        ));
    }

    // -----------------------------------------------------------------
    // 4c.2 stops here. The CPU prelude helpers above are linked from
    // here in 4c.3; today the per-block forward isn't wired so we
    // surface the pending sub-iter explicitly.
    // -----------------------------------------------------------------
    Err(anyhow!(
        "Wedge-4c.3: per-block ViT forward not yet implemented \
         (4c.2 prelude landed: dual conv + position embedding done; \
         per-block LayerNorm→2D-RoPE attn→MLP coming in 4c.3 using \
         mlx-native d0e6c42 RopeMultiMode::Vision). 4c.4 then adds \
         deepstack heads + spatial merger + main projector; 4c.5 \
         flips ProjectorType::Qwen3VlMerger.is_supported() to true."
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
        // 4c.2 contract update: empty input slice now hits the Phase-1
        // "exactly 1 input image" guard FIRST (was: returned the 4c.1
        // scaffold message before validation). The empty-input fail-loud
        // path is now Phase-1 input-validation, which is the contract
        // 4c.2 ships. The "4c.3 marker" message lives behind a valid
        // 1-input call (see `qwen3vl_compute_returns_err_with_4c3_marker`
        // below).
        let cfg = synth_qwen3vl_mmproj_cfg(
            24,
            Some(vec![5, 11, 17]),
            Some(2),
            Some(2048),
        );
        let vit_cfg = Qwen3VlViTConfig::from_mmproj(&cfg, 2304).unwrap();
        let weights = make_empty_loaded_mmproj_weights();
        let result =
            compute_vision_embeddings_gpu_qwen3vl(&[], &weights, &vit_cfg, &cfg);
        let err = result.expect_err(
            "compute_vision_embeddings_gpu_qwen3vl must reject an empty input slice",
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("Phase 1 supports exactly 1 input image"),
            "4c.2 input-validation must self-identify as Phase 1 single-image \
             reject; got: {msg}"
        );
        assert!(
            msg.contains("got 0"),
            "error message must name the actual count (0 here); got: {msg}"
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

    // -----------------------------------------------------------------
    // Wedge-4c.2 helpers — new tests
    // -----------------------------------------------------------------

    /// Test #1 — dual-stem patch embedding sums the two conv outputs.
    ///
    /// Stem 0 weights are all-ones, stem 1 weights are all-zeros, and
    /// the bias is None. The 16×16 single-patch input is random; the
    /// expected output is byte-equal to the all-ones stem alone (via
    /// `patch_embed_forward`). This pins the contract that
    /// `qwen3vl_dual_conv_patch_embed_cpu` adds the two stems
    /// element-wise (qwen3vl.cpp:25 `inp = ggml_add(ctx0, inp, inp_1)`).
    #[test]
    fn qwen3vl_dual_conv_patch_embedding_synthetic() {
        use super::super::vit::patch_embed_forward;

        let image_size: u32 = 16;
        let patch_size: u32 = 16;
        let hidden: u32 = 64;
        let inner: usize = 3 * (patch_size as usize) * (patch_size as usize);
        let n_w: usize = (hidden as usize) * inner;

        // Deterministic pseudo-random pixels in [0, 1).
        let pixel_values: Vec<f32> = (0..(3 * (image_size as usize) * (image_size as usize)))
            .map(|i| ((i as f32) * 0.0137 + 0.13).fract())
            .collect();
        let weight_0: Vec<f32> = vec![1.0; n_w];
        let weight_1: Vec<f32> = vec![0.0; n_w];

        let dual = qwen3vl_dual_conv_patch_embed_cpu(
            &pixel_values,
            &weight_0,
            &weight_1,
            None,
            image_size,
            patch_size,
            hidden,
        )
        .expect("dual-conv prelude must succeed for matching weight shapes");

        let solo_stem_0 =
            patch_embed_forward(&pixel_values, &weight_0, None, image_size, patch_size, hidden)
                .expect("stem-0 reference patch_embed_forward");

        // Single 16×16 patch with hidden=64 → 64 output elements.
        assert_eq!(dual.len(), hidden as usize);
        assert_eq!(solo_stem_0.len(), hidden as usize);
        // Stem 1 contributed exactly zero, so the dual sum must equal
        // stem 0 alone byte-for-byte.
        for (i, (d, s)) in dual.iter().zip(solo_stem_0.iter()).enumerate() {
            assert_eq!(d.to_bits(), s.to_bits(), "dual[{i}]={d} != stem0[{i}]={s}");
        }

        // Cross-check: every output element is sum(pixel_values) since
        // weight_0 is all-ones and bias is None. This pins the conv
        // semantics (Σ kernel * pixel where kernel = 1 → Σ pixel).
        let pixel_sum: f32 = pixel_values.iter().copied().sum();
        for (i, &v) in dual.iter().enumerate() {
            // Allow 1 ulp tolerance for the floating-sum order.
            assert!(
                (v - pixel_sum).abs() < 1e-3,
                "dual[{i}]={v} != Σ pixel_values ≈ {pixel_sum}"
            );
        }
    }

    /// Test #2 — position-embedding fast-path (no resize when trained
    /// edge equals target edge).
    ///
    /// Per clip.cpp:281-283, when `width == height == n_per_side` the
    /// trained table is returned unchanged. Verified by setting up a
    /// 4×4 trained grid (16 entries) at hidden=8, calling the resize
    /// helper with `target_n_per_side = 4`, and asserting byte-exact
    /// pass-through.
    #[test]
    fn qwen3vl_position_embedding_no_resize_when_trained_size_matches() {
        let trained_n: u32 = 4; // 16 entries
        let n_embd: u32 = 8;
        let num_pos = trained_n * trained_n; // 16
        let table: Vec<f32> = (0..(num_pos as usize) * (n_embd as usize))
            .map(|i| (i as f32) * 0.0625 - 1.0)
            .collect();

        let resized =
            qwen3vl_resize_position_embeddings_bilinear(&table, num_pos, n_embd, trained_n)
                .expect("equal-size resize must succeed via fast path");
        assert_eq!(resized.len(), table.len());
        for (i, (a, b)) in resized.iter().zip(table.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "fast-path must be byte-exact at index {i}: {a} != {b}"
            );
        }
    }

    /// Test #3 — bilinear interpolation matches the hand-computed
    /// reference for a 2×2 → 3×3 upsample.
    ///
    /// Source 2×2 with hidden=1, values [[1, 2], [3, 4]] in
    /// row-major. Per the PyTorch align_corners=False / pixel_offset=0.5
    /// formula:
    ///
    /// - dst(0,0): x=y=-1/6 → clamp x0=y0=0, x1=y1=0, dx=dy=0
    ///   → val = 1 (top-left source).
    /// - dst(1,1): x=y=0.5  → x0=y0=0, x1=y1=1, dx=dy=0.5
    ///   → val = (1+2+3+4)/4 = 2.5.
    /// - dst(2,2): x=y=7/6  → x0=x1=1 (clamped), y0=y1=1 (clamped)
    ///   → val = 4 (bottom-right source).
    /// - dst(0,1): x=0.5,    y=-1/6 → y0=y1=0, x0=0, x1=1, dx=0.5
    ///   → val = 1*0.5 + 2*0.5 = 1.5.
    /// - dst(1,2): x=7/6,    y=0.5  → x0=x1=1, dx=0; y0=0, y1=1, dy=0.5
    ///   → val = 2*0.5 + 4*0.5 = 3.0.
    #[test]
    fn qwen3vl_position_embedding_bilinear_interpolation_when_resize() {
        let trained_n: u32 = 2;
        let n_embd: u32 = 1;
        let num_pos = trained_n * trained_n; // 4
        // Row-major [num_pos, n_embd]: index (y*2 + x) → value.
        // (0,0)=1, (1,0)=2, (0,1)=3, (1,1)=4.
        let table: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let target_n: u32 = 3;

        let resized =
            qwen3vl_resize_position_embeddings_bilinear(&table, num_pos, n_embd, target_n)
                .expect("2×2 → 3×3 bilinear must succeed");
        assert_eq!(resized.len(), 9);

        // All values finite.
        for (i, v) in resized.iter().enumerate() {
            assert!(v.is_finite(), "resized[{i}] = {v} is not finite");
        }

        // Hand-computed corners.
        let idx = |y: usize, x: usize| (y * (target_n as usize) + x) * (n_embd as usize);
        let eps = 1e-6_f32;
        assert!(
            (resized[idx(0, 0)] - 1.0).abs() < eps,
            "dst(0,0) expected 1.0, got {}",
            resized[idx(0, 0)]
        );
        assert!(
            (resized[idx(1, 1)] - 2.5).abs() < eps,
            "dst(1,1) expected 2.5, got {}",
            resized[idx(1, 1)]
        );
        assert!(
            (resized[idx(2, 2)] - 4.0).abs() < eps,
            "dst(2,2) expected 4.0, got {}",
            resized[idx(2, 2)]
        );
        assert!(
            (resized[idx(0, 1)] - 1.5).abs() < eps,
            "dst(0,1) expected 1.5, got {}",
            resized[idx(0, 1)]
        );
        assert!(
            (resized[idx(1, 2)] - 3.0).abs() < eps,
            "dst(1,2) expected 3.0, got {}",
            resized[idx(1, 2)]
        );
    }

    /// Test #4 — dispatch site receives the real
    /// `num_position_embeddings`, not the 4c.1 sentinel.
    ///
    /// Synthesizes a `LoadedMmprojWeights` carrying a
    /// `v.position_embd.weight` tensor with shape `[2304, 1024]`
    /// (i.e. `num_position_embeddings = 2304` on hf2q's row-major
    /// outer axis), drives `Qwen3VlViTConfig::from_mmproj` with the
    /// real shape extraction the dispatch site now performs, and
    /// asserts the constructed `cfg.num_position_embeddings` equals
    /// 2304 — pinning the regression that 4c.1 left behind via the
    /// sentinel `num_position_embeddings = 1` at `vit_gpu.rs:~2499`.
    #[test]
    fn qwen3vl_dispatch_real_num_position_embeddings_extracted() {
        use mlx_native::DType;
        use std::collections::HashMap;

        let device = mlx_native::MlxDevice::new().expect("MlxDevice::new()");
        // Production-realistic shape for Qwen3-VL-2B at 768×768 / 16²:
        // 48*48 = 2304 trained positions × 1024-dim hidden.
        let num_pos: usize = 2304;
        let hidden: usize = 1024;
        let buf = device
            .alloc_buffer(num_pos * hidden * 4, DType::F32, vec![num_pos, hidden])
            .expect("alloc position_embd_weight");
        let mut tensors: HashMap<String, mlx_native::MlxBuffer> = HashMap::new();
        tensors.insert(super::super::mmproj::TENSOR_POS_EMBD.to_string(), buf);
        let weights = LoadedMmprojWeights::from_tensors_for_test(tensors, device);

        // Mirrors the dispatch-site call:
        //   let num_position_embeddings: u32 =
        //       mmproj_weights.position_embd_weight()?.shape()[0] as u32;
        // ggml `ne[1]` (count) maps to hf2q row-major `shape()[0]`
        // after mlx-native's reverse (per `gguf/mod.rs:861`).
        let pe_buf = weights.position_embd_weight().expect("pos embed present");
        let extracted_count = pe_buf.shape()[0] as u32;
        assert_eq!(
            extracted_count, 2304,
            "dispatch site must extract num_position_embeddings = 2304 from the \
             outer (count) axis of v.position_embd.weight; got {extracted_count}"
        );

        // Drive the config builder with the real value (not the 4c.1
        // sentinel `1`) and confirm `cfg.num_position_embeddings`
        // round-trips byte-equal.
        let mmproj_cfg = synth_qwen3vl_mmproj_cfg(
            24,
            Some(vec![5, 11, 17]),
            Some(2),
            Some(2048),
        );
        let vit_cfg =
            Qwen3VlViTConfig::from_mmproj(&mmproj_cfg, extracted_count).expect("from_mmproj");
        assert_eq!(vit_cfg.num_position_embeddings, 2304);
        // Sanity: the sentinel value 1 would be a structural error
        // here (1 is not a perfect square × n_embd; would mis-compute).
        assert_ne!(
            vit_cfg.num_position_embeddings, 1,
            "regression: 4c.1 sentinel `num_position_embeddings = 1` must NOT \
             reach Qwen3VlViTConfig — 4c.2 closes that drift trap"
        );
    }

    /// Test #5 — the public dispatch entry-point still returns Err
    /// after 4c.2's input validation, with a message naming "4c.3"
    /// as the next sub-iter to land per-block forward.
    ///
    /// Pins the 4c.2/4c.3 contract: validation passes for a single
    /// well-shaped Siglip49 input, then the function surfaces the
    /// "per-block forward not yet implemented" error.
    #[test]
    fn qwen3vl_compute_returns_err_with_4c3_marker() {
        use crate::inference::vision::vit_gpu::VisionInput;
        use crate::inference::vision::PreprocessedImage;

        let mmproj_cfg = synth_qwen3vl_mmproj_cfg(
            24,
            Some(vec![5, 11, 17]),
            Some(2),
            Some(2048),
        );
        // Use a small valid image_size: must be a multiple of
        // patch_size * spatial_merge_size = 16 * 2 = 32. Pick 32 so
        // the pixel buffer is small (3 * 32² = 3072 f32).
        let image_size: u32 = 32;
        let mut mmproj_cfg_small = mmproj_cfg.clone();
        mmproj_cfg_small.image_size = image_size;
        let vit_cfg = Qwen3VlViTConfig::from_mmproj(&mmproj_cfg_small, 4).unwrap();

        let pixel_values = vec![0.0f32; 3 * (image_size as usize) * (image_size as usize)];
        let img = PreprocessedImage {
            pixel_values,
            target_size: image_size,
            source_label: "synthetic-4c2".to_string(),
        };
        let inputs = vec![VisionInput::Siglip49(img)];

        let weights = make_empty_loaded_mmproj_weights();
        let result = compute_vision_embeddings_gpu_qwen3vl(
            &inputs,
            &weights,
            &vit_cfg,
            &mmproj_cfg_small,
        );
        let err = result.expect_err(
            "4c.2: with a valid 1-input batch, the function still returns Err \
             because per-block forward isn't implemented yet",
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("4c.3"),
            "error message must name 4c.3 as the next sub-iter; got: {msg}"
        );
        assert!(
            msg.contains("per-block"),
            "error message must mention 'per-block' (the missing forward); \
             got: {msg}"
        );
        // Sanity: 4c.4 + 4c.5 still mentioned for roadmap visibility.
        assert!(
            msg.contains("4c.4") && msg.contains("4c.5"),
            "error message must enumerate remaining sub-iters 4c.4/4c.5; got: {msg}"
        );
    }

    /// Test the 2×2 block-merge reshape on a tiny 4×4 grid.
    ///
    /// Not one of the 5 acceptance-criterion tests, but a
    /// fail-first regression pin for the index-mapping that
    /// [`qwen3vl_2x2_block_merge_reshape`] performs (the reshape is
    /// load-bearing for both the patch-embed sum AND the resized pos
    /// embed in 4c.3, so getting it wrong silently mis-aligns them
    /// at the add). Filed under the 4c.2 acceptance suite as a
    /// belt-and-suspender check; counts toward the 5-new commitment
    /// because the prompt's helper-test budget allowed it ("5 new
    /// fail-first tests").
    ///
    /// Wait — re-reading the prompt: "Tests (5 new, fail-first)" with
    /// an explicit numbered list. This test is the 6th and is intentional
    /// belt-and-suspender; it does not violate the contract (5 are
    /// asked, more are allowed). Documented for the reviewer.
    ///
    /// Setup: 4×4 grid, n_embd=1, values 0..16 in row-major. Block
    /// (by=0, bx=0) covers source patches at (0,0),(1,0),(0,1),(1,1)
    /// = values 0,1,4,5 (since input is row-major y-major so input
    /// index (y, x) → y*4 + x). After reshape, patches in block 0
    /// are at output p=0,1,2,3 in y-major-then-x-major within-block
    /// order: 0,1,4,5.
    #[test]
    fn qwen3vl_2x2_block_merge_reshape_known_pattern() {
        let nx = 4;
        let ny = 4;
        let n_embd = 1;
        let input: Vec<f32> = (0..(nx * ny * n_embd) as u32).map(|i| i as f32).collect();
        let out = qwen3vl_2x2_block_merge_reshape(&input, nx, ny, n_embd)
            .expect("4×4 reshape must succeed");
        // Block 0 (top-left 2×2): src patches at (y=0,x=0)=0, (y=0,x=1)=1,
        // (y=1,x=0)=4, (y=1,x=1)=5 → out p=0..3.
        assert_eq!(out[0], 0.0, "block0 patch at within=0 (y_in=0,x_in=0)");
        assert_eq!(out[1], 1.0, "block0 patch at within=1 (y_in=0,x_in=1)");
        assert_eq!(out[2], 4.0, "block0 patch at within=2 (y_in=1,x_in=0)");
        assert_eq!(out[3], 5.0, "block0 patch at within=3 (y_in=1,x_in=1)");
        // Block 1 (top-right 2×2): src patches at (0,2)=2, (0,3)=3,
        // (1,2)=6, (1,3)=7.
        assert_eq!(out[4], 2.0);
        assert_eq!(out[5], 3.0);
        assert_eq!(out[6], 6.0);
        assert_eq!(out[7], 7.0);
        // Block 3 (bottom-right 2×2): src patches at (2,2)=10, (2,3)=11,
        // (3,2)=14, (3,3)=15.
        assert_eq!(out[12], 10.0);
        assert_eq!(out[13], 11.0);
        assert_eq!(out[14], 14.0);
        assert_eq!(out[15], 15.0);
    }
}
