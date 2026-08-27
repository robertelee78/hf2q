//! CHW float-tensor preprocessing for ViT-family vision encoders.
//!
//! Decodes image bytes, resizes to `target_size × target_size`, converts
//! to RGB8, normalizes each channel with the supplied mean/std, and
//! transposes HWC → CHW into a flat `Vec<f32>` of length `3 × size × size`.
//!
//! Matches the standard HF `AutoImageProcessor` behavior for ViT-class
//! models; the only deliberate simplification is we always rescale to the
//! exact `target_size` (no "crop-to-aspect" mode — a later iter can add
//! that when a model needs it).
//!
//! # Normalization constants
//!
//! - `GEMMA4_VISION_CONFIG` — ImageNet mean/std, 896×896 input
//!   (Gemma 4 MoE vision tower default — fixed-resolution SigLIP-49 path).
//! - Other per-model configs land alongside their forward-pass code
//!   (e.g. `NOMIC_VISION_CONFIG` when a Nomic vision encoder ports).
//!
//! # Gemma4V (Gemma 4.6 / Qwen-3.6 era hybrid) preprocessing
//!
//! `preprocess_gemma4v` is a SIBLING entry point — separate from the
//! fixed-resolution `preprocess_rgb_chw` path. It implements the
//! variable-resolution patchifier the peer uses for its gemma4v
//! projector type:
//!
//!   1. Patchify image at native resolution into `patch_size × patch_size`
//!      tiles. Compute `(n_x, n_y)` such that `n_x * n_y` lies in
//!      `[token_min, token_max]` (typically `[252, 280]`); downscale
//!      preserving aspect ratio when the native grid exceeds the cap.
//!   2. Each pixel is mapped `[0, 1] → [-3, +1]` via `4x − 3`. This is
//!      the algebraic collapse of the peer's two-step scale-bias chain:
//!      u8→f32 conversion with mean=std=[0.5, 0.5, 0.5] yields `2x − 1`,
//!      and a `(2.0, -1.0)` scale-bias then
//!      applies `2y − 1` on top of that, composing to
//!      `4x − 3`. The SigLIP-49 fixed-res path's `GEMMA4_VISION_CONFIG`
//!      stops at the `2x − 1` step (no scale-bias follow-up) and is
//!      therefore NOT byte-identical to the gemma4v variable-res path
//!      — they target different model graphs (ADR-005 Phase 2c iter-125,
//!      W56).
//!   3. Returns a `[N_patches, patch_size² × 3]` flat patch buffer plus
//!      per-patch `pos_x` and `pos_y` index arrays so the dual
//!      position-embed lookup
//!      and the per-axis 2D RoPE step can reuse them
//!      downstream without re-deriving the (px, py) → (pos_x, pos_y)
//!      mapping.
//!
//! The existing `preprocess_rgb_chw` is unchanged.
//!
//! # ADR-005 Phase 2c iter-121 (W52) — peer-parity resize
//!
//! Earlier iters used `image::imageops::FilterType::Triangle` for the
//! gemma4v resize step. That's a separable triangle filter with
//! pixel-center sampling and round-to-nearest output — which does NOT
//! match the peer's dynamic-size preprocessor. The
//! peer's algorithm is corner-aligned bilinear interpolation followed
//! by truncation-to-uint8. For sparse-signal fixtures
//! (e.g. four corner dots on white), the difference between
//! corner-aligned + truncation and center-aligned + round-to-nearest
//! is large enough to flip patch-level pixel values and produce a
//! qualitatively different ViT input — exactly what was observed in
//! iter-117 through iter-120 (hf2q text "image-blind" vs llama-mtmd-cli
//! "square frame made of four"). This iter ports the byte-faithful
//! algorithm into hf2q so the variable-resolution patch tensor matches
//! `llama-mtmd-cli`'s for the same input bytes.

use anyhow::{anyhow, Result};
use image::{imageops::FilterType, GenericImageView, ImageBuffer, ImageFormat, Rgb, RgbImage};
use std::io::Cursor;

const MAX_DECODED_IMAGE_PIXELS: u64 = 16_777_216;

fn validate_decoded_image_extent(width: u32, height: u32) -> Result<()> {
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| anyhow!("decoded image dimensions overflow: {width}x{height}"))?;
    if pixels == 0 || pixels > MAX_DECODED_IMAGE_PIXELS {
        return Err(anyhow!(
            "decoded image dimensions {width}x{height} exceed the {}-pixel safety cap",
            MAX_DECODED_IMAGE_PIXELS
        ));
    }
    Ok(())
}

fn validate_decode_dimensions(bytes: &[u8]) -> Result<(u32, u32)> {
    let reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| anyhow!("inspect image header: {error}"))?;
    let (width, height) = reader
        .into_dimensions()
        .map_err(|error| anyhow!("inspect image dimensions: {error}"))?;
    validate_decoded_image_extent(width, height)?;
    Ok((width, height))
}

/// Preprocessing knobs for a specific ViT model family.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessConfig {
    /// Square input side length the ViT expects (e.g. 224, 336, 518, 896).
    pub target_size: u32,
    /// Per-channel mean in `[R, G, B]` order, in [0, 1] pixel-normalized
    /// scale. Typical ImageNet: `[0.485, 0.456, 0.406]`.
    pub mean: [f32; 3],
    /// Per-channel std in `[R, G, B]` order, in [0, 1] scale. ImageNet:
    /// `[0.229, 0.224, 0.225]`.
    pub std: [f32; 3],
}

/// Gemma 4 MoE vision tower preprocessing. Verified against the HF
/// `Gemma3ImageProcessor` defaults for the 26B MoE variant.
pub const GEMMA4_VISION_CONFIG: PreprocessConfig = PreprocessConfig {
    target_size: 896,
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
};

/// Decode image bytes and preprocess to a CHW float tensor.
///
/// Returns `Vec<f32>` of length `3 * config.target_size * config.target_size`,
/// layout `[C=3, H=size, W=size]` (row-major within each channel; channels
/// concatenated). Caller reshapes as needed for the ViT's patch stem.
///
/// # Errors
///
/// - Unrecognized image format (not PNG / JPEG).
/// - Decoding fails (truncated / corrupt).
/// - `target_size == 0` or exceeds `u16::MAX` (defensive).
pub fn preprocess_rgb_chw(bytes: &[u8], config: &PreprocessConfig) -> Result<Vec<f32>> {
    if config.target_size == 0 || config.target_size > u16::MAX as u32 {
        return Err(anyhow!(
            "invalid target_size {}: must be in 1..=65535",
            config.target_size
        ));
    }

    // Sniff format from the first few bytes. `image::guess_format` handles
    // PNG (0x89 PNG) and JPEG (0xFF D8 FF) signatures.
    let fmt = image::guess_format(bytes).map_err(|e| anyhow!("guess_format: {e}"))?;
    match fmt {
        ImageFormat::Png | ImageFormat::Jpeg => {}
        other => {
            return Err(anyhow!(
                "image format {:?} is not supported by this build (only PNG + JPEG)",
                other
            ));
        }
    }

    validate_decode_dimensions(bytes)?;
    let img = image::load_from_memory(bytes).map_err(|e| anyhow!("decode image: {e}"))?;
    let (_w, _h) = img.dimensions();

    // Resize to target × target. `FilterType::Triangle` = bilinear;
    // matches HF's default `Image.BILINEAR` resize mode.
    let resized = img.resize_exact(config.target_size, config.target_size, FilterType::Triangle);
    // Drop alpha and ensure 8-bit depth.
    let rgb = resized.to_rgb8();
    let size = config.target_size as usize;
    let hw = size * size;

    // HWC → CHW + per-channel normalize.
    let mut out = vec![0f32; 3 * hw];
    for (y, row) in rgb.rows().enumerate() {
        for (x, pix) in row.enumerate() {
            let channels = [
                pix[0] as f32 / 255.0,
                pix[1] as f32 / 255.0,
                pix[2] as f32 / 255.0,
            ];
            let idx = y * size + x;
            for (c, &channel) in channels.iter().enumerate() {
                out[c * hw + idx] = (channel - config.mean[c]) / config.std[c];
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Gemma4V variable-resolution preprocessing
// ---------------------------------------------------------------------------

/// Knobs for the gemma4v patchifier. Defaults to the peer's
/// gemma4v projector settings:
///   - `patch_size = 16`
///   - `n_merge = 3` (pool kernel size; pre-pool patch grid axes must
///     be multiples of `n_merge` so the n_merge×n_merge avg-pool
///     produces an exact integer post-pool grid)
///   - `token_min = 252`, `token_max = 280` — **post-pool** token
///     bounds. The peer's token-limit
///     helper converts these to
///     pixel bounds:
///       `image_min_pixels = 252 * patch_size² * n_merge² = 580608`
///       `image_max_pixels = 280 * patch_size² * n_merge² = 645120`.
///     These pixel bounds are what the ratio-preserving size
///     computation actually
///     consumes when picking the resized image dims; the resulting
///     pre-pool patch grid `(n_x, n_y)` is therefore aligned to
///     `align_size = patch_size * n_merge = 48` pixels — i.e. each
///     axis is automatically a multiple of `n_merge`.
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4vPreprocessConfig {
    /// Patch edge length (square; pixel count per patch is `patch_size²`).
    pub patch_size: u32,
    /// Pool kernel size (n_merge × n_merge avg-pool after the ViT).
    /// Pre-pool patch grid axes are constrained to be multiples of this.
    pub n_merge: u32,
    /// Lower bound on the **post-pool** token count, i.e.
    /// `(n_x / n_merge) * (n_y / n_merge)` (inclusive). Small images
    /// are upscaled to meet this floor — the peer does the same
    /// (small inputs degrade quality without it).
    pub token_min: u32,
    /// Upper bound on the **post-pool** token count (inclusive). Large
    /// images are downscaled (preserving aspect ratio) until the
    /// post-pool grid fits.
    pub token_max: u32,
}

/// Default gemma4v config — locked to the peer's reference values so
/// the hf2q output matches the GGUF tower's expected token budget.
pub const GEMMA4V_PREPROCESS_DEFAULT: Gemma4vPreprocessConfig = Gemma4vPreprocessConfig {
    patch_size: 16,
    n_merge: 3,
    token_min: 252,
    token_max: 280,
};

/// Output of `preprocess_gemma4v` — a variable-resolution patch tensor
/// plus the per-patch `(pos_x, pos_y)` index arrays the dual position
/// embedding and 2D RoPE both consume.
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4vPreprocessed {
    /// Flat `[N_patches, 3 × patch_size²]` tensor in row-major order.
    /// Within each row, the `3 × patch_size²` inner dim iterates as
    /// `(c, dy, dx)` — channel-major, pixel-minor — matching the
    /// CHW im2col layout the peer's conv-2d produces over
    /// `inp_raw[ne0=W, ne1=H, ne2=C]`
    /// (`iic*(KH*KW) + ikh*KW + ikw` flat indexing).
    /// The GGUF-stored `v.patch_embd.weight` is CHW per output row
    /// (writer applies `permute(0, 3, 1, 2)`); the linear matmul `out[n][o]
    /// = Σ_k weight[o][k] · patches[n][k]` requires both sides to share
    /// the same `(c, dy, dx)` flat indexing for `k` to refer to the
    /// same spatial-channel position. ADR-005 Phase 2c iter-126 (W57)
    /// fixed the prior HWC `(dy, dx, c)` ordering — that ordering
    /// matched candle's HWC reshape but candle reads HF source weight
    /// directly without the CHW permute, so HWC↔HWC worked there. Hf2q
    /// reads the CHW-permuted GGUF weight, so it requires CHW patches.
    pub patches: Vec<f32>,
    /// Per-patch X-axis position index, length `N_patches`. Patch
    /// `(px, py)` (column, row) at the post-resize grid maps to
    /// `pos_x[py*n_x + px] = px`.
    pub pos_x: Vec<u32>,
    /// Per-patch Y-axis position index, length `N_patches`. Patch
    /// `(px, py)` maps to `pos_y[py*n_x + px] = py`.
    pub pos_y: Vec<u32>,
    /// Patches along the X (column) axis after resizing.
    pub n_x: u32,
    /// Patches along the Y (row) axis after resizing.
    pub n_y: u32,
}

impl Gemma4vPreprocessed {
    /// `n_x * n_y` — convenience for callers that only need the total.
    pub fn n_patches(&self) -> u32 {
        self.n_x.saturating_mul(self.n_y)
    }
}

/// Decode + variable-resolution patchify for the gemma4v vision tower.
///
/// Implements the peer's gemma4v graph input contract:
///
///   - Decode image bytes (PNG/JPEG only, same restriction as
///     `preprocess_rgb_chw`).
///   - Compute the largest aspect-ratio-preserving size `(W, H)` such
///     that `(W/patch) * (H/patch) ≤ token_max`. If the resulting
///     patch count would fall below `token_min`, scale up to the
///     smallest size that meets the floor. `W` and `H` are rounded
///     down to multiples of `patch_size` so patchification is exact.
///   - Resize via bilinear (the peer's resize algorithm for the
///     gemma4v projector).
///   - Patchify into `[N_patches, patch_size² × 3]` with pixel layout
///     `(dy, dx, c)` per patch (matches candle's reshape, see
///     `Gemma4vPreprocessed::patches` doc).
///   - Pixel-scale `4x − 3` so the per-patch values are in `[-3, +1]`.
///     This is the algebraic collapse of the peer's two-step chain:
///     u8→f32 conversion with mean=std=[0.5,0.5,0.5]
///     gives `2x − 1` (range `[-1, +1]`), and a
///     `(2.0, -1.0)` scale-bias then applies `2y − 1` on top, yielding
///     `4x − 3`. Folded here into a single CPU pass so the GPU patch-embd
///     conv sees byte-faithful inputs (ADR-005 Phase 2c iter-125, W56).
///
/// # Errors
///
/// - `cfg.patch_size == 0` or `cfg.token_max == 0` (defensive).
/// - `cfg.token_min > cfg.token_max` (caller bug).
/// - Image format not PNG/JPEG.
/// - Image dimension `< patch_size` after sniff (cannot patchify a
///   single-pixel image into ≥1 patch of `patch_size² × 3` pixels).
pub fn preprocess_gemma4v(
    bytes: &[u8],
    cfg: &Gemma4vPreprocessConfig,
) -> Result<Gemma4vPreprocessed> {
    if cfg.patch_size == 0 {
        return Err(anyhow!("gemma4v: patch_size must be > 0"));
    }
    if cfg.n_merge == 0 {
        return Err(anyhow!("gemma4v: n_merge must be > 0"));
    }
    if cfg.token_max == 0 {
        return Err(anyhow!("gemma4v: token_max must be > 0"));
    }
    if cfg.token_min > cfg.token_max {
        return Err(anyhow!(
            "gemma4v: token_min ({}) > token_max ({})",
            cfg.token_min,
            cfg.token_max
        ));
    }

    let fmt = image::guess_format(bytes).map_err(|e| anyhow!("guess_format: {e}"))?;
    match fmt {
        ImageFormat::Png | ImageFormat::Jpeg => {}
        other => {
            return Err(anyhow!(
                "image format {:?} is not supported by gemma4v preprocess (only PNG + JPEG)",
                other
            ));
        }
    }
    validate_decode_dimensions(bytes)?;
    let img = image::load_from_memory(bytes).map_err(|e| anyhow!("decode image: {e}"))?;
    let (orig_w, orig_h) = img.dimensions();
    if orig_w == 0 || orig_h == 0 {
        return Err(anyhow!("gemma4v: image has zero dimension"));
    }

    let p = cfg.patch_size;
    let (n_x, n_y) =
        compute_gemma4v_patch_grid(orig_w, orig_h, p, cfg.n_merge, cfg.token_min, cfg.token_max)?;
    let target_w = n_x * p;
    let target_h = n_y * p;

    // ADR-005 Phase 2c iter-121 (W52): byte-faithful match against
    // `llama-mtmd-cli`'s dynamic-size preprocessor. The peer's
    // resize runs with padding enabled
    // (the default — gemma4v's projector init
    // does not override it), so we replicate the
    // padded-resize branch with a black pad color ({0,0,0}).
    // Bilinear sampling is corner-aligned
    // with truncation-to-uint8, NOT the image crate's
    // `FilterType::Triangle` (center-aligned, round-to-nearest).
    let src_rgb = img.to_rgb8();
    let rgb = resize_bilinear_pad_peer(&src_rgb, target_w, target_h, [0, 0, 0]);

    let n_patches = (n_x as usize) * (n_y as usize);
    let p_us = p as usize;
    let inner = p_us * p_us * 3;
    let mut patches = vec![0f32; n_patches * inner];
    let mut pos_x = Vec::with_capacity(n_patches);
    let mut pos_y = Vec::with_capacity(n_patches);

    // Patchify in (py, px, c, dy, dx) order — channel-major within
    // each patch row to match the CHW im2col layout consumed by
    // the peer's patch_embd conv (see `Gemma4vPreprocessed::patches`
    // doc) and the CHW-permuted GGUF weight (`gguf.rs:995`).
    // ADR-005 Phase 2c iter-126
    // (W57): prior HWC `(dy, dx, c)` ordering was a mismatch against
    // the GGUF weight's CHW per-output-row layout — uniform-channel
    // patches still produced byte-identical output (Σ over channels
    // collapses), but every non-grayscale patch diverged from peer
    // (parity probe stage 01 max_abs=64.6 with correlation 0.967 even
    // after iter-125's `4x − 3` fix). This fix is the index permutation
    // patches[(c, dy, dx)] = (pix(dy, dx)[c]/255)*4 - 3.
    let p2 = p_us * p_us; // patch_size² (pixels per channel-plane in one patch)
    for py in 0..n_y {
        for px in 0..n_x {
            let patch_idx = (py as usize) * (n_x as usize) + (px as usize);
            let row_base = patch_idx * inner;
            pos_x.push(px);
            pos_y.push(py);
            for dy in 0..p {
                for dx in 0..p {
                    let img_x = px * p + dx;
                    let img_y = py * p + dy;
                    let pix = rgb.get_pixel(img_x, img_y);
                    let pos_in_plane = (dy as usize) * p_us + (dx as usize);
                    // ADR-005 Phase 2c iter-125 (W56): byte-faithful match
                    // against the peer's two-step scale-bias chain.
                    // Step 1 — u8→f32 conversion with
                    // mean=std=[0.5,0.5,0.5]:
                    //     y = (pix/255 - 0.5)/0.5  = 2*pix/255 - 1   ∈ [-1, +1]
                    // Step 2 — a `(2.0, -1.0)` scale-bias:
                    //     z = 2*y + (-1)           = 2*(2*pix/255 - 1) - 1
                    //                              = 4*pix/255 - 3    ∈ [-3, +1]
                    // Folded into a single CPU expression: `4x − 3`. Iter-124
                    // parity probe (W55, 625f94a) showed hf2q was one full
                    // `2x − 1` step short, producing the algebraic identity
                    // hf2q = (peer + 1) / 2; i.e. peer = 2*hf2q + 1.
                    //
                    // `0 * p2` is intentional symmetry with the `1 * p2` /
                    // `2 * p2` lines below — visually parallel CHW layout
                    // makes the channel-stride pattern obvious. Clippy's
                    // `erasing_op` is a false positive here.
                    #[allow(clippy::erasing_op, clippy::identity_op)]
                    {
                        patches[row_base + 0 * p2 + pos_in_plane] =
                            (pix[0] as f32 / 255.0) * 4.0 - 3.0;
                        patches[row_base + 1 * p2 + pos_in_plane] =
                            (pix[1] as f32 / 255.0) * 4.0 - 3.0;
                        patches[row_base + 2 * p2 + pos_in_plane] =
                            (pix[2] as f32 / 255.0) * 4.0 - 3.0;
                    }
                }
            }
        }
    }

    Ok(Gemma4vPreprocessed {
        patches,
        pos_x,
        pos_y,
        n_x,
        n_y,
    })
}

/// Compute `(n_x, n_y)` — the pre-pool patch grid — for a
/// `(orig_w, orig_h)` image given a patch edge `p`, pool kernel size
/// `n_merge`, and `[token_min, token_max]` **post-pool** token bounds.
///
/// This must stay byte-faithful to the peer's ratio-preserving size
/// computation for the gemma4v projector. The pixel bounds come from
/// the post-pool token bounds via
///   `image_min_pixels = token_min * patch_size² * n_merge²`
///   `image_max_pixels = token_max * patch_size² * n_merge²`.
///
/// Algorithm (this is "smart_resize" from the HF transformers code):
///   - `align_size = p * n_merge` (= 48 for gemma4v defaults).
///   - "Always align up first": `h_bar = max(align, round_by(h))`,
///     `w_bar = max(align, round_by(w))`, where `round_by(x)` =
///     `round(x/align)*align`.
///   - If `h_bar*w_bar > max_pixels`: `beta = sqrt(h*w / max_pixels)`;
///     `h_bar = max(align, floor_by(h/beta))`,
///     `w_bar = max(align, floor_by(w/beta))`.
///   - Else if `h_bar*w_bar < min_pixels`:
///     `beta = sqrt(min_pixels / (h*w))`;
///     `h_bar = ceil_by(h*beta)`, `w_bar = ceil_by(w*beta)`.
///
/// Returned `(n_x, n_y) = (w_bar/p, h_bar/p)` — both are multiples of
/// `n_merge` by construction, matching the gemma4v pool kernel's
/// invariant. Post-pool token count = `(n_x/n_merge) * (n_y/n_merge)`.
fn compute_gemma4v_patch_grid(
    orig_w: u32,
    orig_h: u32,
    p: u32,
    n_merge: u32,
    token_min: u32,
    token_max: u32,
) -> Result<(u32, u32)> {
    let align_size: u64 = (p as u64) * (n_merge as u64);
    if align_size == 0 {
        return Err(anyhow!(
            "gemma4v patch grid: align_size = patch_size ({p}) * n_merge ({n_merge}) is zero"
        ));
    }
    // Pixel-area bounds: patch_area = p² * n_merge².
    let patch_area: u64 = (p as u64) * (p as u64) * (n_merge as u64) * (n_merge as u64);
    let min_pixels: u64 = (token_min as u64) * patch_area;
    let max_pixels: u64 = (token_max as u64) * patch_area;

    // Helpers: round / ceil / floor `x` to a multiple of `align_size`.
    let round_by =
        |x: f64| -> u64 { ((x / align_size as f64).round() as i64).max(0) as u64 * align_size };
    let ceil_by =
        |x: f64| -> u64 { ((x / align_size as f64).ceil() as i64).max(0) as u64 * align_size };
    let floor_by =
        |x: f64| -> u64 { ((x / align_size as f64).floor() as i64).max(0) as u64 * align_size };

    let width = orig_w as u64;
    let height = orig_h as u64;

    // "Always align up first".
    let mut h_bar: u64 = align_size.max(round_by(height as f64));
    let mut w_bar: u64 = align_size.max(round_by(width as f64));

    if h_bar * w_bar > max_pixels {
        // Shrink toward max_pixels.
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = align_size.max(floor_by(height as f64 / beta));
        w_bar = align_size.max(floor_by(width as f64 / beta));
    } else if h_bar * w_bar < min_pixels {
        // Grow toward min_pixels.
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = ceil_by(height as f64 * beta);
        w_bar = ceil_by(width as f64 * beta);
    }

    let n_x_u64 = w_bar / (p as u64);
    let n_y_u64 = h_bar / (p as u64);
    if n_x_u64 == 0 || n_y_u64 == 0 || n_x_u64 > u32::MAX as u64 || n_y_u64 > u32::MAX as u64 {
        return Err(anyhow!(
            "gemma4v patch grid: degenerate output ({} x {}) for input ({} x {})",
            n_x_u64,
            n_y_u64,
            orig_w,
            orig_h
        ));
    }
    let n_x = n_x_u64 as u32;
    let n_y = n_y_u64 as u32;

    // Defensive: post-condition. Both axes must be multiples of n_merge
    // (the gemma4v pool kernel invariant). With align_size = p*n_merge
    // and `_bar` computed via *_by_factor on align_size, this is
    // guaranteed mathematically; assert for paranoia.
    debug_assert!(
        n_x % n_merge == 0 && n_y % n_merge == 0,
        "gemma4v patch grid: ({n_x},{n_y}) not aligned to n_merge={n_merge}"
    );

    Ok((n_x, n_y))
}

// ---------------------------------------------------------------------------
// Byte-faithful peer bilinear resize (ADR-005 Phase 2c iter-121, W52)
// ---------------------------------------------------------------------------

/// Corner-aligned bilinear resize matching the peer's bilinear resize
/// byte-for-byte.
///
/// Differences vs `image::imageops::FilterType::Triangle`:
///   - **Sampling alignment**: uses `x_ratio = (src_w-1)/(target_w-1)`
///     (vertex/corner-aligned — corners coincide), NOT center-aligned
///     `(src_w/target_w)` with half-pixel offsets.
///   - **Bounds clamping**: when integer truncation puts `x0` at the last
///     column, `x1 = min(x0+1, src_w-1)` clamps to the last column, so
///     edge pixels are weighted-blended with themselves (degenerate lerp).
///   - **No antialiasing**: even when downscaling, no kernel widening —
///     this is a 2×2 nearest-neighbor lerp regardless of scale ratio.
///   - **u8 cast**: `static_cast<uint8_t>(lerp(top, bottom, yf))` — C++
///     truncation-toward-zero of a non-negative float is `floor`, NOT
///     round-to-nearest.
fn resize_bilinear_peer(src: &RgbImage, target_w: u32, target_h: u32) -> RgbImage {
    let src_w = src.width();
    let src_h = src.height();
    if target_w == 0 || target_h == 0 || src_w == 0 || src_h == 0 {
        return ImageBuffer::new(target_w.max(1), target_h.max(1));
    }
    if src_w == target_w && src_h == target_h {
        return src.clone();
    }

    // Match `mtmd-image.cpp:209-210` exactly (vertex-aligned ratio).
    let x_ratio = if target_w > 1 {
        (src_w as f32 - 1.0) / (target_w as f32 - 1.0)
    } else {
        0.0
    };
    let y_ratio = if target_h > 1 {
        (src_h as f32 - 1.0) / (target_h as f32 - 1.0)
    } else {
        0.0
    };

    let mut dst: RgbImage = ImageBuffer::new(target_w, target_h);
    let src_w_i = src_w as i32;
    let src_h_i = src_h as i32;

    for y in 0..target_h {
        for x in 0..target_w {
            let px = x as f32 * x_ratio;
            let py = y as f32 * y_ratio;

            // `std::min(static_cast<int>(px), src.nx - 1)` —
            // C++ int-cast of non-negative float is truncation = floor.
            let x0 = (px as i32).min(src_w_i - 1).max(0);
            let y0 = (py as i32).min(src_h_i - 1).max(0);
            let x1 = (x0 + 1).min(src_w_i - 1);
            let y1 = (y0 + 1).min(src_h_i - 1);

            let xf = px - (x0 as f32);
            let yf = py - (y0 as f32);

            let p00 = src.get_pixel(x0 as u32, y0 as u32).0;
            let p10 = src.get_pixel(x1 as u32, y0 as u32).0;
            let p01 = src.get_pixel(x0 as u32, y1 as u32).0;
            let p11 = src.get_pixel(x1 as u32, y1 as u32).0;

            let mut out = [0u8; 3];
            for c in 0..3 {
                // lerp(s, e, t) = s + (e - s) * t  (mtmd-image.cpp:558-560)
                let top = (p00[c] as f32) + ((p10[c] as f32) - (p00[c] as f32)) * xf;
                let bottom = (p01[c] as f32) + ((p11[c] as f32) - (p01[c] as f32)) * xf;
                let v = top + (bottom - top) * yf;
                // C++ `static_cast<uint8_t>(positive_float)` = truncation = floor.
                // Clamp to [0, 255] for paranoia (lerp of u8s in [0, 255]
                // with t ∈ [0, 1] is already in-range, but defensive).
                out[c] = v.clamp(0.0, 255.0) as u8;
            }
            dst.put_pixel(x, y, Rgb(out));
        }
    }
    dst
}

/// Resize-with-padding match for the peer's padded resize.
///
///   - Compute `scale = min(target_w/src.nx, target_h/src.ny)` —
///     fit-inside, aspect-ratio preserving.
///   - `new_w = min(ceil(src.nx * scale), target_w)`,
///     `new_h = min(ceil(src.ny * scale), target_h)`.
///   - Bilinear-resize to `(new_w, new_h)` via
///     `resize_bilinear_peer`.
///   - Allocate `target_w × target_h` filled with `pad_color`, composite
///     resized image at `((target_w - new_w)/2, (target_h - new_h)/2)`
///     (center).
///
/// For square inputs where target is square (the common gemma4v case
/// after the ratio-preserving size computation), `new_w == target_w` and
/// `new_h == target_h`, so the padding is a no-op and behavior reduces
/// to plain bilinear. For non-square inputs the center-pad is what
/// the peer emits, and we match it.
fn resize_bilinear_pad_peer(
    src: &RgbImage,
    target_w: u32,
    target_h: u32,
    pad_color: [u8; 3],
) -> RgbImage {
    let src_w = src.width();
    let src_h = src.height();
    if src_w == target_w && src_h == target_h {
        return src.clone();
    }
    if target_w == 0 || target_h == 0 || src_w == 0 || src_h == 0 {
        return ImageBuffer::new(target_w.max(1), target_h.max(1));
    }

    let scale_w = (target_w as f32) / (src_w as f32);
    let scale_h = (target_h as f32) / (src_h as f32);
    let scale = scale_w.min(scale_h);

    let new_w_f = (src_w as f32) * scale;
    let new_h_f = (src_h as f32) * scale;
    // `std::ceil` then min-clamp to target.
    let new_w = (new_w_f.ceil() as i64).min(target_w as i64).max(1) as u32;
    let new_h = (new_h_f.ceil() as i64).min(target_h as i64).max(1) as u32;

    let resized = resize_bilinear_peer(src, new_w, new_h);

    // Fill dst with pad_color.
    let mut dst: RgbImage = ImageBuffer::from_pixel(target_w, target_h, Rgb(pad_color));

    // Composite at center.
    let offset_x = ((target_w - new_w) / 2) as i32;
    let offset_y = ((target_h - new_h) / 2) as i32;
    for y in 0..new_h {
        for x in 0..new_w {
            let dx = (x as i32) + offset_x;
            let dy = (y as i32) + offset_y;
            if dx < 0 || dy < 0 || dx >= target_w as i32 || dy >= target_h as i32 {
                continue;
            }
            let p = *resized.get_pixel(x, y);
            dst.put_pixel(dx as u32, dy as u32, p);
        }
    }
    dst
}

// ---------------------------------------------------------------------------
// Qwen3-VL variable-resolution preprocessing (ADR-005 iter-224 Wedge-4d)
// ---------------------------------------------------------------------------

/// Knobs for the Qwen3-VL "smart_resize" preprocessor, sourced from
/// the peer's projector init and token-limit helper.
///
/// For Qwen3-VL the per-token budget defaults to `[8, 4096]`. The
/// pixel-area bounds are derived as
/// `image_min_pixels = 8 * patch_size² * spatial_merge_size²`,
/// `image_max_pixels = 4096 * patch_size² * spatial_merge_size²`. With
/// `patch_size=16` and `spatial_merge_size=2` (Qwen3-VL canonical):
///   - `image_min_pixels = 8 * 1024 = 8192`
///   - `image_max_pixels = 4096 * 1024 = 4_194_304`
///
/// `align_size = patch_size * spatial_merge_size` (= 32 for canonical
/// Qwen3-VL). Both output axes are multiples of `align_size`.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenVisionPreprocessConfig {
    /// ViT patch edge length (Qwen3-VL: 16). Sourced from
    /// `MmprojConfig.patch_size`.
    pub patch_size: u32,
    /// Spatial-merge degree (Qwen3-VL: 2 → 2×2 merge, 4× token
    /// reduction). Sourced from
    /// `MmprojConfig.spatial_merge_size.expect("Qwen3-VL")`.
    pub spatial_merge_size: u32,
    /// Per-channel mean (Qwen3-VL canonical: `[0.48145466, 0.4578275,
    /// 0.40821073]` — OpenAI CLIP defaults; verify per-mmproj). Sourced
    /// from `MmprojConfig.image_mean`.
    pub image_mean: [f32; 3],
    /// Per-channel std (Qwen3-VL canonical: `[0.26862954, 0.26130258,
    /// 0.27577711]`). Sourced from `MmprojConfig.image_std`.
    pub image_std: [f32; 3],
    /// Pixel-area lower bound (per peer `set_limit_image_tokens`). When
    /// the rounded-up size falls below this floor, the smart-resize
    /// upscales to meet it.
    pub image_min_pixels: u64,
    /// Pixel-area upper bound. When the rounded-up size exceeds this
    /// ceiling, the smart-resize downscales to fit.
    pub image_max_pixels: u64,
}

impl QwenVisionPreprocessConfig {
    /// Build from a parsed `MmprojConfig`. Falls back to peer's
    /// `set_limit_image_tokens(8, 4096)` defaults for the pixel bounds —
    /// these are NOT serialized in the GGUF metadata for canonical
    /// Qwen3-VL files (peer's `KEY_IMAGE_MIN_PIXELS` /
    /// `KEY_IMAGE_MAX_PIXELS` reads at clip.cpp:1294 are scoped to
    /// `DOTS_OCR` and `KIMIK25` only, NOT the QWEN*VL projectors —
    /// those use the hard-coded 8/4096 token limits at clip.cpp:1361).
    ///
    /// # Errors
    /// - `MmprojConfig.spatial_merge_size` is `None` (caller must supply
    ///   a Qwen3-VL mmproj; the dispatch site already enforces this).
    /// - `patch_size` or `spatial_merge_size` is 0.
    pub fn from_mmproj(cfg: &super::mmproj::MmprojConfig) -> anyhow::Result<Self> {
        let sm = cfg.spatial_merge_size.ok_or_else(|| {
            anyhow!(
                "QwenVisionPreprocessConfig::from_mmproj: MmprojConfig.spatial_merge_size is None \
                 — Qwen3-VL mmproj must carry `clip.vision.spatial_merge_size`"
            )
        })?;
        if cfg.patch_size == 0 || sm == 0 {
            return Err(anyhow!(
                "QwenVisionPreprocessConfig::from_mmproj: patch_size ({}) and \
                 spatial_merge_size ({}) must be > 0",
                cfg.patch_size,
                sm
            ));
        }
        // Legacy projector artifacts do not carry processor bounds. Preserve
        // their established token-derived defaults, while preferring exact
        // bounds embedded by current hf2q conversion.
        let patch_area: u64 = (cfg.patch_size as u64).pow(2) * (sm as u64).pow(2);
        let image_min_pixels: u64 = cfg
            .image_min_pixels
            .map(u64::from)
            .unwrap_or(8 * patch_area);
        let image_max_pixels: u64 = cfg
            .image_max_pixels
            .map(u64::from)
            .unwrap_or(4096 * patch_area);
        Ok(Self {
            patch_size: cfg.patch_size,
            spatial_merge_size: sm,
            image_mean: cfg.image_mean,
            image_std: cfg.image_std,
            image_min_pixels,
            image_max_pixels,
        })
    }
}

/// Output of `preprocess_qwen_vision` — variable-resolution pixel tensor in
/// CHW layout plus the post-resize patch grid the placeholder expansion
/// + 3D-mRoPE position synthesis consume.
///
/// **ADR-005 iter-225 Phase-2 (LANDED)**: the ViT
/// `compute_vision_embeddings_gpu_qwen` now accepts genuinely
/// rectangular `[3, target_h, target_w]` input. The center-pad chain
/// is GONE; this preprocessor's output is the smart-resized pixel
/// grid directly:
///
///   1. Smart-resize per peer `calc_size_preserved_ratio` → variable
///      `(target_w, target_h)` aligned to `(patch_size *
///      spatial_merge_size)`.
///   2. Bilinear-resize the source image to that variable target.
///   3. Pixel-normalize CHW → `pixel_values` (length =
///      `3 * target_w * target_h`).
///
/// The reported `n_x_token, n_y_token, n_image_tokens` are the
/// per-image post-merge grid (`target_w / (patch * sm)`,
/// `target_h / (patch * sm)`); they vary across images with different
/// aspect ratios (e.g. landscape 1024×576 → 16×9 = 144 tokens vs
/// canonical Phase-1 24×24 = 576). Total token count for a particular
/// image = `n_x_token * n_y_token`.
///
/// `target_size` is preserved as the trained ViT canvas size (=
/// `mmproj.image_size`) for backward-compat reporting; the
/// authoritative pixel-grid dimensions live in
/// `target_pixel_grid()` (= `(target_w, target_h)`).
#[derive(Debug, Clone, PartialEq)]
pub struct QwenVisionPreprocessed {
    /// Flat `[3, target_h, target_w]` row-major (CHW) tensor.
    /// Caller passes this through as `VisionInput::Siglip49(...)`'s
    /// `pixel_values` and threads `target_pixel_grid()` into
    /// `PreprocessedImage::{pixel_w, pixel_h}` so the ViT consumes the
    /// rectangular grid directly.
    pub pixel_values: Vec<f32>,
    /// Trained ViT canvas size in pixels (= `mmproj.image_size`).
    /// Retained for backward-compat reporting; consumers should read
    /// the per-image rectangular grid from `target_pixel_grid()`.
    pub target_size: u32,
    /// Smart-resized pixel grid width (post-resize, pre-patch). Aligned
    /// to `patch_size * spatial_merge_size`.
    pub target_w: u32,
    /// Smart-resized pixel grid height (post-resize, pre-patch).
    /// Aligned to `patch_size * spatial_merge_size`.
    pub target_h: u32,
    /// Post-merge token grid dimension along the W axis (=
    /// `target_w / (patch * sm)`). Used by 3D-mRoPE position
    /// synthesis: image-patch token at post-merge index `i` has
    /// position `[t, i/n_x_token, i%n_x_token, 0]` (per peer
    /// `tools/mtmd/mtmd.cpp:1295-1304`
    /// `mtmd_image_tokens_get_decoder_pos` with `MTMD_POS_TYPE_MROPE`).
    pub n_x_token: u32,
    /// Post-merge token grid dimension along the H axis (=
    /// `target_h / (patch * sm)`).
    pub n_y_token: u32,
    /// Total post-merge image token count = `n_x_token * n_y_token`.
    /// This is what `expand_image_placeholders` uses to expand the
    /// `<|image_pad|>` placeholder into a contiguous run of repeats.
    pub n_image_tokens: u32,
}

impl QwenVisionPreprocessed {
    /// Authoritative `(W, H)` pixel grid post-smart-resize. Use this
    /// to populate `PreprocessedImage::{pixel_w, pixel_h}` so the ViT
    /// consumes the rectangular content directly (Phase-2 contract).
    pub fn target_pixel_grid(&self) -> (u32, u32) {
        (self.target_w, self.target_h)
    }
}

/// Variable-resolution preprocessor for Qwen vision images.
///
/// The source processor uses variable-resolution smart resize with bicubic
/// sampling. The Phase-1 center-pad accommodation has been removed; the ViT
/// `compute_vision_embeddings_gpu_qwen` now consumes the rectangular
/// `[3, target_h, target_w]` tensor directly:
///
///   1. Decode image bytes (PNG/JPEG; `image::guess_format` rejects
///      anything else).
///   2. Compute `(target_w, target_h)` via
///      `calc_size_preserved_ratio(orig_size, align_size=patch*sm,
///      min_pixels, max_pixels)`. This is "smart_resize" from the HF
///      transformers code — aspect ratio preserved, both axes aligned
///      to `align_size`, area clamped to `[min_pixels, max_pixels]`.
///   3. Bicubic resize the source RGB to `(target_w, target_h)`.
///   4. Pixel-normalize CHW row-major into `pixel_values` (length =
///      `3 * target_w * target_h`, NO padding).
///
/// **`image_size` parameter** (preserved): retained for the
/// backward-compat single-square-canvas reporting path. When the
/// smart-resize yields exactly `(image_size, image_size)` the output
/// is byte-equivalent to the Phase-1 path. The token grid
/// `(n_x_token, n_y_token)` is now PER-IMAGE, derived from
/// `(target_w / stride, target_h / stride)` — different aspect ratios
/// produce different token counts.
///
/// # Errors
/// - Image format not PNG/JPEG.
/// - `image::load_from_memory` decode failure (truncated / corrupt).
/// - Original image has zero dimension.
/// - `cfg.patch_size * cfg.spatial_merge_size == 0`.
/// - `cfg.image_min_pixels > cfg.image_max_pixels`.
/// - `image_size` is not a positive multiple of `patch_size *
///   spatial_merge_size` (preserved as a sanity check on the canonical
///   canvas).
pub fn preprocess_qwen_vision(
    bytes: &[u8],
    cfg: &QwenVisionPreprocessConfig,
    image_size: u32,
) -> Result<QwenVisionPreprocessed> {
    if cfg.patch_size == 0 || cfg.spatial_merge_size == 0 {
        return Err(anyhow!(
            "qwen3vl preprocess: patch_size ({}) and spatial_merge_size ({}) \
             must both be > 0",
            cfg.patch_size,
            cfg.spatial_merge_size
        ));
    }
    let stride = cfg.patch_size * cfg.spatial_merge_size;
    if image_size == 0 || image_size % stride != 0 {
        return Err(anyhow!(
            "qwen3vl preprocess: image_size ({}) must be a positive multiple \
             of patch_size ({}) * spatial_merge_size ({}) = {}",
            image_size,
            cfg.patch_size,
            cfg.spatial_merge_size,
            stride
        ));
    }
    if cfg.image_min_pixels > cfg.image_max_pixels {
        return Err(anyhow!(
            "qwen3vl preprocess: image_min_pixels ({}) > image_max_pixels ({})",
            cfg.image_min_pixels,
            cfg.image_max_pixels
        ));
    }

    let fmt = image::guess_format(bytes).map_err(|e| anyhow!("guess_format: {e}"))?;
    match fmt {
        ImageFormat::Png | ImageFormat::Jpeg => {}
        other => {
            return Err(anyhow!(
                "image format {:?} is not supported by qwen3vl preprocess (only PNG + JPEG)",
                other
            ));
        }
    }
    validate_decode_dimensions(bytes)?;
    let img = image::load_from_memory(bytes).map_err(|e| anyhow!("decode image: {e}"))?;
    let (orig_w, orig_h) = img.dimensions();
    if orig_w == 0 || orig_h == 0 {
        return Err(anyhow!("qwen3vl preprocess: image has zero dimension"));
    }
    let longer = orig_w.max(orig_h) as u64;
    let shorter = orig_w.min(orig_h) as u64;
    if longer > shorter.saturating_mul(200) {
        return Err(anyhow!(
            "qwen3vl preprocess: absolute aspect ratio must be at most 200, got {}x{}",
            orig_w,
            orig_h
        ));
    }

    // Variable-resolution smart-resize is governed by the source
    // processor's exact pixel-area bounds. `image_size` describes the
    // learned position-table grid; it is not a per-axis resize ceiling.
    // Wide and tall inputs can therefore exceed it along one axis while
    // remaining within the validated total pixel budget.
    let (smart_w, smart_h) = qwen_vision_calc_size_preserved_ratio(
        orig_w,
        orig_h,
        stride,
        cfg.image_min_pixels,
        cfg.image_max_pixels,
    )?;

    let (target_w, target_h) = (smart_w, smart_h);

    // Sanity (defense-in-depth): both axes are stride-aligned and
    // non-zero. Any failure indicates a smart-resize contract bug.
    if target_w == 0 || target_h == 0 {
        return Err(anyhow!(
            "qwen3vl preprocess: smart-resize produced degenerate target \
             ({target_w}x{target_h}) for input ({orig_w}x{orig_h})"
        ));
    }
    if target_w % stride != 0 || target_h % stride != 0 {
        return Err(anyhow!(
            "qwen3vl preprocess: smart-resized target ({target_w}x{target_h}) not \
             stride-aligned (stride={stride})"
        ));
    }

    // Step 3: bicubic resize. This is a distinct family contract; the
    // corner-aligned bilinear helper used by Gemma must not be reused here.
    let src_rgb = img.to_rgb8();
    let resized = super::resize_bicubic::resize_rgb8(&src_rgb, target_w, target_h);

    // Step 4: HWC → CHW + per-channel normalize. Output shape is
    // `[3, target_h, target_w]` (CHW row-major) — NO center-pad.
    let tw_us = target_w as usize;
    let th_us = target_h as usize;
    let hw = th_us * tw_us;
    let mut pixel_values = vec![0f32; 3 * hw];
    for (y, row) in resized.rows().enumerate() {
        for (x, pix) in row.enumerate() {
            let idx = y * tw_us + x;
            for c in 0..3 {
                let v = (pix[c] as f32 / 255.0 - cfg.image_mean[c]) / cfg.image_std[c];
                pixel_values[c * hw + idx] = v;
            }
        }
    }

    // Phase-2 per-image token grid: `n_x_token = target_w / stride`,
    // `n_y_token = target_h / stride`. Variable across images with
    // different aspect ratios.
    let n_x_token = target_w / stride;
    let n_y_token = target_h / stride;
    let n_image_tokens = n_x_token * n_y_token;

    Ok(QwenVisionPreprocessed {
        pixel_values,
        target_size: image_size,
        target_w,
        target_h,
        n_x_token,
        n_y_token,
        n_image_tokens,
    })
}

/// Smart-resize contract for the Qwen vision image processor.
/// Algorithm:
///   - `align_size` MUST be > 0.
///   - Always align up first: `h_bar = max(align, round_by(h))`,
///     `w_bar = max(align, round_by(w))` where
///     `round_by(x) = round(x/align)*align`.
///   - If `h_bar*w_bar > max_pixels`:
///     `beta = sqrt(h*w / max_pixels)`,
///     `h_bar = max(align, floor_by(h/beta))`,
///     `w_bar = max(align, floor_by(w/beta))`.
///   - Else if `h_bar*w_bar < min_pixels`:
///     `beta = sqrt(min_pixels / (h*w))`,
///     `h_bar = ceil_by(h*beta)`, `w_bar = ceil_by(w*beta)`.
///
/// Returns `(target_w, target_h)`.
fn qwen_vision_calc_size_preserved_ratio(
    orig_w: u32,
    orig_h: u32,
    align_size: u32,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<(u32, u32)> {
    let align: u64 = align_size as u64;
    if align == 0 {
        return Err(anyhow!("qwen3vl smart_resize: align_size must be > 0"));
    }
    let width = orig_w as u64;
    let height = orig_h as u64;
    if width == 0 || height == 0 {
        return Err(anyhow!(
            "qwen3vl smart_resize: input dimensions must be > 0"
        ));
    }
    if width.max(height) > width.min(height).saturating_mul(200) {
        return Err(anyhow!(
            "qwen3vl smart_resize: absolute aspect ratio must be at most 200, got {}x{}",
            orig_w,
            orig_h
        ));
    }

    let round_by = |x: f64| -> u64 { ((x / align as f64).round() as i64).max(0) as u64 * align };
    let ceil_by = |x: f64| -> u64 { ((x / align as f64).ceil() as i64).max(0) as u64 * align };
    let floor_by = |x: f64| -> u64 { ((x / align as f64).floor() as i64).max(0) as u64 * align };

    let mut h_bar: u64 = align.max(round_by(height as f64));
    let mut w_bar: u64 = align.max(round_by(width as f64));

    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = align.max(floor_by(height as f64 / beta));
        w_bar = align.max(floor_by(width as f64 / beta));
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = ceil_by(height as f64 * beta);
        w_bar = ceil_by(width as f64 * beta);
    }

    if h_bar == 0 || w_bar == 0 || h_bar > u32::MAX as u64 || w_bar > u32::MAX as u64 {
        return Err(anyhow!(
            "qwen3vl smart_resize: degenerate output ({} x {}) for input ({} x {})",
            w_bar,
            h_bar,
            orig_w,
            orig_h
        ));
    }
    Ok((w_bar as u32, h_bar as u32))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_image_extent_is_bounded_before_pixel_allocation() {
        validate_decoded_image_extent(4_096, 4_096).expect("official maximum");
        assert!(validate_decoded_image_extent(4_097, 4_096).is_err());
        assert!(validate_decoded_image_extent(0, 4_096).is_err());
        assert!(validate_decoded_image_extent(u32::MAX, u32::MAX).is_err());
    }
    use image::{ImageBuffer, Rgb, RgbImage};
    use std::io::Cursor;

    /// Encode an in-memory RGB image to PNG bytes for test fixtures.
    fn encode_png(img: &RgbImage) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
            .expect("encode png");
        buf
    }

    #[test]
    fn preprocess_solid_gray_image_produces_expected_shape() {
        // 4×4 solid gray. target=4 (no resize). mean=0.5, std=0.5 → pixel
        // 127/255=0.498 → (0.498-0.5)/0.5 ≈ -0.0039 for every channel.
        let img: RgbImage = ImageBuffer::from_fn(4, 4, |_x, _y| Rgb([127u8, 127, 127]));
        let png = encode_png(&img);
        let cfg = PreprocessConfig {
            target_size: 4,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        };
        let out = preprocess_rgb_chw(&png, &cfg).unwrap();
        assert_eq!(out.len(), 3 * 4 * 4);
        for v in &out {
            assert!(
                (*v + 0.01).abs() < 0.02,
                "expected ~-0.004 per pixel, got {}",
                v
            );
        }
    }

    #[test]
    fn preprocess_resizes_to_target_size() {
        // 8×8 solid blue → resized to 2×2.
        let img: RgbImage = ImageBuffer::from_fn(8, 8, |_x, _y| Rgb([0u8, 0, 255]));
        let png = encode_png(&img);
        let cfg = PreprocessConfig {
            target_size: 2,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        };
        let out = preprocess_rgb_chw(&png, &cfg).unwrap();
        assert_eq!(out.len(), 3 * 2 * 2);
        // Red channel = 0, green = 0, blue = 1.0 (normalized from 255/255).
        for i in 0..4 {
            assert!((out[i] - 0.0).abs() < 1e-5, "R[{}] = {}", i, out[i]);
            assert!((out[4 + i] - 0.0).abs() < 1e-5, "G[{}] = {}", i, out[4 + i]);
            assert!((out[8 + i] - 1.0).abs() < 1e-5, "B[{}] = {}", i, out[8 + i]);
        }
    }

    #[test]
    fn preprocess_normalizes_with_configured_mean_std() {
        // Single-pixel image. Red=200, green=100, blue=50.
        // With mean=[0.1, 0.2, 0.3] and std=[0.5, 0.5, 0.5]:
        //   R: (200/255 - 0.1)/0.5 ≈ (0.7843 - 0.1)/0.5 ≈ 1.3686
        //   G: (100/255 - 0.2)/0.5 ≈ (0.3922 - 0.2)/0.5 ≈ 0.3843
        //   B: (50/255 - 0.3)/0.5 ≈ (0.1961 - 0.3)/0.5 ≈ -0.2078
        let img: RgbImage = ImageBuffer::from_fn(1, 1, |_x, _y| Rgb([200u8, 100, 50]));
        let png = encode_png(&img);
        let cfg = PreprocessConfig {
            target_size: 1,
            mean: [0.1, 0.2, 0.3],
            std: [0.5, 0.5, 0.5],
        };
        let out = preprocess_rgb_chw(&png, &cfg).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.3686).abs() < 1e-3, "R={}", out[0]);
        assert!((out[1] - 0.3843).abs() < 1e-3, "G={}", out[1]);
        assert!((out[2] - (-0.2078)).abs() < 1e-3, "B={}", out[2]);
    }

    #[test]
    fn preprocess_layout_is_chw_not_hwc() {
        // 2×2 image: top-left = red (255,0,0), top-right = green (0,255,0),
        // bottom-left = blue (0,0,255), bottom-right = black (0,0,0).
        // CHW layout: out[0..4]=R, out[4..8]=G, out[8..12]=B.
        // In row-major HWC order, those pixels at positions 0,1,2,3 produce:
        //   R channel: [1, 0, 0, 0]  (only top-left has red)
        //   G channel: [0, 1, 0, 0]  (only top-right has green)
        //   B channel: [0, 0, 1, 0]  (only bottom-left has blue)
        let img: RgbImage = ImageBuffer::from_fn(2, 2, |x, y| match (x, y) {
            (0, 0) => Rgb([255, 0, 0]),
            (1, 0) => Rgb([0, 255, 0]),
            (0, 1) => Rgb([0, 0, 255]),
            _ => Rgb([0, 0, 0]),
        });
        let png = encode_png(&img);
        let cfg = PreprocessConfig {
            target_size: 2,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        };
        let out = preprocess_rgb_chw(&png, &cfg).unwrap();
        // R channel: positions [0..4] = row-major of [TL, TR, BL, BR].
        assert!((out[0] - 1.0).abs() < 1e-5, "R[TL] = {}", out[0]);
        assert!(out[1].abs() < 1e-5, "R[TR] = {}", out[1]);
        assert!(out[2].abs() < 1e-5, "R[BL] = {}", out[2]);
        assert!(out[3].abs() < 1e-5, "R[BR] = {}", out[3]);
        // G channel at offset 4.
        assert!(out[4].abs() < 1e-5);
        assert!((out[5] - 1.0).abs() < 1e-5, "G[TR] = {}", out[5]);
        assert!(out[6].abs() < 1e-5);
        // B channel at offset 8.
        assert!((out[10] - 1.0).abs() < 1e-5, "B[BL] = {}", out[10]);
    }

    #[test]
    fn preprocess_rejects_unsupported_format() {
        // BMP-ish bytes shouldn't pass guess_format. Actually BMP's signature
        // is detectable, so let's use something definitely unrecognized.
        let gibberish = vec![0xABu8; 64];
        let cfg = GEMMA4_VISION_CONFIG.clone();
        let err = preprocess_rgb_chw(&gibberish, &cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("guess_format") || msg.contains("not supported"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn preprocess_rejects_zero_target_size() {
        let img: RgbImage = ImageBuffer::from_pixel(1, 1, Rgb([0u8, 0, 0]));
        let png = encode_png(&img);
        let cfg = PreprocessConfig {
            target_size: 0,
            mean: [0.0; 3],
            std: [1.0; 3],
        };
        let err = preprocess_rgb_chw(&png, &cfg).unwrap_err();
        assert!(format!("{err}").contains("invalid target_size"));
    }

    #[test]
    fn preprocess_accepts_jpeg_input() {
        // Encode as JPEG — verifies the guess_format accepts both.
        let img: RgbImage = ImageBuffer::from_pixel(16, 16, Rgb([128u8, 128, 128]));
        let mut buf: Vec<u8> = Vec::new();
        img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Jpeg)
            .expect("encode jpeg");
        let cfg = PreprocessConfig {
            target_size: 4,
            mean: [0.0; 3],
            std: [1.0; 3],
        };
        let out = preprocess_rgb_chw(&buf, &cfg).unwrap();
        assert_eq!(out.len(), 3 * 4 * 4);
    }

    #[test]
    fn gemma4_vision_config_constants() {
        // Lock the day-one Gemma 4 vision tower preprocessing constants.
        // Changes here must pair with a validation against mlx-lm's
        // Gemma 4 vision output.
        assert_eq!(GEMMA4_VISION_CONFIG.target_size, 896);
        assert_eq!(GEMMA4_VISION_CONFIG.mean, [0.5, 0.5, 0.5]);
        assert_eq!(GEMMA4_VISION_CONFIG.std, [0.5, 0.5, 0.5]);
    }

    // -------------------------------------------------------------------
    // Gemma4V variable-resolution preprocessing tests
    // -------------------------------------------------------------------

    fn encode_solid_png(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
        let img: RgbImage = ImageBuffer::from_pixel(w, h, Rgb(rgb));
        let mut buf: Vec<u8> = Vec::new();
        img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
            .expect("encode png");
        buf
    }

    #[test]
    fn gemma4v_preprocess_default_constants_match_llama_cpp() {
        // Locks the peer's `(252, 280)` token-limit and
        // `n_merge=3`/`patch_size=16` reference values.
        assert_eq!(GEMMA4V_PREPROCESS_DEFAULT.patch_size, 16);
        assert_eq!(GEMMA4V_PREPROCESS_DEFAULT.n_merge, 3);
        assert_eq!(GEMMA4V_PREPROCESS_DEFAULT.token_min, 252);
        assert_eq!(GEMMA4V_PREPROCESS_DEFAULT.token_max, 280);
    }

    #[test]
    fn gemma4v_preprocess_token_budget_post_pool() {
        // Three sizes spanning the budget regimes (small, on-target,
        // large). The post-pool token count `(n_x/n_merge) * (n_y/n_merge)`
        // must land in `[token_min, token_max]` = `[252, 280]`. The
        // pre-pool patch grid axes must each be multiples of
        // `n_merge = 3` so the avg-pool kernel sees an exact integer
        // grid (matches the peer's dynamic-size preprocessor resize
        // with align_size = patch * n_merge).
        let n_merge = GEMMA4V_PREPROCESS_DEFAULT.n_merge;
        for (w, h) in [(64u32, 64), (256, 256), (1024, 1024)] {
            let png = encode_solid_png(w, h, [128, 128, 128]);
            let out = preprocess_gemma4v(&png, &GEMMA4V_PREPROCESS_DEFAULT)
                .unwrap_or_else(|e| panic!("({w},{h}): {e}"));
            assert_eq!(
                out.n_x % n_merge,
                0,
                "({w},{h}) n_x={} not mul of {n_merge}",
                out.n_x
            );
            assert_eq!(
                out.n_y % n_merge,
                0,
                "({w},{h}) n_y={} not mul of {n_merge}",
                out.n_y
            );
            let post_pool = (out.n_x / n_merge) * (out.n_y / n_merge);
            assert!(
                (252..=280).contains(&post_pool),
                "({w},{h}) → pre-pool ({},{}) → post-pool {post_pool} tokens, expected [252, 280]",
                out.n_x,
                out.n_y
            );
            let n = out.n_patches();
            assert_eq!(out.patches.len(), (n as usize) * 16 * 16 * 3);
            assert_eq!(out.pos_x.len(), n as usize);
            assert_eq!(out.pos_y.len(), n as usize);
        }
    }

    #[test]
    fn gemma4v_preprocess_pixel_scaling_4x_minus_3() {
        // ADR-005 Phase 2c iter-125 (W56): expected values updated from the
        // old single-step `2x − 1` algebra (which produced range [-1, +1])
        // to the peer's byte-faithful two-step chain folded as `4x − 3`
        // (range [-3, +1]). Solid black (0) → 4*0 - 3 = -3.0. Solid white
        // (255) → 4*1 - 3 = +1.0. Mid-gray (128) → 4*(128/255) - 3 ≈ -0.992.
        for (rgb, expect) in [([0u8, 0, 0], -3.0_f32), ([255, 255, 255], 1.0)] {
            let png = encode_solid_png(256, 256, rgb);
            let out = preprocess_gemma4v(&png, &GEMMA4V_PREPROCESS_DEFAULT).unwrap();
            // Spot-check a handful of positions across the patch tensor.
            for &i in &[0, out.patches.len() / 2, out.patches.len() - 1] {
                assert!(
                    (out.patches[i] - expect).abs() < 1e-3,
                    "rgb={:?} idx={} got={} expect={}",
                    rgb,
                    i,
                    out.patches[i],
                    expect
                );
            }
        }
        // Mid-gray center: 4 * (128/255) - 3 = 0.5098... - 2.0 - ... actually
        // 128/255 ≈ 0.50196, *4 = 2.00784, -3 = -0.99216.
        let png_mid = encode_solid_png(256, 256, [128, 128, 128]);
        let out_mid = preprocess_gemma4v(&png_mid, &GEMMA4V_PREPROCESS_DEFAULT).unwrap();
        let v = out_mid.patches[0];
        let expect_mid = (128.0_f32 / 255.0) * 4.0 - 3.0; // ≈ -0.99216
        assert!(
            (v - expect_mid).abs() < 1e-3,
            "mid-gray got {v}, expected ≈ {expect_mid}"
        );
    }

    #[test]
    fn gemma4v_preprocess_pixel_range_in_minus_three_plus_one() {
        // ADR-005 Phase 2c iter-125 (W56): expected range updated from
        // [-1, +1] (old one-step `2x − 1`) to [-3, +1] (the peer's
        // byte-faithful
        // two-step chain `4x − 3`). Random-style gradient image — every
        // pixel must end up in [-3, +1].
        let img: RgbImage = ImageBuffer::from_fn(128, 128, |x, y| {
            Rgb([
                (x as u8).wrapping_mul(2),
                (y as u8).wrapping_mul(2),
                ((x ^ y) as u8).wrapping_mul(2),
            ])
        });
        let mut buf: Vec<u8> = Vec::new();
        img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
            .expect("encode png");
        let out = preprocess_gemma4v(&buf, &GEMMA4V_PREPROCESS_DEFAULT).unwrap();
        let max_v = out.patches.iter().cloned().fold(f32::MIN, f32::max);
        let min_v = out.patches.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            min_v >= -3.0 - 1e-6 && max_v <= 1.0 + 1e-6,
            "range out of bounds: [{min_v}, {max_v}]"
        );
    }

    #[test]
    fn gemma4v_preprocess_pos_indices_are_dense_grid() {
        // For a square input, pos_x[idx] == idx % n_x and
        // pos_y[idx] == idx / n_x — i.e. row-major patch ordering.
        let png = encode_solid_png(256, 256, [10, 20, 30]);
        let out = preprocess_gemma4v(&png, &GEMMA4V_PREPROCESS_DEFAULT).unwrap();
        for idx in 0..out.n_patches() as usize {
            let exp_x = (idx as u32) % out.n_x;
            let exp_y = (idx as u32) / out.n_x;
            assert_eq!(out.pos_x[idx], exp_x, "pos_x[{idx}]");
            assert_eq!(out.pos_y[idx], exp_y, "pos_y[{idx}]");
        }
    }

    #[test]
    fn gemma4v_preprocess_rejects_unknown_format() {
        let gibberish = vec![0xABu8; 64];
        let err = preprocess_gemma4v(&gibberish, &GEMMA4V_PREPROCESS_DEFAULT).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("guess_format") || msg.contains("not supported"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn gemma4v_preprocess_rejects_zero_patch_size() {
        let png = encode_solid_png(64, 64, [0, 0, 0]);
        let cfg = Gemma4vPreprocessConfig {
            patch_size: 0,
            ..GEMMA4V_PREPROCESS_DEFAULT.clone()
        };
        let err = preprocess_gemma4v(&png, &cfg).unwrap_err();
        assert!(format!("{err}").contains("patch_size"));
    }

    // -------------------------------------------------------------------
    // ADR-005 Phase 2c iter-121 (W52) — byte-faithful peer resize
    // -------------------------------------------------------------------

    #[test]
    fn resize_bilinear_peer_corner_aligned_identity_2x2_to_3x3() {
        // 2×2 input with each pixel a unique value; resize to 3×3.
        // Corner-aligned bilinear with x_ratio = (2-1)/(3-1) = 0.5 means
        // output position (0,0) samples src(0,0), (2,2) samples src(1,1)
        // (corners are exact). Center (1,1) bilinear-blends all four src.
        // This locks "vertex-aligned" sampling; image::Triangle would
        // produce a different center pixel.
        let mut src: RgbImage = ImageBuffer::new(2, 2);
        src.put_pixel(0, 0, Rgb([0, 0, 0])); // top-left = 0
        src.put_pixel(1, 0, Rgb([100, 100, 100])); // top-right = 100
        src.put_pixel(0, 1, Rgb([200, 200, 200])); // bot-left = 200
        src.put_pixel(1, 1, Rgb([255, 255, 255])); // bot-right = 255

        let dst = resize_bilinear_peer(&src, 3, 3);
        // Corner (0,0) must be exactly src(0,0) = 0 (not blended).
        assert_eq!(dst.get_pixel(0, 0).0[0], 0, "corner (0,0)");
        // Corner (2,2) must be exactly src(1,1) = 255 (not blended).
        assert_eq!(dst.get_pixel(2, 2).0[0], 255, "corner (2,2)");
        // Corner (2,0) must be exactly src(1,0) = 100.
        assert_eq!(dst.get_pixel(2, 0).0[0], 100, "corner (2,0)");
        // Corner (0,2) must be exactly src(0,1) = 200.
        assert_eq!(dst.get_pixel(0, 2).0[0], 200, "corner (0,2)");
        // Center (1,1) at px=py=0.5 → x0=y0=0, xf=yf=0.5.
        // top    = lerp(0, 100, 0.5) = 50
        // bottom = lerp(200, 255, 0.5) = 227.5
        // out    = lerp(50, 227.5, 0.5) = 138.75 → trunc → 138
        assert_eq!(dst.get_pixel(1, 1).0[0], 138, "center (1,1)");
    }

    #[test]
    fn resize_bilinear_peer_truncates_not_rounds() {
        // 1×2 source [0, 1] → resize to 1×3. With x_ratio = (2-1)/(3-1) = 0.5,
        // middle output samples px=0.5 → top=0.5, bottom=0.5, out=0.5.
        // Truncation: 0.5 → 0 (NOT 1 like round-to-nearest).
        let mut src: RgbImage = ImageBuffer::new(1, 2);
        src.put_pixel(0, 0, Rgb([0, 0, 0]));
        src.put_pixel(0, 1, Rgb([1, 1, 1]));
        let dst = resize_bilinear_peer(&src, 1, 3);
        assert_eq!(dst.get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(dst.get_pixel(0, 1).0, [0, 0, 0], "trunc(0.5)=0");
        assert_eq!(dst.get_pixel(0, 2).0, [1, 1, 1]);
    }

    #[test]
    fn resize_bilinear_pad_peer_no_pad_for_square_input() {
        // Square input → square target: padding branch must reduce to
        // plain bilinear (new_w/new_h hit target exactly). Verifies
        // gemma4v's common case (square fixtures) works the same as
        // direct resize.
        let src: RgbImage = ImageBuffer::from_fn(4, 4, |x, _y| Rgb([(x * 50) as u8; 3]));
        let padded = resize_bilinear_pad_peer(&src, 8, 8, [0, 0, 0]);
        let plain = resize_bilinear_peer(&src, 8, 8);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(
                    padded.get_pixel(x, y).0,
                    plain.get_pixel(x, y).0,
                    "({x},{y})"
                );
            }
        }
    }

    #[test]
    fn resize_bilinear_pad_peer_pads_non_square_input() {
        // 4×2 source → 4×4 target. scale = min(4/4, 4/2) = 1.0.
        // new_w = ceil(4*1.0) = 4, new_h = ceil(2*1.0) = 2.
        // Padding adds 1 row of black above and 1 row below the resized
        // image (offset_y = (4-2)/2 = 1).
        let src: RgbImage = ImageBuffer::from_fn(4, 2, |_x, _y| Rgb([200, 100, 50]));
        let dst = resize_bilinear_pad_peer(&src, 4, 4, [0, 0, 0]);
        // Top row must be black pad.
        for x in 0..4 {
            assert_eq!(dst.get_pixel(x, 0).0, [0, 0, 0], "pad top ({x},0)");
        }
        // Middle two rows = resized source = original color (1:1 scale).
        for x in 0..4 {
            assert_eq!(dst.get_pixel(x, 1).0, [200, 100, 50]);
            assert_eq!(dst.get_pixel(x, 2).0, [200, 100, 50]);
        }
        // Bottom row = black pad.
        for x in 0..4 {
            assert_eq!(dst.get_pixel(x, 3).0, [0, 0, 0], "pad bot ({x},3)");
        }
    }

    #[test]
    fn gemma4v_preprocess_uses_llama_cpp_resize_for_four_corner_dots() {
        // 8×8 image with four corner pixels = white, rest = black.
        // After the peer's corner-aligned bilinear resize to a much
        // larger target (e.g. 768×768 from the gemma4v patch grid),
        // the resulting CORNER patches must contain non-zero pixel values
        // (white seeped into the corner via the lerp from the 1-pixel
        // dot). With a center-aligned + antialiased filter the same
        // 1-pixel dot can be smoothed away or shifted, so this test
        // both pins our new resize and is a regression guard against
        // accidentally re-introducing `FilterType::Triangle`.
        let img: RgbImage = ImageBuffer::from_fn(8, 8, |x, y| {
            if (x == 0 || x == 7) && (y == 0 || y == 7) {
                Rgb([255u8, 255, 255])
            } else {
                Rgb([0, 0, 0])
            }
        });
        let mut buf: Vec<u8> = Vec::new();
        img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
            .expect("encode png");
        let out = preprocess_gemma4v(&buf, &GEMMA4V_PREPROCESS_DEFAULT).unwrap();

        // First patch (top-left corner). At least one pixel in the patch
        // must be > 0 (specifically pixel (0,0) of the resized image is
        // exactly src(0,0) = white = +1.0 after the `4x − 3` byte-faithful
        // normalization: 4*1.0 - 3 = +1.0; iter-125 W56 confirmed white
        // is invariant under the algebra change).
        let inner = (16 * 16 * 3) as usize;
        let first_patch = &out.patches[0..inner];
        // ADR-005 Phase 2c iter-126 (W57): inner-axis ordering switched
        // from HWC `(dy, dx, c)` to CHW `(c, dy, dx)`. (c=0, dy=0, dx=0)
        // → index 0*256 + 0*16 + 0 = 0 (still index 0 for the (R, 0, 0)
        // pixel by coincidence — both orderings start at the R-channel
        // top-left pixel).
        assert!(
            (first_patch[0] - 1.0).abs() < 1e-3,
            "first patch (R, 0, 0) should be +1.0 (corner-aligned exact src), got {}",
            first_patch[0]
        );

        // Last patch's bottom-right pixel must also be exactly +1.0
        // (corner of resized image == corner of src). With CHW ordering,
        // (c=0, dy=15, dx=15) → 0*256 + 15*16 + 15 = 255.
        let n_patches = out.n_patches() as usize;
        let last_patch = &out.patches[(n_patches - 1) * inner..n_patches * inner];
        assert!(
            (last_patch[255] - 1.0).abs() < 1e-3,
            "last patch (R, 15, 15) should be +1.0, got {}",
            last_patch[255]
        );
    }

    #[test]
    fn gemma4v_preprocess_rejects_inverted_token_bounds() {
        let png = encode_solid_png(64, 64, [0, 0, 0]);
        let cfg = Gemma4vPreprocessConfig {
            patch_size: 16,
            n_merge: 3,
            token_min: 300,
            token_max: 100,
        };
        let err = preprocess_gemma4v(&png, &cfg).unwrap_err();
        assert!(format!("{err}").contains("token_min"));
    }

    // -----------------------------------------------------------------
    // Qwen3-VL preprocessor tests (ADR-005 iter-224 Wedge-4d)
    // -----------------------------------------------------------------

    /// Canonical Qwen3-VL preprocessor config for tests: patch_size=16,
    /// spatial_merge_size=2, OpenAI-CLIP normalization defaults, peer's
    /// `set_limit_image_tokens(8, 4096)` pixel bounds.
    fn qwen_vision_test_cfg() -> QwenVisionPreprocessConfig {
        QwenVisionPreprocessConfig {
            patch_size: 16,
            spatial_merge_size: 2,
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.26862954, 0.26130258, 0.27577711],
            image_min_pixels: 8 * 16 * 16 * 2 * 2,    // 8192
            image_max_pixels: 4096 * 16 * 16 * 2 * 2, // 4_194_304
        }
    }

    #[test]
    fn qwen_vision_preprocess_pixel_shape_matches_smart_resize_grid() {
        // Phase-2 (iter-225): output pixel grid is the smart-resized
        // rectangular shape, NOT the canvas. Square 256×256 input is
        // already stride-aligned (multiple of 32) and inside
        // [min_pixels=8192, max_pixels=4_194_304], so smart_resize
        // returns (256, 256) unchanged. The output is `[3, 256, 256]`,
        // NOT `[3, 768, 768]` (Phase-1 center-pad path is GONE).
        let png = encode_solid_png(256, 256, [127, 127, 127]);
        let cfg = qwen_vision_test_cfg();
        let image_size = 768; // canonical Qwen3-VL trained canvas
        let out = preprocess_qwen_vision(&png, &cfg, image_size).unwrap();
        assert_eq!(out.target_w, 256);
        assert_eq!(out.target_h, 256);
        assert_eq!(out.pixel_values.len(), 3 * 256 * 256);
        assert_eq!(out.target_size, image_size); // canvas reporting preserved
        assert_eq!(out.target_pixel_grid(), (256, 256));
        // n_x_token = n_y_token = 256 / (16 * 2) = 8
        assert_eq!(out.n_x_token, 8);
        assert_eq!(out.n_y_token, 8);
        assert_eq!(out.n_image_tokens, 8 * 8);
    }

    #[test]
    fn qwen_vision_preprocess_smart_resize_aligned_to_stride() {
        // Input shape that doesn't align — verify smart_resize produces a
        // (target_w, target_h) that is a multiple of stride = patch * sm.
        let stride: u32 = 16 * 2;
        let cfg = qwen_vision_test_cfg();

        // Test the calc helper directly across a range of input sizes.
        for &(orig_w, orig_h) in &[
            (100, 100),
            (200, 50),
            (50, 200),
            (1, 1),       // tiny — should upscale to min_pixels
            (8000, 4000), // huge — should downscale to max_pixels
            (1024, 768),  // landscape standard
        ] {
            let (tw, th) = qwen_vision_calc_size_preserved_ratio(
                orig_w,
                orig_h,
                stride,
                cfg.image_min_pixels,
                cfg.image_max_pixels,
            )
            .expect("smart_resize ok");
            assert!(
                tw % stride == 0 && th % stride == 0,
                "({orig_w}x{orig_h}) → ({tw}x{th}) not aligned to stride={stride}"
            );
            let area = tw as u64 * th as u64;
            assert!(
                area >= cfg.image_min_pixels && area <= cfg.image_max_pixels,
                "({tw}x{th}) area={area} not in [{}, {}]",
                cfg.image_min_pixels,
                cfg.image_max_pixels
            );
        }
    }

    #[test]
    fn qwen_vision_smart_resize_uses_area_bounds_not_image_size_axis_cap() {
        // The learned position-table grid is not a per-axis image ceiling.
        // Smart resize may exceed it along one axis as long as the total
        // area remains within the source processor's bounds.
        let cfg = qwen_vision_test_cfg();
        let image_size = 768u32;
        let stride: u32 = cfg.patch_size * cfg.spatial_merge_size;

        for &(orig_w, orig_h, label) in &[
            (1024u32, 576u32, "landscape"),
            (576, 1024, "portrait"),
            (3840, 2160, "large landscape"),
        ] {
            let (smart_w, smart_h) = qwen_vision_calc_size_preserved_ratio(
                orig_w,
                orig_h,
                stride,
                cfg.image_min_pixels,
                cfg.image_max_pixels,
            )
            .unwrap_or_else(|e| panic!("smart_resize ok for {label}: {e}"));
            let area = u64::from(smart_w) * u64::from(smart_h);
            assert!(
                smart_w > image_size || smart_h > image_size,
                "{label} should prove the axis-cap removal: got {smart_w}x{smart_h}"
            );
            assert!(
                smart_w % stride == 0 && smart_h % stride == 0,
                "{label} smart target ({smart_w}x{smart_h}) not stride={stride} aligned"
            );
            assert!(area >= cfg.image_min_pixels && area <= cfg.image_max_pixels);
        }
    }

    #[test]
    fn qwen_vision_preprocess_aspect_ratio_preserved_in_smart_resize() {
        // 200x50 (4:1) — smart_resize should keep that ratio approximately.
        let stride: u32 = 32;
        let cfg = qwen_vision_test_cfg();
        let (tw, th) = qwen_vision_calc_size_preserved_ratio(
            200,
            50,
            stride,
            cfg.image_min_pixels,
            cfg.image_max_pixels,
        )
        .unwrap();
        // Ratio drift should be small (within stride alignment tolerance).
        let orig_ratio = 200.0_f64 / 50.0;
        let new_ratio = tw as f64 / th as f64;
        assert!(
            (orig_ratio - new_ratio).abs() / orig_ratio < 0.5,
            "smart_resize aspect drift: orig={orig_ratio}, new={new_ratio} ({tw}x{th})"
        );
    }

    #[test]
    fn qwen_vision_preprocess_rejects_aspect_ratio_above_200() {
        let cfg = qwen_vision_test_cfg();
        let err = qwen_vision_calc_size_preserved_ratio(
            201,
            1,
            cfg.patch_size * cfg.spatial_merge_size,
            cfg.image_min_pixels,
            cfg.image_max_pixels,
        )
        .expect_err("201:1 must fail before resize");
        assert!(format!("{err}").contains("aspect ratio"));

        let png = encode_solid_png(201, 1, [127, 127, 127]);
        let err = preprocess_qwen_vision(&png, &cfg, 768)
            .expect_err("the public preprocessor must reject 201:1");
        assert!(format!("{err}").contains("aspect ratio"));
    }

    #[test]
    fn qwen_vision_bicubic_resize_golden() {
        let mut src = RgbImage::new(3, 2);
        let values = [
            [0, 0, 0],
            [255, 0, 0],
            [255, 255, 255],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
        ];
        for (pixel, value) in src.pixels_mut().zip(values) {
            *pixel = Rgb(value);
        }
        let resized = super::super::resize_bicubic::resize_rgb8(&src, 4, 4);
        let actual: Vec<[u8; 3]> = resized.pixels().map(|pixel| pixel.0).collect();
        let expected = vec![
            [0, 0, 0],
            [174, 0, 0],
            [255, 97, 88],
            [255, 255, 255],
            [0, 53, 0],
            [127, 16, 37],
            [222, 91, 112],
            [255, 255, 202],
            [0, 202, 0],
            [33, 61, 141],
            [128, 81, 161],
            [255, 255, 53],
            [0, 255, 0],
            [0, 84, 194],
            [81, 75, 185],
            [255, 255, 0],
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn qwen_vision_preprocess_rejects_misaligned_image_size() {
        let png = encode_solid_png(100, 100, [0, 0, 0]);
        let cfg = qwen_vision_test_cfg();
        // 100 is not a multiple of stride=32.
        let err = preprocess_qwen_vision(&png, &cfg, 100).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("must be a positive multiple"), "got: {msg}");
    }

    #[test]
    fn qwen_vision_preprocess_normalization_mean_std_applied() {
        // Phase-2 (iter-225): no center-pad. Constant 128/255 image,
        // mean=0.5, std=0.5: per-channel value before norm =
        // 128/255 ≈ 0.502, after = (0.502-0.5)/0.5 ≈ 0.004. Every
        // pixel in the smart-resized output should land near +0.004
        // (no pad region exists post-Phase-2).
        let png = encode_solid_png(64, 64, [128, 128, 128]);
        let cfg = QwenVisionPreprocessConfig {
            patch_size: 16,
            spatial_merge_size: 2,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_min_pixels: 64 * 64,       // 4096 (≤ canvas area)
            image_max_pixels: 768u64.pow(2), // 589824
        };
        let out = preprocess_qwen_vision(&png, &cfg, 768).unwrap();
        // Smart-resize for 64×64 input with min_pixels=4096 stays at
        // (64, 64) (already aligned + at min_pixels exactly). Output
        // shape is [3, 64, 64].
        assert_eq!(out.target_w, 64);
        assert_eq!(out.target_h, 64);
        assert_eq!(out.pixel_values.len(), 3 * 64 * 64);
        // Spot-check: center pixel and corner pixel should both be the
        // normalized constant ≈ 0.004 (NO pad region).
        let center_idx = 64 * 32 + 32; // (y=32, x=32) in c=0 plane
        let v = out.pixel_values[center_idx];
        assert!(
            v.abs() < 0.05,
            "center pixel of normalized 128-gray image should be ~0, got {v}"
        );
        // Phase-2: corner is REAL content (not pad). 128-gray normalizes
        // to ~0.004, NOT -1.0 (which would be the Phase-1 pad signature).
        let corner_v = out.pixel_values[0];
        assert!(
            corner_v.abs() < 0.05,
            "Phase-2 corner is REAL content (no pad), expected ~0.004 from \
             128-gray input, got {corner_v}"
        );
    }

    #[test]
    fn qwen_vision_preprocess_from_legacy_mmproj_uses_token_derived_pixel_bounds() {
        let mmcfg = super::super::mmproj::MmprojConfig {
            image_size: 768,
            patch_size: 16,
            num_patches_side: 48,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_attention_heads: 16,
            num_hidden_layers: 24,
            layer_norm_eps: 1e-6,
            projector: super::super::mmproj::ProjectorType::QwenVisionMerger,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_min_pixels: None,
            image_max_pixels: None,
            spatial_merge_size: Some(2),
            projection_dim: Some(2048),
            deepstack_indexes: Some(vec![5, 11, 17]),
        };
        let cfg = QwenVisionPreprocessConfig::from_mmproj(&mmcfg).unwrap();
        // Defaults: image_min_pixels = 8 * 16² * 2² = 8192,
        //           image_max_pixels = 4096 * 16² * 2² = 4_194_304.
        assert_eq!(cfg.image_min_pixels, 8192);
        assert_eq!(cfg.image_max_pixels, 4_194_304);
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.spatial_merge_size, 2);
    }

    #[test]
    fn qwen_vision_preprocess_from_mmproj_uses_embedded_processor_pixel_bounds() {
        let mmcfg = super::super::mmproj::MmprojConfig {
            image_size: 768,
            patch_size: 16,
            num_patches_side: 48,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            layer_norm_eps: 1e-6,
            projector: super::super::mmproj::ProjectorType::QwenVisionMerger,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_min_pixels: Some(65_536),
            image_max_pixels: Some(16_777_216),
            spatial_merge_size: Some(2),
            projection_dim: Some(5120),
            deepstack_indexes: Some(vec![]),
        };
        let cfg = QwenVisionPreprocessConfig::from_mmproj(&mmcfg).unwrap();
        assert_eq!(cfg.image_min_pixels, 65_536);
        assert_eq!(cfg.image_max_pixels, 16_777_216);
    }

    #[test]
    fn qwen_vision_preprocess_from_mmproj_rejects_missing_spatial_merge() {
        let mmcfg = super::super::mmproj::MmprojConfig {
            image_size: 768,
            patch_size: 16,
            num_patches_side: 48,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_attention_heads: 16,
            num_hidden_layers: 24,
            layer_norm_eps: 1e-6,
            projector: super::super::mmproj::ProjectorType::QwenVisionMerger,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_min_pixels: None,
            image_max_pixels: None,
            spatial_merge_size: None, // missing
            projection_dim: Some(2048),
            deepstack_indexes: Some(vec![5, 11, 17]),
        };
        let err = QwenVisionPreprocessConfig::from_mmproj(&mmcfg).unwrap_err();
        assert!(format!("{err}").contains("spatial_merge_size"));
    }

    #[test]
    fn qwen_vision_preprocess_rejects_non_image_bytes() {
        let cfg = qwen_vision_test_cfg();
        let err = preprocess_qwen_vision(&[1, 2, 3, 4, 5], &cfg, 768).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("guess_format") || msg.contains("not supported"),
            "got: {msg}"
        );
    }

    // -----------------------------------------------------------------
    // ADR-005 iter-225 Wedge-4 Phase-2 — variable-resolution preprocess
    // -----------------------------------------------------------------

    /// Phase-2 test #1: square 768×768 input → square output identical
    /// to Phase-1 contract (n_image_tokens=576). Backward-compat pin.
    #[test]
    fn qwen_vision_preprocess_phase2_square_768_matches_phase1_grid() {
        let png = encode_solid_png(768, 768, [80, 90, 100]);
        let cfg = qwen_vision_test_cfg();
        let out = preprocess_qwen_vision(&png, &cfg, 768).unwrap();
        // Smart-resize on a 768×768 input that is already exactly the
        // canonical canvas: target stays at (768, 768).
        assert_eq!(out.target_w, 768);
        assert_eq!(out.target_h, 768);
        assert_eq!(out.pixel_values.len(), 3 * 768 * 768);
        // n_x_token = n_y_token = 768 / (16*2) = 24 → 576 tokens.
        assert_eq!(out.n_x_token, 24);
        assert_eq!(out.n_y_token, 24);
        assert_eq!(out.n_image_tokens, 576);
        assert_eq!(out.target_pixel_grid(), (768, 768));
    }

    /// Landscape input remains at its stride-aligned smart-resize shape;
    /// the position-grid size does not force an axis clamp.
    #[test]
    fn qwen_vision_preprocess_phase2_landscape_1024x576_aspect_preserved() {
        // Use mean=std=0.5 in this test so the no-pad-region invariant
        // is unambiguously distinguishable from real solid-color
        // content (default OpenAI-CLIP mean/std would normalize a
        // pure black pixel to ~-1.5 vs Phase-1 pad ~-1.0; here both
        // map to -1.0 only for true black pixels, which we don't
        // have in this fixture).
        let cfg = QwenVisionPreprocessConfig {
            patch_size: 16,
            spatial_merge_size: 2,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_min_pixels: 8 * 16 * 16 * 2 * 2,    // 8192
            image_max_pixels: 4096 * 16 * 16 * 2 * 2, // 4_194_304
        };
        let png = encode_solid_png(1024, 576, [128, 128, 128]); // mid-gray
        let out = preprocess_qwen_vision(&png, &cfg, 768).unwrap();
        assert_eq!(out.target_w, 1024);
        assert_eq!(out.target_h, 576);
        assert_eq!(out.pixel_values.len(), 3 * 1024 * 576);
        assert_eq!(out.n_x_token, 32);
        assert_eq!(out.n_y_token, 18);
        assert_eq!(out.n_image_tokens, 32 * 18);
        // Mid-gray 128/255 ≈ 0.502 → with mean=std=0.5, normalized ≈ 0.004.
        // A Phase-1 pad region would normalize to (0/255 - 0.5)/0.5 = -1.0.
        // Pin: NO pixel is ≤ -0.5 (no pad leak).
        let mut min_v = f32::INFINITY;
        for &v in out.pixel_values.iter() {
            if v < min_v {
                min_v = v;
            }
        }
        assert!(
            min_v > -0.5,
            "Phase-2 should have NO pad region; mid-gray input must \
             normalize to ~0.004 everywhere (NO -1.0 pad signature). \
             observed min={min_v}"
        );
    }

    /// Portrait input mirrors the landscape grid without an axis clamp.
    #[test]
    fn qwen_vision_preprocess_phase2_portrait_576x1024_aspect_preserved() {
        let png = encode_solid_png(576, 1024, [200, 50, 80]);
        let cfg = qwen_vision_test_cfg();
        let out = preprocess_qwen_vision(&png, &cfg, 768).unwrap();
        assert_eq!(out.target_w, 576);
        assert_eq!(out.target_h, 1024);
        assert_eq!(out.pixel_values.len(), 3 * 1024 * 576);
        assert_eq!(out.n_x_token, 18);
        assert_eq!(out.n_y_token, 32);
        assert_eq!(out.n_image_tokens, 18 * 32);
        // Aspect ratio preserved (within stride alignment).
        assert!(out.n_y_token > out.n_x_token, "portrait → n_y > n_x");
    }
}

#[cfg(test)]
mod guide_fixture_tests {
    /// The getting-started guide's embedded red-PNG fixture must decode
    /// through the strict PNG path hf2q serves with. The original fixture
    /// was a truncated "smallest-PNG" whose declared IDAT length exceeded
    /// the file: tolerant decoders (CoreGraphics) rendered it, the strict
    /// `png`-crate path returned UnexpectedEof, and the guide's vision
    /// check failed with HTTP 400 on a healthy server (2026-08-21).
    const GUIDE_RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAb0lEQVR4nO3PAQkAAAyEwO9feoshgnABdLep8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3IPanc8OLDQitxAAAAAElFTkSuQmCC";

    #[test]
    fn guide_red_png_decodes_to_64x64() {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(GUIDE_RED_PNG_B64)
            .expect("base64 decode");
        // The fixture must be a complete, checksum-valid PNG — not merely
        // decodable by lenient renderers. A well-formed PNG ends with the
        // 12-byte IEND trailer; the original truncated fixture lacked it.
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "PNG magic");
        assert_eq!(
            &bytes[bytes.len() - 12..],
            &[0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xAE, 0x42, 0x60, 0x82],
            "PNG IEND trailer"
        );
        let img = image::load_from_memory(&bytes).expect("image decode");
        assert_eq!(
            image::GenericImageView::dimensions(&img),
            (64, 64),
            "guide fixture dimensions"
        );
        // Solid red as far as the model is concerned.
        let rgb = img.to_rgb8();
        let px = rgb.get_pixel(32, 32);
        assert_eq!(px.0, [255, 0, 0], "center pixel is red");
    }
}
