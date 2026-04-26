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
//! variable-resolution patchifier llama.cpp uses for `PROJECTOR_TYPE_GEMMA4V`
//! (`tools/mtmd/clip.cpp:1334-1343` + `tools/mtmd/models/gemma4v.cpp:4-15`):
//!
//!   1. Patchify image at native resolution into `patch_size × patch_size`
//!      tiles. Compute `(n_x, n_y)` such that `n_x * n_y` lies in
//!      `[token_min, token_max]` (typically `[252, 280]`); downscale
//!      preserving aspect ratio when the native grid exceeds the cap.
//!   2. Each pixel is mapped `[0, 1] → [-1, 1]` via `2x − 1`. This is
//!      mathematically equivalent to mean=std=[0.5, 0.5, 0.5] (the
//!      `(x − 0.5)/0.5 = 2x − 1` identity), so the SigLIP-49 fixed-res
//!      path's `GEMMA4_VISION_CONFIG` produces byte-identical normalized
//!      values for the same pixel — the gemma4v path just stores them
//!      flattened per-patch instead of CHW.
//!   3. Returns a `[N_patches, patch_size² × 3]` flat patch buffer plus
//!      per-patch `pos_x` and `pos_y` index arrays so the dual
//!      position-embed lookup (`tools/mtmd/models/gemma4v.cpp:18-42`)
//!      and the per-axis 2D RoPE step (lines 46-91) can reuse them
//!      downstream without re-deriving the (px, py) → (pos_x, pos_y)
//!      mapping.
//!
//! The existing `preprocess_rgb_chw` is unchanged.

use anyhow::{anyhow, Result};
use image::{imageops::FilterType, GenericImageView, ImageFormat};

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

    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow!("decode image: {e}"))?;
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

/// Knobs for the gemma4v patchifier. Defaults to llama.cpp's
/// `PROJECTOR_TYPE_GEMMA4V` settings:
///   - `patch_size = 16`
///   - `n_merge = 3` (pool kernel size; pre-pool patch grid axes must
///     be multiples of `n_merge` so the n_merge×n_merge avg-pool
///     produces an exact integer post-pool grid)
///   - `token_min = 252`, `token_max = 280` — **post-pool** token
///     bounds per `tools/mtmd/clip.cpp:1341`
///     `set_limit_image_tokens(252, 280)`. The `set_limit_image_tokens`
///     helper at `tools/mtmd/clip-model.h:112-118` converts these to
///     pixel bounds:
///       `image_min_pixels = 252 * patch_size² * n_merge² = 580608`
///       `image_max_pixels = 280 * patch_size² * n_merge² = 645120`.
///     These pixel bounds are what `calc_size_preserved_ratio` actually
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
    /// are upscaled to meet this floor — llama.cpp does the same
    /// (the `@ngxson` rationale: small inputs degrade quality without it).
    pub token_min: u32,
    /// Upper bound on the **post-pool** token count (inclusive). Large
    /// images are downscaled (preserving aspect ratio) until the
    /// post-pool grid fits.
    pub token_max: u32,
}

/// Default gemma4v config — locked to llama.cpp's reference values so
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
    /// Flat `[N_patches, patch_size² × 3]` tensor in row-major order.
    /// Within each row, the `patch_size² × 3` inner dim iterates as
    /// `(dy, dx, c)` — pixel-major within a patch, channel-minor —
    /// matching the candle reference's
    /// `permute(0,2,4,3,5,1).reshape(b, ph*pw, ps*ps*c)`
    /// (`/opt/candle/.../gemma4/vision.rs:146-150`).
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
/// Implements the `clip_graph_gemma4v::build` input contract from
/// `/opt/llama.cpp/tools/mtmd/models/gemma4v.cpp`:
///
///   - Decode image bytes (PNG/JPEG only, same restriction as
///     `preprocess_rgb_chw`).
///   - Compute the largest aspect-ratio-preserving size `(W, H)` such
///     that `(W/patch) * (H/patch) ≤ token_max`. If the resulting
///     patch count would fall below `token_min`, scale up to the
///     smallest size that meets the floor. `W` and `H` are rounded
///     down to multiples of `patch_size` so patchification is exact.
///   - Resize via bilinear (matches llama.cpp's
///     `RESIZE_ALGO_BILINEAR` for `PROJECTOR_TYPE_GEMMA4V`).
///   - Patchify into `[N_patches, patch_size² × 3]` with pixel layout
///     `(dy, dx, c)` per patch (matches candle's reshape, see
///     `Gemma4vPreprocessed::patches` doc).
///   - Pixel-scale `2x − 1` so the per-patch values are in `[-1, 1]`.
///     llama.cpp does this as the first GPU op (`gemma4v.cpp:9`); we
///     fold it into the CPU pre-step because the math is identical
///     to mean=std=[0.5,0.5,0.5] and the existing SigLIP-49 path
///     already runs the equivalent in preprocess.
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
    let img = image::load_from_memory(bytes).map_err(|e| anyhow!("decode image: {e}"))?;
    let (orig_w, orig_h) = img.dimensions();
    if orig_w == 0 || orig_h == 0 {
        return Err(anyhow!("gemma4v: image has zero dimension"));
    }

    let p = cfg.patch_size;
    let (n_x, n_y) = compute_gemma4v_patch_grid(
        orig_w,
        orig_h,
        p,
        cfg.n_merge,
        cfg.token_min,
        cfg.token_max,
    )?;
    let target_w = n_x * p;
    let target_h = n_y * p;

    let resized = img.resize_exact(target_w, target_h, FilterType::Triangle);
    let rgb = resized.to_rgb8();

    let n_patches = (n_x as usize) * (n_y as usize);
    let p_us = p as usize;
    let inner = p_us * p_us * 3;
    let mut patches = vec![0f32; n_patches * inner];
    let mut pos_x = Vec::with_capacity(n_patches);
    let mut pos_y = Vec::with_capacity(n_patches);

    // Patchify in (py, px, dy, dx, c) order — matches candle's
    // `permute(0,2,4,3,5,1).reshape(b, ph*pw, ps*ps*c)`.
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
                    let inner_base = ((dy as usize) * p_us + (dx as usize)) * 3;
                    // 2x − 1 maps [0, 1] → [-1, 1].
                    patches[row_base + inner_base + 0] =
                        (pix[0] as f32 / 255.0) * 2.0 - 1.0;
                    patches[row_base + inner_base + 1] =
                        (pix[1] as f32 / 255.0) * 2.0 - 1.0;
                    patches[row_base + inner_base + 2] =
                        (pix[2] as f32 / 255.0) * 2.0 - 1.0;
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
/// This is a byte-faithful port of llama.cpp's
/// `img_tool::calc_size_preserved_ratio(inp, align_size,
/// min_pixels, max_pixels)`
/// (`/opt/llama.cpp/tools/mtmd/mtmd-image.cpp:144-168`), called by
/// `mtmd_image_preprocessor_dyn_size::preprocess` at
/// `/opt/llama.cpp/tools/mtmd/mtmd-image.cpp:864-873` for
/// `PROJECTOR_TYPE_GEMMA4V`. The pixel bounds come from
/// `set_limit_image_tokens` at `clip-model.h:112-118` which converts
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
    // Pixel-area bounds, mirroring `set_limit_image_tokens`
    // (clip-model.h:112-118): patch_area = p² * n_merge².
    let patch_area: u64 = (p as u64) * (p as u64) * (n_merge as u64) * (n_merge as u64);
    let min_pixels: u64 = (token_min as u64) * patch_area;
    let max_pixels: u64 = (token_max as u64) * patch_area;

    // Helpers: round / ceil / floor `x` to a multiple of `align_size`.
    let round_by = |x: f64| -> u64 {
        ((x / align_size as f64).round() as i64).max(0) as u64 * align_size
    };
    let ceil_by = |x: f64| -> u64 {
        ((x / align_size as f64).ceil() as i64).max(0) as u64 * align_size
    };
    let floor_by = |x: f64| -> u64 {
        ((x / align_size as f64).floor() as i64).max(0) as u64 * align_size
    };

    let width = orig_w as u64;
    let height = orig_h as u64;

    // "Always align up first" — clip-model.h:153-155.
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
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
            assert!(
                (out[8 + i] - 1.0).abs() < 1e-5,
                "B[{}] = {}",
                i,
                out[8 + i]
            );
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
        // Locks the llama.cpp `set_limit_image_tokens(252, 280)` and
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
        // grid (matches llama.cpp's
        // `mtmd_image_preprocessor_dyn_size::preprocess` resize via
        // `calc_size_preserved_ratio` with align_size = patch * n_merge).
        let n_merge = GEMMA4V_PREPROCESS_DEFAULT.n_merge;
        for (w, h) in [(64u32, 64), (256, 256), (1024, 1024)] {
            let png = encode_solid_png(w, h, [128, 128, 128]);
            let out = preprocess_gemma4v(&png, &GEMMA4V_PREPROCESS_DEFAULT)
                .unwrap_or_else(|e| panic!("({w},{h}): {e}"));
            assert_eq!(out.n_x % n_merge, 0, "({w},{h}) n_x={} not mul of {n_merge}", out.n_x);
            assert_eq!(out.n_y % n_merge, 0, "({w},{h}) n_y={} not mul of {n_merge}", out.n_y);
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
    fn gemma4v_preprocess_pixel_scaling_2x_minus_1() {
        // Solid black should land at -1.0 (0 → -1), solid white at +1.0.
        // Solid mid-gray (128) should land at ~0.0 (128/255 ≈ 0.502 → 0.004).
        for (rgb, expect) in [
            ([0u8, 0, 0], -1.0_f32),
            ([255, 255, 255], 1.0),
        ] {
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
        // Mid-gray center
        let png_mid = encode_solid_png(256, 256, [128, 128, 128]);
        let out_mid = preprocess_gemma4v(&png_mid, &GEMMA4V_PREPROCESS_DEFAULT).unwrap();
        let v = out_mid.patches[0];
        // 128/255*2 - 1 ≈ 0.0039
        assert!(v.abs() < 0.01, "mid-gray got {v}, expected ~0.004");
    }

    #[test]
    fn gemma4v_preprocess_pixel_range_in_minus_one_one() {
        // Random-style image (gradient) — every pixel must end up in
        // [-1, 1].
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
            min_v >= -1.0 - 1e-6 && max_v <= 1.0 + 1e-6,
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
}
