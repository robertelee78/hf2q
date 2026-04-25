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
//!   (Gemma 4 MoE vision tower default).
//! - Other per-model configs land alongside their forward-pass code
//!   (e.g. `NOMIC_VISION_CONFIG` when a Nomic vision encoder ports).

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
}
