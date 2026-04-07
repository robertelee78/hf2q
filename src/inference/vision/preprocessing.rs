//! CPU-side image preprocessing for the Gemma 4 vision encoder.
//!
//! Pipeline:
//! 1. Decode image bytes (JPEG, PNG, WebP, GIF first frame) via the `image` crate
//! 2. Aspect-ratio-preserving resize to fit within the patch budget
//! 3. Rescale pixel values to \[0, 1\] (the model's patch embedder does `2*(x-0.5)`)
//! 4. Patchify into (num_patches, patch_size * patch_size * 3) f32 tensor
//! 5. Compute 2D position IDs for each patch
//!
//! All work is done on CPU. The output `PreprocessedImage` is a flat f32 buffer
//! ready to be transferred to a Metal buffer for the vision encoder.

use std::path::Path;

use thiserror::Error;

use super::config::VisionConfig;

/// Errors from image preprocessing.
#[derive(Error, Debug)]
pub enum ImageError {
    #[error("Invalid image: {reason}")]
    Decode { reason: String },

    #[error("Unsupported image format: {format}")]
    UnsupportedFormat { format: String },

    #[error("Image too large: {width}x{height} exceeds maximum allowed dimensions")]
    TooLarge { width: u32, height: u32 },

    #[error("Image file not found: {path}")]
    NotFound { path: String },

    #[error("Path traversal rejected: {path}")]
    PathTraversal { path: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result of preprocessing a single image, ready for the vision encoder.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    /// Flattened patches: shape \[num_patches_padded, patch_pixels\] stored row-major.
    /// `patch_pixels` = patch_size * patch_size * 3.
    pub pixel_values: Vec<f32>,
    /// 2D position IDs for each patch: shape \[num_patches_padded, 2\].
    /// Padding patches have position (-1, -1).
    pub position_ids: Vec<[i32; 2]>,
    /// Actual number of non-padding patches (before padding to max_patches).
    pub num_real_patches: usize,
    /// Number of soft tokens this image will produce after pooling.
    pub num_soft_tokens: usize,
    /// Total padded patch count.
    pub max_patches: usize,
}

/// Decode and preprocess a single image from raw bytes.
///
/// Returns a `PreprocessedImage` with pixel values rescaled to \[0, 1\],
/// patchified, and padded to the maximum patch count.
pub fn preprocess_image(
    image_bytes: &[u8],
    config: &VisionConfig,
) -> Result<PreprocessedImage, ImageError> {
    if image_bytes.is_empty() {
        return Err(ImageError::Decode {
            reason: "Image data is empty (0 bytes)".to_string(),
        });
    }

    // Step 0: Decode the image
    let img = image::load_from_memory(image_bytes).map_err(|e| ImageError::Decode {
        reason: format!("Failed to decode image: {e}"),
    })?;

    let rgb_img = img.to_rgb8();
    let (orig_w, orig_h) = (rgb_img.width(), orig_height(&rgb_img));

    if orig_w == 0 || orig_h == 0 {
        return Err(ImageError::Decode {
            reason: format!("Image has zero dimensions: {orig_w}x{orig_h}"),
        });
    }

    // Step 1: Compute target dimensions (aspect-ratio-preserving resize)
    let patch_size = config.patch_size;
    let pooling_k = config.pooling_kernel_size;
    let max_soft_tokens = config.default_output_length;
    let max_patches = max_soft_tokens * pooling_k * pooling_k;

    let (target_h, target_w) =
        aspect_ratio_preserving_size(orig_h as usize, orig_w as usize, patch_size, max_patches, pooling_k)?;

    // Step 2: Resize the image
    let resized = image::imageops::resize(
        &rgb_img,
        target_w as u32,
        target_h as u32,
        image::imageops::FilterType::Lanczos3,
    );

    // Step 3: Convert to f32 and rescale to [0, 1]
    // Layout: (C, H, W) interleaved in the raw buffer as (H, W, C)
    let (rh, rw) = (resized.height() as usize, resized.width() as usize);
    let raw_pixels = resized.into_raw();

    // Convert to CHW f32 in [0, 1]
    let mut chw = vec![0.0f32; 3 * rh * rw];
    for y in 0..rh {
        for x in 0..rw {
            let idx = (y * rw + x) * 3;
            for c in 0..3 {
                chw[c * rh * rw + y * rw + x] = raw_pixels[idx + c] as f32 / 255.0;
            }
        }
    }

    // Step 4: Patchify
    let patch_h = rh / patch_size;
    let patch_w = rw / patch_size;
    let num_real_patches = patch_h * patch_w;
    let patch_pixels = patch_size * patch_size * 3;

    let mut patches = Vec::with_capacity(num_real_patches * patch_pixels);
    for py in 0..patch_h {
        for px in 0..patch_w {
            // Extract one patch: iterate over (patch_y, patch_x) within the patch,
            // for each channel
            for row in 0..patch_size {
                for col in 0..patch_size {
                    let img_y = py * patch_size + row;
                    let img_x = px * patch_size + col;
                    for c in 0..3 {
                        patches.push(chw[c * rh * rw + img_y * rw + img_x]);
                    }
                }
            }
        }
    }

    // Step 5: Compute 2D position IDs (x, y) for each patch
    let mut position_ids = Vec::with_capacity(num_real_patches);
    for py in 0..patch_h {
        for px in 0..patch_w {
            position_ids.push([px as i32, py as i32]);
        }
    }

    // Step 6: Compute soft token count after pooling
    let pooled_h = patch_h / pooling_k;
    let pooled_w = patch_w / pooling_k;
    let num_soft_tokens = pooled_h * pooled_w;

    // Step 7: Pad patches and positions to max_patches
    let padding_needed = max_patches.saturating_sub(num_real_patches);
    patches.extend(std::iter::repeat(0.0f32).take(padding_needed * patch_pixels));
    position_ids.extend(std::iter::repeat([-1i32, -1i32]).take(padding_needed));

    Ok(PreprocessedImage {
        pixel_values: patches,
        position_ids,
        num_real_patches,
        num_soft_tokens,
        max_patches,
    })
}

/// Extract image bytes from a base64 data URL.
///
/// Supports `data:image/{format};base64,{data}` format.
/// Also handles raw base64 without the data URL prefix.
pub fn decode_base64_image(url: &str) -> Result<Vec<u8>, ImageError> {
    use base64::Engine as _;

    let base64_data = if let Some(comma_pos) = url.find(',') {
        // data:image/xxx;base64,<data>
        let prefix = &url[..comma_pos];
        if !prefix.contains("base64") {
            return Err(ImageError::Decode {
                reason: "Data URL is not base64-encoded".to_string(),
            });
        }
        &url[comma_pos + 1..]
    } else if url.starts_with("data:") {
        return Err(ImageError::Decode {
            reason: "Malformed data URL: missing comma separator".to_string(),
        });
    } else {
        // Assume raw base64
        url
    };

    base64::engine::general_purpose::STANDARD
        .decode(base64_data)
        .map_err(|e| ImageError::Decode {
            reason: format!("Base64 decode failed: {e}"),
        })
}

/// Load image bytes from a local file path.
///
/// Validates against path traversal attacks by rejecting paths containing
/// `..` components or symlinks that escape the filesystem root.
pub fn load_image_from_path(path_str: &str) -> Result<Vec<u8>, ImageError> {
    // Strip file:// prefix if present
    let clean_path = if let Some(stripped) = path_str.strip_prefix("file://") {
        stripped
    } else {
        path_str
    };

    // Reject path traversal
    let path = Path::new(clean_path);
    for component in path.components() {
        if let std::path::Component::ParentDir = component {
            return Err(ImageError::PathTraversal {
                path: path_str.to_string(),
            });
        }
    }

    // Canonicalize to resolve symlinks
    let canonical = path.canonicalize().map_err(|_| ImageError::NotFound {
        path: path_str.to_string(),
    })?;

    // Read the file
    std::fs::read(&canonical).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            ImageError::NotFound {
                path: path_str.to_string(),
            }
        } else {
            ImageError::Io(e)
        }
    })
}

/// Compute the target (height, width) that preserves aspect ratio and fits
/// within the patch budget.
///
/// Dimensions are rounded down to the nearest multiple of
/// `pooling_kernel_size * patch_size` to ensure clean pooling.
fn aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize), ImageError> {
    let total_px = height * width;
    let target_px = max_patches * (patch_size * patch_size);
    let factor = (target_px as f64 / total_px as f64).sqrt();

    let ideal_height = factor * height as f64;
    let ideal_width = factor * width as f64;
    let side_mult = (pooling_kernel_size * patch_size) as f64;

    // Round down to nearest multiple of side_mult
    let mut target_h = (ideal_height / side_mult).floor() as usize * pooling_kernel_size * patch_size;
    let mut target_w = (ideal_width / side_mult).floor() as usize * pooling_kernel_size * patch_size;

    let side_mult_usize = pooling_kernel_size * patch_size;

    // Handle edge cases where one or both dimensions round to 0
    if target_h == 0 && target_w == 0 {
        return Err(ImageError::Decode {
            reason: format!(
                "Image too small to resize: {}x{} cannot produce patches with \
                 side multiple {}",
                width, height, side_mult_usize
            ),
        });
    }

    if target_h == 0 {
        target_h = side_mult_usize;
        let max_side = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult_usize;
        target_w = ((width as f64 / height as f64).floor() as usize * side_mult_usize).min(max_side);
        if target_w == 0 {
            target_w = side_mult_usize;
        }
    } else if target_w == 0 {
        target_w = side_mult_usize;
        let max_side = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult_usize;
        target_h = ((height as f64 / width as f64).floor() as usize * side_mult_usize).min(max_side);
        if target_h == 0 {
            target_h = side_mult_usize;
        }
    }

    Ok((target_h, target_w))
}

/// Helper to get image height (the image crate names are width/height).
fn orig_height(img: &image::RgbImage) -> u32 {
    img.height()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aspect_ratio_preserving_size_square() {
        // 1000x1000 image, patch_size=16, max_patches=2520 (280*9), pool_k=3
        let (h, w) = aspect_ratio_preserving_size(1000, 1000, 16, 2520, 3).unwrap();
        // Both dimensions should be multiples of 48 (3*16)
        assert_eq!(h % 48, 0);
        assert_eq!(w % 48, 0);
        // Should fit within max_patches
        let patches = (h / 16) * (w / 16);
        assert!(patches <= 2520);
    }

    #[test]
    fn test_aspect_ratio_preserving_size_wide() {
        // 480x1920 image
        let (h, w) = aspect_ratio_preserving_size(480, 1920, 16, 2520, 3).unwrap();
        assert_eq!(h % 48, 0);
        assert_eq!(w % 48, 0);
        let patches = (h / 16) * (w / 16);
        assert!(patches <= 2520);
    }

    #[test]
    fn test_decode_base64_image_with_prefix() {
        // A 1x1 red PNG encoded in base64
        let png_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";
        let bytes = decode_base64_image(png_b64).unwrap();
        assert!(!bytes.is_empty());
        // Should start with PNG magic bytes
        assert_eq!(&bytes[..4], &[0x89, b'P', b'N', b'G']);
    }

    #[test]
    fn test_decode_base64_image_raw() {
        // Raw base64 of a 1x1 red PNG
        let raw_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";
        let bytes = decode_base64_image(raw_b64).unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_decode_base64_image_invalid() {
        let result = decode_base64_image("data:image/png;base64,!!!invalid!!!");
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_base64_malformed_data_url() {
        let result = decode_base64_image("data:image/png;nocomma");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_image_path_traversal() {
        let result = load_image_from_path("/etc/../etc/passwd");
        assert!(matches!(result, Err(ImageError::PathTraversal { .. })));
    }

    #[test]
    fn test_load_image_not_found() {
        let result = load_image_from_path("/nonexistent/path/image.png");
        assert!(matches!(result, Err(ImageError::NotFound { .. })));
    }

    #[test]
    fn test_load_image_file_prefix() {
        let result = load_image_from_path("file:///nonexistent/path/image.png");
        assert!(matches!(result, Err(ImageError::NotFound { .. })));
    }

    #[test]
    fn test_preprocess_empty_image() {
        let config = test_vision_config();
        let result = preprocess_image(&[], &config);
        assert!(matches!(result, Err(ImageError::Decode { .. })));
    }

    #[test]
    fn test_preprocess_corrupt_image() {
        let config = test_vision_config();
        let result = preprocess_image(&[0xFF, 0x00, 0x42, 0x13], &config);
        assert!(matches!(result, Err(ImageError::Decode { .. })));
    }

    #[test]
    fn test_preprocess_valid_png() {
        let config = test_vision_config();
        // Create a small 96x96 test image
        let img = image::RgbImage::from_fn(96, 96, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut buf = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut buf);
        image::ImageEncoder::write_image(
            encoder,
            img.as_raw(),
            96,
            96,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();

        let result = preprocess_image(&buf, &config).unwrap();

        // The 96x96 image gets resized by aspect_ratio_preserving_size:
        // max_patches = 280 * 9 = 2520, target_px = 2520 * 256 = 645120
        // factor = sqrt(645120 / 9216) ~ 8.37, ideal = 803.3
        // side_mult = 48, target = floor(803.3/48)*48 = 768
        // patches per side: 768/16 = 48, total: 48*48 = 2304
        assert_eq!(result.num_real_patches, 2304);
        // Pooled: 48/3=16 per side, 16*16=256 soft tokens
        assert_eq!(result.num_soft_tokens, 256);
        // Padded to max_patches = 280 * 9 = 2520
        assert_eq!(result.max_patches, 2520);
        assert_eq!(result.pixel_values.len(), 2520 * 16 * 16 * 3);
        assert_eq!(result.position_ids.len(), 2520);
        // Check first position is (0, 0)
        assert_eq!(result.position_ids[0], [0, 0]);
        // Check padding positions start after real patches
        assert_eq!(result.position_ids[2304], [-1, -1]);
        // Check pixel values are in [0, 1]
        for &v in result.pixel_values.iter().take(result.num_real_patches * 16 * 16 * 3) {
            assert!(v >= 0.0 && v <= 1.0, "Pixel value out of range: {v}");
        }
    }

    fn test_vision_config() -> VisionConfig {
        VisionConfig {
            hidden_size: 1152,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            intermediate_size: 4304,
            patch_size: 16,
            head_dim: 72,
            rms_norm_eps: 1e-6,
            rope_theta: 100.0,
            pooling_kernel_size: 3,
            position_embedding_size: 10240,
            default_output_length: 280,
            standardize: true,
            text_hidden_size: 2816,
            image_token_id: 258880,
            boi_token_id: 255999,
            eoi_token_id: 258882,
            vision_soft_tokens_per_image: 280,
        }
    }
}
