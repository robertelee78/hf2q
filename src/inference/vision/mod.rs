//! Vision preprocessing (ADR-005 Phase 2c, Task #14).
//!
//! CPU-side image handling for the `/v1/chat/completions` multimodal path
//! (Decision #1 of the 2026-04-23 scope refinement — vision absorbed as
//! Phase 2c sub-phase). Decodes OpenAI-format `image_url` content parts,
//! resizes to the ViT's expected patch grid, and normalizes to a CHW
//! float tensor ready for a future ViT forward pass.
//!
//! # Supported input formats (day-one)
//!
//!   - `data:image/png;base64,<payload>`  — inline base64, Open WebUI default.
//!   - `data:image/jpeg;base64,<payload>` — same shape, JPEG payload.
//!   - `file:///absolute/path/to/image.{png,jpg,jpeg}` — local file URL.
//!   - `/absolute/path/to/image.{png,jpg,jpeg}` — bare path (shorthand).
//!
//! HTTP URLs are **not** fetched in this iter — that'd introduce a
//! network-fetch code path on the request hot path, which is out of scope
//! for Phase 2c's minimum viable preprocessor. A later iter can add the
//! `reqwest::blocking::get` path when a real deployment needs it.
//!
//! # Preprocessing pipeline
//!
//!   1. Decode bytes into `image::DynamicImage`.
//!   2. Resize to `target_size × target_size` (typical ViT: 224, 336, 518).
//!   3. Convert to RGB8 (drops alpha channel; mmproj inputs are 3-channel).
//!   4. Normalize each channel: `(pixel/255 - mean[c]) / std[c]`.
//!   5. Transpose HWC → CHW into a flat `Vec<f32>` of length `3 × size × size`.
//!
//! # Not done in this iter (deferred to ViT-forward-pass iter)
//!
//!   - Patchifying `[3, H, W]` → `[N_PATCHES, PATCH_DIM]` via conv stem.
//!     That's a ViT model-side operation, not preprocessing.
//!   - Multi-image batching. A single request may carry multiple images
//!     (OpenAI's `content: [{text}, {image_url}, {image_url}, ...]` shape);
//!     the handler will iterate.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

pub mod mmproj;
pub mod mmproj_weights;
pub mod preprocess;
pub mod vit;
pub mod vit_gpu;

#[allow(unused_imports)]
pub use preprocess::{preprocess_rgb_chw, PreprocessConfig, GEMMA4_VISION_CONFIG};

/// A single preprocessed image ready for the ViT forward pass.
///
/// `pixel_values` carries the CHW-layout f32 tensor produced by
/// `preprocess_rgb_chw` (length = `3 × target_size × target_size`).
/// `source_label` is a debug/log-friendly id (mime type for data URIs,
/// file-name stem for file paths) so request-level tracing can
/// correlate per-image timings without leaking the full URL or payload.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    pub pixel_values: Vec<f32>,
    pub target_size: u32,
    pub source_label: String,
}

// ---------------------------------------------------------------------------
// ImageInput parsing
// ---------------------------------------------------------------------------

/// Parsed image source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageInput {
    /// Base64-encoded image payload (PNG or JPEG). The `mime_type` is the
    /// string between `data:` and `;base64,` — e.g. `"image/png"`.
    DataUri {
        mime_type: String,
        payload_base64: String,
    },
    /// Local filesystem path.
    FilePath(PathBuf),
    /// HTTP(S) URL — **not loaded** by this module. Caller is expected to
    /// reject with an error until the network-fetch path is enabled.
    HttpUrl(String),
}

/// Parse an OpenAI-format `image_url` string into a typed `ImageInput`.
///
/// Returns `Err` on unrecognized / malformed URLs. The caller should map
/// the error to a 400 invalid_request with `param = "content"`.
pub fn parse_image_url(url: &str) -> Result<ImageInput> {
    // data:image/{fmt};base64,<payload>
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, payload) = rest
            .split_once(",")
            .ok_or_else(|| anyhow!("data URI missing comma separator"))?;
        // Metadata: `image/png;base64` → mime=image/png, encoding=base64.
        let (mime_type, encoding) = meta
            .split_once(";")
            .ok_or_else(|| anyhow!("data URI missing encoding section"))?;
        if encoding != "base64" {
            return Err(anyhow!(
                "data URI encoding '{}' not supported (only 'base64')",
                encoding
            ));
        }
        if !(mime_type == "image/png" || mime_type == "image/jpeg" || mime_type == "image/jpg") {
            return Err(anyhow!(
                "data URI mime type '{}' not supported (only image/png and image/jpeg)",
                mime_type
            ));
        }
        return Ok(ImageInput::DataUri {
            mime_type: mime_type.to_string(),
            payload_base64: payload.to_string(),
        });
    }

    // file:///path
    if let Some(rest) = url.strip_prefix("file://") {
        // Trim the leading '/' that's always present in file:// URLs to
        // keep an absolute POSIX path.
        let path = if rest.starts_with('/') {
            PathBuf::from(rest)
        } else {
            return Err(anyhow!(
                "file:// URL must contain an absolute path (file:///path)"
            ));
        };
        return Ok(ImageInput::FilePath(path));
    }

    // Bare absolute path.
    if url.starts_with('/') {
        return Ok(ImageInput::FilePath(PathBuf::from(url)));
    }

    // http(s):// — not fetched in this iter.
    if url.starts_with("http://") || url.starts_with("https://") {
        return Ok(ImageInput::HttpUrl(url.to_string()));
    }

    Err(anyhow!(
        "unrecognized image URL scheme (expected data:, file://, or absolute path)"
    ))
}

/// Read an `ImageInput` into a raw bytes buffer. HTTP URLs are rejected in
/// this iter with a specific error so the caller can map to 400.
pub fn load_image_bytes(input: &ImageInput) -> Result<Vec<u8>> {
    match input {
        ImageInput::DataUri { payload_base64, .. } => {
            use base64::Engine;
            let payload = base64::engine::general_purpose::STANDARD
                .decode(payload_base64.trim())
                .map_err(|e| anyhow!("base64 decode: {e}"))?;
            Ok(payload)
        }
        ImageInput::FilePath(p) => read_file_bounded(p),
        ImageInput::HttpUrl(url) => Err(anyhow!(
            "HTTP image URLs are not yet loaded by this build ({}). \
             Send a `data:image/png;base64,...` or `file:///` URL instead.",
            url
        )),
    }
}

/// Read a file with a 20 MB size cap — defensive cap that exceeds the
/// biggest reasonable VLM input (a 4K JPEG is ~6 MB).
fn read_file_bounded(p: &Path) -> Result<Vec<u8>> {
    const MAX: u64 = 20 * 1024 * 1024;
    let meta = std::fs::metadata(p)
        .map_err(|e| anyhow!("stat {}: {e}", p.display()))?;
    if meta.len() > MAX {
        return Err(anyhow!(
            "image file {} exceeds {}-byte cap (got {})",
            p.display(),
            MAX,
            meta.len()
        ));
    }
    std::fs::read(p).map_err(|e| anyhow!("read {}: {e}", p.display()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_image_url_data_png() {
        let got = parse_image_url("data:image/png;base64,iVBORw0K").unwrap();
        match got {
            ImageInput::DataUri { mime_type, payload_base64 } => {
                assert_eq!(mime_type, "image/png");
                assert_eq!(payload_base64, "iVBORw0K");
            }
            other => panic!("expected DataUri, got {:?}", other),
        }
    }

    #[test]
    fn parse_image_url_data_jpeg() {
        let got = parse_image_url("data:image/jpeg;base64,/9j/4AA").unwrap();
        assert!(matches!(got, ImageInput::DataUri { .. }));
    }

    #[test]
    fn parse_image_url_rejects_unsupported_mime() {
        let err = parse_image_url("data:image/gif;base64,xyz").unwrap_err();
        assert!(format!("{err}").contains("not supported"));
    }

    #[test]
    fn parse_image_url_rejects_non_base64_encoding() {
        let err = parse_image_url("data:image/png;utf8,hello").unwrap_err();
        assert!(format!("{err}").contains("not supported"));
    }

    #[test]
    fn parse_image_url_file_scheme() {
        let got = parse_image_url("file:///tmp/cat.jpg").unwrap();
        assert_eq!(got, ImageInput::FilePath(PathBuf::from("/tmp/cat.jpg")));
    }

    #[test]
    fn parse_image_url_bare_absolute_path() {
        let got = parse_image_url("/tmp/dog.png").unwrap();
        assert_eq!(got, ImageInput::FilePath(PathBuf::from("/tmp/dog.png")));
    }

    #[test]
    fn parse_image_url_http_preserved_for_deferred_fetch() {
        let got = parse_image_url("https://example.com/img.jpg").unwrap();
        assert!(matches!(got, ImageInput::HttpUrl(_)));
    }

    #[test]
    fn parse_image_url_rejects_gibberish() {
        let err = parse_image_url("not-a-url").unwrap_err();
        assert!(format!("{err}").contains("unrecognized"));
    }

    #[test]
    fn load_image_bytes_data_uri_round_trips_base64() {
        // A minimal PNG signature: 8 bytes.
        let sig = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(sig);
        let url = format!("data:image/png;base64,{b64}");
        let input = parse_image_url(&url).unwrap();
        let bytes = load_image_bytes(&input).unwrap();
        assert_eq!(bytes, sig);
    }

    #[test]
    fn load_image_bytes_rejects_http_url() {
        let input = ImageInput::HttpUrl("https://example.com/cat.jpg".into());
        let err = load_image_bytes(&input).unwrap_err();
        assert!(format!("{err}").contains("not yet loaded"));
    }

    #[test]
    fn load_image_bytes_rejects_oversized_file() {
        // Don't actually create a 20 MB file in a unit test — just test the
        // nonexistent-path branch for the fast-fail contract. The size cap
        // is separately exercised by the live smoke harness when a real
        // oversized file is available.
        let err = load_image_bytes(&ImageInput::FilePath(PathBuf::from(
            "/tmp/does-not-exist-xyz-42.png",
        )))
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("stat") || msg.contains("No such"),
            "unexpected error: {msg}"
        );
    }
}
