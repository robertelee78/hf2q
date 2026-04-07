//! Vision encoder module for Gemma 4 multimodal inference.
//!
//! Implements the full SigLIP-based vision pipeline:
//! 1. CPU image preprocessing (decode, resize, patchify)
//! 2. GPU vision encoder (27 transformer layers)
//! 3. Spatial pooling and standardization
//! 4. Projection from vision hidden space to text hidden space
//! 5. Token injection into the text model sequence
//!
//! The vision encoder weights are loaded from the model's safetensors files
//! with the `model.vision_tower.*` and `model.embed_vision.*` prefixes.

pub mod preprocessing;
pub mod encoder;
pub mod config;

// Public re-exports for convenience. Currently used by the serve handlers
// via full paths, but these are the intended public API surface.
#[allow(unused_imports)]
pub use config::VisionConfig;
#[allow(unused_imports)]
pub use encoder::VisionEncoder;
#[allow(unused_imports)]
pub use preprocessing::{preprocess_image, ImageError, PreprocessedImage};
