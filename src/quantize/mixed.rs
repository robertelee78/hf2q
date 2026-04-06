//! Mixed-bit quantizer with --sensitive-layers support (Epic 5).
//!
//! Not yet implemented.

use thiserror::Error;

/// Errors from mixed-bit quantization.
#[derive(Error, Debug)]
pub enum MixedQuantError {
    #[error("Mixed-bit quantization is not yet implemented (Epic 5)")]
    NotImplemented,

    #[error("Invalid sensitive layer range: {0}")]
    InvalidLayerRange(String),
}
