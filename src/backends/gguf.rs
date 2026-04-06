//! GGUF output backend (future).
//!
//! Not yet implemented.

use thiserror::Error;

/// Errors from GGUF backend operations.
#[derive(Error, Debug)]
pub enum GgufError {
    #[error("GGUF output backend is not yet implemented")]
    NotImplemented,
}
