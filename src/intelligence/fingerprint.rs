//! Model fingerprinting from config.json (Epic 6).
//!
//! Not yet implemented.

use thiserror::Error;

/// Errors from model fingerprinting.
#[derive(Error, Debug)]
pub enum FingerprintError {
    #[error("Model fingerprinting is not yet implemented (Epic 6)")]
    NotImplemented,
}
