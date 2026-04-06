//! CoreML output backend (Epic 8).
//!
//! Produces .mlpackage directories for Apple Neural Engine acceleration.
//! Not yet implemented.

use thiserror::Error;

/// Errors from CoreML backend operations.
#[derive(Error, Debug)]
pub enum CoremlError {
    #[error("CoreML output backend is not yet implemented (Epic 8)")]
    NotImplemented,

    #[error("Model architecture is not compatible with CoreML: {reason}")]
    IncompatibleArchitecture { reason: String },

    #[error("CoreML compilation failed: {reason}")]
    CompilationFailed { reason: String },
}
