//! NVFP4 output backend (future).
//!
//! Not yet implemented.

use thiserror::Error;

/// Errors from NVFP4 backend operations.
#[derive(Error, Debug)]
pub enum Nvfp4Error {
    #[error("NVFP4 output backend is not yet implemented")]
    NotImplemented,
}
