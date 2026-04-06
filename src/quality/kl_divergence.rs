//! KL divergence measurement (Epic 4).
//!
//! Not yet implemented.

use thiserror::Error;

/// Errors from KL divergence computation.
#[derive(Error, Debug)]
pub enum KlDivergenceError {
    #[error("KL divergence measurement is not yet implemented (Epic 4)")]
    NotImplemented,
}
