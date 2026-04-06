//! Perplexity delta measurement (Epic 4).
//!
//! Not yet implemented.

use thiserror::Error;

/// Errors from perplexity computation.
#[derive(Error, Debug)]
pub enum PerplexityError {
    #[error("Perplexity measurement is not yet implemented (Epic 4)")]
    NotImplemented,
}
