//! Cosine similarity of layer activations (Epic 4).
//!
//! Not yet implemented.

use thiserror::Error;

/// Errors from cosine similarity computation.
#[derive(Error, Debug)]
pub enum CosineSimilarityError {
    #[error("Cosine similarity measurement is not yet implemented (Epic 4)")]
    NotImplemented,
}
