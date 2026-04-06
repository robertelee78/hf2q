//! RuVector self-learning integration (Epic 7).
//!
//! Stores and retrieves conversion results for auto mode optimization.
//! Not yet implemented.

use thiserror::Error;

/// Errors from RuVector operations.
#[derive(Error, Debug)]
pub enum RuVectorError {
    #[error("RuVector integration is not yet implemented (Epic 7)")]
    NotImplemented,

    #[error("RuVector database not accessible at {path}: {reason}")]
    DatabaseUnavailable { path: String, reason: String },

    #[error("RuVector query failed: {reason}")]
    QueryFailed { reason: String },

    #[error("RuVector store failed: {reason}")]
    StoreFailed { reason: String },
}
