//! Rule-based heuristic quant selection (Epic 6).
//!
//! Not yet implemented.

use thiserror::Error;

/// Errors from heuristic resolution.
#[derive(Error, Debug)]
pub enum HeuristicsError {
    #[error("Heuristic auto mode is not yet implemented (Epic 6)")]
    NotImplemented,
}
