//! MLX inference runner via mlx-rs (Epic 4).
//!
//! Not yet implemented.

use thiserror::Error;

/// Errors specific to the MLX runner.
#[derive(Error, Debug)]
pub enum MlxRunnerError {
    #[error("MLX inference runner is not yet implemented (Epic 4)")]
    NotImplemented,

    #[error("Metal backend not available")]
    NoMetal,
}
