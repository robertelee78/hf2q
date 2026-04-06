//! HuggingFace Hub download integration (Epic 3).
//!
//! Downloads model files from HuggingFace Hub via the `hf-hub` crate,
//! with fallback to the `hf` CLI.

use std::path::PathBuf;

use thiserror::Error;

use crate::progress::ProgressReporter;

/// Errors from HF download operations.
#[derive(Error, Debug)]
pub enum DownloadError {
    #[error("HuggingFace Hub download is not yet implemented (Epic 3). Use --input with a local model directory.")]
    NotImplemented,

    #[error("Failed to download from HuggingFace Hub: {reason}")]
    DownloadFailed { reason: String },

    #[error("Authentication failed: {reason}. Check HF_TOKEN env var or ~/.huggingface/token")]
    AuthError { reason: String },

    #[error("Repository not found: {repo}")]
    RepoNotFound { repo: String },

    #[error("hf CLI not found on PATH — install with: pip install huggingface_hub[cli]")]
    HfCliNotFound,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Download a model from HuggingFace Hub to a local cache directory.
///
/// Returns the path to the local directory containing the downloaded model files.
///
/// Not yet fully implemented (Epic 3) — currently returns an error.
pub fn download_model(
    _repo_id: &str,
    _progress: &ProgressReporter,
) -> Result<PathBuf, DownloadError> {
    Err(DownloadError::NotImplemented)
}
