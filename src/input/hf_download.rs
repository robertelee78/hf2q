//! HuggingFace Hub download integration (Epic 3).
//!
//! Downloads model files from HuggingFace Hub via the `hf-hub` crate,
//! with fallback to the `hf` CLI if the crate fails.
//!
//! Token resolution order (Story 3.2):
//! 1. HF_TOKEN environment variable
//! 2. ~/.cache/huggingface/token file (hf-hub default)
//! 3. ~/.huggingface/token file (legacy path)
//!
//! Cache: Uses standard hf-hub cache directory. Subsequent runs with
//! the same repo skip re-download.

use std::path::PathBuf;

use thiserror::Error;
use tracing::{debug, info, warn};

use crate::progress::ProgressReporter;

/// Errors from HF download operations.
#[derive(Error, Debug)]
pub enum DownloadError {
    #[error(
        "Failed to download from HuggingFace Hub: {reason}\n\
         \n\
         Troubleshooting:\n\
         - Check your network connection\n\
         - For gated models, ensure you have accepted the license at huggingface.co\n\
         - Set HF_TOKEN env var or run: huggingface-cli login\n\
         - Install hf CLI as fallback: pip install huggingface_hub[cli]"
    )]
    DownloadFailed { reason: String },

    #[error(
        "Authentication failed for repository '{repo}'.\n\
         \n\
         This model may be gated or private. To access it:\n\
         1. Accept the model license at https://huggingface.co/{repo}\n\
         2. Set your token: export HF_TOKEN=hf_xxxx\n\
            Or create ~/.huggingface/token with your token\n\
         3. Alternatively, run: huggingface-cli login"
    )]
    AuthError { repo: String },

    #[error(
        "Repository not found: {repo}\n\
         \n\
         Check that the repository ID is correct (format: org/model-name).\n\
         Example: google/gemma-3-27b"
    )]
    RepoNotFound { repo: String },

    #[error(
        "No model files found in repository '{repo}'.\n\
         The repository exists but contains no safetensors files."
    )]
    NoModelFiles { repo: String },

    #[error(
        "hf CLI fallback also failed: {reason}\n\
         \n\
         Install the HuggingFace CLI: pip install huggingface_hub[cli]\n\
         Then try again."
    )]
    CliFallbackFailed { reason: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Files we need to download from a model repo.
const REQUIRED_FILES: &[&str] = &["config.json"];

/// Files we want to download if present (non-fatal if missing).
const OPTIONAL_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "generation_config.json",
];

/// Download a model from HuggingFace Hub.
///
/// Returns the path to the local directory containing the downloaded model files.
/// Uses hf-hub crate cache — subsequent calls with the same repo skip re-download.
pub fn download_model(
    repo_id: &str,
    progress: &ProgressReporter,
) -> Result<PathBuf, DownloadError> {
    info!(repo = %repo_id, "Downloading model from HuggingFace Hub");

    // Attempt 1: Use hf-hub crate (primary method)
    match download_via_hf_hub(repo_id, progress) {
        Ok(path) => Ok(path),
        Err(e) => {
            warn!("hf-hub crate download failed: {}. Trying hf CLI fallback...", e);

            // Attempt 2: Fall back to hf CLI on PATH (Story 3.2)
            match download_via_hf_cli(repo_id, progress) {
                Ok(path) => Ok(path),
                Err(cli_err) => {
                    // Both methods failed — report both errors
                    Err(DownloadError::DownloadFailed {
                        reason: format!(
                            "hf-hub crate: {}. hf CLI: {}",
                            e, cli_err
                        ),
                    })
                }
            }
        }
    }
}

/// Download model files using the hf-hub crate.
fn download_via_hf_hub(
    repo_id: &str,
    progress: &ProgressReporter,
) -> Result<PathBuf, DownloadError> {
    use hf_hub::api::sync::ApiBuilder;

    // Resolve auth token (Story 3.2)
    let token = resolve_auth_token();
    debug!(has_token = token.is_some(), "Auth token resolution");

    // Build the API client
    let mut builder = ApiBuilder::new()
        .with_progress(true);

    if let Some(t) = token {
        builder = builder.with_token(Some(t));
    }

    let api = builder.build().map_err(|e| {
        DownloadError::DownloadFailed {
            reason: format!("Failed to initialize HuggingFace API client: {}", e),
        }
    })?;

    let repo = api.model(repo_id.to_string());

    // Get repo info to discover files
    let repo_info = repo.info().map_err(|e| {
        let err_str = format!("{}", e);
        if err_str.contains("401") || err_str.contains("403") || err_str.contains("auth") {
            DownloadError::AuthError {
                repo: repo_id.to_string(),
            }
        } else if err_str.contains("404") || err_str.contains("not found") {
            DownloadError::RepoNotFound {
                repo: repo_id.to_string(),
            }
        } else {
            DownloadError::DownloadFailed {
                reason: format!("Failed to get repository info: {}", e),
            }
        }
    })?;

    // Identify files to download
    let all_files: Vec<String> = repo_info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .collect();

    debug!(file_count = all_files.len(), "Repository file listing retrieved");

    // Find safetensors files
    let safetensors_files: Vec<&String> = all_files
        .iter()
        .filter(|f| f.ends_with(".safetensors"))
        .collect();

    let index_file: Option<&String> = all_files
        .iter()
        .find(|f| f.as_str() == "model.safetensors.index.json");

    if safetensors_files.is_empty() {
        return Err(DownloadError::NoModelFiles {
            repo: repo_id.to_string(),
        });
    }

    // Build the list of files to download
    let mut files_to_download: Vec<&str> = Vec::new();

    // Required files
    for required in REQUIRED_FILES {
        if all_files.iter().any(|f| f.as_str() == *required) {
            files_to_download.push(required);
        } else {
            return Err(DownloadError::DownloadFailed {
                reason: format!("Required file '{}' not found in repository", required),
            });
        }
    }

    // Optional files (silently skip if missing)
    for optional in OPTIONAL_FILES {
        if all_files.iter().any(|f| f.as_str() == *optional) {
            files_to_download.push(optional);
        }
    }

    // Index file
    if let Some(idx) = index_file {
        files_to_download.push(idx.as_str());
    }

    // Safetensors shards
    for sf in &safetensors_files {
        files_to_download.push(sf.as_str());
    }

    // Download all files with progress
    let total_files = files_to_download.len();
    let pb = progress.bar(total_files as u64, "Downloading model files");

    let mut downloaded_path: Option<PathBuf> = None;

    for filename in &files_to_download {
        debug!(file = %filename, "Downloading");

        let local_path = repo.get(filename).map_err(|e| {
            let err_str = format!("{}", e);
            if err_str.contains("401") || err_str.contains("403") {
                DownloadError::AuthError {
                    repo: repo_id.to_string(),
                }
            } else {
                DownloadError::DownloadFailed {
                    reason: format!("Failed to download '{}': {}", filename, e),
                }
            }
        })?;

        // The first downloaded file tells us where the cache directory is
        if downloaded_path.is_none() {
            if let Some(parent) = local_path.parent() {
                downloaded_path = Some(parent.to_path_buf());
            }
        }

        pb.inc(1);
    }

    pb.finish_with_message(format!("Downloaded {} files", total_files));

    let model_dir = downloaded_path.ok_or_else(|| DownloadError::DownloadFailed {
        reason: "No files were downloaded".to_string(),
    })?;

    info!(path = %model_dir.display(), "Model downloaded to cache");

    Ok(model_dir)
}

/// Attempt to download using the `hf` CLI as a fallback.
fn download_via_hf_cli(
    repo_id: &str,
    _progress: &ProgressReporter,
) -> Result<PathBuf, DownloadError> {
    // Check if hf CLI is available — try both 'hf' and 'huggingface-cli'
    let hf_check = std::process::Command::new("hf")
        .arg("--version")
        .output();

    match hf_check {
        Ok(output) if output.status.success() => {
            debug!(
                "hf CLI found: {}",
                String::from_utf8_lossy(&output.stdout).trim()
            );
            download_with_cli_command("hf", repo_id)
        }
        _ => {
            let hfcli_check = std::process::Command::new("huggingface-cli")
                .arg("--version")
                .output();

            match hfcli_check {
                Ok(output) if output.status.success() => {
                    debug!(
                        "huggingface-cli found: {}",
                        String::from_utf8_lossy(&output.stdout).trim()
                    );
                    download_with_cli_command("huggingface-cli", repo_id)
                }
                _ => {
                    Err(DownloadError::CliFallbackFailed {
                        reason: "Neither 'hf' nor 'huggingface-cli' found on PATH".to_string(),
                    })
                }
            }
        }
    }
}

/// Run the actual CLI download command.
fn download_with_cli_command(cmd: &str, repo_id: &str) -> Result<PathBuf, DownloadError> {
    info!(cmd = %cmd, repo = %repo_id, "Downloading via CLI");

    let output = std::process::Command::new(cmd)
        .args(["download", repo_id])
        .output()
        .map_err(|e| DownloadError::CliFallbackFailed {
            reason: format!("Failed to execute '{}': {}", cmd, e),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(DownloadError::CliFallbackFailed {
            reason: format!("{} download failed: {}", cmd, stderr.trim()),
        });
    }

    // Parse the output to find the download directory
    // The hf CLI prints the snapshot path on the last line
    let stdout = String::from_utf8_lossy(&output.stdout);
    let download_path = stdout
        .lines()
        .last()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| DownloadError::CliFallbackFailed {
            reason: format!("{} produced no output path", cmd),
        })?;

    if !download_path.exists() {
        return Err(DownloadError::CliFallbackFailed {
            reason: format!(
                "Downloaded path does not exist: {}",
                download_path.display()
            ),
        });
    }

    info!(path = %download_path.display(), "Model downloaded via CLI");

    Ok(download_path)
}

/// Resolve an HF auth token from available sources (Story 3.2).
///
/// Resolution order:
/// 1. HF_TOKEN environment variable
/// 2. HUGGING_FACE_HUB_TOKEN environment variable (legacy)
/// 3. Standard hf-hub token path (~/.cache/huggingface/token)
/// 4. Legacy token path (~/.huggingface/token)
fn resolve_auth_token() -> Option<String> {
    // 1. Environment variable
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            debug!("Using HF_TOKEN from environment");
            return Some(token);
        }
    }

    // 2. Legacy env var
    if let Ok(token) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        if !token.is_empty() {
            debug!("Using HUGGING_FACE_HUB_TOKEN from environment");
            return Some(token);
        }
    }

    // 3. Standard hf-hub cache token path
    if let Some(home) = home_dir() {
        let cache_token = home.join(".cache").join("huggingface").join("token");
        if let Some(token) = read_token_file(&cache_token) {
            debug!(path = %cache_token.display(), "Using token from cache directory");
            return Some(token);
        }

        // 4. Legacy token path
        let legacy_token = home.join(".huggingface").join("token");
        if let Some(token) = read_token_file(&legacy_token) {
            debug!(path = %legacy_token.display(), "Using token from legacy path");
            return Some(token);
        }
    }

    debug!("No HuggingFace auth token found");
    None
}

/// Read a token from a file, returning None if the file doesn't exist or is empty.
fn read_token_file(path: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Get the user's home directory.
fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("USERPROFILE")
                .ok()
                .map(PathBuf::from)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_auth_token_from_env() {
        // Save and restore original env
        let original = std::env::var("HF_TOKEN").ok();
        std::env::set_var("HF_TOKEN", "test_token_12345");

        let token = resolve_auth_token();
        assert_eq!(token, Some("test_token_12345".to_string()));

        // Restore
        match original {
            Some(val) => std::env::set_var("HF_TOKEN", val),
            None => std::env::remove_var("HF_TOKEN"),
        }
    }

    #[test]
    fn test_resolve_auth_token_empty_env() {
        let original = std::env::var("HF_TOKEN").ok();
        let original2 = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();
        std::env::set_var("HF_TOKEN", "");
        std::env::set_var("HUGGING_FACE_HUB_TOKEN", "");

        // Should not return empty string
        let token = resolve_auth_token();
        if let Some(ref t) = token {
            assert!(!t.is_empty());
        }

        match original {
            Some(val) => std::env::set_var("HF_TOKEN", val),
            None => std::env::remove_var("HF_TOKEN"),
        }
        match original2 {
            Some(val) => std::env::set_var("HUGGING_FACE_HUB_TOKEN", val),
            None => std::env::remove_var("HUGGING_FACE_HUB_TOKEN"),
        }
    }

    #[test]
    fn test_read_token_file_missing() {
        assert!(read_token_file(std::path::Path::new("/nonexistent/path/token")).is_none());
    }

    #[test]
    fn test_read_token_file_valid() {
        let tmp = tempfile::tempdir().unwrap();
        let token_path = tmp.path().join("token");
        std::fs::write(&token_path, "hf_test_token_abc\n").unwrap();

        let token = read_token_file(&token_path);
        assert_eq!(token, Some("hf_test_token_abc".to_string()));
    }

    #[test]
    fn test_read_token_file_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let token_path = tmp.path().join("token");
        std::fs::write(&token_path, "  \n").unwrap();

        let token = read_token_file(&token_path);
        assert!(token.is_none());
    }

    #[test]
    fn test_home_dir_returns_something() {
        let home = home_dir();
        assert!(home.is_some());
    }

    #[test]
    fn test_download_error_messages_are_actionable() {
        let err = DownloadError::AuthError {
            repo: "meta-llama/Llama-3.1-8B".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("HF_TOKEN"));
        assert!(msg.contains("huggingface.co"));
        assert!(msg.contains("huggingface-cli login"));
    }

    #[test]
    fn test_download_error_repo_not_found() {
        let err = DownloadError::RepoNotFound {
            repo: "nonexistent/model".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("org/model-name"));
    }
}
