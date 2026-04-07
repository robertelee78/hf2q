//! HuggingFace Hub download wrapper for inference models.
//!
//! Downloads model files from HuggingFace Hub and caches them at
//! `~/.cache/hf2q/models/<org>/<repo>/`. Subsequent runs with the same
//! model identifier skip the download and use the cache.
//!
//! This module follows the same patterns as `input::hf_download` but differs
//! in cache management: the inference cache is persistent and user-visible,
//! whereas the conversion download uses hf-hub's internal cache.

use std::path::{Path, PathBuf};

use thiserror::Error;
use tracing::{debug, info};

/// Errors from hub download operations.
#[derive(Error, Debug)]
pub enum HubError {
    #[error(
        "Failed to download model from HuggingFace Hub: {reason}\n\
         \n\
         Troubleshooting:\n\
         - Check your network connection\n\
         - For gated models, ensure you have accepted the license at huggingface.co\n\
         - Set HF_TOKEN env var or run: huggingface-cli login"
    )]
    DownloadFailed { reason: String },

    #[error(
        "Authentication failed for repository '{repo}'.\n\
         \n\
         This model may be gated or private. To access it:\n\
         1. Accept the model license at https://huggingface.co/{repo}\n\
         2. Set your token: export HF_TOKEN=hf_xxxx\n\
         3. Alternatively, run: huggingface-cli login"
    )]
    AuthError { repo: String },

    #[error(
        "Repository not found: {repo}\n\
         \n\
         Check that the repository ID is correct (format: org/model-name).\n\
         Example: mlx-community/gemma-4-26b-a4b-it-4bit"
    )]
    RepoNotFound { repo: String },

    #[error(
        "No safetensors files found in repository '{repo}'.\n\
         The repository exists but contains no model weight files."
    )]
    NoModelFiles { repo: String },

    #[error("Local model path does not exist: {path}")]
    LocalPathNotFound { path: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Files required to be present for a valid model.
const REQUIRED_FILES: &[&str] = &["config.json"];

/// Files to download if present (non-fatal if missing).
const OPTIONAL_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "generation_config.json",
    "quantization_config.json",
    "chat_template.jinja",
];

/// Resolve a model identifier to a local path.
///
/// If `model` looks like a local filesystem path (contains `/` or `\` and exists
/// on disk, or starts with `.` or `/`), it is returned directly. Otherwise it is
/// treated as a HuggingFace Hub model identifier and downloaded.
pub fn resolve_model_path(model: &str) -> Result<PathBuf, HubError> {
    let path = Path::new(model);

    // Check for explicit local paths
    if path.is_absolute() || model.starts_with('.') {
        if path.exists() && path.is_dir() {
            info!(path = %path.display(), "Using local model directory");
            return Ok(path.to_path_buf());
        }
        return Err(HubError::LocalPathNotFound {
            path: model.to_string(),
        });
    }

    // Check if it looks like a local directory that exists
    if path.exists() && path.is_dir() {
        info!(path = %path.display(), "Using local model directory");
        return Ok(path.to_path_buf());
    }

    // Check if it's a Hub identifier (org/model format)
    if model.contains('/') && !path.exists() {
        return download_model(model);
    }

    // If it doesn't contain '/' and doesn't exist locally, it might be a model
    // name without org prefix — return a helpful error
    Err(HubError::DownloadFailed {
        reason: format!(
            "'{}' is not a local directory and doesn't look like a Hub model ID. \
             Hub model IDs use the format: org/model-name (e.g., mlx-community/gemma-4-26b-a4b-it-4bit)",
            model
        ),
    })
}

/// Get the hf2q model cache directory.
///
/// Returns `~/.cache/hf2q/models/`. Creates the directory if it doesn't exist.
fn cache_dir() -> Result<PathBuf, HubError> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| HubError::DownloadFailed {
            reason: "Could not determine home directory".to_string(),
        })?;

    let cache = PathBuf::from(home)
        .join(".cache")
        .join("hf2q")
        .join("models");

    std::fs::create_dir_all(&cache)?;
    Ok(cache)
}

/// Download a model from HuggingFace Hub.
///
/// Returns the local cache path. Files are cached at
/// `~/.cache/hf2q/models/<org>/<repo>/`. If the model is already cached
/// (config.json exists in cache), the download is skipped.
pub fn download_model(model_id: &str) -> Result<PathBuf, HubError> {
    let cache = cache_dir()?;

    // Build cache path from model ID (org/repo -> org/repo/)
    let model_cache_dir = cache.join(model_id);

    // Check if already cached — config.json is the minimum indicator
    if model_cache_dir.join("config.json").exists() {
        info!(
            path = %model_cache_dir.display(),
            "Model found in cache, skipping download"
        );
        return Ok(model_cache_dir);
    }

    info!(repo = %model_id, "Downloading model from HuggingFace Hub");

    download_via_hf_hub(model_id, &model_cache_dir)
}

/// Download model files using the hf-hub crate.
fn download_via_hf_hub(
    repo_id: &str,
    target_dir: &PathBuf,
) -> Result<PathBuf, HubError> {
    use hf_hub::api::sync::ApiBuilder;

    // Resolve auth token
    let token = resolve_auth_token();
    debug!(has_token = token.is_some(), "Auth token resolution");

    let mut builder = ApiBuilder::new().with_progress(true);

    if let Some(t) = token {
        builder = builder.with_token(Some(t));
    }

    let api = builder.build().map_err(|e| HubError::DownloadFailed {
        reason: format!("Failed to initialize HuggingFace API client: {}", e),
    })?;

    let repo = api.model(repo_id.to_string());

    // Get repo info to discover files
    let repo_info = repo.info().map_err(|e| {
        let err_str = format!("{}", e);
        if err_str.contains("401") || err_str.contains("403") || err_str.contains("auth") {
            HubError::AuthError {
                repo: repo_id.to_string(),
            }
        } else if err_str.contains("404") || err_str.contains("not found") {
            HubError::RepoNotFound {
                repo: repo_id.to_string(),
            }
        } else {
            HubError::DownloadFailed {
                reason: format!("Failed to get repository info: {}", e),
            }
        }
    })?;

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
        return Err(HubError::NoModelFiles {
            repo: repo_id.to_string(),
        });
    }

    // Build the download list
    let mut files_to_download: Vec<&str> = Vec::new();

    // Required files
    for required in REQUIRED_FILES {
        if all_files.iter().any(|f| f.as_str() == *required) {
            files_to_download.push(required);
        } else {
            return Err(HubError::DownloadFailed {
                reason: format!("Required file '{}' not found in repository", required),
            });
        }
    }

    // Optional files
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

    // Create target directory
    std::fs::create_dir_all(target_dir)?;

    let total_files = files_to_download.len();
    info!(
        file_count = total_files,
        "Downloading model files to {}",
        target_dir.display()
    );

    // Download each file via hf-hub (which handles its own caching), then
    // symlink or copy into our cache directory.
    for filename in &files_to_download {
        debug!(file = %filename, "Downloading");

        let local_path = repo.get(filename).map_err(|e| {
            let err_str = format!("{}", e);
            if err_str.contains("401") || err_str.contains("403") {
                HubError::AuthError {
                    repo: repo_id.to_string(),
                }
            } else {
                HubError::DownloadFailed {
                    reason: format!("Failed to download '{}': {}", filename, e),
                }
            }
        })?;

        // Create any subdirectories needed in the target path
        let target_path = target_dir.join(filename);
        if let Some(parent) = target_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Symlink the hf-hub cached file into our cache directory.
        // Fall back to hard link, then copy.
        if !target_path.exists() {
            if std::os::unix::fs::symlink(&local_path, &target_path).is_err() {
                if std::fs::hard_link(&local_path, &target_path).is_err() {
                    std::fs::copy(&local_path, &target_path).map_err(|e| {
                        HubError::DownloadFailed {
                            reason: format!(
                                "Failed to copy '{}' to cache: {}",
                                filename, e
                            ),
                        }
                    })?;
                }
            }
        }
    }

    info!(
        path = %target_dir.display(),
        "Model downloaded and cached ({} files)",
        total_files
    );

    Ok(target_dir.clone())
}

/// Resolve an HF auth token from available sources.
///
/// Resolution order:
/// 1. HF_TOKEN environment variable
/// 2. HUGGING_FACE_HUB_TOKEN environment variable (legacy)
/// 3. Standard hf-hub token path (~/.cache/huggingface/token)
/// 4. Legacy token path (~/.huggingface/token)
fn resolve_auth_token() -> Option<String> {
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            return Some(token);
        }
    }

    if let Ok(token) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        if !token.is_empty() {
            return Some(token);
        }
    }

    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()?;
    let home = PathBuf::from(home);

    let cache_token = home.join(".cache").join("huggingface").join("token");
    if let Some(token) = read_token_file(&cache_token) {
        return Some(token);
    }

    let legacy_token = home.join(".huggingface").join("token");
    if let Some(token) = read_token_file(&legacy_token) {
        return Some(token);
    }

    None
}

/// Read a token from a file, returning None if the file doesn't exist or is empty.
fn read_token_file(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_model_path_local_dir() {
        // An existing directory should be returned directly
        let tmp = tempfile::tempdir().unwrap();
        let result = resolve_model_path(tmp.path().to_str().unwrap());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), tmp.path());
    }

    #[test]
    fn test_resolve_model_path_absolute_nonexistent() {
        let result = resolve_model_path("/nonexistent/model/path");
        assert!(result.is_err());
        match result.unwrap_err() {
            HubError::LocalPathNotFound { path } => {
                assert!(path.contains("nonexistent"));
            }
            other => panic!("Expected LocalPathNotFound, got: {}", other),
        }
    }

    #[test]
    fn test_resolve_model_path_relative_dot_nonexistent() {
        let result = resolve_model_path("./nonexistent_model");
        assert!(result.is_err());
        match result.unwrap_err() {
            HubError::LocalPathNotFound { .. } => {}
            other => panic!("Expected LocalPathNotFound, got: {}", other),
        }
    }

    #[test]
    fn test_resolve_model_path_hub_format_recognized() {
        // Verify that org/model format is recognized as a Hub identifier
        // (not treated as a local path). Use a fake model to avoid actual download.
        let result = resolve_model_path("fake-org-abc123/nonexistent-model-xyz789");
        // Should attempt download and fail, but NOT return LocalPathNotFound
        match &result {
            Ok(_) => {} // If it somehow succeeds (cached), that's fine
            Err(HubError::LocalPathNotFound { .. }) => {
                panic!("Hub-format ID should not produce LocalPathNotFound")
            }
            Err(_) => {} // Any download error is expected
        }
    }

    #[test]
    fn test_cache_dir_creates_directory() {
        let dir = cache_dir().unwrap();
        assert!(dir.exists());
        assert!(dir.ends_with("hf2q/models"));
    }

    #[test]
    fn test_resolve_auth_token_from_env() {
        let original = std::env::var("HF_TOKEN").ok();
        std::env::set_var("HF_TOKEN", "test_token_hub");
        let token = resolve_auth_token();
        assert_eq!(token, Some("test_token_hub".to_string()));
        match original {
            Some(val) => std::env::set_var("HF_TOKEN", val),
            None => std::env::remove_var("HF_TOKEN"),
        }
    }

    #[test]
    fn test_read_token_file_missing() {
        assert!(read_token_file(Path::new("/nonexistent/path/token")).is_none());
    }

    #[test]
    fn test_read_token_file_valid() {
        let tmp = tempfile::tempdir().unwrap();
        let token_path = tmp.path().join("token");
        std::fs::write(&token_path, "hf_test_token_xyz\n").unwrap();
        let token = read_token_file(&token_path);
        assert_eq!(token, Some("hf_test_token_xyz".to_string()));
    }

    #[test]
    fn test_read_token_file_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let token_path = tmp.path().join("token");
        std::fs::write(&token_path, "  \n").unwrap();
        assert!(read_token_file(&token_path).is_none());
    }

    #[test]
    fn test_error_messages_are_actionable() {
        let err = HubError::AuthError {
            repo: "test/model".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("HF_TOKEN"));
        assert!(msg.contains("huggingface.co"));
    }
}
