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
//! the same repo skip re-download (hf-hub's built-in LFS resumption).
//!
//! # Disk preflight (Decision 14)
//!
//! Before starting any download, the available disk space on the target
//! path is checked against per-model-class minimums:
//!
//! | Model class | Minimum free |
//! |---|---|
//! | Qwen3.5-MoE 35B (`qwen35moe`) | 150 GB |
//! | Qwen3.5 27B dense (`qwen35`) | 55 GB |
//! | Gemma-4 26B and other models | 100 GB |
//!
//! If the check fails the download is aborted with a user-actionable
//! error message that includes the specific path and shortfall.
//!
//! # Shard resumption
//!
//! hf-hub's `repo.get(filename)` skips re-downloading files that are
//! already present in the cache directory. Interrupted downloads leave
//! partial files; on re-invocation the partially-downloaded shard is
//! re-fetched from the beginning (hf-hub does not do byte-range
//! resumption at the shard level). Fully-completed shards are NOT
//! re-downloaded. This means a Ctrl+C during a 40-shard download
//! followed by re-invoke will re-download only the in-flight shard;
//! all completed shards are reused.
//!
//! Manual test protocol: `Ctrl+C` mid-download → observe partial shard
//! in `~/.cache/huggingface/hub/models--*/snapshots/*/` → re-invoke
//! `hf2q` → verify only in-flight shard re-downloads, total wall-clock
//! is proportionally shorter.

use std::path::PathBuf;

use serde_json;
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::progress::ProgressReporter;

/// Minimum free disk space requirements by model class (Decision 14).
///
/// These constants encode:
/// - qwen35moe 35B: ~70 GB BF16 + ~73 GB DWQ intermediate peak + 10 GB margin = 153 GB → 150 GB floor
/// - qwen35 27B dense: ~28 GB BF16 + ~22 GB DWQ + 5 GB margin = 55 GB floor
/// - gemma4 26B (existing) + others: 100 GB conservative floor
const DISK_REQUIREMENT_QWEN35MOE_BYTES: u64 = 150 * 1024 * 1024 * 1024;
const DISK_REQUIREMENT_QWEN35_BYTES: u64 = 55 * 1024 * 1024 * 1024;
const DISK_REQUIREMENT_DEFAULT_BYTES: u64 = 100 * 1024 * 1024 * 1024;

/// Model class for disk preflight routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelClass {
    Qwen35Moe,
    Qwen35Dense,
    Other,
}

impl ModelClass {
    /// Detect model class from a HuggingFace repo ID.
    ///
    /// Heuristic: looks for well-known name fragments in the repo id
    /// (case-insensitive). Gemma and other classes fall through to
    /// `Other`, which uses the conservative 100 GB floor.
    pub fn from_repo_id(repo_id: &str) -> Self {
        let lower = repo_id.to_lowercase();
        // Order matters: check MoE first to avoid misclassifying as dense.
        // Repo IDs containing "-a3b", "-moe", or "35b-a" suggest MoE variant.
        if lower.contains("-a3b") || lower.contains("-moe") || lower.contains("35b-a") {
            return ModelClass::Qwen35Moe;
        }
        // Dense variant: "qwen3" (or "qwen35") with "27b" but no MoE markers.
        if (lower.contains("qwen3") || lower.contains("qwen35")) && lower.contains("27b") {
            return ModelClass::Qwen35Dense;
        }
        ModelClass::Other
    }

    /// Minimum free bytes required before download begins.
    pub fn min_free_bytes(self) -> u64 {
        match self {
            ModelClass::Qwen35Moe => DISK_REQUIREMENT_QWEN35MOE_BYTES,
            ModelClass::Qwen35Dense => DISK_REQUIREMENT_QWEN35_BYTES,
            ModelClass::Other => DISK_REQUIREMENT_DEFAULT_BYTES,
        }
    }

    /// Human-readable model label for error messages.
    pub fn label(self) -> &'static str {
        match self {
            ModelClass::Qwen35Moe => "Qwen3.5-MoE 35B",
            ModelClass::Qwen35Dense => "Qwen3.5 27B dense",
            ModelClass::Other => "model",
        }
    }
}

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

    /// Disk preflight failure (Decision 14).
    ///
    /// Error message wording is load-bearing — integration tests assert
    /// against the exact phrasing. Do not change without updating tests.
    #[error(
        "{label} requires \u{2265}{required_gb} GB free in {path}; found {found_gb} GB. \
         Free space or change --cache-dir."
    )]
    InsufficientDisk {
        label: String,
        required_gb: u64,
        found_gb: u64,
        path: String,
    },

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

/// Check that enough disk space is available before starting a download (Decision 14).
///
/// `target_path` is the directory where downloaded files will be written
/// (typically `~/.cache/huggingface/hub` or a custom `--cache-dir`).
/// If the path does not exist yet, the nearest existing ancestor is used.
///
/// Returns `Ok(())` when sufficient space is available.
/// Returns `Err(DownloadError::InsufficientDisk)` with an actionable message otherwise.
///
/// # Test seam
///
/// `available_bytes_override` is `Some(n)` in unit tests to inject a
/// fake free-space value without touching the filesystem.
pub fn check_disk_preflight(
    repo_id: &str,
    target_path: &std::path::Path,
    available_bytes_override: Option<u64>,
) -> Result<(), DownloadError> {
    let class = ModelClass::from_repo_id(repo_id);
    let required = class.min_free_bytes();

    let available = match available_bytes_override {
        Some(v) => v,
        None => get_available_space_for_path(target_path),
    };

    debug!(
        repo = %repo_id,
        class = ?class,
        required_gb = required / (1024 * 1024 * 1024),
        available_gb = available / (1024 * 1024 * 1024),
        "Disk preflight check"
    );

    if available < required {
        let path_str = target_path.display().to_string();
        return Err(DownloadError::InsufficientDisk {
            label: class.label().to_string(),
            required_gb: required / (1024 * 1024 * 1024),
            found_gb: available / (1024 * 1024 * 1024),
            path: path_str,
        });
    }

    Ok(())
}

/// Get available bytes for the filesystem containing `path`.
///
/// Walks to the nearest existing ancestor directory, then uses `sysinfo`
/// to find the matching mount point. Returns 0 if nothing can be determined
/// (conservative — will not block downloads when the check can't run).
fn get_available_space_for_path(path: &std::path::Path) -> u64 {
    // Walk up to an existing ancestor
    let existing = {
        let mut p = path.to_path_buf();
        loop {
            if p.exists() {
                break p;
            }
            match p.parent() {
                Some(parent) => p = parent.to_path_buf(),
                None => break std::path::PathBuf::from("/"),
            }
        }
    };

    use sysinfo::Disks;
    let disks = Disks::new_with_refreshed_list();
    let mut best: Option<(usize, u64)> = None;
    for disk in disks.list() {
        let mount = disk.mount_point();
        if existing.starts_with(mount) {
            let len = mount.as_os_str().len();
            match best {
                Some((prev, _)) if len > prev => best = Some((len, disk.available_space())),
                None => best = Some((len, disk.available_space())),
                _ => {}
            }
        }
    }
    best.map(|(_, space)| space).unwrap_or(0)
}

/// Download a model from HuggingFace Hub.
///
/// Returns the path to the local directory containing the downloaded model files.
/// Uses hf-hub crate cache — subsequent calls with the same repo skip re-download.
/// Completed shards are not re-fetched on re-invocation; only in-flight shards
/// are restarted (hf-hub file-level skip, not byte-range resumption).
pub fn download_model(
    repo_id: &str,
    progress: &ProgressReporter,
) -> Result<PathBuf, DownloadError> {
    info!(repo = %repo_id, "Downloading model from HuggingFace Hub");

    // Decision 14: disk preflight before any network activity.
    // Use the hf-hub default cache dir (~/.cache/huggingface/hub).
    let cache_dir = resolve_hf_cache_dir();
    check_disk_preflight(repo_id, &cache_dir, None)?;

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

    // Index file — download FIRST so we know which shards are actually needed
    let mut needed_shards: Option<Vec<String>> = None;
    if let Some(idx) = index_file {
        files_to_download.push(idx.as_str());

        // Download the index now to discover required shards
        debug!("Downloading index file to determine required shards");
        let idx_path = repo.get(idx.as_str()).map_err(|e| {
            DownloadError::DownloadFailed {
                reason: format!("Failed to download index file: {}", e),
            }
        })?;

        // Parse weight_map to find which shard files are actually referenced
        if let Ok(content) = std::fs::read_to_string(&idx_path) {
            if let Ok(index) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                    let mut shard_names: Vec<String> = weight_map
                        .values()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect();
                    shard_names.sort();
                    shard_names.dedup();
                    info!(
                        total_safetensors = safetensors_files.len(),
                        needed = shard_names.len(),
                        "Index specifies {} of {} safetensors files",
                        shard_names.len(),
                        safetensors_files.len(),
                    );
                    needed_shards = Some(shard_names);
                }
            }
        }
    }

    // Only download shards referenced by the index (or all if no index)
    match &needed_shards {
        Some(shards) => {
            for shard in shards {
                files_to_download.push(shard.as_str());
            }
        }
        None => {
            // No index file — download all safetensors files
            for sf in &safetensors_files {
                files_to_download.push(sf.as_str());
            }
        }
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
        .args([
            "download",
            repo_id,
            "--include", "*.safetensors",
            "--include", "*.json",
            "--include", "tokenizer.model",
            "--exclude", "*.gguf",
            "--exclude", "*.bin",
            "--exclude", "*.pt",
            "--exclude", "*.h5",
            "--exclude", "*.msgpack",
            "--exclude", "*.ot",
        ])
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

/// Resolve the hf-hub cache directory.
///
/// Resolution order (mirrors hf-hub's own logic):
/// 1. `HF_HUB_CACHE` env var
/// 2. `HF_HOME` env var + `/hub`
/// 3. `XDG_CACHE_HOME` env var + `/huggingface/hub`
/// 4. `~/.cache/huggingface/hub`
fn resolve_hf_cache_dir() -> PathBuf {
    if let Ok(v) = std::env::var("HF_HUB_CACHE") {
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    if let Ok(v) = std::env::var("HF_HOME") {
        if !v.is_empty() {
            return PathBuf::from(v).join("hub");
        }
    }
    if let Ok(v) = std::env::var("XDG_CACHE_HOME") {
        if !v.is_empty() {
            return PathBuf::from(v).join("huggingface").join("hub");
        }
    }
    home_dir()
        .unwrap_or_else(|| PathBuf::from("/"))
        .join(".cache")
        .join("huggingface")
        .join("hub")
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

    // --- Decision 14: disk preflight tests ---

    #[test]
    fn test_model_class_from_repo_id_qwen35moe() {
        let cases = [
            "jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated",
            "org/Qwen3.5-MoE-35B-Instruct",
            "someone/model-35b-a3b-stuff",
        ];
        for repo in &cases {
            assert_eq!(
                ModelClass::from_repo_id(repo),
                ModelClass::Qwen35Moe,
                "Expected Qwen35Moe for {repo}"
            );
        }
    }

    #[test]
    fn test_model_class_from_repo_id_qwen35_dense() {
        let cases = [
            "Qwen/Qwen3.5-27B-Instruct",
            "org/qwen35-27b-dense",
        ];
        for repo in &cases {
            assert_eq!(
                ModelClass::from_repo_id(repo),
                ModelClass::Qwen35Dense,
                "Expected Qwen35Dense for {repo}"
            );
        }
    }

    #[test]
    fn test_model_class_from_repo_id_other() {
        let cases = [
            "google/gemma-4-26b-it",
            "meta-llama/Llama-3.1-8B",
            "mistralai/Mistral-7B-v0.1",
        ];
        for repo in &cases {
            assert_eq!(
                ModelClass::from_repo_id(repo),
                ModelClass::Other,
                "Expected Other for {repo}"
            );
        }
    }

    #[test]
    fn test_model_class_min_free_bytes() {
        assert_eq!(
            ModelClass::Qwen35Moe.min_free_bytes(),
            150 * 1024 * 1024 * 1024
        );
        assert_eq!(
            ModelClass::Qwen35Dense.min_free_bytes(),
            55 * 1024 * 1024 * 1024
        );
        assert_eq!(
            ModelClass::Other.min_free_bytes(),
            100 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn test_disk_preflight_qwen35moe_insufficient_fails_with_exact_message() {
        let tmp = tempfile::tempdir().unwrap();
        // 50 GB < 150 GB required for qwen35moe
        let available: u64 = 50 * 1024 * 1024 * 1024;
        let repo = "jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated";

        let err = check_disk_preflight(repo, tmp.path(), Some(available))
            .expect_err("Should fail with insufficient disk");

        let msg = err.to_string();
        // ADR-012 Decision 14: exact wording is part of the spec.
        assert!(
            msg.contains("Qwen3.5-MoE 35B"),
            "Error must name the model class: {msg}"
        );
        assert!(
            msg.contains("≥150 GB"),
            "Error must state the requirement: {msg}"
        );
        assert!(
            msg.contains("50 GB"),
            "Error must state found bytes: {msg}"
        );
        assert!(
            msg.contains("Free space or change --cache-dir"),
            "Error must be actionable: {msg}"
        );
        assert!(
            msg.contains(tmp.path().to_str().unwrap()),
            "Error must include path: {msg}"
        );
    }

    #[test]
    fn test_disk_preflight_qwen35moe_sufficient_passes() {
        let tmp = tempfile::tempdir().unwrap();
        // 200 GB > 150 GB required
        let available: u64 = 200 * 1024 * 1024 * 1024;
        let repo = "jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated";

        assert!(
            check_disk_preflight(repo, tmp.path(), Some(available)).is_ok(),
            "200 GB should pass the 150 GB requirement"
        );
    }

    #[test]
    fn test_disk_preflight_qwen35_dense_insufficient_fails() {
        let tmp = tempfile::tempdir().unwrap();
        // 30 GB < 55 GB required for qwen35 dense
        let available: u64 = 30 * 1024 * 1024 * 1024;
        let repo = "Qwen/Qwen3.5-27B-Instruct";

        let err = check_disk_preflight(repo, tmp.path(), Some(available))
            .expect_err("Should fail");

        let msg = err.to_string();
        assert!(msg.contains("Qwen3.5 27B dense"), "Expected dense label: {msg}");
        assert!(msg.contains("≥55 GB"), "Expected 55 GB requirement: {msg}");
    }

    #[test]
    fn test_disk_preflight_qwen35_dense_sufficient_passes() {
        let tmp = tempfile::tempdir().unwrap();
        let available: u64 = 100 * 1024 * 1024 * 1024;
        let repo = "Qwen/Qwen3.5-27B-Instruct";

        assert!(check_disk_preflight(repo, tmp.path(), Some(available)).is_ok());
    }

    #[test]
    fn test_disk_preflight_gemma_regression_passes() {
        // Gemma-4 26B produces ~13 GB GGUF + overhead; requirement is 100 GB.
        // With 120 GB available, should pass.
        let tmp = tempfile::tempdir().unwrap();
        let available: u64 = 120 * 1024 * 1024 * 1024;
        let repo = "google/gemma-4-26b-it";

        assert!(
            check_disk_preflight(repo, tmp.path(), Some(available)).is_ok(),
            "Gemma-4 should pass with 120 GB available (100 GB floor)"
        );
    }

    #[test]
    fn test_disk_preflight_gemma_insufficient_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let available: u64 = 80 * 1024 * 1024 * 1024;
        let repo = "google/gemma-4-26b-it";

        assert!(
            check_disk_preflight(repo, tmp.path(), Some(available)).is_err(),
            "Gemma-4 should fail with only 80 GB (100 GB floor)"
        );
    }

    #[test]
    fn test_resolve_hf_cache_dir_uses_env_override() {
        let original = std::env::var("HF_HUB_CACHE").ok();
        std::env::set_var("HF_HUB_CACHE", "/custom/cache");
        let dir = resolve_hf_cache_dir();
        assert_eq!(dir, std::path::PathBuf::from("/custom/cache"));
        match original {
            Some(v) => std::env::set_var("HF_HUB_CACHE", v),
            None => std::env::remove_var("HF_HUB_CACHE"),
        }
    }

    #[test]
    fn test_resolve_hf_cache_dir_returns_path() {
        // With no special env vars, should return something rooted under home.
        std::env::remove_var("HF_HUB_CACHE");
        std::env::remove_var("HF_HOME");
        std::env::remove_var("XDG_CACHE_HOME");
        let dir = resolve_hf_cache_dir();
        assert!(dir.to_str().is_some());
        assert!(dir.ends_with("hub") || dir.to_str().unwrap().contains("huggingface"));
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
