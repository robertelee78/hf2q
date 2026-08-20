//! HuggingFace Hub download integration (Epic 3).
//!
//! Downloads model files from Hugging Face Hub through the in-process
//! `hf-hub` client. Production never shells out to `hf` or
//! `huggingface-cli`.
//!
//! Token resolution order (Story 3.2):
//! 1. HF_TOKEN environment variable
//! 2. HUGGING_FACE_HUB_TOKEN environment variable (legacy)
//! 3. ~/.cache/huggingface/token file (hf-hub default)
//! 4. ~/.huggingface/token file (legacy path)
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
//! The preparation-resolution seam performs only the repository-info lookup:
//! it consumes the host-checked plan, binds its exact original reference to an
//! immutable commit and bounded name inventory, and returns an inert sealed
//! plan before any payload transfer.
//!
//! Manual test protocol: `Ctrl+C` mid-download → observe partial shard
//! in `~/.cache/huggingface/hub/models--*/snapshots/*/` → re-invoke
//! `hf2q` → verify only in-flight shard re-downloads, total wall-clock
//! is proportionally shorter.

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use thiserror::Error;
use tracing::{debug, info};

use crate::core::integrity::{verify_shard, IntegrityError, ShardIntegrity};
use crate::input::hf_reference::{HfModelReference, HfReferenceError, ResolvedHfModelReference};
use crate::input::model_recipe::{
    ModelPreparationPlan, QWEN38_ACCEPTED_REVISION, QWEN38_REPOSITORY_ID,
};
use crate::progress::ProgressReporter;

mod resolution;

use resolution::{bind_model_preparation_resolution, resolve_repository_info};
#[cfg(test)]
pub(in crate::input) use resolution::{
    bind_model_preparation_resolution_for_test, resolve_repository_info_for_test,
};
pub use resolution::{
    ModelPreparationResolutionError, ResolvedModelPreparationPlan, ResolvedModelRepository,
};

const CANONICAL_HF_ENDPOINT: &str = "https://huggingface.co";
const DEFAULT_HF_REVISION: &str = "main";
pub(super) const MAX_HF_REPO_FILES: usize = 4096;
pub(super) const MAX_HF_SMALL_METADATA_BYTES: u64 = 16 * 1024 * 1024;
pub(super) const MAX_HF_TOKENIZER_BYTES: u64 = 512 * 1024 * 1024;

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
        "Failed to download from Hugging Face Hub: {reason}\n\
         \n\
         Troubleshooting:\n\
         - Check your network connection\n\
         - For gated models, ensure you have accepted the license at huggingface.co\n\
         - Set HF_TOKEN for private or gated repositories"
    )]
    DownloadFailed { reason: String },

    #[error(
        "Authentication failed for repository '{repo}'.\n\
         \n\
         This model may be gated or private. To access it:\n\
         1. Accept the model license at https://huggingface.co/{repo}\n\
         2. Set your token: export HF_TOKEN=hf_xxxx\n\
            Or create ~/.huggingface/token with your token\n\
         3. Retry the same hf2q command"
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

    #[error(transparent)]
    InvalidReference(#[from] HfReferenceError),

    #[error(transparent)]
    Integrity(#[from] IntegrityError),

    #[error("invalid or unsupported Hugging Face repository inventory: {reason}")]
    InvalidRepositoryInventory { reason: String },

    #[error("a file-specific Hugging Face URL cannot be converted as a model repository")]
    FileReferenceUnsupported,

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
    "README.md",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "merges.txt",
    "vocab.json",
];

/// A completed native download bound to the exact Hub commit resolved before
/// any model payload was transferred.
#[derive(Debug)]
pub struct DownloadedModel {
    local_dir: PathBuf,
    reference: ResolvedHfModelReference,
    manifest: Vec<ShardIntegrity>,
}

impl DownloadedModel {
    pub fn local_dir(&self) -> &Path {
        &self.local_dir
    }

    pub fn reference(&self) -> &ResolvedHfModelReference {
        &self.reference
    }

    pub fn manifest(&self) -> &[ShardIntegrity] {
        &self.manifest
    }

    pub fn into_parts(self) -> (PathBuf, ResolvedHfModelReference, Vec<ShardIntegrity>) {
        (self.local_dir, self.reference, self.manifest)
    }
}

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
    let reference = HfModelReference::parse(repo_id, None)?;
    download_model_reference(reference, progress).map(|download| download.local_dir)
}

/// Resolve a parsed model reference to one immutable commit, then download the
/// exact source-format inventory required by conversion from that commit.
pub fn download_model_reference(
    reference: HfModelReference,
    progress: &ProgressReporter,
) -> Result<DownloadedModel, DownloadError> {
    if reference.filename().is_some() {
        return Err(DownloadError::FileReferenceUnsupported);
    }
    info!(repo = %reference.repo_id(), "Downloading model from Hugging Face Hub");

    // Decision 14: disk preflight before any network activity.
    // Use the hf-hub default cache dir (~/.cache/huggingface/hub).
    let cache_dir = resolve_hf_cache_dir();
    check_disk_preflight(reference.repo_id(), &cache_dir, None)?;
    download_via_hf_hub(reference, &cache_dir, progress)
}

/// Resolve one parsed model reference to an exact commit and bounded
/// repository inventory without transferring model payloads.
pub fn resolve_model_reference(
    reference: HfModelReference,
) -> Result<ResolvedModelRepository, DownloadError> {
    if reference.filename().is_some() {
        return Err(DownloadError::FileReferenceUnsupported);
    }
    let cache_dir = resolve_hf_cache_dir();
    let api = build_hub_api(&cache_dir, false)?;
    resolve_with_api(&api, reference)
}

/// Consume a host-checked preparation plan only after resolving the exact
/// original reference through the pinned in-process Hub boundary.
pub fn resolve_model_preparation_plan(
    plan: ModelPreparationPlan,
) -> Result<ResolvedModelPreparationPlan, ModelPreparationResolutionError> {
    let resolution = resolve_model_reference(plan.reference().clone())?;
    Ok(bind_model_preparation_resolution(plan, resolution)?)
}

/// Download model files using the hf-hub crate.
fn download_via_hf_hub(
    reference: HfModelReference,
    cache_dir: &Path,
    progress: &ProgressReporter,
) -> Result<DownloadedModel, DownloadError> {
    use hf_hub::{Repo, RepoType};

    let api = build_hub_api(cache_dir, true)?;
    let (resolved, inventory) = resolve_with_api(&api, reference)?.into_download_parts();
    let repo = api.repo(Repo::with_revision(
        resolved.repo_id().to_owned(),
        RepoType::Model,
        resolved.revision().to_owned(),
    ));

    debug!(
        file_count = inventory.len(),
        revision = resolved.revision(),
        "Repository file listing retrieved"
    );
    let initial_files = initial_download_files(&inventory)?;
    let mut downloaded_path = None;
    let mut manifest = Vec::with_capacity(initial_files.len());
    for filename in &initial_files {
        let record = fetch_expected_file_metadata(&api, &repo, &resolved, filename)?;
        let local = download_file(&repo, resolved.repo_id(), filename)?;
        bind_snapshot_parent(&mut downloaded_path, &local, filename, resolved.revision())?;
        verify_shard(resolved.repo_id(), resolved.revision(), &local, &record)?;
        manifest.push(record);
    }
    let model_dir = downloaded_path
        .clone()
        .ok_or_else(|| DownloadError::DownloadFailed {
            reason: "No model metadata files were downloaded".to_owned(),
        })?;
    let required_shards =
        crate::input::integrity::required_weight_shards(&model_dir).map_err(|error| {
            DownloadError::InvalidRepositoryInventory {
                reason: error.to_string(),
            }
        })?;
    let files_to_download = complete_download_files(&inventory, &initial_files, &required_shards)?;

    let additional_files = files_to_download
        .iter()
        .filter(|filename| !initial_files.contains(filename))
        .collect::<Vec<_>>();
    let pb = progress.bar(additional_files.len() as u64, "Downloading model weights");

    for filename in additional_files {
        debug!(file = %filename, "Downloading");
        let record = fetch_expected_file_metadata(&api, &repo, &resolved, filename)?;
        let local_path = download_file(&repo, resolved.repo_id(), filename)?;
        bind_snapshot_parent(
            &mut downloaded_path,
            &local_path,
            filename,
            resolved.revision(),
        )?;
        verify_shard(
            resolved.repo_id(),
            resolved.revision(),
            &local_path,
            &record,
        )?;
        manifest.push(record);
        pb.inc(1);
    }

    pb.finish_with_message(format!(
        "Selected {} exact source files",
        files_to_download.len()
    ));

    info!(path = %model_dir.display(), "Model downloaded to cache");

    manifest.sort_by(|left, right| left.filename.cmp(&right.filename));
    Ok(DownloadedModel {
        local_dir: model_dir,
        reference: resolved,
        manifest,
    })
}

fn build_hub_api(
    cache_dir: &Path,
    progress: bool,
) -> Result<hf_hub::api::sync::Api, DownloadError> {
    use hf_hub::api::sync::ApiBuilder;

    let token = resolve_auth_token();
    debug!(has_token = token.is_some(), "Auth token resolution");

    // Pin the official endpoint even when HF_ENDPOINT is set in the process.
    let mut builder = ApiBuilder::new()
        .with_endpoint(CANONICAL_HF_ENDPOINT.to_owned())
        .with_cache_dir(cache_dir.to_path_buf())
        .with_progress(progress);
    if let Some(token) = token {
        builder = builder.with_token(Some(token));
    }
    builder
        .build()
        .map_err(|error| DownloadError::DownloadFailed {
            reason: format!("Failed to initialize Hugging Face API client: {error}"),
        })
}

fn resolve_with_api(
    api: &hf_hub::api::sync::Api,
    reference: HfModelReference,
) -> Result<ResolvedModelRepository, DownloadError> {
    use hf_hub::{Repo, RepoType};

    if reference.filename().is_some() {
        return Err(DownloadError::FileReferenceUnsupported);
    }
    let requested_revision = reference
        .requested_revision()
        .unwrap_or_else(|| default_revision_for(reference.repo_id()))
        .to_owned();
    let lookup_repo = api.repo(Repo::with_revision(
        reference.repo_id().to_owned(),
        RepoType::Model,
        requested_revision.clone(),
    ));
    let repo_info = lookup_repo.info().map_err(|error| {
        let message = error.to_string();
        if message.contains("401") || message.contains("403") || message.contains("auth") {
            DownloadError::AuthError {
                repo: reference.repo_id().to_owned(),
            }
        } else if message.contains("404") || message.contains("not found") {
            DownloadError::RepoNotFound {
                repo: reference.repo_id().to_owned(),
            }
        } else {
            DownloadError::DownloadFailed {
                reason: format!("Failed to get repository info: {error}"),
            }
        }
    })?;
    resolve_repository_info(reference, &requested_revision, &repo_info)
}

fn fetch_expected_file_metadata(
    api: &hf_hub::api::sync::Api,
    repo: &hf_hub::api::sync::ApiRepo,
    resolved: &ResolvedHfModelReference,
    filename: &str,
) -> Result<ShardIntegrity, DownloadError> {
    let metadata =
        api.metadata(&repo.url(filename))
            .map_err(|error| DownloadError::DownloadFailed {
                reason: format!("Failed to fetch immutable metadata for `{filename}`: {error}"),
            })?;
    validate_file_metadata(
        filename,
        resolved.revision(),
        metadata.commit_hash(),
        metadata.etag(),
        metadata.size() as u64,
    )
}

pub(super) fn validate_file_metadata(
    filename: &str,
    expected_revision: &str,
    returned_revision: &str,
    etag: &str,
    size: u64,
) -> Result<ShardIntegrity, DownloadError> {
    validate_repo_filename(filename)?;
    if returned_revision.len() != 40
        || !returned_revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || !returned_revision.eq_ignore_ascii_case(expected_revision)
    {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "metadata for `{filename}` returned commit `{returned_revision}` instead of `{expected_revision}`"
            ),
        });
    }
    if size == 0 {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!("metadata for `{filename}` reported an empty file"),
        });
    }
    if let Some(cap) = metadata_size_cap(filename) {
        if size > cap {
            return Err(DownloadError::InvalidRepositoryInventory {
                reason: format!("metadata file `{filename}` is {size} bytes; limit is {cap}"),
            });
        }
    }
    let record = ShardIntegrity::from_metadata(filename, etag, size);
    let has_supported_identity = record.sha256.is_some()
        || (record.hf_etag.len() == 40
            && record.hf_etag.bytes().all(|byte| byte.is_ascii_hexdigit()));
    if !has_supported_identity {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!("metadata for `{filename}` has no supported immutable identity"),
        });
    }
    if filename.ends_with(".safetensors") && !record.is_lfs {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!("weight shard `{filename}` has no strong LFS SHA-256 identity"),
        });
    }
    Ok(record)
}

pub(super) fn metadata_size_cap(filename: &str) -> Option<u64> {
    if filename.ends_with(".safetensors") {
        None
    } else if matches!(
        filename,
        "tokenizer.json" | "tokenizer.model" | "merges.txt" | "vocab.json"
    ) {
        Some(MAX_HF_TOKENIZER_BYTES)
    } else {
        Some(MAX_HF_SMALL_METADATA_BYTES)
    }
}

pub(super) fn default_revision_for(repo_id: &str) -> &'static str {
    if repo_id == QWEN38_REPOSITORY_ID {
        QWEN38_ACCEPTED_REVISION
    } else {
        DEFAULT_HF_REVISION
    }
}

pub(super) fn validate_repo_inventory<'a>(
    filenames: impl IntoIterator<Item = &'a str>,
) -> Result<BTreeSet<String>, DownloadError> {
    let mut inventory = BTreeSet::new();
    for filename in filenames {
        if inventory.len() == MAX_HF_REPO_FILES {
            return Err(DownloadError::InvalidRepositoryInventory {
                reason: format!("more than {MAX_HF_REPO_FILES} files"),
            });
        }
        validate_repo_filename(filename)?;
        if !inventory.insert(filename.to_owned()) {
            return Err(DownloadError::InvalidRepositoryInventory {
                reason: format!("duplicate file `{filename}`"),
            });
        }
    }
    Ok(inventory)
}

fn validate_repo_filename(filename: &str) -> Result<(), DownloadError> {
    let path = Path::new(filename);
    let component_count = path.components().count();
    let valid = !filename.is_empty()
        && filename.len() <= crate::input::hf_reference::MAX_HF_FILENAME_BYTES
        && component_count <= crate::input::hf_reference::MAX_HF_FILENAME_COMPONENTS
        && !filename.contains('\\')
        && filename.is_ascii()
        && filename.bytes().all(|byte| !byte.is_ascii_control())
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)));
    if valid {
        Ok(())
    } else {
        Err(DownloadError::InvalidRepositoryInventory {
            reason: format!("unsafe file path `{filename}`"),
        })
    }
}

pub(super) fn initial_download_files(
    inventory: &BTreeSet<String>,
) -> Result<Vec<String>, DownloadError> {
    for required in REQUIRED_FILES {
        if !inventory.contains(*required) {
            return Err(DownloadError::InvalidRepositoryInventory {
                reason: format!("required file `{required}` is absent"),
            });
        }
    }
    let has_index = inventory.contains("model.safetensors.index.json");
    if !has_index && !inventory.contains("model.safetensors") {
        return Err(DownloadError::NoModelFiles {
            repo: "selected model repository".to_owned(),
        });
    }

    let mut files = REQUIRED_FILES
        .iter()
        .map(|filename| (*filename).to_owned())
        .collect::<Vec<_>>();
    if has_index {
        files.push("model.safetensors.index.json".to_owned());
    }
    files.extend(
        OPTIONAL_FILES
            .iter()
            .filter(|filename| inventory.contains(**filename))
            .map(|filename| (*filename).to_owned()),
    );
    Ok(files)
}

pub(super) fn complete_download_files(
    inventory: &BTreeSet<String>,
    initial_files: &[String],
    required_shards: &[String],
) -> Result<Vec<String>, DownloadError> {
    if required_shards.is_empty() {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: "weight index selected no safetensors shards".to_owned(),
        });
    }
    let mut files = initial_files.iter().cloned().collect::<BTreeSet<_>>();
    for shard in required_shards {
        validate_repo_filename(shard)?;
        if !shard.ends_with(".safetensors") || !inventory.contains(shard) {
            return Err(DownloadError::InvalidRepositoryInventory {
                reason: format!("required shard `{shard}` is absent or not safetensors"),
            });
        }
        files.insert(shard.clone());
    }
    Ok(files.into_iter().collect())
}

fn download_file(
    repo: &hf_hub::api::sync::ApiRepo,
    repo_id: &str,
    filename: &str,
) -> Result<PathBuf, DownloadError> {
    repo.get(filename).map_err(|error| {
        let rendered = error.to_string();
        if rendered.contains("401") || rendered.contains("403") {
            DownloadError::AuthError {
                repo: repo_id.to_owned(),
            }
        } else {
            DownloadError::DownloadFailed {
                reason: format!("Failed to download `{filename}`: {error}"),
            }
        }
    })
}

pub(super) fn bind_snapshot_parent(
    selected: &mut Option<PathBuf>,
    local_file: &Path,
    filename: &str,
    expected_revision: &str,
) -> Result<(), DownloadError> {
    let mut snapshot = local_file.to_path_buf();
    for _ in Path::new(filename).components() {
        if !snapshot.pop() {
            return Err(DownloadError::InvalidRepositoryInventory {
                reason: format!("download path for `{filename}` has no snapshot parent"),
            });
        }
    }
    if snapshot.file_name().and_then(|name| name.to_str()) != Some(expected_revision) {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "downloaded `{filename}` resolved outside exact snapshot `{expected_revision}`"
            ),
        });
    }
    match selected {
        Some(existing) if existing != &snapshot => Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "downloaded files crossed snapshots `{}` and `{}`",
                existing.display(),
                snapshot.display()
            ),
        }),
        Some(_) => Ok(()),
        None => {
            *selected = Some(snapshot);
            Ok(())
        }
    }
}

/// Resolve an HF auth token from available sources (Story 3.2).
///
/// Resolution order:
/// 1. HF_TOKEN environment variable
/// 2. HUGGING_FACE_HUB_TOKEN environment variable (legacy)
/// 3. Standard hf-hub token path (~/.cache/huggingface/token)
/// 4. Legacy token path (~/.huggingface/token)
pub(crate) fn resolve_auth_token() -> Option<String> {
    let hf_token = std::env::var("HF_TOKEN").ok();
    let legacy_env_token = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();
    let home = home_dir();
    let cache_token = home
        .as_ref()
        .map(|home| home.join(".cache").join("huggingface").join("token"));
    let legacy_token = home
        .as_ref()
        .map(|home| home.join(".huggingface").join("token"));

    let resolved = resolve_auth_token_from_inputs(
        hf_token.as_deref(),
        legacy_env_token.as_deref(),
        cache_token.as_deref(),
        legacy_token.as_deref(),
    );
    match resolved {
        Some((token, AuthTokenSource::HfTokenEnv)) => {
            debug!("Using HF_TOKEN from environment");
            Some(token)
        }
        Some((token, AuthTokenSource::LegacyEnv)) => {
            debug!("Using HUGGING_FACE_HUB_TOKEN from environment");
            Some(token)
        }
        Some((token, AuthTokenSource::CacheFile)) => {
            if let Some(path) = cache_token.as_ref() {
                debug!(path = %path.display(), "Using token from cache directory");
            }
            Some(token)
        }
        Some((token, AuthTokenSource::LegacyFile)) => {
            if let Some(path) = legacy_token.as_ref() {
                debug!(path = %path.display(), "Using token from legacy path");
            }
            Some(token)
        }
        None => {
            debug!("No HuggingFace auth token found");
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuthTokenSource {
    HfTokenEnv,
    LegacyEnv,
    CacheFile,
    LegacyFile,
}

/// Pure auth resolver used by the production environment wrapper and tests.
///
/// Process environment variables are global mutable state. Passing their
/// captured values and the candidate token paths into this helper keeps the
/// precedence contract deterministic without making parallel tests mutate
/// `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`.
fn resolve_auth_token_from_inputs(
    hf_token: Option<&str>,
    legacy_env_token: Option<&str>,
    cache_token_path: Option<&std::path::Path>,
    legacy_token_path: Option<&std::path::Path>,
) -> Option<(String, AuthTokenSource)> {
    if let Some(token) = hf_token.filter(|token| !token.is_empty()) {
        return Some((token.to_owned(), AuthTokenSource::HfTokenEnv));
    }
    if let Some(token) = legacy_env_token.filter(|token| !token.is_empty()) {
        return Some((token.to_owned(), AuthTokenSource::LegacyEnv));
    }
    if let Some(token) = cache_token_path.and_then(read_token_file) {
        return Some((token, AuthTokenSource::CacheFile));
    }
    legacy_token_path
        .and_then(read_token_file)
        .map(|token| (token, AuthTokenSource::LegacyFile))
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
    let hf_hub_cache = std::env::var("HF_HUB_CACHE").ok();
    let hf_home = std::env::var("HF_HOME").ok();
    let xdg_cache_home = std::env::var("XDG_CACHE_HOME").ok();
    let home = home_dir();
    resolve_hf_cache_dir_from_inputs(
        hf_hub_cache.as_deref(),
        hf_home.as_deref(),
        xdg_cache_home.as_deref(),
        home.as_deref(),
    )
}

/// Pure cache-path resolver used by the production environment wrapper and
/// deterministic tests. Keeping process-global environment access outside
/// this function prevents parallel unit tests from changing one another's
/// inputs.
fn resolve_hf_cache_dir_from_inputs(
    hf_hub_cache: Option<&str>,
    hf_home: Option<&str>,
    xdg_cache_home: Option<&str>,
    home: Option<&std::path::Path>,
) -> PathBuf {
    if let Some(value) = hf_hub_cache.filter(|value| !value.is_empty()) {
        return PathBuf::from(value);
    }
    if let Some(value) = hf_home.filter(|value| !value.is_empty()) {
        return PathBuf::from(value).join("hub");
    }
    if let Some(value) = xdg_cache_home.filter(|value| !value.is_empty()) {
        return PathBuf::from(value).join("huggingface").join("hub");
    }
    home.map(PathBuf::from)
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
        .or_else(|| std::env::var("USERPROFILE").ok().map(PathBuf::from))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_auth_token_from_env() {
        let token = resolve_auth_token_from_inputs(
            Some("test_token_12345"),
            Some("legacy_token"),
            None,
            None,
        );
        assert_eq!(
            token,
            Some(("test_token_12345".to_string(), AuthTokenSource::HfTokenEnv,))
        );

        assert_eq!(
            resolve_auth_token_from_inputs(Some(""), Some("legacy_token"), None, None),
            Some(("legacy_token".to_owned(), AuthTokenSource::LegacyEnv))
        );
    }

    #[test]
    fn test_resolve_auth_token_empty_env() {
        assert_eq!(
            resolve_auth_token_from_inputs(Some(""), Some(""), None, None),
            None
        );
    }

    #[test]
    fn test_resolve_auth_token_file_fallback_precedence() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cache_token = tmp.path().join("cache-token");
        let legacy_token = tmp.path().join("legacy-token");
        std::fs::write(&cache_token, "hf_cache\n").expect("cache token");
        std::fs::write(&legacy_token, "hf_legacy\n").expect("legacy token");

        assert_eq!(
            resolve_auth_token_from_inputs(
                Some(""),
                Some(""),
                Some(&cache_token),
                Some(&legacy_token),
            ),
            Some(("hf_cache".to_owned(), AuthTokenSource::CacheFile))
        );
        std::fs::write(&cache_token, "  \n").expect("empty cache token");
        assert_eq!(
            resolve_auth_token_from_inputs(None, None, Some(&cache_token), Some(&legacy_token)),
            Some(("hf_legacy".to_owned(), AuthTokenSource::LegacyFile))
        );
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
        let cases = ["Qwen/Qwen3.5-27B-Instruct", "org/qwen35-27b-dense"];
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
        assert_eq!(ModelClass::Other.min_free_bytes(), 100 * 1024 * 1024 * 1024);
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
        assert!(msg.contains("50 GB"), "Error must state found bytes: {msg}");
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

        let err = check_disk_preflight(repo, tmp.path(), Some(available)).expect_err("Should fail");

        let msg = err.to_string();
        assert!(
            msg.contains("Qwen3.5 27B dense"),
            "Expected dense label: {msg}"
        );
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
    fn test_resolve_hf_cache_dir_input_precedence() {
        let home = std::path::Path::new("/home/tester");
        assert_eq!(
            resolve_hf_cache_dir_from_inputs(
                Some("/hub-cache"),
                Some("/hf-home"),
                Some("/xdg-cache"),
                Some(home),
            ),
            std::path::PathBuf::from("/hub-cache")
        );
        assert_eq!(
            resolve_hf_cache_dir_from_inputs(
                Some(""),
                Some("/hf-home"),
                Some("/xdg-cache"),
                Some(home),
            ),
            std::path::PathBuf::from("/hf-home/hub")
        );
        assert_eq!(
            resolve_hf_cache_dir_from_inputs(None, None, Some("/xdg-cache"), Some(home)),
            std::path::PathBuf::from("/xdg-cache/huggingface/hub")
        );
    }

    #[test]
    fn test_resolve_hf_cache_dir_fallbacks_are_deterministic() {
        assert_eq!(
            resolve_hf_cache_dir_from_inputs(
                None,
                Some(""),
                Some(""),
                Some(std::path::Path::new("/home/tester")),
            ),
            std::path::PathBuf::from("/home/tester/.cache/huggingface/hub")
        );
        assert_eq!(
            resolve_hf_cache_dir_from_inputs(None, None, None, None),
            std::path::PathBuf::from("/.cache/huggingface/hub")
        );
    }

    #[test]
    fn test_download_error_messages_are_actionable() {
        let err = DownloadError::AuthError {
            repo: "meta-llama/Llama-3.1-8B".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("HF_TOKEN"));
        assert!(msg.contains("huggingface.co"));
        assert!(!msg.contains("huggingface-cli"));
    }

    #[test]
    fn test_download_error_repo_not_found() {
        let err = DownloadError::RepoNotFound {
            repo: "nonexistent/model".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("org/model-name"));
    }

    /// Opt-in live proof that the accepted Qwen3.8 revision resolves through
    /// the exact production endpoint without transferring model payloads.
    #[test]
    fn live_qwen38_repository_info_matches_the_accepted_commit() {
        if std::env::var("HF2Q_NETWORK_TESTS").ok().as_deref() != Some("1") {
            eprintln!("skipping network test (set HF2Q_NETWORK_TESTS=1 to run)");
            return;
        }
        use hf_hub::api::sync::ApiBuilder;
        use hf_hub::{Repo, RepoType};

        let api = ApiBuilder::new()
            .with_endpoint(CANONICAL_HF_ENDPOINT.to_owned())
            .with_progress(false)
            .build()
            .expect("build exact-origin Hub client");
        let info = api
            .repo(Repo::with_revision(
                QWEN38_REPOSITORY_ID.to_owned(),
                RepoType::Model,
                QWEN38_ACCEPTED_REVISION.to_owned(),
            ))
            .info()
            .expect("fetch Qwen3.8 repository info");
        let reference = HfModelReference::parse(QWEN38_REPOSITORY_ID, None).unwrap();
        let resolved = resolve_repository_info(reference, QWEN38_ACCEPTED_REVISION, &info).unwrap();
        assert_eq!(resolved.reference().revision(), QWEN38_ACCEPTED_REVISION);
        assert!(resolved.contains("config.json"));
        assert!(resolved.contains("model.safetensors.index.json"));
        let exact_repo = api.repo(Repo::with_revision(
            QWEN38_REPOSITORY_ID.to_owned(),
            RepoType::Model,
            QWEN38_ACCEPTED_REVISION.to_owned(),
        ));
        let config =
            fetch_expected_file_metadata(&api, &exact_repo, resolved.reference(), "config.json")
                .expect("authenticate bounded Qwen3.8 config metadata");
        assert_eq!(config.filename, "config.json");
        assert!(config.bytes > 0 && config.bytes <= MAX_HF_SMALL_METADATA_BYTES);
    }
}
