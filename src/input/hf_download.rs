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
//! Cache: Uses the standard hf-hub cache directory. Subsequent runs with
//! the same exact revision reuse verified blobs. `hf-hub` 1.x transparently
//! selects native Xet range reconstruction for Xet-backed Hub objects and
//! concurrent file transfers for bounded snapshots.
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
//! Completed blobs are not re-downloaded. Transfers write to `.incomplete`
//! paths and publish cache blobs atomically only after the native Xet or HTTP
//! operation succeeds. After interruption, completed blobs are reused and
//! incomplete work is reconstructed safely by the selected transport.
//!
//! Manual test protocol: `Ctrl+C` mid-download → observe partial shard
//! in `~/.cache/huggingface/hub/models--*/snapshots/*/` → re-invoke
//! `hf2q` → verify only in-flight shard re-downloads, total wall-clock
//! is proportionally shorter.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use thiserror::Error;
use tracing::{debug, info};

use crate::core::integrity::{verify_shard, IntegrityError, ShardIntegrity};
use crate::input::hf_reference::{HfModelReference, HfReferenceError, ResolvedHfModelReference};
use crate::progress::{HubDownloadObserver, HubDownloadSnapshot, ProgressReporter};

type HubRepo = hf_hub::HFRepositorySync<hf_hub::RepoTypeModel>;

struct HubApi {
    client: hf_hub::HFClientSync,
    metadata: reqwest::blocking::Client,
}

struct HubFileMetadata {
    commit_hash: String,
    etag: String,
    file_size: u64,
    xet_hash: Option<String>,
}

mod gguf_probe;
mod resolution;

use resolution::resolve_repository_info;
#[cfg(test)]
pub(in crate::input) use resolution::resolve_repository_info_for_test;
pub use resolution::ResolvedModelRepository;

const CANONICAL_HF_ENDPOINT: &str = "https://huggingface.co";
const DEFAULT_HF_REVISION: &str = "main";
const QWEN38_REPOSITORY_ID: &str = "Qwen/Qwen3.8-27B";
const QWEN38_ACCEPTED_REVISION: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
const HF_SNAPSHOT_MAX_WORKERS: usize = 8;
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

    #[error("hosted GGUF is incompatible with the production runtime: {reason}")]
    IncompatibleHostedGguf { reason: String },

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

/// Metadata-only view of one GGUF-like entry in a resolved Hub repository.
/// This is a deliberately narrow projection of the private repository
/// inventory: callers cannot select arbitrary source files through it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct HubGgufArtifact {
    pub repository: String,
    pub revision: String,
    pub filename: String,
    pub bytes: u64,
    pub sha256: String,
    /// Filename-derived hint only. The GGUF header remains authoritative.
    pub quant_hint: Option<String>,
    pub role: String,
    pub selectable: bool,
    pub unavailable_reason: Option<String>,
}

impl HubGgufArtifact {
    /// Immutable model-pool/request identity. The strong object identity is
    /// included so a mutable branch or same-named replacement cannot alias an
    /// already resident engine.
    pub fn request_model(&self) -> String {
        format!(
            "hf://{}@{}/{}#{}",
            self.repository, self.revision, self.filename, self.sha256
        )
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct HubGgufCatalog {
    pub schema_version: String,
    pub repository: String,
    pub revision: String,
    #[serde(default)]
    pub requires_projector: bool,
    /// Exact sum of the immutable safetensors shards selected by the native
    /// downloader. This is metadata-only admission evidence; no weight payload
    /// is transferred while building the catalog.
    #[serde(default)]
    pub source_weight_bytes: Option<u64>,
    #[serde(default)]
    pub source_uncached_weight_bytes: Option<u64>,
    pub artifacts: Vec<HubGgufArtifact>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeSourcePlan {
    pub repository: String,
    pub revision: String,
    pub total_weight_bytes: u64,
    pub uncached_weight_bytes: u64,
    pub total_metadata_bytes: u64,
    pub uncached_metadata_bytes: u64,
    pub requires_projector: bool,
    /// Fail-closed pre-payload upper bound for every GGUF byte emitted from
    /// this source inventory, including kept F32 roles and metadata.
    pub output_upper_bound_bytes: u64,
    metadata_records: Vec<ShardIntegrity>,
    weight_records: Vec<ShardIntegrity>,
}

impl NativeSourcePlan {
    pub fn uncached_source_bytes(&self) -> Result<u64, DownloadError> {
        self.uncached_weight_bytes
            .checked_add(self.uncached_metadata_bytes)
            .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
                reason: "uncached native source byte total overflowed u64".into(),
            })
    }
}

pub(crate) struct PreparedNativePlanningSource {
    _workspace: tempfile::TempDir,
    model_dir: PathBuf,
    source_plan: NativeSourcePlan,
    source_bundle_sha256: String,
}

impl PreparedNativePlanningSource {
    pub(crate) fn path(&self) -> &Path {
        &self.model_dir
    }

    pub(crate) fn source_plan(&self) -> &NativeSourcePlan {
        &self.source_plan
    }

    pub(crate) fn source_bundle_sha256(&self) -> &str {
        &self.source_bundle_sha256
    }
}

/// Build a private sparse, metadata-only view of one exact native source.
/// Every byte present in the sparse weights came from an identity-bound Hub
/// range response; tensor payload pages are holes and are never requested.
pub(crate) fn prepare_native_planning_source(
    reference: HfModelReference,
) -> Result<PreparedNativePlanningSource, DownloadError> {
    let source_plan = resolve_native_source_plan(reference)?;
    let cache_dir = resolve_hf_cache_dir();
    let workspace = tempfile::Builder::new()
        .prefix("hf2q-native-plan-")
        .tempdir_in(&cache_dir)
        .map_err(download_failed)?;
    let model_dir = workspace.path().join("source");
    fs::create_dir(&model_dir).map_err(download_failed)?;

    let repo_root = cache_dir.join(hub_model_cache_folder(&source_plan.repository));
    let blob_root = repo_root
        .join("blobs")
        .canonicalize()
        .map_err(download_failed)?;
    for record in &source_plan.metadata_records {
        let snapshot = cached_hub_file_path(
            &cache_dir,
            &source_plan.repository,
            &source_plan.revision,
            &record.filename,
            record.bytes,
        )
        .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "staged native metadata `{}` disappeared before planning",
                record.filename
            ),
        })?;
        let blob = snapshot.canonicalize().map_err(download_failed)?;
        if !blob.starts_with(&blob_root) {
            return Err(DownloadError::InvalidRepositoryInventory {
                reason: format!(
                    "staged native metadata `{}` resolved outside its exact Hub blob namespace",
                    record.filename
                ),
            });
        }
        let destination = model_dir.join(&record.filename);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(download_failed)?;
        }
        fs::hard_link(&blob, &destination).map_err(download_failed)?;
        verify_shard(
            &source_plan.repository,
            &source_plan.revision,
            &destination,
            record,
        )?;
    }

    for record in &source_plan.weight_records {
        let sha256 =
            record
                .sha256
                .as_deref()
                .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
                    reason: format!(
                        "native weight `{}` has no strong LFS SHA-256 identity",
                        record.filename
                    ),
                })?;
        let identity = AuthenticatedHubFile {
            repository: &source_plan.repository,
            revision: &source_plan.revision,
            filename: &record.filename,
            bytes: record.bytes,
            sha256,
            kind: "safetensors",
        };
        let (prefix_length, header, header_end) = fetch_authenticated_safetensors_header(
            record.bytes,
            &record.filename,
            |start, end| fetch_authenticated_file_range(&identity, start, end),
        )?;
        let destination = model_dir.join(&record.filename);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(download_failed)?;
        }
        let mut sparse = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&destination)
            .map_err(download_failed)?;
        sparse.write_all(&prefix_length).map_err(download_failed)?;
        sparse.write_all(&header).map_err(download_failed)?;
        sparse.set_len(record.bytes).map_err(download_failed)?;
        sparse.sync_all().map_err(download_failed)?;
        ensure_sparse_planning_file(&destination, header_end, record.bytes)?;
    }

    let records = source_plan
        .metadata_records
        .iter()
        .chain(source_plan.weight_records.iter())
        .map(crate::core::provenance::SourceShard::from_integrity)
        .collect::<Vec<_>>();
    let source_bundle_sha256 = crate::core::provenance::compute_source_bundle_sha256(&records)
        .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
            reason: "native planning source has no strong bundle identity".into(),
        })?;
    Ok(PreparedNativePlanningSource {
        _workspace: workspace,
        model_dir,
        source_plan,
        source_bundle_sha256,
    })
}

fn authenticated_safetensors_header_end(
    length_prefix: &[u8],
    logical_bytes: u64,
    filename: &str,
) -> Result<u64, DownloadError> {
    let header_bytes = u64::from_le_bytes(length_prefix.try_into().map_err(|_| {
        DownloadError::InvalidRepositoryInventory {
            reason: format!("native weight `{filename}` has no safetensors length"),
        }
    })?);
    const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100_000_000;
    let header_end = 8_u64.checked_add(header_bytes).ok_or_else(|| {
        DownloadError::InvalidRepositoryInventory {
            reason: format!("native weight `{filename}` header length overflowed"),
        }
    })?;
    if header_bytes == 0
        || header_bytes > MAX_SAFETENSORS_HEADER_BYTES
        || header_end >= logical_bytes
    {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "native weight `{filename}` has invalid safetensors header length {header_bytes}"
            ),
        });
    }
    Ok(header_end)
}

fn fetch_authenticated_safetensors_header(
    logical_bytes: u64,
    filename: &str,
    mut fetch: impl FnMut(u64, u64) -> Result<Vec<u8>, DownloadError>,
) -> Result<(Vec<u8>, Vec<u8>, u64), DownloadError> {
    let length_prefix = fetch(0, 7)?;
    let header_end = authenticated_safetensors_header_end(&length_prefix, logical_bytes, filename)?;
    let header = fetch(8, header_end - 1)?;
    if header.len() as u64 != header_end - 8 {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!("native weight `{filename}` returned an incomplete safetensors header"),
        });
    }
    Ok((length_prefix, header, header_end))
}

#[cfg(unix)]
fn ensure_sparse_planning_file(
    path: &Path,
    authenticated_prefix_bytes: u64,
    logical_bytes: u64,
) -> Result<(), DownloadError> {
    use std::os::unix::fs::MetadataExt;

    let metadata = path.metadata().map_err(download_failed)?;
    let allocated = metadata.blocks().saturating_mul(512);
    let allowance = authenticated_prefix_bytes.saturating_add(1024 * 1024);
    if metadata.len() != logical_bytes || allocated > allowance {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "planning workspace does not preserve sparse allocation for {} (logical={logical_bytes}, allocated={allocated}, prefix={authenticated_prefix_bytes})",
                path.display()
            ),
        });
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_sparse_planning_file(
    path: &Path,
    _authenticated_prefix_bytes: u64,
    _logical_bytes: u64,
) -> Result<(), DownloadError> {
    Err(DownloadError::InvalidRepositoryInventory {
        reason: format!(
            "native sparse planning is unavailable on this platform for {}",
            path.display()
        ),
    })
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

    let available = available_bytes_override
        .or_else(|| get_available_space_for_path(target_path))
        .unwrap_or(0);

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
/// to find the matching mount point. `None` means capacity could not be
/// established and exact transfer admission must fail closed.
fn get_available_space_for_path(path: &std::path::Path) -> Option<u64> {
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
    best.map(|(_, space)| space)
}

/// Download a model from HuggingFace Hub.
///
/// Returns the path to the local directory containing the downloaded model files.
/// Uses hf-hub crate cache — subsequent calls with the same repo skip re-download.
/// Completed blobs are not re-fetched on re-invocation. Interrupted Xet work
/// may reuse CAS/chunk state, but only completed blobs become snapshot entries.
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

    // Resolve exact shard sizes and already-valid cache presence before the
    // first weight transfer. The old family-name floor rejected fully cached
    // models and could admit source bytes that left no room for conversion.
    let cache_dir = resolve_hf_cache_dir();
    let plan = resolve_native_source_plan(reference.clone())?;
    let uncached_source_bytes = plan.uncached_source_bytes()?;
    if uncached_source_bytes > 0 {
        check_exact_disk_preflight(
            format!("native source cache for {}", plan.repository),
            &cache_dir,
            uncached_source_bytes,
            None,
        )?;
    }
    download_via_hf_hub(plan, &cache_dir, progress)
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

/// Resolve the hosted GGUF choices for one repository without transferring
/// model payload. Every returned artifact is bound to the same exact commit
/// and a strong Hugging Face LFS identity.
pub fn resolve_hub_gguf_catalog(
    reference: HfModelReference,
) -> Result<HubGgufCatalog, DownloadError> {
    if reference.filename().is_some() {
        return Err(DownloadError::FileReferenceUnsupported);
    }
    let cache_dir = resolve_hf_cache_dir();
    let api = build_hub_api(&cache_dir, false)?;
    let (resolved, inventory) = resolve_with_api(&api, reference)?.into_download_parts();
    let repo = hub_model_repo(&api, resolved.repo_id());
    let requires_projector =
        resolve_repository_projector_requirement_best_effort(&api, &repo, &resolved, &inventory);
    let mut artifacts = Vec::new();
    let gguf_filenames = inventory
        .iter()
        .filter(|filename| filename.to_ascii_lowercase().ends_with(".gguf"))
        .collect::<Vec<_>>();
    if gguf_filenames.len() > 128 {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "repository exposes {} GGUF entries; diagnostic catalog limit is 128",
                gguf_filenames.len()
            ),
        });
    }
    for filename in gguf_filenames {
        let record = fetch_expected_file_metadata(&api, &repo, &resolved, filename)?;
        let sha256 = record
            .sha256
            .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
                reason: format!("GGUF artifact `{filename}` has no strong LFS SHA-256 identity"),
            })?;
        let (role, quant_hint, unavailable_reason) = classify_hub_gguf(filename);
        artifacts.push(HubGgufArtifact {
            repository: resolved.repo_id().to_owned(),
            revision: resolved.revision().to_owned(),
            filename: filename.clone(),
            bytes: record.bytes,
            sha256,
            quant_hint,
            role: role.to_owned(),
            selectable: unavailable_reason.is_none() && role == "text_model",
            unavailable_reason,
        });
    }
    artifacts.sort_by(|left, right| left.filename.cmp(&right.filename));
    Ok(HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".to_owned(),
        repository: resolved.repo_id().to_owned(),
        revision: resolved.revision().to_owned(),
        requires_projector,
        // Source shard HEADs are intentionally deferred until every hosted
        // candidate has been ruled out. Hosted is the fastest-startup path.
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts,
    })
}

pub fn resolve_native_source_plan(
    reference: HfModelReference,
) -> Result<NativeSourcePlan, DownloadError> {
    if reference.filename().is_some() {
        return Err(DownloadError::FileReferenceUnsupported);
    }
    let cache_dir = resolve_hf_cache_dir();
    let api = build_hub_api(&cache_dir, false)?;
    let (resolved, inventory) = resolve_with_api(&api, reference)?.into_download_parts();
    let repo = hub_model_repo(&api, resolved.repo_id());
    // Metadata is a bounded first stage: establish every exact extent and
    // admit the aggregate before the first cache write. Once staged it is
    // intentionally not counted again in the weight/output preflight; the
    // subsequent free-space measurement already reflects those bytes.
    let metadata_records =
        resolve_source_metadata_records(&api, &repo, &resolved, &inventory, &cache_dir)?;
    let (total_metadata_bytes, _uncached_metadata_bytes) = checked_inventory_extents(
        metadata_records
            .iter()
            .map(|(_, record, cached)| (record.bytes, *cached)),
        "native source metadata",
    )?;
    admit_and_stage_native_metadata(
        resolved.repo_id(),
        &cache_dir,
        &metadata_records,
        None,
        |records| stage_native_source_metadata(&repo, &resolved, records),
    )?;
    let weight_records = resolve_source_weight_records(
        &api, &repo, &resolved, &inventory, &cache_dir,
    )?
    .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
        reason: "repository has no supported model.safetensors source inventory".into(),
    })?;
    let (total_weight_bytes, uncached_weight_bytes) = checked_inventory_extents(
        weight_records
            .iter()
            .map(|(_, record, cached)| (record.bytes, *cached)),
        "native source weights",
    )?;
    let config = inspect_repository_config(&api, &repo, &resolved, &inventory)?;
    let requires_projector = config_requires_projector(&config);
    let output_upper_bound_bytes = native_output_upper_bound_bytes(total_weight_bytes, &config);
    Ok(NativeSourcePlan {
        repository: resolved.repo_id().to_owned(),
        revision: resolved.revision().to_owned(),
        total_weight_bytes,
        uncached_weight_bytes,
        total_metadata_bytes,
        uncached_metadata_bytes: 0,
        requires_projector,
        output_upper_bound_bytes,
        metadata_records: metadata_records
            .into_iter()
            .map(|(_, record, _)| record)
            .collect(),
        weight_records: weight_records
            .into_iter()
            .map(|(_, record, _)| record)
            .collect(),
    })
}

fn resolve_source_metadata_records(
    api: &HubApi,
    repo: &HubRepo,
    resolved: &ResolvedHfModelReference,
    inventory: &BTreeSet<String>,
    cache_dir: &Path,
) -> Result<Vec<(String, ShardIntegrity, bool)>, DownloadError> {
    let mut records = Vec::new();
    for filename in initial_download_files(inventory)? {
        let record = fetch_expected_file_metadata(api, repo, resolved, &filename)?;
        let cached = cached_hub_file_path(
            cache_dir,
            resolved.repo_id(),
            resolved.revision(),
            &filename,
            record.bytes,
        )
        .is_some();
        records.push((filename, record, cached));
    }
    Ok(records)
}

fn stage_native_source_metadata(
    repo: &HubRepo,
    resolved: &ResolvedHfModelReference,
    records: &[(String, ShardIntegrity, bool)],
) -> Result<(), DownloadError> {
    for (filename, record, _) in records {
        let path = download_file(
            repo,
            resolved.repo_id(),
            resolved.revision(),
            filename,
            None,
        )?;
        verify_shard(resolved.repo_id(), resolved.revision(), &path, record)?;
    }
    Ok(())
}

fn admit_and_stage_native_metadata(
    repository: &str,
    cache_dir: &Path,
    records: &[(String, ShardIntegrity, bool)],
    available_bytes_override: Option<u64>,
    stage: impl FnOnce(&[(String, ShardIntegrity, bool)]) -> Result<(), DownloadError>,
) -> Result<(), DownloadError> {
    let (_, uncached) = checked_inventory_extents(
        records
            .iter()
            .map(|(_, record, cached)| (record.bytes, *cached)),
        "native source metadata",
    )?;
    if uncached > 0 {
        check_exact_disk_preflight(
            format!("native source metadata for {repository}"),
            cache_dir,
            uncached,
            available_bytes_override,
        )?;
    }
    stage(records)
}

fn checked_inventory_extents(
    files: impl IntoIterator<Item = (u64, bool)>,
    label: &str,
) -> Result<(u64, u64), DownloadError> {
    let mut total = 0_u64;
    let mut uncached = 0_u64;
    for (bytes, cached) in files {
        total =
            total
                .checked_add(bytes)
                .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
                    reason: format!("{label} byte total overflowed u64"),
                })?;
        if !cached {
            uncached = uncached.checked_add(bytes).ok_or_else(|| {
                DownloadError::InvalidRepositoryInventory {
                    reason: format!("uncached {label} byte total overflowed u64"),
                }
            })?;
        }
    }
    Ok((total, uncached))
}

fn resolve_source_weight_records(
    api: &HubApi,
    repo: &HubRepo,
    resolved: &ResolvedHfModelReference,
    inventory: &BTreeSet<String>,
    cache_dir: &Path,
) -> Result<Option<Vec<(String, ShardIntegrity, bool)>>, DownloadError> {
    const INDEX: &str = "model.safetensors.index.json";
    const SINGLE: &str = "model.safetensors";

    let required = if inventory.contains(INDEX) {
        let record = fetch_expected_file_metadata(api, repo, resolved, INDEX)?;
        let path = download_file(repo, resolved.repo_id(), resolved.revision(), INDEX, None)?;
        verify_shard(resolved.repo_id(), resolved.revision(), &path, &record)?;
        let bytes = fs::read(&path)?;
        crate::input::integrity::required_weight_shards_from_bytes(&bytes, &path).map_err(
            |error| DownloadError::InvalidRepositoryInventory {
                reason: error.to_string(),
            },
        )?
    } else if inventory.contains(SINGLE) {
        vec![SINGLE.to_owned()]
    } else {
        return Ok(None);
    };

    let mut records = Vec::new();
    for filename in required {
        if !inventory.contains(&filename) {
            return Err(DownloadError::InvalidRepositoryInventory {
                reason: format!(
                    "safetensors index references `{filename}` outside the exact repository inventory"
                ),
            });
        }
        let record = fetch_expected_file_metadata(api, repo, resolved, &filename)?;
        let cached = cached_hub_file_path(
            cache_dir,
            resolved.repo_id(),
            resolved.revision(),
            &filename,
            record.bytes,
        )
        .is_some();
        records.push((filename, record, cached));
    }
    Ok((!records.is_empty()).then_some(records))
}

fn inspect_repository_config(
    api: &HubApi,
    repo: &HubRepo,
    resolved: &ResolvedHfModelReference,
    inventory: &BTreeSet<String>,
) -> Result<serde_json::Value, DownloadError> {
    const CONFIG: &str = "config.json";
    inspect_repository_config_with(inventory.contains(CONFIG), || {
        let record = fetch_expected_file_metadata(api, repo, resolved, CONFIG)?;
        let path = download_file(repo, resolved.repo_id(), resolved.revision(), CONFIG, None)?;
        verify_shard(resolved.repo_id(), resolved.revision(), &path, &record)?;
        let bytes = fs::read(&path)?;
        serde_json::from_slice(&bytes).map_err(|error| DownloadError::InvalidRepositoryInventory {
            reason: format!("exact config.json is invalid JSON: {error}"),
        })
    })
}

fn inspect_repository_config_with(
    config_present: bool,
    inspect: impl FnOnce() -> Result<serde_json::Value, DownloadError>,
) -> Result<serde_json::Value, DownloadError> {
    if !config_present {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: "native source repository has no exact config.json metadata".into(),
        });
    }
    inspect()
}

fn resolve_repository_projector_requirement_best_effort(
    api: &HubApi,
    repo: &HubRepo,
    resolved: &ResolvedHfModelReference,
    inventory: &BTreeSet<String>,
) -> bool {
    if !inventory.contains("config.json") {
        return false;
    }
    match inspect_repository_config(api, repo, resolved, inventory) {
        Ok(config) => config_requires_projector(&config),
        Err(error) => {
            tracing::warn!(
                repository = %resolved.repo_id(),
                revision = %resolved.revision(),
                error = %error,
                "could not inspect bounded exact-revision multimodal repository metadata"
            );
            false
        }
    }
}

fn config_requires_projector(config: &serde_json::Value) -> bool {
    config
        .get("vision_config")
        .is_some_and(serde_json::Value::is_object)
}

pub(crate) fn native_output_upper_bound_bytes(
    source_weight_bytes: u64,
    config: &serde_json::Value,
) -> u64 {
    // The converter's largest wire type is F32. Ordinary sources use at
    // least two bytes per element; FP8 uses one; MXFP4 packs two elements per
    // byte. Unknown quantized source encodings take the MXFP4 worst case.
    let quantization = config.get("quantization_config");
    let method = quantization
        .and_then(|value| {
            value
                .get("quant_method")
                .or_else(|| value.get("quant_type"))
        })
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let expansion = if quantization.is_none() {
        2_u64
    } else if method.contains("mxfp4") || method.contains("fp4") {
        8
    } else if method.contains("fp8") || method.contains("float8") {
        4
    } else {
        8
    };
    // Tokenizer/metadata inputs are independently capped (512 MiB tokenizer,
    // 16 MiB per small metadata file). Two GiB covers their GGUF encoding,
    // tensor table, synthesized tensors, and alignment without relying on a
    // quant-ratio estimate.
    const NON_WEIGHT_ALLOWANCE: u64 = 2 * 1024 * 1024 * 1024;
    source_weight_bytes
        .saturating_mul(expansion)
        .saturating_add(NON_WEIGHT_ALLOWANCE)
}

/// Authenticate and parse only the selected GGUF's bounded header prefix
/// before transferring its tensor payload. The Hugging Face redirect is
/// bound to the exact catalog commit, LFS digest, and total bytes; bearer
/// credentials are never forwarded to the signed cross-origin CDN URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubGgufHeaderCompatibility {
    pub architecture: String,
    pub requires_projector: bool,
}

pub fn validate_hub_gguf_header_compatibility(
    artifact: &HubGgufArtifact,
) -> Result<HubGgufHeaderCompatibility, DownloadError> {
    const INITIAL_PREFIX_BYTES: u64 = 1024 * 1024;
    const MAX_PREFIX_BYTES: u64 = 64 * 1024 * 1024;
    let mut prefix = Vec::new();
    let mut fetched = 0_u64;
    let mut target = INITIAL_PREFIX_BYTES.min(artifact.bytes);
    let mut last_parse_error = None;

    while fetched < artifact.bytes && fetched < MAX_PREFIX_BYTES {
        let end = target.min(artifact.bytes).saturating_sub(1);
        let bytes = fetch_authenticated_hub_range(artifact, fetched, end)?;
        prefix.extend_from_slice(&bytes);
        fetched = end.saturating_add(1);

        match gguf_probe::parse_bounded_header(&prefix, artifact.bytes) {
            Ok(header) => {
                validate_probed_gguf_header(&header, artifact)?;
                validate_qwen_hosted_prefix(&prefix, &header)?;
                return Ok(HubGgufHeaderCompatibility {
                    architecture: header.architecture,
                    requires_projector: header.requires_projector,
                });
            }
            Err(error) => last_parse_error = Some(error),
        }
        if fetched >= MAX_PREFIX_BYTES || fetched >= artifact.bytes {
            break;
        }
        target = target
            .saturating_mul(2)
            .min(MAX_PREFIX_BYTES)
            .min(artifact.bytes);
    }
    Err(DownloadError::IncompatibleHostedGguf {
        reason: format!(
            "selected GGUF `{}` header did not parse within the {} MiB authenticated preflight cap: {}",
            artifact.filename,
            MAX_PREFIX_BYTES / (1024 * 1024),
            last_parse_error.unwrap_or_else(|| "no header bytes returned".into())
        ),
    })
}

/// Apply the hosted GGUF admission contract to exact-size bytes the operator
/// already owns. This is a bounded structural/runtime check, not an immutable
/// payload proof: callers that publish or copy the bytes must separately bind
/// the complete SHA-256. No Hub range request is performed here.
pub fn validate_local_hub_gguf_compatibility(
    path: &Path,
    artifact: &HubGgufArtifact,
) -> Result<HubGgufHeaderCompatibility, DownloadError> {
    let retained = crate::core::bounded_file::StableRegularFile::open_exact(path, artifact.bytes)
        .map_err(download_failed)?
        .ok_or_else(|| DownloadError::IncompatibleHostedGguf {
            reason: "owned GGUF changed or is not an exact-size regular file".to_owned(),
        })?;
    validate_retained_local_hub_gguf_compatibility(&retained, artifact)
}

pub(crate) fn validate_retained_local_hub_gguf_compatibility(
    retained: &crate::core::bounded_file::StableRegularFile,
    artifact: &HubGgufArtifact,
) -> Result<HubGgufHeaderCompatibility, DownloadError> {
    use std::io::Read;

    const MAX_PREFIX_BYTES: u64 = 64 * 1024 * 1024;
    let mut prefix = Vec::new();
    retained
        .try_clone()
        .map_err(download_failed)?
        .take(MAX_PREFIX_BYTES.min(artifact.bytes))
        .read_to_end(&mut prefix)
        .map_err(download_failed)?;
    let header = gguf_probe::parse_bounded_header(&prefix, artifact.bytes).map_err(|reason| {
        DownloadError::IncompatibleHostedGguf {
            reason: format!(
                "owned GGUF `{}` header did not parse within the {} MiB local preflight cap: {reason}",
                artifact.filename,
                MAX_PREFIX_BYTES / (1024 * 1024)
            ),
        }
    })?;
    validate_probed_gguf_header(&header, artifact)?;
    let gguf =
        mlx_native::gguf::GgufFile::from_file(retained.try_clone().map_err(download_failed)?)
            .map_err(|error| DownloadError::IncompatibleHostedGguf {
                reason: format!("owned GGUF does not parse with the runtime reader: {error}"),
            })?;
    validate_qwen_runtime_admission(&gguf)
        .map_err(|reason| DownloadError::IncompatibleHostedGguf { reason })?;
    if !retained.is_stable().map_err(download_failed)? {
        return Err(DownloadError::IncompatibleHostedGguf {
            reason: "owned GGUF changed during local admission".to_owned(),
        });
    }
    Ok(HubGgufHeaderCompatibility {
        architecture: header.architecture,
        requires_projector: header.requires_projector,
    })
}

fn fetch_authenticated_hub_range(
    artifact: &HubGgufArtifact,
    start: u64,
    end: u64,
) -> Result<Vec<u8>, DownloadError> {
    fetch_authenticated_file_range(
        &AuthenticatedHubFile {
            repository: &artifact.repository,
            revision: &artifact.revision,
            filename: &artifact.filename,
            bytes: artifact.bytes,
            sha256: &artifact.sha256,
            kind: "GGUF",
        },
        start,
        end,
    )
}

struct AuthenticatedHubFile<'a> {
    repository: &'a str,
    revision: &'a str,
    filename: &'a str,
    bytes: u64,
    sha256: &'a str,
    kind: &'static str,
}

fn fetch_authenticated_file_range(
    artifact: &AuthenticatedHubFile<'_>,
    start: u64,
    end: u64,
) -> Result<Vec<u8>, DownloadError> {
    use reqwest::header::{AUTHORIZATION, CONTENT_RANGE, LOCATION, RANGE};
    use std::io::Read;
    use std::time::Duration;

    let mut url = reqwest::Url::parse(CANONICAL_HF_ENDPOINT).map_err(download_failed)?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| DownloadError::DownloadFailed {
                reason: "canonical Hugging Face endpoint cannot accept path segments".into(),
            })?;
        segments.pop_if_empty();
        segments.extend(artifact.repository.split('/'));
        segments.push("resolve");
        segments.push(&artifact.revision);
        segments.extend(artifact.filename.split('/'));
    }
    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(60))
        .user_agent(concat!("hf2q/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(download_failed)?;
    let range = format!("bytes={start}-{end}");
    let mut request = client.get(url).header(RANGE, &range);
    if let Some(token) = resolve_auth_token() {
        request = request.header(AUTHORIZATION, format!("Bearer {token}"));
    }
    let response = request.send().map_err(download_failed)?;
    let status = response.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(DownloadError::AuthError {
            repo: artifact.repository.to_owned(),
        });
    }
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(DownloadError::RepoNotFound {
            repo: artifact.repository.to_owned(),
        });
    }
    validate_authenticated_range_identity(response.headers(), artifact)?;

    let response = if status.is_redirection() {
        let location = response
            .headers()
            .get(LOCATION)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
                reason: format!(
                    "exact {} range response omitted its signed redirect",
                    artifact.kind
                ),
            })?;
        let signed = reqwest::Url::parse(location).map_err(|error| {
            DownloadError::InvalidRepositoryInventory {
                reason: format!("exact {} range redirect is invalid: {error}", artifact.kind),
            }
        })?;
        let host = signed.host_str().unwrap_or_default();
        let trusted_hf_delivery_host = host == "huggingface.co"
            || host.ends_with(".huggingface.co")
            || host == "hf.co"
            || host.ends_with(".hf.co");
        if signed.scheme() != "https"
            || !trusted_hf_delivery_host
            || !signed.username().is_empty()
            || signed.password().is_some()
            || signed.fragment().is_some()
        {
            return Err(DownloadError::InvalidRepositoryInventory {
                reason: format!(
                    "exact {} range redirect is not a credential-free trusted Hugging Face HTTPS URL",
                    artifact.kind
                ),
            });
        }
        // Deliberately build a fresh request without Authorization. The
        // signed CDN URL is already capability-bearing and cross-origin.
        client
            .get(signed)
            .header(RANGE, &range)
            .send()
            .map_err(download_failed)?
    } else {
        response
    };
    if response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "exact {} range returned HTTP {}, expected 206 Partial Content",
                artifact.kind,
                response.status()
            ),
        });
    }
    let expected_len = end
        .checked_sub(start)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
            reason: format!("{} range length overflow", artifact.kind),
        })?;
    let expected_content_range = format!("bytes {start}-{end}/{}", artifact.bytes);
    if response
        .headers()
        .get(CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        != Some(expected_content_range.as_str())
    {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "exact {} range response did not bind `{expected_content_range}`",
                artifact.kind
            ),
        });
    }
    let mut bytes = Vec::with_capacity(expected_len as usize);
    response
        .take(expected_len.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(download_failed)?;
    if bytes.len() as u64 != expected_len {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "exact {} range returned {} bytes, expected {expected_len}",
                artifact.kind,
                bytes.len()
            ),
        });
    }
    Ok(bytes)
}

#[cfg(test)]
fn validate_hub_range_identity(
    headers: &reqwest::header::HeaderMap,
    artifact: &HubGgufArtifact,
) -> Result<(), DownloadError> {
    validate_authenticated_range_identity(
        headers,
        &AuthenticatedHubFile {
            repository: &artifact.repository,
            revision: &artifact.revision,
            filename: &artifact.filename,
            bytes: artifact.bytes,
            sha256: &artifact.sha256,
            kind: "GGUF",
        },
    )
}

fn validate_authenticated_range_identity(
    headers: &reqwest::header::HeaderMap,
    artifact: &AuthenticatedHubFile<'_>,
) -> Result<(), DownloadError> {
    let text = |name: &'static str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
                reason: format!("exact {} range response omitted `{name}`", artifact.kind),
            })
    };
    let commit = text("x-repo-commit")?;
    let bytes = text("x-linked-size")?.parse::<u64>().map_err(|_| {
        DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "exact {} range response has invalid x-linked-size",
                artifact.kind
            ),
        }
    })?;
    let etag = text("x-linked-etag")?
        .trim()
        .trim_start_matches("W/")
        .trim_matches('"');
    if !commit.eq_ignore_ascii_case(&artifact.revision)
        || bytes != artifact.bytes
        || !etag.eq_ignore_ascii_case(&artifact.sha256)
    {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "exact {} range identity changed (commit={commit}, bytes={bytes}, sha256={etag})",
                artifact.kind
            ),
        });
    }
    Ok(())
}

fn validate_probed_gguf_header(
    header: &gguf_probe::ProbedGgufHeader,
    artifact: &HubGgufArtifact,
) -> Result<(), DownloadError> {
    let arch = header.architecture.as_str();
    if !matches!(arch, "qwen35" | "qwen35moe") {
        return Err(DownloadError::IncompatibleHostedGguf {
            reason: format!(
                "selected GGUF `{}` architecture {arch:?} has no complete hosted tensor-layout admission contract in this build",
                artifact.filename
            ),
        });
    }
    let file_type = hosted_quant_from_file_type(header.file_type).ok_or_else(|| {
        DownloadError::IncompatibleHostedGguf {
            reason: format!(
                "selected GGUF `{}` has an unsupported general.file_type",
                artifact.filename
            ),
        }
    })?;
    if artifact.quant_hint.as_deref() != Some(file_type) {
        return Err(DownloadError::IncompatibleHostedGguf {
            reason: format!(
                "selected GGUF `{}` header quant {file_type} disagrees with catalog quant {:?}",
                artifact.filename, artifact.quant_hint
            ),
        });
    }
    if header.tensor_count == 0 || header.tensor_data_offset >= artifact.bytes {
        return Err(DownloadError::IncompatibleHostedGguf {
            reason: format!(
                "selected GGUF `{}` has no bounded tensor payload",
                artifact.filename
            ),
        });
    }
    if header.token_embedding_type.is_none() || !header.has_output_norm || !header.has_block_tensor
    {
        return Err(DownloadError::IncompatibleHostedGguf {
            reason: format!(
                "selected GGUF {} is missing a required token embedding, output norm, or block-0 tensor sentinel",
                artifact.filename
            ),
        });
    }
    if let Some(reason) = header.incompatible_tensor.as_deref() {
        return Err(DownloadError::IncompatibleHostedGguf {
            reason: format!(
                "selected GGUF `{}` has a runtime-incompatible tensor layout: {reason}",
                artifact.filename
            ),
        });
    }
    if matches!(arch, "qwen35" | "qwen35moe") {
        if let Some(ggml_type) = header.token_embedding_type {
            // Qwen3.5/Qwen3.8 performs a direct native embedding gather.
            // mlx-native 0.11.2 has no Q3_K (or Q4_0/F16) gather route even
            // though its matrix kernels support those types. Reject that
            // per-tensor layout before downloading the payload; native hf2q
            // conversion promotes its embedding to a supported type.
            if !matches!(ggml_type, 0 | 8 | 10 | 12 | 13 | 14) {
                return Err(DownloadError::IncompatibleHostedGguf {
                    reason: format!(
                        "selected GGUF `{}` uses unsupported Qwen native token embedding GGML type {ggml_type}",
                        artifact.filename
                    ),
                });
            }
        }
    }
    Ok(())
}

fn validate_qwen_hosted_prefix(
    prefix: &[u8],
    header: &gguf_probe::ProbedGgufHeader,
) -> Result<(), DownloadError> {
    use std::io::Write;

    let key_prefix = match header.architecture.as_str() {
        "qwen35" => "qwen35",
        "qwen35moe" => "qwen35moe",
        _ => return Ok(()),
    };
    // Qwen35Config constructs one entry per declared block. Bound that scalar
    // before invoking the shared runtime parser so hostile metadata cannot
    // turn a bounded network prefix into an unbounded allocation.
    let mut file = tempfile::tempfile().map_err(download_failed)?;
    file.write_all(prefix).map_err(download_failed)?;
    let gguf = mlx_native::gguf::GgufFile::from_file(file).map_err(|error| {
        DownloadError::IncompatibleHostedGguf {
            reason: format!("bounded GGUF prefix does not parse with the runtime reader: {error}"),
        }
    })?;
    let block_count = gguf
        .metadata_u32(&format!("{key_prefix}.block_count"))
        .ok_or_else(|| DownloadError::IncompatibleHostedGguf {
            reason: format!("GGUF is missing required {key_prefix}.block_count metadata"),
        })?;
    if block_count == 0 || block_count > 4096 || u64::from(block_count) > header.tensor_count {
        return Err(DownloadError::IncompatibleHostedGguf {
            reason: format!(
                "GGUF declares invalid {key_prefix}.block_count={block_count} for {} tensors",
                header.tensor_count
            ),
        });
    }
    validate_qwen_runtime_admission(&gguf)
        .map_err(|reason| DownloadError::IncompatibleHostedGguf { reason })
}

pub(crate) fn validate_qwen_runtime_admission(
    gguf: &mlx_native::gguf::GgufFile,
) -> Result<(), String> {
    let key_prefix = match gguf.metadata_string("general.architecture").unwrap_or("") {
        "qwen35" => "qwen35",
        "qwen35moe" => "qwen35moe",
        _ => return Ok(()),
    };
    let block_count = gguf
        .metadata_u32(&format!("{key_prefix}.block_count"))
        .ok_or_else(|| format!("GGUF is missing required {key_prefix}.block_count metadata"))?;
    let tensor_count = gguf.tensor_names().len() as u64;
    if block_count == 0 || block_count > 4096 || u64::from(block_count) > tensor_count {
        return Err(format!(
            "GGUF declares invalid {key_prefix}.block_count={block_count} for {tensor_count} tensors"
        ));
    }
    let cfg = crate::inference::models::qwen35::Qwen35Config::from_gguf(gguf)
        .map_err(|error| format!("Qwen runtime metadata admission failed: {error}"))?;
    crate::inference::models::qwen35::tokenizer::build_tokenizer_from_gguf(gguf)
        .map_err(|error| format!("Qwen tokenizer metadata admission failed: {error}"))?;
    validate_qwen_operational_config(&cfg)
        .map_err(|reason| format!("Qwen operational config admission failed: {reason}"))?;
    validate_qwen_hosted_topology(gguf, &cfg)
        .map_err(|reason| format!("Qwen tensor topology admission failed: {reason}"))?;
    crate::inference::models::qwen35::mtp_weights_load::validate_mtp_tensor_topology(gguf, &cfg)
        .map_err(|error| format!("Qwen MTP topology admission failed: {error}"))
}

fn validate_qwen_operational_config(
    cfg: &crate::inference::models::qwen35::Qwen35Config,
) -> Result<(), String> {
    if cfg.hidden_size == 0
        || cfg.num_hidden_layers == 0
        || cfg.num_attention_heads == 0
        || cfg.num_key_value_heads == 0
        || cfg.head_dim == 0
    {
        return Err("hidden, layer, Q-head, KV-head, and head dimensions must be nonzero".into());
    }
    if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
        return Err(format!(
            "Q head count {} is not divisible by KV head count {}",
            cfg.num_attention_heads, cfg.num_key_value_heads
        ));
    }
    if cfg.linear_num_key_heads == 0
        || cfg.linear_num_value_heads == 0
        || cfg.linear_num_value_heads % cfg.linear_num_key_heads != 0
        || cfg.linear_key_head_dim == 0
        || cfg.linear_value_head_dim == 0
        || cfg.linear_conv_kernel_dim == 0
    {
        return Err("linear-attention head counts/dimensions/kernel are not executable".into());
    }
    let mrope_sum = cfg.mrope_section.iter().try_fold(0_u32, |sum, value| {
        sum.checked_add(*value)
            .ok_or_else(|| "mRoPE section sum overflow".to_owned())
    })?;
    if cfg.rotary_dim == 0
        || cfg.rotary_dim > cfg.head_dim
        || cfg.rotary_dim % 2 != 0
        || mrope_sum != cfg.rotary_dim / 2
    {
        return Err(format!(
            "rotary/mRoPE dimensions are incoherent: rotary_dim={}, head_dim={}, sections={:?}",
            cfg.rotary_dim, cfg.head_dim, cfg.mrope_section
        ));
    }
    if !cfg.rope_theta.is_finite()
        || cfg.rope_theta <= 0.0
        || !cfg.rms_norm_eps.is_finite()
        || cfg.rms_norm_eps <= 0.0
        || cfg.max_position_embeddings == 0
        || cfg.vocab_size == 0
    {
        return Err(
            "rope, norm, context, and vocabulary scalars must be finite and positive".into(),
        );
    }
    match cfg.variant {
        crate::inference::models::qwen35::Qwen35Variant::Dense => {
            if cfg.intermediate_size.is_none_or(|value| value == 0) {
                return Err("dense feed-forward length must be nonzero".into());
            }
        }
        crate::inference::models::qwen35::Qwen35Variant::Moe => {
            let moe = cfg
                .moe
                .as_ref()
                .ok_or_else(|| "MoE configuration is absent".to_owned())?;
            if moe.num_experts == 0
                || moe.num_experts_per_tok == 0
                || moe.num_experts_per_tok > moe.num_experts
                || moe.moe_intermediate_size == 0
                || moe.shared_expert_intermediate_size == 0
            {
                return Err(format!(
                    "MoE expert routing is not executable: experts={}, top_k={}, expert_ffn={}, shared_ffn={}",
                    moe.num_experts,
                    moe.num_experts_per_tok,
                    moe.moe_intermediate_size,
                    moe.shared_expert_intermediate_size
                ));
            }
        }
    }
    Ok(())
}

fn validate_qwen_hosted_topology(
    gguf: &mlx_native::gguf::GgufFile,
    cfg: &crate::inference::models::qwen35::Qwen35Config,
) -> Result<(), String> {
    use crate::inference::models::qwen35::{Qwen35LayerKind, Qwen35Variant};

    fn checked_product(values: &[u32], label: &str) -> Result<usize, String> {
        values.iter().try_fold(1_usize, |product, value| {
            product
                .checked_mul(*value as usize)
                .ok_or_else(|| format!("{label} dimension product overflow"))
        })
    }

    fn require_shape(
        gguf: &mlx_native::gguf::GgufFile,
        name: &str,
        expected: &[usize],
    ) -> Result<(), String> {
        let info = gguf
            .tensor_info(name)
            .ok_or_else(|| format!("missing required tensor `{name}`"))?;
        if info.shape != expected {
            return Err(format!(
                "tensor `{name}` shape {:?} != expected {expected:?}",
                info.shape
            ));
        }
        Ok(())
    }

    let h = cfg.hidden_size as usize;
    if h == 0 || cfg.num_hidden_layers == 0 {
        return Err("hidden size and normal block count must be nonzero".into());
    }
    let token_count = match gguf.metadata("tokenizer.ggml.tokens") {
        Some(mlx_native::gguf::MetadataValue::Array(tokens)) if !tokens.is_empty() => tokens.len(),
        _ => return Err("missing nonempty tokenizer.ggml.tokens array".into()),
    };
    let token = gguf
        .tensor_info("token_embd.weight")
        .ok_or_else(|| "missing required tensor `token_embd.weight`".to_owned())?;
    if token.shape.len() != 2 || token.shape[1] != h || token.shape[0] < token_count {
        return Err(format!(
            "token_embd.weight shape {:?} cannot cover {token_count} tokenizer rows at hidden size {h}",
            token.shape
        ));
    }
    require_shape(gguf, "output_norm.weight", &[h])?;
    let output_rows = gguf
        .tensor_info("output.weight")
        .map(|output| output.shape.first().copied())
        .flatten()
        .unwrap_or(token.shape[0]);
    if let Some(output) = gguf.tensor_info("output.weight") {
        if output.shape.len() != 2 || output.shape[1] != h {
            return Err(format!(
                "output.weight shape {:?} is not [vocab,{h}]",
                output.shape
            ));
        }
    }
    if token.shape[0] < output_rows {
        return Err(format!(
            "token_embd.weight rows {} cannot cover resolved output-head rows {output_rows}",
            token.shape[0]
        ));
    }

    let q_rows = checked_product(&[cfg.num_attention_heads, cfg.head_dim], "full-attention Q")?;
    let kv_rows = checked_product(
        &[cfg.num_key_value_heads, cfg.head_dim],
        "full-attention KV",
    )?;
    let nk_d = checked_product(
        &[cfg.linear_num_key_heads, cfg.linear_key_head_dim],
        "linear-attention K",
    )?;
    let nv_d = checked_product(
        &[cfg.linear_num_value_heads, cfg.linear_value_head_dim],
        "linear-attention V",
    )?;
    let qkv_rows = nk_d
        .checked_mul(2)
        .and_then(|value| value.checked_add(nv_d))
        .ok_or_else(|| "linear-attention QKV dimension overflow".to_owned())?;
    let q_projection_rows = q_rows
        .checked_mul(2)
        .ok_or_else(|| "full-attention gated Q projection dimension overflow".to_owned())?;

    if cfg.layer_types.len() != cfg.num_hidden_layers as usize {
        return Err("runtime layer-kind topology length differs from block count".into());
    }
    for (layer, kind) in cfg.layer_types.iter().copied().enumerate() {
        let p = format!("blk.{layer}");
        require_shape(gguf, &format!("{p}.attn_norm.weight"), &[h])?;
        require_shape(gguf, &format!("{p}.post_attention_norm.weight"), &[h])?;
        match kind {
            Qwen35LayerKind::FullAttention => {
                require_shape(gguf, &format!("{p}.attn_q.weight"), &[q_projection_rows, h])?;
                require_shape(gguf, &format!("{p}.attn_k.weight"), &[kv_rows, h])?;
                require_shape(gguf, &format!("{p}.attn_v.weight"), &[kv_rows, h])?;
                require_shape(
                    gguf,
                    &format!("{p}.attn_q_norm.weight"),
                    &[cfg.head_dim as usize],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.attn_k_norm.weight"),
                    &[cfg.head_dim as usize],
                )?;
                require_shape(gguf, &format!("{p}.attn_output.weight"), &[h, q_rows])?;
            }
            Qwen35LayerKind::LinearAttention => {
                require_shape(gguf, &format!("{p}.attn_qkv.weight"), &[qkv_rows, h])?;
                require_shape(gguf, &format!("{p}.attn_gate.weight"), &[nv_d, h])?;
                require_shape(
                    gguf,
                    &format!("{p}.ssm_conv1d.weight"),
                    &[qkv_rows, cfg.linear_conv_kernel_dim as usize],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ssm_alpha.weight"),
                    &[cfg.linear_num_value_heads as usize, h],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ssm_beta.weight"),
                    &[cfg.linear_num_value_heads as usize, h],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ssm_dt.bias"),
                    &[cfg.linear_num_value_heads as usize],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ssm_a"),
                    &[cfg.linear_num_value_heads as usize],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ssm_norm.weight"),
                    &[cfg.linear_value_head_dim as usize],
                )?;
                require_shape(gguf, &format!("{p}.ssm_out.weight"), &[h, nv_d])?;
            }
        }

        match cfg.variant {
            Qwen35Variant::Dense => {
                let intermediate = cfg
                    .intermediate_size
                    .ok_or_else(|| "dense Qwen config has no intermediate size".to_owned())?
                    as usize;
                require_shape(gguf, &format!("{p}.ffn_gate.weight"), &[intermediate, h])?;
                require_shape(gguf, &format!("{p}.ffn_up.weight"), &[intermediate, h])?;
                require_shape(gguf, &format!("{p}.ffn_down.weight"), &[h, intermediate])?;
                let tensor_type = |role: &str| {
                    gguf.tensor_info(&format!("{p}.ffn_{role}.weight"))
                        .map(|info| info.ggml_type)
                        .ok_or_else(|| format!("{p}.ffn_{role}.weight is missing"))
                };
                crate::inference::models::qwen35::weight_loader::validate_qwen35_dense_ffn_storage(
                    layer as u32,
                    tensor_type("gate")?,
                    tensor_type("up")?,
                    tensor_type("down")?,
                )
                .map_err(|error| error.to_string())?;
            }
            Qwen35Variant::Moe => {
                let moe = cfg
                    .moe
                    .as_ref()
                    .ok_or_else(|| "MoE Qwen config has no expert topology".to_owned())?;
                let experts = moe.num_experts as usize;
                let expert_intermediate = moe.moe_intermediate_size as usize;
                let shared_intermediate = moe.shared_expert_intermediate_size as usize;
                require_shape(gguf, &format!("{p}.ffn_gate_inp.weight"), &[experts, h])?;
                require_shape(
                    gguf,
                    &format!("{p}.ffn_gate_exps.weight"),
                    &[experts, expert_intermediate, h],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ffn_up_exps.weight"),
                    &[experts, expert_intermediate, h],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ffn_down_exps.weight"),
                    &[experts, h, expert_intermediate],
                )?;
                require_shape(gguf, &format!("{p}.ffn_gate_inp_shexp.weight"), &[1, h])?;
                require_shape(
                    gguf,
                    &format!("{p}.ffn_gate_shexp.weight"),
                    &[shared_intermediate, h],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ffn_up_shexp.weight"),
                    &[shared_intermediate, h],
                )?;
                require_shape(
                    gguf,
                    &format!("{p}.ffn_down_shexp.weight"),
                    &[h, shared_intermediate],
                )?;
                let gate = gguf
                    .tensor_info(&format!("{p}.ffn_gate_exps.weight"))
                    .expect("shape validated");
                let up = gguf
                    .tensor_info(&format!("{p}.ffn_up_exps.weight"))
                    .expect("shape validated");
                if gate.ggml_type != up.ggml_type {
                    return Err(format!(
                        "{p} expert gate/up GGML types differ ({:?} vs {:?})",
                        gate.ggml_type, up.ggml_type
                    ));
                }
            }
        }
    }
    Ok(())
}

fn hosted_quant_from_file_type(file_type: u32) -> Option<&'static str> {
    use crate::quantize::ggml_quants::GgufFtype;
    match file_type {
        value if value == GgufFtype::MostlyQ2_K as u32 => Some("Q2_K"),
        value if value == GgufFtype::MostlyQ3_K_M as u32 => Some("Q3_K_M"),
        value if value == GgufFtype::MostlyQ4_K_M as u32 => Some("Q4_K_M"),
        value if value == GgufFtype::MostlyQ5_K_M as u32 => Some("Q5_K_M"),
        value if value == GgufFtype::MostlyQ6_K as u32 => Some("Q6_K"),
        value if value == GgufFtype::MostlyQ8_0 as u32 => Some("Q8_0"),
        _ => None,
    }
}

fn download_failed(error: impl std::fmt::Display) -> DownloadError {
    DownloadError::DownloadFailed {
        reason: error.to_string(),
    }
}

/// Download and authenticate exactly one artifact previously returned by
/// [`resolve_hub_gguf_catalog`]. No source weights or sibling GGUFs are read.
pub fn download_hub_gguf(artifact: &HubGgufArtifact) -> Result<PathBuf, DownloadError> {
    if !artifact.selectable || artifact.role != "text_model" {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!("GGUF artifact `{}` is not selectable", artifact.filename),
        });
    }
    let transfer_progress = ProgressReporter::new();
    download_hub_artifact(
        artifact,
        "text_model",
        Some(transfer_progress.hub_download("Downloading hosted GGUF")),
    )
    .map(|download| download.snapshot_path)
}

/// Foreground-safe progress from one hosted Hub payload transfer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HubTransferProgress {
    pub completed_bytes: u64,
    pub total_bytes: u64,
    pub bytes_per_second: Option<u64>,
    pub elapsed_ms: u64,
    pub complete: bool,
}

/// Download a hosted text artifact while keeping terminal ownership with the
/// caller. The native Xet handler writes only output-agnostic atomics; this
/// synchronous foreground loop coalesces those updates before invoking the
/// caller's potentially stateful startup renderer or IPC publisher.
pub(crate) fn download_hub_gguf_with_progress(
    artifact: &HubGgufArtifact,
    progress: &mut dyn FnMut(HubTransferProgress),
) -> Result<DownloadedHubArtifact, DownloadError> {
    if !artifact.selectable || artifact.role != "text_model" {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!("GGUF artifact `{}` is not selectable", artifact.filename),
        });
    }
    download_hub_artifact_observed(artifact, "text_model", progress)
}

pub(crate) fn download_hub_companion_with_progress(
    artifact: &HubGgufArtifact,
    progress: &mut dyn FnMut(HubTransferProgress),
) -> Result<DownloadedHubArtifact, DownloadError> {
    if artifact.role != "companion" || artifact.selectable {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "GGUF artifact `{}` is not a projector companion",
                artifact.filename
            ),
        });
    }
    download_hub_artifact_observed(artifact, "companion", progress)
}

pub(crate) struct DownloadedHubArtifact {
    pub(crate) snapshot_path: PathBuf,
    pub(crate) blob_path: PathBuf,
    pub(crate) retained: crate::core::bounded_file::StableRegularFile,
}

fn download_hub_artifact(
    artifact: &HubGgufArtifact,
    expected_role: &str,
    progress: Option<hf_hub::progress::Progress>,
) -> Result<DownloadedHubArtifact, DownloadError> {
    validate_repo_filename(&artifact.filename)?;
    if !hosted_gguf_identity_valid_for_role(artifact, expected_role) {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: "hosted GGUF identity is incomplete or malformed".to_owned(),
        });
    }
    let cache_dir = resolve_hf_cache_dir();
    let api = build_hub_api(&cache_dir, true)?;
    let repo = hub_model_repo(&api, &artifact.repository);
    let resolved = HfModelReference::parse(&artifact.repository, Some(&artifact.revision))?
        .resolve(&artifact.revision)?;
    let record = fetch_expected_file_metadata(&api, &repo, &resolved, &artifact.filename)?;
    if record.bytes != artifact.bytes || record.sha256.as_deref() != Some(artifact.sha256.as_str())
    {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "GGUF artifact `{}` changed after catalog resolution",
                artifact.filename
            ),
        });
    }
    if let Some(path) = cached_hub_artifact_path_in(&cache_dir, artifact) {
        return authenticate_hub_artifact(&cache_dir, artifact, &path, &record);
    }
    check_artifact_disk_preflight(&artifact.repository, &cache_dir, artifact.bytes)?;
    let path = download_file(
        &repo,
        &artifact.repository,
        &artifact.revision,
        &artifact.filename,
        progress,
    )?;
    authenticate_hub_artifact(&cache_dir, artifact, &path, &record)
}

fn authenticate_hub_artifact(
    cache_dir: &Path,
    artifact: &HubGgufArtifact,
    snapshot_path: &Path,
    record: &ShardIntegrity,
) -> Result<DownloadedHubArtifact, DownloadError> {
    let mut retained = crate::core::bounded_file::StableRegularFile::open_operator_path_exact(
        snapshot_path,
        record.bytes,
    )
    .map_err(download_failed)?
    .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
        reason: format!(
            "downloaded hosted artifact `{}` is not a stable exact-size cache file",
            artifact.filename
        ),
    })?;
    let blob_path = retained
        .canonical_path_for_identity()
        .map_err(download_failed)?
        .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "downloaded hosted artifact `{}` has no stable cache blob identity",
                artifact.filename
            ),
        })?;
    let blob_root = cache_dir
        .join(hub_model_cache_folder(&artifact.repository))
        .join("blobs")
        .canonicalize()
        .map_err(download_failed)?;
    if blob_path.parent() != Some(blob_root.as_path())
        || blob_path.file_name().and_then(|name| name.to_str()) != Some(artifact.sha256.as_str())
    {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "downloaded hosted artifact `{}` resolved outside its authenticated repository blob store",
                artifact.filename
            ),
        });
    }
    let actual_sha256 = retained.sha256().map_err(download_failed)?.ok_or_else(|| {
        DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "downloaded hosted artifact `{}` changed during digest verification",
                artifact.filename
            ),
        }
    })?;
    if !actual_sha256.eq_ignore_ascii_case(&artifact.sha256) {
        return Err(DownloadError::Integrity(IntegrityError::ShardMismatch {
            repo: artifact.repository.clone(),
            revision: artifact.revision.clone(),
            filename: artifact.filename.clone(),
            expected: artifact.sha256.clone(),
            actual: actual_sha256,
            local_path: snapshot_path.display().to_string(),
        }));
    }
    if !retained.is_stable().map_err(download_failed)? {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "downloaded hosted artifact `{}` changed after digest verification",
                artifact.filename
            ),
        });
    }
    Ok(DownloadedHubArtifact {
        snapshot_path: snapshot_path.to_path_buf(),
        blob_path,
        retained,
    })
}

fn download_hub_artifact_observed(
    artifact: &HubGgufArtifact,
    expected_role: &str,
    progress: &mut dyn FnMut(HubTransferProgress),
) -> Result<DownloadedHubArtifact, DownloadError> {
    let observer = HubDownloadObserver::new();
    let handler = observer.progress();
    let started = Instant::now();
    std::thread::scope(|scope| {
        let transfer =
            scope.spawn(|| download_hub_artifact(artifact, expected_role, Some(handler)));
        let mut last_sequence = 0;
        loop {
            publish_hub_transfer_progress(&observer, started, &mut last_sequence, progress);
            if transfer.is_finished() {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let result = transfer.join().map_err(|_| DownloadError::DownloadFailed {
            reason: "native Hugging Face download worker panicked".to_owned(),
        })?;
        publish_hub_transfer_progress(&observer, started, &mut last_sequence, progress);
        result
    })
}

fn publish_hub_transfer_progress(
    observer: &HubDownloadObserver,
    started: Instant,
    last_sequence: &mut u64,
    progress: &mut dyn FnMut(HubTransferProgress),
) {
    let snapshot = observer.snapshot();
    if snapshot.sequence == 0 || snapshot.sequence == *last_sequence || snapshot.total_bytes == 0 {
        return;
    }
    *last_sequence = snapshot.sequence;
    progress(hub_transfer_progress(snapshot, started.elapsed()));
}

fn hub_transfer_progress(snapshot: HubDownloadSnapshot, elapsed: Duration) -> HubTransferProgress {
    HubTransferProgress {
        completed_bytes: snapshot.completed_bytes.min(snapshot.total_bytes),
        total_bytes: snapshot.total_bytes,
        bytes_per_second: snapshot.bytes_per_second,
        elapsed_ms: elapsed.as_millis().min(u128::from(u64::MAX)) as u64,
        complete: snapshot.complete,
    }
}

/// Return exact-revision Hub-cache bytes for a catalog-bound GGUF without
/// performing metadata or payload I/O. Size is checked here. Serve/chat may
/// admit the retained file structurally in place; publication and copying
/// still require callers to bind the complete digest.
pub(crate) fn cached_hub_gguf_path(artifact: &HubGgufArtifact) -> Option<PathBuf> {
    cached_hub_artifact_path_in(&resolve_hf_cache_dir(), artifact)
}

/// Retain a managed final-leaf link only when it still targets the active
/// standard Hub cache's exact repository blob and exact-revision snapshot.
/// This is a cheap repeat-start check: the blob name is the authenticated LFS
/// SHA-256 already recorded in the regular managed sidecar, so no model-sized
/// rehash is needed after first publication.
pub(crate) fn retain_managed_hub_cache_link(
    path: &Path,
    repository: &str,
    revision: &str,
    hub_filename: &str,
    bytes: u64,
    sha256: &str,
) -> Result<Option<crate::core::bounded_file::StableRegularFile>, DownloadError> {
    retain_managed_hub_cache_link_in(
        &resolve_hf_cache_dir(),
        path,
        repository,
        revision,
        hub_filename,
        bytes,
        sha256,
    )
}

pub(crate) fn retain_managed_hub_cache_link_in(
    cache_dir: &Path,
    path: &Path,
    repository: &str,
    revision: &str,
    hub_filename: &str,
    bytes: u64,
    sha256: &str,
) -> Result<Option<crate::core::bounded_file::StableRegularFile>, DownloadError> {
    validate_repo_filename(hub_filename)?;
    if revision.len() != 40
        || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        || sha256.len() != 64
        || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: "managed Hub cache link has malformed immutable identity".into(),
        });
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {}
        Ok(_) => return Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    }
    let Some(retained) =
        crate::core::bounded_file::StableRegularFile::open_operator_path_exact(path, bytes)
            .map_err(download_failed)?
    else {
        return Ok(None);
    };
    let Some(blob_path) = retained
        .canonical_path_for_identity()
        .map_err(download_failed)?
    else {
        return Ok(None);
    };
    let blob_root = match cache_dir
        .join(hub_model_cache_folder(repository))
        .join("blobs")
        .canonicalize()
    {
        Ok(path) => path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if blob_path.parent() != Some(blob_root.as_path())
        || blob_path.file_name().and_then(|name| name.to_str()) != Some(sha256)
    {
        return Ok(None);
    }
    let Some(snapshot_path) =
        cached_hub_file_path(cache_dir, repository, revision, hub_filename, bytes)
    else {
        return Ok(None);
    };
    let Some(snapshot) = crate::core::bounded_file::StableRegularFile::open_operator_path_exact(
        &snapshot_path,
        bytes,
    )
    .map_err(download_failed)?
    else {
        return Ok(None);
    };
    Ok((retained.identity() == snapshot.identity()
        && retained.is_stable().map_err(download_failed)?
        && snapshot.is_stable().map_err(download_failed)?)
    .then_some(retained))
}

/// Recognize only the dangling link spelling hf2q itself publishes for one
/// immutable repository blob. This permits a cleared Hub cache to be repaired
/// by redownload without treating an arbitrary or retargeted symlink as a
/// cache miss.
pub(crate) fn managed_hub_cache_link_is_expected_dangling(
    path: &Path,
    repository: &str,
    sha256: &str,
) -> Result<bool, DownloadError> {
    managed_hub_cache_link_is_expected_dangling_in(
        &resolve_hf_cache_dir(),
        path,
        repository,
        sha256,
    )
}

pub(crate) fn managed_hub_cache_link_is_expected_dangling_in(
    cache_dir: &Path,
    path: &Path,
    repository: &str,
    sha256: &str,
) -> Result<bool, DownloadError> {
    if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Ok(false);
    }
    HfModelReference::parse(repository, None)?;
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {}
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    }
    match fs::metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Ok(_) => return Ok(false),
        Err(error) => return Err(error.into()),
    }
    let target = fs::read_link(path)?;
    let cache_authority = match cache_dir.canonicalize() {
        Ok(path) => path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && cache_dir.is_absolute() => {
            cache_dir.to_path_buf()
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::env::current_dir()?.join(cache_dir)
        }
        Err(error) => return Err(error.into()),
    };
    let expected = cache_authority
        .join(hub_model_cache_folder(repository))
        .join("blobs")
        .join(sha256);
    Ok(target == expected)
}

#[cfg(test)]
pub(crate) fn cached_hub_gguf_path_in(
    cache_dir: &Path,
    artifact: &HubGgufArtifact,
) -> Option<PathBuf> {
    cached_hub_artifact_path_in(cache_dir, artifact)
}

fn cached_hub_artifact_path_in(cache_dir: &Path, artifact: &HubGgufArtifact) -> Option<PathBuf> {
    cached_hub_file_path(
        cache_dir,
        &artifact.repository,
        &artifact.revision,
        &artifact.filename,
        artifact.bytes,
    )
}

fn cached_hub_file_path(
    cache_dir: &Path,
    repository: &str,
    revision: &str,
    filename: &str,
    bytes: u64,
) -> Option<PathBuf> {
    if validate_repo_filename(filename).is_err()
        || HfModelReference::parse(repository, Some(revision)).is_err()
    {
        return None;
    }
    let direct_snapshot = (revision.len() == 40
        && revision.bytes().all(|byte| byte.is_ascii_hexdigit()))
    .then(|| {
        cache_dir
            .join(hub_model_cache_folder(repository))
            .join("snapshots")
            .join(revision)
            .join(filename)
    })
    .filter(|path| path.exists());
    let path = direct_snapshot?;
    let metadata = path.metadata().ok()?;
    (metadata.is_file() && metadata.len() == bytes).then_some(path)
}

fn hosted_gguf_identity_valid_for_role(artifact: &HubGgufArtifact, expected_role: &str) -> bool {
    let (inferred_role, inferred_quant, unavailable_reason) = classify_hub_gguf(&artifact.filename);
    artifact.filename.to_ascii_lowercase().ends_with(".gguf")
        && artifact.bytes > 0
        && artifact.revision.len() == 40
        && artifact
            .revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        && artifact.sha256.len() == 64
        && artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        && inferred_role == expected_role
        && inferred_quant == artifact.quant_hint
        && artifact.role == inferred_role
        && match expected_role {
            "text_model" => {
                artifact.selectable
                    && artifact.unavailable_reason.is_none()
                    && unavailable_reason.is_none()
            }
            "companion" => {
                !artifact.selectable
                    && artifact.quant_hint.is_none()
                    && artifact.unavailable_reason.is_some()
            }
            _ => false,
        }
}

/// Exact-byte disk preflight for materializing a hosted artifact outside the
/// hf-hub cache (for example into the canonical XDG data directory).
pub fn check_hub_artifact_destination(
    repo_id: &str,
    destination: &Path,
    artifact_bytes: u64,
) -> Result<(), DownloadError> {
    check_artifact_disk_preflight(repo_id, destination, artifact_bytes)
}

/// Preflight an exact native-conversion output plan on the destination
/// filesystem before creating or streaming the temporary GGUF.
pub fn check_conversion_output_preflight(
    repo_id: &str,
    destination: &Path,
    planned_output_bytes: u64,
) -> Result<(), DownloadError> {
    check_exact_disk_preflight(
        format!("native hf2q conversion for {repo_id}"),
        destination,
        planned_output_bytes,
        None,
    )
}

/// Preflight the complete native fallback before source weight transfer. The
/// source cache and managed output may share a filesystem; in that case their
/// remaining extents are admitted as one aggregate instead of independently.
pub fn check_native_source_conversion_plan(
    plan: &NativeSourcePlan,
    destination: &Path,
    planned_output_bytes: u64,
) -> Result<(), DownloadError> {
    let cache_dir = resolve_hf_cache_dir();
    let shared = same_filesystem(&cache_dir, destination);
    let (cache_extent, destination_extent) = native_source_conversion_extents(
        plan.uncached_source_bytes()?,
        planned_output_bytes,
        shared,
    )?;
    if shared {
        check_exact_disk_preflight(
            format!("native source plus hf2q conversion for {}", plan.repository),
            destination,
            destination_extent,
            None,
        )
    } else {
        if cache_extent > 0 {
            check_exact_disk_preflight(
                format!("native source cache for {}", plan.repository),
                &cache_dir,
                cache_extent,
                None,
            )?;
        }
        check_exact_disk_preflight(
            format!("native hf2q conversion for {}", plan.repository),
            destination,
            destination_extent,
            None,
        )
    }
}

fn native_source_conversion_extents(
    uncached_source_bytes: u64,
    planned_output_bytes: u64,
    shared_filesystem: bool,
) -> Result<(u64, u64), DownloadError> {
    if shared_filesystem {
        let aggregate = uncached_source_bytes
            .checked_add(planned_output_bytes)
            .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
                reason: "native source plus output disk plan overflowed u64".into(),
            })?;
        Ok((0, aggregate))
    } else {
        Ok((uncached_source_bytes, planned_output_bytes))
    }
}

/// Preflight the complete hosted-artifact plan before payload transfer.
/// The Hub cache is the sole model-sized allocation. Managed publication is a
/// tiny symlink even when its destination is on another filesystem.
pub fn check_hub_artifact_plan(
    artifact: &HubGgufArtifact,
    _destination: &Path,
) -> Result<(), DownloadError> {
    let cache_dir = resolve_hf_cache_dir();
    if cached_hub_artifact_path_in(&cache_dir, artifact).is_none() {
        check_artifact_disk_preflight(&artifact.repository, &cache_dir, artifact.bytes)?;
    }
    Ok(())
}

/// Admit a text GGUF and its selected projector as one disk transaction before
/// either payload is transferred. Only uncached Hub bytes are model-sized;
/// managed destinations receive tiny authenticated symlinks.
pub fn check_hub_artifact_pair_plan(
    text: &HubGgufArtifact,
    text_destination: &Path,
    projector: Option<(&HubGgufArtifact, &Path)>,
) -> Result<(), DownloadError> {
    check_hub_artifact_pair_plan_from_state(
        text,
        text_destination,
        false,
        projector.map(|(artifact, destination)| (artifact, destination, false)),
    )
}

pub fn check_hub_artifact_pair_plan_from_state(
    text: &HubGgufArtifact,
    text_destination: &Path,
    text_destination_exact: bool,
    projector: Option<(&HubGgufArtifact, &Path, bool)>,
) -> Result<(), DownloadError> {
    let Some((projector, _projector_destination, projector_destination_exact)) = projector else {
        return if text_destination_exact {
            Ok(())
        } else {
            check_hub_artifact_plan(text, text_destination)
        };
    };
    let cache_dir = resolve_hf_cache_dir();
    let text_uncached = (!text_destination_exact
        && cached_hub_artifact_path_in(&cache_dir, text).is_none())
    .then_some(text.bytes)
    .unwrap_or(0);
    let projector_uncached = (!projector_destination_exact
        && cached_hub_artifact_path_in(&cache_dir, projector).is_none())
    .then_some(projector.bytes)
    .unwrap_or(0);
    let cache_required = text_uncached
        .checked_add(projector_uncached)
        .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
            reason: "hosted text plus projector cache plan overflowed u64".into(),
        })?;
    if cache_required > 0 {
        check_artifact_disk_preflight(&text.repository, &cache_dir, cache_required)?;
    }
    Ok(())
}

/// Preflight materializing an already-local text artifact and optional
/// projector into their explicit destinations as one copy plan. Same-device
/// clones consume no model-sized extent; cross-device copies are charged
/// together when they land on one filesystem.
pub fn check_local_artifact_pair_plan(
    repository: &str,
    text_source: &Path,
    text_destination: &Path,
    text_bytes: u64,
    text_destination_exact: bool,
    projector: Option<(&Path, &Path, u64, bool)>,
) -> Result<(), DownloadError> {
    check_local_artifact_pair_plan_with_devices(
        repository,
        text_source,
        source_device_id(text_source),
        text_destination,
        text_bytes,
        text_destination_exact,
        projector.map(|(source, destination, bytes, exact)| {
            (source, source_device_id(source), destination, bytes, exact)
        }),
    )
}

pub fn check_local_artifact_pair_plan_with_devices(
    repository: &str,
    text_source: &Path,
    text_source_device: Option<u64>,
    text_destination: &Path,
    text_bytes: u64,
    text_destination_exact: bool,
    projector: Option<(&Path, Option<u64>, &Path, u64, bool)>,
) -> Result<(), DownloadError> {
    let text_copy = (!text_destination_exact
        && !same_filesystem_authority(text_source, text_source_device, text_destination))
    .then_some(text_bytes)
    .unwrap_or(0);
    let Some((
        projector_source,
        projector_source_device,
        projector_destination,
        projector_bytes,
        projector_destination_exact,
    )) = projector
    else {
        if text_copy > 0 {
            check_exact_disk_preflight(
                format!("local text artifact for {repository}"),
                text_destination,
                text_copy,
                None,
            )?;
        }
        return Ok(());
    };
    let projector_copy = (!projector_destination_exact
        && !same_filesystem_authority(
            projector_source,
            projector_source_device,
            projector_destination,
        ))
    .then_some(projector_bytes)
    .unwrap_or(0);
    if same_filesystem(text_destination, projector_destination) {
        let required = text_copy.checked_add(projector_copy).ok_or_else(|| {
            DownloadError::InvalidRepositoryInventory {
                reason: "local text plus projector destination plan overflowed u64".into(),
            }
        })?;
        if required > 0 {
            check_exact_disk_preflight(
                format!("local text/projector pair for {repository}"),
                text_destination,
                required,
                None,
            )?;
        }
    } else {
        if text_copy > 0 {
            check_exact_disk_preflight(
                format!("local text artifact for {repository}"),
                text_destination,
                text_copy,
                None,
            )?;
        }
        if projector_copy > 0 {
            check_exact_disk_preflight(
                format!("local projector artifact for {repository}"),
                projector_destination,
                projector_copy,
                None,
            )?;
        }
    }
    Ok(())
}

/// Preflight an already-retained local adoption plan against the exact
/// destination directory descriptors that publication will use.
pub fn check_local_artifact_pair_plan_with_authorities(
    repository: &str,
    text_source_device: Option<u64>,
    text_destination: &Path,
    text_destination_device: Option<u64>,
    text_destination_available: Option<u64>,
    text_bytes: u64,
    text_destination_exact: bool,
    projector: Option<(Option<u64>, &Path, Option<u64>, Option<u64>, u64, bool)>,
) -> Result<(), DownloadError> {
    let text_copy = (!text_destination_exact
        && text_source_device
            .zip(text_destination_device)
            .is_none_or(|(source, destination)| source != destination))
    .then_some(text_bytes)
    .unwrap_or(0);
    let Some((
        projector_source_device,
        projector_destination,
        projector_destination_device,
        projector_destination_available,
        projector_bytes,
        projector_destination_exact,
    )) = projector
    else {
        if text_copy > 0 {
            check_exact_disk_capacity(
                format!("local text artifact for {repository}"),
                text_destination,
                text_copy,
                text_destination_available,
            )?;
        }
        return Ok(());
    };
    let projector_copy = (!projector_destination_exact
        && projector_source_device
            .zip(projector_destination_device)
            .is_none_or(|(source, destination)| source != destination))
    .then_some(projector_bytes)
    .unwrap_or(0);
    if text_destination_device
        .zip(projector_destination_device)
        .is_some_and(|(text, projector)| text == projector)
    {
        let required = text_copy.checked_add(projector_copy).ok_or_else(|| {
            DownloadError::InvalidRepositoryInventory {
                reason: "local text plus projector authority plan overflowed u64".into(),
            }
        })?;
        if required > 0 {
            check_exact_disk_capacity(
                format!("local text/projector pair for {repository}"),
                text_destination,
                required,
                text_destination_available.or(projector_destination_available),
            )?;
        }
    } else {
        if text_copy > 0 {
            check_exact_disk_capacity(
                format!("local text artifact for {repository}"),
                text_destination,
                text_copy,
                text_destination_available,
            )?;
        }
        if projector_copy > 0 {
            check_exact_disk_capacity(
                format!("local projector artifact for {repository}"),
                projector_destination,
                projector_copy,
                projector_destination_available,
            )?;
        }
    }
    Ok(())
}

/// Preflight adopting an already-local text GGUF and downloading an exact
/// hosted projector as one disk transaction. This mixed source plan is
/// deliberately separate from [`check_hub_artifact_pair_plan`]: the local
/// text bytes must never be charged as a Hub download, while the projector
/// cache extent must not be omitted merely because the text is already local.
pub fn check_local_text_hosted_projector_plan(
    repository: &str,
    text_source: &Path,
    text_destination: &Path,
    text_bytes: u64,
    text_destination_exact: bool,
    projector: &HubGgufArtifact,
    projector_destination: &Path,
    projector_destination_exact: bool,
) -> Result<(), DownloadError> {
    check_local_text_hosted_projector_plan_with_device(
        repository,
        text_source,
        source_device_id(text_source),
        text_destination,
        text_bytes,
        text_destination_exact,
        projector,
        projector_destination,
        projector_destination_exact,
    )
}

pub fn check_local_text_hosted_projector_plan_with_device(
    repository: &str,
    text_source: &Path,
    text_source_device: Option<u64>,
    text_destination: &Path,
    text_bytes: u64,
    text_destination_exact: bool,
    projector: &HubGgufArtifact,
    _projector_destination: &Path,
    projector_destination_exact: bool,
) -> Result<(), DownloadError> {
    let cache_dir = resolve_hf_cache_dir();
    let text_copy = (!text_destination_exact
        && !same_filesystem_authority(text_source, text_source_device, text_destination))
    .then_some(text_bytes)
    .unwrap_or(0);
    let projector_uncached = (!projector_destination_exact
        && cached_hub_artifact_path_in(&cache_dir, projector).is_none())
    .then_some(projector.bytes)
    .unwrap_or(0);
    let cache_and_text_share = same_filesystem(&cache_dir, text_destination);
    let (cache_required, text_required) =
        local_text_hosted_projector_extents(text_copy, projector_uncached, cache_and_text_share)?;
    if cache_required > 0 {
        check_exact_disk_preflight(
            format!("hosted projector cache for {repository}"),
            &cache_dir,
            cache_required,
            None,
        )?;
    }
    if text_required > 0 {
        check_exact_disk_preflight(
            format!("local text plus hosted projector for {repository}"),
            text_destination,
            text_required,
            None,
        )?;
    }
    Ok(())
}

pub fn check_local_text_hosted_projector_plan_with_authorities(
    repository: &str,
    text_source_device: Option<u64>,
    text_destination: &Path,
    text_destination_device: Option<u64>,
    text_destination_available: Option<u64>,
    text_bytes: u64,
    text_destination_exact: bool,
    projector: &HubGgufArtifact,
    _projector_destination: &Path,
    _projector_destination_device: Option<u64>,
    _projector_destination_available: Option<u64>,
    projector_destination_exact: bool,
) -> Result<(), DownloadError> {
    let cache_dir = resolve_hf_cache_dir();
    let cache_device = source_device_id(&cache_dir);
    let cache_available = get_available_space_for_path(&cache_dir);
    let text_copy = (!text_destination_exact
        && text_source_device
            .zip(text_destination_device)
            .is_none_or(|(source, destination)| source != destination))
    .then_some(text_bytes)
    .unwrap_or(0);
    let projector_uncached = (!projector_destination_exact
        && cached_hub_artifact_path_in(&cache_dir, projector).is_none())
    .then_some(projector.bytes)
    .unwrap_or(0);
    let cache_and_text_share = cache_device
        .zip(text_destination_device)
        .is_some_and(|(cache, text)| cache == text);
    let (cache_required, text_required) =
        local_text_hosted_projector_extents(text_copy, projector_uncached, cache_and_text_share)?;
    if cache_required > 0 {
        check_exact_disk_capacity(
            format!("hosted projector cache for {repository}"),
            &cache_dir,
            cache_required,
            cache_available,
        )?;
    }
    if text_required > 0 {
        check_exact_disk_capacity(
            format!("local text plus hosted projector for {repository}"),
            text_destination,
            text_required,
            text_destination_available,
        )?;
    }
    Ok(())
}

/// Preflight downloading an exact hosted text GGUF while adopting an already
/// local projector. The retained projector source consumes no Hub-cache
/// extent, and all real allocations are aggregated by destination device.
pub fn check_hosted_text_local_projector_plan(
    text: &HubGgufArtifact,
    text_destination: &Path,
    text_destination_exact: bool,
    projector_source: &Path,
    projector_destination: &Path,
    projector_bytes: u64,
    projector_destination_exact: bool,
) -> Result<(), DownloadError> {
    check_hosted_text_local_projector_plan_with_device(
        text,
        text_destination,
        text_destination_exact,
        projector_source,
        source_device_id(projector_source),
        projector_destination,
        projector_bytes,
        projector_destination_exact,
    )
}

pub fn check_hosted_text_local_projector_plan_with_device(
    text: &HubGgufArtifact,
    _text_destination: &Path,
    text_destination_exact: bool,
    projector_source: &Path,
    projector_source_device: Option<u64>,
    projector_destination: &Path,
    projector_bytes: u64,
    projector_destination_exact: bool,
) -> Result<(), DownloadError> {
    let cache_dir = resolve_hf_cache_dir();
    let cache_extent = (!text_destination_exact
        && cached_hub_artifact_path_in(&cache_dir, text).is_none())
    .then_some(text.bytes)
    .unwrap_or(0);
    let projector_extent = (!projector_destination_exact
        && !same_filesystem_authority(
            projector_source,
            projector_source_device,
            projector_destination,
        ))
    .then_some(projector_bytes)
    .unwrap_or(0);
    let (cache_required, projector_required) = hosted_text_local_projector_extents(
        cache_extent,
        projector_extent,
        same_filesystem(&cache_dir, projector_destination),
    )?;
    if cache_required > 0 {
        check_exact_disk_preflight(
            format!("hosted text cache for {}", text.repository),
            &cache_dir,
            cache_required,
            None,
        )?;
    }
    if projector_required > 0 {
        check_exact_disk_preflight(
            format!("local projector destination for {}", text.repository),
            projector_destination,
            projector_required,
            None,
        )?;
    }
    Ok(())
}

fn local_text_hosted_projector_extents(
    text_copy: u64,
    projector_uncached: u64,
    cache_and_text_share_filesystem: bool,
) -> Result<(u64, u64), DownloadError> {
    if cache_and_text_share_filesystem {
        Ok((
            projector_uncached.checked_add(text_copy).ok_or_else(|| {
                DownloadError::InvalidRepositoryInventory {
                    reason: "local text plus hosted projector cache plan overflowed u64".into(),
                }
            })?,
            0,
        ))
    } else {
        Ok((projector_uncached, text_copy))
    }
}

fn hosted_text_local_projector_extents(
    text_uncached: u64,
    projector_copy: u64,
    cache_and_projector_share_filesystem: bool,
) -> Result<(u64, u64), DownloadError> {
    if cache_and_projector_share_filesystem {
        Ok((
            text_uncached.checked_add(projector_copy).ok_or_else(|| {
                DownloadError::InvalidRepositoryInventory {
                    reason: "hosted text plus local projector cache plan overflowed u64".into(),
                }
            })?,
            0,
        ))
    } else {
        Ok((text_uncached, projector_copy))
    }
}

#[cfg(test)]
fn hosted_pair_required_extents(
    text_uncached: u64,
    projector_uncached: u64,
) -> Result<u64, DownloadError> {
    text_uncached
        .checked_add(projector_uncached)
        .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
            reason: "hosted pair cache extent overflowed u64".into(),
        })
}

#[cfg(unix)]
fn same_filesystem(left: &Path, right: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;

    existing_ancestor(left)
        .and_then(|left| left.metadata().ok())
        .zip(existing_ancestor(right).and_then(|right| right.metadata().ok()))
        .is_some_and(|(left, right)| left.dev() == right.dev())
}

#[cfg(unix)]
fn source_device_id(path: &Path) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;

    existing_ancestor(path)
        .and_then(|path| path.metadata().ok())
        .map(|metadata| metadata.dev())
}

#[cfg(not(unix))]
fn source_device_id(_path: &Path) -> Option<u64> {
    None
}

fn same_filesystem_authority(
    fallback_source: &Path,
    source_device: Option<u64>,
    destination: &Path,
) -> bool {
    source_device
        .zip(source_device_id(destination))
        .map_or_else(
            || same_filesystem(fallback_source, destination),
            |(source, destination)| source == destination,
        )
}

#[cfg(not(unix))]
fn same_filesystem(_left: &Path, _right: &Path) -> bool {
    false
}

fn existing_ancestor(path: &Path) -> Option<&Path> {
    let mut current = Some(path);
    while let Some(candidate) = current {
        if candidate.exists() {
            return Some(candidate);
        }
        current = candidate.parent();
    }
    None
}

fn classify_hub_gguf(filename: &str) -> (&'static str, Option<String>, Option<String>) {
    let lower = filename.to_ascii_lowercase();
    let basename = lower.rsplit('/').next().unwrap_or(&lower);
    if basename.starts_with("mmproj") || basename.contains("-mmproj") {
        return (
            "companion",
            None,
            Some("vision projector companion; not a text model".to_owned()),
        );
    }
    let stem = basename.strip_suffix(".gguf").unwrap_or(basename);
    if stem.rsplit_once("-of-").is_some_and(|(left, right)| {
        left.rsplit('-')
            .next()
            .is_some_and(|part| part.len() == 5 && part.bytes().all(|byte| byte.is_ascii_digit()))
            && right.len() == 5
            && right.bytes().all(|byte| byte.is_ascii_digit())
    }) {
        return (
            "text_model",
            infer_filename_quant(stem),
            Some("split GGUF sets are not supported by the current loader".to_owned()),
        );
    }
    let quant_hint = infer_filename_quant(stem);
    if quant_hint.as_deref() == Some("BF16") {
        return (
            "text_model",
            quant_hint,
            Some("BF16 GGUF weights are not supported by the current mlx-native loader".to_owned()),
        );
    }
    if quant_hint.is_none() {
        return (
            "text_model",
            None,
            Some("GGUF quant type cannot be established from metadata-only inventory".to_owned()),
        );
    }
    ("text_model", quant_hint, None)
}

fn infer_filename_quant(stem: &str) -> Option<String> {
    ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0", "bf16"]
        .into_iter()
        .find(|quant| stem.ends_with(quant))
        .map(|quant| quant.to_ascii_uppercase())
}

fn check_artifact_disk_preflight(
    repo_id: &str,
    cache_dir: &Path,
    artifact_bytes: u64,
) -> Result<(), DownloadError> {
    check_exact_disk_preflight(
        format!("hosted GGUF from {repo_id}"),
        cache_dir,
        artifact_bytes,
        None,
    )
}

fn check_exact_disk_preflight(
    label: String,
    destination: &Path,
    artifact_bytes: u64,
    available_bytes_override: Option<u64>,
) -> Result<(), DownloadError> {
    if artifact_bytes == 0 {
        return Ok(());
    }
    let available = available_bytes_override.or_else(|| get_available_space_for_path(destination));
    check_exact_disk_capacity(label, destination, artifact_bytes, available)
}

fn check_exact_disk_capacity(
    label: String,
    destination: &Path,
    artifact_bytes: u64,
    available_bytes: Option<u64>,
) -> Result<(), DownloadError> {
    if artifact_bytes == 0 {
        return Ok(());
    }
    let required = artifact_bytes.saturating_add(2 * 1024 * 1024 * 1024);
    let available = available_bytes.unwrap_or(0);
    if available < required {
        return Err(DownloadError::InsufficientDisk {
            label,
            required_gb: required.div_ceil(1024 * 1024 * 1024),
            found_gb: available / (1024 * 1024 * 1024),
            path: destination.display().to_string(),
        });
    }
    Ok(())
}

/// Download model files using the hf-hub crate.
fn download_via_hf_hub(
    plan: NativeSourcePlan,
    cache_dir: &Path,
    progress: &ProgressReporter,
) -> Result<DownloadedModel, DownloadError> {
    let api = build_hub_api(cache_dir, true)?;
    let resolved =
        HfModelReference::parse(&plan.repository, Some(&plan.revision))?.resolve(&plan.revision)?;
    let repo = hub_model_repo(&api, resolved.repo_id());

    debug!(
        metadata_files = plan.metadata_records.len(),
        weight_files = plan.weight_records.len(),
        revision = resolved.revision(),
        max_workers = HF_SNAPSHOT_MAX_WORKERS,
        "Downloading bounded exact-revision model snapshot through native Xet"
    );
    let mut downloaded_path = None;
    let mut manifest = Vec::with_capacity(plan.metadata_records.len() + plan.weight_records.len());
    for record in &plan.metadata_records {
        let local = cached_hub_file_path(
            cache_dir,
            resolved.repo_id(),
            resolved.revision(),
            &record.filename,
            record.bytes,
        )
        .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "staged native metadata `{}` disappeared before payload transfer",
                record.filename
            ),
        })?;
        bind_snapshot_parent(
            &mut downloaded_path,
            &local,
            &record.filename,
            resolved.revision(),
        )?;
        verify_shard(resolved.repo_id(), resolved.revision(), &local, record)?;
        manifest.push(record.clone());
    }
    let model_dir = downloaded_path
        .clone()
        .ok_or_else(|| DownloadError::DownloadFailed {
            reason: "No model metadata files were downloaded".to_owned(),
        })?;
    let weight_filenames = plan
        .weight_records
        .iter()
        .map(|record| record.filename.clone())
        .collect::<Vec<_>>();
    download_selected_snapshot(
        &repo,
        resolved.repo_id(),
        resolved.revision(),
        &weight_filenames,
        progress,
    )?;

    let pb = progress.bar(plan.weight_records.len() as u64, "Verifying model weights");
    for record in &plan.weight_records {
        let local_path = cached_hub_file_path(
            cache_dir,
            resolved.repo_id(),
            resolved.revision(),
            &record.filename,
            record.bytes,
        )
        .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "selected snapshot omitted required weight `{}`",
                record.filename
            ),
        })?;
        bind_snapshot_parent(
            &mut downloaded_path,
            &local_path,
            &record.filename,
            resolved.revision(),
        )?;
        verify_shard(resolved.repo_id(), resolved.revision(), &local_path, record)?;
        manifest.push(record.clone());
        pb.inc(1);
    }

    pb.finish_with_message(format!("Selected {} exact source files", manifest.len()));

    info!(path = %model_dir.display(), "Model downloaded to cache");

    manifest.sort_by(|left, right| left.filename.cmp(&right.filename));
    Ok(DownloadedModel {
        local_dir: model_dir,
        reference: resolved,
        manifest,
    })
}

fn download_selected_snapshot(
    repo: &HubRepo,
    repo_id: &str,
    revision: &str,
    filenames: &[String],
    progress: &ProgressReporter,
) -> Result<PathBuf, DownloadError> {
    if filenames.is_empty() {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: "bounded snapshot selected no model payload files".to_owned(),
        });
    }
    let allow_patterns = exact_hub_allow_patterns(filenames)?;
    info!(
        repo = repo_id,
        revision,
        selected_files = filenames.len(),
        max_workers = HF_SNAPSHOT_MAX_WORKERS,
        "Starting concurrent Hugging Face snapshot download through native Xet"
    );
    let snapshot = repo
        .snapshot_download()
        .revision(revision.to_owned())
        .allow_patterns(allow_patterns)
        .max_workers(HF_SNAPSHOT_MAX_WORKERS)
        .progress(progress.hub_download("Downloading model weights"))
        .send()
        .map_err(|error| map_hub_download_error(error, repo_id, "selected snapshot"))?;
    if snapshot.file_name().and_then(|name| name.to_str()) != Some(revision) {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "selected snapshot resolved to `{}` instead of exact commit `{revision}`",
                snapshot.display()
            ),
        });
    }
    Ok(snapshot)
}

fn exact_hub_allow_patterns(filenames: &[String]) -> Result<Vec<String>, DownloadError> {
    let mut patterns = Vec::with_capacity(filenames.len());
    for filename in filenames {
        validate_repo_filename(filename)?;
        patterns.push(globset::escape(filename));
    }
    Ok(patterns)
}

fn hub_model_repo(api: &HubApi, repo_id: &str) -> HubRepo {
    let (owner, name) = hf_hub::split_id(repo_id);
    api.client.model(owner.to_owned(), name.to_owned())
}

fn hub_model_cache_folder(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

fn map_hub_repository_error(error: hf_hub::HFError, repo_id: &str) -> DownloadError {
    match error {
        hf_hub::HFError::AuthRequired { .. } | hf_hub::HFError::Forbidden { .. } => {
            DownloadError::AuthError {
                repo: repo_id.to_owned(),
            }
        }
        hf_hub::HFError::RepoNotFound { .. } | hf_hub::HFError::RevisionNotFound { .. } => {
            DownloadError::RepoNotFound {
                repo: repo_id.to_owned(),
            }
        }
        error => DownloadError::DownloadFailed {
            reason: format!("Failed to get repository info: {error}"),
        },
    }
}

fn map_hub_download_error(error: hf_hub::HFError, repo_id: &str, operation: &str) -> DownloadError {
    match error {
        hf_hub::HFError::AuthRequired { .. } | hf_hub::HFError::Forbidden { .. } => {
            DownloadError::AuthError {
                repo: repo_id.to_owned(),
            }
        }
        error => DownloadError::DownloadFailed {
            reason: format!("Failed to download {operation}: {error}"),
        },
    }
}

fn build_hub_api(cache_dir: &Path, _progress: bool) -> Result<HubApi, DownloadError> {
    let token = resolve_auth_token();
    debug!(has_token = token.is_some(), "Auth token resolution");

    // Pin the official endpoint even when HF_ENDPOINT is set in the process.
    let mut builder = hf_hub::HFClient::builder()
        .endpoint(CANONICAL_HF_ENDPOINT)
        .cache_dir(cache_dir)
        .user_agent(concat!("hf2q/", env!("CARGO_PKG_VERSION")));
    if let Some(token) = token {
        builder = builder.token(token);
    }
    let client = builder
        .build_sync()
        .map_err(|error| DownloadError::DownloadFailed {
            reason: format!("Failed to initialize Hugging Face API client: {error}"),
        })?;
    let metadata = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(60))
        .user_agent(concat!("hf2q/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(download_failed)?;
    Ok(HubApi { client, metadata })
}

fn resolve_with_api(
    api: &HubApi,
    reference: HfModelReference,
) -> Result<ResolvedModelRepository, DownloadError> {
    if reference.filename().is_some() {
        return Err(DownloadError::FileReferenceUnsupported);
    }
    let requested_revision = reference
        .requested_revision()
        .unwrap_or_else(|| default_revision_for(reference.repo_id()))
        .to_owned();
    let lookup_repo = hub_model_repo(api, reference.repo_id());
    let repo_info = lookup_repo
        .info()
        .revision(requested_revision.clone())
        .send()
        .map_err(|error| map_hub_repository_error(error, reference.repo_id()))?;
    let returned_sha = repo_info
        .sha
        .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
            reason: "repository lookup omitted its resolved commit".to_owned(),
        })?;
    let siblings = repo_info
        .siblings
        .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
            reason: "repository lookup omitted its file inventory".to_owned(),
        })?;
    resolve_repository_info(
        reference,
        &requested_revision,
        &returned_sha,
        siblings.iter().map(|sibling| sibling.rfilename.as_str()),
    )
}

fn fetch_expected_file_metadata(
    api: &HubApi,
    _repo: &HubRepo,
    resolved: &ResolvedHfModelReference,
    filename: &str,
) -> Result<ShardIntegrity, DownloadError> {
    let metadata = fetch_hub_file_metadata(api, resolved, filename)?;
    require_native_xet_payload(filename, metadata.xet_hash.as_deref())?;
    debug!(
        repo = resolved.repo_id(),
        revision = resolved.revision(),
        filename,
        transport = if metadata.xet_hash.is_some() {
            "xet"
        } else {
            "http"
        },
        "Authenticated immutable Hugging Face file metadata"
    );
    validate_file_metadata(
        filename,
        resolved.revision(),
        &metadata.commit_hash,
        &metadata.etag,
        metadata.file_size,
    )
}

fn require_native_xet_payload(filename: &str, xet_hash: Option<&str>) -> Result<(), DownloadError> {
    if !filename.ends_with(".safetensors") && !filename.ends_with(".gguf") {
        return Ok(());
    }
    let valid_xet_hash = xet_hash
        .is_some_and(|hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()));
    if valid_xet_hash {
        Ok(())
    } else {
        Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "model payload `{filename}` is not available through native Hugging Face Xet"
            ),
        })
    }
}

pub(crate) fn fetch_selected_hub_integrity(
    repo_id: &str,
    revision: &str,
    filenames: &[String],
) -> Result<Vec<ShardIntegrity>, DownloadError> {
    let reference = HfModelReference::parse(repo_id, Some(revision))?;
    let api = build_hub_api(&resolve_hf_cache_dir(), false)?;
    let (resolved, _) = resolve_with_api(&api, reference)?.into_download_parts();
    let repo = hub_model_repo(&api, repo_id);
    filenames
        .iter()
        .map(|filename| fetch_expected_file_metadata(&api, &repo, &resolved, filename))
        .collect()
}

fn fetch_hub_file_metadata(
    api: &HubApi,
    resolved: &ResolvedHfModelReference,
    filename: &str,
) -> Result<HubFileMetadata, DownloadError> {
    use reqwest::header::{AUTHORIZATION, CONTENT_LENGTH, ETAG};

    validate_repo_filename(filename)?;
    let mut url = reqwest::Url::parse(CANONICAL_HF_ENDPOINT).map_err(download_failed)?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| DownloadError::DownloadFailed {
                reason: "canonical Hugging Face endpoint cannot accept path segments".into(),
            })?;
        segments.pop_if_empty();
        segments.extend(resolved.repo_id().split('/'));
        segments.push("resolve");
        segments.push(resolved.revision());
        segments.extend(filename.split('/'));
    }
    let mut request = api.metadata.head(url.clone());
    if let Some(token) = resolve_auth_token() {
        request = request.header(AUTHORIZATION, format!("Bearer {token}"));
    }
    let response = request
        .send()
        .map_err(|error| DownloadError::DownloadFailed {
            reason: format!("Failed to fetch immutable metadata for `{filename}`: {error}"),
        })?;
    let status = response.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(DownloadError::AuthError {
            repo: resolved.repo_id().to_owned(),
        });
    }
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(DownloadError::RepoNotFound {
            repo: resolved.repo_id().to_owned(),
        });
    }
    if !status.is_success() && !status.is_redirection() {
        return Err(DownloadError::DownloadFailed {
            reason: format!(
                "Failed to fetch immutable metadata for `{filename}`: HTTP {status} at {url}"
            ),
        });
    }
    let headers = response.headers();
    let text = |name: &'static str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
                reason: format!("metadata for `{filename}` omitted `{name}`"),
            })
    };
    let commit_hash = text("x-repo-commit")?.to_owned();
    let etag = headers
        .get("x-linked-etag")
        .or_else(|| headers.get(ETAG))
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            value
                .trim()
                .trim_start_matches("W/")
                .trim_matches('"')
                .to_owned()
        })
        .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
            reason: format!("metadata for `{filename}` omitted its immutable ETag"),
        })?;
    let file_size = headers
        .get("x-linked-size")
        .or_else(|| headers.get(CONTENT_LENGTH))
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| DownloadError::InvalidRepositoryInventory {
            reason: format!("metadata for `{filename}` omitted its byte length"),
        })?;
    let xet_hash = headers
        .get("x-xet-hash")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    Ok(HubFileMetadata {
        commit_hash,
        etag,
        file_size,
        xet_hash,
    })
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
    if (filename.ends_with(".safetensors") || filename.ends_with(".gguf")) && !record.is_lfs {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!("model payload `{filename}` has no strong LFS SHA-256 identity"),
        });
    }
    Ok(record)
}

pub(super) fn metadata_size_cap(filename: &str) -> Option<u64> {
    if filename.ends_with(".safetensors") || filename.ends_with(".gguf") {
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

#[cfg(test)]
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
    repo: &HubRepo,
    repo_id: &str,
    revision: &str,
    filename: &str,
    progress: Option<hf_hub::progress::Progress>,
) -> Result<PathBuf, DownloadError> {
    let request = repo
        .download_file()
        .filename(filename.to_owned())
        .revision(revision.to_owned());
    let result = match progress {
        Some(progress) => request.progress(progress).send(),
        None => request.send(),
    };
    result.map_err(|error| map_hub_download_error(error, repo_id, &format!("`{filename}`")))
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

/// Canonical Hugging Face cache root used by bounded read-only local
/// inventory. This performs path resolution only and creates nothing.
pub(crate) fn hf_hub_cache_dir() -> PathBuf {
    resolve_hf_cache_dir()
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
pub(crate) mod tests {
    use super::*;

    #[test]
    fn selected_snapshot_patterns_are_literal_and_do_not_broaden_inventory() {
        let selected = vec!["weights/model[01]*?.safetensors".to_owned()];
        let patterns = exact_hub_allow_patterns(&selected).unwrap();
        let matcher = globset::Glob::new(&patterns[0]).unwrap().compile_matcher();
        assert!(matcher.is_match(&selected[0]));
        assert!(!matcher.is_match("weights/model0-extra.safetensors"));
        assert!(!matcher.is_match("weights/model1.safetensors"));
    }

    #[test]
    fn model_cache_folder_matches_the_standard_hub_layout() {
        assert_eq!(
            hub_model_cache_folder("owner/model"),
            "models--owner--model"
        );
    }

    #[test]
    fn model_payloads_require_native_xet_while_git_metadata_does_not() {
        let xet_hash = "a".repeat(64);
        assert!(require_native_xet_payload("model.safetensors", Some(&xet_hash)).is_ok());
        assert!(require_native_xet_payload("model.gguf", None).is_err());
        assert!(require_native_xet_payload("model.gguf", Some("not-a-xet-hash")).is_err());
        assert!(require_native_xet_payload("config.json", None).is_ok());
    }

    fn write_test_gguf(path: &Path, arch: &str, file_type: u32) -> u64 {
        fn string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&3_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        string(&mut bytes, arch);
        string(&mut bytes, "general.file_type");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&file_type.to_le_bytes());
        for (name, inner, ggml_type, offset) in [
            ("token_embd.weight", 256_u64, 12_u32, 0_u64),
            ("output_norm.weight", 1, 0, 144),
            ("blk.0.attn_norm.weight", 1, 0, 148),
        ] {
            string(&mut bytes, name);
            bytes.extend_from_slice(&1_u32.to_le_bytes());
            bytes.extend_from_slice(&inner.to_le_bytes());
            bytes.extend_from_slice(&ggml_type.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        let aligned = bytes.len().div_ceil(32) * 32;
        bytes.resize(aligned + 152, 0);
        std::fs::write(path, &bytes).unwrap();
        bytes.len() as u64
    }

    pub(crate) fn write_complete_qwen_test_gguf(path: &Path) -> u64 {
        use crate::quantize::ggml_quants::GgmlType;

        write_complete_qwen_test_gguf_with_dense_types(
            path,
            GgmlType::Q4_0,
            GgmlType::Q4_0,
            GgmlType::Q4_0,
        )
    }

    fn write_complete_qwen_test_gguf_with_dense_types(
        path: &Path,
        dense_gate: crate::quantize::ggml_quants::GgmlType,
        dense_up: crate::quantize::ggml_quants::GgmlType,
        dense_down: crate::quantize::ggml_quants::GgmlType,
    ) -> u64 {
        write_complete_qwen_test_gguf_with_options(
            path,
            dense_gate,
            dense_up,
            dense_down,
            false,
            crate::quantize::ggml_quants::GgmlType::F32,
            None,
        )
    }

    fn write_complete_qwen_test_gguf_with_options(
        path: &Path,
        dense_gate: crate::quantize::ggml_quants::GgmlType,
        dense_up: crate::quantize::ggml_quants::GgmlType,
        dense_down: crate::quantize::ggml_quants::GgmlType,
        include_mtp: bool,
        mtp_projection_type: crate::quantize::ggml_quants::GgmlType,
        mtp_dedicated_embedding_type: Option<crate::quantize::ggml_quants::GgmlType>,
    ) -> u64 {
        write_complete_qwen_test_gguf_with_vocab_rows(
            path,
            dense_gate,
            dense_up,
            dense_down,
            include_mtp,
            mtp_projection_type,
            mtp_dedicated_embedding_type,
            2,
            None,
            2,
            2,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn write_complete_qwen_test_gguf_with_vocab_rows(
        path: &Path,
        dense_gate: crate::quantize::ggml_quants::GgmlType,
        dense_up: crate::quantize::ggml_quants::GgmlType,
        dense_down: crate::quantize::ggml_quants::GgmlType,
        include_mtp: bool,
        mtp_projection_type: crate::quantize::ggml_quants::GgmlType,
        mtp_dedicated_embedding_type: Option<crate::quantize::ggml_quants::GgmlType>,
        token_rows: usize,
        output_rows: Option<usize>,
        mtp_embedding_rows: usize,
        mtp_head_rows: usize,
    ) -> u64 {
        use crate::backends::gguf::types::MetaValue;
        use crate::backends::gguf::writer::GgufWriter;
        use crate::quantize::ggml_quants::GgmlType;
        use std::io::Cursor;

        let mut metadata = vec![
            ("general.architecture", MetaValue::String("qwen35".into())),
            (
                "general.file_type",
                MetaValue::U32(crate::quantize::ggml_quants::GgufFtype::MostlyQ4_K_M as u32),
            ),
            (
                "qwen35.block_count",
                MetaValue::U32(if include_mtp { 2 } else { 1 }),
            ),
            ("qwen35.embedding_length", MetaValue::U32(32)),
            ("qwen35.attention.head_count", MetaValue::U32(1)),
            ("qwen35.attention.head_count_kv", MetaValue::U32(1)),
            ("qwen35.attention.key_length", MetaValue::U32(32)),
            ("qwen35.attention.value_length", MetaValue::U32(32)),
            (
                "qwen35.attention.layer_norm_rms_epsilon",
                MetaValue::F32(1e-6),
            ),
            ("qwen35.context_length", MetaValue::U32(128)),
            ("qwen35.rope.freq_base", MetaValue::F32(1_000_000.0)),
            ("qwen35.rope.dimension_count", MetaValue::U32(32)),
            (
                "qwen35.rope.dimension_sections",
                MetaValue::ArrayI32(vec![4, 4, 4, 4]),
            ),
            ("qwen35.full_attention_interval", MetaValue::U32(0)),
            ("qwen35.ssm.state_size", MetaValue::U32(32)),
            ("qwen35.ssm.group_count", MetaValue::U32(1)),
            ("qwen35.ssm.inner_size", MetaValue::U32(32)),
            ("qwen35.ssm.conv_kernel", MetaValue::U32(4)),
            ("qwen35.vocab_size", MetaValue::U32(2)),
            ("qwen35.feed_forward_length", MetaValue::U32(32)),
            ("tokenizer.ggml.pre", MetaValue::String("qwen35".into())),
            (
                "tokenizer.ggml.tokens",
                MetaValue::ArrayString(vec!["a".into(), "b".into()]),
            ),
            ("tokenizer.ggml.merges", MetaValue::ArrayString(Vec::new())),
            ("tokenizer.ggml.token_type", MetaValue::ArrayI32(vec![1, 1])),
        ];
        if include_mtp {
            metadata.push(("qwen35.nextn_predict_layers", MetaValue::U32(1)));
            metadata.push((
                "qwen35.nextn.use_dedicated_embeddings",
                MetaValue::Bool(mtp_dedicated_embedding_type.is_some()),
            ));
        }
        let mut tensors = vec![
            ("token_embd.weight", vec![32, token_rows], GgmlType::F32),
            ("output_norm.weight", vec![32], GgmlType::F32),
            ("blk.0.attn_norm.weight", vec![32], GgmlType::F32),
            ("blk.0.post_attention_norm.weight", vec![32], GgmlType::F32),
            ("blk.0.attn_qkv.weight", vec![32, 96], GgmlType::Q4_0),
            ("blk.0.attn_gate.weight", vec![32, 32], GgmlType::Q4_0),
            ("blk.0.ssm_conv1d.weight", vec![4, 96], GgmlType::F32),
            ("blk.0.ssm_alpha.weight", vec![32, 1], GgmlType::Q4_0),
            ("blk.0.ssm_dt.bias", vec![1], GgmlType::F32),
            ("blk.0.ssm_beta.weight", vec![32, 1], GgmlType::Q4_0),
            ("blk.0.ssm_a", vec![1], GgmlType::F32),
            ("blk.0.ssm_norm.weight", vec![32], GgmlType::F32),
            ("blk.0.ssm_out.weight", vec![32, 32], GgmlType::Q4_0),
            ("blk.0.ffn_gate.weight", vec![32, 32], dense_gate),
            ("blk.0.ffn_up.weight", vec![32, 32], dense_up),
            ("blk.0.ffn_down.weight", vec![32, 32], dense_down),
        ];
        if let Some(rows) = output_rows {
            tensors.push(("output.weight", vec![32, rows], GgmlType::F32));
        }
        if include_mtp {
            tensors.extend([
                ("blk.1.nextn.enorm.weight", vec![32], GgmlType::F32),
                ("blk.1.nextn.hnorm.weight", vec![32], GgmlType::F32),
                (
                    "blk.1.nextn.eh_proj.weight",
                    vec![64, 32],
                    mtp_projection_type,
                ),
                (
                    "blk.1.nextn.shared_head_norm.weight",
                    vec![32],
                    GgmlType::F32,
                ),
                ("blk.1.attn_norm.weight", vec![32], GgmlType::F32),
                ("blk.1.post_attention_norm.weight", vec![32], GgmlType::F32),
                ("blk.1.attn_q.weight", vec![32, 32], GgmlType::F32),
                ("blk.1.attn_k.weight", vec![32, 32], GgmlType::F32),
                ("blk.1.attn_v.weight", vec![32, 32], GgmlType::F32),
                ("blk.1.attn_output.weight", vec![32, 32], GgmlType::F32),
                ("blk.1.attn_q_norm.weight", vec![32], GgmlType::F32),
                ("blk.1.attn_k_norm.weight", vec![32], GgmlType::F32),
                ("blk.1.ffn_gate.weight", vec![32, 32], GgmlType::F32),
                ("blk.1.ffn_up.weight", vec![32, 32], GgmlType::F32),
                ("blk.1.ffn_down.weight", vec![32, 32], GgmlType::F32),
            ]);
            if let Some(embed_type) = mtp_dedicated_embedding_type {
                tensors.push((
                    "blk.1.nextn.embed_tokens.weight",
                    vec![32, mtp_embedding_rows],
                    embed_type,
                ));
                tensors.push((
                    "blk.1.nextn.shared_head_head.weight",
                    vec![32, mtp_head_rows],
                    GgmlType::F32,
                ));
            }
        }

        let cursor = Cursor::new(Vec::new());
        let mut writer = GgufWriter::new(cursor);
        writer
            .write_header(tensors.len() as u64, metadata.len() as u64)
            .unwrap();
        for (key, value) in &metadata {
            writer.write_metadata_kv(key, value).unwrap();
        }
        for (name, dims, ggml_type) in &tensors {
            let dims = dims.iter().map(|&dim| dim as u64).collect::<Vec<_>>();
            writer.reserve_tensor_info(name, &dims, *ggml_type).unwrap();
        }
        writer.pad_to_alignment().unwrap();
        for (index, (_, dims, ggml_type)) in tensors.iter().enumerate() {
            let rows = dims.iter().skip(1).product::<usize>();
            let n_per_row = dims[0];
            let bytes = vec![0_u8; rows * ggml_type.row_size(n_per_row)];
            writer.stream_tensor_payload(index, &bytes).unwrap();
        }
        writer.finalize().unwrap();
        let bytes = writer.into_inner().into_inner();
        std::fs::write(path, &bytes).unwrap();
        bytes.len() as u64
    }

    #[test]
    fn authenticated_range_identity_and_selected_header_must_match_catalog() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("header.gguf");
        let bytes = write_test_gguf(
            &path,
            "qwen35",
            crate::quantize::ggml_quants::GgufFtype::MostlyQ4_K_M as u32,
        );
        let artifact = HubGgufArtifact {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            filename: "model-q4_k_m.gguf".into(),
            bytes,
            sha256: "b".repeat(64),
            quant_hint: Some("Q4_K_M".into()),
            role: "text_model".into(),
            selectable: true,
            unavailable_reason: None,
        };
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-repo-commit", artifact.revision.parse().unwrap());
        headers.insert("x-linked-size", artifact.bytes.to_string().parse().unwrap());
        headers.insert(
            "x-linked-etag",
            format!("\"{}\"", artifact.sha256).parse().unwrap(),
        );
        validate_hub_range_identity(&headers, &artifact).unwrap();
        let prefix = std::fs::read(&path).unwrap();
        let header = gguf_probe::parse_bounded_header(&prefix, artifact.bytes).unwrap();
        validate_probed_gguf_header(&header, &artifact).unwrap();

        let mut wrong_quant = artifact.clone();
        wrong_quant.quant_hint = Some("Q8_0".into());
        assert!(validate_probed_gguf_header(&header, &wrong_quant).is_err());
        headers.insert(
            "x-linked-size",
            (artifact.bytes + 1).to_string().parse().unwrap(),
        );
        assert!(validate_hub_range_identity(&headers, &artifact).is_err());
    }

    #[test]
    fn complete_qwen_prefix_admits_runtime_metadata_tokenizer_and_tensor_topology() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("complete-qwen.gguf");
        let logical_bytes = write_complete_qwen_test_gguf(&path);
        let prefix = std::fs::read(path).unwrap();
        let header = gguf_probe::parse_bounded_header(&prefix, logical_bytes).unwrap();
        validate_qwen_hosted_prefix(&prefix, &header).unwrap();
    }

    #[test]
    fn complete_qwen_mtp_prefix_admits_the_same_one_layer_topology_as_runtime() {
        use crate::quantize::ggml_quants::GgmlType;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("complete-qwen-mtp.gguf");
        let logical_bytes = write_complete_qwen_test_gguf_with_options(
            &path,
            GgmlType::Q4_0,
            GgmlType::Q4_0,
            GgmlType::Q4_0,
            true,
            GgmlType::F32,
            None,
        );
        let prefix = std::fs::read(&path).unwrap();
        let header = gguf_probe::parse_bounded_header(&prefix, logical_bytes).unwrap();
        validate_qwen_hosted_prefix(&prefix, &header).unwrap();
        let gguf = mlx_native::gguf::GgufFile::open(&path).unwrap();
        let cfg = crate::inference::models::qwen35::Qwen35Config::from_gguf(&gguf).unwrap();
        assert_eq!(cfg.mtp_num_hidden_layers, 1);
        crate::inference::models::qwen35::mtp_weights_load::validate_mtp_tensor_topology(
            &gguf, &cfg,
        )
        .unwrap();
    }

    #[test]
    fn qwen_admission_rejects_embedding_tables_smaller_than_selected_heads() {
        use crate::quantize::ggml_quants::GgmlType;

        let directory = tempfile::tempdir().unwrap();
        let main = directory.path().join("main-vocab-mismatch.gguf");
        let main_bytes = write_complete_qwen_test_gguf_with_vocab_rows(
            &main,
            GgmlType::Q4_0,
            GgmlType::Q4_0,
            GgmlType::Q4_0,
            false,
            GgmlType::F32,
            None,
            2,
            Some(3),
            2,
            2,
        );
        let prefix = fs::read(&main).unwrap();
        let header = gguf_probe::parse_bounded_header(&prefix, main_bytes).unwrap();
        let error = validate_qwen_hosted_prefix(&prefix, &header).unwrap_err();
        assert!(error.to_string().contains("output-head rows"), "{error}");

        let mtp = directory.path().join("mtp-vocab-mismatch.gguf");
        let mtp_bytes = write_complete_qwen_test_gguf_with_vocab_rows(
            &mtp,
            GgmlType::Q4_0,
            GgmlType::Q4_0,
            GgmlType::Q4_0,
            true,
            GgmlType::F32,
            Some(GgmlType::F32),
            3,
            Some(3),
            2,
            3,
        );
        let prefix = fs::read(&mtp).unwrap();
        let header = gguf_probe::parse_bounded_header(&prefix, mtp_bytes).unwrap();
        let error = validate_qwen_hosted_prefix(&prefix, &header).unwrap_err();
        assert!(error.to_string().contains("MTP embedding rows"), "{error}");
    }

    #[test]
    fn mtp_hosted_admission_rejects_shape_correct_non_executable_storage() {
        use crate::quantize::ggml_quants::GgmlType;

        let directory = tempfile::tempdir().unwrap();
        let projection = directory.path().join("mtp-bad-projection.gguf");
        let projection_bytes = write_complete_qwen_test_gguf_with_options(
            &projection,
            GgmlType::Q4_0,
            GgmlType::Q4_0,
            GgmlType::Q4_0,
            true,
            GgmlType::F16,
            None,
        );
        let prefix = std::fs::read(&projection).unwrap();
        let header = gguf_probe::parse_bounded_header(&prefix, projection_bytes).unwrap();
        let error = validate_qwen_hosted_prefix(&prefix, &header).unwrap_err();
        assert!(error.to_string().contains("eh_proj.weight"), "{error}");

        let embedding = directory.path().join("mtp-bad-embedding.gguf");
        let embedding_bytes = write_complete_qwen_test_gguf_with_options(
            &embedding,
            GgmlType::Q4_0,
            GgmlType::Q4_0,
            GgmlType::Q4_0,
            true,
            GgmlType::F32,
            Some(GgmlType::F16),
        );
        let prefix = std::fs::read(&embedding).unwrap();
        let header = gguf_probe::parse_bounded_header(&prefix, embedding_bytes).unwrap();
        let error = validate_qwen_hosted_prefix(&prefix, &header).unwrap_err();
        assert!(error.to_string().contains("embed_tokens.weight"), "{error}");
    }

    #[test]
    fn qwen_dense_hosted_admission_is_isomorphic_to_runtime_storage() {
        use crate::quantize::ggml_quants::GgmlType;

        let directory = tempfile::tempdir().unwrap();
        for (name, gate, up, down) in [
            ("float.gguf", GgmlType::F16, GgmlType::F32, GgmlType::F16),
            (
                "matched-quant.gguf",
                GgmlType::Q4_0,
                GgmlType::Q4_0,
                GgmlType::Q8_0,
            ),
        ] {
            let path = directory.path().join(name);
            let logical_bytes =
                write_complete_qwen_test_gguf_with_dense_types(&path, gate, up, down);
            let prefix = std::fs::read(path).unwrap();
            let header = gguf_probe::parse_bounded_header(&prefix, logical_bytes).unwrap();
            assert!(header.incompatible_tensor.is_none());
            validate_qwen_hosted_prefix(&prefix, &header).unwrap();
        }

        let mismatched = directory.path().join("mismatched-quant.gguf");
        let logical_bytes = write_complete_qwen_test_gguf_with_dense_types(
            &mismatched,
            GgmlType::Q4_0,
            GgmlType::Q8_0,
            GgmlType::Q4_0,
        );
        let prefix = std::fs::read(mismatched).unwrap();
        let header = gguf_probe::parse_bounded_header(&prefix, logical_bytes).unwrap();
        assert!(header.incompatible_tensor.is_none());
        let error = validate_qwen_hosted_prefix(&prefix, &header).unwrap_err();
        assert!(
            error.to_string().contains("gate/up quant types differ"),
            "{error}"
        );
    }

    #[test]
    fn exact_owned_hub_bytes_use_local_compatibility_admission_without_a_range_request() {
        let dir = tempfile::tempdir().unwrap();
        let complete = dir.path().join("complete-qwen-q4_k_m.gguf");
        let bytes = write_complete_qwen_test_gguf(&complete);
        let artifact = HubGgufArtifact {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            filename: "complete-qwen-q4_k_m.gguf".into(),
            bytes,
            sha256: crate::core::sha256::compute_file_sha256(&complete).unwrap(),
            quant_hint: Some("Q4_K_M".into()),
            role: "text_model".into(),
            selectable: true,
            unavailable_reason: None,
        };
        validate_local_hub_gguf_compatibility(&complete, &artifact).unwrap();

        let incomplete = dir.path().join("incomplete-qwen-q4_k_m.gguf");
        let mut incompatible = artifact;
        incompatible.filename = "incomplete-qwen-q4_k_m.gguf".into();
        incompatible.bytes = write_test_gguf(
            &incomplete,
            "qwen35",
            crate::quantize::ggml_quants::GgufFtype::MostlyQ4_K_M as u32,
        );
        incompatible.sha256 = crate::core::sha256::compute_file_sha256(&incomplete).unwrap();
        assert!(matches!(
            validate_local_hub_gguf_compatibility(&incomplete, &incompatible),
            Err(DownloadError::IncompatibleHostedGguf { .. })
        ));
    }

    #[test]
    fn exact_revision_hub_cache_lookup_reuses_an_existing_quant_without_transfer() {
        let cache = tempfile::tempdir().unwrap();
        let revision = "a".repeat(40);
        let relative = Path::new("nested/model-q8_0.gguf");
        let path = cache
            .path()
            .join("models--owner--model")
            .join("snapshots")
            .join(&revision)
            .join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"already downloaded hosted quant").unwrap();
        let artifact = HubGgufArtifact {
            repository: "owner/model".into(),
            revision,
            filename: relative.to_string_lossy().into_owned(),
            bytes: std::fs::metadata(&path).unwrap().len(),
            sha256: crate::core::sha256::compute_file_sha256(&path).unwrap(),
            quant_hint: Some("Q8_0".into()),
            role: "text_model".into(),
            selectable: true,
            unavailable_reason: None,
        };
        assert_eq!(
            cached_hub_gguf_path_in(cache.path(), &artifact)
                .unwrap()
                .canonicalize()
                .unwrap(),
            path.canonicalize().unwrap()
        );
    }

    #[test]
    fn qwen_operational_scalars_reject_non_executable_runtime_configs() {
        use crate::inference::models::qwen35::{
            Qwen35Config, Qwen35LayerKind, Qwen35MoeConfig, Qwen35Variant,
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("complete-qwen.gguf");
        write_complete_qwen_test_gguf(&path);
        let gguf = mlx_native::gguf::GgufFile::open(&path).unwrap();
        let base = Qwen35Config::from_gguf(&gguf).unwrap();

        let mut zero_heads = base.clone();
        zero_heads.num_attention_heads = 0;
        assert!(validate_qwen_operational_config(&zero_heads).is_err());

        let mut indivisible_heads = base.clone();
        indivisible_heads.num_attention_heads = 3;
        indivisible_heads.num_key_value_heads = 2;
        assert!(validate_qwen_operational_config(&indivisible_heads).is_err());

        let mut incoherent_rope = base.clone();
        incoherent_rope.mrope_section = [8, 8, 8, 8];
        assert!(validate_qwen_operational_config(&incoherent_rope).is_err());

        let mut invalid_moe = base;
        invalid_moe.variant = Qwen35Variant::Moe;
        invalid_moe.intermediate_size = None;
        invalid_moe.moe = Some(Qwen35MoeConfig {
            moe_intermediate_size: 32,
            num_experts: 2,
            num_experts_per_tok: 3,
            shared_expert_intermediate_size: 32,
        });
        assert!(validate_qwen_operational_config(&invalid_moe).is_err());

        let mut overflowing_projection = Qwen35Config::from_gguf(&gguf).unwrap();
        overflowing_projection.num_attention_heads = u32::MAX;
        overflowing_projection.head_dim = u32::MAX;
        overflowing_projection.layer_types = vec![Qwen35LayerKind::FullAttention];
        assert!(validate_qwen_hosted_topology(&gguf, &overflowing_projection).is_err());
    }

    #[test]
    fn incomplete_qwen_prefix_is_rejected_before_payload_transfer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("incomplete-qwen.gguf");
        let logical_bytes = write_test_gguf(
            &path,
            "qwen35",
            crate::quantize::ggml_quants::GgufFtype::MostlyQ4_K_M as u32,
        );
        let prefix = std::fs::read(path).unwrap();
        let header = gguf_probe::parse_bounded_header(&prefix, logical_bytes).unwrap();
        let error = validate_qwen_hosted_prefix(&prefix, &header).unwrap_err();
        assert!(
            matches!(error, DownloadError::IncompatibleHostedGguf { .. }),
            "{error}"
        );
    }

    #[test]
    fn qwen_q3_embedding_layout_is_rejected_before_payload_transfer() {
        let artifact = HubGgufArtifact {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            filename: "model-q3_k_m.gguf".into(),
            bytes: 4096,
            sha256: "b".repeat(64),
            quant_hint: Some("Q3_K_M".into()),
            role: "text_model".into(),
            selectable: true,
            unavailable_reason: None,
        };
        let incompatible = gguf_probe::ProbedGgufHeader {
            architecture: "qwen35".into(),
            requires_projector: false,
            file_type: crate::quantize::ggml_quants::GgufFtype::MostlyQ3_K_M as u32,
            tensor_count: 1,
            tensor_data_offset: 256,
            token_embedding_type: Some(11),
            incompatible_tensor: Some(
                "token_embd.weight uses unsupported GGML type 11 for qwen35".into(),
            ),
            has_output_norm: true,
            has_block_tensor: true,
        };
        let error = validate_probed_gguf_header(&incompatible, &artifact).unwrap_err();
        assert!(matches!(
            error,
            DownloadError::IncompatibleHostedGguf { .. }
        ));

        let supported = gguf_probe::ProbedGgufHeader {
            token_embedding_type: Some(14),
            incompatible_tensor: None,
            ..incompatible
        };
        validate_probed_gguf_header(&supported, &artifact).unwrap();
    }

    #[test]
    fn family_sentinels_and_unproven_hosted_families_fall_back_before_download() {
        let artifact = HubGgufArtifact {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            filename: "model-q4_k_m.gguf".into(),
            bytes: 4096,
            sha256: "b".repeat(64),
            quant_hint: Some("Q4_K_M".into()),
            role: "text_model".into(),
            selectable: true,
            unavailable_reason: None,
        };
        let missing = gguf_probe::ProbedGgufHeader {
            architecture: "qwen35".into(),
            requires_projector: false,
            file_type: crate::quantize::ggml_quants::GgufFtype::MostlyQ4_K_M as u32,
            tensor_count: 1,
            tensor_data_offset: 256,
            token_embedding_type: None,
            incompatible_tensor: None,
            has_output_norm: false,
            has_block_tensor: false,
        };
        assert!(matches!(
            validate_probed_gguf_header(&missing, &artifact),
            Err(DownloadError::IncompatibleHostedGguf { .. })
        ));

        let mut deepseek = gguf_probe::ProbedGgufHeader {
            architecture: "deepseek4".into(),
            requires_projector: false,
            file_type: crate::quantize::ggml_quants::GgufFtype::MostlyQ4_K_M as u32,
            tensor_count: 3,
            tensor_data_offset: 256,
            token_embedding_type: Some(12),
            incompatible_tensor: None,
            has_output_norm: true,
            has_block_tensor: true,
        };
        assert!(matches!(
            validate_probed_gguf_header(&deepseek, &artifact),
            Err(DownloadError::IncompatibleHostedGguf { .. })
        ));
        deepseek.token_embedding_type = Some(10);
        assert!(matches!(
            validate_probed_gguf_header(&deepseek, &artifact),
            Err(DownloadError::IncompatibleHostedGguf { .. })
        ));
    }

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

    #[test]
    fn exact_repository_config_distinguishes_multimodal_from_stray_companions() {
        assert!(config_requires_projector(&serde_json::json!({
            "model_type": "qwen3_5",
            "vision_config": {"model_type": "qwen3_vl"}
        })));
        assert!(!config_requires_projector(&serde_json::json!({
            "model_type": "qwen3_5"
        })));
    }

    #[test]
    fn native_projector_planning_fails_closed_when_exact_config_inspection_fails() {
        let failed = inspect_repository_config_with(true, || {
            Err(DownloadError::InvalidRepositoryInventory {
                reason: "forced authenticated config failure".into(),
            })
        });
        assert!(failed.is_err());
        assert!(inspect_repository_config_with(false, || Ok(serde_json::json!({}))).is_err());
        let config =
            inspect_repository_config_with(true, || Ok(serde_json::json!({"vision_config": {}})))
                .unwrap();
        assert!(config_requires_projector(&config));
    }

    #[test]
    fn native_inventory_counts_all_uncached_optional_metadata_before_payload() {
        let mib = 1024_u64 * 1024;
        let inventory = validate_repo_inventory([
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer.model",
            "merges.txt",
            "vocab.json",
        ])
        .unwrap();
        let selected = initial_download_files(&inventory).unwrap();
        for filename in [
            "config.json",
            "tokenizer.json",
            "tokenizer.model",
            "merges.txt",
            "vocab.json",
        ] {
            assert!(selected.iter().any(|value| value == filename));
        }
        let (total, uncached) = checked_inventory_extents(
            [
                (16 * mib, false),
                (512 * mib, false),
                (512 * mib, false),
                (512 * mib, false),
                (512 * mib, false),
            ],
            "native source metadata",
        )
        .unwrap();
        assert_eq!(total, 2 * 1024 * mib + 16 * mib);
        assert_eq!(uncached, total);
        let (_, with_one_cached) = checked_inventory_extents(
            [(16 * mib, false), (512 * mib, true), (1536 * mib, false)],
            "native source metadata",
        )
        .unwrap();
        assert_eq!(with_one_cached, total - 512 * mib);
        assert!(checked_inventory_extents(
            [(u64::MAX, false), (1, false)],
            "native source metadata",
        )
        .is_err());

        let plan = NativeSourcePlan {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            total_weight_bytes: 10 * 1024 * mib,
            uncached_weight_bytes: 10 * 1024 * mib,
            total_metadata_bytes: total,
            uncached_metadata_bytes: uncached,
            requires_projector: false,
            output_upper_bound_bytes: 22 * 1024 * mib,
            metadata_records: Vec::new(),
            weight_records: Vec::new(),
        };
        assert_eq!(
            plan.uncached_source_bytes().unwrap(),
            12 * 1024 * mib + 16 * mib
        );
        assert_eq!(
            native_source_conversion_extents(
                plan.uncached_source_bytes().unwrap(),
                18 * 1024 * mib,
                false,
            )
            .unwrap(),
            (12 * 1024 * mib + 16 * mib, 18 * 1024 * mib)
        );
        assert_eq!(
            native_source_conversion_extents(
                plan.uncached_source_bytes().unwrap(),
                18 * 1024 * mib,
                true,
            )
            .unwrap(),
            (0, 30 * 1024 * mib + 16 * mib)
        );
    }

    #[test]
    fn native_metadata_stage_preflights_before_first_cache_write() {
        let records = vec![
            (
                "config.json".to_owned(),
                ShardIntegrity::from_metadata("config.json", &"a".repeat(40), 16),
                false,
            ),
            (
                "tokenizer.json".to_owned(),
                ShardIntegrity::from_metadata("tokenizer.json", &"b".repeat(40), 32),
                false,
            ),
        ];
        let reserve = 2 * 1024_u64.pow(3);
        let stage_calls = std::cell::Cell::new(0_usize);
        let error = admit_and_stage_native_metadata(
            "owner/model",
            Path::new("/planned/cache"),
            &records,
            Some(reserve + 48 - 1),
            |_| {
                stage_calls.set(stage_calls.get() + 1);
                Ok(())
            },
        )
        .unwrap_err();
        assert!(matches!(error, DownloadError::InsufficientDisk { .. }));
        assert_eq!(
            stage_calls.get(),
            0,
            "admission failure must precede every cache write"
        );

        admit_and_stage_native_metadata(
            "owner/model",
            Path::new("/planned/cache"),
            &records,
            Some(reserve + 48),
            |_| {
                stage_calls.set(stage_calls.get() + 1);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(stage_calls.get(), 1);
    }

    #[test]
    fn authenticated_sparse_source_fetches_only_safetensors_headers() {
        let header = br#"{"weight":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
        let mut object = Vec::new();
        object.extend_from_slice(&(header.len() as u64).to_le_bytes());
        object.extend_from_slice(header);
        object.extend_from_slice(&[1, 2, 3, 4]);
        let mut ranges = Vec::new();
        let (_, fetched_header, header_end) = fetch_authenticated_safetensors_header(
            object.len() as u64,
            "model.safetensors",
            |start, end| {
                ranges.push((start, end));
                Ok(object[start as usize..=end as usize].to_vec())
            },
        )
        .unwrap();
        assert_eq!(fetched_header, header);
        assert_eq!(
            ranges,
            vec![(0, 7), (8, header_end - 1)],
            "no requested range may reach the tensor payload"
        );
        assert_eq!(header_end, object.len() as u64 - 4);

        assert!(authenticated_safetensors_header_end(&[0; 8], 1024, "bad").is_err());
        assert!(authenticated_safetensors_header_end(
            &(100_000_001_u64).to_le_bytes(),
            200_000_000,
            "bad",
        )
        .is_err());
    }

    #[test]
    fn native_output_bound_covers_kept_f32_fp16_fp8_and_mxfp4_sources() {
        let gib = 1024_u64 * 1024 * 1024;
        assert_eq!(
            native_output_upper_bound_bytes(10 * gib, &serde_json::json!({})),
            22 * gib
        );
        assert_eq!(
            native_output_upper_bound_bytes(
                10 * gib,
                &serde_json::json!({"quantization_config": {"quant_method": "fp8"}}),
            ),
            42 * gib
        );
        assert_eq!(
            native_output_upper_bound_bytes(
                10 * gib,
                &serde_json::json!({"quantization_config": {"quant_method": "mxfp4"}}),
            ),
            82 * gib
        );
    }

    #[test]
    fn mixed_repository_ggufs_are_classified_without_source_fallback() {
        for (filename, quant) in [
            ("gguf/model-q2_k.gguf", "Q2_K"),
            ("gguf/model-q3_k_m.gguf", "Q3_K_M"),
            ("gguf/model-q4_k_m.gguf", "Q4_K_M"),
            ("gguf/model-q5_k_m.gguf", "Q5_K_M"),
            ("gguf/model-q6_k.gguf", "Q6_K"),
            ("gguf/model-q8_0.gguf", "Q8_0"),
        ] {
            assert_eq!(
                classify_hub_gguf(filename),
                ("text_model", Some(quant.to_owned()), None)
            );
        }
        let mmproj = classify_hub_gguf("gguf/mmproj-model-f16.gguf");
        assert_eq!(mmproj.0, "companion");
        assert!(mmproj.2.unwrap().contains("not a text model"));
        let bf16 = classify_hub_gguf("gguf/model-bf16.gguf");
        assert_eq!(bf16.1.as_deref(), Some("BF16"));
        assert!(bf16.2.unwrap().contains("not supported"));
        let split = classify_hub_gguf("model-q6_k-00001-of-00002.gguf");
        assert!(split.2.unwrap().contains("split GGUF"));
    }

    #[test]
    fn hosted_companion_identity_is_role_specific_and_not_text_selectable() {
        let companion = HubGgufArtifact {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            filename: "gguf/mmproj-model-f16.gguf".into(),
            bytes: 1024,
            sha256: "b".repeat(64),
            quant_hint: None,
            role: "companion".into(),
            selectable: false,
            unavailable_reason: Some("vision projector companion; not a text model".into()),
        };
        assert!(hosted_gguf_identity_valid_for_role(&companion, "companion"));
        assert!(!hosted_gguf_identity_valid_for_role(
            &companion,
            "text_model"
        ));
    }

    #[test]
    fn native_conversion_preflight_uses_exact_plan_plus_safety_headroom() {
        let gib = 1024_u64 * 1024 * 1024;
        let error = check_exact_disk_preflight(
            "native hf2q conversion for owner/model".into(),
            Path::new("/planned/output.gguf"),
            4 * gib,
            Some(5 * gib),
        )
        .unwrap_err();
        assert!(error.to_string().contains("native hf2q conversion"));
        assert!(check_exact_disk_preflight(
            "native hf2q conversion for owner/model".into(),
            Path::new("/planned/output.gguf"),
            4 * gib,
            Some(6 * gib),
        )
        .is_ok());
    }

    #[test]
    fn exact_transfer_preflight_fails_closed_on_unknown_or_zero_capacity() {
        for available in [None, Some(0)] {
            let error = check_exact_disk_capacity(
                "hosted GGUF pair".into(),
                Path::new("/unknown-volume/model.gguf"),
                1,
                available,
            )
            .unwrap_err();
            assert!(matches!(error, DownloadError::InsufficientDisk { .. }));
        }
        assert!(check_exact_disk_capacity(
            "already materialized".into(),
            Path::new("/unknown-volume/model.gguf"),
            0,
            None,
        )
        .is_ok());
    }

    #[test]
    fn retained_cross_device_pair_preflight_aggregates_on_destination_authority() {
        let directory = tempfile::tempdir().unwrap();
        let text_destination = directory.path().join("text.gguf");
        let projector_destination = directory.path().join("mmproj.gguf");
        let gib = 1024_u64 * 1024 * 1024;

        let error = check_local_artifact_pair_plan_with_authorities(
            "owner/model",
            Some(1),
            &text_destination,
            Some(9),
            Some(3 * gib),
            gib,
            false,
            Some((
                Some(2),
                &projector_destination,
                Some(9),
                Some(3 * gib),
                gib,
                false,
            )),
        )
        .expect_err("two cross-device GiB copies plus headroom require four GiB");
        assert!(matches!(error, DownloadError::InsufficientDisk { .. }));

        check_local_artifact_pair_plan_with_authorities(
            "owner/model",
            Some(1),
            &text_destination,
            Some(9),
            Some(5 * gib),
            gib,
            false,
            Some((
                Some(2),
                &projector_destination,
                Some(9),
                Some(5 * gib),
                gib,
                false,
            )),
        )
        .unwrap();
    }

    #[test]
    fn native_source_and_output_are_admitted_as_one_shared_filesystem_extent() {
        let gib = 1024_u64 * 1024 * 1024;
        assert_eq!(
            native_source_conversion_extents(55 * gib, 17 * gib, true).unwrap(),
            (0, 72 * gib)
        );
        assert_eq!(
            native_source_conversion_extents(55 * gib, 17 * gib, false).unwrap(),
            (55 * gib, 17 * gib)
        );
        assert_eq!(
            native_source_conversion_extents(0, 17 * gib, true).unwrap(),
            (0, 17 * gib),
            "already-valid cached source bytes must not be charged twice"
        );
        assert!(native_source_conversion_extents(u64::MAX, 1, true).is_err());
    }

    #[test]
    fn hosted_multimodal_pair_only_reserves_hub_cache_payload_bytes() {
        let gib = 1024_u64 * 1024 * 1024;
        assert_eq!(
            hosted_pair_required_extents(20 * gib, 2 * gib).unwrap(),
            22 * gib
        );
        assert_eq!(
            hosted_pair_required_extents(0, 2 * gib).unwrap(),
            2 * gib,
            "a cached text model reserves only the missing projector cache bytes"
        );
        assert_eq!(
            hosted_pair_required_extents(0, 0).unwrap(),
            0,
            "managed symlinks require no model-sized destination extent"
        );
        assert!(check_exact_disk_preflight(
            "already cached hosted pair".into(),
            Path::new("/full/cache"),
            0,
            Some(1),
        )
        .is_ok());
        assert!(hosted_pair_required_extents(u64::MAX, 1).is_err());
    }

    #[test]
    fn exact_local_text_projector_destinations_require_zero_new_extent() {
        assert!(check_local_artifact_pair_plan(
            "owner/model",
            Path::new("/source-volume/text.gguf"),
            Path::new("/full-destination/text.gguf"),
            u64::MAX,
            true,
            Some((
                Path::new("/source-volume/mmproj.gguf"),
                Path::new("/full-destination/mmproj.gguf"),
                u64::MAX,
                true,
            )),
        )
        .is_ok());
    }

    #[test]
    fn mixed_local_and_hosted_extents_charge_only_real_payload_allocations() {
        let gib = 1024_u64.pow(3);
        assert_eq!(
            local_text_hosted_projector_extents(10 * gib, 10 * gib, true).unwrap(),
            (20 * gib, 0),
            "a local text copy and hosted download on one device are additive"
        );
        assert_eq!(
            local_text_hosted_projector_extents(10 * gib, 10 * gib, false).unwrap(),
            (10 * gib, 10 * gib),
            "a hosted projector symlink consumes no destination payload extent"
        );
        assert_eq!(
            hosted_text_local_projector_extents(10 * gib, 2 * gib, true).unwrap(),
            (12 * gib, 0),
            "hosted text cache bytes and a local projector copy aggregate on one device"
        );
        assert_eq!(
            hosted_text_local_projector_extents(10 * gib, 2 * gib, false).unwrap(),
            (10 * gib, 2 * gib)
        );
        assert!(local_text_hosted_projector_extents(u64::MAX, 1, true).is_err());
        assert!(hosted_text_local_projector_extents(u64::MAX, 1, true).is_err());
    }

    #[test]
    fn hosted_artifact_request_identity_includes_revision_filename_and_hash() {
        let artifact = HubGgufArtifact {
            repository: "owner/model".to_owned(),
            revision: "a".repeat(40),
            filename: "gguf/model-q6_k.gguf".to_owned(),
            bytes: 42,
            sha256: "b".repeat(64),
            quant_hint: Some("Q6_K".to_owned()),
            role: "text_model".to_owned(),
            selectable: true,
            unavailable_reason: None,
        };
        let identity = artifact.request_model();
        assert!(identity.contains("owner/model@aaaaaaaa"));
        assert!(identity.contains("gguf/model-q6_k.gguf"));
        assert!(identity.ends_with(&"b".repeat(64)));
    }

    #[test]
    fn hosted_gguf_transfer_rejects_forged_non_gguf_identity_before_network() {
        let artifact = HubGgufArtifact {
            repository: "owner/model".to_owned(),
            revision: "main".to_owned(),
            filename: "model-00001-of-00012.safetensors".to_owned(),
            bytes: 42,
            sha256: "not-a-hash".to_owned(),
            quant_hint: Some("Q6_K".to_owned()),
            role: "text_model".to_owned(),
            selectable: true,
            unavailable_reason: None,
        };
        let error = download_hub_gguf(&artifact).unwrap_err();
        assert!(error
            .to_string()
            .contains("identity is incomplete or malformed"));
    }

    #[test]
    fn qwen38_hosted_q5_transfer_identity_is_admitted_before_hub_access() {
        let artifact = HubGgufArtifact {
            repository: "owner/model".to_owned(),
            revision: "a".repeat(40),
            filename: "model-q5_k_m.gguf".to_owned(),
            bytes: 42,
            sha256: "b".repeat(64),
            quant_hint: Some("Q5_K_M".to_owned()),
            role: "text_model".to_owned(),
            selectable: true,
            unavailable_reason: None,
        };
        assert!(hosted_gguf_identity_valid_for_role(&artifact, "text_model"));

        let mut forged = artifact.clone();
        forged.quant_hint = Some("Q6_K".to_owned());
        assert!(!hosted_gguf_identity_valid_for_role(&forged, "text_model"));
    }

    /// Metadata-only regression proof for the mixed repository that exposed
    /// the diagnostic-chat source-download defect. The selected GGUF probe is
    /// bounded to its authenticated header prefix; no tensor payload is read.
    #[test]
    fn live_mixed_qwen38_repository_catalogs_hosted_ggufs_only() {
        if std::env::var("HF2Q_NETWORK_TESTS").ok().as_deref() != Some("1") {
            eprintln!("skipping network test (set HF2Q_NETWORK_TESTS=1 to run)");
            return;
        }
        const REVISION: &str = "0a72776892f98db49381fdf69f4b9982222ec9dc";
        let reference =
            HfModelReference::parse("jenerallee78/Qwen3.8-27B-Abliterated-SFT", Some(REVISION))
                .unwrap();
        let catalog = resolve_hub_gguf_catalog(reference).unwrap();
        assert_eq!(catalog.revision, REVISION);
        let selectable = catalog
            .artifacts
            .iter()
            .filter(|artifact| artifact.selectable)
            .map(|artifact| artifact.filename.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            selectable,
            vec![
                "gguf/qwen38-abliterated-sft-hf2q-q4_k_m.gguf",
                "gguf/qwen38-abliterated-sft-q5_k_m.gguf",
                "gguf/qwen38-abliterated-sft-q6_k.gguf",
                "gguf/qwen38-abliterated-sft-q8_0.gguf",
            ]
        );
        let q4 = catalog
            .artifacts
            .iter()
            .find(|artifact| artifact.quant_hint.as_deref() == Some("Q4_K_M"))
            .unwrap();
        validate_hub_gguf_header_compatibility(q4).unwrap();
    }

    /// Opt-in live proof that the accepted Qwen3.8 revision resolves through
    /// the exact production endpoint without transferring model payloads.
    #[test]
    fn live_qwen38_repository_info_matches_the_accepted_commit() {
        if std::env::var("HF2Q_NETWORK_TESTS").ok().as_deref() != Some("1") {
            eprintln!("skipping network test (set HF2Q_NETWORK_TESTS=1 to run)");
            return;
        }
        let api =
            build_hub_api(&resolve_hf_cache_dir(), false).expect("build exact-origin Hub client");
        let exact_repo = hub_model_repo(&api, QWEN38_REPOSITORY_ID);
        let info = exact_repo
            .info()
            .revision(QWEN38_ACCEPTED_REVISION)
            .send()
            .expect("fetch Qwen3.8 repository info");
        let reference = HfModelReference::parse(QWEN38_REPOSITORY_ID, None).unwrap();
        let sha = info.sha.expect("resolved commit");
        let siblings = info.siblings.expect("repository inventory");
        let resolved = resolve_repository_info(
            reference,
            QWEN38_ACCEPTED_REVISION,
            &sha,
            siblings.iter().map(|sibling| sibling.rfilename.as_str()),
        )
        .unwrap();
        assert_eq!(resolved.reference().revision(), QWEN38_ACCEPTED_REVISION);
        assert!(resolved.contains("config.json"));
        assert!(resolved.contains("model.safetensors.index.json"));
        let config =
            fetch_expected_file_metadata(&api, &exact_repo, resolved.reference(), "config.json")
                .expect("authenticate bounded Qwen3.8 config metadata");
        assert_eq!(config.filename, "config.json");
        assert!(config.bytes > 0 && config.bytes <= MAX_HF_SMALL_METADATA_BYTES);
    }

    /// Opt-in end-to-end proof that the production snapshot path selects the
    /// native Xet transport while retaining exact-file and SHA-256 contracts.
    #[test]
    fn live_xet_snapshot_preserves_exact_selection_and_integrity() {
        if std::env::var("HF2Q_NETWORK_TESTS").ok().as_deref() != Some("1") {
            eprintln!("skipping network test (set HF2Q_NETWORK_TESTS=1 to run)");
            return;
        }
        const REPO: &str = "hf-internal-testing/tiny-random-bert";
        const REVISION: &str = "f171d7baecaf37b5da5a3616d8833b9969753535";
        const FILE: &str = "pytorch_model.bin";

        let cache = tempfile::tempdir().unwrap();
        let api = build_hub_api(cache.path(), true).unwrap();
        let repo = hub_model_repo(&api, REPO);
        let resolved = HfModelReference::parse(REPO, Some(REVISION))
            .unwrap()
            .resolve(REVISION)
            .unwrap();
        let metadata = fetch_hub_file_metadata(&api, &resolved, FILE).expect("fetch Xet metadata");
        assert!(
            metadata.xet_hash.is_some(),
            "fixture must remain Xet-backed"
        );
        require_native_xet_payload("fixture.safetensors", metadata.xet_hash.as_deref())
            .expect("live Xet identity must satisfy the production payload contract");
        let record = validate_file_metadata(
            FILE,
            REVISION,
            &metadata.commit_hash,
            &metadata.etag,
            metadata.file_size,
        )
        .unwrap();
        let snapshot = download_selected_snapshot(
            &repo,
            REPO,
            REVISION,
            &[FILE.to_owned()],
            &ProgressReporter::new(),
        )
        .expect("download selected Xet snapshot");
        let path = snapshot.join(FILE);
        assert!(path.is_file());
        assert!(!snapshot.join("config.json").exists());
        verify_shard(REPO, REVISION, &path, &record).unwrap();
    }
}
