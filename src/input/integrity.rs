//! ADR-005 Phase 3 — HuggingFace shard integrity check (HTTP + FS).
//!
//! What lives here (post-B1.3, 2026-05-16):
//!
//! - [`fetch_repo_shard_metadata`] — issues HEAD requests to HF Hub via
//!   `hf-hub::Api::metadata`, parses `x-linked-etag` (LFS SHA-256) /
//!   plain `etag` (Git blob SHA-1) for each local file.
//! - [`verify_repo`] — convenience wrapper that calls fetch then
//!   `core::integrity::verify_shard` for every record.
//! - `enumerate_local_files` / `walk_dir` — snapshot-directory walker.
//!
//! What moved to [`crate::core::integrity`] (B1.3):
//!
//! - `ShardIntegrity` + `IntegrityError` + `verify_shard` + `shard_path`.
//!   Pure data + file-system + crypto — no HTTP / hf-hub coupling.
//!   Now lives on the foundation side of the workspace split so the
//!   serve / cache halves can consume the types without dragging in
//!   the HF download stack.
//!
//! The contract between the two halves is the same as it always was:
//!
//! > After the bytes land on disk, do they match what HuggingFace says
//! > should be there?  If not, refuse to proceed with a named,
//! > actionable error.
//!
//! `verify_repo` fails fast on the first byte-mismatch — there is no
//! "summary mode" that prints all mismatches and continues, because
//! corruption is a refuse-to-proceed event.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path};

use hf_hub::api::sync::ApiBuilder;
use serde::de::{self, IgnoredAny, MapAccess, Visitor};
use serde::Deserialize;
use tracing::{debug, info, warn};

use crate::core::integrity::{verify_shard, IntegrityError, ShardIntegrity};

const CANONICAL_HF_ENDPOINT: &str = "https://huggingface.co";
const MAX_VERIFIED_SOURCE_FILES: usize = 4096;
pub(super) const MAX_HF_SAFETENSORS_INDEX_BYTES: u64 = 16 * 1024 * 1024;
pub(super) const MAX_HF_SAFETENSORS_INDEX_ENTRIES: usize = 262_144;
const MAX_HF_TENSOR_NAME_BYTES: usize = 1024;

/// Type-state proof that every local source file was checked and every
/// index-required weight shard matched a strong HuggingFace LFS identity.
#[derive(Debug, Clone)]
pub struct VerifiedSourceManifest {
    records: Vec<ShardIntegrity>,
}

impl VerifiedSourceManifest {
    pub fn records(&self) -> &[ShardIntegrity] {
        &self.records
    }

    #[cfg(test)]
    pub(crate) fn for_test(records: Vec<ShardIntegrity>) -> Self {
        Self { records }
    }
}

fn validate_relative_source_path(filename: &str) -> Result<(), IntegrityError> {
    let path = Path::new(filename);
    if filename.is_empty()
        || filename.len() > crate::input::hf_reference::MAX_HF_FILENAME_BYTES
        || path.components().count() > crate::input::hf_reference::MAX_HF_FILENAME_COMPONENTS
        || !filename.is_ascii()
        || filename.contains('\\')
        || filename.bytes().any(|byte| byte.is_ascii_control())
        || path.is_absolute()
        || !path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(IntegrityError::InvalidSourceManifest {
            reason: format!("unsafe relative source path `{filename}`"),
        });
    }
    Ok(())
}

/// Read the exact safetensors shard set consumed by conversion.
pub fn required_weight_shards(local_dir: &Path) -> Result<Vec<String>, IntegrityError> {
    let index_path = local_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let metadata = std::fs::metadata(&index_path)?;
        if metadata.len() > MAX_HF_SAFETENSORS_INDEX_BYTES {
            return Err(IntegrityError::InvalidSourceManifest {
                reason: format!(
                    "{} is {} bytes; limit is {MAX_HF_SAFETENSORS_INDEX_BYTES}",
                    index_path.display(),
                    metadata.len()
                ),
            });
        }
        let raw = std::fs::read(&index_path)?;
        if raw.len() as u64 != metadata.len() {
            return Err(IntegrityError::InvalidSourceManifest {
                reason: format!("{} changed while it was read", index_path.display()),
            });
        }
        let parsed: BoundedSafetensorsIndex = serde_json::from_slice(&raw).map_err(|error| {
            IntegrityError::InvalidSourceManifest {
                reason: format!("parse {}: {error}", index_path.display()),
            }
        })?;
        if parsed.required.is_empty() {
            return Err(IntegrityError::InvalidSourceManifest {
                reason: "weight_map does not reference any safetensors shards".into(),
            });
        }
        return Ok(parsed.required.into_iter().collect());
    }
    Ok(vec!["model.safetensors".to_owned()])
}

struct BoundedSafetensorsIndex {
    required: BTreeSet<String>,
}

impl<'de> Deserialize<'de> for BoundedSafetensorsIndex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(SafetensorsIndexVisitor)
    }
}

struct SafetensorsIndexVisitor;

impl<'de> Visitor<'de> for SafetensorsIndexVisitor {
    type Value = BoundedSafetensorsIndex;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a safetensors index with metadata and one weight_map")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut metadata_seen = false;
        let mut required = None;
        while let Some(field) = map.next_key::<String>()? {
            match field.as_str() {
                "metadata" if !metadata_seen => {
                    metadata_seen = true;
                    map.next_value::<IgnoredAny>()?;
                }
                "metadata" => return Err(de::Error::duplicate_field("metadata")),
                "weight_map" if required.is_none() => {
                    required = Some(map.next_value::<BoundedWeightMap>()?.0);
                }
                "weight_map" => return Err(de::Error::duplicate_field("weight_map")),
                _ => {
                    return Err(de::Error::unknown_field(
                        &field,
                        &["metadata", "weight_map"],
                    ))
                }
            }
        }
        Ok(BoundedSafetensorsIndex {
            required: required.ok_or_else(|| de::Error::missing_field("weight_map"))?,
        })
    }
}

struct BoundedWeightMap(BTreeSet<String>);

impl<'de> Deserialize<'de> for BoundedWeightMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(WeightMapVisitor)
    }
}

struct WeightMapVisitor;

impl<'de> Visitor<'de> for WeightMapVisitor {
    type Value = BoundedWeightMap;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded tensor-to-safetensors map")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut tensors = BTreeSet::new();
        let mut shards = BTreeSet::new();
        while let Some((tensor, filename)) = map.next_entry::<String, String>()? {
            if tensors.len() == MAX_HF_SAFETENSORS_INDEX_ENTRIES {
                return Err(de::Error::custom(format_args!(
                    "weight_map exceeds {MAX_HF_SAFETENSORS_INDEX_ENTRIES} entries"
                )));
            }
            if tensor.is_empty()
                || tensor.len() > MAX_HF_TENSOR_NAME_BYTES
                || tensor
                    .bytes()
                    .any(|byte| byte == 0 || byte.is_ascii_control())
            {
                return Err(de::Error::custom("invalid bounded tensor name"));
            }
            if !tensors.insert(tensor.clone()) {
                return Err(de::Error::custom(format_args!(
                    "duplicate weight_map tensor {tensor:?}"
                )));
            }
            validate_relative_source_path(&filename).map_err(de::Error::custom)?;
            if !filename.ends_with(".safetensors") {
                return Err(de::Error::custom(format_args!(
                    "weight_map[{tensor:?}] references non-safetensors {filename:?}"
                )));
            }
            if shards.len() == MAX_VERIFIED_SOURCE_FILES && !shards.contains(&filename) {
                return Err(de::Error::custom(format_args!(
                    "weight_map references more than {MAX_VERIFIED_SOURCE_FILES} shards"
                )));
            }
            shards.insert(filename);
        }
        Ok(BoundedWeightMap(shards))
    }
}

/// Offline verification seam used by remote conversion and hermetic tests.
pub fn verify_conversion_manifest(
    repo: &str,
    revision: &str,
    local_dir: &Path,
    manifest: Vec<ShardIntegrity>,
) -> Result<VerifiedSourceManifest, IntegrityError> {
    if manifest.is_empty() || manifest.len() > MAX_VERIFIED_SOURCE_FILES {
        return Err(IntegrityError::InvalidSourceManifest {
            reason: format!(
                "verified source manifest must contain 1..={MAX_VERIFIED_SOURCE_FILES} files"
            ),
        });
    }
    if !local_dir.join("config.json").is_file() {
        return Err(IntegrityError::LocalFileMissing {
            filename: "config.json".into(),
            path: local_dir.join("config.json").display().to_string(),
        });
    }
    let mut by_name = BTreeMap::new();
    for record in &manifest {
        validate_relative_source_path(&record.filename)?;
        if by_name.insert(record.filename.as_str(), record).is_some() {
            return Err(IntegrityError::DuplicateManifestEntry {
                filename: record.filename.clone(),
            });
        }
    }
    // Authenticate every selected byte before parsing the index. The index is
    // repository-controlled structure that chooses which large payloads gain
    // conversion authority.
    for record in &manifest {
        verify_shard(repo, revision, &local_dir.join(&record.filename), record)?;
    }
    let required = required_weight_shards(local_dir)?;
    for required in ["config.json", "model.safetensors.index.json"] {
        if (required == "config.json" || local_dir.join(required).is_file())
            && !by_name.contains_key(required)
        {
            return Err(IntegrityError::RequiredShardMissing {
                filename: required.to_owned(),
            });
        }
    }
    for filename in &required {
        let record =
            by_name
                .get(filename.as_str())
                .ok_or_else(|| IntegrityError::RequiredShardMissing {
                    filename: filename.clone(),
                })?;
        let strong_lfs_identity = record.is_lfs
            && record
                .sha256
                .as_deref()
                .is_some_and(|sha| sha.len() == 64 && sha.chars().all(|c| c.is_ascii_hexdigit()));
        if !strong_lfs_identity {
            return Err(IntegrityError::RequiredShardNotLfs {
                filename: filename.clone(),
                etag: record.hf_etag.clone(),
            });
        }
    }
    let mut records = manifest;
    records.sort_by(|a, b| a.filename.cmp(&b.filename));
    Ok(VerifiedSourceManifest { records })
}

/// Fetch the live manifest at an immutable revision and verify it before
/// remote conversion opens any safetensors shard.
pub fn verify_remote_conversion_source(
    repo: &str,
    revision: &str,
    local_dir: &Path,
) -> Result<VerifiedSourceManifest, IntegrityError> {
    validate_immutable_revision(revision)?;
    let manifest = fetch_repo_shard_metadata(repo, revision, local_dir)?;
    verify_conversion_manifest(repo, revision, local_dir, manifest)
}

/// Verify exactly the bounded source inventory selected by the native
/// downloader. Unrelated files that another process placed in the shared Hub
/// snapshot cannot expand conversion authority or receipt contents.
pub fn verify_remote_conversion_files(
    repo: &str,
    revision: &str,
    local_dir: &Path,
    selected_files: &[String],
) -> Result<VerifiedSourceManifest, IntegrityError> {
    validate_immutable_revision(revision)?;
    let manifest = fetch_selected_repo_metadata(repo, revision, selected_files)?;
    verify_conversion_manifest(repo, revision, local_dir, manifest)
}

fn validate_immutable_revision(revision: &str) -> Result<(), IntegrityError> {
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(IntegrityError::InvalidSourceManifest {
            reason: format!("remote revision `{revision}` is not an exact 40-hex commit"),
        });
    }
    Ok(())
}

/// Fetch the per-shard integrity manifest for a HuggingFace repo.
///
/// Walks the local snapshot directory `local_dir`, and for every file
/// (not just `.safetensors` — also `config.json`, `tokenizer.json`, etc.)
/// queries `https://huggingface.co/<repo>/resolve/<revision>/<file>` for
/// the size + etag pair.  Network calls go through `hf-hub`'s sync `Api`,
/// which already handles redirect-follow + token resolution.
///
/// Returns one [`ShardIntegrity`] per file actually on disk under
/// `local_dir`.  Files in the HF repo that were never downloaded (e.g. the
/// optional ONNX export) are NOT included — we only verify what we have.
///
/// `revision` is typically `"main"` for the rolling tip; pass an explicit
/// commit SHA when the snapshot dir's revision is pinned.
pub fn fetch_repo_shard_metadata(
    repo: &str,
    revision: &str,
    local_dir: &Path,
) -> Result<Vec<ShardIntegrity>, IntegrityError> {
    let local_files = enumerate_local_files(local_dir)?;
    if local_files.is_empty() {
        return Ok(Vec::new());
    }
    fetch_selected_repo_metadata(repo, revision, &local_files)
}

fn fetch_selected_repo_metadata(
    repo: &str,
    revision: &str,
    selected_files: &[String],
) -> Result<Vec<ShardIntegrity>, IntegrityError> {
    use hf_hub::{Repo, RepoType};

    if selected_files.is_empty() || selected_files.len() > MAX_VERIFIED_SOURCE_FILES {
        return Err(IntegrityError::InvalidSourceManifest {
            reason: format!(
                "selected source inventory must contain 1..={MAX_VERIFIED_SOURCE_FILES} files"
            ),
        });
    }
    let mut unique = BTreeSet::new();
    for filename in selected_files {
        validate_relative_source_path(filename)?;
        if !unique.insert(filename.as_str()) {
            return Err(IntegrityError::DuplicateManifestEntry {
                filename: filename.clone(),
            });
        }
    }

    let api = ApiBuilder::new()
        .with_endpoint(CANONICAL_HF_ENDPOINT.to_owned())
        .with_token(crate::input::hf_download::resolve_auth_token())
        .build()
        .map_err(|error| IntegrityError::MetadataFetchFailed {
            repo: repo.to_owned(),
            revision: revision.to_owned(),
            reason: format!("Failed to build hf-hub API client: {error}"),
        })?;

    // Use Repo::with_revision so the URLs use the pinned revision when one
    // is supplied; this matches the snapshot layout `snapshots/<rev>/`.
    let repo_handle = api.repo(Repo::with_revision(
        repo.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let mut out = Vec::with_capacity(selected_files.len());
    for filename in selected_files {
        let url = repo_handle.url(filename);
        debug!(repo, revision, filename, "Fetching HF metadata for shard");
        let metadata = api
            .metadata(&url)
            .map_err(|e| IntegrityError::MetadataFetchFailed {
                repo: repo.to_string(),
                revision: revision.to_string(),
                reason: format!("HEAD {url}: {e}"),
            })?;
        out.push(ShardIntegrity::from_metadata(
            filename,
            metadata.etag(),
            metadata.size() as u64,
        ));
    }
    Ok(out)
}

/// Enumerate files in a local snapshot directory, ignoring hidden files
/// (e.g. `.locks/` from hf-hub) and following symlinks (hf-hub stores
/// `snapshots/<rev>/file` as a symlink into `blobs/<oid>`, so
/// `metadata().is_file()` on the symlink resolves correctly).
fn enumerate_local_files(dir: &Path) -> std::io::Result<Vec<String>> {
    let mut out = Vec::new();
    walk_dir(dir, dir, 0, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk_dir(root: &Path, dir: &Path, depth: usize, out: &mut Vec<String>) -> std::io::Result<()> {
    if depth > crate::input::hf_reference::MAX_HF_FILENAME_COMPONENTS {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Hugging Face snapshot directory depth exceeds the source bound",
        ));
    }
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for entry in read {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with('.') {
            continue; // skip .locks etc.
        }
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            walk_dir(root, &path, depth + 1, out)?;
        } else if file_type.is_file()
            || (file_type.is_symlink() && std::fs::metadata(&path)?.is_file())
        {
            if out.len() == MAX_VERIFIED_SOURCE_FILES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Hugging Face snapshot contains too many source files",
                ));
            }
            // Compute path relative to the snapshot root.
            let rel = path
                .strip_prefix(root)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| name_str.into_owned());
            out.push(rel);
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Hugging Face snapshot contains an unsupported file type",
            ));
        }
    }
    Ok(())
}

/// Combined fetch + per-shard verify across the whole repo.
///
/// Returns the verified [`ShardIntegrity`] vector on PASS so callers can
/// persist it into the cache manifest (see
/// `ModelCache::record_source_with_shards`).
///
/// Refuses with [`IntegrityError::ShardMismatch`] / [`IntegrityError::SizeMismatch`]
/// on the first failure — no "all results, then summary" mode.
pub fn verify_repo(
    repo: &str,
    revision: &str,
    local_dir: &Path,
) -> Result<Vec<ShardIntegrity>, IntegrityError> {
    info!(repo, revision, dir = %local_dir.display(), "HF integrity check");
    let shards = fetch_repo_shard_metadata(repo, revision, local_dir)?;
    let n = shards.len();
    if n == 0 {
        warn!(
            repo,
            revision,
            dir = %local_dir.display(),
            "no files enumerated for integrity check; nothing to verify"
        );
        return Ok(shards);
    }
    for s in &shards {
        let path = local_dir.join(&s.filename);
        verify_shard(repo, revision, &path, s)?;
    }
    let lfs_count = shards.iter().filter(|s| s.is_lfs).count();
    info!(
        repo,
        revision,
        files = n,
        lfs_verified = lfs_count,
        "integrity check passed"
    );
    Ok(shards)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::sha256::compute_file_sha256;
    use std::fs;
    use tempfile::TempDir;

    fn make_shard_file(tmp: &Path, name: &str, contents: &[u8]) -> std::path::PathBuf {
        let p = tmp.join(name);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&p, contents).unwrap();
        p
    }

    fn record(tmp: &Path, name: &str, lfs: bool) -> ShardIntegrity {
        let path = tmp.join(name);
        let sha = compute_file_sha256(&path).unwrap();
        let git_sha = crate::core::integrity::compute_git_blob_sha1(
            &path,
            fs::metadata(&path).unwrap().len(),
        )
        .unwrap();
        ShardIntegrity {
            filename: name.to_string(),
            bytes: fs::metadata(path).unwrap().len(),
            sha256: lfs.then_some(sha.clone()),
            hf_etag: if lfs { sha } else { git_sha },
            is_lfs: lfs,
        }
    }

    #[test]
    fn enumerate_local_files_returns_relative_paths_sorted() {
        let tmp = TempDir::new().unwrap();
        make_shard_file(tmp.path(), "config.json", b"{}");
        make_shard_file(tmp.path(), "model-00001-of-00002.safetensors", b"a");
        make_shard_file(tmp.path(), "model-00002-of-00002.safetensors", b"b");
        make_shard_file(tmp.path(), ".locks/foo.lock", b"");
        make_shard_file(tmp.path(), "subdir/extra.txt", b"x");

        let files = enumerate_local_files(tmp.path()).unwrap();
        assert_eq!(
            files,
            vec![
                "config.json".to_string(),
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string(),
                "subdir/extra.txt".to_string(),
            ]
        );
    }

    #[test]
    fn conversion_manifest_verifies_exact_index_shards_offline() {
        let tmp = TempDir::new().unwrap();
        make_shard_file(tmp.path(), "config.json", b"{}");
        make_shard_file(tmp.path(), "model-00001-of-00002.safetensors", b"first");
        make_shard_file(tmp.path(), "model-00002-of-00002.safetensors", b"second");
        make_shard_file(
            tmp.path(),
            "model.safetensors.index.json",
            br#"{"weight_map":{"a":"model-00002-of-00002.safetensors","b":"model-00001-of-00002.safetensors"}}"#,
        );
        let records = vec![
            record(tmp.path(), "model-00002-of-00002.safetensors", true),
            record(tmp.path(), "config.json", false),
            record(tmp.path(), "model.safetensors.index.json", false),
            record(tmp.path(), "model-00001-of-00002.safetensors", true),
        ];

        let verified =
            verify_conversion_manifest("org/model", &"a".repeat(40), tmp.path(), records)
                .expect("valid fixture");
        let names: Vec<_> = verified
            .records()
            .iter()
            .map(|r| r.filename.as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                "config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
            ]
        );
    }

    #[test]
    fn conversion_manifest_fails_closed_on_missing_required_shard() {
        let tmp = TempDir::new().unwrap();
        make_shard_file(tmp.path(), "config.json", b"{}");
        make_shard_file(
            tmp.path(),
            "model.safetensors.index.json",
            br#"{"weight_map":{"a":"missing.safetensors"}}"#,
        );
        let err = verify_conversion_manifest(
            "org/model",
            &"b".repeat(40),
            tmp.path(),
            vec![
                record(tmp.path(), "config.json", false),
                record(tmp.path(), "model.safetensors.index.json", false),
            ],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            IntegrityError::RequiredShardMissing { filename } if filename == "missing.safetensors"
        ));
    }

    #[test]
    fn conversion_manifest_requires_strong_lfs_identity_for_weights() {
        let tmp = TempDir::new().unwrap();
        make_shard_file(tmp.path(), "config.json", b"{}");
        make_shard_file(tmp.path(), "model.safetensors", b"weights");
        let err = verify_conversion_manifest(
            "org/model",
            &"c".repeat(40),
            tmp.path(),
            vec![
                record(tmp.path(), "config.json", false),
                record(tmp.path(), "model.safetensors", false),
            ],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            IntegrityError::RequiredShardNotLfs { filename, .. } if filename == "model.safetensors"
        ));
    }

    #[test]
    fn conversion_manifest_rejects_index_traversal() {
        let tmp = TempDir::new().unwrap();
        make_shard_file(tmp.path(), "config.json", b"{}");
        make_shard_file(
            tmp.path(),
            "model.safetensors.index.json",
            br#"{"weight_map":{"a":"../escape.safetensors"}}"#,
        );
        let err = required_weight_shards(tmp.path()).unwrap_err();
        assert!(matches!(err, IntegrityError::InvalidSourceManifest { .. }));
    }

    #[test]
    fn safetensors_index_is_size_bounded_before_json_parsing() {
        let tmp = TempDir::new().unwrap();
        let index = tmp.path().join("model.safetensors.index.json");
        let mut at_cap = br#"{"weight_map":{"a":"model.safetensors"}}"#.to_vec();
        at_cap.resize(MAX_HF_SAFETENSORS_INDEX_BYTES as usize, b' ');
        std::fs::write(&index, at_cap).unwrap();
        assert_eq!(
            required_weight_shards(tmp.path()).unwrap(),
            vec!["model.safetensors"]
        );

        let file = std::fs::File::create(&index).unwrap();
        file.set_len(MAX_HF_SAFETENSORS_INDEX_BYTES + 1).unwrap();
        let err = required_weight_shards(tmp.path()).unwrap_err();
        assert!(matches!(err, IntegrityError::InvalidSourceManifest { .. }));
        assert!(err.to_string().contains("limit"));
    }

    #[test]
    fn safetensors_index_rejects_duplicate_and_unknown_structure() {
        let tmp = TempDir::new().unwrap();
        let index = tmp.path().join("model.safetensors.index.json");
        for hostile in [
            br#"{"weight_map":{"a":"one.safetensors","a":"two.safetensors"}}"#.as_slice(),
            br#"{"weight_map":{"a":"one.safetensors"},"weight_map":{"b":"two.safetensors"}}"#,
            br#"{"weight_map":{"a":"one.safetensors"},"unexpected":true}"#,
            br#"{"weight_map":{"a":7}}"#,
        ] {
            std::fs::write(&index, hostile).unwrap();
            assert!(
                required_weight_shards(tmp.path()).is_err(),
                "accepted hostile index: {}",
                String::from_utf8_lossy(hostile)
            );
        }
    }

    #[test]
    fn safetensors_index_entry_cap_is_exact() {
        let tmp = TempDir::new().unwrap();
        let index = tmp.path().join("model.safetensors.index.json");
        let build = |count: usize| {
            let mut raw = String::from("{\"weight_map\":{");
            for entry in 0..count {
                if entry != 0 {
                    raw.push(',');
                }
                use std::fmt::Write as _;
                write!(raw, "\"t{entry}\":\"model.safetensors\"").unwrap();
            }
            raw.push_str("}}");
            raw
        };
        let at_cap = build(MAX_HF_SAFETENSORS_INDEX_ENTRIES);
        assert!((at_cap.len() as u64) < MAX_HF_SAFETENSORS_INDEX_BYTES);
        std::fs::write(&index, at_cap).unwrap();
        assert_eq!(
            required_weight_shards(tmp.path()).unwrap(),
            vec!["model.safetensors"]
        );

        std::fs::write(&index, build(MAX_HF_SAFETENSORS_INDEX_ENTRIES + 1)).unwrap();
        assert!(required_weight_shards(tmp.path()).is_err());
    }

    #[test]
    fn remote_verifier_rejects_mutable_revision_before_network() {
        let tmp = TempDir::new().unwrap();
        let err = verify_remote_conversion_source("org/model", "main", tmp.path()).unwrap_err();
        assert!(matches!(err, IntegrityError::InvalidSourceManifest { .. }));
    }

    #[test]
    fn selected_remote_inventory_is_nonempty_bounded_and_duplicate_free_before_network() {
        let tmp = TempDir::new().unwrap();
        let revision = "d".repeat(40);
        assert!(matches!(
            verify_remote_conversion_files("org/model", &revision, tmp.path(), &[]),
            Err(IntegrityError::InvalidSourceManifest { .. })
        ));
        assert!(matches!(
            verify_remote_conversion_files(
                "org/model",
                &revision,
                tmp.path(),
                &["config.json".to_owned(), "config.json".to_owned()],
            ),
            Err(IntegrityError::DuplicateManifestEntry { .. })
        ));
        let over_cap = (0..=MAX_VERIFIED_SOURCE_FILES)
            .map(|index| format!("file-{index}.json"))
            .collect::<Vec<_>>();
        assert!(matches!(
            verify_remote_conversion_files("org/model", &revision, tmp.path(), &over_cap),
            Err(IntegrityError::InvalidSourceManifest { .. })
        ));
    }

    #[test]
    fn conversion_manifest_requires_selected_config_and_index_identity() {
        let tmp = TempDir::new().unwrap();
        make_shard_file(tmp.path(), "config.json", b"{}");
        make_shard_file(tmp.path(), "model.safetensors", b"weights");
        let missing_config = verify_conversion_manifest(
            "org/model",
            &"e".repeat(40),
            tmp.path(),
            vec![record(tmp.path(), "model.safetensors", true)],
        )
        .unwrap_err();
        assert!(matches!(
            missing_config,
            IntegrityError::RequiredShardMissing { filename } if filename == "config.json"
        ));

        make_shard_file(
            tmp.path(),
            "model.safetensors.index.json",
            br#"{"weight_map":{"a":"model.safetensors"}}"#,
        );
        let missing_index = verify_conversion_manifest(
            "org/model",
            &"e".repeat(40),
            tmp.path(),
            vec![
                record(tmp.path(), "config.json", false),
                record(tmp.path(), "model.safetensors", true),
            ],
        )
        .unwrap_err();
        assert!(matches!(
            missing_index,
            IntegrityError::RequiredShardMissing { filename }
                if filename == "model.safetensors.index.json"
        ));
    }

    #[test]
    fn verify_repo_empty_dir_returns_empty_vec_and_warns() {
        let tmp = TempDir::new().unwrap();
        let shards = verify_repo("org/empty", "main", tmp.path()).unwrap();
        assert!(shards.is_empty());
    }

    /// Network-gated: real verification against a small public HF repo.
    /// Skipped unless `HF2Q_NETWORK_TESTS=1` is set so default `cargo test`
    /// runs are hermetic.
    #[test]
    fn verify_repo_network_smoke() {
        if std::env::var("HF2Q_NETWORK_TESTS").ok().as_deref() != Some("1") {
            eprintln!("skipping network test (set HF2Q_NETWORK_TESTS=1 to run)");
            return;
        }
        let api = ApiBuilder::new()
            .with_endpoint(CANONICAL_HF_ENDPOINT.to_owned())
            .build()
            .expect("build exact-origin Hub client");
        use hf_hub::Repo;
        let repo = api.repo(Repo::with_revision(
            "hf-internal-testing/tiny-random-bert".into(),
            hf_hub::RepoType::Model,
            "main".into(),
        ));
        let url = repo.url("config.json");
        let metadata = api.metadata(&url).expect("fetch config metadata");
        assert!(metadata.size() > 0);
        let s =
            ShardIntegrity::from_metadata("config.json", metadata.etag(), metadata.size() as u64);
        // config.json is a small Git-managed file, not LFS.
        assert!(!s.is_lfs);
        assert_eq!(s.hf_etag.len(), 40);
    }
}
