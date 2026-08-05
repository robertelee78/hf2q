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
use std::path::{Component, Path};

use hf_hub::api::sync::ApiBuilder;
use tracing::{debug, info, warn};

use crate::core::integrity::{verify_shard, IntegrityError, ShardIntegrity};

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
}

fn validate_relative_source_path(filename: &str) -> Result<(), IntegrityError> {
    let path = Path::new(filename);
    if filename.is_empty()
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
    let mut required = BTreeSet::new();
    if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value =
            serde_json::from_str(&raw).map_err(|error| IntegrityError::InvalidSourceManifest {
                reason: format!("parse {}: {error}", index_path.display()),
            })?;
        let weight_map = index
            .get("weight_map")
            .and_then(|value| value.as_object())
            .ok_or_else(|| IntegrityError::InvalidSourceManifest {
                reason: format!("{} has no `weight_map` object", index_path.display()),
            })?;
        for (tensor, value) in weight_map {
            let filename = value
                .as_str()
                .ok_or_else(|| IntegrityError::InvalidSourceManifest {
                    reason: format!("weight_map[{tensor:?}] is not a string"),
                })?;
            validate_relative_source_path(filename)?;
            if !filename.ends_with(".safetensors") {
                return Err(IntegrityError::InvalidSourceManifest {
                    reason: format!(
                        "weight_map[{tensor:?}] references non-safetensors `{filename}`"
                    ),
                });
            }
            required.insert(filename.to_string());
        }
    } else {
        required.insert("model.safetensors".to_string());
    }
    if required.is_empty() {
        return Err(IntegrityError::InvalidSourceManifest {
            reason: "weight_map does not reference any safetensors shards".into(),
        });
    }
    Ok(required.into_iter().collect())
}

/// Offline verification seam used by remote conversion and hermetic tests.
pub fn verify_conversion_manifest(
    repo: &str,
    revision: &str,
    local_dir: &Path,
    manifest: Vec<ShardIntegrity>,
) -> Result<VerifiedSourceManifest, IntegrityError> {
    if !local_dir.join("config.json").is_file() {
        return Err(IntegrityError::LocalFileMissing {
            filename: "config.json".into(),
            path: local_dir.join("config.json").display().to_string(),
        });
    }
    let required = required_weight_shards(local_dir)?;
    let mut by_name = BTreeMap::new();
    for record in &manifest {
        validate_relative_source_path(&record.filename)?;
        if by_name.insert(record.filename.as_str(), record).is_some() {
            return Err(IntegrityError::DuplicateManifestEntry {
                filename: record.filename.clone(),
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
    for record in &manifest {
        verify_shard(repo, revision, &local_dir.join(&record.filename), record)?;
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
    if revision.len() != 40 || !revision.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(IntegrityError::InvalidSourceManifest {
            reason: format!("remote revision `{revision}` is not an exact 40-hex commit"),
        });
    }
    let manifest = fetch_repo_shard_metadata(repo, revision, local_dir)?;
    verify_conversion_manifest(repo, revision, local_dir, manifest)
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
    use hf_hub::{Repo, RepoType};

    let api = ApiBuilder::new()
        .build()
        .map_err(|e| IntegrityError::MetadataFetchFailed {
            repo: repo.to_string(),
            revision: revision.to_string(),
            reason: format!("Failed to build hf-hub API client: {e}"),
        })?;

    // Use Repo::with_revision so the URLs use the pinned revision when one
    // is supplied; this matches the snapshot layout `snapshots/<rev>/`.
    let repo_handle = api.repo(Repo::with_revision(
        repo.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let local_files = enumerate_local_files(local_dir)?;
    let mut out = Vec::with_capacity(local_files.len());
    for filename in &local_files {
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
    walk_dir(dir, dir, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk_dir(root: &Path, dir: &Path, out: &mut Vec<String>) -> std::io::Result<()> {
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
        let md = std::fs::metadata(&path)?;
        if md.is_dir() {
            walk_dir(root, &path, out)?;
        } else if md.is_file() {
            // Compute path relative to the snapshot root.
            let rel = path
                .strip_prefix(root)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| name_str.into_owned());
            out.push(rel);
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
        ShardIntegrity {
            filename: name.to_string(),
            bytes: fs::metadata(path).unwrap().len(),
            sha256: lfs.then_some(sha.clone()),
            hf_etag: if lfs { sha } else { "git-etag".into() },
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
            vec![record(tmp.path(), "config.json", false)],
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
    fn remote_verifier_rejects_mutable_revision_before_network() {
        let tmp = TempDir::new().unwrap();
        let err = verify_remote_conversion_source("org/model", "main", tmp.path()).unwrap_err();
        assert!(matches!(err, IntegrityError::InvalidSourceManifest { .. }));
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
        if let Ok(api) = ApiBuilder::new().build() {
            use hf_hub::Repo;
            let repo = api.repo(Repo::with_revision(
                "hf-internal-testing/tiny-random-bert".into(),
                hf_hub::RepoType::Model,
                "main".into(),
            ));
            let url = repo.url("config.json");
            if let Ok(metadata) = api.metadata(&url) {
                assert!(metadata.size() > 0);
                let s = ShardIntegrity::from_metadata(
                    "config.json",
                    metadata.etag(),
                    metadata.size() as u64,
                );
                // config.json is a small text file → non-LFS → no sha256.
                assert!(!s.is_lfs);
            }
        }
    }
}
