//! HuggingFace shard integrity types + offline per-shard verification.
//!
//! Migrated 2026-05-16 from `src/input/integrity.rs` as part of the
//! v0.1.0 workspace split (B1.3).  The HTTP-bound pieces
//! (`fetch_repo_shard_metadata`, `verify_repo`, the local-snapshot
//! walker) stay in `src/input/integrity.rs` because they depend on
//! `hf-hub`'s `ApiBuilder` — that's an HF-download concern that lives
//! on the convert side of the workspace split.
//!
//! What lives here (post-B1.3):
//!
//! - [`IntegrityError`] — the error enum that callers
//!   (`hf2q convert`, `hf2q serve`, the auto-pipeline) match on to
//!   produce refuse-to-proceed diagnostics.
//! - [`ShardIntegrity`] — the per-shard integrity record persisted
//!   into cache manifests and Source-bundle SHA-256 inputs.
//! - [`ShardIntegrity::from_metadata`] — adapter from an
//!   `hf-hub::Metadata` etag (kept here so cache-side callers don't
//!   need to pull in `hf-hub` types).
//! - [`verify_shard`] — pure file-system + crypto check (file exists,
//!   size matches, sha256 matches for LFS files).
//! - [`shard_path`] — `local_dir.join(filename)` helper.
//!
//! What stays in `src/input/integrity.rs`:
//!
//! - `fetch_repo_shard_metadata` — issues HEAD to HF Hub via
//!   `hf-hub::Api::metadata`.
//! - `verify_repo` — convenience wrapper that calls fetch then
//!   `verify_shard` for every record.  Lives next to fetch because
//!   the two share the HTTP / token-resolution stack.
//! - `enumerate_local_files` / `walk_dir` — snapshot-directory
//!   walker, used by fetch.
//!
//! # Failure semantics
//!
//! Per `feedback_no_shortcuts.md`, integrity verification is on by
//! default; `--no-integrity` is an operator override for development
//! workflows + air-gapped setups.  `verify_shard` fails fast on the
//! first byte-mismatch — there is no "summary mode" that prints all
//! mismatches and continues, because corruption is a refuse-to-proceed
//! event, not a quality-of-life issue.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::debug;

use crate::core::sha256::compute_file_sha256;

/// Errors from integrity-check operations.
#[derive(Error, Debug)]
pub enum IntegrityError {
    #[error("Failed to query HuggingFace metadata for {repo}@{revision}: {reason}")]
    MetadataFetchFailed {
        repo: String,
        revision: String,
        reason: String,
    },

    #[error(
        "Local file missing during integrity check: shard '{filename}' \
         expected at {path}"
    )]
    LocalFileMissing { filename: String, path: String },

    /// Strong-error: a byte-level integrity mismatch.  Message wording is
    /// load-bearing — refuse-to-proceed callers (convert / serve) print
    /// this verbatim, and tests assert against the field names.
    #[error(
        "Integrity check failed for shard '{filename}' \
         (repo {repo}@{revision}): \
         expected SHA-256 {expected}, computed {actual}. \
         The downloaded file does not match HuggingFace's recorded hash. \
         Possible causes: corrupted download, MITM, or the source repo \
         was force-pushed since the last cache. \
         Re-run after `rm -rf {local_path}` to refetch, or pass \
         --no-integrity to skip (NOT recommended)."
    )]
    ShardMismatch {
        repo: String,
        revision: String,
        filename: String,
        expected: String,
        actual: String,
        local_path: String,
    },

    #[error(
        "Integrity check failed for shard '{filename}': \
         expected size {expected_bytes} bytes, file on disk is {actual_bytes} bytes. \
         File is truncated or has trailing data."
    )]
    SizeMismatch {
        filename: String,
        expected_bytes: u64,
        actual_bytes: u64,
    },

    #[error("I/O error during integrity check: {0}")]
    Io(#[from] std::io::Error),
}

/// Per-shard integrity record.  Persisted into the cache manifest by
/// `ModelCache::record_source_with_shards` so that future serve-time
/// loads can verify without re-fetching HF metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShardIntegrity {
    /// Shard filename relative to the snapshot directory
    /// (e.g. `model-00001-of-00009.safetensors`, `config.json`).
    pub filename: String,
    /// Authoritative size in bytes per HF's metadata endpoint.
    pub bytes: u64,
    /// SHA-256 (lowercase hex) of the LFS object, when HuggingFace returns
    /// `x-linked-etag` for the file.  `None` for non-LFS files
    /// (small JSON / text where HF's etag is a Git blob SHA-1, not
    /// directly comparable to a file SHA-256).
    pub sha256: Option<String>,
    /// Raw etag as returned by HF (LFS sha256 hex when LFS, Git-style
    /// SHA-1 of `blob <size>\0<contents>` otherwise).  Stored verbatim
    /// for traceability; equality checks use [`Self::sha256`].
    pub hf_etag: String,
    /// `true` iff the etag came from `x-linked-etag` (i.e. the file is
    /// LFS-managed and the etag IS the file SHA-256).
    pub is_lfs: bool,
}

impl ShardIntegrity {
    /// Build from an `hf-hub` `Metadata` plus filename.  The
    /// crate's `Metadata::etag()` already prefers `x-linked-etag` over
    /// plain `etag` — but it doesn't tell us *which* one it picked.  We
    /// re-derive `is_lfs` from the etag's shape: an x-linked-etag is a
    /// 64-character lowercase hex SHA-256; a Git blob etag is a 40-char
    /// SHA-1.  Anything else (or a quoted etag with internal punctuation)
    /// is treated as non-LFS.
    pub fn from_metadata(filename: &str, etag: &str, size: u64) -> Self {
        let trimmed = etag.trim().trim_matches('"');
        let is_lfs = trimmed.len() == 64 && trimmed.chars().all(|c| c.is_ascii_hexdigit());
        Self {
            filename: filename.to_string(),
            bytes: size,
            sha256: if is_lfs {
                Some(trimmed.to_lowercase())
            } else {
                None
            },
            hf_etag: trimmed.to_string(),
            is_lfs,
        }
    }
}

/// Verify a single local file against an expected [`ShardIntegrity`]
/// record.
///
/// Order of checks (cheapest-first; fail fast):
///
/// 1. File exists.
/// 2. File size matches `expected.bytes`.
/// 3. *(LFS only)* SHA-256 of the file's bytes matches `expected.sha256`.
///
/// Step 3 is skipped when `expected.is_lfs == false` — HF's plain etag is
/// a Git blob SHA-1 (`SHA1("blob <size>\0<contents>")`) that is not
/// directly comparable to a file SHA-256.
pub fn verify_shard(
    repo: &str,
    revision: &str,
    local_path: &Path,
    expected: &ShardIntegrity,
) -> Result<(), IntegrityError> {
    let metadata = match std::fs::metadata(local_path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(IntegrityError::LocalFileMissing {
                filename: expected.filename.clone(),
                path: local_path.display().to_string(),
            });
        }
        Err(e) => return Err(IntegrityError::Io(e)),
    };

    let actual_bytes = metadata.len();
    if actual_bytes != expected.bytes {
        return Err(IntegrityError::SizeMismatch {
            filename: expected.filename.clone(),
            expected_bytes: expected.bytes,
            actual_bytes,
        });
    }

    // Non-LFS files (config.json, tokenizer.json) carry a Git-blob SHA-1
    // etag that doesn't match a file SHA-256 — record the size-only PASS
    // and move on.
    let Some(expected_sha) = expected.sha256.as_ref() else {
        debug!(
            filename = %expected.filename,
            "size match (non-LFS file; no sha256 verification possible)"
        );
        return Ok(());
    };

    let actual_sha = compute_file_sha256(local_path)?;
    if actual_sha.eq_ignore_ascii_case(expected_sha) {
        debug!(filename = %expected.filename, "sha256 match");
        Ok(())
    } else {
        Err(IntegrityError::ShardMismatch {
            repo: repo.to_string(),
            revision: revision.to_string(),
            filename: expected.filename.clone(),
            expected: expected_sha.clone(),
            actual: actual_sha,
            local_path: local_path.display().to_string(),
        })
    }
}

/// Build a [`PathBuf`] for a single shard within a snapshot dir.  Public
/// because callers (e.g. ADR-014 iter-204 streaming convert) need to
/// resolve shard paths consistently across the convert + serve halves.
#[inline]
pub fn shard_path(local_dir: &Path, filename: &str) -> PathBuf {
    local_dir.join(filename)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::fs;
    use tempfile::TempDir;

    // ── ShardIntegrity::from_metadata ────────────────────────────────────

    #[test]
    fn from_metadata_lfs_etag_marks_lfs_and_records_sha256() {
        let etag = "f9343d7d7ec5c3d8bcced056c438fc9f1d3819e9ca3d42418a40857050e10e20";
        let s = ShardIntegrity::from_metadata("model.safetensors", etag, 12345);
        assert!(s.is_lfs);
        assert_eq!(s.sha256.as_deref(), Some(etag));
        assert_eq!(s.hf_etag, etag);
        assert_eq!(s.bytes, 12345);
    }

    #[test]
    fn from_metadata_lfs_etag_uppercase_normalized_to_lowercase() {
        let etag = "F9343D7D7EC5C3D8BCCED056C438FC9F1D3819E9CA3D42418A40857050E10E20";
        let s = ShardIntegrity::from_metadata("model.safetensors", etag, 1);
        assert!(s.is_lfs);
        assert_eq!(
            s.sha256.as_deref(),
            Some("f9343d7d7ec5c3d8bcced056c438fc9f1d3819e9ca3d42418a40857050e10e20")
        );
    }

    #[test]
    fn from_metadata_quoted_etag_unwrapped() {
        let etag = "\"f9343d7d7ec5c3d8bcced056c438fc9f1d3819e9ca3d42418a40857050e10e20\"";
        let s = ShardIntegrity::from_metadata("model.safetensors", etag, 1);
        assert!(s.is_lfs);
    }

    #[test]
    fn from_metadata_git_blob_etag_marks_non_lfs() {
        let etag = "0123456789abcdef0123456789abcdef01234567";
        let s = ShardIntegrity::from_metadata("config.json", etag, 1024);
        assert!(!s.is_lfs);
        assert!(s.sha256.is_none());
        assert_eq!(s.hf_etag, etag);
    }

    #[test]
    fn from_metadata_garbage_etag_marks_non_lfs() {
        let s = ShardIntegrity::from_metadata("foo", "not-a-hash", 0);
        assert!(!s.is_lfs);
        assert!(s.sha256.is_none());
    }

    // ── verify_shard ─────────────────────────────────────────────────────

    fn make_shard_file(tmp: &Path, name: &str, contents: &[u8]) -> PathBuf {
        let p = tmp.join(name);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&p, contents).unwrap();
        p
    }

    fn lfs_record(name: &str, contents: &[u8]) -> ShardIntegrity {
        let mut h = Sha256::new();
        h.update(contents);
        let sha = hex::encode(h.finalize());
        ShardIntegrity::from_metadata(name, &sha, contents.len() as u64)
    }

    #[test]
    fn verify_shard_pass_lfs() {
        let tmp = TempDir::new().unwrap();
        let contents = b"the actual safetensors bytes";
        let path = make_shard_file(tmp.path(), "model-00001.safetensors", contents);
        let expected = lfs_record("model-00001.safetensors", contents);
        verify_shard("org/repo", "main", &path, &expected).expect("hash matches");
    }

    #[test]
    fn verify_shard_fail_sha_mismatch_names_filename_and_hashes() {
        let tmp = TempDir::new().unwrap();
        let contents = b"good bytes";
        let path = make_shard_file(tmp.path(), "model.safetensors", contents);
        let bad = b"different bytes that hash differently";
        let expected = lfs_record("model.safetensors", bad);
        let expected = ShardIntegrity {
            bytes: contents.len() as u64,
            ..expected
        };
        let err = verify_shard("org/repo", "main", &path, &expected)
            .expect_err("should mismatch");
        let msg = format!("{err}");
        assert!(msg.contains("model.safetensors"), "msg: {msg}");
        assert!(msg.contains("expected SHA-256"), "msg: {msg}");
        assert!(msg.contains("--no-integrity"), "msg: {msg}");
        assert!(matches!(err, IntegrityError::ShardMismatch { .. }));
    }

    #[test]
    fn verify_shard_fail_size_mismatch_short_circuits_before_hashing() {
        let tmp = TempDir::new().unwrap();
        let contents = b"only 12 bytes";
        let path = make_shard_file(tmp.path(), "model.safetensors", contents);
        let expected = ShardIntegrity {
            filename: "model.safetensors".into(),
            bytes: 9999,
            sha256: Some("0".repeat(64)),
            hf_etag: "0".repeat(64),
            is_lfs: true,
        };
        let err = verify_shard("org/repo", "main", &path, &expected)
            .expect_err("size mismatch");
        assert!(matches!(err, IntegrityError::SizeMismatch { .. }));
        let msg = format!("{err}");
        assert!(msg.contains("9999"), "msg: {msg}");
        assert!(msg.contains("13"), "msg: {msg}");
    }

    #[test]
    fn verify_shard_missing_file_named_in_error() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("nope.safetensors");
        let expected = ShardIntegrity {
            filename: "nope.safetensors".into(),
            bytes: 100,
            sha256: Some("0".repeat(64)),
            hf_etag: "0".repeat(64),
            is_lfs: true,
        };
        let err = verify_shard("org/repo", "main", &path, &expected)
            .expect_err("missing");
        assert!(matches!(err, IntegrityError::LocalFileMissing { .. }));
        let msg = format!("{err}");
        assert!(msg.contains("nope.safetensors"), "msg: {msg}");
    }

    #[test]
    fn verify_shard_non_lfs_skips_hash_after_size_match() {
        let tmp = TempDir::new().unwrap();
        let contents = br#"{"hidden_size": 4096}"#;
        let path = make_shard_file(tmp.path(), "config.json", contents);
        let expected = ShardIntegrity {
            filename: "config.json".into(),
            bytes: contents.len() as u64,
            sha256: None,
            hf_etag: "0123456789abcdef0123456789abcdef01234567".into(),
            is_lfs: false,
        };
        verify_shard("org/repo", "main", &path, &expected)
            .expect("non-LFS files only need size to match");
    }
}
