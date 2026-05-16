//! Generic file SHA-256 helpers — 1-MiB-streaming hasher used by every
//! caller in the convert + serve pipelines (cache manifest, integrity
//! verification, provenance source-bundle hashing).
//!
//! Migrated 2026-05-16 from `src/serve/cache.rs` as part of the v0.1.0
//! workspace split (B1.2).  The old paths
//! `crate::serve::cache::{compute_file_sha256, sha256_file}` remain as
//! `#[deprecated]` re-export shims until the workspace split removes
//! them.
//!
//! Two surfaces:
//!
//! - [`compute_file_sha256`] — the canonical implementation; returns
//!   `std::io::Result<String>` so the OS-level failure surface stays
//!   visible to callers that want to handle it directly.
//! - [`sha256_file`] — `anyhow::Result`-shaped wrapper that adds an
//!   `open: <path>` context.  Kept for back-compat with the
//!   pre-iter-211 callsites that already use `anyhow::Result`.  New
//!   code SHOULD prefer [`compute_file_sha256`].

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Streaming SHA-256 of a file's contents as lowercase hex.
///
/// 1 MiB read window — chosen so the hash cost stays I/O-bound on a
/// stock M-series machine (Sha256 throughput ~ 600 MB/s; APFS read
/// throughput ~ 4 GB/s).  Smaller windows pay more per-syscall overhead;
/// larger windows waste memory without throughput gains.
///
/// Returns `std::io::Result<String>` so callers can match on the
/// specific OS-level error class (e.g. `NotFound`, `PermissionDenied`)
/// without parsing strings.  See [`sha256_file`] for the
/// `anyhow::Result` wrapper.
///
/// ADR-005 Phase 4 iter-211 — added as the canonical 1-MiB-streaming
/// hasher; [`sha256_file`] now delegates here so all callers (cache,
/// auto-pipeline, Phase 3 integrity) share one implementation.
pub fn compute_file_sha256(path: &Path) -> std::io::Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// SHA-256 of a file as lowercase hex.  Used by `QuantEntry` writers.
/// Public so iter-203 (integrity check) can reuse it without re-creating
/// yet another sha256 helper.
///
/// Thin `anyhow::Result`-shaped wrapper around [`compute_file_sha256`]
/// that adds the `open: <path>` context expected by every existing
/// caller.  Kept for back-compat with iter-203 / iter-205 callsites —
/// new code should prefer [`compute_file_sha256`] when `io::Result` is
/// sufficient.
pub fn sha256_file(path: &Path) -> Result<String> {
    compute_file_sha256(path).with_context(|| format!("open: {}", path.display()))
}
