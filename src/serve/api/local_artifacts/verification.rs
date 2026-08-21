use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use super::{is_hex, quant_from_file_type};
use crate::serve::quant_select::QuantType;

#[derive(Debug)]
pub struct LocalVerificationRequest<'a> {
    pub root: &'a Path,
    pub artifact: &'a Path,
    pub bytes: u64,
    pub sha256: &'a str,
    pub quant: QuantType,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct LocalVerificationReceipt {
    pub path: PathBuf,
}

/// Direct-child verification entry point. The server supplies every field
/// from its retained opaque authority; chat clients cannot submit paths or
/// weaken this check.
pub fn verify_local_artifact(
    request: LocalVerificationRequest<'_>,
) -> Result<LocalVerificationReceipt> {
    if !is_hex(request.sha256, 64) {
        bail!("expected local artifact SHA-256 is malformed");
    }
    let root_meta = fs::symlink_metadata(request.root).context("inspect verification root")?;
    if root_meta.file_type().is_symlink() || !root_meta.is_dir() {
        bail!("verification root is not a non-symlink directory");
    }
    let canonical_root = request
        .root
        .canonicalize()
        .context("canonicalize verification root")?;
    let before = fs::symlink_metadata(request.artifact).context("inspect local artifact")?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() != request.bytes {
        bail!("local artifact type or size changed after cataloging");
    }
    let canonical = request
        .artifact
        .canonicalize()
        .context("canonicalize local artifact")?;
    if !canonical.starts_with(&canonical_root) {
        bail!("local artifact escaped its configured root");
    }
    let actual_sha = crate::core::sha256::compute_file_sha256(&canonical)
        .context("hash selected local artifact")?;
    if !actual_sha.eq_ignore_ascii_case(request.sha256) {
        bail!("selected local artifact SHA-256 no longer matches its hf2q authority");
    }
    let header = mlx_native::gguf::GgufFile::open(&canonical)
        .context("open selected local GGUF after hashing")?;
    if header
        .metadata_u32("general.file_type")
        .and_then(quant_from_file_type)
        != Some(request.quant)
    {
        bail!("selected local artifact GGUF quant no longer matches its hf2q authority");
    }
    let after = fs::symlink_metadata(&canonical).context("re-stat selected local artifact")?;
    if !same_file_snapshot(&before, &after) || after.len() != request.bytes {
        bail!("selected local artifact changed while it was being verified");
    }
    Ok(LocalVerificationReceipt { path: canonical })
}

#[cfg(unix)]
fn same_file_snapshot(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;

    left.dev() == right.dev()
        && left.ino() == right.ino()
        && left.len() == right.len()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
}

#[cfg(not(unix))]
fn same_file_snapshot(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.len() == right.len() && left.modified().ok() == right.modified().ok()
}
