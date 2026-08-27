use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use super::{is_hex, quant_from_file_type};
use crate::serve::multi_model::{ArtifactFileStamp, TextArtifactIdentity};
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
    pub sha256: String,
    pub file_stamp: ArtifactFileStamp,
}

impl LocalVerificationReceipt {
    pub fn text_identity(&self) -> TextArtifactIdentity {
        TextArtifactIdentity {
            sha256: self.sha256.clone(),
            stamp: self.file_stamp.clone(),
        }
    }
}

pub(crate) struct VerifiedLocalArtifact {
    pub(crate) receipt: LocalVerificationReceipt,
    pub(crate) retained: crate::core::bounded_file::StableRegularFile,
}

/// Direct-child verification entry point. The server supplies every field
/// from its retained opaque authority; chat clients cannot submit paths or
/// weaken this check.
pub fn verify_local_artifact(
    request: LocalVerificationRequest<'_>,
) -> Result<LocalVerificationReceipt> {
    let opened =
        crate::core::bounded_file::StableRegularFile::open_exact(request.artifact, request.bytes)?
            .context("local artifact type or size changed after cataloging")?;
    Ok(verify_retained_local_artifact(request, opened)?.receipt)
}

pub(crate) fn verify_retained_local_artifact(
    request: LocalVerificationRequest<'_>,
    opened: crate::core::bounded_file::StableRegularFile,
) -> Result<VerifiedLocalArtifact> {
    verify_retained_local_artifact_with_progress(request, opened, |_| {})
}

pub(crate) fn verify_retained_local_artifact_with_progress(
    request: LocalVerificationRequest<'_>,
    mut opened: crate::core::bounded_file::StableRegularFile,
    mut progress: impl FnMut(u64),
) -> Result<VerifiedLocalArtifact> {
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
    let canonical = request
        .artifact
        .canonicalize()
        .context("canonicalize local artifact")?;
    if !canonical.starts_with(&canonical_root) {
        bail!("local artifact escaped its configured root");
    }
    let before_stamp = ArtifactFileStamp::inspect(&canonical)?;
    if before_stamp.bytes() != request.bytes {
        bail!("local artifact size changed before content verification");
    }
    let actual_sha = opened
        .sha256_with_progress(&mut progress)
        .context("hash selected local artifact")?
        .context("selected local artifact changed while it was being hashed")?;
    if !actual_sha.eq_ignore_ascii_case(request.sha256) {
        bail!("selected local artifact SHA-256 no longer matches its hf2q authority");
    }
    let header = mlx_native::gguf::GgufFile::from_file(opened.try_clone()?)
        .context("open selected local GGUF after hashing")?;
    if header
        .metadata_u32("general.file_type")
        .and_then(quant_from_file_type)
        != Some(request.quant)
    {
        bail!("selected local artifact GGUF quant no longer matches its hf2q authority");
    }
    let after_stamp = ArtifactFileStamp::inspect(&canonical)?;
    if before_stamp != after_stamp {
        bail!("selected local artifact changed while it was being verified");
    }
    if !opened.is_stable()? {
        bail!("selected local artifact changed while it was being verified");
    }
    Ok(VerifiedLocalArtifact {
        receipt: LocalVerificationReceipt {
            path: canonical,
            sha256: actual_sha,
            file_stamp: after_stamp,
        },
        retained: opened,
    })
}
