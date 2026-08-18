//! Exact release-archive preparation before filesystem publication.
//!
//! This bounded context begins with a listing-first, exact-inventory ZIP
//! verifier. It deliberately owns no extraction, codesign, installed marker,
//! receipt, prepared-version, activation, installer, or CLI authority yet.

mod archive;
mod deflate;

use std::io::{Read, Seek};

use crate::distribution::install_state::ArtifactStageError;
use crate::distribution::install_state::VerifiedArchiveFile;
use crate::distribution::update_transport::VerifiedReleaseBundle;

#[derive(Debug, thiserror::Error)]
pub(super) enum PreparedReleaseError {
    #[error("the authenticated release archive is outside the supported ZIP profile")]
    ArchiveProfile,
    #[error("the authenticated release archive could not be read completely")]
    ArchiveRead,
    #[error("the authenticated release archive changed after download")]
    ArchiveIntegrity(#[from] ArtifactStageError),
}

/// Exact archive/manifest agreement without extraction or install authority.
///
/// The wrapper is intentionally non-cloneable and non-serializable. Future
/// preparation work may consume it only after adding lock-held freshness,
/// descriptor-relative extraction, and macOS signing verification.
pub(super) struct ArchiveBoundRelease {
    bundle: VerifiedReleaseBundle,
}

impl std::fmt::Debug for ArchiveBoundRelease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ArchiveBoundRelease")
            .finish_non_exhaustive()
    }
}

pub(super) fn bind_archive(
    mut bundle: VerifiedReleaseBundle,
) -> Result<ArchiveBoundRelease, PreparedReleaseError> {
    let (manifest_bytes, manifest, archive_file) = bundle.preparation_parts_mut();
    validate_archive_reader(archive_file, manifest_bytes, manifest)?;
    Ok(ArchiveBoundRelease { bundle })
}

trait ArchiveIntegrity {
    fn revalidate_for_preparation(&mut self) -> Result<(), PreparedReleaseError>;
}

impl ArchiveIntegrity for VerifiedArchiveFile {
    fn revalidate_for_preparation(&mut self) -> Result<(), PreparedReleaseError> {
        self.revalidate().map_err(PreparedReleaseError::from)
    }
}

fn validate_archive_reader<A: Read + Seek + ArchiveIntegrity>(
    archive_file: &mut A,
    manifest_bytes: &[u8],
    manifest: &crate::distribution::schema::ReleaseManifestV1,
) -> Result<(), PreparedReleaseError> {
    archive_file.revalidate_for_preparation()?;
    archive::verify_archive(archive_file, manifest_bytes, manifest)?;
    archive_file.revalidate_for_preparation()?;
    Ok(())
}

#[cfg(test)]
mod tests;
