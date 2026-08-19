//! Exact release-archive preparation before filesystem publication.
//!
//! This bounded context validates a listing-first exact-inventory ZIP and then
//! materializes it only into private descriptor-relative inert staging while
//! retaining the shared installation lock. It owns no codesign, signed-mode
//! normalization, publication, marker, receipt, prepared-version, activation,
//! installer, or CLI authority.

mod archive;
mod deflate;
mod extract;
mod macho;

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
    #[error(transparent)]
    Authentication(#[from] crate::distribution::update_auth::ArtifactFetchAuthorizationError),
    #[error(transparent)]
    Extraction(#[from] crate::distribution::install_state::ExtractionError),
}

/// Exact archive/manifest agreement before lock-held inert extraction.
///
/// The wrapper is intentionally non-cloneable and non-serializable. Future
/// Only this module's extraction coordinator may consume it; macOS signing and
/// durable version publication remain separate future authority boundaries.
pub(super) struct ArchiveBoundRelease<'a> {
    bundle: VerifiedReleaseBundle<'a>,
    profile: archive::VerifiedArchiveProfile,
}

impl std::fmt::Debug for ArchiveBoundRelease<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ArchiveBoundRelease")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
impl ArchiveBoundRelease<'_> {
    pub(in crate::distribution) fn fail_archive_revalidation_after_for_test(
        &mut self,
        calls: usize,
    ) {
        let (_, _, archive) = self.bundle.preparation_parts_mut();
        archive.fail_revalidation_after_for_test(calls);
    }
}

pub(super) fn bind_archive<'a>(
    mut bundle: VerifiedReleaseBundle<'a>,
) -> Result<ArchiveBoundRelease<'a>, PreparedReleaseError> {
    let (manifest_bytes, manifest, archive_file) = bundle.preparation_parts_mut();
    let profile = validate_archive_reader(archive_file, manifest_bytes, manifest)?;
    Ok(ArchiveBoundRelease { bundle, profile })
}

pub(super) fn extract_release(
    release: ArchiveBoundRelease<'_>,
) -> Result<extract::ExtractedRelease<'_>, PreparedReleaseError> {
    extract::extract_release(release)
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
) -> Result<archive::VerifiedArchiveProfile, PreparedReleaseError> {
    archive_file.revalidate_for_preparation()?;
    let profile = archive::verify_archive(archive_file, manifest_bytes, manifest)?;
    archive_file.revalidate_for_preparation()?;
    Ok(profile)
}

#[cfg(test)]
mod macho_tests;
#[cfg(test)]
mod tests;
