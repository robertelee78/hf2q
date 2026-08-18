use std::io::{Read, Seek, SeekFrom};

use zip::{CompressionMethod, ZipArchive};

use super::archive::VerifiedArchiveProfile;
use super::{ArchiveBoundRelease, PreparedReleaseError};
use crate::distribution::install_state::{ExtractedReleaseTree, ReleaseExtractionStage};
use crate::distribution::schema::ReleaseManifestV1;
use crate::distribution::update_auth::PostLocalIoReleaseAuthorization;

/// Inert exact extracted tree with the successful post-I/O TUF replay.
///
/// No path, file descriptor, executable mode, publication, or activation
/// authority is exposed by this value.
pub(in crate::distribution) struct ExtractedRelease<'a> {
    _authentication: PostLocalIoReleaseAuthorization<'a>,
    _tree: ExtractedReleaseTree,
    _manifest_bytes: Box<[u8]>,
    _manifest: ReleaseManifestV1,
    _profile: VerifiedArchiveProfile,
}

impl std::fmt::Debug for ExtractedRelease<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExtractedRelease")
            .finish_non_exhaustive()
    }
}

pub(in crate::distribution) fn extract_release(
    release: ArchiveBoundRelease<'_>,
) -> Result<ExtractedRelease<'_>, PreparedReleaseError> {
    let ArchiveBoundRelease { bundle, profile } = release;
    let mut parts = bundle.into_preparation_parts();
    let preparation = parts.authorization.lock_for_preparation()?;
    let mut stage = preparation.open_extraction_stage(&parts.manifest_bytes, &parts.manifest)?;
    parts.archive.revalidate()?;
    extract_entries(
        &mut parts.archive,
        &parts.manifest_bytes,
        &parts.manifest,
        &profile,
        &mut stage,
    )?;
    parts.archive.revalidate()?;
    let tree = stage.finish()?;
    let authentication = preparation.reauthenticate_after_local_io()?;
    Ok(ExtractedRelease {
        _authentication: authentication,
        _tree: tree,
        _manifest_bytes: parts.manifest_bytes,
        _manifest: parts.manifest,
        _profile: profile,
    })
}

pub(super) trait ExtractionSink {
    fn resume_manifest(&mut self, source: &mut dyn Read) -> Result<(), PreparedReleaseError>;
    fn resume_payload(
        &mut self,
        file: &crate::distribution::schema::BundleFileV1,
        source: &mut dyn Read,
    ) -> Result<(), PreparedReleaseError>;
}

impl ExtractionSink for ReleaseExtractionStage<'_> {
    fn resume_manifest(&mut self, source: &mut dyn Read) -> Result<(), PreparedReleaseError> {
        Ok(ReleaseExtractionStage::resume_manifest(self, source)?)
    }

    fn resume_payload(
        &mut self,
        file: &crate::distribution::schema::BundleFileV1,
        source: &mut dyn Read,
    ) -> Result<(), PreparedReleaseError> {
        Ok(ReleaseExtractionStage::resume_payload(self, file, source)?)
    }
}

pub(super) fn extract_entries<A: Read + Seek, S: ExtractionSink>(
    archive_file: &mut A,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
    profile: &VerifiedArchiveProfile,
    stage: &mut S,
) -> Result<(), PreparedReleaseError> {
    archive_file
        .seek(SeekFrom::Start(0))
        .map_err(|_| PreparedReleaseError::ArchiveRead)?;
    let mut archive =
        ZipArchive::new(&mut *archive_file).map_err(|_| PreparedReleaseError::ArchiveProfile)?;
    if archive.len() != profile.entries().len()
        || profile.entries().len() != manifest.files().len() + 1
    {
        return Err(PreparedReleaseError::ArchiveProfile);
    }
    for (index, descriptor) in profile.entries().iter().enumerate() {
        let mut entry = archive
            .by_index(index)
            .map_err(|_| PreparedReleaseError::ArchiveProfile)?;
        let method = match entry.compression() {
            CompressionMethod::Stored => 0,
            CompressionMethod::Deflated => 8,
            _ => return Err(PreparedReleaseError::ArchiveProfile),
        };
        if entry.name_raw() != descriptor.name().as_bytes()
            || entry.name() != descriptor.name()
            || method != descriptor.method()
            || entry.size() != u64::from(descriptor.uncompressed_size())
            || entry.compressed_size() != u64::from(descriptor.compressed_size())
            || entry.data_start() != descriptor.data_start()
        {
            return Err(PreparedReleaseError::ArchiveProfile);
        }
        if index == 0 {
            if descriptor.name() != "release-manifest.json"
                || entry.size() != exact_manifest.len() as u64
            {
                return Err(PreparedReleaseError::ArchiveProfile);
            }
            stage.resume_manifest(&mut entry)?;
        } else {
            let file = manifest
                .files()
                .get(index - 1)
                .ok_or(PreparedReleaseError::ArchiveProfile)?;
            if descriptor.name() != file.path().as_str() {
                return Err(PreparedReleaseError::ArchiveProfile);
            }
            stage.resume_payload(file, &mut entry)?;
        }
    }
    drop(archive);
    archive_file
        .seek(SeekFrom::Start(0))
        .map_err(|_| PreparedReleaseError::ArchiveRead)?;
    Ok(())
}
