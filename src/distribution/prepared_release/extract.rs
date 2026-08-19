use std::io::{Read, Seek, SeekFrom};

#[cfg(target_os = "macos")]
use std::fs::File;

use zip::{CompressionMethod, ZipArchive};

use super::archive::VerifiedArchiveProfile;
use super::{ArchiveBoundRelease, PreparedReleaseError};
use crate::distribution::install_state::{ExtractedReleaseTree, ReleaseExtractionStage};
use crate::distribution::schema::ReleaseManifestV1;
use crate::distribution::update_auth::PostLocalIoReleaseAuthorization;

#[cfg(target_os = "macos")]
use super::codesign::{DeveloperIdVerification, SigningPolicy};
#[cfg(target_os = "macos")]
use crate::distribution::install_state::NormalizedExtractedReleaseTree;

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

/// Inert, signed-mode tree with a repeated lock-held Developer ID proof.
///
/// This value remains private staging and grants no publication, marker,
/// receipt, prepared-version, activation, path, or descriptor authority.
#[cfg(target_os = "macos")]
pub(in crate::distribution) struct SignedModeNormalizedRelease<'a> {
    _authentication: PostLocalIoReleaseAuthorization<'a>,
    _tree: NormalizedExtractedReleaseTree,
    _manifest_bytes: Box<[u8]>,
    _manifest: ReleaseManifestV1,
    _profile: VerifiedArchiveProfile,
    _developer_id: DeveloperIdVerification,
}

impl std::fmt::Debug for ExtractedRelease<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExtractedRelease")
            .finish_non_exhaustive()
    }
}

#[cfg(target_os = "macos")]
impl std::fmt::Debug for SignedModeNormalizedRelease<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SignedModeNormalizedRelease")
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

#[cfg(target_os = "macos")]
pub(super) fn verify_and_normalize_release<'a>(
    release: ExtractedRelease<'a>,
    policy: &SigningPolicy,
) -> Result<SignedModeNormalizedRelease<'a>, PreparedReleaseError> {
    verify_and_normalize_release_with(
        release,
        |path, file, manifest, binding| {
            super::macho::verify_file(file, manifest)?;
            Ok(super::codesign::verify_path(
                path, manifest, policy, binding,
            )?)
        },
        || {},
        PostLocalIoReleaseAuthorization::reauthenticate_after_mode_normalization,
    )
}

#[cfg(target_os = "macos")]
fn verify_and_normalize_release_with<'a>(
    release: ExtractedRelease<'a>,
    mut verify: impl FnMut(
        &std::path::Path,
        &File,
        &ReleaseManifestV1,
        crate::distribution::install_state::ExecutableReleaseBinding,
    ) -> Result<DeveloperIdVerification, PreparedReleaseError>,
    after_normalization: impl FnOnce(),
    reauthenticate: impl FnOnce(
        PostLocalIoReleaseAuthorization<'a>,
    ) -> Result<
        PostLocalIoReleaseAuthorization<'a>,
        crate::distribution::update_auth::ArtifactFetchAuthorizationError,
    >,
) -> Result<SignedModeNormalizedRelease<'a>, PreparedReleaseError> {
    let ExtractedRelease {
        _authentication: authentication,
        _tree: tree,
        _manifest_bytes: manifest_bytes,
        _manifest: manifest,
        _profile: profile,
    } = release;

    let first_verification: DeveloperIdVerification = authentication.with_extracted_executable(
        &tree,
        &manifest_bytes,
        &manifest,
        |path, file, binding| verify(path, file, &manifest, binding),
    )?;
    let tree = authentication.normalize_extracted_release(
        first_verification,
        tree,
        &manifest_bytes,
        &manifest,
    )?;
    after_normalization();
    let authentication = reauthenticate(authentication)?;
    let developer_id = authentication.with_normalized_executable(
        &tree,
        &manifest_bytes,
        &manifest,
        |path, file, binding| verify(path, file, &manifest, binding),
    )?;
    authentication.verify_normalized_release_tree(&tree, &manifest_bytes, &manifest)?;
    Ok(SignedModeNormalizedRelease {
        _authentication: authentication,
        _tree: tree,
        _manifest_bytes: manifest_bytes,
        _manifest: manifest,
        _profile: profile,
        _developer_id: developer_id,
    })
}

#[cfg(all(test, target_os = "macos"))]
pub(in crate::distribution) fn verify_and_normalize_release_for_test<'a>(
    release: ExtractedRelease<'a>,
    post_normalization_samples: Option<Vec<jiff::Timestamp>>,
    fail_verification_call: Option<usize>,
) -> Result<SignedModeNormalizedRelease<'a>, PreparedReleaseError> {
    use std::os::unix::fs::MetadataExt;

    let mut verification_call = 0_usize;
    let mut first_identity = None;
    verify_and_normalize_release_with(
        release,
        |path, file, _manifest, binding| {
            verification_call += 1;
            let metadata = file
                .metadata()
                .map_err(|_| PreparedReleaseError::CodeSigning)?;
            let identity = (path.to_path_buf(), metadata.dev(), metadata.ino());
            if let Some(first) = &first_identity {
                assert_eq!(
                    first, &identity,
                    "both native verifier brackets must bind the same path and inode"
                );
            } else {
                first_identity = Some(identity);
            }
            if fail_verification_call == Some(verification_call) {
                return Err(PreparedReleaseError::CodeSigning);
            }
            Ok(DeveloperIdVerification::for_test(binding))
        },
        || {},
        |authorization| match post_normalization_samples {
            Some(samples) => {
                authorization.reauthenticate_after_mode_normalization_for_test(samples)
            }
            None => authorization.reauthenticate_after_mode_normalization(),
        },
    )
}

#[cfg(all(test, target_os = "macos"))]
pub(in crate::distribution) fn verify_and_normalize_release_with_hook_for_test<'a>(
    release: ExtractedRelease<'a>,
    after_normalization: impl FnOnce(),
) -> Result<SignedModeNormalizedRelease<'a>, PreparedReleaseError> {
    verify_and_normalize_release_with(
        release,
        |_path, _file, _manifest, binding| Ok(DeveloperIdVerification::for_test(binding)),
        after_normalization,
        PostLocalIoReleaseAuthorization::reauthenticate_after_mode_normalization,
    )
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
