use jiff::Timestamp;

use super::{reauthenticate_release, ArtifactFetchAuthorizationError, FinalArtifactAuthorization};
use crate::distribution::install_state::metadata::{
    lock_metadata_state, LockedMetadataState, MetadataStateAuthorization,
};
use crate::distribution::schema::{ReleaseManifestV1, ReleaseVersion, Sha256Digest, UpdateChannel};
use crate::distribution::update_auth::model::EmbeddedTrustRoot;
use crate::distribution::update_auth::target_set::AuthenticatedReleaseTargets;
use crate::distribution::update_auth::verifier::ClockSource;

/// One-use identity for the deterministic private extraction tree.
#[derive(Debug)]
pub(in crate::distribution) struct ExtractionStageAuthorization {
    version: ReleaseVersion,
    archive_sha256: Sha256Digest,
}

/// Shared-lock release proof held across inert local preparation I/O.
///
/// This capability can neither publish a version nor activate it. Its only
/// authority is to bracket a private extraction with exact selected-metadata
/// replay while the installation lock remains held.
pub(in crate::distribution) struct LockedReleasePreparation<'a> {
    authorization: &'a MetadataStateAuthorization,
    anchor: &'a EmbeddedTrustRoot,
    locked: LockedMetadataState,
    targets: AuthenticatedReleaseTargets,
    last_sample: Timestamp,
}

/// Current selected-release proof after the complete local-I/O bracket.
pub(in crate::distribution) struct PostLocalIoReleaseAuthorization<'a> {
    _authorization: &'a MetadataStateAuthorization,
    _anchor: &'a EmbeddedTrustRoot,
    _locked: LockedMetadataState,
    targets: AuthenticatedReleaseTargets,
    authenticated_at: Timestamp,
}

impl std::fmt::Debug for PostLocalIoReleaseAuthorization<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PostLocalIoReleaseAuthorization")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for LockedReleasePreparation<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LockedReleasePreparation")
            .finish_non_exhaustive()
    }
}

impl ExtractionStageAuthorization {
    pub(in crate::distribution) fn version(&self) -> &ReleaseVersion {
        &self.version
    }

    pub(in crate::distribution) fn stage_name(&self) -> String {
        format!(
            ".extract-v{}-{}",
            self.version.as_str(),
            self.archive_sha256.as_str()
        )
    }

    #[cfg(test)]
    pub(in crate::distribution) fn for_test(version: &str, archive_bytes: &[u8]) -> Self {
        use sha2::{Digest, Sha256};

        Self {
            version: ReleaseVersion::parse_stable("test.version", version.to_owned())
                .expect("test release version"),
            archive_sha256: Sha256Digest::parse(
                "test.archive_sha256",
                hex::encode(Sha256::digest(archive_bytes)),
            )
            .expect("test archive digest"),
        }
    }
}

impl<'a> FinalArtifactAuthorization<'a> {
    pub(in crate::distribution) fn lock_for_preparation(
        self,
    ) -> Result<LockedReleasePreparation<'a>, ArtifactFetchAuthorizationError> {
        self.lock_for_preparation_with_clock(ClockSource::System)
    }

    fn lock_for_preparation_with_clock(
        self,
        clock: ClockSource,
    ) -> Result<LockedReleasePreparation<'a>, ArtifactFetchAuthorizationError> {
        let locked = lock_metadata_state(self.authorization)?;
        let targets = reauthenticate_release(
            self.authorization,
            self.anchor,
            &locked,
            &self.targets,
            self.authenticated_at,
            clock,
        )?;
        let last_sample = targets.authenticated_at();
        Ok(LockedReleasePreparation {
            authorization: self.authorization,
            anchor: self.anchor,
            locked,
            targets,
            last_sample,
        })
    }

    #[cfg(test)]
    fn with_locked_extraction_with_clocks<E>(
        self,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
        lock_clock: ClockSource,
        post_io_clock: ClockSource,
        operation: impl for<'lock> FnOnce(
            &mut crate::distribution::install_state::ReleaseExtractionStage<'lock>,
        ) -> Result<(), E>,
    ) -> Result<PostLocalIoReleaseAuthorization<'a>, E>
    where
        E: From<ArtifactFetchAuthorizationError>
            + From<crate::distribution::install_state::ExtractionError>,
    {
        let preparation = self
            .lock_for_preparation_with_clock(lock_clock)
            .map_err(E::from)?;
        let mut stage = preparation
            .open_extraction_stage(exact_manifest, manifest)
            .map_err(E::from)?;
        operation(&mut stage)?;
        drop(stage.finish().map_err(E::from)?);
        preparation
            .reauthenticate_after_local_io_with_clock(post_io_clock)
            .map_err(E::from)
    }
}

impl<'a> LockedReleasePreparation<'a> {
    pub(in crate::distribution) fn open_extraction_stage(
        &self,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
    ) -> Result<
        crate::distribution::install_state::ReleaseExtractionStage<'_>,
        crate::distribution::install_state::ExtractionError,
    > {
        let deterministic = manifest
            .to_deterministic_json()
            .map_err(|_| crate::distribution::install_state::ExtractionError::Integrity)?;
        if !self.targets.manifest().matches_bytes(exact_manifest)
            || manifest.version() != self.targets.version()
            || manifest.target() != self.targets.target()
            || manifest.channel() != UpdateChannel::Stable
            || deterministic != exact_manifest
        {
            return Err(crate::distribution::install_state::ExtractionError::Integrity);
        }
        self.locked.open_release_extraction(
            ExtractionStageAuthorization {
                version: self.targets.version().clone(),
                archive_sha256: self.targets.archive().sha256().clone(),
            },
            exact_manifest,
            manifest,
        )
    }

    pub(in crate::distribution) fn reauthenticate_after_local_io(
        self,
    ) -> Result<PostLocalIoReleaseAuthorization<'a>, ArtifactFetchAuthorizationError> {
        self.reauthenticate_after_local_io_with_clock(ClockSource::System)
    }

    fn reauthenticate_after_local_io_with_clock(
        self,
        clock: ClockSource,
    ) -> Result<PostLocalIoReleaseAuthorization<'a>, ArtifactFetchAuthorizationError> {
        let current = reauthenticate_release(
            self.authorization,
            self.anchor,
            &self.locked,
            &self.targets,
            self.last_sample,
            clock,
        )?;
        let authenticated_at = current.authenticated_at();
        Ok(PostLocalIoReleaseAuthorization {
            _authorization: self.authorization,
            _anchor: self.anchor,
            _locked: self.locked,
            targets: current,
            authenticated_at,
        })
    }
}

#[cfg(test)]
impl<'a> FinalArtifactAuthorization<'a> {
    pub(in crate::distribution::update_auth) fn lock_for_preparation_for_test(
        self,
        samples: impl IntoIterator<Item = Timestamp>,
    ) -> Result<LockedReleasePreparation<'a>, ArtifactFetchAuthorizationError> {
        self.lock_for_preparation_with_clock(ClockSource::scripted(samples))
    }

    pub(in crate::distribution::update_auth) fn with_locked_extraction_for_test<E>(
        self,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
        lock_samples: impl IntoIterator<Item = Timestamp>,
        post_io_samples: impl IntoIterator<Item = Timestamp>,
        operation: impl for<'lock> FnOnce(
            &mut crate::distribution::install_state::ReleaseExtractionStage<'lock>,
        ) -> Result<(), E>,
    ) -> Result<PostLocalIoReleaseAuthorization<'a>, E>
    where
        E: From<ArtifactFetchAuthorizationError>
            + From<crate::distribution::install_state::ExtractionError>,
    {
        self.with_locked_extraction_with_clocks(
            exact_manifest,
            manifest,
            ClockSource::scripted(lock_samples),
            ClockSource::scripted(post_io_samples),
            operation,
        )
    }
}

#[cfg(test)]
impl<'a> LockedReleasePreparation<'a> {
    pub(in crate::distribution::update_auth) fn reauthenticate_after_local_io_for_test(
        self,
        samples: impl IntoIterator<Item = Timestamp>,
    ) -> Result<PostLocalIoReleaseAuthorization<'a>, ArtifactFetchAuthorizationError> {
        self.reauthenticate_after_local_io_with_clock(ClockSource::scripted(samples))
    }
}

#[cfg(test)]
impl PostLocalIoReleaseAuthorization<'_> {
    pub(in crate::distribution::update_auth) fn authenticated_at_for_test(&self) -> Timestamp {
        self.authenticated_at
    }
}
