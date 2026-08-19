use jiff::Timestamp;

use super::{reauthenticate_release, ArtifactFetchAuthorizationError, FinalArtifactAuthorization};
use crate::distribution::install_state::metadata::{
    lock_metadata_state, LockedMetadataState, MetadataStateAuthorization,
};
use crate::distribution::schema::{ReleaseManifestV1, ReleaseVersion, Sha256Digest, UpdateChannel};
use crate::distribution::update_auth::model::EmbeddedTrustRoot;
use crate::distribution::update_auth::target_set::AuthenticatedReleaseTargets;
use crate::distribution::update_auth::verifier::ClockSource;
use crate::distribution::update_auth::TufVerifierError;

/// One-use identity for the deterministic private extraction tree.
#[derive(Debug)]
pub(in crate::distribution) struct ExtractionStageAuthorization {
    version: ReleaseVersion,
    archive_sha256: Sha256Digest,
}

/// Exact selected-release identity authorized for first-version publication.
///
/// The fields are private so local paths, parsed JSON, or caller-supplied
/// digests cannot mint this capability. Install-state may inspect the bound
/// values, but only the current-time TUF replay in this module constructs it.
pub(in crate::distribution) struct PreparedVersionAuthorization {
    installation_id: String,
    state_root: String,
    version: ReleaseVersion,
    target: crate::distribution::schema::TargetTriple,
    manifest_sha256: Sha256Digest,
    archive_sha256: Sha256Digest,
    metadata_versions: [u64; 4],
}

#[derive(Debug, thiserror::Error)]
pub(in crate::distribution) enum PreparedVersionCommitError {
    #[error(transparent)]
    Authentication(#[from] ArtifactFetchAuthorizationError),
    #[error(transparent)]
    Publication(#[from] crate::distribution::install_state::PreparedVersionError),
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

/// One-use proof that the exact selected release was replayed immediately
/// before its prepared-version commit.
pub(in crate::distribution) struct PreparedPublicationAuthorization {
    authorization: PreparedVersionAuthorization,
    authenticated_at: Timestamp,
    earliest_expiry: Timestamp,
}

/// Final one-use freshness token binding exactly one verified receipt.
pub(in crate::distribution) struct PreparedActivationAuthorization {
    receipt_sha256: [u8; 32],
}

/// Borrowed one-use selector-boundary guard. Install-state can invoke only its
/// fail-closed check; it cannot construct or replace the replay/clock proof.
pub(in crate::distribution) struct PreparedVersionCommitGuard<'guard, 'state> {
    authenticated_at: &'guard mut Timestamp,
    publication: PreparedPublicationAuthorization,
    clock: ClockSource,
    error: Option<TufVerifierError>,
    used: bool,
    _lifetime: std::marker::PhantomData<&'state MetadataStateAuthorization>,
}

/// Final current-time TUF proof retained through prepared-version durability.
///
/// Only this capability may turn a published tree plus its exact native proof
/// into activation input. It is consumed whether final verification succeeds
/// or fails.
pub(in crate::distribution) struct FinalPreparedVersionAuthorization<'a> {
    authorization: PreparedVersionAuthorization,
    locked: LockedMetadataState,
    authenticated_at: Timestamp,
    earliest_expiry: Timestamp,
    _lifetime: std::marker::PhantomData<&'a MetadataStateAuthorization>,
}

impl std::fmt::Debug for PostLocalIoReleaseAuthorization<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PostLocalIoReleaseAuthorization")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for FinalPreparedVersionAuthorization<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FinalPreparedVersionAuthorization")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for PreparedPublicationAuthorization {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedPublicationAuthorization")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for PreparedActivationAuthorization {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedActivationAuthorization")
            .finish_non_exhaustive()
    }
}

impl PreparedActivationAuthorization {
    fn for_receipt(receipt: &[u8]) -> Self {
        use sha2::{Digest, Sha256};

        Self {
            receipt_sha256: Sha256::digest(receipt).into(),
        }
    }

    pub(in crate::distribution) fn matches_receipt(&self, receipt: &[u8]) -> bool {
        use sha2::{Digest, Sha256};

        self.receipt_sha256 == <[u8; 32]>::from(Sha256::digest(receipt))
    }
}

impl std::fmt::Debug for PreparedVersionCommitGuard<'_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedVersionCommitGuard")
            .finish_non_exhaustive()
    }
}

impl PreparedVersionCommitGuard<'_, '_> {
    pub(in crate::distribution) fn check_at_commit_boundary(
        &mut self,
        authorization: &PreparedVersionAuthorization,
    ) -> Result<(), crate::distribution::install_state::PreparedVersionError> {
        if self.used
            || !self
                .publication
                .authorization
                .exactly_matches(authorization)
            || self.publication.authenticated_at != *self.authenticated_at
        {
            self.error = Some(TufVerifierError::DurableCommitMismatch);
            return Err(crate::distribution::install_state::PreparedVersionError::Integrity);
        }
        self.used = true;
        let sample = match self.clock.sample() {
            Ok(sample) => sample,
            Err(error) => {
                self.error = Some(error);
                return Err(crate::distribution::install_state::PreparedVersionError::Integrity);
            }
        };
        if sample < self.publication.authenticated_at {
            self.error = Some(TufVerifierError::ClockRollback);
            return Err(crate::distribution::install_state::PreparedVersionError::Integrity);
        }
        if sample >= self.publication.earliest_expiry {
            self.error = Some(TufVerifierError::ExpiredMetadata);
            return Err(crate::distribution::install_state::PreparedVersionError::Integrity);
        }
        *self.authenticated_at = sample;
        Ok(())
    }
}

impl std::fmt::Debug for LockedReleasePreparation<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LockedReleasePreparation")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for PreparedVersionAuthorization {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedVersionAuthorization")
            .finish_non_exhaustive()
    }
}

impl PreparedVersionAuthorization {
    fn exactly_matches(&self, other: &Self) -> bool {
        self.installation_id == other.installation_id
            && self.state_root == other.state_root
            && self.version == other.version
            && self.target == other.target
            && self.manifest_sha256 == other.manifest_sha256
            && self.archive_sha256 == other.archive_sha256
            && self.metadata_versions == other.metadata_versions
    }

    pub(in crate::distribution) fn installation_id(&self) -> &str {
        &self.installation_id
    }

    pub(in crate::distribution) fn state_root(&self) -> &str {
        &self.state_root
    }

    pub(in crate::distribution) fn version(&self) -> &ReleaseVersion {
        &self.version
    }

    pub(in crate::distribution) fn target(&self) -> crate::distribution::schema::TargetTriple {
        self.target
    }

    pub(in crate::distribution) fn manifest_sha256(&self) -> &Sha256Digest {
        &self.manifest_sha256
    }

    pub(in crate::distribution) fn archive_sha256(&self) -> &Sha256Digest {
        &self.archive_sha256
    }

    pub(in crate::distribution) fn metadata_versions(&self) -> [u64; 4] {
        self.metadata_versions
    }
}

impl<'a> PostLocalIoReleaseAuthorization<'a> {
    pub(in crate::distribution) fn prepared_version_authorization(
        &self,
    ) -> PreparedVersionAuthorization {
        prepared_authorization(&self.targets)
    }

    pub(in crate::distribution) fn authenticated_at(&self) -> Timestamp {
        self.authenticated_at
    }

    pub(in crate::distribution) fn stage_normalized_prepared_version(
        &self,
        developer_id: crate::distribution::prepared_release::DeveloperIdVerification,
        tree: crate::distribution::install_state::NormalizedExtractedReleaseTree,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
    ) -> Result<
        crate::distribution::install_state::PreparedVersionState,
        crate::distribution::install_state::PreparedVersionError,
    > {
        let installed_at = u64::try_from(self.authenticated_at.as_second())
            .map_err(|_| crate::distribution::install_state::PreparedVersionError::Integrity)?;
        self._locked.stage_normalized_prepared_version(
            &self.prepared_version_authorization(),
            developer_id,
            tree,
            exact_manifest,
            manifest,
            installed_at,
        )
    }

    pub(in crate::distribution) fn recover_prepared_version(
        &self,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
    ) -> Result<
        Option<crate::distribution::install_state::PreparedVersionState>,
        crate::distribution::install_state::PreparedVersionError,
    > {
        let recovery_reference = u64::try_from(self.authenticated_at.as_second())
            .map_err(|_| crate::distribution::install_state::PreparedVersionError::Integrity)?;
        self._locked.recover_prepared_version(
            &self.prepared_version_authorization(),
            exact_manifest,
            manifest,
            recovery_reference,
        )
    }

    #[cfg(target_os = "macos")]
    pub(in crate::distribution) fn with_prepared_executable<R, E>(
        &self,
        state: &crate::distribution::install_state::PreparedVersionState,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
        operation: impl FnOnce(
            &std::path::Path,
            &std::fs::File,
            crate::distribution::install_state::ExecutableReleaseBinding,
        ) -> Result<R, E>,
    ) -> Result<R, E>
    where
        E: From<crate::distribution::install_state::PreparedVersionError>,
    {
        self._locked
            .with_prepared_executable(state, exact_manifest, manifest, operation)
    }

    pub(in crate::distribution) fn verify_prepared_version_tree(
        &self,
        state: &crate::distribution::install_state::PreparedVersionState,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
    ) -> Result<(), crate::distribution::install_state::PreparedVersionError> {
        self._locked.verify_prepared_version_tree(
            state,
            &self.prepared_version_authorization(),
            exact_manifest,
            manifest,
        )
    }

    pub(in crate::distribution) fn publish_pending_prepared_version(
        &mut self,
        publication: PreparedPublicationAuthorization,
        pending: crate::distribution::install_state::PreparedVersionState,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
        developer_id: crate::distribution::prepared_release::DeveloperIdVerification,
    ) -> Result<
        crate::distribution::install_state::PublishedPreparedVersion,
        PreparedVersionCommitError,
    > {
        self.publish_pending_prepared_version_with_clock(
            publication,
            pending,
            exact_manifest,
            manifest,
            developer_id,
            ClockSource::System,
        )
    }

    fn publish_pending_prepared_version_with_clock(
        &mut self,
        publication: PreparedPublicationAuthorization,
        pending: crate::distribution::install_state::PreparedVersionState,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
        developer_id: crate::distribution::prepared_release::DeveloperIdVerification,
        clock: ClockSource,
    ) -> Result<
        crate::distribution::install_state::PublishedPreparedVersion,
        PreparedVersionCommitError,
    > {
        let current_authorization = self.prepared_version_authorization();
        if !publication
            .authorization
            .exactly_matches(&current_authorization)
            || publication.authenticated_at != self.authenticated_at
            || publication.earliest_expiry != self.targets.earliest_expiry()
        {
            return Err(ArtifactFetchAuthorizationError::from(
                TufVerifierError::DurableCommitMismatch,
            )
            .into());
        }
        let authorization = self.prepared_version_authorization();
        let mut guard = PreparedVersionCommitGuard {
            authenticated_at: &mut self.authenticated_at,
            publication,
            clock,
            error: None,
            used: false,
            _lifetime: std::marker::PhantomData,
        };
        let result = self._locked.publish_pending_prepared_version(
            &authorization,
            pending,
            exact_manifest,
            manifest,
            developer_id,
            &mut guard,
        );
        if let Some(error) = guard.error {
            return Err(ArtifactFetchAuthorizationError::from(error).into());
        }
        result.map_err(PreparedVersionCommitError::from)
    }

    pub(in crate::distribution) fn finish_published_prepared_version(
        &self,
        published: &crate::distribution::install_state::PublishedPreparedVersion,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
    ) -> Result<(), crate::distribution::install_state::PreparedVersionError> {
        self._locked.finish_published_prepared_version(
            &self.prepared_version_authorization(),
            published,
            exact_manifest,
            manifest,
        )
    }

    pub(in crate::distribution) fn reauthenticate_after_prepared_publication(
        self,
    ) -> Result<FinalPreparedVersionAuthorization<'a>, ArtifactFetchAuthorizationError> {
        let current = reauthenticate_release(
            self._authorization,
            self._anchor,
            &self._locked,
            &self.targets,
            self.authenticated_at,
            ClockSource::System,
        )?;
        let authenticated_at = current.authenticated_at();
        let earliest_expiry = current.earliest_expiry();
        Ok(FinalPreparedVersionAuthorization {
            authorization: prepared_authorization(&current),
            locked: self._locked,
            authenticated_at,
            earliest_expiry,
            _lifetime: std::marker::PhantomData,
        })
    }

    #[cfg(target_os = "macos")]
    pub(in crate::distribution) fn with_extracted_executable<R, E>(
        &self,
        tree: &crate::distribution::install_state::ExtractedReleaseTree,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
        operation: impl FnOnce(
            &std::path::Path,
            &std::fs::File,
            crate::distribution::install_state::ExecutableReleaseBinding,
        ) -> Result<R, E>,
    ) -> Result<R, E>
    where
        E: From<crate::distribution::install_state::ExtractionError>,
    {
        self._locked
            .with_extracted_executable(tree, exact_manifest, manifest, operation)
    }

    pub(in crate::distribution) fn normalize_extracted_release(
        &self,
        developer_id: crate::distribution::prepared_release::DeveloperIdVerification,
        tree: crate::distribution::install_state::ExtractedReleaseTree,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
    ) -> Result<
        crate::distribution::install_state::NormalizedExtractedReleaseTree,
        crate::distribution::install_state::ExtractionError,
    > {
        self._locked
            .normalize_extracted_release(developer_id, tree, exact_manifest, manifest)
    }

    pub(in crate::distribution) fn reauthenticate_after_mode_normalization(
        self,
    ) -> Result<Self, ArtifactFetchAuthorizationError> {
        self.reauthenticate_after_mode_normalization_with_clock(ClockSource::System)
    }

    pub(in crate::distribution) fn reauthenticate_before_prepared_publication(
        self,
    ) -> Result<(Self, PreparedPublicationAuthorization), ArtifactFetchAuthorizationError> {
        self.reauthenticate_before_prepared_publication_with_clock(ClockSource::System)
    }

    fn reauthenticate_before_prepared_publication_with_clock(
        self,
        clock: ClockSource,
    ) -> Result<(Self, PreparedPublicationAuthorization), ArtifactFetchAuthorizationError> {
        let current = self.reauthenticate_after_mode_normalization_with_clock(clock)?;
        let publication = PreparedPublicationAuthorization {
            authorization: current.prepared_version_authorization(),
            authenticated_at: current.authenticated_at,
            earliest_expiry: current.targets.earliest_expiry(),
        };
        Ok((current, publication))
    }

    fn reauthenticate_after_mode_normalization_with_clock(
        self,
        clock: ClockSource,
    ) -> Result<Self, ArtifactFetchAuthorizationError> {
        let current = reauthenticate_release(
            self._authorization,
            self._anchor,
            &self._locked,
            &self.targets,
            self.authenticated_at,
            clock,
        )?;
        let authenticated_at = current.authenticated_at();
        Ok(Self {
            _authorization: self._authorization,
            _anchor: self._anchor,
            _locked: self._locked,
            targets: current,
            authenticated_at,
        })
    }

    #[cfg(target_os = "macos")]
    pub(in crate::distribution) fn with_normalized_executable<R, E>(
        &self,
        tree: &crate::distribution::install_state::NormalizedExtractedReleaseTree,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
        operation: impl FnOnce(
            &std::path::Path,
            &std::fs::File,
            crate::distribution::install_state::ExecutableReleaseBinding,
        ) -> Result<R, E>,
    ) -> Result<R, E>
    where
        E: From<crate::distribution::install_state::ExtractionError>,
    {
        self._locked
            .with_normalized_executable(tree, exact_manifest, manifest, operation)
    }

    pub(in crate::distribution) fn verify_normalized_release_tree(
        &self,
        tree: &crate::distribution::install_state::NormalizedExtractedReleaseTree,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
    ) -> Result<(), crate::distribution::install_state::ExtractionError> {
        self._locked
            .verify_normalized_release_tree(tree, exact_manifest, manifest)
    }
}

impl FinalPreparedVersionAuthorization<'_> {
    pub(in crate::distribution) fn authenticate_published_version(
        self,
        state: crate::distribution::install_state::PreparedVersionState,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
        developer_id: crate::distribution::prepared_release::DeveloperIdVerification,
    ) -> Result<
        crate::distribution::install_state::AuthenticatedPreparedVersion,
        PreparedVersionCommitError,
    > {
        self.authenticate_published_version_with_clock(
            state,
            exact_manifest,
            manifest,
            developer_id,
            ClockSource::System,
        )
    }

    fn authenticate_published_version_with_clock(
        self,
        state: crate::distribution::install_state::PreparedVersionState,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
        developer_id: crate::distribution::prepared_release::DeveloperIdVerification,
        mut clock: ClockSource,
    ) -> Result<
        crate::distribution::install_state::AuthenticatedPreparedVersion,
        PreparedVersionCommitError,
    > {
        let prepared = self.locked.authenticate_published_prepared_version(
            &self.authorization,
            state,
            exact_manifest,
            manifest,
            developer_id,
        )?;
        let sample = clock
            .sample()
            .map_err(ArtifactFetchAuthorizationError::from)?;
        if sample < self.authenticated_at {
            return Err(
                ArtifactFetchAuthorizationError::from(TufVerifierError::ClockRollback).into(),
            );
        }
        if sample >= self.earliest_expiry {
            return Err(
                ArtifactFetchAuthorizationError::from(TufVerifierError::ExpiredMetadata).into(),
            );
        }
        let authorization = PreparedActivationAuthorization::for_receipt(prepared.receipt_bytes());
        Ok(prepared.authenticate(authorization)?)
    }

    #[cfg(test)]
    pub(in crate::distribution) fn authenticate_published_version_for_test(
        self,
        state: crate::distribution::install_state::PreparedVersionState,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
        developer_id: crate::distribution::prepared_release::DeveloperIdVerification,
        samples: impl IntoIterator<Item = Timestamp>,
    ) -> Result<
        crate::distribution::install_state::AuthenticatedPreparedVersion,
        PreparedVersionCommitError,
    > {
        self.authenticate_published_version_with_clock(
            state,
            exact_manifest,
            manifest,
            developer_id,
            ClockSource::scripted(samples),
        )
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
    pub(in crate::distribution) fn has_recoverable_prepared_version(
        &self,
    ) -> Result<bool, crate::distribution::install_state::PreparedVersionError> {
        self.locked
            .has_recoverable_prepared_version(&prepared_authorization(&self.targets))
    }
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

fn prepared_authorization(targets: &AuthenticatedReleaseTargets) -> PreparedVersionAuthorization {
    PreparedVersionAuthorization {
        installation_id: targets.installation_id().to_owned(),
        state_root: targets.state_root().to_owned(),
        version: targets.version().clone(),
        target: targets.target(),
        manifest_sha256: targets.manifest().sha256().clone(),
        archive_sha256: targets.archive().sha256().clone(),
        metadata_versions: targets.metadata_versions(),
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
impl<'a> PostLocalIoReleaseAuthorization<'a> {
    pub(in crate::distribution::update_auth) fn authenticated_at_for_test(&self) -> Timestamp {
        self.authenticated_at
    }

    pub(in crate::distribution) fn reauthenticate_after_mode_normalization_for_test(
        self,
        samples: impl IntoIterator<Item = Timestamp>,
    ) -> Result<Self, ArtifactFetchAuthorizationError> {
        self.reauthenticate_after_mode_normalization_with_clock(ClockSource::scripted(samples))
    }

    pub(in crate::distribution) fn reauthenticate_before_prepared_publication_for_test(
        self,
        samples: impl IntoIterator<Item = Timestamp>,
    ) -> Result<(Self, PreparedPublicationAuthorization), ArtifactFetchAuthorizationError> {
        self.reauthenticate_before_prepared_publication_with_clock(ClockSource::scripted(samples))
    }

    pub(in crate::distribution) fn publish_pending_prepared_version_for_test(
        &mut self,
        publication: PreparedPublicationAuthorization,
        pending: crate::distribution::install_state::PreparedVersionState,
        exact_manifest: &[u8],
        manifest: &ReleaseManifestV1,
        developer_id: crate::distribution::prepared_release::DeveloperIdVerification,
        samples: impl IntoIterator<Item = Timestamp>,
    ) -> Result<
        crate::distribution::install_state::PublishedPreparedVersion,
        PreparedVersionCommitError,
    > {
        self.publish_pending_prepared_version_with_clock(
            publication,
            pending,
            exact_manifest,
            manifest,
            developer_id,
            ClockSource::scripted(samples),
        )
    }

    pub(in crate::distribution) fn reauthenticate_after_prepared_publication_for_test(
        self,
        samples: impl IntoIterator<Item = Timestamp>,
    ) -> Result<FinalPreparedVersionAuthorization<'a>, ArtifactFetchAuthorizationError> {
        let current = reauthenticate_release(
            self._authorization,
            self._anchor,
            &self._locked,
            &self.targets,
            self.authenticated_at,
            ClockSource::scripted(samples),
        )?;
        let authenticated_at = current.authenticated_at();
        let earliest_expiry = current.earliest_expiry();
        Ok(FinalPreparedVersionAuthorization {
            authorization: prepared_authorization(&current),
            locked: self._locked,
            authenticated_at,
            earliest_expiry,
            _lifetime: std::marker::PhantomData,
        })
    }
}
