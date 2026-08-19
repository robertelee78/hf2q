use jiff::Timestamp;

use super::model::EmbeddedTrustRoot;
use super::target_set::{
    authenticate_selected_targets, AuthenticatedReleaseTargets, AuthenticatedTargetDescriptor,
    AuthenticatedTargetSet,
};
use super::verifier::ClockSource;
use super::TufVerifierError;
use crate::distribution::install_state::metadata::{
    lock_metadata_state, MetadataJournalError, MetadataStateAuthorization,
};
use crate::distribution::install_state::{
    ActiveInstalledReleaseFloor, ArtifactStageError, EphemeralArtifactStage,
    LiveInstalledReleaseFloor,
};
use crate::distribution::schema::{ReleaseVersion, TargetTriple};

mod extraction;

pub(in crate::distribution) use extraction::{
    ExtractionStageAuthorization, LockedReleasePreparation, PostLocalIoReleaseAuthorization,
    PreparedActivationAuthorization, PreparedPublicationAuthorization,
    PreparedVersionAuthorization, PreparedVersionCommitError, PreparedVersionCommitGuard,
};

#[derive(Debug, thiserror::Error)]
pub(in crate::distribution) enum ArtifactFetchAuthorizationError {
    #[error(transparent)]
    Authentication(#[from] TufVerifierError),
    #[error(transparent)]
    Journal(#[from] MetadataJournalError),
    #[error(transparent)]
    Stage(#[from] ArtifactStageError),
    #[error("the one-use archive stage has already been created")]
    ArchiveStageAlreadyCreated,
    #[error("the archive stage was not created before final authentication")]
    ArchiveStageMissing,
}

/// One-use request consumed by the locked install-state staging boundary.
///
/// Only a freshly reauthenticated release plan can construct this value.
#[derive(Debug)]
pub(in crate::distribution) struct ArchiveStageAuthorization {
    expected_length: u64,
}

/// One-use authority to fetch only the authenticated stable channel pointer.
///
/// The compiled trust anchor and state authorization remain private to the
/// signed-update bounded context; transport cannot supply or replace either.
#[derive(Debug)]
pub(in crate::distribution) struct ArtifactFetchAuthorization<'a> {
    authorization: &'a MetadataStateAuthorization,
    anchor: &'a EmbeddedTrustRoot,
    targets: AuthenticatedTargetSet,
}

/// Pointer-bound release fetch authority with cross-phase freshness state.
///
/// It may create one anonymous archive stage only after a lock-held ordinary
/// authority reread. It remains neither extraction nor installation authority.
#[derive(Debug)]
pub(in crate::distribution) struct BoundArtifactFetchAuthorization<'a> {
    authorization: &'a MetadataStateAuthorization,
    anchor: &'a EmbeddedTrustRoot,
    targets: AuthenticatedReleaseTargets,
    last_sample: Timestamp,
    release_floor: LiveInstalledReleaseFloor,
    archive_stage_created: bool,
}

/// Result of binding the exact authenticated stable pointer.
///
/// Already-current is a successful no-download outcome. Only `Fetch` carries
/// artifact transport authority, and that authority retains the exact live
/// installed-release floor observed under the shared lock.
pub(in crate::distribution) enum ArtifactPointerBinding<'a> {
    Fetch(BoundArtifactFetchAuthorization<'a>),
    AlreadyCurrent(AuthenticatedAlreadyCurrentRelease),
}

/// Fresh TUF plus live-install proof that the selected release is active.
///
/// This is diagnostic state only. It grants no download, activation,
/// downgrade, rollback, or filesystem mutation authority.
pub(in crate::distribution) struct AuthenticatedAlreadyCurrentRelease {
    targets: AuthenticatedReleaseTargets,
    floor: ActiveInstalledReleaseFloor,
}

/// Final current-time proof retained by the inert downloaded bundle.
pub(in crate::distribution) struct FinalArtifactAuthorization<'a> {
    authorization: &'a MetadataStateAuthorization,
    anchor: &'a EmbeddedTrustRoot,
    targets: AuthenticatedReleaseTargets,
    authenticated_at: Timestamp,
    release_floor: LiveInstalledReleaseFloor,
}

impl std::fmt::Debug for ArtifactPointerBinding<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fetch(_) => formatter.write_str("ArtifactPointerBinding::Fetch(..)"),
            Self::AlreadyCurrent(_) => {
                formatter.write_str("ArtifactPointerBinding::AlreadyCurrent(..)")
            }
        }
    }
}

impl std::fmt::Debug for AuthenticatedAlreadyCurrentRelease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AuthenticatedAlreadyCurrentRelease")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for FinalArtifactAuthorization<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FinalArtifactAuthorization")
            .finish_non_exhaustive()
    }
}

/// Deliberately dormant until a real stable root is compiled into hf2q.
pub(super) fn begin_artifact_fetch<'a>(
    authorization: &'a MetadataStateAuthorization,
    anchor: &'a EmbeddedTrustRoot,
) -> Result<ArtifactFetchAuthorization<'a>, ArtifactFetchAuthorizationError> {
    let targets = authenticate_selected_targets(authorization, anchor)?;
    Ok(ArtifactFetchAuthorization {
        authorization,
        anchor,
        targets,
    })
}

impl<'a> ArtifactFetchAuthorization<'a> {
    pub(in crate::distribution) fn pointer(&self) -> &AuthenticatedTargetDescriptor {
        self.targets.pointer()
    }

    pub(in crate::distribution) fn bind_pointer(
        self,
        exact_pointer_bytes: &[u8],
    ) -> Result<ArtifactPointerBinding<'a>, ArtifactFetchAuthorizationError> {
        self.bind_pointer_with_clock(exact_pointer_bytes, ClockSource::System)
    }

    fn bind_pointer_with_clock(
        self,
        exact_pointer_bytes: &[u8],
        clock: ClockSource,
    ) -> Result<ArtifactPointerBinding<'a>, ArtifactFetchAuthorizationError> {
        let last_sample = self.targets.authenticated_at();
        let expected = self.targets.bind_channel_pointer(exact_pointer_bytes)?;
        let locked = lock_metadata_state(self.authorization)?;
        let targets = reauthenticate_release(
            self.authorization,
            self.anchor,
            &locked,
            &expected,
            last_sample,
            clock,
        )?;
        let release_floor = locked.read_live_installed_release_floor()?;
        let disposition = classify_release(&release_floor, &targets)?;
        if disposition == AutomaticReleaseDisposition::AlreadyCurrent {
            let LiveInstalledReleaseFloor::Active(floor) = release_floor else {
                return Err(TufVerifierError::InstalledReleaseChanged.into());
            };
            return Ok(ArtifactPointerBinding::AlreadyCurrent(
                AuthenticatedAlreadyCurrentRelease { targets, floor },
            ));
        }
        Ok(ArtifactPointerBinding::Fetch(
            BoundArtifactFetchAuthorization {
                authorization: self.authorization,
                anchor: self.anchor,
                last_sample: targets.authenticated_at(),
                targets,
                release_floor,
                archive_stage_created: false,
            },
        ))
    }
}

impl<'a> ArtifactPointerBinding<'a> {
    pub(in crate::distribution) fn into_fetch(self) -> Option<BoundArtifactFetchAuthorization<'a>> {
        match self {
            Self::Fetch(fetch) => Some(fetch),
            Self::AlreadyCurrent(_) => None,
        }
    }
}

impl AuthenticatedAlreadyCurrentRelease {
    pub(in crate::distribution) fn version(&self) -> &ReleaseVersion {
        self.targets.version()
    }

    pub(in crate::distribution) fn receipt_sha256(&self) -> [u8; 32] {
        self.floor.receipt_sha256()
    }
}

impl<'a> BoundArtifactFetchAuthorization<'a> {
    pub(in crate::distribution) fn version(&self) -> &ReleaseVersion {
        self.targets.version()
    }

    pub(in crate::distribution) fn target(&self) -> TargetTriple {
        self.targets.target()
    }

    pub(in crate::distribution) fn manifest(&self) -> &AuthenticatedTargetDescriptor {
        self.targets.manifest()
    }

    pub(in crate::distribution) fn archive(&self) -> &AuthenticatedTargetDescriptor {
        self.targets.archive()
    }

    pub(in crate::distribution) fn exact_pointer_bytes(&self) -> &[u8] {
        self.targets.exact_pointer_bytes()
    }

    /// Reauthenticate the exact selected generation under the shared lock and
    /// create its anonymous archive stage before the large network transfer.
    pub(in crate::distribution) fn create_archive_stage(
        &mut self,
    ) -> Result<EphemeralArtifactStage, ArtifactFetchAuthorizationError> {
        self.create_archive_stage_with_clock(ClockSource::System)
    }

    fn create_archive_stage_with_clock(
        &mut self,
        clock: ClockSource,
    ) -> Result<EphemeralArtifactStage, ArtifactFetchAuthorizationError> {
        if self.archive_stage_created {
            return Err(ArtifactFetchAuthorizationError::ArchiveStageAlreadyCreated);
        }
        let locked = lock_metadata_state(self.authorization)?;
        let current = self.reauthenticate_locked(&locked, clock)?;
        require_same_release_floor(&locked, &self.release_floor)?;
        let stage = locked.create_ephemeral_artifact_stage(ArchiveStageAuthorization {
            expected_length: current.archive().length(),
        })?;
        self.archive_stage_created = true;
        self.last_sample = current.authenticated_at();
        self.targets = current;
        Ok(stage)
    }

    /// Repeat the same lock-held current-time proof after every external byte
    /// has been read. A generation change discards the staged result even when
    /// its target descriptors happen to be identical.
    pub(in crate::distribution) fn finalize(
        self,
    ) -> Result<FinalArtifactAuthorization<'a>, ArtifactFetchAuthorizationError> {
        self.finalize_with_clock(ClockSource::System)
    }

    fn finalize_with_clock(
        mut self,
        clock: ClockSource,
    ) -> Result<FinalArtifactAuthorization<'a>, ArtifactFetchAuthorizationError> {
        if !self.archive_stage_created {
            return Err(ArtifactFetchAuthorizationError::ArchiveStageMissing);
        }
        let locked = lock_metadata_state(self.authorization)?;
        let current = self.reauthenticate_locked(&locked, clock)?;
        require_same_release_floor(&locked, &self.release_floor)?;
        self.last_sample = current.authenticated_at();
        Ok(FinalArtifactAuthorization {
            authorization: self.authorization,
            anchor: self.anchor,
            authenticated_at: self.last_sample,
            targets: current,
            release_floor: self.release_floor,
        })
    }

    fn reauthenticate_locked(
        &self,
        locked: &crate::distribution::install_state::metadata::LockedMetadataState,
        clock: ClockSource,
    ) -> Result<AuthenticatedReleaseTargets, ArtifactFetchAuthorizationError> {
        reauthenticate_release(
            self.authorization,
            self.anchor,
            locked,
            &self.targets,
            self.last_sample,
            clock,
        )
    }
}

impl ArchiveStageAuthorization {
    pub(in crate::distribution) fn expected_length(&self) -> u64 {
        self.expected_length
    }
}

impl<'a> FinalArtifactAuthorization<'a> {
    pub(in crate::distribution) fn targets(&self) -> &AuthenticatedReleaseTargets {
        &self.targets
    }

    pub(in crate::distribution) fn authenticated_at(&self) -> Timestamp {
        self.authenticated_at
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AutomaticReleaseDisposition {
    InitialInstall,
    Upgrade,
    AlreadyCurrent,
}

pub(super) fn classify_release(
    floor: &LiveInstalledReleaseFloor,
    targets: &AuthenticatedReleaseTargets,
) -> Result<AutomaticReleaseDisposition, TufVerifierError> {
    classify_release_identity(
        floor,
        targets.version(),
        targets.target(),
        targets.manifest().sha256(),
        targets.archive().sha256(),
    )
}

pub(super) fn classify_release_identity(
    floor: &LiveInstalledReleaseFloor,
    version: &ReleaseVersion,
    target: TargetTriple,
    manifest_sha256: &crate::distribution::schema::Sha256Digest,
    archive_sha256: &crate::distribution::schema::Sha256Digest,
) -> Result<AutomaticReleaseDisposition, TufVerifierError> {
    let LiveInstalledReleaseFloor::Active(active) = floor else {
        return Ok(AutomaticReleaseDisposition::InitialInstall);
    };
    if active.target() != target {
        return Err(TufVerifierError::InstalledReleaseEquivocation);
    }
    match version.cmp(active.version()) {
        std::cmp::Ordering::Less => Err(TufVerifierError::InstalledReleaseRollback),
        std::cmp::Ordering::Greater => Ok(AutomaticReleaseDisposition::Upgrade),
        std::cmp::Ordering::Equal
            if manifest_sha256 == active.manifest_sha256()
                && archive_sha256 == active.archive_sha256() =>
        {
            Ok(AutomaticReleaseDisposition::AlreadyCurrent)
        }
        std::cmp::Ordering::Equal => Err(TufVerifierError::InstalledReleaseEquivocation),
    }
}

pub(super) fn require_same_release_floor(
    locked: &crate::distribution::install_state::metadata::LockedMetadataState,
    expected: &LiveInstalledReleaseFloor,
) -> Result<(), ArtifactFetchAuthorizationError> {
    if &locked.read_live_installed_release_floor()? != expected {
        return Err(TufVerifierError::InstalledReleaseChanged.into());
    }
    Ok(())
}

pub(super) fn reauthenticate_release(
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
    locked: &crate::distribution::install_state::metadata::LockedMetadataState,
    expected: &AuthenticatedReleaseTargets,
    last_sample: Timestamp,
    mut clock: ClockSource,
) -> Result<AuthenticatedReleaseTargets, ArtifactFetchAuthorizationError> {
    let stored = locked
        .read_selected_for_authority()?
        .ok_or(TufVerifierError::NoSelectedMetadata)?;
    let set = super::target_set::authenticate_stored_targets_after(
        authorization,
        anchor,
        stored,
        &mut clock,
        Some(last_sample),
    )?;
    let current = set.bind_channel_pointer(expected.exact_pointer_bytes())?;
    if current.authenticated_at() < last_sample {
        return Err(TufVerifierError::ClockRollback.into());
    }
    if !expected.exactly_matches_bound_release(&current) {
        return Err(TufVerifierError::DurableCommitMismatch.into());
    }
    Ok(current)
}

#[cfg(test)]
pub(super) fn begin_artifact_fetch_for_test<'a>(
    authorization: &'a MetadataStateAuthorization,
    anchor: &'a EmbeddedTrustRoot,
    samples: impl IntoIterator<Item = Timestamp>,
) -> Result<ArtifactFetchAuthorization<'a>, ArtifactFetchAuthorizationError> {
    let targets =
        super::target_set::authenticate_selected_targets_for_test(authorization, anchor, samples)?;
    Ok(ArtifactFetchAuthorization {
        authorization,
        anchor,
        targets,
    })
}

#[cfg(test)]
impl<'a> ArtifactFetchAuthorization<'a> {
    pub(super) fn bind_pointer_for_test(
        self,
        exact_pointer_bytes: &[u8],
        samples: impl IntoIterator<Item = Timestamp>,
    ) -> Result<ArtifactPointerBinding<'a>, ArtifactFetchAuthorizationError> {
        self.bind_pointer_with_clock(exact_pointer_bytes, ClockSource::scripted(samples))
    }
}

#[cfg(test)]
impl<'a> BoundArtifactFetchAuthorization<'a> {
    pub(super) fn create_archive_stage_for_test(
        &mut self,
        samples: impl IntoIterator<Item = Timestamp>,
    ) -> Result<EphemeralArtifactStage, ArtifactFetchAuthorizationError> {
        self.create_archive_stage_with_clock(ClockSource::scripted(samples))
    }

    pub(super) fn finalize_for_test(
        self,
        samples: impl IntoIterator<Item = Timestamp>,
    ) -> Result<FinalArtifactAuthorization<'a>, ArtifactFetchAuthorizationError> {
        self.finalize_with_clock(ClockSource::scripted(samples))
    }
}
