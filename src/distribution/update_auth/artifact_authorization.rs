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
use crate::distribution::install_state::{ArtifactStageError, EphemeralArtifactStage};
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
    archive_stage_created: bool,
}

/// Final current-time proof retained by the inert downloaded bundle.
pub(in crate::distribution) struct FinalArtifactAuthorization<'a> {
    authorization: &'a MetadataStateAuthorization,
    anchor: &'a EmbeddedTrustRoot,
    targets: AuthenticatedReleaseTargets,
    authenticated_at: Timestamp,
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
    ) -> Result<BoundArtifactFetchAuthorization<'a>, ArtifactFetchAuthorizationError> {
        let last_sample = self.targets.authenticated_at();
        let targets = self.targets.bind_channel_pointer(exact_pointer_bytes)?;
        Ok(BoundArtifactFetchAuthorization {
            authorization: self.authorization,
            anchor: self.anchor,
            targets,
            last_sample,
            archive_stage_created: false,
        })
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
        self.last_sample = current.authenticated_at();
        Ok(FinalArtifactAuthorization {
            authorization: self.authorization,
            anchor: self.anchor,
            authenticated_at: self.last_sample,
            targets: current,
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
