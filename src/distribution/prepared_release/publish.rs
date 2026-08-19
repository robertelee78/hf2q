use std::fs::File;

use super::codesign::{DeveloperIdVerification, SigningPolicy};
use super::extract::{self, SignedModeNormalizedRelease};
use super::{bind_archive, PreparedReleaseError};
use crate::distribution::install_state::{AuthenticatedPreparedVersion, PreparedVersionState};
use crate::distribution::schema::ReleaseManifestV1;
use crate::distribution::update_auth::PostLocalIoReleaseAuthorization;
use crate::distribution::update_transport::VerifiedReleaseBundle;

pub(in crate::distribution) enum PreparedReleaseOutcome {
    Prepared(AuthenticatedPreparedVersion),
    AlreadyPrepared(AuthenticatedPreparedVersion),
}

impl std::fmt::Debug for PreparedReleaseOutcome {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Prepared(_) => formatter.write_str("PreparedReleaseOutcome::Prepared(..)"),
            Self::AlreadyPrepared(_) => {
                formatter.write_str("PreparedReleaseOutcome::AlreadyPrepared(..)")
            }
        }
    }
}

/// Dormant first-install preparation coordinator.
///
/// No production caller can construct `SigningPolicy` until the real stable
/// Team ID is compiled. When enabled, this is the sole path from downloaded
/// bytes to `AuthenticatedPreparedVersion`.
#[allow(dead_code)]
pub(super) fn prepare_release<'a>(
    bundle: VerifiedReleaseBundle<'a>,
    policy: &SigningPolicy,
) -> Result<PreparedReleaseOutcome, PreparedReleaseError> {
    prepare_release_with(
        bundle,
        PublicationClocks::System,
        |path, file, manifest, binding| {
            super::macho::verify_file(file, manifest)?;
            Ok(super::codesign::verify_path(
                path, manifest, policy, binding,
            )?)
        },
    )
}

fn prepare_release_with<'a>(
    bundle: VerifiedReleaseBundle<'a>,
    clocks: PublicationClocks,
    mut verify: impl FnMut(
        &std::path::Path,
        &File,
        &ReleaseManifestV1,
        crate::distribution::install_state::ExecutableReleaseBinding,
    ) -> Result<DeveloperIdVerification, PreparedReleaseError>,
) -> Result<PreparedReleaseOutcome, PreparedReleaseError> {
    let bound = bind_archive(bundle)?;
    let super::ArchiveBoundRelease { bundle, profile } = bound;
    let crate::distribution::update_transport::ReleasePreparationParts {
        authorization,
        manifest_bytes,
        manifest,
        archive,
    } = bundle.into_preparation_parts();
    let preparation = authorization.lock_for_preparation()?;

    if preparation.has_recoverable_prepared_version()? {
        let authentication = preparation.reauthenticate_after_local_io()?;
        let state = authentication
            .recover_prepared_version(&manifest_bytes, &manifest)?
            .ok_or(crate::distribution::install_state::PreparedVersionError::Integrity)?;
        drop(archive);
        drop(profile);
        return finish_state(
            authentication,
            state,
            manifest_bytes,
            manifest,
            true,
            clocks,
            &mut verify,
        );
    }

    let extracted =
        extract::extract_with_preparation(manifest_bytes, manifest, archive, profile, preparation)?;
    let normalized = extract::verify_and_normalize_release_with(
        extracted,
        |path, file, manifest, binding| verify(path, file, manifest, binding),
        || {},
        PostLocalIoReleaseAuthorization::reauthenticate_after_mode_normalization,
    )?;
    publish_normalized(normalized, clocks, &mut verify)
}

fn publish_normalized(
    normalized: SignedModeNormalizedRelease<'_>,
    clocks: PublicationClocks,
    verify: &mut impl FnMut(
        &std::path::Path,
        &File,
        &ReleaseManifestV1,
        crate::distribution::install_state::ExecutableReleaseBinding,
    ) -> Result<DeveloperIdVerification, PreparedReleaseError>,
) -> Result<PreparedReleaseOutcome, PreparedReleaseError> {
    let SignedModeNormalizedRelease {
        authentication,
        tree,
        manifest_bytes,
        manifest,
        profile,
        developer_id,
    } = normalized;
    drop(profile);
    let pending = authentication.stage_normalized_prepared_version(
        developer_id,
        tree,
        &manifest_bytes,
        &manifest,
    )?;
    finish_state(
        authentication,
        pending,
        manifest_bytes,
        manifest,
        false,
        clocks,
        verify,
    )
}

fn finish_state(
    mut authentication: PostLocalIoReleaseAuthorization<'_>,
    mut state: PreparedVersionState,
    manifest_bytes: Box<[u8]>,
    manifest: ReleaseManifestV1,
    already_published: bool,
    mut clocks: PublicationClocks,
    verify: &mut impl FnMut(
        &std::path::Path,
        &File,
        &ReleaseManifestV1,
        crate::distribution::install_state::ExecutableReleaseBinding,
    ) -> Result<DeveloperIdVerification, PreparedReleaseError>,
) -> Result<PreparedReleaseOutcome, PreparedReleaseError> {
    let version = manifest.version().as_str().to_owned();
    let was_published = matches!(state, PreparedVersionState::Published(_));
    if !was_published {
        authentication.verify_prepared_version_tree(&state, &manifest_bytes, &manifest)?;
        let developer_id = authentication.with_prepared_executable(
            &state,
            &manifest_bytes,
            &manifest,
            |path, file, binding| verify(path, file, &manifest, binding),
        )?;
        let (mut current_authentication, publication) =
            clocks.before_publication(authentication)?;
        let published = clocks.selector_boundary(
            &mut current_authentication,
            publication,
            state,
            &manifest_bytes,
            &manifest,
            developer_id,
        )?;
        authentication = current_authentication;
        state = PreparedVersionState::Published(published);
    }

    let postcommit = || -> Result<AuthenticatedPreparedVersion, PreparedReleaseError> {
        authentication.verify_prepared_version_tree(&state, &manifest_bytes, &manifest)?;
        let developer_id = authentication.with_prepared_executable(
            &state,
            &manifest_bytes,
            &manifest,
            |path, file, binding| verify(path, file, &manifest, binding),
        )?;
        clocks.authenticate_after_publication(
            authentication,
            state,
            &manifest_bytes,
            &manifest,
            developer_id,
        )
    };
    let authenticated = postcommit().map_err(|error| error.after_prepared_commit(&version))?;
    if already_published || was_published {
        Ok(PreparedReleaseOutcome::AlreadyPrepared(authenticated))
    } else {
        Ok(PreparedReleaseOutcome::Prepared(authenticated))
    }
}

#[cfg(test)]
pub(in crate::distribution) fn prepare_release_for_test(
    bundle: VerifiedReleaseBundle<'_>,
) -> Result<PreparedReleaseOutcome, PreparedReleaseError> {
    prepare_release_with(
        bundle,
        PublicationClocks::System,
        |_path, _file, _manifest, binding| Ok(DeveloperIdVerification::for_test(binding)),
    )
}

enum PublicationClocks {
    System,
    #[cfg(test)]
    Scripted {
        before_publication: Vec<jiff::Timestamp>,
        selector_boundary: Vec<jiff::Timestamp>,
        after_publication: Vec<jiff::Timestamp>,
    },
}

impl PublicationClocks {
    fn before_publication<'a>(
        &mut self,
        authentication: PostLocalIoReleaseAuthorization<'a>,
    ) -> Result<
        (
            PostLocalIoReleaseAuthorization<'a>,
            crate::distribution::update_auth::PreparedPublicationAuthorization,
        ),
        crate::distribution::update_auth::ArtifactFetchAuthorizationError,
    > {
        match self {
            Self::System => authentication.reauthenticate_before_prepared_publication(),
            #[cfg(test)]
            Self::Scripted {
                before_publication, ..
            } => authentication
                .reauthenticate_before_prepared_publication_for_test(before_publication.drain(..)),
        }
    }

    fn selector_boundary(
        &mut self,
        authentication: &mut PostLocalIoReleaseAuthorization<'_>,
        publication: crate::distribution::update_auth::PreparedPublicationAuthorization,
        state: PreparedVersionState,
        manifest_bytes: &[u8],
        manifest: &ReleaseManifestV1,
        developer_id: DeveloperIdVerification,
    ) -> Result<
        crate::distribution::install_state::PublishedPreparedVersion,
        crate::distribution::update_auth::PreparedVersionCommitError,
    > {
        match self {
            Self::System => authentication.publish_pending_prepared_version(
                publication,
                state,
                manifest_bytes,
                manifest,
                developer_id,
            ),
            #[cfg(test)]
            Self::Scripted {
                selector_boundary, ..
            } => authentication.publish_pending_prepared_version_for_test(
                publication,
                state,
                manifest_bytes,
                manifest,
                developer_id,
                selector_boundary.drain(..),
            ),
        }
    }

    fn authenticate_after_publication(
        self,
        authentication: PostLocalIoReleaseAuthorization<'_>,
        state: PreparedVersionState,
        manifest_bytes: &[u8],
        manifest: &ReleaseManifestV1,
        developer_id: DeveloperIdVerification,
    ) -> Result<AuthenticatedPreparedVersion, PreparedReleaseError> {
        match self {
            Self::System => {
                let final_authorization =
                    authentication.reauthenticate_after_prepared_publication()?;
                Ok(final_authorization.authenticate_published_version(
                    state,
                    manifest_bytes,
                    manifest,
                    developer_id,
                )?)
            }
            #[cfg(test)]
            Self::Scripted {
                mut after_publication,
                ..
            } => {
                let final_sample = after_publication
                    .pop()
                    .ok_or(crate::distribution::update_auth::TufVerifierError::ClockRollback)
                    .map_err(
                        crate::distribution::update_auth::ArtifactFetchAuthorizationError::from,
                    )?;
                let final_authorization = authentication
                    .reauthenticate_after_prepared_publication_for_test(after_publication)?;
                Ok(final_authorization.authenticate_published_version_for_test(
                    state,
                    manifest_bytes,
                    manifest,
                    developer_id,
                    [final_sample],
                )?)
            }
        }
    }
}

#[cfg(test)]
pub(in crate::distribution) fn prepare_release_for_test_with_clocks(
    bundle: VerifiedReleaseBundle<'_>,
    before_publication: Vec<jiff::Timestamp>,
    selector_boundary: Vec<jiff::Timestamp>,
    after_publication: Vec<jiff::Timestamp>,
) -> Result<PreparedReleaseOutcome, PreparedReleaseError> {
    prepare_release_with(
        bundle,
        PublicationClocks::Scripted {
            before_publication,
            selector_boundary,
            after_publication,
        },
        |_path, _file, _manifest, binding| Ok(DeveloperIdVerification::for_test(binding)),
    )
}
