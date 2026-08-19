//! Origin-locked, exact-byte update transport.
//!
//! This sibling keeps HTTP out of `update_auth`. It accepts only one-use
//! verifier requests or sealed authenticated target capabilities. Metadata is
//! committed only after the transport-free verifier yields a sealed candidate;
//! artifact results remain inert exact-byte proofs. It owns no CLI,
//! extraction, prepared-version, or activation authority.

mod fetch;
mod http;
mod metadata;
mod origin;

#[cfg(test)]
mod tests;

use crate::distribution::install_state::{ArtifactStageError, VerifiedArchiveFile};
use crate::distribution::schema::{ReleaseManifestError, ReleaseManifestV1};
use crate::distribution::update_auth::{
    ArtifactFetchAuthorizationError, FinalArtifactAuthorization, TufVerifierError,
};

#[derive(Debug, thiserror::Error)]
pub(in crate::distribution) enum UpdateTransportError {
    #[error("the update HTTP client could not be constructed")]
    Client(#[source] reqwest::Error),
    #[error("the update request failed")]
    Network(#[source] reqwest::Error),
    #[error("the update response did not match the origin policy")]
    OriginPolicy,
    #[error("the update response used an unsupported HTTP status")]
    Status,
    #[error("the update response headers are malformed or conflicting")]
    Headers,
    #[error("the update response used a transformed content encoding")]
    ContentEncoding,
    #[error("the update response length does not match signed metadata")]
    Length,
    #[error("the update response digest does not match signed metadata")]
    Digest,
    #[error("the release manifest identity does not match the authenticated release")]
    ManifestIdentity,
    #[error("the update response body could not be read")]
    BodyRead,
    #[error(transparent)]
    Stage(#[from] ArtifactStageError),
    #[error(transparent)]
    Manifest(#[from] ReleaseManifestError),
    #[error(transparent)]
    Authentication(#[from] TufVerifierError),
    #[error(transparent)]
    FetchAuthorization(#[from] ArtifactFetchAuthorizationError),
}

/// Exact external manifest and archive bytes for one authenticated plan.
///
/// This is deliberately not prepared-version or installation authority. A
/// later lock-held coordinator must reauthenticate the selected generation,
/// validate the embedded manifest and ZIP inventory, verify code signing,
/// and durably publish a version before activation is possible.
pub(in crate::distribution) struct VerifiedReleaseBundle<'a> {
    authorization: FinalArtifactAuthorization<'a>,
    manifest_bytes: Box<[u8]>,
    manifest: ReleaseManifestV1,
    archive: VerifiedArchiveFile,
}

/// One-use transport output consumed only by the prepared-release boundary.
///
/// This remains inert transport data. In particular, it contains no archive
/// profile, extracted-tree proof, publication capability, or activation
/// authority.
pub(super) struct ReleasePreparationParts<'a> {
    pub(super) authorization: FinalArtifactAuthorization<'a>,
    pub(super) manifest_bytes: Box<[u8]>,
    pub(super) manifest: ReleaseManifestV1,
    pub(super) archive: VerifiedArchiveFile,
}

impl std::fmt::Debug for VerifiedReleaseBundle<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VerifiedReleaseBundle")
            .finish_non_exhaustive()
    }
}

impl<'a> VerifiedReleaseBundle<'a> {
    pub(super) fn preparation_parts_mut(
        &mut self,
    ) -> (&[u8], &ReleaseManifestV1, &mut VerifiedArchiveFile) {
        (&self.manifest_bytes, &self.manifest, &mut self.archive)
    }

    pub(super) fn into_preparation_parts(self) -> ReleasePreparationParts<'a> {
        let Self {
            authorization,
            manifest_bytes,
            manifest,
            archive,
        } = self;
        ReleasePreparationParts {
            authorization,
            manifest_bytes,
            manifest,
            archive,
        }
    }

    #[cfg(test)]
    pub(in crate::distribution) fn from_test_parts(
        authorization: FinalArtifactAuthorization<'a>,
        manifest_bytes: Box<[u8]>,
        manifest: ReleaseManifestV1,
        archive: VerifiedArchiveFile,
    ) -> Self {
        Self {
            authorization,
            manifest_bytes,
            manifest,
            archive,
        }
    }
}
