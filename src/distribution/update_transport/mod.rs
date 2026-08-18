//! Origin-locked, exact-byte update artifact transport.
//!
//! This sibling keeps HTTP out of `update_auth`. It accepts only sealed,
//! authenticated target capabilities and returns inert exact-byte proofs; it
//! owns no CLI, extraction, prepared-version, or activation authority.

mod fetch;
mod http;
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
pub(in crate::distribution) struct VerifiedReleaseBundle {
    authorization: FinalArtifactAuthorization,
    manifest_bytes: Box<[u8]>,
    manifest: ReleaseManifestV1,
    archive: VerifiedArchiveFile,
}

impl std::fmt::Debug for VerifiedReleaseBundle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VerifiedReleaseBundle")
            .finish_non_exhaustive()
    }
}

impl VerifiedReleaseBundle {
    pub(super) fn preparation_parts_mut(
        &mut self,
    ) -> (&[u8], &ReleaseManifestV1, &mut VerifiedArchiveFile) {
        (&self.manifest_bytes, &self.manifest, &mut self.archive)
    }
}
