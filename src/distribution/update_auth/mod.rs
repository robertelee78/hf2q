//! Transport-free signed update-metadata authentication.
//!
//! This bounded context deliberately owns no HTTP client, URL, retry policy,
//! target lookup, install path, or CLI surface. It turns one exact, bounded
//! sequence of TUF responses into a sealed journal candidate. It owns no
//! generic or caller-directed target lookup. A parsed role or
//! a structurally valid journal generation is never authority by itself.

mod artifact_authorization;
mod commit;
mod model;
mod profile;
mod replay;
mod strict_json;
mod target_set;
mod verifier;

#[cfg(test)]
mod test_repository;
#[cfg(test)]
mod tests;

pub(in crate::distribution) use artifact_authorization::{
    ArchiveStageAuthorization, ArtifactFetchAuthorization, ArtifactFetchAuthorizationError,
    BoundArtifactFetchAuthorization, ExtractionStageAuthorization, FinalArtifactAuthorization,
    PostLocalIoReleaseAuthorization,
};
pub(in crate::distribution) use commit::AdvancingCommitGuard;
pub(in crate::distribution) use model::{ExactMetadataRole, VerifiedMetadataCandidate};
pub(in crate::distribution) use target_set::AuthenticatedTargetDescriptor;

#[derive(Debug, thiserror::Error)]
pub(in crate::distribution) enum TufVerifierError {
    #[error("signed metadata exceeds its role byte bound")]
    MetadataSize,
    #[error("signed metadata contains a duplicate JSON key")]
    DuplicateJsonKey,
    #[error("signed metadata is malformed or outside the v1 profile")]
    MalformedMetadata,
    #[error("signed metadata authentication failed")]
    AuthenticationFailed,
    #[error("the trusted metadata anchor does not match the compiled root")]
    AnchorMismatch,
    #[error("the trusted metadata clock moved backward")]
    ClockRollback,
    #[error("signed metadata is expired at the verification reference time")]
    ExpiredMetadata,
    #[error("signed metadata moved below or equivocated with a durable floor")]
    RollbackOrEquivocation,
    #[error("the metadata response does not match the outstanding request")]
    UnexpectedResponse,
    #[error("required signed metadata was reported missing")]
    RequiredMetadataMissing,
    #[error("the v1 lifetime root-rotation bound was exceeded")]
    RootRotationLimit,
    #[error("the trusted root version cannot be incremented")]
    RootVersionExhausted,
    #[error("the verifier has not reached a complete authenticated transcript")]
    IncompleteTranscript,
    #[error("the committed metadata does not exactly match the authenticated candidate")]
    DurableCommitMismatch,
    #[error("no selected signed-metadata generation is available")]
    NoSelectedMetadata,
    #[error("the authenticated targets role is outside the stable repository profile")]
    UnsupportedTargetProfile,
    #[error("the authenticated targets inventory is malformed or incomplete")]
    TargetInventory,
    #[error("signed metadata attempted to rewrite or remove an immutable release target")]
    RetainedReleaseMutation,
    #[error("the channel pointer does not match the authenticated release targets")]
    TargetBinding,
    #[error(transparent)]
    ChannelPointer(#[from] crate::distribution::schema::ChannelPointerError),
    #[error(transparent)]
    Journal(#[from] super::install_state::metadata::MetadataJournalError),
}
