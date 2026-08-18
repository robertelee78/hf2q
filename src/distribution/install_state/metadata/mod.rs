//! Crash-durable authenticated update-metadata state.
//!
//! The module is deliberately dormant: no network verifier or CLI path can
//! construct a production candidate yet.  It first freezes the exact local
//! wire contract and capability split needed by the signed-update adapter.

mod journal;
mod schema;

#[cfg(test)]
mod tests;

use jiff::Timestamp;

use super::{ExplicitRootAuthorization, InstallStateError};

#[derive(Debug, thiserror::Error)]
pub(super) enum MetadataJournalError {
    #[error("invalid authenticated metadata journal: {0}")]
    Invalid(&'static str),
    #[error("metadata generation {sequence} was committed, but final durability is unknown")]
    CommittedDurabilityUnknown {
        sequence: u64,
        #[source]
        source: Box<MetadataJournalError>,
    },
    #[error(transparent)]
    InstallState(#[from] InstallStateError),
}

impl MetadataJournalError {
    fn after_commit(self, sequence: u64) -> Self {
        Self::CommittedDurabilityUnknown {
            sequence,
            source: Box::new(self),
        }
    }
}

/// Exact, bounded role bytes authenticated by the future verifier.
#[derive(Debug)]
pub(super) struct ExactMetadataRole {
    request_name: String,
    version: u64,
    bytes: Box<[u8]>,
}

/// Fresh metadata evidence. Parsing JSON cannot construct this capability.
///
/// The production constructor will live only in the tokenized transport-free
/// TUF verifier. This schema slice intentionally exposes no such constructor.
#[derive(Debug)]
pub(super) struct VerifiedMetadataCandidate {
    installation_id: String,
    state_root: String,
    repository_id: String,
    channel: String,
    verification_started_at: Timestamp,
    verification_completed_at: Timestamp,
    anchor_root: ExactMetadataRole,
    root_chain: Vec<ExactMetadataRole>,
    trusted_root: ExactMetadataRole,
    timestamp: ExactMetadataRole,
    snapshot: ExactMetadataRole,
    targets: ExactMetadataRole,
}

/// Explicit association between one state root and one installation identity.
///
/// The production constructor will be added only when setup/ownership state
/// can supply a live validated installation ID. A path alone cannot make a
/// copied metadata journal authoritative for another installation.
#[derive(Debug)]
pub(super) struct MetadataStateAuthorization {
    root: ExplicitRootAuthorization,
    installation_id: String,
}

#[cfg(test)]
impl MetadataStateAuthorization {
    fn for_test(root: ExplicitRootAuthorization, installation_id: &str) -> Self {
        Self {
            root,
            installation_id: installation_id.to_owned(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MetadataCommitOutcome {
    Committed { sequence: u64 },
    AlreadyCommitted { sequence: u64 },
}

/// Structurally complete selected bytes. This is not cryptographic authority;
/// the future verifier must reconstruct a durable baseline from these exact
/// bytes before it may issue any metadata request.
#[derive(Debug)]
pub(super) struct StoredMetadataGeneration {
    pub(super) sequence: u64,
    pub(super) generation_receipt: Box<[u8]>,
    pub(super) anchor_root: Box<[u8]>,
    pub(super) root_chain: Vec<Box<[u8]>>,
    pub(super) trusted_root: Box<[u8]>,
    pub(super) timestamp: Box<[u8]>,
    pub(super) snapshot: Box<[u8]>,
    pub(super) targets: Box<[u8]>,
}
