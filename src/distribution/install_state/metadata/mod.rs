//! Crash-durable authenticated update-metadata state.
//!
//! The module is deliberately dormant: the transport-free TUF verifier can
//! construct a metadata-only candidate, but no network transport, target
//! lookup, installer, or CLI path can use it yet. This module owns only the
//! exact local wire contract and crash-durable journal capability split.

mod journal;
pub(in crate::distribution) mod schema;

pub(in crate::distribution) use journal::{
    lock_metadata_state, read_selected, LockedMetadataState,
};

#[cfg(test)]
pub(in crate::distribution) use journal::{commit_candidate_for_test, Barrier, FaultPlan};

#[cfg(test)]
mod tests;

#[cfg(test)]
use super::{
    bootstrap_installation_identity_for_test, ExplicitRootAuthorization, IdentityFaultPlan,
};
use super::{DurableInstallationIdentity, InstallStateError};
pub(super) use crate::distribution::update_auth::{ExactMetadataRole, VerifiedMetadataCandidate};

#[derive(Debug, thiserror::Error)]
pub(in crate::distribution) enum MetadataJournalError {
    #[error("invalid authenticated metadata journal: {0}")]
    Invalid(&'static str),
    #[error("metadata generation {sequence} was committed, but final durability is unknown")]
    CommittedDurabilityUnknown {
        sequence: u64,
        #[source]
        source: Box<MetadataJournalError>,
    },
    #[error("verified metadata failed its lock-held precommit gate")]
    PrecommitRejected,
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

/// Live association between one state root and its durable identity inode.
///
/// A path or copied UUID cannot construct this capability, and disk bytes do
/// not self-authorize a copied metadata journal for another installation.
#[derive(Debug)]
pub(in crate::distribution) struct MetadataStateAuthorization {
    pub(super) identity: DurableInstallationIdentity,
}

#[cfg(test)]
impl MetadataStateAuthorization {
    pub(crate) fn for_test(root: ExplicitRootAuthorization, installation_id: &str) -> Self {
        Self::from_identity(
            bootstrap_installation_identity_for_test(
                root,
                installation_id,
                IdentityFaultPlan::default(),
            )
            .expect("bootstrap test installation identity")
            .into_identity(),
        )
    }

    pub(crate) fn for_test_path(root: &std::path::Path, installation_id: &str) -> Self {
        Self::for_test(
            ExplicitRootAuthorization::new(root).expect("explicit test root authorization"),
            installation_id,
        )
    }
}

impl MetadataStateAuthorization {
    pub(in crate::distribution) fn from_identity(identity: DurableInstallationIdentity) -> Self {
        Self { identity }
    }

    pub(in crate::distribution) fn installation_id(&self) -> &str {
        self.identity.installation_id().as_str()
    }

    pub(in crate::distribution) fn state_root(&self) -> &str {
        self.identity.state_root().as_str()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::distribution) enum MetadataCommitOutcome {
    Committed { sequence: u64 },
    AlreadyCommitted { sequence: u64 },
}

/// Result of lock-held restart cleanup for a transaction that never became
/// selected. Cleanup never authenticates or promotes its bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::distribution) enum MetadataRestartCleanup {
    Clean,
    DiscardedUnselected { sequence: u64 },
}

/// Structurally complete selected bytes. This is not cryptographic authority;
/// the TUF verifier must reconstruct a durable floor from these exact bytes
/// and the compiled anchor before it may issue any metadata request.
#[derive(Debug)]
pub(in crate::distribution) struct StoredMetadataGeneration {
    sequence: u64,
    generation_receipt: Box<[u8]>,
    anchor_root: Box<[u8]>,
    root_chain: Vec<Box<[u8]>>,
    trusted_root: Box<[u8]>,
    timestamp: Box<[u8]>,
    snapshot: Box<[u8]>,
    targets: Box<[u8]>,
}

impl StoredMetadataGeneration {
    pub(in crate::distribution) fn sequence(&self) -> u64 {
        self.sequence
    }

    pub(in crate::distribution) fn generation_receipt(&self) -> &[u8] {
        &self.generation_receipt
    }

    pub(in crate::distribution) fn anchor_root(&self) -> &[u8] {
        &self.anchor_root
    }

    pub(in crate::distribution) fn root_chain(&self) -> &[Box<[u8]>] {
        &self.root_chain
    }

    pub(in crate::distribution) fn trusted_root(&self) -> &[u8] {
        &self.trusted_root
    }

    pub(in crate::distribution) fn timestamp(&self) -> &[u8] {
        &self.timestamp
    }

    pub(in crate::distribution) fn snapshot(&self) -> &[u8] {
        &self.snapshot
    }

    pub(in crate::distribution) fn targets(&self) -> &[u8] {
        &self.targets
    }

    #[allow(clippy::type_complexity)]
    pub(in crate::distribution) fn into_authenticated_bytes(
        self,
    ) -> (
        Box<[u8]>,
        Vec<Box<[u8]>>,
        Box<[u8]>,
        Box<[u8]>,
        Box<[u8]>,
        Box<[u8]>,
    ) {
        (
            self.anchor_root,
            self.root_chain,
            self.trusted_root,
            self.timestamp,
            self.snapshot,
            self.targets,
        )
    }
}
