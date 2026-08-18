//! Candidate-neutral observations used only by the spike.

use jiff::Timestamp;
use sha2::{Digest, Sha256};
use thiserror::Error;

pub(crate) const MAX_ROOT_BYTES: usize = 1024 * 1024;
pub(crate) const MAX_TIMESTAMP_BYTES: usize = 1024 * 1024;
pub(crate) const MAX_SNAPSHOT_BYTES: usize = 1024 * 1024;
pub(crate) const MAX_TARGETS_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RoleKind {
    Root,
    Timestamp,
    Snapshot,
    Targets,
}

impl RoleKind {
    pub(crate) fn max_bytes(self) -> usize {
        match self {
            Self::Root => MAX_ROOT_BYTES,
            Self::Timestamp => MAX_TIMESTAMP_BYTES,
            Self::Snapshot => MAX_SNAPSHOT_BYTES,
            Self::Targets => MAX_TARGETS_BYTES,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RoleFloor {
    pub(crate) version: u64,
    pub(crate) raw_sha256: [u8; 32],
}

impl RoleFloor {
    pub(crate) fn from_bytes(version: u64, bytes: &[u8]) -> Self {
        Self {
            version,
            raw_sha256: sha256(bytes),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CommittedGeneration {
    pub(crate) update_start_floor: Timestamp,
    pub(crate) root: RoleFloor,
    pub(crate) timestamp: RoleFloor,
    pub(crate) snapshot: RoleFloor,
    pub(crate) targets: RoleFloor,
    pub(crate) raw_root: Vec<u8>,
    pub(crate) raw_timestamp: Vec<u8>,
    pub(crate) raw_snapshot: Vec<u8>,
    pub(crate) raw_targets: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CapturedRole {
    pub(crate) request_name: String,
    pub(crate) version: u64,
    pub(crate) raw: Vec<u8>,
    pub(crate) raw_sha256: [u8; 32],
}

#[derive(Debug, Error)]
pub(crate) enum SpikeError {
    #[error("clock moved below the committed floor")]
    ClockRollback,
    #[error("candidate metadata is malformed")]
    MalformedMetadata,
    #[error("metadata contains a duplicate JSON key")]
    DuplicateJsonKey,
    #[error("metadata exceeds the absolute role limit")]
    MetadataTooLarge,
    #[error("transport transcript violates the top-level-only policy")]
    TransportPolicy,
    #[error("candidate verifier rejected the repository")]
    CandidateRejected,
    #[error("captured raw metadata does not match the verifier result")]
    CorrelationMismatch,
    #[error("metadata role rolled back below its committed version floor")]
    VersionRollback,
    #[error("equal-version metadata is not byte-identical to the committed envelope")]
    EqualVersionChangedBytes,
    #[error("metadata expired at the wrapper's sampled update time")]
    ExpiredAtWrapperTime,
    #[error("top-level targets contains a delegation block")]
    DelegationsForbidden,
    #[error("application target descriptors do not cross-bind")]
    ApplicationBinding,
    #[error("scratch datastore did not preserve a bounded clock sample")]
    InvalidCandidateClock,
    #[error("spike filesystem operation failed")]
    Io(#[from] std::io::Error),
}

pub(crate) fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

pub(crate) fn check_floor(role: &CapturedRole, floor: &RoleFloor) -> Result<(), SpikeError> {
    if role.version < floor.version {
        return Err(SpikeError::VersionRollback);
    }
    if role.version == floor.version && role.raw_sha256 != floor.raw_sha256 {
        return Err(SpikeError::EqualVersionChangedBytes);
    }
    Ok(())
}
