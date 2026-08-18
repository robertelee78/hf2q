use jiff::Timestamp;

use super::verifier::VerificationState;
use super::TufVerifierError;

pub(in crate::distribution) const MAX_ROOT_BYTES: usize = 1024 * 1024;
pub(in crate::distribution) const MAX_TIMESTAMP_BYTES: usize = 1024 * 1024;
pub(in crate::distribution) const MAX_SNAPSHOT_BYTES: usize = 1024 * 1024;
pub(in crate::distribution) const MAX_TARGETS_BYTES: usize = 4 * 1024 * 1024;
pub(in crate::distribution) const MAX_ROOT_ROTATIONS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MetadataRoleKind {
    Root,
    Timestamp,
    Snapshot,
    Targets,
}

impl MetadataRoleKind {
    pub(super) const fn maximum_bytes(self) -> usize {
        match self {
            Self::Root | Self::Timestamp | Self::Snapshot => MAX_ROOT_BYTES,
            Self::Targets => MAX_TARGETS_BYTES,
        }
    }
}

/// Exact signed bytes and the request identity that authenticated them.
#[derive(Debug)]
pub(in crate::distribution) struct ExactMetadataRole {
    request_name: String,
    version: u64,
    bytes: Box<[u8]>,
}

impl ExactMetadataRole {
    pub(super) fn new(request_name: String, version: u64, bytes: Box<[u8]>) -> Self {
        Self {
            request_name,
            version,
            bytes,
        }
    }

    pub(in crate::distribution) fn request_name(&self) -> &str {
        &self.request_name
    }

    pub(in crate::distribution) fn version(&self) -> u64 {
        self.version
    }

    pub(in crate::distribution) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(super) fn exactly_matches(&self, other: &Self) -> bool {
        self.request_name == other.request_name
            && self.version == other.version
            && self.bytes == other.bytes
    }

    #[cfg(test)]
    pub(crate) fn for_test(request_name: &str, version: u64, bytes: Vec<u8>) -> Self {
        Self::new(request_name.to_owned(), version, bytes.into_boxed_slice())
    }

    #[cfg(test)]
    pub(crate) fn set_request_name_for_test(&mut self, request_name: &str) {
        self.request_name = request_name.to_owned();
    }
}

/// Fresh metadata evidence. JSON parsing cannot construct this capability.
///
/// This authorizes only a metadata-journal commit. It exposes no target lookup
/// and cannot construct an installed-version capability.
#[derive(Debug)]
pub(in crate::distribution) struct VerifiedMetadataCandidate {
    installation_id: String,
    state_root: String,
    repository_id: String,
    channel: String,
    verification_started_at: Timestamp,
    verification_completed_at: Timestamp,
    anchor_root: ExactMetadataRole,
    root_chain: Vec<ExactMetadataRole>,
    trusted_root: ExactMetadataRole,
    /// Prior selected root that authorized the TUF step-11 online-role reset.
    timestamp_snapshot_floor_reset_from_root_version: Option<u64>,
    timestamp: ExactMetadataRole,
    snapshot: ExactMetadataRole,
    targets: ExactMetadataRole,
}

impl VerifiedMetadataCandidate {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        installation_id: String,
        state_root: String,
        verification_started_at: Timestamp,
        verification_completed_at: Timestamp,
        anchor_root: ExactMetadataRole,
        root_chain: Vec<ExactMetadataRole>,
        trusted_root: ExactMetadataRole,
        timestamp_snapshot_floor_reset_from_root_version: Option<u64>,
        timestamp: ExactMetadataRole,
        snapshot: ExactMetadataRole,
        targets: ExactMetadataRole,
    ) -> Self {
        Self {
            installation_id,
            state_root,
            repository_id: "hf2q".to_owned(),
            channel: "stable".to_owned(),
            verification_started_at,
            verification_completed_at,
            anchor_root,
            root_chain,
            trusted_root,
            timestamp_snapshot_floor_reset_from_root_version,
            timestamp,
            snapshot,
            targets,
        }
    }

    pub(in crate::distribution) fn installation_id(&self) -> &str {
        &self.installation_id
    }

    pub(in crate::distribution) fn state_root(&self) -> &str {
        &self.state_root
    }

    pub(in crate::distribution) fn repository_id(&self) -> &str {
        &self.repository_id
    }

    pub(in crate::distribution) fn channel(&self) -> &str {
        &self.channel
    }

    pub(in crate::distribution) fn verification_started_at(&self) -> Timestamp {
        self.verification_started_at
    }

    pub(in crate::distribution) fn verification_completed_at(&self) -> Timestamp {
        self.verification_completed_at
    }

    pub(in crate::distribution) fn anchor_root(&self) -> &ExactMetadataRole {
        &self.anchor_root
    }

    pub(in crate::distribution) fn root_chain(&self) -> &[ExactMetadataRole] {
        &self.root_chain
    }

    pub(in crate::distribution) fn trusted_root(&self) -> &ExactMetadataRole {
        &self.trusted_root
    }

    pub(in crate::distribution) fn timestamp_snapshot_floor_reset_from_root(
        &self,
    ) -> Option<&ExactMetadataRole> {
        let version = self.timestamp_snapshot_floor_reset_from_root_version?;
        if self.anchor_root.version() == version {
            return Some(&self.anchor_root);
        }
        self.root_chain
            .iter()
            .find(|root| root.version() == version)
    }

    pub(in crate::distribution) fn timestamp(&self) -> &ExactMetadataRole {
        &self.timestamp
    }

    pub(in crate::distribution) fn snapshot(&self) -> &ExactMetadataRole {
        &self.snapshot
    }

    pub(in crate::distribution) fn targets(&self) -> &ExactMetadataRole {
        &self.targets
    }

    pub(super) fn exactly_matches(&self, other: &Self) -> bool {
        self.installation_id == other.installation_id
            && self.state_root == other.state_root
            && self.repository_id == other.repository_id
            && self.channel == other.channel
            && self.verification_started_at == other.verification_started_at
            && self.verification_completed_at == other.verification_completed_at
            && self.anchor_root.exactly_matches(&other.anchor_root)
            && self.root_chain.len() == other.root_chain.len()
            && self
                .root_chain
                .iter()
                .zip(&other.root_chain)
                .all(|(left, right)| left.exactly_matches(right))
            && self.trusted_root.exactly_matches(&other.trusted_root)
            && self.timestamp_snapshot_floor_reset_from_root_version
                == other.timestamp_snapshot_floor_reset_from_root_version
            && self.timestamp.exactly_matches(&other.timestamp)
            && self.snapshot.exactly_matches(&other.snapshot)
            && self.targets.exactly_matches(&other.targets)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_test(
        installation_id: String,
        state_root: String,
        verification_started_at: Timestamp,
        verification_completed_at: Timestamp,
        anchor_root: ExactMetadataRole,
        root_chain: Vec<ExactMetadataRole>,
        trusted_root: ExactMetadataRole,
        timestamp: ExactMetadataRole,
        snapshot: ExactMetadataRole,
        targets: ExactMetadataRole,
    ) -> Self {
        Self::new(
            installation_id,
            state_root,
            verification_started_at,
            verification_completed_at,
            anchor_root,
            root_chain,
            trusted_root,
            None,
            timestamp,
            snapshot,
            targets,
        )
    }

    #[cfg(test)]
    pub(crate) fn set_state_root_for_test(&mut self, state_root: &str) {
        self.state_root = state_root.to_owned();
    }

    #[cfg(test)]
    pub(crate) fn replace_root_for_test(&mut self, index: usize, role: ExactMetadataRole) {
        self.root_chain[index] = role;
    }

    #[cfg(test)]
    pub(crate) fn set_timestamp_snapshot_floor_reset_for_test(&mut self, version: Option<u64>) {
        self.timestamp_snapshot_floor_reset_from_root_version = version;
    }

    #[cfg(test)]
    pub(crate) fn snapshot_mut_for_test(&mut self) -> &mut ExactMetadataRole {
        &mut self.snapshot
    }
}

/// One exact request derived from already-authenticated parent metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RequestSpec {
    kind: MetadataRoleKind,
    relative_name: String,
    maximum_bytes: usize,
}

impl RequestSpec {
    pub(super) fn new(kind: MetadataRoleKind, relative_name: String) -> Self {
        Self {
            kind,
            relative_name,
            maximum_bytes: kind.maximum_bytes(),
        }
    }

    pub(super) fn role(&self) -> MetadataRoleKind {
        self.kind
    }

    pub(super) fn relative_name(&self) -> &str {
        &self.relative_name
    }

    pub(super) fn maximum_bytes(&self) -> usize {
        self.maximum_bytes
    }
}

/// The transport must consume this token with exactly one response. Keeping
/// verifier state inside the token makes stale and out-of-order reuse
/// unrepresentable without unsafe code.
#[derive(Debug)]
pub(super) struct PendingMetadataRequest {
    pub(super) state: VerificationState,
    pub(super) spec: RequestSpec,
}

impl PendingMetadataRequest {
    pub(super) fn spec(&self) -> &RequestSpec {
        &self.spec
    }

    pub(super) fn respond(
        self,
        response: MetadataResponse,
    ) -> Result<VerificationStep, TufVerifierError> {
        super::verifier::respond(self, response)
    }
}

#[derive(Debug)]
pub(super) enum MetadataResponse {
    Found(Box<[u8]>),
    ConfirmedNotFound,
}

#[derive(Debug)]
pub(super) enum VerificationStep {
    Request(PendingMetadataRequest),
    Candidate(VerifiedMetadataCandidate),
}

/// Pinned root bytes compiled into a future release binary. The constructor is
/// crate-private and accepts only a static byte slice; runtime input never
/// becomes a trust anchor.
#[derive(Debug)]
pub(super) struct EmbeddedTrustRoot {
    bytes: &'static [u8],
}

impl EmbeddedTrustRoot {
    pub(super) const fn from_compiled(bytes: &'static [u8]) -> Self {
        Self { bytes }
    }

    pub(super) const fn bytes(&self) -> &'static [u8] {
        self.bytes
    }
}
