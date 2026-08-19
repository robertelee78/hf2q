use super::*;
use crate::distribution::schema::FirstStandaloneInstallRecord;

#[derive(Debug, thiserror::Error)]
pub(in crate::distribution) enum PreparedVersionError {
    #[error("the prepared-version transaction does not match the authenticated release")]
    Integrity,
    #[error(transparent)]
    Extraction(#[from] ExtractionError),
    #[error("prepared version {version} was published, but final durability is unknown")]
    PublishedDurabilityUnknown {
        version: String,
        #[source]
        source: Box<PreparedVersionError>,
    },
}

impl PreparedVersionError {
    pub(super) fn after_commit(self, version: &str) -> Self {
        Self::PublishedDurabilityUnknown {
            version: version.to_owned(),
            source: Box::new(self),
        }
    }
}

impl From<crate::distribution::install_state::InstallStateError> for PreparedVersionError {
    fn from(error: crate::distribution::install_state::InstallStateError) -> Self {
        Self::Extraction(ExtractionError::from(error))
    }
}

pub(in crate::distribution) enum PreparedVersionState {
    Pending(PendingPreparedVersion),
    Published(PublishedPreparedVersion),
}

pub(in crate::distribution) struct PendingPreparedVersion {
    pub(super) prepared: Directory,
    pub(super) versions: Directory,
    pub(super) tree: Directory,
    pub(super) name: String,
    pub(super) record: FirstStandaloneInstallRecord,
}

pub(in crate::distribution) struct PublishedPreparedVersion {
    pub(super) versions: Directory,
    pub(super) tree: Directory,
    pub(super) record: FirstStandaloneInstallRecord,
}

/// Fully reopened and media-synced published tree. This remains structural
/// evidence until update-auth supplies its final post-I/O freshness token.
pub(in crate::distribution) struct VerifiedPublishedPreparedVersion {
    pub(super) receipt_bytes: Vec<u8>,
}

impl std::fmt::Debug for PreparedVersionState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending(_) => formatter.write_str("PreparedVersionState::Pending(..)"),
            Self::Published(_) => formatter.write_str("PreparedVersionState::Published(..)"),
        }
    }
}

impl std::fmt::Debug for PendingPreparedVersion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PendingPreparedVersion")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for PublishedPreparedVersion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PublishedPreparedVersion")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for VerifiedPublishedPreparedVersion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VerifiedPublishedPreparedVersion")
            .finish_non_exhaustive()
    }
}

impl VerifiedPublishedPreparedVersion {
    pub(in crate::distribution) fn receipt_bytes(&self) -> &[u8] {
        &self.receipt_bytes
    }

    pub(in crate::distribution) fn authenticate(
        self,
        authorization: crate::distribution::update_auth::PreparedActivationAuthorization,
    ) -> Result<
        crate::distribution::install_state::AuthenticatedPreparedVersion,
        PreparedVersionError,
    > {
        if !authorization.matches_receipt(&self.receipt_bytes) {
            return Err(PreparedVersionError::Integrity);
        }
        Ok(
            crate::distribution::install_state::AuthenticatedPreparedVersion {
                receipt_bytes: self.receipt_bytes,
            },
        )
    }
}
