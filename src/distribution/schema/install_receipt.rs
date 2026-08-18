use std::fmt;

use serde::Serialize;

use super::common::{ReleaseVersion, SchemaValueError, Sha256Digest, TargetTriple};
use super::installed_version_marker::InstalledVersionMarkerV2;

pub const INSTALL_RECEIPT_KIND: &str = "hf2q.install-receipt";
pub const INSTALL_RECEIPT_SCHEMA_VERSION: u32 = 1;
pub const MAX_INSTALL_RECEIPT_BYTES: usize = 64 * 1024;
pub const STATE_LAYOUT_SCHEMA_V1: u32 = 1;
pub const INSTALLATION_LAYOUT_SCHEMA_V1: u32 = 1;

const MAX_ABSOLUTE_ROOT_BYTES: usize = 4096;

#[derive(Debug, thiserror::Error)]
pub enum InstallReceiptError {
    #[error("{document} exceeds the {limit}-byte input limit ({actual} bytes)")]
    InputTooLarge {
        document: &'static str,
        limit: usize,
        actual: usize,
    },
    #[error("{document} JSON is invalid at line {line}, column {column} ({category})")]
    Json {
        document: &'static str,
        line: usize,
        column: usize,
        category: &'static str,
    },
    #[error("unsupported {0} kind discriminator")]
    UnsupportedKind(&'static str),
    #[error("unsupported {document} schema version {actual}")]
    UnsupportedSchema { document: &'static str, actual: u32 },
    #[error("invalid install-state field `{field}`: {reason}")]
    InvalidField { field: &'static str, reason: String },
    #[error("install receipt contains {actual} retained releases; at most {limit} are allowed")]
    TooManyRetained { limit: usize, actual: usize },
    #[error("install receipt contains duplicate release version `{0}`")]
    DuplicateVersion(String),
    #[error("install receipt owner and update route are incompatible")]
    OwnerRouteMismatch,
    #[error("install receipt transition is inconsistent: {0}")]
    TransitionMismatch(&'static str),
    #[error("installed-version marker digest does not match the exact recorded bytes")]
    MarkerDigestMismatch,
    #[error("installed-version marker is not in hf2q's canonical byte encoding")]
    NonCanonicalMarkerEncoding,
}

impl InstallReceiptError {
    pub(crate) fn invalid(field: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidField {
            field,
            reason: reason.into(),
        }
    }
}

impl From<SchemaValueError> for InstallReceiptError {
    fn from(error: SchemaValueError) -> Self {
        Self::InvalidField {
            field: error.field,
            reason: error.reason,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct InstallationId(String);

impl InstallationId {
    pub(crate) fn parse(value: String) -> Result<Self, InstallReceiptError> {
        let parsed = uuid::Uuid::parse_str(&value).map_err(|_| {
            InstallReceiptError::invalid(
                "installation_id",
                "must be a canonical lowercase RFC 4122 version-4 UUID",
            )
        })?;
        if parsed.hyphenated().to_string() != value
            || parsed.get_version() != Some(uuid::Version::Random)
            || parsed.get_variant() != uuid::Variant::RFC4122
        {
            return Err(InstallReceiptError::invalid(
                "installation_id",
                "must be a canonical lowercase RFC 4122 version-4 UUID",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct AbsoluteInstallPath(String);

impl AbsoluteInstallPath {
    pub(crate) fn parse(field: &'static str, value: String) -> Result<Self, InstallReceiptError> {
        let suffix = value.strip_prefix('/');
        if value.len() > MAX_ABSOLUTE_ROOT_BYTES
            || value == "/"
            || value.ends_with('/')
            || value.contains("//")
            || value.contains('\\')
            || value.chars().any(char::is_control)
            || suffix.is_none_or(str::is_empty)
            || suffix.is_some_and(|path| {
                path.split('/').any(|component| {
                    component.is_empty()
                        || component == "."
                        || component == ".."
                        || component.as_bytes().len() > 255
                })
            })
        {
            return Err(InstallReceiptError::invalid(
                field,
                "must be a canonical, non-root absolute UTF-8 path",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AbsoluteInstallPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum OwnerFamily {
    #[serde(rename = "standalone")]
    Standalone,
    #[serde(rename = "homebrew")]
    Homebrew,
    #[serde(rename = "cargo-registry")]
    CargoRegistry,
    #[serde(rename = "unknown/manual")]
    UnknownManual,
}

impl OwnerFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Standalone => "standalone",
            Self::Homebrew => "homebrew",
            Self::CargoRegistry => "cargo-registry",
            Self::UnknownManual => "unknown/manual",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum UpdateRoute {
    #[serde(rename = "standalone")]
    Standalone,
    #[serde(rename = "brew")]
    Brew,
    #[serde(rename = "cargo-install")]
    CargoInstall,
    #[serde(rename = "cargo-binstall")]
    CargoBinstall,
}

impl UpdateRoute {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Standalone => "standalone",
            Self::Brew => "brew",
            Self::CargoInstall => "cargo-install",
            Self::CargoBinstall => "cargo-binstall",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecordedBundleIdentityV1 {
    release_manifest_sha256: Sha256Digest,
    archive_sha256: Sha256Digest,
    #[serde(skip_serializing_if = "Option::is_none")]
    installed_version_marker_sha256: Option<Sha256Digest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    installation_sequence: Option<u64>,
}

impl RecordedBundleIdentityV1 {
    pub fn release_manifest_sha256(&self) -> &Sha256Digest {
        &self.release_manifest_sha256
    }

    pub fn archive_sha256(&self) -> &Sha256Digest {
        &self.archive_sha256
    }

    pub fn installed_version_marker_sha256(&self) -> Option<&Sha256Digest> {
        self.installed_version_marker_sha256.as_ref()
    }

    pub fn installation_sequence(&self) -> Option<u64> {
        self.installation_sequence
    }

    pub fn is_standalone_installation(&self) -> bool {
        self.installed_version_marker_sha256.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InstalledReleaseV1 {
    version: ReleaseVersion,
    target: TargetTriple,
    #[serde(skip_serializing_if = "Option::is_none")]
    bundle: Option<RecordedBundleIdentityV1>,
}

impl InstalledReleaseV1 {
    pub fn version(&self) -> &ReleaseVersion {
        &self.version
    }

    pub fn target(&self) -> TargetTriple {
        self.target
    }

    pub fn bundle(&self) -> Option<&RecordedBundleIdentityV1> {
        self.bundle.as_ref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TransitionKind {
    #[serde(rename = "install")]
    Install,
    #[serde(rename = "update")]
    Update,
    #[serde(rename = "rollback")]
    Rollback,
    #[serde(rename = "confirmed-migration")]
    ConfirmedMigration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransitionEndpointV1 {
    owner_family: OwnerFamily,
    release: InstalledReleaseV1,
}

impl TransitionEndpointV1 {
    pub fn owner_family(&self) -> OwnerFamily {
        self.owner_family
    }

    pub fn release(&self) -> &InstalledReleaseV1 {
        &self.release
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum RecordedTransitionEvidenceV1 {
    #[serde(rename = "verified-update-metadata")]
    UpdateMetadataVersions {
        root_version: u64,
        timestamp_version: u64,
        snapshot_version: u64,
        targets_version: u64,
    },
    #[serde(rename = "package-manager")]
    PackageManagerRoute { route: UpdateRoute },
    #[serde(rename = "retained-release")]
    RetainedReleaseManifest {
        release_manifest_sha256: Sha256Digest,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SuccessfulTransitionV1 {
    sequence: u64,
    #[serde(rename = "type")]
    transition_type: TransitionKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    from: Option<TransitionEndpointV1>,
    to: TransitionEndpointV1,
    #[serde(rename = "authority")]
    recorded_evidence: RecordedTransitionEvidenceV1,
    completed_at_unix_seconds: u64,
}

impl SuccessfulTransitionV1 {
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn transition_type(&self) -> TransitionKind {
        self.transition_type
    }

    pub fn from(&self) -> Option<&TransitionEndpointV1> {
        self.from.as_ref()
    }

    pub fn to(&self) -> &TransitionEndpointV1 {
        &self.to
    }

    /// Returns unauthenticated audit evidence recorded by the transition.
    ///
    /// Callers must reverify this claim against live update, package-manager,
    /// or installed-release evidence before authorizing any mutation.
    pub fn recorded_evidence(&self) -> &RecordedTransitionEvidenceV1 {
        &self.recorded_evidence
    }

    pub fn completed_at_unix_seconds(&self) -> u64 {
        self.completed_at_unix_seconds
    }
}

/// Structurally validated local install state.
///
/// This is still an unauthenticated projection. Filesystem mutation must take
/// a future `VerifiedInstallOwnership` capability produced from live evidence,
/// never this parsed receipt or one of its path strings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InstallReceiptV1 {
    kind: String,
    schema_version: u32,
    package: String,
    state_layout_schema: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    installation_layout_schema: Option<u32>,
    installation_id: InstallationId,
    state_root: AbsoluteInstallPath,
    installation_root: AbsoluteInstallPath,
    owner_family: OwnerFamily,
    #[serde(skip_serializing_if = "Option::is_none")]
    update_route: Option<UpdateRoute>,
    active: InstalledReleaseV1,
    retained: Vec<InstalledReleaseV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_successful_transition: Option<SuccessfulTransitionV1>,
}

impl InstallReceiptV1 {
    pub fn parse_and_validate(bytes: &[u8]) -> Result<Self, InstallReceiptError> {
        validation::parse_install_receipt(bytes)
    }

    pub fn to_deterministic_json(&self) -> Result<Vec<u8>, InstallReceiptError> {
        deterministic_json("install receipt", self)
    }

    pub fn installation_id(&self) -> &InstallationId {
        &self.installation_id
    }

    pub fn state_root(&self) -> &AbsoluteInstallPath {
        &self.state_root
    }

    pub fn installation_root(&self) -> &AbsoluteInstallPath {
        &self.installation_root
    }

    pub fn owner_family(&self) -> OwnerFamily {
        self.owner_family
    }

    pub fn update_route(&self) -> Option<UpdateRoute> {
        self.update_route
    }

    pub fn active(&self) -> &InstalledReleaseV1 {
        &self.active
    }

    pub fn retained(&self) -> &[InstalledReleaseV1] {
        &self.retained
    }

    pub fn last_successful_transition(&self) -> Option<&SuccessfulTransitionV1> {
        self.last_successful_transition.as_ref()
    }
}

#[allow(dead_code)]
pub(super) fn build_first_install_receipt(
    marker: &InstalledVersionMarkerV2,
    marker_sha256: &Sha256Digest,
) -> Result<(InstallReceiptV1, Vec<u8>), InstallReceiptError> {
    let release = marker.release();
    let bundle = RecordedBundleIdentityV1 {
        release_manifest_sha256: release.release_manifest_sha256().clone(),
        archive_sha256: release.archive_sha256().clone(),
        installed_version_marker_sha256: Some(marker_sha256.clone()),
        installation_sequence: Some(1),
    };
    let active = InstalledReleaseV1 {
        version: release.version().clone(),
        target: release.target(),
        bundle: Some(bundle),
    };
    let (root_version, timestamp_version, snapshot_version, targets_version) =
        marker.prepared_from().role_versions();
    let receipt = InstallReceiptV1 {
        kind: INSTALL_RECEIPT_KIND.to_owned(),
        schema_version: INSTALL_RECEIPT_SCHEMA_VERSION,
        package: "hf2q".to_owned(),
        state_layout_schema: STATE_LAYOUT_SCHEMA_V1,
        installation_layout_schema: Some(INSTALLATION_LAYOUT_SCHEMA_V1),
        installation_id: marker.installation_id().clone(),
        state_root: marker.installation_root().clone(),
        installation_root: marker.installation_root().clone(),
        owner_family: OwnerFamily::Standalone,
        update_route: Some(UpdateRoute::Standalone),
        active: active.clone(),
        retained: Vec::new(),
        last_successful_transition: Some(SuccessfulTransitionV1 {
            sequence: 1,
            transition_type: TransitionKind::Install,
            from: None,
            to: TransitionEndpointV1 {
                owner_family: OwnerFamily::Standalone,
                release: active,
            },
            recorded_evidence: RecordedTransitionEvidenceV1::UpdateMetadataVersions {
                root_version,
                timestamp_version,
                snapshot_version,
                targets_version,
            },
            completed_at_unix_seconds: marker.installed_at_unix_seconds(),
        }),
    };
    let receipt_bytes = receipt.to_deterministic_json()?;
    let receipt = InstallReceiptV1::parse_and_validate(&receipt_bytes)?;
    Ok((receipt, receipt_bytes))
}

pub(super) fn validate_envelope(
    actual_kind: &str,
    actual_schema: u32,
    package: &str,
    expected_kind: &str,
    expected_schema: u32,
    document: &'static str,
) -> Result<(), InstallReceiptError> {
    if actual_kind != expected_kind {
        return Err(InstallReceiptError::UnsupportedKind(document));
    }
    if actual_schema != expected_schema {
        return Err(InstallReceiptError::UnsupportedSchema {
            document,
            actual: actual_schema,
        });
    }
    if package != "hf2q" {
        return Err(InstallReceiptError::invalid(
            "package",
            "must be exactly `hf2q`",
        ));
    }
    Ok(())
}

pub(super) fn deterministic_json<T: Serialize>(
    document: &'static str,
    value: &T,
) -> Result<Vec<u8>, InstallReceiptError> {
    let mut bytes =
        serde_json::to_vec(value).map_err(|error| sanitize_json_error(document, error))?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub(super) fn sanitize_json_error(
    document: &'static str,
    error: serde_json::Error,
) -> InstallReceiptError {
    let category = match error.classify() {
        serde_json::error::Category::Io => "I/O",
        serde_json::error::Category::Syntax => "syntax",
        serde_json::error::Category::Data => "data",
        serde_json::error::Category::Eof => "unexpected EOF",
    };
    InstallReceiptError::Json {
        document,
        line: error.line(),
        column: error.column(),
        category,
    }
}

#[path = "install_receipt_validation.rs"]
mod validation;

#[cfg(test)]
#[path = "install_receipt_tests.rs"]
mod tests;
