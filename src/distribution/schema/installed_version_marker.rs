use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::common::{ReleaseVersion, Sha256Digest, TargetTriple};
use super::install_receipt::{
    deterministic_json, sanitize_json_error, validate_envelope, AbsoluteInstallPath,
    InstallReceiptError, InstallationId, INSTALLATION_LAYOUT_SCHEMA_V1,
};

pub const INSTALLED_VERSION_MARKER_KIND: &str = "hf2q.installed-version";
pub const INSTALLED_VERSION_MARKER_SCHEMA_VERSION: u32 = 2;
pub const MAX_INSTALLED_VERSION_MARKER_BYTES: usize = 16 * 1024;

/// Immutable marker written inside a fully verified standalone version.
///
/// Marker v2 records the metadata-role versions that prepared the release so
/// an exact activation receipt remains reconstructible after a crash or a
/// later metadata refresh. These fields are audit evidence, not authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InstalledVersionMarkerV2 {
    kind: String,
    schema_version: u32,
    package: String,
    installation_layout_schema: u32,
    installation_id: InstallationId,
    installation_root: AbsoluteInstallPath,
    release: MarkerReleaseV1,
    prepared_from: RecordedPreparationEvidenceV2,
    installation_sequence: u64,
    installed_at_unix_seconds: u64,
}

impl InstalledVersionMarkerV2 {
    /// Structurally validates an untrusted marker without authenticating it.
    pub fn parse_and_validate(bytes: &[u8]) -> Result<Self, InstallReceiptError> {
        parse_installed_version_marker(bytes)
    }

    /// Binds and validates the exact marker bytes recorded by an activation.
    ///
    /// Ownership verification must use this entry point. Hashing a parsed or
    /// reserialized JSON value is never equivalent to hashing `bytes`.
    pub fn parse_and_validate_exact(
        bytes: &[u8],
        expected_sha256: &Sha256Digest,
    ) -> Result<Self, InstallReceiptError> {
        if bytes.len() > MAX_INSTALLED_VERSION_MARKER_BYTES {
            return Err(InstallReceiptError::InputTooLarge {
                document: "installed-version marker",
                limit: MAX_INSTALLED_VERSION_MARKER_BYTES,
                actual: bytes.len(),
            });
        }
        let actual = hex::encode(Sha256::digest(bytes));
        if actual != expected_sha256.as_str() {
            return Err(InstallReceiptError::MarkerDigestMismatch);
        }
        let marker = Self::parse_and_validate(bytes)?;
        if marker.to_deterministic_json()? != bytes {
            return Err(InstallReceiptError::NonCanonicalMarkerEncoding);
        }
        Ok(marker)
    }

    pub fn to_deterministic_json(&self) -> Result<Vec<u8>, InstallReceiptError> {
        deterministic_json("installed-version marker", self)
    }

    pub fn installation_id(&self) -> &InstallationId {
        &self.installation_id
    }

    pub fn installation_root(&self) -> &AbsoluteInstallPath {
        &self.installation_root
    }

    pub fn release(&self) -> &MarkerReleaseV1 {
        &self.release
    }

    /// Returns unauthenticated preparation evidence recorded by this marker.
    pub fn prepared_from(&self) -> &RecordedPreparationEvidenceV2 {
        &self.prepared_from
    }

    pub fn installation_sequence(&self) -> u64 {
        self.installation_sequence
    }

    pub fn installed_at_unix_seconds(&self) -> u64 {
        self.installed_at_unix_seconds
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(super) fn first_install(
        installation_id: InstallationId,
        installation_root: AbsoluteInstallPath,
        version: ReleaseVersion,
        target: TargetTriple,
        release_manifest_sha256: Sha256Digest,
        archive_sha256: Sha256Digest,
        prepared_from: RecordedPreparationEvidenceV2,
        installed_at_unix_seconds: u64,
    ) -> Self {
        Self {
            kind: INSTALLED_VERSION_MARKER_KIND.to_owned(),
            schema_version: INSTALLED_VERSION_MARKER_SCHEMA_VERSION,
            package: "hf2q".to_owned(),
            installation_layout_schema: INSTALLATION_LAYOUT_SCHEMA_V1,
            installation_id,
            installation_root,
            release: MarkerReleaseV1 {
                version,
                target,
                release_manifest_sha256,
                archive_sha256,
            },
            prepared_from,
            installation_sequence: 1,
            installed_at_unix_seconds,
        }
    }
}

/// Metadata-role versions recorded when a standalone version was prepared.
///
/// Parsing or constructing this value does not authenticate update metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum RecordedPreparationEvidenceV2 {
    #[serde(rename = "verified-update-metadata")]
    UpdateMetadataVersions {
        root_version: u64,
        timestamp_version: u64,
        snapshot_version: u64,
        targets_version: u64,
    },
}

impl RecordedPreparationEvidenceV2 {
    pub(in crate::distribution) fn verified_update_metadata(
        root_version: u64,
        timestamp_version: u64,
        snapshot_version: u64,
        targets_version: u64,
    ) -> Result<Self, InstallReceiptError> {
        if [
            root_version,
            timestamp_version,
            snapshot_version,
            targets_version,
        ]
        .contains(&0)
        {
            return Err(InstallReceiptError::invalid(
                "prepared_from",
                "verified metadata role versions must be nonzero",
            ));
        }
        Ok(Self::UpdateMetadataVersions {
            root_version,
            timestamp_version,
            snapshot_version,
            targets_version,
        })
    }

    pub fn role_versions(&self) -> (u64, u64, u64, u64) {
        match self {
            Self::UpdateMetadataVersions {
                root_version,
                timestamp_version,
                snapshot_version,
                targets_version,
            } => (
                *root_version,
                *timestamp_version,
                *snapshot_version,
                *targets_version,
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MarkerReleaseV1 {
    version: ReleaseVersion,
    target: TargetTriple,
    release_manifest_sha256: Sha256Digest,
    archive_sha256: Sha256Digest,
}

impl MarkerReleaseV1 {
    pub fn version(&self) -> &ReleaseVersion {
        &self.version
    }

    pub fn target(&self) -> TargetTriple {
        self.target
    }

    pub fn release_manifest_sha256(&self) -> &Sha256Digest {
        &self.release_manifest_sha256
    }

    pub fn archive_sha256(&self) -> &Sha256Digest {
        &self.archive_sha256
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawInstalledVersionMarkerV2 {
    kind: String,
    schema_version: u32,
    package: String,
    installation_layout_schema: u32,
    installation_id: String,
    installation_root: String,
    release: RawMarkerReleaseV1,
    prepared_from: RawRecordedPreparationEvidenceV2,
    installation_sequence: u64,
    installed_at_unix_seconds: u64,
}

#[derive(Debug, Deserialize)]
struct RawMarkerEnvelope {
    kind: String,
    schema_version: u32,
    package: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
enum RawRecordedPreparationEvidenceV2 {
    #[serde(rename = "verified-update-metadata")]
    VerifiedUpdateMetadata {
        root_version: u64,
        timestamp_version: u64,
        snapshot_version: u64,
        targets_version: u64,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMarkerReleaseV1 {
    version: String,
    target: String,
    release_manifest_sha256: String,
    archive_sha256: String,
}

fn parse_installed_version_marker(
    bytes: &[u8],
) -> Result<InstalledVersionMarkerV2, InstallReceiptError> {
    if bytes.len() > MAX_INSTALLED_VERSION_MARKER_BYTES {
        return Err(InstallReceiptError::InputTooLarge {
            document: "installed-version marker",
            limit: MAX_INSTALLED_VERSION_MARKER_BYTES,
            actual: bytes.len(),
        });
    }
    let envelope: RawMarkerEnvelope = serde_json::from_slice(bytes)
        .map_err(|error| sanitize_json_error("installed-version marker", error))?;
    validate_envelope(
        &envelope.kind,
        envelope.schema_version,
        &envelope.package,
        INSTALLED_VERSION_MARKER_KIND,
        INSTALLED_VERSION_MARKER_SCHEMA_VERSION,
        "installed-version marker",
    )?;
    let raw: RawInstalledVersionMarkerV2 = serde_json::from_slice(bytes)
        .map_err(|error| sanitize_json_error("installed-version marker", error))?;
    validate_marker(raw)
}

fn validate_marker(
    raw: RawInstalledVersionMarkerV2,
) -> Result<InstalledVersionMarkerV2, InstallReceiptError> {
    validate_envelope(
        &raw.kind,
        raw.schema_version,
        &raw.package,
        INSTALLED_VERSION_MARKER_KIND,
        INSTALLED_VERSION_MARKER_SCHEMA_VERSION,
        "installed-version marker",
    )?;
    if raw.installation_layout_schema != INSTALLATION_LAYOUT_SCHEMA_V1 {
        return Err(InstallReceiptError::invalid(
            "installation_layout_schema",
            "must equal the supported standalone v1 installation layout schema",
        ));
    }
    if raw.installation_sequence == 0 {
        return Err(InstallReceiptError::invalid(
            "installation_sequence",
            "must be nonzero",
        ));
    }
    if raw.installed_at_unix_seconds == 0 {
        return Err(InstallReceiptError::invalid(
            "installed_at_unix_seconds",
            "must be nonzero",
        ));
    }
    let prepared_from = match raw.prepared_from {
        RawRecordedPreparationEvidenceV2::VerifiedUpdateMetadata {
            root_version,
            timestamp_version,
            snapshot_version,
            targets_version,
        } => RecordedPreparationEvidenceV2::verified_update_metadata(
            root_version,
            timestamp_version,
            snapshot_version,
            targets_version,
        )?,
    };
    Ok(InstalledVersionMarkerV2 {
        kind: raw.kind,
        schema_version: raw.schema_version,
        package: raw.package,
        installation_layout_schema: raw.installation_layout_schema,
        installation_id: InstallationId::parse(raw.installation_id)?,
        installation_root: AbsoluteInstallPath::parse("installation_root", raw.installation_root)?,
        release: MarkerReleaseV1 {
            version: ReleaseVersion::parse_stable("release.version", raw.release.version)?,
            target: TargetTriple::parse("release.target", raw.release.target)?,
            release_manifest_sha256: Sha256Digest::parse(
                "release.release_manifest_sha256",
                raw.release.release_manifest_sha256,
            )?,
            archive_sha256: Sha256Digest::parse(
                "release.archive_sha256",
                raw.release.archive_sha256,
            )?,
        },
        prepared_from,
        installation_sequence: raw.installation_sequence,
        installed_at_unix_seconds: raw.installed_at_unix_seconds,
    })
}
