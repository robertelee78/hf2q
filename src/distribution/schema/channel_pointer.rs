use serde::{Deserialize, Serialize};

use super::common::{ReleaseVersion, SchemaValueError, Sha256Digest, TargetTriple, UpdateChannel};
use super::release_manifest::MAX_RELEASE_MANIFEST_BYTES;
use super::target_name::LogicalTargetName;

pub const CHANNEL_POINTER_KIND: &str = "hf2q.update-channel-pointer";
pub const CHANNEL_POINTER_SCHEMA_VERSION: u32 = 1;
pub const MAX_CHANNEL_POINTER_BYTES: usize = 16 * 1024;
pub const MAX_RELEASE_ARCHIVE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum ChannelPointerError {
    #[error("channel pointer exceeds the {limit}-byte input limit ({actual} bytes)")]
    InputTooLarge { limit: usize, actual: usize },
    #[error("channel pointer JSON is invalid at line {line}, column {column} ({category})")]
    Json {
        line: usize,
        column: usize,
        category: &'static str,
    },
    #[error("unsupported channel pointer kind discriminator")]
    UnsupportedKind,
    #[error("unsupported channel pointer schema version {0}")]
    UnsupportedSchema(u32),
    #[error("invalid channel pointer field `{field}`: {reason}")]
    InvalidField { field: &'static str, reason: String },
}

impl From<SchemaValueError> for ChannelPointerError {
    fn from(error: SchemaValueError) -> Self {
        Self::InvalidField {
            field: error.field,
            reason: error.reason,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChannelPointerV1 {
    kind: String,
    schema_version: u32,
    package: String,
    repository_id: String,
    channel: UpdateChannel,
    version: ReleaseVersion,
    target: TargetTriple,
    manifest: ReleaseTargetDescriptorV1,
    archive: ReleaseTargetDescriptorV1,
}

impl ChannelPointerV1 {
    pub fn parse_and_validate(bytes: &[u8]) -> Result<Self, ChannelPointerError> {
        if bytes.len() > MAX_CHANNEL_POINTER_BYTES {
            return Err(ChannelPointerError::InputTooLarge {
                limit: MAX_CHANNEL_POINTER_BYTES,
                actual: bytes.len(),
            });
        }
        let raw: RawChannelPointerV1 =
            serde_json::from_slice(bytes).map_err(sanitize_json_error)?;
        Self::try_from(raw)
    }

    #[cfg(test)]
    pub(crate) fn new(
        version: ReleaseVersion,
        target: TargetTriple,
        manifest_length: u64,
        manifest_sha256: Sha256Digest,
        archive_length: u64,
        archive_sha256: Sha256Digest,
    ) -> Result<Self, ChannelPointerError> {
        validate_length(
            "manifest.length",
            manifest_length,
            MAX_RELEASE_MANIFEST_BYTES as u64,
        )?;
        validate_length("archive.length", archive_length, MAX_RELEASE_ARCHIVE_BYTES)?;
        Ok(Self {
            kind: CHANNEL_POINTER_KIND.to_owned(),
            schema_version: CHANNEL_POINTER_SCHEMA_VERSION,
            package: "hf2q".to_owned(),
            repository_id: "hf2q".to_owned(),
            channel: UpdateChannel::Stable,
            manifest: ReleaseTargetDescriptorV1 {
                name: LogicalTargetName::release_manifest(&version, target),
                length: manifest_length,
                sha256: manifest_sha256,
            },
            archive: ReleaseTargetDescriptorV1 {
                name: LogicalTargetName::release_archive(&version, target),
                length: archive_length,
                sha256: archive_sha256,
            },
            version,
            target,
        })
    }

    pub fn to_deterministic_json(&self) -> Result<Vec<u8>, ChannelPointerError> {
        let mut bytes = serde_json::to_vec(self).map_err(sanitize_json_error)?;
        bytes.push(b'\n');
        if bytes.len() > MAX_CHANNEL_POINTER_BYTES {
            return Err(ChannelPointerError::InputTooLarge {
                limit: MAX_CHANNEL_POINTER_BYTES,
                actual: bytes.len(),
            });
        }
        Ok(bytes)
    }

    pub fn channel(&self) -> UpdateChannel {
        self.channel
    }

    pub fn version(&self) -> &ReleaseVersion {
        &self.version
    }

    pub fn target(&self) -> TargetTriple {
        self.target
    }

    pub fn manifest(&self) -> &ReleaseTargetDescriptorV1 {
        &self.manifest
    }

    pub fn archive(&self) -> &ReleaseTargetDescriptorV1 {
        &self.archive
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReleaseTargetDescriptorV1 {
    name: LogicalTargetName,
    length: u64,
    sha256: Sha256Digest,
}

impl ReleaseTargetDescriptorV1 {
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    pub fn length(&self) -> u64 {
        self.length
    }

    pub fn sha256(&self) -> &Sha256Digest {
        &self.sha256
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawChannelPointerV1 {
    kind: String,
    schema_version: u32,
    package: String,
    repository_id: String,
    channel: String,
    version: String,
    target: String,
    manifest: RawReleaseTargetDescriptorV1,
    archive: RawReleaseTargetDescriptorV1,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawReleaseTargetDescriptorV1 {
    name: String,
    length: u64,
    sha256: String,
}

impl TryFrom<RawChannelPointerV1> for ChannelPointerV1 {
    type Error = ChannelPointerError;

    fn try_from(raw: RawChannelPointerV1) -> Result<Self, Self::Error> {
        if raw.kind != CHANNEL_POINTER_KIND {
            return Err(ChannelPointerError::UnsupportedKind);
        }
        if raw.schema_version != CHANNEL_POINTER_SCHEMA_VERSION {
            return Err(ChannelPointerError::UnsupportedSchema(raw.schema_version));
        }
        if raw.package != "hf2q" {
            return Err(invalid("package", "must be hf2q"));
        }
        if raw.repository_id != "hf2q" {
            return Err(invalid("repository_id", "must be hf2q"));
        }
        let channel = UpdateChannel::parse("channel", raw.channel)?;
        let version = ReleaseVersion::parse_stable("version", raw.version)?;
        let target = TargetTriple::parse("target", raw.target)?;
        let expected_manifest = LogicalTargetName::release_manifest(&version, target);
        let expected_archive = LogicalTargetName::release_archive(&version, target);
        let manifest = descriptor(
            "manifest",
            raw.manifest,
            expected_manifest,
            MAX_RELEASE_MANIFEST_BYTES as u64,
        )?;
        let archive = descriptor(
            "archive",
            raw.archive,
            expected_archive,
            MAX_RELEASE_ARCHIVE_BYTES,
        )?;
        Ok(Self {
            kind: raw.kind,
            schema_version: raw.schema_version,
            package: raw.package,
            repository_id: raw.repository_id,
            channel,
            version,
            target,
            manifest,
            archive,
        })
    }
}

fn descriptor(
    field: &'static str,
    raw: RawReleaseTargetDescriptorV1,
    expected_name: LogicalTargetName,
    maximum: u64,
) -> Result<ReleaseTargetDescriptorV1, ChannelPointerError> {
    let name_field = if field == "manifest" {
        "manifest.name"
    } else {
        "archive.name"
    };
    let name = LogicalTargetName::parse(name_field, raw.name)?;
    if name != expected_name {
        return Err(invalid(
            name_field,
            "must equal the canonical name derived from version and target",
        ));
    }
    let length_field = if field == "manifest" {
        "manifest.length"
    } else {
        "archive.length"
    };
    validate_length(length_field, raw.length, maximum)?;
    let digest_field = if field == "manifest" {
        "manifest.sha256"
    } else {
        "archive.sha256"
    };
    Ok(ReleaseTargetDescriptorV1 {
        name,
        length: raw.length,
        sha256: Sha256Digest::parse(digest_field, raw.sha256)?,
    })
}

fn validate_length(
    field: &'static str,
    value: u64,
    maximum: u64,
) -> Result<(), ChannelPointerError> {
    if value == 0 || value > maximum {
        return Err(invalid(
            field,
            format!("must be between 1 and {maximum} bytes"),
        ));
    }
    Ok(())
}

fn invalid(field: &'static str, reason: impl Into<String>) -> ChannelPointerError {
    ChannelPointerError::InvalidField {
        field,
        reason: reason.into(),
    }
}

fn sanitize_json_error(error: serde_json::Error) -> ChannelPointerError {
    let category = match error.classify() {
        serde_json::error::Category::Io => "I/O",
        serde_json::error::Category::Syntax => "syntax",
        serde_json::error::Category::Data => "data",
        serde_json::error::Category::Eof => "unexpected EOF",
    };
    ChannelPointerError::Json {
        line: error.line(),
        column: error.column(),
        category,
    }
}

#[cfg(test)]
#[path = "channel_pointer_tests.rs"]
mod tests;
