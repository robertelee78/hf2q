use serde::{Deserialize, Serialize};

use super::{AbsoluteInstallPath, InstallReceiptError, InstallationId, STATE_LAYOUT_SCHEMA_V1};

pub(in crate::distribution) const INSTALLATION_IDENTITY_KIND: &str = "hf2q.installation-identity";
pub(in crate::distribution) const INSTALLATION_IDENTITY_SCHEMA_VERSION: u32 = 1;
pub(in crate::distribution) const MAX_INSTALLATION_IDENTITY_BYTES: usize = 16 * 1024;

#[derive(Debug, thiserror::Error)]
pub(crate) enum InstallationIdentityError {
    #[error("installation identity exceeds the {limit}-byte input limit ({actual} bytes)")]
    InputTooLarge { limit: usize, actual: usize },
    #[error("installation identity JSON is invalid at line {line}, column {column} ({category})")]
    Json {
        line: usize,
        column: usize,
        category: &'static str,
    },
    #[error("unsupported installation identity kind discriminator")]
    UnsupportedKind,
    #[error("unsupported installation identity schema version {0}")]
    UnsupportedSchema(u32),
    #[error("invalid installation identity field `{field}`: {reason}")]
    InvalidField { field: &'static str, reason: String },
    #[error("installation identity is not in hf2q's canonical byte encoding")]
    NonCanonicalEncoding,
}

/// Immutable, root-bound installation identity wire.
///
/// Parsing proves only structural validity. Filesystem authority is created
/// separately after the exact bytes and named inode are revalidated beneath
/// an explicitly authorized state root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(in crate::distribution) struct InstallationIdentityV1 {
    kind: String,
    schema_version: u32,
    state_layout_schema: u32,
    package: String,
    installation_id: InstallationId,
    state_root: AbsoluteInstallPath,
}

impl InstallationIdentityV1 {
    pub(in crate::distribution) fn new(
        installation_id: InstallationId,
        state_root: AbsoluteInstallPath,
    ) -> Self {
        Self {
            kind: INSTALLATION_IDENTITY_KIND.to_owned(),
            schema_version: INSTALLATION_IDENTITY_SCHEMA_VERSION,
            state_layout_schema: STATE_LAYOUT_SCHEMA_V1,
            package: "hf2q".to_owned(),
            installation_id,
            state_root,
        }
    }

    pub(in crate::distribution) fn parse_and_validate(
        bytes: &[u8],
    ) -> Result<Self, InstallationIdentityError> {
        if bytes.len() > MAX_INSTALLATION_IDENTITY_BYTES {
            return Err(InstallationIdentityError::InputTooLarge {
                limit: MAX_INSTALLATION_IDENTITY_BYTES,
                actual: bytes.len(),
            });
        }
        let raw: RawInstallationIdentityV1 =
            serde_json::from_slice(bytes).map_err(sanitize_json_error)?;
        let identity = Self::try_from(raw)?;
        if identity.to_deterministic_json()? != bytes {
            return Err(InstallationIdentityError::NonCanonicalEncoding);
        }
        Ok(identity)
    }

    pub(in crate::distribution) fn to_deterministic_json(
        &self,
    ) -> Result<Vec<u8>, InstallationIdentityError> {
        let mut bytes = serde_json::to_vec(self).map_err(sanitize_json_error)?;
        bytes.push(b'\n');
        if bytes.len() > MAX_INSTALLATION_IDENTITY_BYTES {
            return Err(InstallationIdentityError::InputTooLarge {
                limit: MAX_INSTALLATION_IDENTITY_BYTES,
                actual: bytes.len(),
            });
        }
        Ok(bytes)
    }

    pub(in crate::distribution) fn installation_id(&self) -> &InstallationId {
        &self.installation_id
    }

    pub(in crate::distribution) fn state_root(&self) -> &AbsoluteInstallPath {
        &self.state_root
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawInstallationIdentityV1 {
    kind: String,
    schema_version: u32,
    state_layout_schema: u32,
    package: String,
    installation_id: String,
    state_root: String,
}

impl TryFrom<RawInstallationIdentityV1> for InstallationIdentityV1 {
    type Error = InstallationIdentityError;

    fn try_from(raw: RawInstallationIdentityV1) -> Result<Self, Self::Error> {
        if raw.kind != INSTALLATION_IDENTITY_KIND {
            return Err(InstallationIdentityError::UnsupportedKind);
        }
        if raw.schema_version != INSTALLATION_IDENTITY_SCHEMA_VERSION {
            return Err(InstallationIdentityError::UnsupportedSchema(
                raw.schema_version,
            ));
        }
        if raw.state_layout_schema != STATE_LAYOUT_SCHEMA_V1 {
            return Err(invalid(
                "state_layout_schema",
                "must equal the supported standalone state-layout schema",
            ));
        }
        if raw.package != "hf2q" {
            return Err(invalid("package", "must be hf2q"));
        }
        Ok(Self {
            kind: raw.kind,
            schema_version: raw.schema_version,
            state_layout_schema: raw.state_layout_schema,
            package: raw.package,
            installation_id: InstallationId::parse(raw.installation_id).map_err(map_value_error)?,
            state_root: AbsoluteInstallPath::parse("state_root", raw.state_root)
                .map_err(map_value_error)?,
        })
    }
}

fn invalid(field: &'static str, reason: impl Into<String>) -> InstallationIdentityError {
    InstallationIdentityError::InvalidField {
        field,
        reason: reason.into(),
    }
}

fn map_value_error(error: InstallReceiptError) -> InstallationIdentityError {
    match error {
        InstallReceiptError::InvalidField { field, reason } => {
            InstallationIdentityError::InvalidField { field, reason }
        }
        other => invalid("identity", other.to_string()),
    }
}

fn sanitize_json_error(error: serde_json::Error) -> InstallationIdentityError {
    let category = match error.classify() {
        serde_json::error::Category::Io => "io",
        serde_json::error::Category::Syntax => "syntax",
        serde_json::error::Category::Data => "data",
        serde_json::error::Category::Eof => "eof",
    };
    InstallationIdentityError::Json {
        line: error.line(),
        column: error.column(),
        category,
    }
}

#[cfg(test)]
#[path = "installation_identity_tests.rs"]
mod tests;
