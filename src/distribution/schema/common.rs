use std::fmt;

use serde::Serialize;

const MAX_RELEASE_VERSION_BYTES: usize = 64;
const MAX_BUNDLE_PATH_BYTES: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SchemaValueError {
    pub(crate) field: &'static str,
    pub(crate) reason: String,
}

impl SchemaValueError {
    pub(crate) fn new(field: &'static str, reason: impl Into<String>) -> Self {
        Self {
            field,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct Sha256Digest(String);

impl Sha256Digest {
    pub(crate) fn parse(field: &'static str, value: String) -> Result<Self, SchemaValueError> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(SchemaValueError::new(
                field,
                "must be exactly 64 lowercase hexadecimal characters",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct GitCommit(String);

impl GitCommit {
    pub(crate) fn parse(field: &'static str, value: String) -> Result<Self, SchemaValueError> {
        if value.len() != 40
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(SchemaValueError::new(
                field,
                "must be exactly 40 lowercase hexadecimal characters",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ReleaseVersion(String);

impl ReleaseVersion {
    pub(crate) fn parse_stable(
        field: &'static str,
        value: String,
    ) -> Result<Self, SchemaValueError> {
        if value.is_empty() || value.len() > MAX_RELEASE_VERSION_BYTES {
            return Err(SchemaValueError::new(
                field,
                "must be a bounded semantic version",
            ));
        }
        let parsed = semver::Version::parse(&value)
            .map_err(|_| SchemaValueError::new(field, "must be a canonical semantic version"))?;
        if !parsed.pre.is_empty() || !parsed.build.is_empty() {
            return Err(SchemaValueError::new(
                field,
                "stable releases cannot contain prerelease or build metadata",
            ));
        }
        if parsed.to_string() != value {
            return Err(SchemaValueError::new(
                field,
                "semantic version is not in canonical form",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn parsed(&self) -> semver::Version {
        semver::Version::parse(&self.0).expect("validated release version")
    }
}

impl PartialOrd for ReleaseVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ReleaseVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.parsed().cmp(&other.parsed())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TargetTriple {
    #[serde(rename = "aarch64-apple-darwin")]
    Aarch64AppleDarwin,
}

impl TargetTriple {
    pub(crate) fn parse(field: &'static str, value: String) -> Result<Self, SchemaValueError> {
        match value.as_str() {
            "aarch64-apple-darwin" => Ok(Self::Aarch64AppleDarwin),
            _ => Err(SchemaValueError::new(
                field,
                "v1 supports only aarch64-apple-darwin",
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Aarch64AppleDarwin => "aarch64-apple-darwin",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum UpdateChannel {
    #[serde(rename = "stable")]
    Stable,
}

impl UpdateChannel {
    pub(crate) fn parse(field: &'static str, value: String) -> Result<Self, SchemaValueError> {
        match value.as_str() {
            "stable" => Ok(Self::Stable),
            _ => Err(SchemaValueError::new(
                field,
                "v1 supports only the stable channel",
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stable => "stable",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct MacOsVersion(String);

impl MacOsVersion {
    pub(crate) fn parse(field: &'static str, value: String) -> Result<Self, SchemaValueError> {
        if value.is_empty() || value.len() > 16 {
            return Err(SchemaValueError::new(
                field,
                "must be a bounded major.minor or major.minor.patch version",
            ));
        }
        let parts: Vec<_> = value.split('.').collect();
        if !(2..=3).contains(&parts.len())
            || parts.iter().any(|part| {
                part.is_empty()
                    || !part.bytes().all(|byte| byte.is_ascii_digit())
                    || (part.len() > 1 && part.starts_with('0'))
                    || part.parse::<u16>().is_err()
            })
            || (parts.len() == 3 && parts[2] == "0")
            || parts[0] == "0"
        {
            return Err(SchemaValueError::new(
                field,
                "must be canonical major.minor or nonzero-patch major.minor.patch decimal components",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn numeric_components(&self) -> impl Iterator<Item = u16> + '_ {
        self.0
            .split('.')
            .map(|part| part.parse::<u16>().expect("validated macOS version"))
            .chain(std::iter::repeat(0))
            .take(3)
    }
}

impl PartialOrd for MacOsVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MacOsVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.numeric_components().cmp(other.numeric_components())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct BundlePath(String);

impl BundlePath {
    pub(crate) fn parse(field: &'static str, value: String) -> Result<Self, SchemaValueError> {
        if value.is_empty()
            || value.len() > MAX_BUNDLE_PATH_BYTES
            || !value.is_ascii()
            || value.starts_with('/')
            || value.ends_with('/')
            || value.contains("//")
            || value.contains('\\')
            || value.contains(':')
            || value.bytes().any(|byte| byte.is_ascii_control())
            || !value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'.' | b'_' | b'-')
            })
            || value.split('/').any(|component| {
                component.is_empty()
                    || component == "."
                    || component == ".."
                    || component.len() > 255
            })
        {
            return Err(SchemaValueError::new(
                field,
                "must be a canonical, safe ASCII bundle-relative path",
            ));
        }

        if value == "release-manifest.json" {
            return Err(SchemaValueError::new(
                field,
                "release-manifest.json is the reserved envelope and is not self-inventoried",
            ));
        }

        let allowed = value == "bin/hf2q"
            || matches!(
                value.as_str(),
                "libexec/serve_qwen38_opencode.sh"
                    | "libexec/serve_qwen36_opencode.sh"
                    | "libexec/serve_gemma4_opencode.sh"
                    | "libexec/serve_deepseek4_opencode.sh"
            )
            || value
                .strip_prefix("share/doc/hf2q/")
                .is_some_and(|suffix| !suffix.is_empty())
            || value
                .strip_prefix("share/licenses/hf2q/")
                .is_some_and(|suffix| !suffix.is_empty());
        if !allowed {
            return Err(SchemaValueError::new(
                field,
                "path is outside the v1 release-bundle whitelist",
            ));
        }

        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, Serialize)]
pub enum BundleEntryType {
    #[serde(rename = "regular")]
    Regular,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, Serialize)]
pub enum FileMode {
    #[serde(rename = "0644")]
    Data,
    #[serde(rename = "0755")]
    Executable,
}

impl FileMode {
    pub fn as_octal(self) -> u32 {
        match self {
            Self::Data => 0o644,
            Self::Executable => 0o755,
        }
    }
}
