use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::common::{
    BundleEntryType, BundlePath, FileMode, GitCommit, MacOsVersion, ReleaseVersion,
    SchemaValueError, Sha256Digest, TargetTriple, UpdateChannel,
};

pub const RELEASE_MANIFEST_KIND: &str = "hf2q.release-manifest";
pub const RELEASE_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const MAX_RELEASE_MANIFEST_BYTES: usize = 1024 * 1024;
pub const MAX_BUNDLE_FILES: usize = 4096;
pub const MAX_BUNDLE_DIRECTORIES: usize = 4096;
pub const MAX_DYNAMIC_DEPENDENCIES: usize = 256;
pub const MAX_BUNDLE_PAYLOAD_BYTES: u64 = 4 * 1024 * 1024 * 1024;

const SUPPORTED_INSTALLER_PROTOCOL: u32 = 1;
const SUPPORTED_UPDATER_PROTOCOL: u32 = 1;
const SUPPORTED_LAUNCHER_REGISTRY_SCHEMA: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum ReleaseManifestError {
    #[error("release manifest exceeds the {limit}-byte input limit ({actual} bytes)")]
    InputTooLarge { limit: usize, actual: usize },
    #[error("release manifest JSON is invalid at line {line}, column {column} ({category})")]
    Json {
        line: usize,
        column: usize,
        category: &'static str,
    },
    #[error("unsupported release manifest kind discriminator")]
    UnsupportedKind,
    #[error("unsupported release manifest schema version {0}")]
    UnsupportedSchema(u32),
    #[error("invalid release manifest field `{field}`: {reason}")]
    InvalidField { field: &'static str, reason: String },
    #[error("release manifest contains too many {collection}: {actual} exceeds {limit}")]
    TooManyEntries {
        collection: &'static str,
        limit: usize,
        actual: usize,
    },
    #[error("release manifest file inventory is not strictly sorted at `{0}`")]
    UnsortedInventory(String),
    #[error("release manifest contains duplicate path `{0}`")]
    DuplicatePath(String),
    #[error("release manifest paths `{first}` and `{second}` collide on the target filesystem")]
    PathCollision { first: String, second: String },
    #[error("release manifest is missing required {0}")]
    MissingRequired(&'static str),
    #[error("release manifest payload size overflow")]
    PayloadSizeOverflow,
    #[error("release manifest payload is {actual} bytes, above the {limit}-byte limit")]
    PayloadTooLarge { limit: u64, actual: u64 },
    #[error(
        "release requires {capability} protocol/schema {required}, but this verifier supports {supported}"
    )]
    UnsupportedCompatibility {
        capability: &'static str,
        required: u32,
        supported: u32,
    },
}

impl ReleaseManifestError {
    pub(crate) fn invalid(field: &'static str, reason: impl Into<String>) -> ReleaseManifestError {
        Self::InvalidField {
            field,
            reason: reason.into(),
        }
    }
}

impl From<SchemaValueError> for ReleaseManifestError {
    fn from(error: SchemaValueError) -> Self {
        Self::InvalidField {
            field: error.field,
            reason: error.reason,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReleaseManifestV1 {
    kind: String,
    schema_version: u32,
    package: String,
    version: ReleaseVersion,
    target: TargetTriple,
    minimum_macos: MacOsVersion,
    source_commit: GitCommit,
    channel: UpdateChannel,
    code_signing: CodeSigningIdentityV1,
    compatibility: CompatibilityV1,
    files: Vec<BundleFileV1>,
    #[serde(skip)]
    derived_directories: Vec<String>,
    non_system_dynamic_dependencies: Vec<DynamicDependencyV1>,
}

impl ReleaseManifestV1 {
    /// Parses untrusted JSON and enforces every v1 structural invariant.
    ///
    /// This does not authenticate the input. A future trust adapter must first
    /// verify the exact external manifest bytes against signed target metadata.
    pub fn parse_and_validate(bytes: &[u8]) -> Result<Self, ReleaseManifestError> {
        if bytes.len() > MAX_RELEASE_MANIFEST_BYTES {
            return Err(ReleaseManifestError::InputTooLarge {
                limit: MAX_RELEASE_MANIFEST_BYTES,
                actual: bytes.len(),
            });
        }
        let raw: RawReleaseManifestV1 =
            serde_json::from_slice(bytes).map_err(sanitize_json_error)?;
        Self::try_from(raw)
    }

    /// Emits the single deterministic encoding produced by hf2q.
    ///
    /// The byte digest authenticates a manifest; JSON reserialization is never
    /// treated as an equivalent signed representation.
    pub fn to_deterministic_json(&self) -> Result<Vec<u8>, ReleaseManifestError> {
        let mut bytes = serde_json::to_vec(self).map_err(sanitize_json_error)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    pub fn version(&self) -> &ReleaseVersion {
        &self.version
    }

    pub fn target(&self) -> TargetTriple {
        self.target
    }

    pub fn minimum_macos(&self) -> &MacOsVersion {
        &self.minimum_macos
    }

    pub fn source_commit(&self) -> &GitCommit {
        &self.source_commit
    }

    pub fn channel(&self) -> UpdateChannel {
        self.channel
    }

    pub fn compatibility(&self) -> &CompatibilityV1 {
        &self.compatibility
    }

    pub fn files(&self) -> &[BundleFileV1] {
        &self.files
    }

    pub(crate) fn derived_directories(&self) -> &[String] {
        &self.derived_directories
    }

    pub fn payload_bytes(&self) -> u64 {
        self.files.iter().map(|file| file.size).sum()
    }

    pub fn code_signing(&self) -> &CodeSigningIdentityV1 {
        &self.code_signing
    }

    pub fn non_system_dynamic_dependencies(&self) -> &[DynamicDependencyV1] {
        &self.non_system_dynamic_dependencies
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CodeSigningIdentityV1 {
    team_id: String,
    identifier: String,
    certificate_common_name: String,
}

impl CodeSigningIdentityV1 {
    pub fn team_id(&self) -> &str {
        &self.team_id
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn certificate_common_name(&self) -> &str {
        &self.certificate_common_name
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompatibilityV1 {
    minimum_installer_protocol: u32,
    minimum_updater_protocol: u32,
    launcher_registry_schema: u32,
}

impl CompatibilityV1 {
    pub fn minimum_installer_protocol(&self) -> u32 {
        self.minimum_installer_protocol
    }

    pub fn minimum_updater_protocol(&self) -> u32 {
        self.minimum_updater_protocol
    }

    pub fn launcher_registry_schema(&self) -> u32 {
        self.launcher_registry_schema
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BundleFileV1 {
    path: BundlePath,
    #[serde(rename = "type")]
    file_type: BundleEntryType,
    size: u64,
    mode: FileMode,
    sha256: Sha256Digest,
}

impl BundleFileV1 {
    pub fn path(&self) -> &BundlePath {
        &self.path
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn file_type(&self) -> BundleEntryType {
        self.file_type
    }

    pub fn mode(&self) -> FileMode {
        self.mode
    }

    pub fn sha256(&self) -> &Sha256Digest {
        &self.sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicDependencyV1 {
    consumer: BundlePath,
    install_name: String,
}

impl DynamicDependencyV1 {
    pub fn consumer(&self) -> &BundlePath {
        &self.consumer
    }

    pub fn install_name(&self) -> &str {
        &self.install_name
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawReleaseManifestV1 {
    kind: String,
    schema_version: u32,
    package: String,
    version: String,
    target: String,
    minimum_macos: String,
    source_commit: String,
    channel: String,
    code_signing: RawCodeSigningIdentityV1,
    compatibility: RawCompatibilityV1,
    files: Vec<RawBundleFileV1>,
    non_system_dynamic_dependencies: Vec<RawDynamicDependencyV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCodeSigningIdentityV1 {
    team_id: String,
    identifier: String,
    certificate_common_name: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCompatibilityV1 {
    minimum_installer_protocol: u32,
    minimum_updater_protocol: u32,
    launcher_registry_schema: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBundleFileV1 {
    path: String,
    #[serde(rename = "type")]
    file_type: BundleEntryType,
    size: u64,
    mode: FileMode,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDynamicDependencyV1 {
    consumer: String,
    install_name: String,
}

#[derive(Default)]
struct BundleTreeInventory {
    files: BTreeMap<String, String>,
    directories: BTreeMap<String, String>,
}

impl BundleTreeInventory {
    fn record(&mut self, path: &BundlePath) -> Result<(), ReleaseManifestError> {
        let exact = path.as_str();
        let case_folded = exact.to_ascii_lowercase();

        if let Some(first) = self.files.get(&case_folded) {
            return if first == exact {
                Err(ReleaseManifestError::DuplicatePath(exact.to_owned()))
            } else {
                Err(ReleaseManifestError::PathCollision {
                    first: first.clone(),
                    second: exact.to_owned(),
                })
            };
        }
        if let Some(first) = self.directories.get(&case_folded) {
            return Err(ReleaseManifestError::PathCollision {
                first: first.clone(),
                second: exact.to_owned(),
            });
        }

        for (index, _) in case_folded.match_indices('/') {
            let folded_directory = &case_folded[..index];
            let exact_directory = &exact[..index];
            if let Some(first) = self.files.get(folded_directory) {
                return Err(ReleaseManifestError::PathCollision {
                    first: first.clone(),
                    second: exact.to_owned(),
                });
            }
            if let Some(first) = self.directories.get(folded_directory) {
                if first != exact_directory {
                    return Err(ReleaseManifestError::PathCollision {
                        first: first.clone(),
                        second: exact.to_owned(),
                    });
                }
                continue;
            }
            if self.directories.len() == MAX_BUNDLE_DIRECTORIES {
                return Err(ReleaseManifestError::TooManyEntries {
                    collection: "derived directories",
                    limit: MAX_BUNDLE_DIRECTORIES,
                    actual: MAX_BUNDLE_DIRECTORIES + 1,
                });
            }
            self.directories
                .insert(folded_directory.to_owned(), exact_directory.to_owned());
        }

        self.files.insert(case_folded, exact.to_owned());
        Ok(())
    }

    fn into_directories(self) -> Vec<String> {
        let mut directories: Vec<_> = self.directories.into_values().collect();
        directories.sort();
        directories
    }
}

impl TryFrom<RawReleaseManifestV1> for ReleaseManifestV1 {
    type Error = ReleaseManifestError;

    fn try_from(raw: RawReleaseManifestV1) -> Result<Self, Self::Error> {
        if raw.kind != RELEASE_MANIFEST_KIND {
            return Err(ReleaseManifestError::UnsupportedKind);
        }
        if raw.schema_version != RELEASE_MANIFEST_SCHEMA_VERSION {
            return Err(ReleaseManifestError::UnsupportedSchema(raw.schema_version));
        }
        if raw.package != "hf2q" {
            return Err(ReleaseManifestError::invalid(
                "package",
                "must be exactly `hf2q`",
            ));
        }
        if raw.files.len() > MAX_BUNDLE_FILES {
            return Err(ReleaseManifestError::TooManyEntries {
                collection: "files",
                limit: MAX_BUNDLE_FILES,
                actual: raw.files.len(),
            });
        }
        if raw.non_system_dynamic_dependencies.len() > MAX_DYNAMIC_DEPENDENCIES {
            return Err(ReleaseManifestError::TooManyEntries {
                collection: "non-system dynamic dependencies",
                limit: MAX_DYNAMIC_DEPENDENCIES,
                actual: raw.non_system_dynamic_dependencies.len(),
            });
        }

        let version = ReleaseVersion::parse_stable("version", raw.version)?;
        let target = TargetTriple::parse("target", raw.target)?;
        let minimum_macos = MacOsVersion::parse("minimum_macos", raw.minimum_macos)?;
        let source_commit = GitCommit::parse("source_commit", raw.source_commit)?;
        let channel = UpdateChannel::parse("channel", raw.channel)?;
        let code_signing = validate_code_signing(raw.code_signing)?;
        let compatibility = validate_compatibility(raw.compatibility)?;

        let mut files = Vec::with_capacity(raw.files.len());
        let mut tree = BundleTreeInventory::default();
        let mut previous: Option<String> = None;
        let mut payload_bytes = 0_u64;
        let mut has_binary = false;
        let mut has_doc = false;
        let mut has_license = false;
        for raw_file in raw.files {
            let path = BundlePath::parse("files[].path", raw_file.path)?;
            tree.record(&path)?;
            if previous
                .as_deref()
                .is_some_and(|prior| prior >= path.as_str())
            {
                return Err(ReleaseManifestError::UnsortedInventory(
                    path.as_str().to_owned(),
                ));
            }
            previous = Some(path.as_str().to_owned());

            let is_binary = path.as_str() == "bin/hf2q";
            let is_launcher = path.as_str().starts_with("libexec/");
            if (is_binary || is_launcher) && raw_file.mode != FileMode::Executable {
                return Err(ReleaseManifestError::invalid(
                    "files[].mode",
                    format!("{} must have mode 0755", path.as_str()),
                ));
            }
            if !(is_binary || is_launcher) && raw_file.mode != FileMode::Data {
                return Err(ReleaseManifestError::invalid(
                    "files[].mode",
                    format!("{} must have mode 0644", path.as_str()),
                ));
            }
            if is_binary && raw_file.size == 0 {
                return Err(ReleaseManifestError::invalid(
                    "files[].size",
                    "bin/hf2q cannot be empty",
                ));
            }

            payload_bytes = payload_bytes
                .checked_add(raw_file.size)
                .ok_or(ReleaseManifestError::PayloadSizeOverflow)?;
            has_binary |= is_binary;
            has_doc |= path.as_str().starts_with("share/doc/hf2q/");
            has_license |= path.as_str().starts_with("share/licenses/hf2q/");
            files.push(BundleFileV1 {
                path,
                file_type: raw_file.file_type,
                size: raw_file.size,
                mode: raw_file.mode,
                sha256: Sha256Digest::parse("files[].sha256", raw_file.sha256)?,
            });
        }
        if payload_bytes > MAX_BUNDLE_PAYLOAD_BYTES {
            return Err(ReleaseManifestError::PayloadTooLarge {
                limit: MAX_BUNDLE_PAYLOAD_BYTES,
                actual: payload_bytes,
            });
        }
        if !has_binary {
            return Err(ReleaseManifestError::MissingRequired("bin/hf2q"));
        }
        if !has_doc {
            return Err(ReleaseManifestError::MissingRequired(
                "share/doc/hf2q payload",
            ));
        }
        if !has_license {
            return Err(ReleaseManifestError::MissingRequired(
                "share/licenses/hf2q payload",
            ));
        }

        let file_modes: BTreeMap<_, _> = files
            .iter()
            .map(|file| (file.path.as_str(), file.mode))
            .collect();
        let mut dependencies = Vec::with_capacity(raw.non_system_dynamic_dependencies.len());
        let mut prior_dependency: Option<String> = None;
        for raw_dependency in raw.non_system_dynamic_dependencies {
            let consumer = BundlePath::parse(
                "non_system_dynamic_dependencies[].consumer",
                raw_dependency.consumer,
            )?;
            if file_modes.get(consumer.as_str()) != Some(&FileMode::Executable) {
                return Err(ReleaseManifestError::invalid(
                    "non_system_dynamic_dependencies[].consumer",
                    "must name an executable inventoried bundle file",
                ));
            }
            validate_install_name(&raw_dependency.install_name)?;
            let key = format!("{}\0{}", consumer.as_str(), raw_dependency.install_name);
            if prior_dependency
                .as_deref()
                .is_some_and(|prior| prior >= key.as_str())
            {
                return Err(ReleaseManifestError::invalid(
                    "non_system_dynamic_dependencies",
                    "must be strictly sorted and unique by consumer then install_name",
                ));
            }
            prior_dependency = Some(key);
            dependencies.push(DynamicDependencyV1 {
                consumer,
                install_name: raw_dependency.install_name,
            });
        }

        let derived_directories = tree.into_directories();
        Ok(Self {
            kind: raw.kind,
            schema_version: raw.schema_version,
            package: raw.package,
            version,
            target,
            minimum_macos,
            source_commit,
            channel,
            code_signing,
            compatibility,
            files,
            derived_directories,
            non_system_dynamic_dependencies: dependencies,
        })
    }
}

fn validate_code_signing(
    raw: RawCodeSigningIdentityV1,
) -> Result<CodeSigningIdentityV1, ReleaseManifestError> {
    if raw.team_id.len() != 10
        || !raw
            .team_id
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit())
    {
        return Err(ReleaseManifestError::invalid(
            "code_signing.team_id",
            "must be exactly 10 uppercase ASCII letters or digits",
        ));
    }
    if raw.identifier.is_empty()
        || raw.identifier.len() > 128
        || !raw.identifier.is_ascii()
        || raw.identifier.starts_with('.')
        || raw.identifier.ends_with('.')
        || raw.identifier.contains("..")
        || raw.identifier.split('.').any(|component| {
            !component
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_alphanumeric)
                || !component
                    .as_bytes()
                    .last()
                    .is_some_and(u8::is_ascii_alphanumeric)
        })
        || !raw
            .identifier
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))
    {
        return Err(ReleaseManifestError::invalid(
            "code_signing.identifier",
            "must be a bounded canonical signing identifier",
        ));
    }
    let expected_suffix = format!(" ({})", raw.team_id);
    let identity_body = raw
        .certificate_common_name
        .strip_prefix("Developer ID Application: ")
        .and_then(|value| value.strip_suffix(&expected_suffix));
    if raw.certificate_common_name.chars().count() > 256
        || raw.certificate_common_name.chars().any(char::is_control)
        || identity_body.is_none_or(str::is_empty)
    {
        return Err(ReleaseManifestError::invalid(
            "code_signing.certificate_common_name",
            "must be a Developer ID Application common name ending in the exact team ID",
        ));
    }
    Ok(CodeSigningIdentityV1 {
        team_id: raw.team_id,
        identifier: raw.identifier,
        certificate_common_name: raw.certificate_common_name,
    })
}

fn validate_compatibility(
    raw: RawCompatibilityV1,
) -> Result<CompatibilityV1, ReleaseManifestError> {
    validate_supported_capability(
        "installer",
        raw.minimum_installer_protocol,
        SUPPORTED_INSTALLER_PROTOCOL,
    )?;
    validate_supported_capability(
        "updater",
        raw.minimum_updater_protocol,
        SUPPORTED_UPDATER_PROTOCOL,
    )?;
    validate_supported_capability(
        "launcher registry",
        raw.launcher_registry_schema,
        SUPPORTED_LAUNCHER_REGISTRY_SCHEMA,
    )?;
    Ok(CompatibilityV1 {
        minimum_installer_protocol: raw.minimum_installer_protocol,
        minimum_updater_protocol: raw.minimum_updater_protocol,
        launcher_registry_schema: raw.launcher_registry_schema,
    })
}

fn validate_supported_capability(
    capability: &'static str,
    required: u32,
    supported: u32,
) -> Result<(), ReleaseManifestError> {
    if required == 0 {
        return Err(ReleaseManifestError::invalid(
            "compatibility",
            format!("{capability} protocol/schema cannot be zero"),
        ));
    }
    if required > supported {
        return Err(ReleaseManifestError::UnsupportedCompatibility {
            capability,
            required,
            supported,
        });
    }
    Ok(())
}

fn validate_install_name(value: &str) -> Result<(), ReleaseManifestError> {
    let allowed_prefix = ["@rpath/", "@loader_path/", "@executable_path/"]
        .into_iter()
        .find(|prefix| value.starts_with(prefix));
    let Some(prefix) = allowed_prefix else {
        return Err(ReleaseManifestError::invalid(
            "non_system_dynamic_dependencies[].install_name",
            "must use @rpath, @loader_path, or @executable_path",
        ));
    };
    let suffix = &value[prefix.len()..];
    if value.len() > 512
        || !value.is_ascii()
        || suffix.is_empty()
        || suffix.starts_with('/')
        || suffix.ends_with('/')
        || suffix.contains("//")
        || suffix.contains('\\')
        || !suffix.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'.' | b'_' | b'-' | b'+')
        })
        || suffix
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
        || value.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err(ReleaseManifestError::invalid(
            "non_system_dynamic_dependencies[].install_name",
            "must be a canonical loader-relative install name",
        ));
    }
    Ok(())
}

fn sanitize_json_error(error: serde_json::Error) -> ReleaseManifestError {
    let category = match error.classify() {
        serde_json::error::Category::Io => "I/O",
        serde_json::error::Category::Syntax => "syntax",
        serde_json::error::Category::Data => "data",
        serde_json::error::Category::Eof => "unexpected EOF",
    };
    ReleaseManifestError::Json {
        line: error.line(),
        column: error.column(),
        category,
    }
}

#[cfg(test)]
#[path = "release_manifest_tests.rs"]
mod tests;
