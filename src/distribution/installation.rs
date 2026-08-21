//! Fail-closed ownership for the installation that is running this process.
//!
//! ADR-045 permits one owner, never a best-effort guess. Standalone ownership
//! comes from its adjacent marker. Cargo ownership comes from Cargo's own two
//! tracking files under the root derived from `<root>/bin/hf2q`. A standard
//! source build is recognized only at Cargo's conventional target paths.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::{self, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

mod cargo;
mod manager;

pub(crate) use self::cargo::{
    reconcile_uninstall as reconcile_cargo_uninstall, reconcile_update as reconcile_cargo_update,
};
pub(crate) use self::manager::ManagerCommand;

const STANDALONE_MARKER: &str = ".hf2q-standalone.json";
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub(crate) enum InstallationError {
    #[error("invalid hf2q installation evidence: {0}")]
    Invalid(String),
    #[error("ambiguous hf2q installation ownership: {0}")]
    Ambiguous(String),
    #[error("installation manager command failed: {0}")]
    Manager(String),
    #[error("installation filesystem operation `{operation}` failed: {source}")]
    Io {
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },
}

impl InstallationError {
    fn io(operation: &'static str, source: std::io::Error) -> Self {
        Self::Io { operation, source }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Installation {
    Standalone {
        install_dir: PathBuf,
    },
    Cargo {
        root: PathBuf,
        version: semver::Version,
        source: CargoSource,
        options: CargoInstallOptions,
    },
    SourceDevelopment {
        workspace_root: PathBuf,
        profile: SourceProfile,
    },
    Unmanaged {
        executable: PathBuf,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CargoInstallOptions {
    pub(crate) version_req: Option<String>,
    pub(crate) features: BTreeSet<String>,
    pub(crate) all_features: bool,
    pub(crate) no_default_features: bool,
    pub(crate) profile: String,
    pub(crate) target: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CargoSource {
    CratesIo,
    Path(PathBuf),
    Git(CargoGitSource),
    OtherRegistry(String),
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CargoGitSource {
    pub(crate) repository: String,
    pub(crate) selector: Option<CargoGitSelector>,
    pub(crate) resolved_revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CargoGitSelector {
    Branch(String),
    Tag(String),
    Rev(String),
}

impl CargoSource {
    pub(crate) fn description(&self) -> String {
        match self {
            Self::CratesIo => "crates.io registry".to_owned(),
            Self::Path(path) => format!("local path {}", path.display()),
            Self::Git(source) => {
                let selector = match &source.selector {
                    Some(CargoGitSelector::Branch(value)) => format!("branch {value}"),
                    Some(CargoGitSelector::Tag(value)) => format!("tag {value}"),
                    Some(CargoGitSelector::Rev(value)) => format!("revision {value}"),
                    None => "default branch".to_owned(),
                };
                format!("Git repository {} ({selector})", source.repository)
            }
            Self::OtherRegistry(source) => format!("custom registry {source}"),
            Self::Other(source) => format!("unsupported Cargo source {source}"),
        }
    }

    pub(crate) fn same_channel(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Git(left), Self::Git(right)) => {
                left.repository == right.repository && left.selector == right.selector
            }
            _ => self == other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceProfile {
    Debug,
    Release,
}

impl SourceProfile {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }
}

pub(crate) fn detect(executable: &Path) -> Result<Installation, InstallationError> {
    detect_for_context(
        executable,
        env!("CARGO_PKG_VERSION"),
        Path::new(env!("CARGO_MANIFEST_DIR")),
    )
}

#[cfg(test)]
pub(super) fn detect_with_manifest_root(
    executable: &Path,
    manifest_root: &Path,
) -> Result<Installation, InstallationError> {
    detect_for_context(executable, env!("CARGO_PKG_VERSION"), manifest_root)
}

fn detect_for_context(
    executable: &Path,
    expected_version: &str,
    manifest_root: &Path,
) -> Result<Installation, InstallationError> {
    let executable = canonical_executable(executable)?;
    let standalone = inspect_standalone(&executable)?;
    let cargo = cargo::inspect(&executable, expected_version)?;
    match (standalone, cargo) {
        (Some(_), Some(_)) => {
            return Err(InstallationError::Ambiguous(format!(
                "{} is claimed by both standalone and Cargo metadata",
                executable.display()
            )))
        }
        (Some(install_dir), None) => return Ok(Installation::Standalone { install_dir }),
        (None, Some((root, record))) => {
            return Ok(Installation::Cargo {
                root,
                version: record.version,
                source: record.source,
                options: record.options,
            })
        }
        (None, None) => {}
    }
    if let Some((workspace_root, profile)) = inspect_source(&executable, manifest_root)? {
        return Ok(Installation::SourceDevelopment {
            workspace_root,
            profile,
        });
    }
    Ok(Installation::Unmanaged { executable })
}

fn canonical_executable(executable: &Path) -> Result<PathBuf, InstallationError> {
    let canonical = fs::canonicalize(executable)
        .map_err(|error| InstallationError::io("canonicalize running executable", error))?;
    let metadata = fs::metadata(&canonical)
        .map_err(|error| InstallationError::io("inspect running executable", error))?;
    if !metadata.is_file()
        || metadata.uid() != rustix::process::geteuid().as_raw()
        || metadata.mode() & 0o022 != 0
        || metadata.nlink() != 1
    {
        return Err(InstallationError::Invalid(
            "the running executable is not a current-user-controlled single-link regular file"
                .to_owned(),
        ));
    }
    Ok(canonical)
}

fn inspect_standalone(executable: &Path) -> Result<Option<PathBuf>, InstallationError> {
    let directory = executable.parent().ok_or_else(|| {
        InstallationError::Invalid("the running executable has no parent directory".to_owned())
    })?;
    if !path_entry_exists(&directory.join(STANDALONE_MARKER))? {
        return Ok(None);
    }
    super::standalone::verify_running_installation(executable)
        .map(Some)
        .map_err(|error| InstallationError::Invalid(error.to_string()))
}

fn inspect_source(
    executable: &Path,
    manifest_root: &Path,
) -> Result<Option<(PathBuf, SourceProfile)>, InstallationError> {
    if executable.file_name() != Some(OsStr::new("hf2q")) {
        return Ok(None);
    }
    let Some(profile_dir) = executable.parent() else {
        return Ok(None);
    };
    let profile = match profile_dir.file_name().and_then(OsStr::to_str) {
        Some("debug") => SourceProfile::Debug,
        Some("release") => SourceProfile::Release,
        _ => return Ok(None),
    };
    let Some(before_profile) = profile_dir.parent() else {
        return Ok(None);
    };
    let workspace_root = if before_profile.file_name() == Some(OsStr::new("target")) {
        before_profile.parent()
    } else if before_profile.parent().and_then(Path::file_name) == Some(OsStr::new("target")) {
        before_profile.parent().and_then(Path::parent)
    } else {
        None
    };
    let Some(workspace_root) = workspace_root else {
        return Ok(None);
    };
    let canonical_manifest_root = match fs::canonicalize(manifest_root) {
        Ok(path) => path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(InstallationError::io(
                "canonicalize compiled source root",
                error,
            ))
        }
    };
    if workspace_root != canonical_manifest_root {
        return Ok(None);
    }
    let manifest_path = workspace_root.join("Cargo.toml");
    if !path_entry_exists(&manifest_path)? {
        return Ok(None);
    }
    let bytes = read_bounded_manifest(&manifest_path, MAX_MANIFEST_BYTES)?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| InstallationError::Invalid("source Cargo.toml is not UTF-8".to_owned()))?;
    let value: toml::Value = toml::from_str(text)
        .map_err(|_| InstallationError::Invalid("source Cargo.toml is malformed".to_owned()))?;
    if value
        .get("package")
        .and_then(toml::Value::as_table)
        .and_then(|package| package.get("name"))
        .and_then(toml::Value::as_str)
        != Some("hf2q")
    {
        return Ok(None);
    }
    Ok(Some((workspace_root.to_owned(), profile)))
}

fn read_bounded_manifest(path: &Path, maximum: u64) -> Result<Vec<u8>, InstallationError> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| InstallationError::io("open source manifest", error))?;
    let metadata = file
        .metadata()
        .map_err(|error| InstallationError::io("inspect source manifest", error))?;
    if !metadata.is_file()
        || metadata.mode() & 0o022 != 0
        || metadata.len() == 0
        || metadata.len() > maximum
    {
        return Err(InstallationError::Invalid(
            "source Cargo.toml is not a bounded non-writable regular file".to_owned(),
        ));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| InstallationError::io("read source manifest", error))?;
    if bytes.len() as u64 > maximum {
        return Err(InstallationError::Invalid(
            "source Cargo.toml exceeds the size bound".to_owned(),
        ));
    }
    Ok(bytes)
}

fn path_entry_exists(path: &Path) -> Result<bool, InstallationError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(InstallationError::io(
            "inspect installation evidence",
            error,
        )),
    }
}
