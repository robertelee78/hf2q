//! Exact ownership receipt for shell-completion lifecycle cleanup.
//!
//! Registration files are removed only when their complete SHA-256 still
//! matches the bytes recorded after reconciliation. Startup files are never
//! removed wholesale: only the uniquely marked block is removed, and only
//! when that block's digest still matches. Operator edits therefore fail
//! closed and survive update, rollback, and uninstall.

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use super::completion_install::{atomic_replace_with_hook, capture_regular_target, ExpectedTarget};
use super::completion_startup::{self, StartupCleanup};

const SCHEMA_VERSION: u8 = 1;
const RECEIPT_FILE: &str = "completion-ownership-v1.json";
const MAX_RECEIPT_BYTES: u64 = 64 * 1024;
const MAX_ARTIFACTS: usize = 32;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    path: PathBuf,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema_version: u8,
    package: String,
    registrations: Vec<Artifact>,
    startup_blocks: Vec<Artifact>,
}

#[derive(Debug, Default)]
pub(crate) struct CompletionCleanup {
    pub(crate) removed: Vec<PathBuf>,
    pub(crate) preserved: Vec<String>,
}

pub(crate) fn owned_paths() -> Result<Vec<PathBuf>, String> {
    let Some(receipt) = load()? else {
        return Ok(Vec::new());
    };
    let mut paths = receipt
        .registrations
        .iter()
        .chain(&receipt.startup_blocks)
        .map(|artifact| artifact.path.clone())
        .collect::<Vec<_>>();
    if let Some(path) = receipt_path() {
        paths.push(path);
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

pub(super) fn record(registrations: &[PathBuf], startup_files: &[PathBuf]) -> Result<(), String> {
    let mut receipt = load()?.unwrap_or_else(empty_receipt);
    let current_registrations = digest_regular_files(registrations)?;
    let current_startup = digest_startup_blocks(startup_files)?;
    if current_registrations.is_empty() && current_startup.is_empty() {
        return Ok(());
    }
    receipt.registrations = merge(receipt.registrations, current_registrations)?;
    receipt.startup_blocks = merge(receipt.startup_blocks, current_startup)?;
    persist(&receipt)
}

pub(crate) fn cleanup_owned() -> CompletionCleanup {
    let mut summary = CompletionCleanup::default();
    let receipt = match load() {
        Ok(Some(receipt)) => receipt,
        Ok(None) => return summary,
        Err(error) => {
            summary
                .preserved
                .push(format!("completion ownership receipt: {error}"));
            return summary;
        }
    };

    for artifact in &receipt.registrations {
        match remove_exact_regular(artifact) {
            Ok(RemoveExact::Removed) => summary.removed.push(artifact.path.clone()),
            Ok(RemoveExact::Absent) => {}
            Ok(RemoveExact::Preserved(reason)) => summary.preserved.push(format!(
                "completion registration {}: {reason}",
                artifact.path.display()
            )),
            Err(error) => summary.preserved.push(format!(
                "completion registration {}: {error}",
                artifact.path.display()
            )),
        }
    }
    for artifact in &receipt.startup_blocks {
        match completion_startup::remove_managed_block(&artifact.path, &artifact.sha256) {
            Ok(StartupCleanup::Removed) => summary.removed.push(artifact.path.clone()),
            Ok(StartupCleanup::Absent) => {}
            Ok(StartupCleanup::Preserved(reason)) => summary.preserved.push(format!(
                "completion startup {}: {reason}",
                artifact.path.display()
            )),
            Err(error) => summary.preserved.push(format!(
                "completion startup {}: {error}",
                artifact.path.display()
            )),
        }
    }

    if summary.preserved.is_empty() {
        if let Some(path) = receipt_path() {
            match remove_receipt(&path) {
                Ok(true) => summary.removed.push(path),
                Ok(false) => {}
                Err(error) => summary
                    .preserved
                    .push(format!("completion ownership receipt: {error}")),
            }
        }
    }
    summary.removed.sort();
    summary.removed.dedup();
    summary.preserved.sort();
    summary
}

fn empty_receipt() -> Receipt {
    Receipt {
        schema_version: SCHEMA_VERSION,
        package: "hf2q".to_owned(),
        registrations: Vec::new(),
        startup_blocks: Vec::new(),
    }
}

fn digest_regular_files(paths: &[PathBuf]) -> Result<Vec<Artifact>, String> {
    let mut artifacts = Vec::new();
    for path in paths {
        validate_absolute(path)?;
        let metadata = fs::symlink_metadata(path)
            .map_err(|error| format!("stat {}: {error}", path.display()))?;
        if !metadata.file_type().is_file() {
            return Err(format!("{} is not a regular file", path.display()));
        }
        let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
        artifacts.push(Artifact {
            path: path.clone(),
            sha256: sha256(&bytes),
        });
    }
    Ok(artifacts)
}

fn digest_startup_blocks(paths: &[PathBuf]) -> Result<Vec<Artifact>, String> {
    let mut artifacts = Vec::new();
    for path in paths {
        validate_absolute(path)?;
        let digest = completion_startup::managed_block_digest(path)?
            .ok_or_else(|| format!("managed block missing from {}", path.display()))?;
        artifacts.push(Artifact {
            path: path.clone(),
            sha256: digest,
        });
    }
    Ok(artifacts)
}

fn merge(previous: Vec<Artifact>, current: Vec<Artifact>) -> Result<Vec<Artifact>, String> {
    let mut by_path = BTreeMap::new();
    for artifact in previous.into_iter().chain(current) {
        validate_artifact(&artifact)?;
        by_path.insert(artifact.path.clone(), artifact);
    }
    if by_path.len() > MAX_ARTIFACTS {
        return Err(format!(
            "completion ownership exceeds {MAX_ARTIFACTS} artifacts"
        ));
    }
    Ok(by_path.into_values().collect())
}

fn receipt_path() -> Option<PathBuf> {
    let root = std::env::var_os("XDG_STATE_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
                .map(|home| home.join(".local/state"))
        })?;
    Some(root.join("hf2q").join(RECEIPT_FILE))
}

fn load() -> Result<Option<Receipt>, String> {
    let Some(path) = receipt_path() else {
        return Ok(None);
    };
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("stat {}: {error}", path.display())),
    };
    if !metadata.file_type().is_file()
        || metadata.uid() != rustix::process::geteuid().as_raw()
        || metadata.permissions().mode() & 0o077 != 0
        || metadata.len() > MAX_RECEIPT_BYTES
    {
        return Err(format!(
            "{} is not a bounded private current-user regular file",
            path.display()
        ));
    }
    let bytes = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let receipt: Receipt = serde_json::from_slice(&bytes)
        .map_err(|error| format!("decode {}: {error}", path.display()))?;
    validate_receipt(&receipt)?;
    Ok(Some(receipt))
}

fn validate_receipt(receipt: &Receipt) -> Result<(), String> {
    if receipt.schema_version != SCHEMA_VERSION || receipt.package != "hf2q" {
        return Err("unsupported completion ownership receipt identity".to_owned());
    }
    if receipt.registrations.len() + receipt.startup_blocks.len() > MAX_ARTIFACTS {
        return Err(format!(
            "completion ownership exceeds {MAX_ARTIFACTS} artifacts"
        ));
    }
    for artifact in receipt.registrations.iter().chain(&receipt.startup_blocks) {
        validate_artifact(artifact)?;
    }
    Ok(())
}

fn validate_artifact(artifact: &Artifact) -> Result<(), String> {
    validate_absolute(&artifact.path)?;
    if artifact.sha256.len() != 64
        || !artifact
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("invalid digest for {}", artifact.path.display()));
    }
    Ok(())
}

fn validate_absolute(path: &Path) -> Result<(), String> {
    if path.is_absolute() {
        Ok(())
    } else {
        Err(format!(
            "completion ownership path is relative: {}",
            path.display()
        ))
    }
}

fn persist(receipt: &Receipt) -> Result<(), String> {
    let path = receipt_path().ok_or_else(|| "HOME/XDG state root is unset".to_owned())?;
    let parent = path
        .parent()
        .ok_or_else(|| "completion receipt has no parent".to_owned())?;
    fs::create_dir_all(parent).map_err(|error| format!("create {}: {error}", parent.display()))?;
    let mut bytes = serde_json::to_vec_pretty(receipt)
        .map_err(|error| format!("encode completion ownership receipt: {error}"))?;
    bytes.push(b'\n');
    if bytes.len() as u64 > MAX_RECEIPT_BYTES {
        return Err("completion ownership receipt exceeds its size bound".to_owned());
    }
    let expected = match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_file() => capture_regular_target(&path)
            .map_err(|error| format!("read {}: {error}", path.display()))?,
        Ok(_) => return Err(format!("{} is not a regular file", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => ExpectedTarget::Absent,
        Err(error) => return Err(format!("stat {}: {error}", path.display())),
    };
    if expected.bytes() == bytes {
        return Ok(());
    }
    atomic_replace_with_hook(
        parent,
        &path,
        &bytes,
        0o600,
        &expected,
        "completion-receipt",
        || {},
    )
    .map_err(|error| format!("write {}: {error}", path.display()))
}

enum RemoveExact {
    Removed,
    Absent,
    Preserved(String),
}

fn remove_exact_regular(artifact: &Artifact) -> Result<RemoveExact, String> {
    let metadata = match fs::symlink_metadata(&artifact.path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(RemoveExact::Absent);
        }
        Err(error) => return Err(format!("stat failed: {error}")),
    };
    if !metadata.file_type().is_file() {
        return Ok(RemoveExact::Preserved(
            "path is no longer a regular file".to_owned(),
        ));
    }
    let expected =
        capture_regular_target(&artifact.path).map_err(|error| format!("read failed: {error}"))?;
    if sha256(expected.bytes()) != artifact.sha256 {
        return Ok(RemoveExact::Preserved(
            "file was modified after installation".to_owned(),
        ));
    }
    expected
        .revalidate(&artifact.path)
        .map_err(|error| format!("final revalidation failed: {error}"))?;
    fs::remove_file(&artifact.path).map_err(|error| format!("remove failed: {error}"))?;
    Ok(RemoveExact::Removed)
}

fn remove_receipt(path: &Path) -> Result<bool, String> {
    let expected = match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => capture_regular_target(path)
            .map_err(|error| format!("read {}: {error}", path.display()))?,
        Ok(_) => return Err(format!("{} is not a regular file", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(format!("stat {}: {error}", path.display())),
    };
    expected
        .revalidate(path)
        .map_err(|error| format!("revalidate {}: {error}", path.display()))?;
    fs::remove_file(path).map_err(|error| format!("remove {}: {error}", path.display()))?;
    Ok(true)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_registration_cleanup_is_digest_gated() {
        let root = tempfile::tempdir().unwrap();
        let exact = root.path().join("hf2q");
        fs::write(&exact, b"managed registration\n").unwrap();
        let artifact = Artifact {
            path: exact.clone(),
            sha256: sha256(b"managed registration\n"),
        };
        assert!(matches!(
            remove_exact_regular(&artifact).unwrap(),
            RemoveExact::Removed
        ));
        assert!(!exact.exists());

        let modified = root.path().join("_hf2q");
        fs::write(&modified, b"operator modification\n").unwrap();
        let stale = Artifact {
            path: modified.clone(),
            sha256: sha256(b"original managed registration\n"),
        };
        assert!(matches!(
            remove_exact_regular(&stale).unwrap(),
            RemoveExact::Preserved(_)
        ));
        assert_eq!(fs::read(modified).unwrap(), b"operator modification\n");
    }

    #[test]
    fn receipt_validation_rejects_relative_paths_and_uppercase_digests() {
        let relative = Artifact {
            path: PathBuf::from("relative/hf2q"),
            sha256: "0".repeat(64),
        };
        assert!(validate_artifact(&relative).is_err());
        let uppercase = Artifact {
            path: PathBuf::from("/tmp/hf2q"),
            sha256: "A".repeat(64),
        };
        assert!(validate_artifact(&uppercase).is_err());
    }
}
