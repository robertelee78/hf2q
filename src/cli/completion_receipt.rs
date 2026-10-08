//! Exact ownership receipt for shell-completion lifecycle cleanup.
//!
//! Registration files are removed only when their complete SHA-256 still
//! matches the bytes recorded after reconciliation. Startup files are never
//! removed wholesale: only the uniquely marked block is removed, and only
//! when that block's digest still matches. Operator edits therefore fail
//! closed and survive update, rollback, and uninstall.
//!
//! The receipt is per-user, so it only ever registers and removes paths inside
//! the operator's `$HOME`. A machine-wide path (for example a Homebrew
//! `site-functions` file recorded by an older hf2q) is shared with other
//! accounts and installations; it is never registered, and a legacy entry for
//! one is skipped untouched on cleanup and pruned on the next record (#246).

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
use std::path::{Component, Path, PathBuf};

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
    let Some(receipt_path) = receipt_path() else {
        return Ok(Vec::new());
    };
    owned_paths_in(&receipt_path, user_home().as_deref())
}

pub(super) fn owned_paths_in(
    receipt_path: &Path,
    home: Option<&Path>,
) -> Result<Vec<PathBuf>, String> {
    let Some(receipt) = load_from(receipt_path)? else {
        return Ok(Vec::new());
    };
    let mut paths = receipt
        .registrations
        .iter()
        .chain(&receipt.startup_blocks)
        .map(|artifact| artifact.path.clone())
        .filter(|path| within_home(path, home))
        .collect::<Vec<_>>();
    paths.push(receipt_path.to_path_buf());
    paths.sort();
    paths.dedup();
    Ok(paths)
}

pub(super) fn record(registrations: &[PathBuf], startup_files: &[PathBuf]) -> Result<(), String> {
    let path = receipt_path().ok_or_else(|| "HOME/XDG state root is unset".to_owned())?;
    record_in(&path, user_home().as_deref(), registrations, startup_files)
}

/// Record the current artifacts in the receipt at `receipt_path`. Only paths
/// inside `home` are registered; any previously recorded entry outside it (a
/// legacy machine-wide registration) is pruned without touching the file.
pub(super) fn record_in(
    receipt_path: &Path,
    home: Option<&Path>,
    registrations: &[PathBuf],
    startup_files: &[PathBuf],
) -> Result<(), String> {
    let in_home = |paths: &[PathBuf]| {
        paths
            .iter()
            .filter(|path| within_home(path, home))
            .cloned()
            .collect::<Vec<_>>()
    };
    let loaded = load_from(receipt_path)?;
    let had_receipt = loaded.is_some();
    let mut receipt = loaded.unwrap_or_else(empty_receipt);
    let current_registrations = digest_regular_files(&in_home(registrations))?;
    let current_startup = digest_startup_blocks(&in_home(startup_files))?;
    if current_registrations.is_empty() && current_startup.is_empty() && !had_receipt {
        return Ok(());
    }
    let keep = |artifacts: Vec<Artifact>| {
        artifacts
            .into_iter()
            .filter(|artifact| within_home(&artifact.path, home))
            .collect::<Vec<_>>()
    };
    receipt.registrations = merge(keep(receipt.registrations), current_registrations)?;
    receipt.startup_blocks = merge(keep(receipt.startup_blocks), current_startup)?;
    persist_to(receipt_path, &receipt)
}

pub(crate) fn cleanup_owned() -> CompletionCleanup {
    match receipt_path() {
        Some(path) => cleanup_owned_in(&path, user_home().as_deref()),
        None => CompletionCleanup::default(),
    }
}

/// Remove the unchanged artifacts recorded in the receipt at `receipt_path`.
/// Entries outside `home` are never touched: they are machine-wide paths that
/// other accounts or installations may rely on, so they are skipped rather
/// than removed or reported as preserved hf2q state.
pub(super) fn cleanup_owned_in(receipt_path: &Path, home: Option<&Path>) -> CompletionCleanup {
    let mut summary = CompletionCleanup::default();
    let receipt = match load_from(receipt_path) {
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
        if !within_home(&artifact.path, home) {
            tracing::debug!(path = %artifact.path.display(), "skipping completion registration outside HOME");
            continue;
        }
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
        if !within_home(&artifact.path, home) {
            tracing::debug!(path = %artifact.path.display(), "skipping completion startup block outside HOME");
            continue;
        }
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
        match remove_receipt(receipt_path) {
            Ok(true) => summary.removed.push(receipt_path.to_path_buf()),
            Ok(false) => {}
            Err(error) => summary
                .preserved
                .push(format!("completion ownership receipt: {error}")),
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

/// The operator's home directory, from `$HOME`. `None` when unset or empty;
/// then nothing is considered in-home and nothing is registered or removed.
fn user_home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// True only when `path` is an absolute, `..`-free path whose parent directory
/// resolves (through any symlinks) to a location inside `home`. Machine-wide
/// directories such as `/opt/homebrew/share/zsh/site-functions` or
/// `/usr/local/share/...` are therefore never in scope for the per-user
/// receipt. A missing parent is resolved through its nearest existing
/// ancestor. A home that resolves to `/` matches nothing.
fn within_home(path: &Path, home: Option<&Path>) -> bool {
    let Some(home) = home else {
        return false;
    };
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return false;
    }
    let Ok(home) = fs::canonicalize(home) else {
        return false;
    };
    if home.parent().is_none() {
        return false;
    }
    let mut probe = path.parent();
    while let Some(dir) = probe {
        if let Ok(real) = fs::canonicalize(dir) {
            return real.starts_with(&home);
        }
        probe = dir.parent();
    }
    false
}

fn load_from(path: &Path) -> Result<Option<Receipt>, String> {
    let metadata = match fs::symlink_metadata(path) {
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
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
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

fn persist_to(path: &Path, receipt: &Receipt) -> Result<(), String> {
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
    let expected = match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => capture_regular_target(path)
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
        path,
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

    #[test]
    fn only_paths_inside_home_are_in_scope() {
        let root = tempfile::tempdir().unwrap();
        let home = root.path().join("home");
        fs::create_dir_all(home.join(".local/share/zsh/site-functions")).unwrap();
        let home = Some(home.as_path());
        let inside = root
            .path()
            .join("home/.local/share/zsh/site-functions/_hf2q");
        assert!(within_home(&inside, home));
        // A not-yet-created directory still resolves through its ancestor.
        assert!(within_home(
            &root.path().join("home/.config/fish/completions/hf2q.fish"),
            home
        ));
        for outside in [
            PathBuf::from("/opt/homebrew/share/zsh/site-functions/_hf2q"),
            PathBuf::from("/usr/local/share/zsh/site-functions/_hf2q"),
            root.path().join("elsewhere/_hf2q"),
            root.path().join("home/../elsewhere/_hf2q"),
            PathBuf::from("relative/_hf2q"),
        ] {
            assert!(!within_home(&outside, home), "{}", outside.display());
        }
        assert!(!within_home(&inside, None));
        // A home that escapes through a symlinked directory is resolved.
        let link = root.path().join("home/link-out");
        fs::create_dir_all(root.path().join("elsewhere")).unwrap();
        std::os::unix::fs::symlink(root.path().join("elsewhere"), &link).unwrap();
        assert!(!within_home(&link.join("_hf2q"), home));
    }

    /// A receipt written by an older hf2q that registered a machine-wide
    /// Homebrew-style path: uninstall must leave that path untouched (another
    /// installation may own it) while still removing the per-user artifacts,
    /// and the next record prunes the legacy entry without touching the file.
    #[test]
    fn legacy_receipt_entry_outside_home_is_left_untouched() {
        let root = tempfile::tempdir().unwrap();
        let home = root.path().join("home");
        let shared_dir = root.path().join("opt/homebrew/share/zsh/site-functions");
        let user_dir = home.join(".local/share/zsh/site-functions");
        fs::create_dir_all(&shared_dir).unwrap();
        fs::create_dir_all(&user_dir).unwrap();
        let shared = shared_dir.join("_hf2q");
        let user = user_dir.join("_hf2q");
        let bytes = b"#compdef hf2q\n# hf2q-managed dynamic completion\n";
        fs::write(&shared, bytes).unwrap();
        fs::write(&user, bytes).unwrap();
        let receipt_path = home.join(".local/state/hf2q").join(RECEIPT_FILE);
        let legacy = Receipt {
            schema_version: SCHEMA_VERSION,
            package: "hf2q".to_owned(),
            registrations: vec![
                Artifact {
                    path: shared.clone(),
                    sha256: sha256(bytes),
                },
                Artifact {
                    path: user.clone(),
                    sha256: sha256(bytes),
                },
            ],
            startup_blocks: Vec::new(),
        };
        persist_to(&receipt_path, &legacy).unwrap();

        let owned = owned_paths_in(&receipt_path, Some(&home)).unwrap();
        assert!(owned.contains(&user));
        assert!(!owned.contains(&shared), "{owned:?}");

        let cleanup = cleanup_owned_in(&receipt_path, Some(&home));
        assert!(cleanup.preserved.is_empty(), "{:?}", cleanup.preserved);
        assert!(cleanup.removed.contains(&user));
        assert!(cleanup.removed.contains(&receipt_path));
        assert!(!cleanup.removed.contains(&shared));
        assert!(!user.exists());
        assert!(!receipt_path.exists());
        assert_eq!(fs::read(&shared).unwrap(), bytes);

        // Recording over a legacy receipt prunes the out-of-home entry and
        // leaves the shared file alone.
        fs::write(&user, bytes).unwrap();
        persist_to(&receipt_path, &legacy).unwrap();
        record_in(&receipt_path, Some(&home), &[user.clone()], &[]).unwrap();
        let pruned = load_from(&receipt_path).unwrap().unwrap();
        assert_eq!(
            pruned
                .registrations
                .iter()
                .map(|artifact| artifact.path.clone())
                .collect::<Vec<_>>(),
            vec![user.clone()]
        );
        assert_eq!(fs::read(&shared).unwrap(), bytes);

        // An out-of-home registration offered by the current run is never
        // recorded either.
        record_in(&receipt_path, Some(&home), &[shared.clone()], &[]).unwrap();
        let after = load_from(&receipt_path).unwrap().unwrap();
        assert!(after.registrations.iter().all(|a| a.path != shared));
    }
}
