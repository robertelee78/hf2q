use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use super::fault::prepared_barrier;
use super::*;
use crate::distribution::install_state::file;
use crate::distribution::install_state::{host, PENDING_ACTIVATION, PENDING_CURRENT};
use crate::distribution::schema::RecordedPreparationEvidenceV2;

impl PublishedPreparedVersion {
    pub(in crate::distribution) fn receipt_bytes(&self) -> &[u8] {
        self.record.receipt_bytes()
    }
}

pub(super) fn build_record(
    locked: &LockedInstallationIdentity,
    authorization: &PreparedVersionAuthorization,
    installed_at: u64,
) -> Result<FirstStandaloneInstallRecord, PreparedVersionError> {
    if installed_at == 0 {
        return Err(PreparedVersionError::Integrity);
    }
    let [root, timestamp, snapshot, targets] = authorization.metadata_versions();
    let evidence =
        RecordedPreparationEvidenceV2::verified_update_metadata(root, timestamp, snapshot, targets)
            .map_err(crate::distribution::install_state::InstallStateError::Receipt)?;
    FirstStandaloneInstallRecord::build(
        locked.installation_id().clone(),
        locked.state_root().clone(),
        authorization.version().clone(),
        authorization.target(),
        authorization.manifest_sha256().clone(),
        authorization.archive_sha256().clone(),
        evidence,
        installed_at,
    )
    .map_err(crate::distribution::install_state::InstallStateError::Receipt)
    .map_err(PreparedVersionError::from)
}

pub(super) fn require_record(
    locked: &LockedInstallationIdentity,
    authorization: &PreparedVersionAuthorization,
    record: &FirstStandaloneInstallRecord,
    installed_at: u64,
) -> Result<(), PreparedVersionError> {
    let expected = build_record(locked, authorization, installed_at)?;
    if expected.marker_bytes() != record.marker_bytes()
        || expected.receipt_bytes() != record.receipt_bytes()
    {
        return Err(PreparedVersionError::Integrity);
    }
    Ok(())
}

pub(super) fn require_identity(
    locked: &LockedInstallationIdentity,
    authorization: &PreparedVersionAuthorization,
) -> Result<(), PreparedVersionError> {
    if locked.installation_id().as_str() != authorization.installation_id()
        || locked.state_root().as_str() != authorization.state_root()
    {
        return Err(PreparedVersionError::Integrity);
    }
    Ok(())
}

pub(super) fn require_release(
    locked: &LockedInstallationIdentity,
    authorization: &PreparedVersionAuthorization,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
) -> Result<(), PreparedVersionError> {
    require_identity(locked, authorization)?;
    host::require_compatible_host(manifest.minimum_macos()).map_err(PreparedVersionError::from)?;
    if manifest.version() != authorization.version()
        || manifest.target() != authorization.target()
        || manifest
            .to_deterministic_json()
            .map_err(|_| PreparedVersionError::Integrity)?
            != exact_manifest
        || hex::encode(Sha256::digest(exact_manifest)) != authorization.manifest_sha256().as_str()
    {
        return Err(PreparedVersionError::Integrity);
    }
    Ok(())
}

pub(super) fn optional_directory_nonempty(
    parent: &Directory,
    name: &str,
    bound: usize,
) -> Result<bool, crate::distribution::install_state::InstallStateError> {
    match unix::entry_identity(parent, name)? {
        None => Ok(false),
        Some(identity) => {
            let directory = unix::open_directory_at(parent, name, Some(0o700), true)?;
            if directory.device() != identity.device || directory.inode() != identity.inode {
                return Err(
                    crate::distribution::install_state::InstallStateError::InvalidLayout(
                        "installation namespace changed while checking metadata advancement",
                    ),
                );
            }
            Ok(!unix::list_names_bounded(&directory, bound)?.is_empty())
        }
    }
}

pub(super) fn require_no_activation_state(
    locked: &LockedInstallationIdentity,
) -> Result<(), PreparedVersionError> {
    for name in [
        "activations",
        "current",
        PENDING_ACTIVATION,
        PENDING_CURRENT,
        "uninstall",
    ] {
        if unix::entry_identity(locked.root(), name)?.is_some() {
            return Err(PreparedVersionError::Integrity);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum IntentKind {
    Partial,
    Ready,
}

#[derive(Clone, Copy)]
pub(super) struct PreparedInventory {
    pub(super) installed_at: Option<u64>,
    pub(super) kind: Option<IntentKind>,
    pub(super) pending: bool,
}

pub(super) fn classify_prepared(
    prepared: Option<&Directory>,
    authorization: &PreparedVersionAuthorization,
) -> Result<PreparedInventory, PreparedVersionError> {
    let Some(prepared) = prepared else {
        return Ok(PreparedInventory {
            installed_at: None,
            kind: None,
            pending: false,
        });
    };
    let names = unix::list_names_bounded(prepared, 2)?;
    let expected_pending = pending_name(authorization);
    let mut installed_at = None;
    let mut kind = None;
    let mut pending = false;
    for name in names {
        if name == expected_pending {
            if pending {
                return Err(PreparedVersionError::Integrity);
            }
            let _ = unix::open_directory_at(prepared, &name, Some(0o700), true)?;
            pending = true;
            continue;
        }
        let (timestamp, parsed_kind) = parse_marker_name(authorization, &name)?;
        if installed_at.replace(timestamp).is_some() {
            return Err(PreparedVersionError::Integrity);
        }
        kind = Some(parsed_kind);
        let _ = unix::open_private_regular_file(prepared, &name)?;
    }
    Ok(PreparedInventory {
        installed_at,
        kind,
        pending,
    })
}

pub(super) fn classify_versions(
    versions: Option<&Directory>,
    authorization: &PreparedVersionAuthorization,
) -> Result<bool, PreparedVersionError> {
    let Some(versions) = versions else {
        return Ok(false);
    };
    let names = unix::list_names_bounded(versions, 1)?;
    if names.is_empty() {
        return Ok(false);
    }
    if names != BTreeSet::from([authorization.version().as_str().to_owned()]) {
        return Err(PreparedVersionError::Integrity);
    }
    let _ = unix::open_directory_at(
        versions,
        authorization.version().as_str(),
        Some(0o700),
        true,
    )?;
    Ok(true)
}

pub(super) fn ensure_marker_intent(
    prepared: &Directory,
    authorization: &PreparedVersionAuthorization,
    record: &FirstStandaloneInstallRecord,
    inventory: PreparedInventory,
) -> Result<String, PreparedVersionError> {
    let installed_at = record.marker().installed_at_unix_seconds();
    let partial = marker_name(authorization, installed_at, IntentKind::Partial);
    let ready = marker_name(authorization, installed_at, IntentKind::Ready);
    match inventory.kind {
        None => {
            let marker =
                file::write_or_resume_private_file(prepared, &partial, record.marker_bytes())?;
            unix::full_sync_file(&marker)?;
            prepared_barrier()?;
            unix::rename_noreplace(prepared, &partial, prepared, &ready)?;
            unix::sync_directory(prepared)?;
            prepared_barrier()?;
        }
        Some(IntentKind::Partial) => {
            if inventory.installed_at != Some(installed_at) {
                return Err(PreparedVersionError::Integrity);
            }
            let marker =
                file::write_or_resume_private_file(prepared, &partial, record.marker_bytes())?;
            unix::full_sync_file(&marker)?;
            prepared_barrier()?;
            unix::rename_noreplace(prepared, &partial, prepared, &ready)?;
            unix::sync_directory(prepared)?;
            prepared_barrier()?;
        }
        Some(IntentKind::Ready) => {
            if inventory.installed_at != Some(installed_at) {
                return Err(PreparedVersionError::Integrity);
            }
            require_exact_file(prepared, &ready, record.marker_bytes())?;
        }
    }
    Ok(ready)
}

pub(super) fn record_from_tree(
    tree: &Directory,
) -> Result<FirstStandaloneInstallRecord, PreparedVersionError> {
    let (_, bytes, _) = file::read_regular_file(
        tree,
        MARKER_NAME,
        0o600,
        crate::distribution::schema::MAX_INSTALLED_VERSION_MARKER_BYTES,
    )?;
    FirstStandaloneInstallRecord::reconstruct_from_exact_marker(&bytes)
        .map_err(crate::distribution::install_state::InstallStateError::Receipt)
        .map_err(PreparedVersionError::from)
}

pub(super) fn require_exact_file(
    parent: &Directory,
    name: &str,
    expected: &[u8],
) -> Result<(), PreparedVersionError> {
    let (file, bytes, identity) = file::read_regular_file(parent, name, 0o600, expected.len())?;
    if identity.size != expected.len() as u64 || bytes != expected {
        return Err(PreparedVersionError::Integrity);
    }
    unix::verify_named_identity(parent, name, identity)?;
    unix::full_sync_file(&file)?;
    Ok(())
}

pub(super) fn verify_tree(
    tree: &Directory,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
    record: &FirstStandaloneInstallRecord,
) -> Result<(), PreparedVersionError> {
    let mut files = expected_files(exact_manifest, manifest);
    files.push(ExpectedFile {
        path: MARKER_NAME.to_owned(),
        size: record.marker_bytes().len() as u64,
        sha256: hex::encode(Sha256::digest(record.marker_bytes())),
        final_mode: 0o600,
    });
    let directories = manifest.derived_directories().to_vec();
    let expected = expected_tree(&directories, &files)?;
    let states = scan_tree(tree, &expected, &files, &directories)?;
    if states
        .iter()
        .any(|state| *state != FileState::CompleteFinal)
    {
        return Err(PreparedVersionError::Integrity);
    }
    Ok(())
}

pub(super) fn sync_tree(
    tree: &Directory,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
    record: &FirstStandaloneInstallRecord,
) -> Result<(), PreparedVersionError> {
    verify_tree(tree, exact_manifest, manifest, record)?;
    let mut files = expected_files(exact_manifest, manifest);
    files.push(ExpectedFile {
        path: MARKER_NAME.to_owned(),
        size: record.marker_bytes().len() as u64,
        sha256: hex::encode(Sha256::digest(record.marker_bytes())),
        final_mode: 0o600,
    });
    for expected in &files {
        let (parent, name) = open_existing_parent(tree, &expected.path)?;
        let (file, identity) =
            unix::open_regular_file_with_mode(&parent, name, expected.final_mode)?;
        verify_file(&file, identity, expected)?;
        unix::verify_named_identity(&parent, name, identity)?;
        unix::full_sync_file(&file)?;
        prepared_barrier()?;
        unix::verify_named_identity(&parent, name, identity)?;
    }
    for path in directory_normalization_order(manifest.derived_directories()) {
        let directory = open_existing_directory(tree, path)?;
        if directory.mode() != 0o755 {
            return Err(PreparedVersionError::Integrity);
        }
        unix::sync_directory(&directory)?;
        prepared_barrier()?;
    }
    unix::sync_directory(tree)?;
    prepared_barrier()?;
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn prepared_binding(
    locked: &LockedInstallationIdentity,
    state: &PreparedVersionState,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
) -> Result<ExecutableReleaseBinding, PreparedVersionError> {
    with_prepared_executable(
        locked,
        state,
        exact_manifest,
        manifest,
        |_path, _file, binding| Ok::<_, PreparedVersionError>(binding),
    )
}

#[cfg(target_os = "macos")]
pub(super) fn with_tree_executable<R, E>(
    locked: &LockedInstallationIdentity,
    retained_parent: &Directory,
    retained_tree: &Directory,
    parent_name: &str,
    tree_name: &str,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
    operation: impl FnOnce(&std::path::Path, &std::fs::File, ExecutableReleaseBinding) -> Result<R, E>,
) -> Result<R, E>
where
    E: From<PreparedVersionError>,
{
    let binary = expected_files(exact_manifest, manifest)
        .into_iter()
        .find(|file| file.path == "bin/hf2q")
        .ok_or(PreparedVersionError::Integrity)
        .map_err(E::from)?;
    let open = || -> Result<_, PreparedVersionError> {
        let live = locked.reopen()?;
        let parent = if parent_name == PREPARED {
            unix::open_directory_at(&live.update, parent_name, Some(0o700), true)?
        } else {
            unix::open_directory_at(&live.root, parent_name, Some(0o700), true)?
        };
        if !parent.same_object(retained_parent) {
            return Err(PreparedVersionError::Integrity);
        }
        let tree = unix::open_directory_at(&parent, tree_name, Some(0o700), true)?;
        if !tree.same_object(retained_tree) {
            return Err(PreparedVersionError::Integrity);
        }
        let (binary_parent, name) = open_existing_parent(&tree, &binary.path)?;
        let (file, identity) =
            unix::open_regular_file_with_mode(&binary_parent, name, binary.final_mode)?;
        verify_file(&file, identity, &binary)?;
        unix::verify_named_identity(&binary_parent, name, identity)?;
        let path = unix::file_descriptor_path(&file)?;
        if path != unix::directory_descriptor_path(&tree)?.join(&binary.path) {
            return Err(PreparedVersionError::Integrity);
        }
        let binding = ExecutableReleaseBinding {
            extractions_device: parent.device(),
            extractions_inode: parent.inode(),
            stage_device: tree.device(),
            stage_inode: tree.inode(),
            executable: identity,
            executable_path: path.clone(),
            manifest_sha256: Sha256::digest(exact_manifest).into(),
        };
        Ok((file, identity, path, binding))
    };
    let (file, identity, path, binding) = open().map_err(E::from)?;
    let result = operation(&path, &file, binding)?;
    let (reopened, reopened_identity, reopened_path, _) = open().map_err(E::from)?;
    if identity != reopened_identity || path != reopened_path {
        return Err(E::from(PreparedVersionError::Integrity));
    }
    verify_file(&reopened, reopened_identity, &binary)
        .map_err(PreparedVersionError::from)
        .map_err(E::from)?;
    Ok(result)
}

pub(super) fn verify_named_tree(
    locked: &LockedInstallationIdentity,
    retained_parent: &Directory,
    retained_tree: &Directory,
    parent_name: &str,
    tree_name: &str,
) -> Result<(), PreparedVersionError> {
    let live = locked.reopen()?;
    let parent = if parent_name == PREPARED {
        unix::open_directory_at(&live.update, parent_name, Some(0o700), true)?
    } else {
        unix::open_directory_at(&live.root, parent_name, Some(0o700), true)?
    };
    if !parent.same_object(retained_parent) {
        return Err(PreparedVersionError::Integrity);
    }
    let tree = unix::open_directory_at(&parent, tree_name, Some(0o700), true)?;
    if !tree.same_object(retained_tree) {
        return Err(PreparedVersionError::Integrity);
    }
    Ok(())
}

pub(super) fn open_optional_directory(
    parent: &Directory,
    name: &str,
) -> Result<Option<Directory>, PreparedVersionError> {
    match unix::entry_identity(parent, name)? {
        None => Ok(None),
        Some(_) => Ok(Some(unix::open_directory_at(
            parent,
            name,
            Some(0o700),
            true,
        )?)),
    }
}

pub(super) fn pending_name(authorization: &PreparedVersionAuthorization) -> String {
    format!(
        ".pending-v{}-{}",
        authorization.version().as_str(),
        authorization.archive_sha256().as_str()
    )
}

pub(super) fn marker_name(
    authorization: &PreparedVersionAuthorization,
    installed_at: u64,
    kind: IntentKind,
) -> String {
    let suffix = match kind {
        IntentKind::Partial => MARKER_PARTIAL_SUFFIX,
        IntentKind::Ready => MARKER_READY_SUFFIX,
    };
    format!(
        "{MARKER_PREFIX}{}-{}{MARKER_TIME_SEPARATOR}{installed_at:020}{suffix}",
        authorization.version().as_str(),
        authorization.archive_sha256().as_str(),
    )
}

pub(super) fn parse_marker_name(
    authorization: &PreparedVersionAuthorization,
    name: &str,
) -> Result<(u64, IntentKind), PreparedVersionError> {
    let base = format!(
        "{MARKER_PREFIX}{}-{}{MARKER_TIME_SEPARATOR}",
        authorization.version().as_str(),
        authorization.archive_sha256().as_str()
    );
    let remaining = name
        .strip_prefix(&base)
        .ok_or(PreparedVersionError::Integrity)?;
    let (digits, kind) = if let Some(digits) = remaining.strip_suffix(MARKER_PARTIAL_SUFFIX) {
        (digits, IntentKind::Partial)
    } else if let Some(digits) = remaining.strip_suffix(MARKER_READY_SUFFIX) {
        (digits, IntentKind::Ready)
    } else {
        return Err(PreparedVersionError::Integrity);
    };
    if digits.len() != 20 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(PreparedVersionError::Integrity);
    }
    let installed_at = digits
        .parse::<u64>()
        .map_err(|_| PreparedVersionError::Integrity)?;
    if installed_at == 0 || format!("{installed_at:020}") != digits {
        return Err(PreparedVersionError::Integrity);
    }
    Ok((installed_at, kind))
}
