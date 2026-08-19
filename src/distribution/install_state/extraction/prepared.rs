#[cfg(test)]
use std::sync::atomic::AtomicUsize;

use super::*;
use crate::distribution::schema::{FirstStandaloneInstallRecord, ReleaseManifestV1};
use crate::distribution::update_auth::PreparedVersionAuthorization;

const PREPARED: &str = "prepared";
const VERSIONS: &str = "versions";
const MARKER_NAME: &str = "version-installation.json";
const MARKER_PREFIX: &str = ".marker-v";
const MARKER_TIME_SEPARATOR: &str = "-t";
const MARKER_PARTIAL_SUFFIX: &str = ".partial";
const MARKER_READY_SUFFIX: &str = ".ready";

#[cfg(test)]
static ABORT_AFTER_PREPARED_BARRIER: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static FAIL_AFTER_PREPARED_BARRIER: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static OBSERVED_PREPARED_BARRIERS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
std::thread_local! {
    static PREPARED_PRECOMMIT_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

pub(in crate::distribution::install_state) fn has_recoverable_version(
    locked: &LockedInstallationIdentity,
    authorization: &PreparedVersionAuthorization,
) -> Result<bool, PreparedVersionError> {
    require_identity(locked, authorization)?;
    require_no_activation_state(locked)?;
    let live = locked.reopen()?;
    let prepared = open_optional_directory(&live.update, PREPARED)?;
    let versions = open_optional_directory(&live.root, VERSIONS)?;
    let pending_name = pending_name(authorization);
    let prepared_inventory = classify_prepared(prepared.as_ref(), authorization)?;
    let final_present = classify_versions(versions.as_ref(), authorization)?;
    if prepared_inventory.pending && final_present {
        return Err(PreparedVersionError::Integrity);
    }
    if prepared_inventory.pending {
        let prepared = prepared.as_ref().ok_or(PreparedVersionError::Integrity)?;
        let _ = unix::open_directory_at(prepared, &pending_name, Some(0o700), true)?;
    }
    Ok(prepared_inventory.pending || final_present)
}

pub(in crate::distribution::install_state) fn require_metadata_advancement_safe(
    locked: &LockedInstallationIdentity,
) -> Result<(), crate::distribution::install_state::InstallStateError> {
    let live = locked.reopen()?;
    if let Some(prepared_identity) = unix::entry_identity(&live.update, PREPARED)? {
        let prepared = unix::open_directory_at(&live.update, PREPARED, Some(0o700), true)?;
        if prepared_identity.device != prepared.device()
            || prepared_identity.inode != prepared.inode()
            || !unix::list_names_bounded(&prepared, 2)?.is_empty()
        {
            return Err(
                crate::distribution::install_state::InstallStateError::InvalidLayout(
                    "prepared-version intent blocks metadata advancement",
                ),
            );
        }
    }

    let current = unix::entry_identity(&live.root, "current")?;
    let pending_current = unix::entry_identity(&live.root, super::super::PENDING_CURRENT)?;
    let versions_nonempty = optional_directory_nonempty(&live.root, VERSIONS, 1)?;
    let activations_nonempty = optional_directory_nonempty(&live.root, "activations", 2)?;
    if current.is_none() && (pending_current.is_some() || versions_nonempty || activations_nonempty)
    {
        return Err(
            crate::distribution::install_state::InstallStateError::InvalidLayout(
                "unactivated prepared version blocks metadata advancement",
            ),
        );
    }
    Ok(())
}

pub(in crate::distribution::install_state) fn stage_normalized_version(
    locked: &LockedInstallationIdentity,
    authorization: &PreparedVersionAuthorization,
    developer_id: crate::distribution::prepared_release::DeveloperIdVerification,
    retained: NormalizedExtractedReleaseTree,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
    proposed_installed_at: u64,
) -> Result<PendingPreparedVersion, PreparedVersionError> {
    require_release(locked, authorization, exact_manifest, manifest)?;
    require_no_activation_state(locked)?;
    let current_binding = super::modes::executable_binding(
        locked,
        &retained._extractions,
        &retained._stage,
        &retained.stage_name,
        exact_manifest,
        manifest,
        true,
    )?;
    if !developer_id.matches(current_binding) {
        return Err(PreparedVersionError::Integrity);
    }
    verify_normalized_release_tree(locked, &retained, exact_manifest, manifest)?;

    let live = locked.reopen()?;
    let prepared = unix::ensure_private_directory(&live.update, PREPARED)?;
    let versions = unix::ensure_private_directory(&live.root, VERSIONS)?;
    if classify_versions(Some(&versions), authorization)? {
        return Err(PreparedVersionError::Integrity);
    }
    let inventory = classify_prepared(Some(&prepared), authorization)?;
    if inventory.pending {
        return Err(PreparedVersionError::Integrity);
    }
    let installed_at = inventory.installed_at.unwrap_or(proposed_installed_at);
    if installed_at > proposed_installed_at {
        return Err(PreparedVersionError::Integrity);
    }
    let record = build_record(locked, authorization, installed_at)?;
    let ready_name = ensure_marker_intent(&prepared, authorization, &record, inventory)?;

    let live = locked.reopen()?;
    let extractions = unix::open_directory_at(&live.update, EXTRACTIONS, Some(0o700), true)?;
    if !extractions.same_object(&retained._extractions) {
        return Err(PreparedVersionError::Integrity);
    }
    let stage = unix::open_directory_at(&extractions, &retained.stage_name, Some(0o700), true)?;
    if !stage.same_object(&retained._stage) {
        return Err(PreparedVersionError::Integrity);
    }
    let live_prepared = unix::open_directory_at(&live.update, PREPARED, Some(0o700), true)?;
    if !live_prepared.same_object(&prepared) {
        return Err(PreparedVersionError::Integrity);
    }
    let live_versions = unix::open_directory_at(&live.root, VERSIONS, Some(0o700), true)?;
    if !live_versions.same_object(&versions)
        || !unix::list_names_bounded(&live_versions, 1)?.is_empty()
    {
        return Err(PreparedVersionError::Integrity);
    }
    let pending_name = pending_name(authorization);
    if unix::entry_identity(&live_prepared, &pending_name)?.is_some() {
        return Err(PreparedVersionError::Integrity);
    }

    unix::rename_noreplace(
        &extractions,
        &retained.stage_name,
        &live_prepared,
        &pending_name,
    )?;
    prepared_barrier()?;
    unix::sync_directory(&extractions)?;
    prepared_barrier()?;
    unix::sync_directory(&live_prepared)?;
    prepared_barrier()?;
    let pending = unix::open_directory_at(&live_prepared, &pending_name, Some(0o700), true)?;
    if !pending.same_object(&stage) {
        return Err(PreparedVersionError::Integrity);
    }
    unix::rename_noreplace(&live_prepared, &ready_name, &pending, MARKER_NAME)?;
    prepared_barrier()?;
    unix::sync_directory(&pending)?;
    prepared_barrier()?;
    unix::sync_directory(&live_prepared)?;
    prepared_barrier()?;
    verify_tree(&pending, exact_manifest, manifest, &record)?;
    prepared_barrier()?;
    locked.full_sync_endpoint()?;
    prepared_barrier()?;
    Ok(PendingPreparedVersion {
        prepared: live_prepared,
        versions: live_versions,
        tree: pending,
        name: pending_name,
        record,
    })
}

pub(in crate::distribution::install_state) fn recover_prepared_version(
    locked: &LockedInstallationIdentity,
    authorization: &PreparedVersionAuthorization,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
    recovery_reference: u64,
) -> Result<Option<PreparedVersionState>, PreparedVersionError> {
    require_release(locked, authorization, exact_manifest, manifest)?;
    require_no_activation_state(locked)?;
    let live = locked.reopen()?;
    let prepared = open_optional_directory(&live.update, PREPARED)?;
    let versions = open_optional_directory(&live.root, VERSIONS)?;
    let inventory = classify_prepared(prepared.as_ref(), authorization)?;
    let final_present = classify_versions(versions.as_ref(), authorization)?;
    if inventory.pending && final_present {
        return Err(PreparedVersionError::Integrity);
    }
    if final_present {
        if !matches!(
            inventory,
            PreparedInventory {
                installed_at: None,
                pending: false,
                ..
            }
        ) {
            return Err(PreparedVersionError::Integrity);
        }
        let versions = versions.ok_or(PreparedVersionError::Integrity)?;
        let tree = unix::open_directory_at(
            &versions,
            authorization.version().as_str(),
            Some(0o700),
            true,
        )?;
        let record = record_from_tree(&tree)?;
        if record.marker().installed_at_unix_seconds() > recovery_reference {
            return Err(PreparedVersionError::Integrity);
        }
        require_record(
            locked,
            authorization,
            &record,
            record.marker().installed_at_unix_seconds(),
        )?;
        verify_tree(&tree, exact_manifest, manifest, &record)?;
        return Ok(Some(PreparedVersionState::Published(
            PublishedPreparedVersion {
                versions,
                tree,
                record,
            },
        )));
    }
    if !inventory.pending {
        return Ok(None);
    }
    let prepared = prepared.ok_or(PreparedVersionError::Integrity)?;
    let pending =
        unix::open_directory_at(&prepared, &pending_name(authorization), Some(0o700), true)?;
    let record = if unix::entry_identity(&pending, MARKER_NAME)?.is_some() {
        if inventory.installed_at.is_some() {
            return Err(PreparedVersionError::Integrity);
        }
        record_from_tree(&pending)?
    } else {
        let installed_at = inventory
            .installed_at
            .ok_or(PreparedVersionError::Integrity)?;
        if installed_at > recovery_reference {
            return Err(PreparedVersionError::Integrity);
        }
        if inventory.kind != Some(IntentKind::Ready) {
            return Err(PreparedVersionError::Integrity);
        }
        let record = build_record(locked, authorization, installed_at)?;
        let ready = marker_name(authorization, installed_at, IntentKind::Ready);
        require_exact_file(&prepared, &ready, record.marker_bytes())?;
        unix::rename_noreplace(&prepared, &ready, &pending, MARKER_NAME)?;
        prepared_barrier()?;
        unix::sync_directory(&pending)?;
        prepared_barrier()?;
        unix::sync_directory(&prepared)?;
        prepared_barrier()?;
        record
    };
    if record.marker().installed_at_unix_seconds() > recovery_reference {
        return Err(PreparedVersionError::Integrity);
    }
    require_record(
        locked,
        authorization,
        &record,
        record.marker().installed_at_unix_seconds(),
    )?;
    verify_tree(&pending, exact_manifest, manifest, &record)?;
    prepared_barrier()?;
    locked.full_sync_endpoint()?;
    prepared_barrier()?;
    let versions = versions.unwrap_or(unix::ensure_private_directory(&live.root, VERSIONS)?);
    if !unix::list_names_bounded(&versions, 1)?.is_empty() {
        return Err(PreparedVersionError::Integrity);
    }
    Ok(Some(PreparedVersionState::Pending(
        PendingPreparedVersion {
            prepared,
            versions,
            tree: pending,
            name: pending_name(authorization),
            record,
        },
    )))
}

#[cfg(target_os = "macos")]
pub(in crate::distribution::install_state) fn with_prepared_executable<R, E>(
    locked: &LockedInstallationIdentity,
    state: &PreparedVersionState,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
    operation: impl FnOnce(&std::path::Path, &std::fs::File, ExecutableReleaseBinding) -> Result<R, E>,
) -> Result<R, E>
where
    E: From<PreparedVersionError>,
{
    let (parent, tree, expected_parent, expected_name) = match state {
        PreparedVersionState::Pending(pending) => (
            &pending.prepared,
            &pending.tree,
            PREPARED,
            pending.name.as_str(),
        ),
        PreparedVersionState::Published(published) => (
            &published.versions,
            &published.tree,
            VERSIONS,
            published.record.marker().release().version().as_str(),
        ),
    };
    with_tree_executable(
        locked,
        parent,
        tree,
        expected_parent,
        expected_name,
        exact_manifest,
        manifest,
        operation,
    )
}

pub(in crate::distribution::install_state) fn verify_prepared_version_tree(
    locked: &LockedInstallationIdentity,
    state: &PreparedVersionState,
    authorization: &PreparedVersionAuthorization,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
) -> Result<(), PreparedVersionError> {
    require_release(locked, authorization, exact_manifest, manifest)?;
    match state {
        PreparedVersionState::Pending(pending) => {
            verify_named_tree(
                locked,
                &pending.prepared,
                &pending.tree,
                PREPARED,
                &pending.name,
            )?;
            require_record(
                locked,
                authorization,
                &pending.record,
                pending.record.marker().installed_at_unix_seconds(),
            )?;
            verify_tree(&pending.tree, exact_manifest, manifest, &pending.record)
        }
        PreparedVersionState::Published(published) => {
            verify_named_tree(
                locked,
                &published.versions,
                &published.tree,
                VERSIONS,
                authorization.version().as_str(),
            )?;
            require_record(
                locked,
                authorization,
                &published.record,
                published.record.marker().installed_at_unix_seconds(),
            )?;
            verify_tree(&published.tree, exact_manifest, manifest, &published.record)
        }
    }
}

pub(in crate::distribution::install_state) fn publish_pending_version(
    locked: &LockedInstallationIdentity,
    authorization: &PreparedVersionAuthorization,
    pending: PendingPreparedVersion,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
    developer_id: crate::distribution::prepared_release::DeveloperIdVerification,
    precommit: impl FnOnce() -> Result<(), PreparedVersionError>,
) -> Result<PublishedPreparedVersion, PreparedVersionError> {
    require_release(locked, authorization, exact_manifest, manifest)?;
    let state = PreparedVersionState::Pending(pending);
    verify_prepared_version_tree(locked, &state, authorization, exact_manifest, manifest)?;
    let current = prepared_binding(locked, &state, exact_manifest, manifest)?;
    if !developer_id.matches(current) {
        return Err(PreparedVersionError::Integrity);
    }
    let PreparedVersionState::Pending(pending) = state else {
        unreachable!()
    };
    let live = locked.reopen()?;
    let prepared = unix::open_directory_at(&live.update, PREPARED, Some(0o700), true)?;
    let versions = unix::open_directory_at(&live.root, VERSIONS, Some(0o700), true)?;
    if !prepared.same_object(&pending.prepared)
        || !versions.same_object(&pending.versions)
        || unix::entry_identity(&versions, authorization.version().as_str())?.is_some()
    {
        return Err(PreparedVersionError::Integrity);
    }
    let tree = unix::open_directory_at(&prepared, &pending.name, Some(0o700), true)?;
    if !tree.same_object(&pending.tree) {
        return Err(PreparedVersionError::Integrity);
    }
    prepared_precommit_hook();
    let live = locked.reopen()?;
    let live_prepared = unix::open_directory_at(&live.update, PREPARED, Some(0o700), true)?;
    let live_versions = unix::open_directory_at(&live.root, VERSIONS, Some(0o700), true)?;
    let live_tree = unix::open_directory_at(&live_prepared, &pending.name, Some(0o700), true)?;
    if !live_prepared.same_object(&prepared)
        || !live_versions.same_object(&versions)
        || !live_tree.same_object(&tree)
        || unix::entry_identity(&live_versions, authorization.version().as_str())?.is_some()
    {
        return Err(PreparedVersionError::Integrity);
    }
    precommit()?;
    unix::rename_noreplace(
        &live_prepared,
        &pending.name,
        &live_versions,
        authorization.version().as_str(),
    )?;
    prepared_barrier().map_err(|error| error.after_commit(authorization.version().as_str()))?;
    let finalize = || -> Result<PublishedPreparedVersion, PreparedVersionError> {
        unix::sync_directory(&live_prepared)?;
        prepared_barrier()?;
        unix::sync_directory(&live_versions)?;
        prepared_barrier()?;
        let published = unix::open_directory_at(
            &live_versions,
            authorization.version().as_str(),
            Some(0o700),
            true,
        )?;
        if !published.same_object(&tree) {
            return Err(PreparedVersionError::Integrity);
        }
        prepared_barrier()?;
        Ok(PublishedPreparedVersion {
            versions: live_versions,
            tree: published,
            record: pending.record,
        })
    };
    finalize().map_err(|error| error.after_commit(authorization.version().as_str()))
}

pub(in crate::distribution::install_state) fn finish_published_version(
    locked: &LockedInstallationIdentity,
    authorization: &PreparedVersionAuthorization,
    published: &PublishedPreparedVersion,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
) -> Result<(), PreparedVersionError> {
    let state = PreparedVersionState::Published(PublishedPreparedVersion {
        versions: unix::duplicate_directory(&published.versions)?,
        tree: unix::duplicate_directory(&published.tree)?,
        record: FirstStandaloneInstallRecord::reconstruct_from_exact_marker(
            published.record.marker_bytes(),
        )
        .map_err(crate::distribution::install_state::InstallStateError::Receipt)?,
    });
    verify_prepared_version_tree(locked, &state, authorization, exact_manifest, manifest)?;
    sync_tree(&published.tree, exact_manifest, manifest, &published.record)?;
    unix::sync_directory(&published.versions)?;
    prepared_barrier()?;
    unix::sync_directory(locked.root())?;
    prepared_barrier()?;
    locked.full_sync_endpoint()?;
    prepared_barrier()?;
    Ok(())
}

pub(in crate::distribution::install_state) fn authenticate_published_version(
    locked: &LockedInstallationIdentity,
    authorization: &PreparedVersionAuthorization,
    state: PreparedVersionState,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
    developer_id: crate::distribution::prepared_release::DeveloperIdVerification,
) -> Result<VerifiedPublishedPreparedVersion, PreparedVersionError> {
    let PreparedVersionState::Published(published) = state else {
        return Err(PreparedVersionError::Integrity);
    };
    let state = PreparedVersionState::Published(published);
    verify_prepared_version_tree(locked, &state, authorization, exact_manifest, manifest)?;
    let current = prepared_binding(locked, &state, exact_manifest, manifest)?;
    if !developer_id.matches(current) {
        return Err(PreparedVersionError::Integrity);
    }
    let PreparedVersionState::Published(published) = state else {
        unreachable!()
    };
    finish_published_version(locked, authorization, &published, exact_manifest, manifest)?;
    Ok(VerifiedPublishedPreparedVersion {
        receipt_bytes: published.record.receipt_bytes().to_vec(),
    })
}

mod fault;
mod support;
mod types;

#[cfg(test)]
pub(in crate::distribution) use fault::{
    abort_after_prepared_barrier, fail_after_prepared_barrier, observed_prepared_barriers,
    reset_observed_prepared_barriers, run_prepared_crash_worker, set_prepared_precommit_hook,
};
use fault::{prepared_barrier, prepared_precommit_hook};
use support::{
    build_record, classify_prepared, classify_versions, ensure_marker_intent, marker_name,
    open_optional_directory, optional_directory_nonempty, pending_name, prepared_binding,
    record_from_tree, require_exact_file, require_identity, require_no_activation_state,
    require_record, require_release, sync_tree, verify_named_tree, verify_tree,
    with_tree_executable, IntentKind, PreparedInventory,
};
pub(in crate::distribution) use types::{
    PendingPreparedVersion, PreparedVersionError, PreparedVersionState, PublishedPreparedVersion,
    VerifiedPublishedPreparedVersion,
};
