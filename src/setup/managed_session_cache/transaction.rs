use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::num::NonZeroU64;

use sha2::{Digest, Sha256};

use super::catalog::{catalog_name, parse_catalog_name, sha256_hex, CatalogEntryV1, CatalogV1};
use super::unix::{self, Directory, EntryIdentity, StoreLock};
use super::{
    encode_object, EncodedObject, ManagedCacheBarrier, ManagedCheckpointKey, ManagedCommitOutcome,
    ManagedSessionCache, ManagedSessionCacheError, StoreDirectories, CATALOGS_DIR, CATALOG_PARTIAL,
    LOCK_NAME, MAX_CATALOG_BYTES, MAX_CHECKPOINT_BYTES, MAX_LIVE_ENTRIES, MAX_OBJECTS,
    MAX_OBJECT_BYTES, MAX_QUARANTINE_ENTRIES, MAX_RETAINED_CATALOGS, MIN_FREE_BYTES,
    MIN_MANAGED_CAPACITY_BYTES, OBJECTS_DIR, OBJECT_HEADER_BYTES, OBJECT_MAGIC, OBJECT_PARTIAL,
    OBJECT_VERSION, PENDING_DIR, QUARANTINE_DIR,
};
use crate::setup::fs::RuntimeConfigBinding;

const MAX_PENDING_NAMES: usize = 2;
const MAX_OBJECT_SHARDS: usize = 256;
const METADATA_BLOCK_MARGIN: u64 = 16;

#[cfg(test)]
thread_local! {
    static TEST_VOLUME_SPACE: std::cell::Cell<Option<(u64, u64, u64)>> = const {
        std::cell::Cell::new(None)
    };
}

pub(super) fn preflight_existing_layout(
    sessions: &Directory,
) -> Result<(), ManagedSessionCacheError> {
    let names = unix::list_names_bounded(sessions, 6)?;
    for name in names {
        match name.as_str() {
            LOCK_NAME => {
                let identity = unix::inspect_owned_regular(sessions, LOCK_NAME, 1)?;
                if identity.size() != 0 {
                    return Err(ManagedSessionCacheError::InvalidLayout(
                        "managed cache lock is not empty",
                    ));
                }
            }
            PENDING_DIR => {
                let pending = unix::open_directory(sessions, PENDING_DIR)?;
                verify_pending_inventory(&pending)?;
            }
            OBJECTS_DIR => {
                let objects = unix::open_directory(sessions, OBJECTS_DIR)?;
                inventory_objects(&objects)?;
            }
            CATALOGS_DIR => {
                let catalogs = unix::open_directory(sessions, CATALOGS_DIR)?;
                preflight_catalog_inventory(&catalogs)?;
            }
            QUARANTINE_DIR => {
                let quarantine = unix::open_directory(sessions, QUARANTINE_DIR)?;
                verify_quarantine(&quarantine)?;
            }
            _ => {
                return Err(ManagedSessionCacheError::InvalidLayout(
                    "sessions directory contains an unknown managed entry",
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn with_test_volume_space<T>(
    volume: u64,
    available: u64,
    fragment: u64,
    action: impl FnOnce() -> T,
) -> T {
    TEST_VOLUME_SPACE.with(|slot| {
        let previous = slot.replace(Some((volume, available, fragment)));
        let result = action();
        slot.set(previous);
        result
    })
}

pub(super) fn recover(
    binding: &RuntimeConfigBinding,
    directories: &StoreDirectories,
    sessions: &Directory,
    lock: &StoreLock,
    limit_bytes: NonZeroU64,
) -> Result<(CatalogV1, Vec<String>), ManagedSessionCacheError> {
    let revalidate = || revalidate_recovery(binding, sessions, directories, lock);
    revalidate()?;
    clean_pending(&directories.pending, &revalidate)?;
    revalidate()?;
    let (catalog, names) = load_highest_catalog(&directories.catalogs)?;
    let objects = inventory_objects(&directories.objects)?;
    validate_catalog_object_shapes(&catalog, &objects, limit_bytes)?;
    clean_orphans_with_parent(&directories.objects, &catalog, &objects, &revalidate)?;
    revalidate()?;
    if inventory_charge(sessions, directories)? > limit_bytes.get() {
        return Err(ManagedSessionCacheError::QuotaExceeded);
    }
    verify_catalog_objects(&directories.objects, &catalog, &objects, limit_bytes.get())?;
    verify_quarantine(&directories.quarantine)?;
    let names = prune_old_catalogs(&directories.catalogs, names, &revalidate)?;
    revalidate()?;
    unix::sync_directory(&directories.pending)?;
    unix::sync_directory(&directories.objects)?;
    unix::sync_directory(&directories.catalogs)?;
    unix::sync_directory(&directories.quarantine)?;
    unix::full_sync_lock(lock)?;
    revalidate()?;
    Ok((catalog, names))
}

pub(super) fn require_structural_capacity(
    sessions: &Directory,
    limit_bytes: NonZeroU64,
) -> Result<(), ManagedSessionCacheError> {
    enforce_free_floor(sessions, 0, 0)?;
    let stat = unix::volume_space(sessions)?;
    let fragment = stat.f_frsize.max(1);
    let conservative = fragment
        .checked_mul(32)
        .ok_or(ManagedSessionCacheError::QuotaExceeded)?
        .max(MIN_MANAGED_CAPACITY_BYTES);
    if limit_bytes.get() < conservative {
        return Err(ManagedSessionCacheError::QuotaExceeded);
    }
    Ok(())
}

pub(super) fn commit(
    store: &ManagedSessionCache,
    key: ManagedCheckpointKey,
    payload: &[u8],
) -> Result<ManagedCommitOutcome, ManagedSessionCacheError> {
    commit_with_hook(store, key, payload, |_barrier| {
        #[cfg(test)]
        super::tests::abort_at_managed_cache_barrier(_barrier);
        Ok(())
    })
}

#[cfg(test)]
pub(super) fn commit_with_test_hook(
    store: &ManagedSessionCache,
    key: ManagedCheckpointKey,
    payload: &[u8],
    hook: impl FnMut(ManagedCacheBarrier) -> Result<(), ManagedSessionCacheError>,
) -> Result<ManagedCommitOutcome, ManagedSessionCacheError> {
    commit_with_hook(store, key, payload, hook)
}

fn commit_with_hook(
    store: &ManagedSessionCache,
    key: ManagedCheckpointKey,
    payload: &[u8],
    mut hook: impl FnMut(ManagedCacheBarrier) -> Result<(), ManagedSessionCacheError>,
) -> Result<ManagedCommitOutcome, ManagedSessionCacheError> {
    let object = encode_object(key, payload)?;
    let object_bytes = object.total_bytes;
    let object_digest = object.digest.clone();
    let key_hex = key.hex();
    let mut state = store.state.lock().map_err(|_| {
        ManagedSessionCacheError::Filesystem("managed cache mutex poisoned".to_owned())
    })?;
    if state.needs_recovery {
        return Err(ManagedSessionCacheError::RecoveryRequired);
    }
    let result = (|| {
        reconcile_uncommitted(store, &state)?;
        store.revalidate_endpoint()?;
        hook(ManagedCacheBarrier::AuthorizationValidated)?;
        let replaced_digest = state
            .catalog
            .find(&key_hex)
            .map(|entry| entry.object_sha256.clone());
        if let Some(existing) = state.catalog.find(&key_hex) {
            if existing.object_sha256 == object_digest && existing.object_bytes == object_bytes {
                let restored = read_object(
                    &store.directories.objects,
                    key,
                    &existing.object_sha256,
                    existing.object_bytes,
                    store.limit_bytes.get(),
                )?;
                if restored == payload {
                    store.revalidate_endpoint()?;
                    return Ok(ManagedCommitOutcome::AlreadyPresent);
                }
            }
        }

        let (proposed, proposed_bytes) = make_room(
            store,
            &mut state,
            key_hex.clone(),
            object_digest.clone(),
            object_bytes,
            &mut hook,
        )?;
        enforce_free_floor(
            &store.directories.pending,
            object_bytes,
            proposed_bytes.len(),
        )?;

        let shard_name = &object_digest[..2];
        let shard = unix::ensure_directory(&store.directories.objects, shard_name)?;
        unix::verify_directory(&store.directories.objects, shard_name, &shard)?;
        let current_charge = inventory_charge(&store.sessions, &store.directories)?;
        let stat = unix::volume_space(&store.directories.pending)?;
        let block = stat.f_frsize.max(1);
        let planned = round_up(object_bytes, block)?
            .checked_add(round_up(proposed_bytes.len() as u64, block)?)
            .and_then(|value| value.checked_add(block.saturating_mul(METADATA_BLOCK_MARGIN)))
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
        if current_charge
            .checked_add(planned)
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?
            > store.limit_bytes.get()
        {
            return Err(ManagedSessionCacheError::QuotaExceeded);
        }

        let partial_identity =
            write_object_partial(&store.directories.pending, OBJECT_PARTIAL, &object)?;
        hook(ManagedCacheBarrier::ObjectPartialSynced)?;
        store.revalidate_endpoint()?;
        unix::verify_named(&store.directories.pending, OBJECT_PARTIAL, partial_identity)?;
        hook(ManagedCacheBarrier::BeforeObjectPublish)?;
        store.revalidate_endpoint()?;
        unix::verify_named(&store.directories.pending, OBJECT_PARTIAL, partial_identity)?;
        let final_name = format!("{object_digest}.checkpoint");
        let created = unix::rename_noreplace(
            &store.directories.pending,
            OBJECT_PARTIAL,
            &shard,
            &final_name,
        )?;
        if created {
            unix::verify_named(&shard, &final_name, partial_identity)?;
        }
        hook(ManagedCacheBarrier::ObjectPublished)?;
        if created {
            unix::sync_directory(&shard)?;
            unix::sync_directory(&store.directories.pending)?;
        } else {
            remove_exact_if_present(&store.directories.pending, OBJECT_PARTIAL, partial_identity)?;
        }
        unix::verify_directory(&store.directories.objects, shard_name, &shard)?;
        if created {
            unix::verify_named(&shard, &final_name, partial_identity)?;
        }
        let final_identity = if created {
            partial_identity
        } else {
            let identity = unix::inspect_owned_regular(&shard, &final_name, 1)?;
            if identity.size() != object_bytes {
                return Err(ManagedSessionCacheError::InvalidLayout(
                    "content-addressed managed object has the wrong length",
                ));
            }
            identity
        };
        unix::full_sync_named(&shard, &final_name, final_identity)?;
        hook(ManagedCacheBarrier::ObjectDirectorySynced)?;
        let adopted = read_object(
            &store.directories.objects,
            key,
            &object_digest,
            object_bytes,
            store.limit_bytes.get(),
        )?;
        if adopted != payload {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "content-addressed managed object did not match its digest",
            ));
        }
        store.revalidate_endpoint()?;
        enforce_free_floor(&store.directories.pending, 0, proposed_bytes.len())?;
        if inventory_charge(&store.sessions, &store.directories)? > store.limit_bytes.get() {
            if !state
                .catalog
                .entries()
                .iter()
                .any(|entry| entry.object_sha256 == object_digest)
            {
                let identity = unix::inspect_owned_regular(&shard, &final_name, 1)?;
                unix::remove_file(&shard, &final_name, identity)?;
                unix::sync_directory(&shard)?;
            }
            return Err(ManagedSessionCacheError::QuotaExceeded);
        }

        reserve_catalog_transaction(store, &mut state, proposed_bytes.len())?;
        let replaced_identity = replaced_digest
            .as_deref()
            .filter(|digest| *digest != object_digest)
            .filter(|digest| {
                !proposed
                    .entries()
                    .iter()
                    .any(|entry| entry.object_sha256 == **digest)
            })
            .map(|digest| inspect_object_identity(&store.directories.objects, digest))
            .transpose()?;
        let published_name = publish_catalog(store, &proposed, &proposed_bytes, &mut hook)?;
        state.catalog = proposed;
        state.retained_catalog_names.push(published_name);
        if let Some((digest, identity)) = replaced_identity {
            store.revalidate_endpoint().map_err(committed_unknown)?;
            remove_object_expected(&store.directories.objects, &digest, identity)
                .map_err(committed_unknown)?;
        }
        store.revalidate_endpoint().map_err(committed_unknown)?;
        unix::full_sync_lock(&store.lock).map_err(committed_unknown)?;
        hook(ManagedCacheBarrier::EndpointSynced).map_err(|error| {
            ManagedSessionCacheError::CommittedDurabilityUnknown(error.to_string())
        })?;
        if inventory_charge(&store.sessions, &store.directories).map_err(committed_unknown)?
            > store.limit_bytes.get()
        {
            return Err(ManagedSessionCacheError::CommittedDurabilityUnknown(
                "committed managed cache exceeds its aggregate cap".to_owned(),
            ));
        }
        Ok(ManagedCommitOutcome::Published)
    })();
    if matches!(
        result,
        Err(ManagedSessionCacheError::CommittedDurabilityUnknown(_))
    ) {
        state.needs_recovery = true;
    }
    result
}

pub(super) fn restore(
    store: &ManagedSessionCache,
    key: ManagedCheckpointKey,
) -> Result<Option<Vec<u8>>, ManagedSessionCacheError> {
    let state = store.state.lock().map_err(|_| {
        ManagedSessionCacheError::Filesystem("managed cache mutex poisoned".to_owned())
    })?;
    if state.needs_recovery {
        return Err(ManagedSessionCacheError::RecoveryRequired);
    }
    store.revalidate_endpoint()?;
    let Some(entry) = state.catalog.find(&key.hex()) else {
        return Ok(None);
    };
    let payload = read_object(
        &store.directories.objects,
        key,
        &entry.object_sha256,
        entry.object_bytes,
        store.limit_bytes.get(),
    )?;
    store.revalidate_endpoint()?;
    Ok(Some(payload))
}

fn reconcile_uncommitted(
    store: &ManagedSessionCache,
    state: &super::StoreState,
) -> Result<(), ManagedSessionCacheError> {
    let revalidate = || store.revalidate_endpoint();
    revalidate()?;
    clean_pending(&store.directories.pending, &revalidate)?;
    let objects = inventory_objects(&store.directories.objects)?;
    validate_catalog_object_shapes(&state.catalog, &objects, store.limit_bytes)?;
    clean_orphans_with_parent(
        &store.directories.objects,
        &state.catalog,
        &objects,
        &revalidate,
    )?;
    revalidate()?;
    if inventory_charge(&store.sessions, &store.directories)? > store.limit_bytes.get() {
        return Err(ManagedSessionCacheError::QuotaExceeded);
    }
    Ok(())
}

fn make_room(
    store: &ManagedSessionCache,
    state: &mut super::StoreState,
    key_hex: String,
    object_digest: String,
    object_bytes: u64,
    hook: &mut impl FnMut(ManagedCacheBarrier) -> Result<(), ManagedSessionCacheError>,
) -> Result<(CatalogV1, Vec<u8>), ManagedSessionCacheError> {
    loop {
        if state.catalog.find(&key_hex).is_none()
            && state.catalog.entries().len() >= MAX_LIVE_ENTRIES
        {
            evict_one(store, state, hook)?;
            continue;
        }
        let proposed = state.catalog.with_entry(CatalogEntryV1::new(
            key_hex.clone(),
            object_digest.clone(),
            object_bytes,
        ))?;
        let proposed_bytes = proposed.to_canonical_bytes()?;
        let charge = inventory_charge(&store.sessions, &store.directories)?;
        let stat = unix::volume_space(&store.directories.pending)?;
        let block = stat.f_frsize.max(1);
        let required = round_up(object_bytes, block)?
            .checked_add(round_up(proposed_bytes.len() as u64, block)?)
            .and_then(|value| value.checked_add(block.saturating_mul(METADATA_BLOCK_MARGIN)))
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
        if charge
            .checked_add(required)
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?
            <= store.limit_bytes.get()
        {
            return Ok((proposed, proposed_bytes));
        }
        evict_one(store, state, hook)?;
    }
}

fn evict_one(
    store: &ManagedSessionCache,
    state: &mut super::StoreState,
    hook: &mut impl FnMut(ManagedCacheBarrier) -> Result<(), ManagedSessionCacheError>,
) -> Result<(), ManagedSessionCacheError> {
    let victim = state
        .catalog
        .entries()
        .iter()
        .min_by_key(|entry| (entry.last_committed_generation, entry.key_sha256.as_str()))
        .cloned()
        .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    let victim_identity =
        inspect_object_identity(&store.directories.objects, &victim.object_sha256)?;
    let removed = BTreeSet::from([victim.key_sha256.clone()]);
    let pruned = state.catalog.without_keys(&removed)?;
    let bytes = pruned.to_canonical_bytes()?;
    enforce_free_floor(&store.directories.pending, 0, bytes.len())?;
    reserve_catalog_transaction(store, state, bytes.len())?;
    let name = publish_catalog(store, &pruned, &bytes, hook)?;
    state.catalog = pruned;
    state.retained_catalog_names.push(name);
    if !state
        .catalog
        .entries()
        .iter()
        .any(|entry| entry.object_sha256 == victim.object_sha256)
    {
        store.revalidate_endpoint().map_err(committed_unknown)?;
        remove_object_expected(
            &store.directories.objects,
            &victim.object_sha256,
            victim_identity.1,
        )
        .map_err(committed_unknown)?;
    }
    Ok(())
}

fn reserve_catalog_transaction(
    store: &ManagedSessionCache,
    state: &mut super::StoreState,
    bytes: usize,
) -> Result<(), ManagedSessionCacheError> {
    while state.retained_catalog_names.len() >= MAX_RETAINED_CATALOGS {
        prune_one_catalog_history(store, state)?;
    }
    loop {
        let charge = inventory_charge(&store.sessions, &store.directories)?;
        let stat = unix::volume_space(&store.directories.pending)?;
        let block = stat.f_frsize.max(1);
        let planned = round_up(bytes as u64, block)?
            .checked_add(block.saturating_mul(METADATA_BLOCK_MARGIN))
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
        if charge
            .checked_add(planned)
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?
            <= store.limit_bytes.get()
        {
            return Ok(());
        }
        if state.retained_catalog_names.len() <= 1 {
            return Err(ManagedSessionCacheError::QuotaExceeded);
        }
        prune_one_catalog_history(store, state)?;
    }
}

fn prune_one_catalog_history(
    store: &ManagedSessionCache,
    state: &mut super::StoreState,
) -> Result<(), ManagedSessionCacheError> {
    if state.retained_catalog_names.len() <= 1 {
        return Err(ManagedSessionCacheError::QuotaExceeded);
    }
    state.retained_catalog_names.sort_by_key(|name| {
        parse_catalog_name(name)
            .map(|entry| entry.0)
            .unwrap_or(u64::MAX)
    });
    let stale = state.retained_catalog_names[0].clone();
    store.revalidate_endpoint()?;
    let identity = unix::inspect_owned_regular(&store.directories.catalogs, &stale, 1)?;
    unix::remove_file(&store.directories.catalogs, &stale, identity)?;
    unix::sync_directory(&store.directories.catalogs)?;
    store.revalidate_endpoint()?;
    state.retained_catalog_names.remove(0);
    Ok(())
}

fn publish_catalog(
    store: &ManagedSessionCache,
    catalog: &CatalogV1,
    bytes: &[u8],
    hook: &mut impl FnMut(ManagedCacheBarrier) -> Result<(), ManagedSessionCacheError>,
) -> Result<String, ManagedSessionCacheError> {
    let directories = &store.directories;
    let digest = sha256_hex(bytes);
    let final_name = catalog_name(catalog.generation(), &digest);
    let partial_identity = write_partial(&directories.pending, CATALOG_PARTIAL, bytes)?;
    hook(ManagedCacheBarrier::CatalogPartialSynced)?;
    store.revalidate_endpoint()?;
    unix::verify_named(&directories.pending, CATALOG_PARTIAL, partial_identity)?;
    hook(ManagedCacheBarrier::BeforeCatalogPublish)?;
    store.revalidate_endpoint()?;
    unix::verify_named(&directories.pending, CATALOG_PARTIAL, partial_identity)?;
    let created = unix::rename_noreplace(
        &directories.pending,
        CATALOG_PARTIAL,
        &directories.catalogs,
        &final_name,
    )?;
    let final_identity = if created {
        unix::verify_named(&directories.catalogs, &final_name, partial_identity)
            .map_err(committed_unknown)?;
        partial_identity
    } else {
        let identity = unix::inspect_owned_regular(&directories.catalogs, &final_name, 1)
            .map_err(committed_unknown)?;
        if identity.size() != bytes.len() as u64 {
            return Err(ManagedSessionCacheError::CommittedDurabilityUnknown(
                "immutable managed catalog has the wrong length".to_owned(),
            ));
        }
        identity
    };
    hook(ManagedCacheBarrier::CatalogPublished)
        .map_err(|error| ManagedSessionCacheError::CommittedDurabilityUnknown(error.to_string()))?;
    if !created {
        unix::full_sync_named(&directories.catalogs, &final_name, final_identity)
            .map_err(committed_unknown)?;
        let existing = read_exact_owned(&directories.catalogs, &final_name, bytes.len())
            .map_err(committed_unknown)?;
        if existing != bytes {
            return Err(ManagedSessionCacheError::CommittedDurabilityUnknown(
                "immutable managed catalog name was reused for different bytes".to_owned(),
            ));
        }
        remove_exact_if_present(&directories.pending, CATALOG_PARTIAL, partial_identity)
            .map_err(committed_unknown)?;
    }
    unix::sync_directory(&directories.catalogs)
        .map_err(|error| ManagedSessionCacheError::CommittedDurabilityUnknown(error.to_string()))?;
    hook(ManagedCacheBarrier::CatalogDirectorySynced)
        .map_err(|error| ManagedSessionCacheError::CommittedDurabilityUnknown(error.to_string()))?;
    unix::sync_directory(&directories.pending)
        .map_err(|error| ManagedSessionCacheError::CommittedDurabilityUnknown(error.to_string()))?;
    unix::full_sync_named(&directories.catalogs, &final_name, final_identity)
        .map_err(committed_unknown)?;
    let existing = read_exact_owned(&directories.catalogs, &final_name, bytes.len())
        .map_err(|error| ManagedSessionCacheError::CommittedDurabilityUnknown(error.to_string()))?;
    if existing != bytes {
        return Err(ManagedSessionCacheError::CommittedDurabilityUnknown(
            "published catalog bytes changed".to_owned(),
        ));
    }
    Ok(final_name)
}

fn write_partial(
    pending: &Directory,
    name: &str,
    bytes: &[u8],
) -> Result<EntryIdentity, ManagedSessionCacheError> {
    remove_if_present(pending, name)?;
    let (mut file, identity) = unix::create_private(pending, name)?;
    file.write_all(bytes).map_err(|error| {
        if error.raw_os_error() == Some(libc::ENOSPC) {
            ManagedSessionCacheError::StorageFull
        } else {
            ManagedSessionCacheError::Filesystem(format!("write managed cache partial: {error}"))
        }
    })?;
    file.flush()
        .map_err(|error| ManagedSessionCacheError::Filesystem(error.to_string()))?;
    unix::full_sync(&file)?;
    let actual = unix::identity_after_io(&file, pending, identity)?;
    if actual.size() != bytes.len() as u64 {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache partial has the wrong exact length",
        ));
    }
    unix::verify_named(pending, name, actual)?;
    Ok(actual)
}

fn write_object_partial(
    pending: &Directory,
    name: &str,
    object: &EncodedObject<'_>,
) -> Result<EntryIdentity, ManagedSessionCacheError> {
    remove_if_present(pending, name)?;
    let (mut file, identity) = unix::create_private(pending, name)?;
    for bytes in [object.header.as_slice(), object.payload] {
        file.write_all(bytes).map_err(|error| {
            if error.raw_os_error() == Some(libc::ENOSPC) {
                ManagedSessionCacheError::StorageFull
            } else {
                ManagedSessionCacheError::Filesystem(format!(
                    "write managed checkpoint partial: {error}"
                ))
            }
        })?;
    }
    file.flush()
        .map_err(|error| ManagedSessionCacheError::Filesystem(error.to_string()))?;
    unix::full_sync(&file)?;
    let actual = unix::identity_after_io(&file, pending, identity)?;
    if actual.size() != object.total_bytes {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed checkpoint partial has the wrong exact length",
        ));
    }
    unix::verify_named(pending, name, actual)?;
    Ok(actual)
}

fn read_object(
    objects: &Directory,
    key: ManagedCheckpointKey,
    digest: &str,
    expected_bytes: u64,
    maximum_bytes: u64,
) -> Result<Vec<u8>, ManagedSessionCacheError> {
    let shard = unix::open_directory(objects, &digest[..2])?;
    let name = format!("{digest}.checkpoint");
    read_and_verify_object(
        &shard,
        &name,
        key,
        digest,
        expected_bytes,
        maximum_bytes,
        true,
    )?
    .ok_or(ManagedSessionCacheError::InvalidLayout(
        "managed checkpoint payload was not retained",
    ))
}

fn verify_object(
    objects: &Directory,
    key: ManagedCheckpointKey,
    digest: &str,
    expected_bytes: u64,
    maximum_bytes: u64,
) -> Result<(), ManagedSessionCacheError> {
    let shard = unix::open_directory(objects, &digest[..2])?;
    let name = format!("{digest}.checkpoint");
    read_and_verify_object(
        &shard,
        &name,
        key,
        digest,
        expected_bytes,
        maximum_bytes,
        false,
    )?;
    Ok(())
}

fn read_and_verify_object(
    shard: &Directory,
    name: &str,
    key: ManagedCheckpointKey,
    digest: &str,
    expected_bytes: u64,
    maximum_bytes: u64,
    retain_payload: bool,
) -> Result<Option<Vec<u8>>, ManagedSessionCacheError> {
    if expected_bytes < OBJECT_HEADER_BYTES as u64
        || expected_bytes > MAX_OBJECT_BYTES
        || expected_bytes > maximum_bytes
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed object exceeds its authorized exact bound",
        ));
    }
    let (mut file, identity) = unix::open_private(shard, name, false)?;
    if identity.size() != expected_bytes {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed object length mismatched",
        ));
    }
    let mut header = [0u8; OBJECT_HEADER_BYTES];
    file.read_exact(&mut header)
        .map_err(|error| ManagedSessionCacheError::Filesystem(error.to_string()))?;
    if &header[..8] != OBJECT_MAGIC
        || u32::from_le_bytes(header[8..12].try_into().unwrap()) != OBJECT_VERSION
        || header[12..44] != key.0
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed checkpoint compatibility binding mismatched",
        ));
    }
    let payload_len = u64::from_le_bytes(header[44..52].try_into().unwrap());
    if payload_len == 0
        || payload_len > MAX_CHECKPOINT_BYTES
        || payload_len.checked_add(OBJECT_HEADER_BYTES as u64) != Some(expected_bytes)
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed checkpoint declares an invalid length",
        ));
    }
    let mut object_hasher = Sha256::new();
    object_hasher.update(header);
    let mut payload_hasher = Sha256::new();
    let mut payload = if retain_payload {
        let capacity = usize::try_from(payload_len).map_err(|_| {
            ManagedSessionCacheError::InvalidLayout("managed checkpoint is not addressable")
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(capacity).map_err(|_| {
            ManagedSessionCacheError::Filesystem(
                "managed checkpoint payload allocation failed".to_owned(),
            )
        })?;
        Some(bytes)
    } else {
        None
    };
    let mut remaining = payload_len;
    let mut buffer = [0u8; 64 * 1024];
    while remaining != 0 {
        let chunk = usize::try_from(remaining.min(buffer.len() as u64)).unwrap();
        file.read_exact(&mut buffer[..chunk])
            .map_err(|error| ManagedSessionCacheError::Filesystem(error.to_string()))?;
        object_hasher.update(&buffer[..chunk]);
        payload_hasher.update(&buffer[..chunk]);
        if let Some(payload) = &mut payload {
            payload.extend_from_slice(&buffer[..chunk]);
        }
        remaining -= chunk as u64;
    }
    let mut extra = [0u8; 1];
    if file
        .read(&mut extra)
        .map_err(|error| ManagedSessionCacheError::Filesystem(error.to_string()))?
        != 0
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed checkpoint has trailing bytes",
        ));
    }
    if hex::encode(object_hasher.finalize()) != digest
        || payload_hasher.finalize().as_slice() != &header[52..84]
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed checkpoint digest mismatched",
        ));
    }
    let after = unix::identity_after_io(&file, shard, identity)?;
    if after != identity {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed checkpoint changed while reading",
        ));
    }
    unix::verify_named(shard, name, identity)?;
    Ok(payload)
}

fn read_exact_owned(
    parent: &Directory,
    name: &str,
    expected: usize,
) -> Result<Vec<u8>, ManagedSessionCacheError> {
    let (mut file, identity) = unix::open_private(parent, name, false)?;
    if identity.size() != expected as u64 {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache file length mismatched",
        ));
    }
    let mut bytes = vec![0; expected];
    file.read_exact(&mut bytes)
        .map_err(|error| ManagedSessionCacheError::Filesystem(error.to_string()))?;
    let mut extra = [0u8; 1];
    if file
        .read(&mut extra)
        .map_err(|error| ManagedSessionCacheError::Filesystem(error.to_string()))?
        != 0
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache file has trailing bytes",
        ));
    }
    let after = unix::identity_after_io(&file, parent, identity)?;
    if after != identity {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache file changed while reading",
        ));
    }
    unix::verify_named(parent, name, identity)?;
    Ok(bytes)
}

fn load_highest_catalog(
    catalogs: &Directory,
) -> Result<(CatalogV1, Vec<String>), ManagedSessionCacheError> {
    let names = unix::list_names_bounded(catalogs, MAX_RETAINED_CATALOGS + 2)?;
    if names.is_empty() {
        return Ok((CatalogV1::empty(), Vec::new()));
    }
    let mut generations = BTreeMap::new();
    let mut parsed = Vec::new();
    for name in names {
        let (generation, digest) = parse_catalog_name(&name).ok_or(
            ManagedSessionCacheError::InvalidLayout("managed catalog name is invalid"),
        )?;
        if generations.insert(generation, digest.to_owned()).is_some() {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalogs contain an ambiguous generation",
            ));
        }
        let identity = unix::inspect_owned_regular(catalogs, &name, 1)?;
        if identity.size() == 0 || identity.size() > MAX_CATALOG_BYTES as u64 {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog final exceeds its byte cap",
            ));
        }
        let bytes = read_exact_owned(catalogs, &name, identity.size() as usize)?;
        if sha256_hex(&bytes) != digest {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog filename digest mismatched",
            ));
        }
        let catalog = CatalogV1::parse_exact(&bytes)?;
        if catalog.generation() != generation {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog filename generation mismatched",
            ));
        }
        parsed.push((generation, name, catalog));
    }
    parsed.sort_by_key(|entry| entry.0);
    let selected = parsed
        .last()
        .ok_or(ManagedSessionCacheError::InvalidLayout(
            "managed catalog set is empty",
        ))?
        .2
        .clone();
    Ok((selected, parsed.into_iter().map(|entry| entry.1).collect()))
}

struct ObjectRecord {
    shard: String,
    name: String,
    identity: EntryIdentity,
}

fn inventory_objects(
    objects: &Directory,
) -> Result<BTreeMap<String, ObjectRecord>, ManagedSessionCacheError> {
    let shards = unix::list_names_bounded(objects, MAX_OBJECT_SHARDS)?;
    let mut found = BTreeMap::new();
    for shard_name in shards {
        if shard_name.len() != 2 || !is_lower_hex(&shard_name) {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed object shard name is invalid",
            ));
        }
        let shard = unix::open_directory(objects, &shard_name)?;
        let names = unix::list_names_bounded(&shard, MAX_OBJECTS + 1)?;
        for name in names {
            let digest =
                name.strip_suffix(".checkpoint")
                    .ok_or(ManagedSessionCacheError::InvalidLayout(
                        "managed object name is invalid",
                    ))?;
            if digest.len() != 64 || !is_lower_hex(digest) || &digest[..2] != shard_name {
                return Err(ManagedSessionCacheError::InvalidLayout(
                    "managed object name does not match its shard",
                ));
            }
            if found.len() >= MAX_OBJECTS {
                return Err(ManagedSessionCacheError::InvalidLayout(
                    "managed object inventory exceeds its cap",
                ));
            }
            let identity = unix::inspect_owned_regular(&shard, &name, 1)?;
            found.insert(
                digest.to_owned(),
                ObjectRecord {
                    shard: shard_name.clone(),
                    name,
                    identity,
                },
            );
        }
    }
    Ok(found)
}

fn validate_catalog_object_shapes(
    catalog: &CatalogV1,
    objects: &BTreeMap<String, ObjectRecord>,
    limit_bytes: NonZeroU64,
) -> Result<(), ManagedSessionCacheError> {
    for entry in catalog.entries() {
        if entry.object_bytes > limit_bytes.get() {
            return Err(ManagedSessionCacheError::QuotaExceeded);
        }
        let object =
            objects
                .get(&entry.object_sha256)
                .ok_or(ManagedSessionCacheError::InvalidLayout(
                    "managed catalog references a missing object",
                ))?;
        if object.identity.size() != entry.object_bytes {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog object length mismatched",
            ));
        }
    }
    Ok(())
}

fn verify_catalog_objects(
    objects_dir: &Directory,
    catalog: &CatalogV1,
    objects: &BTreeMap<String, ObjectRecord>,
    maximum_bytes: u64,
) -> Result<(), ManagedSessionCacheError> {
    for entry in catalog.entries() {
        let object =
            objects
                .get(&entry.object_sha256)
                .ok_or(ManagedSessionCacheError::InvalidLayout(
                    "managed catalog references a missing object",
                ))?;
        if object.identity.size() != entry.object_bytes {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog object length mismatched",
            ));
        }
        let key = ManagedCheckpointKey::from_hex(&entry.key_sha256)?;
        verify_object(
            objects_dir,
            key,
            &entry.object_sha256,
            entry.object_bytes,
            maximum_bytes,
        )?;
    }
    Ok(())
}

fn committed_unknown(error: ManagedSessionCacheError) -> ManagedSessionCacheError {
    match error {
        ManagedSessionCacheError::CommittedDurabilityUnknown(_) => error,
        other => ManagedSessionCacheError::CommittedDurabilityUnknown(other.to_string()),
    }
}

fn clean_orphans_with_parent(
    objects_dir: &Directory,
    catalog: &CatalogV1,
    objects: &BTreeMap<String, ObjectRecord>,
    revalidate: &impl Fn() -> Result<(), ManagedSessionCacheError>,
) -> Result<(), ManagedSessionCacheError> {
    let referenced: BTreeSet<_> = catalog
        .entries()
        .iter()
        .map(|entry| entry.object_sha256.as_str())
        .collect();
    for (digest, object) in objects {
        if !referenced.contains(digest.as_str()) {
            revalidate()?;
            let shard = unix::open_directory(objects_dir, &object.shard)?;
            unix::remove_file(&shard, &object.name, object.identity)?;
            unix::sync_directory(&shard)?;
            revalidate()?;
        }
    }
    Ok(())
}

fn clean_pending(
    pending: &Directory,
    revalidate: &impl Fn() -> Result<(), ManagedSessionCacheError>,
) -> Result<(), ManagedSessionCacheError> {
    let names = unix::list_names_bounded(pending, MAX_PENDING_NAMES + 1)?;
    for name in names {
        if name != OBJECT_PARTIAL && name != CATALOG_PARTIAL {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed pending directory contains an unknown name",
            ));
        }
        revalidate()?;
        let identity = unix::inspect_owned_regular(pending, &name, 1)?;
        unix::remove_file(pending, &name, identity)?;
        revalidate()?;
    }
    unix::sync_directory(pending)
}

fn verify_pending_inventory(pending: &Directory) -> Result<(), ManagedSessionCacheError> {
    for name in unix::list_names_bounded(pending, MAX_PENDING_NAMES + 1)? {
        if name != OBJECT_PARTIAL && name != CATALOG_PARTIAL {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed pending directory contains an unknown name",
            ));
        }
        unix::inspect_owned_regular(pending, &name, 1)?;
    }
    Ok(())
}

fn preflight_catalog_inventory(catalogs: &Directory) -> Result<(), ManagedSessionCacheError> {
    let mut generations = BTreeSet::new();
    for name in unix::list_names_bounded(catalogs, MAX_RETAINED_CATALOGS + 2)? {
        let (generation, _) = parse_catalog_name(&name).ok_or(
            ManagedSessionCacheError::InvalidLayout("managed catalog name is invalid"),
        )?;
        if !generations.insert(generation) {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalogs contain an ambiguous generation",
            ));
        }
        let identity = unix::inspect_owned_regular(catalogs, &name, 1)?;
        if identity.size() == 0 || identity.size() > MAX_CATALOG_BYTES as u64 {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog final exceeds its byte cap",
            ));
        }
    }
    Ok(())
}

fn verify_quarantine(quarantine: &Directory) -> Result<(), ManagedSessionCacheError> {
    let names = unix::list_names_bounded(quarantine, MAX_QUARANTINE_ENTRIES + 1)?;
    for name in names {
        if !valid_quarantine_name(&name) {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed quarantine contains an unknown name",
            ));
        }
        unix::inspect_owned_regular(quarantine, &name, 1)?;
    }
    Ok(())
}

fn prune_old_catalogs(
    catalogs: &Directory,
    mut names: Vec<String>,
    revalidate: &impl Fn() -> Result<(), ManagedSessionCacheError>,
) -> Result<Vec<String>, ManagedSessionCacheError> {
    names.sort_by_key(|name| parse_catalog_name(name).map(|entry| entry.0));
    while names.len() > MAX_RETAINED_CATALOGS {
        let name = names.remove(0);
        revalidate()?;
        let identity = unix::inspect_owned_regular(catalogs, &name, 1)?;
        unix::remove_file(catalogs, &name, identity)?;
        unix::sync_directory(catalogs)?;
        revalidate()?;
    }
    Ok(names)
}

fn inspect_object_identity(
    objects: &Directory,
    digest: &str,
) -> Result<(String, EntryIdentity), ManagedSessionCacheError> {
    let shard = unix::open_directory(objects, &digest[..2])?;
    let name = format!("{digest}.checkpoint");
    let identity = unix::inspect_owned_regular(&shard, &name, 1)?;
    Ok((digest.to_owned(), identity))
}

fn remove_object_expected(
    objects: &Directory,
    digest: &str,
    expected: EntryIdentity,
) -> Result<(), ManagedSessionCacheError> {
    let shard = unix::open_directory(objects, &digest[..2])?;
    let name = format!("{digest}.checkpoint");
    unix::remove_file(&shard, &name, expected)?;
    unix::sync_directory(&shard)
}

fn remove_if_present(parent: &Directory, name: &str) -> Result<(), ManagedSessionCacheError> {
    match unix::entry_identity(parent, name)? {
        Some(_) => {
            let identity = unix::inspect_owned_regular(parent, name, 1)?;
            unix::remove_file(parent, name, identity)?;
            unix::sync_directory(parent)
        }
        None => Ok(()),
    }
}

fn remove_exact_if_present(
    parent: &Directory,
    name: &str,
    expected: EntryIdentity,
) -> Result<(), ManagedSessionCacheError> {
    match unix::entry_identity(parent, name)? {
        Some(actual) if actual == expected => {
            unix::remove_file(parent, name, expected)?;
            unix::sync_directory(parent)
        }
        Some(_) => Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache partial changed before cleanup",
        )),
        None => Ok(()),
    }
}

fn revalidate_recovery(
    binding: &RuntimeConfigBinding,
    sessions: &Directory,
    directories: &StoreDirectories,
    lock: &StoreLock,
) -> Result<(), ManagedSessionCacheError> {
    binding
        .revalidate()
        .map_err(|error| ManagedSessionCacheError::StaleAuthorization(error.to_string()))?;
    unix::verify_directory(sessions, PENDING_DIR, &directories.pending)?;
    unix::verify_directory(sessions, OBJECTS_DIR, &directories.objects)?;
    unix::verify_directory(sessions, CATALOGS_DIR, &directories.catalogs)?;
    unix::verify_directory(sessions, QUARANTINE_DIR, &directories.quarantine)?;
    unix::verify_lock(sessions, LOCK_NAME, lock)
}

pub(super) fn verify_managed_root_inventory(
    sessions: &Directory,
) -> Result<(), ManagedSessionCacheError> {
    let expected = BTreeSet::from([
        LOCK_NAME.to_owned(),
        PENDING_DIR.to_owned(),
        OBJECTS_DIR.to_owned(),
        CATALOGS_DIR.to_owned(),
        QUARANTINE_DIR.to_owned(),
    ]);
    if unix::list_names_bounded(sessions, expected.len() + 1)? != expected {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "sessions directory contains an unknown managed entry",
        ));
    }
    Ok(())
}

pub(super) fn inventory_charge(
    sessions: &Directory,
    directories: &StoreDirectories,
) -> Result<u64, ManagedSessionCacheError> {
    verify_managed_root_inventory(sessions)?;
    let mut total = 0u64;
    for directory in [
        &directories.pending,
        &directories.objects,
        &directories.catalogs,
        &directories.quarantine,
    ] {
        total = total
            .checked_add(unix::directory_charge(directory)?)
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    }
    let lock = unix::inspect_owned_regular(sessions, LOCK_NAME, 1)?;
    total = total
        .checked_add(lock.charge())
        .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    for name in unix::list_names_bounded(&directories.pending, MAX_PENDING_NAMES + 1)? {
        if name != OBJECT_PARTIAL && name != CATALOG_PARTIAL {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed pending directory contains an unknown name",
            ));
        }
        total = total
            .checked_add(unix::inspect_owned_regular(&directories.pending, &name, 1)?.charge())
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    }
    let mut generations = BTreeSet::new();
    for name in unix::list_names_bounded(&directories.catalogs, MAX_RETAINED_CATALOGS + 2)? {
        let (generation, _) = parse_catalog_name(&name).ok_or(
            ManagedSessionCacheError::InvalidLayout("managed catalog name is invalid"),
        )?;
        if !generations.insert(generation) {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalogs contain an ambiguous generation",
            ));
        }
        total = total
            .checked_add(unix::inspect_owned_regular(&directories.catalogs, &name, 1)?.charge())
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    }
    for name in unix::list_names_bounded(&directories.quarantine, MAX_QUARANTINE_ENTRIES + 1)? {
        if !valid_quarantine_name(&name) {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed quarantine contains an unknown name",
            ));
        }
        total = total
            .checked_add(unix::inspect_owned_regular(&directories.quarantine, &name, 1)?.charge())
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    }
    let shard_names = unix::list_names_bounded(&directories.objects, MAX_OBJECT_SHARDS)?;
    for shard_name in &shard_names {
        if shard_name.len() != 2 || !is_lower_hex(shard_name) {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed object shard name is invalid",
            ));
        }
    }
    let objects = inventory_objects(&directories.objects)?;
    for shard_name in shard_names {
        let shard = unix::open_directory(&directories.objects, &shard_name)?;
        total = total
            .checked_add(unix::directory_charge(&shard)?)
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    }
    for object in objects.values() {
        total = total
            .checked_add(object.identity.charge())
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    }
    Ok(total)
}

fn enforce_free_floor(
    directory: &Directory,
    object_bytes: u64,
    catalog_bytes: usize,
) -> Result<(), ManagedSessionCacheError> {
    #[cfg(test)]
    let injected = TEST_VOLUME_SPACE.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let injected: Option<(u64, u64, u64)> = None;
    let (volume, available, fragment) = if let Some((volume, available, fragment)) = injected {
        (volume, available, fragment.max(1))
    } else {
        let stat = unix::volume_space(directory)?;
        let fragment = stat.f_frsize.max(1);
        (
            stat.f_blocks
                .checked_mul(fragment)
                .ok_or(ManagedSessionCacheError::FreeSpaceFloor)?,
            stat.f_bavail
                .checked_mul(fragment)
                .ok_or(ManagedSessionCacheError::FreeSpaceFloor)?,
            fragment,
        )
    };
    let fifteen_percent = volume
        .checked_mul(15)
        .and_then(|value| value.checked_add(99))
        .map(|value| value / 100)
        .ok_or(ManagedSessionCacheError::FreeSpaceFloor)?;
    let reserve = MIN_FREE_BYTES.max(fifteen_percent);
    let required = reserve
        .checked_add(round_up(object_bytes, fragment)?)
        .and_then(|value| value.checked_add(round_up(catalog_bytes as u64, fragment).ok()?))
        .and_then(|value| value.checked_add(fragment.saturating_mul(METADATA_BLOCK_MARGIN)))
        .ok_or(ManagedSessionCacheError::FreeSpaceFloor)?;
    if available < required {
        return Err(ManagedSessionCacheError::FreeSpaceFloor);
    }
    Ok(())
}

fn round_up(value: u64, unit: u64) -> Result<u64, ManagedSessionCacheError> {
    value
        .checked_add(unit - 1)
        .map(|rounded| rounded / unit * unit)
        .ok_or(ManagedSessionCacheError::QuotaExceeded)
}

fn is_lower_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_quarantine_name(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".checkpoint") else {
        return false;
    };
    let mut parts = stem.splitn(3, '-');
    let Some(sequence) = parts.next() else {
        return false;
    };
    let Some(reason) = parts.next() else {
        return false;
    };
    let Some(digest) = parts.next() else {
        return false;
    };
    sequence.len() == 20
        && sequence.bytes().all(|byte| byte.is_ascii_digit())
        && !reason.is_empty()
        && reason.len() <= 32
        && reason
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
        && digest.len() == 64
        && is_lower_hex(digest)
}
