use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::num::NonZeroU64;

use sha2::{Digest, Sha256};

use super::catalog::{catalog_name, parse_catalog_name, sha256_hex, CatalogEntryV1, CatalogV1};
use super::unix::{self, Directory, EntryIdentity, StoreLock};
use super::{
    encode_object, EncodedObject, ManagedCacheBarrier, ManagedCheckpointKey, ManagedCommitOutcome,
    ManagedSessionCache, ManagedSessionCacheError, RetainedCatalog, RetainedObject,
    StoreDirectories, CATALOGS_DIR, CATALOG_PARTIAL, LOCK_NAME, MAX_CATALOG_BYTES,
    MAX_CHECKPOINT_BYTES, MAX_LIVE_ENTRIES, MAX_OBJECTS, MAX_OBJECT_BYTES, MAX_QUARANTINE_ENTRIES,
    MAX_RETAINED_CATALOGS, MIN_FREE_BYTES, MIN_MANAGED_CAPACITY_BYTES, OBJECTS_DIR,
    OBJECT_HEADER_BYTES, OBJECT_MAGIC, OBJECT_PARTIAL, OBJECT_VERSION, PENDING_DIR, QUARANTINE_DIR,
};
use crate::setup::fs::RuntimeConfigBinding;

const MAX_PENDING_NAMES: usize = 2;
const MAX_OBJECT_SHARDS: usize = 256;
const METADATA_BLOCK_MARGIN: u64 = 16;
const OBJECT_WRITE_CHUNK_BYTES: usize = 64 * 1024;

#[cfg(test)]
thread_local! {
    static TEST_VOLUME_SPACE: std::cell::Cell<Option<(u64, u64, u64)>> = const {
        std::cell::Cell::new(None)
    };
    static TEST_IO_FAULT: std::cell::Cell<Option<(TestIoFault, bool)>> = const {
        std::cell::Cell::new(None)
    };
    static TEST_OBJECT_WRITE_FAILURE_LEN: std::cell::Cell<Option<u64>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TestIoFault {
    ObjectWrite,
    ObjectFullSync,
    ObjectFinalFullSync,
    ObjectDirectorySync,
    ObjectPendingDirectorySync,
    CatalogWrite,
    CatalogFullSync,
    CatalogDirectorySync,
    CatalogFinalFullSync,
    PendingDirectorySync,
    EndpointLockFullSync,
    DeletionDirectorySync,
    CatalogHistoryDirectorySync,
}

pub(super) fn preflight_existing_layout(
    sessions: &Directory,
) -> Result<bool, ManagedSessionCacheError> {
    let names = unix::list_names_bounded(sessions, 6)?;
    let complete = names.len() == 5;
    for name in names {
        match name.as_str() {
            LOCK_NAME => {
                let identity = unix::inspect_or_normalize_empty_lock(sessions, LOCK_NAME)?;
                if identity.size() != 0 {
                    return Err(ManagedSessionCacheError::InvalidLayout(
                        "managed cache lock is not empty",
                    ));
                }
            }
            PENDING_DIR => {
                let pending = unix::open_recoverable_directory(sessions, PENDING_DIR)?;
                verify_pending_inventory(&pending)?;
            }
            OBJECTS_DIR => {
                let objects = unix::open_recoverable_directory(sessions, OBJECTS_DIR)?;
                inventory_objects(&objects)?;
            }
            CATALOGS_DIR => {
                let catalogs = unix::open_recoverable_directory(sessions, CATALOGS_DIR)?;
                preflight_catalog_inventory(&catalogs)?;
            }
            QUARANTINE_DIR => {
                let quarantine = unix::open_recoverable_directory(sessions, QUARANTINE_DIR)?;
                verify_quarantine(&quarantine)?;
            }
            _ => {
                return Err(ManagedSessionCacheError::InvalidLayout(
                    "sessions directory contains an unknown managed entry",
                ));
            }
        }
    }
    Ok(complete)
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

#[cfg(test)]
pub(super) fn set_test_volume_space(volume: u64, available: u64, fragment: u64) {
    TEST_VOLUME_SPACE.with(|slot| slot.set(Some((volume, available, fragment))));
}

#[cfg(test)]
pub(super) fn reset_test_object_write_failure_len() {
    TEST_OBJECT_WRITE_FAILURE_LEN.with(|slot| slot.set(None));
}

#[cfg(test)]
pub(super) fn take_test_object_write_failure_len() -> Option<u64> {
    TEST_OBJECT_WRITE_FAILURE_LEN.with(std::cell::Cell::take)
}

#[cfg(test)]
pub(super) fn with_test_io_fault<T>(fault: TestIoFault, action: impl FnOnce() -> T) -> T {
    TEST_IO_FAULT.with(|slot| {
        let previous = slot.replace(Some((fault, false)));
        let result = action();
        assert_eq!(
            slot.get(),
            Some((fault, true)),
            "injected I/O fault was not reached"
        );
        slot.set(previous);
        result
    })
}

#[cfg(test)]
fn test_io(fault: TestIoFault) -> Result<(), ManagedSessionCacheError> {
    TEST_IO_FAULT.with(|slot| {
        if slot.get().is_some_and(|(candidate, _)| candidate == fault) {
            slot.set(Some((fault, true)));
            Err(ManagedSessionCacheError::StorageFull)
        } else {
            Ok(())
        }
    })
}

#[cfg(test)]
fn test_io_fault_active(fault: TestIoFault) -> bool {
    TEST_IO_FAULT.with(|slot| {
        if slot.get().is_some_and(|(candidate, _)| candidate == fault) {
            slot.set(Some((fault, true)));
            true
        } else {
            false
        }
    })
}

pub(super) fn recover(
    binding: &RuntimeConfigBinding,
    directories: &StoreDirectories,
    sessions: &Directory,
    lock: &StoreLock,
    limit_bytes: NonZeroU64,
) -> Result<
    (
        CatalogV1,
        Vec<RetainedCatalog>,
        BTreeMap<String, RetainedObject>,
    ),
    ManagedSessionCacheError,
> {
    let revalidate = || revalidate_recovery(binding, sessions, directories, lock);
    revalidate()?;
    let (catalog, names) = load_highest_catalog(&directories.catalogs)?;
    let mut objects = inventory_objects(&directories.objects)?;
    validate_catalog_object_shapes(&catalog, &objects, limit_bytes)?;
    verify_catalog_objects(&directories.objects, &catalog, &objects, limit_bytes.get())?;
    verify_quarantine(&directories.quarantine)?;
    revalidate()?;
    clean_pending(&directories.pending, &revalidate)?;
    clean_orphans_with_parent(&directories.objects, &catalog, &objects, &revalidate)?;
    let referenced: BTreeSet<_> = catalog
        .entries()
        .iter()
        .map(|entry| entry.object_sha256.as_str())
        .collect();
    objects.retain(|digest, _| referenced.contains(digest.as_str()));
    revalidate()?;
    if inventory_charge(sessions, directories)? > limit_bytes.get() {
        return Err(ManagedSessionCacheError::QuotaExceeded);
    }
    let names = prune_old_catalogs(&directories.catalogs, names, &revalidate)?;
    revalidate()?;
    unix::sync_directory(&directories.pending)?;
    unix::sync_directory(&directories.objects)?;
    unix::sync_directory(&directories.catalogs)?;
    unix::sync_directory(&directories.quarantine)?;
    unix::sync_directory(sessions)?;
    unix::full_sync_lock(lock)?;
    revalidate()?;
    Ok((catalog, names, objects))
}

pub(super) fn require_minimum_capacity(
    sessions: &Directory,
    limit_bytes: NonZeroU64,
) -> Result<(), ManagedSessionCacheError> {
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

pub(super) fn require_new_layout_free_space(
    sessions: &Directory,
) -> Result<(), ManagedSessionCacheError> {
    enforce_free_floor(sessions, 0, 0)
}

include!("transaction/commit.rs");
include!("transaction/maintenance.rs");
include!("transaction/io.rs");
include!("transaction/catalog_state.rs");
include!("transaction/inventory.rs");
include!("transaction/quota.rs");
