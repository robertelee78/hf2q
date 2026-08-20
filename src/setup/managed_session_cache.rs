//! Dormant descriptor-authorized managed session-cache coordinator.
//!
//! This module deliberately has no serving, environment, CLI, model, or
//! codec dependency. It consumes the positive setup capability and owns the
//! only filesystem mutation surface for the future managed cache.

mod catalog;
mod transaction;
mod unix;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::Mutex;

use sha2::{Digest, Sha256};
use thiserror::Error;

use self::catalog::CatalogV1;
use self::unix::{Directory, StoreLock};
use super::fs::RuntimeConfigBinding;

pub(super) const MAX_CATALOG_BYTES: usize = 8 * 1024 * 1024;
pub(super) const MAX_LIVE_ENTRIES: usize = 4096;
pub(super) const MAX_OBJECTS: usize = 8192;
pub(super) const MAX_RETAINED_CATALOGS: usize = 4;
pub(super) const MAX_QUARANTINE_ENTRIES: usize = 128;
pub(super) const MAX_CHECKPOINT_BYTES: u64 = 100 * 1024 * 1024 * 1024;
pub(super) const MIN_FREE_BYTES: u64 = 20 * 1024 * 1024 * 1024;
pub(super) const MIN_MANAGED_CAPACITY_BYTES: u64 = 1024 * 1024;

pub(super) const LOCK_NAME: &str = ".managed-session-cache.lock";
pub(super) const PENDING_DIR: &str = "pending";
pub(super) const OBJECTS_DIR: &str = "objects";
pub(super) const CATALOGS_DIR: &str = "catalogs";
pub(super) const QUARANTINE_DIR: &str = "quarantine";
pub(super) const OBJECT_PARTIAL: &str = ".object-v1.partial";
pub(super) const CATALOG_PARTIAL: &str = ".catalog-v1.partial";

pub(super) const OBJECT_MAGIC: &[u8; 8] = b"HF2QMSC1";
pub(super) const OBJECT_VERSION: u32 = 1;
pub(super) const OBJECT_HEADER_BYTES: usize = 8 + 4 + 32 + 8 + 32;
pub(super) const MAX_OBJECT_BYTES: u64 = MAX_CHECKPOINT_BYTES + OBJECT_HEADER_BYTES as u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ManagedCacheBarrier {
    DirectoryCreatedBeforeMode,
    FileCreatedBeforeMode,
    AuthorizationValidated,
    ObjectPartialSynced,
    BeforeObjectPublish,
    ObjectPublished,
    ObjectDirectorySynced,
    CatalogPartialSynced,
    BeforeCatalogPublish,
    CatalogPublished,
    CatalogDirectorySynced,
    BeforeCatalogHistoryPrune,
    CatalogHistoryPruned,
    BeforeObjectDelete,
    ObjectDeleted,
    EndpointSynced,
}

impl ManagedCacheBarrier {
    #[cfg(test)]
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::DirectoryCreatedBeforeMode => "directory-created-before-mode",
            Self::FileCreatedBeforeMode => "file-created-before-mode",
            Self::AuthorizationValidated => "authorization-validated",
            Self::ObjectPartialSynced => "object-partial-synced",
            Self::BeforeObjectPublish => "before-object-publish",
            Self::ObjectPublished => "object-published",
            Self::ObjectDirectorySynced => "object-directory-synced",
            Self::CatalogPartialSynced => "catalog-partial-synced",
            Self::BeforeCatalogPublish => "before-catalog-publish",
            Self::CatalogPublished => "catalog-published",
            Self::CatalogDirectorySynced => "catalog-directory-synced",
            Self::BeforeCatalogHistoryPrune => "before-catalog-history-prune",
            Self::CatalogHistoryPruned => "catalog-history-pruned",
            Self::BeforeObjectDelete => "before-object-delete",
            Self::ObjectDeleted => "object-deleted",
            Self::EndpointSynced => "endpoint-synced",
        }
    }
}

#[derive(Debug, Error)]
pub(super) enum ManagedSessionCacheError {
    #[error("managed session cache is busy")]
    Busy,
    #[error("managed session cache is missing required state")]
    Missing,
    #[error("managed session cache exceeds its configured aggregate quota")]
    QuotaExceeded,
    #[error("managed session cache would cross the volume free-space floor")]
    FreeSpaceFloor,
    #[error("managed session cache storage is full")]
    StorageFull,
    #[error("managed session cache must be reopened to reconcile a possible commit")]
    RecoveryRequired,
    #[error("managed session cache layout is unsafe: {0}")]
    InvalidLayout(&'static str),
    #[error("managed session cache filesystem: {0}")]
    Filesystem(String),
    #[error("managed session cache authorization is stale: {0}")]
    StaleAuthorization(String),
    #[error("managed session cache catalog may have committed but durability is unknown: {0}")]
    CommittedDurabilityUnknown(String),
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct ManagedCheckpointKey([u8; 32]);

impl ManagedCheckpointKey {
    /// Create the content-addressed logical key from a complete canonical
    /// compatibility receipt. The receipt schema is owned by the future
    /// family adapter; this store never interprets or weakens it.
    pub(super) fn from_canonical_receipt(receipt: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"hf2q-managed-session-checkpoint-key-v1\0");
        hasher.update(receipt);
        Self(hasher.finalize().into())
    }

    fn hex(self) -> String {
        hex::encode(self.0)
    }

    fn from_hex(value: &str) -> Result<Self, ManagedSessionCacheError> {
        let bytes = hex::decode(value).map_err(|_| {
            ManagedSessionCacheError::InvalidLayout("managed checkpoint key hex is invalid")
        })?;
        let bytes: [u8; 32] = bytes.try_into().map_err(|_| {
            ManagedSessionCacheError::InvalidLayout("managed checkpoint key length is invalid")
        })?;
        Ok(Self(bytes))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ManagedCommitOutcome {
    Published,
    AlreadyPresent,
}

struct StoreDirectories {
    pending: Directory,
    objects: Directory,
    catalogs: Directory,
    quarantine: Directory,
}

struct StoreState {
    catalog: CatalogV1,
    retained_catalogs: Vec<RetainedCatalog>,
    retained_objects: BTreeMap<String, RetainedObject>,
    needs_recovery: bool,
}

struct RetainedCatalog {
    name: String,
    identity: unix::EntryIdentity,
}

#[derive(Clone)]
struct RetainedObject {
    shard: String,
    shard_identity: unix::EntryIdentity,
    name: String,
    identity: unix::EntryIdentity,
}

/// One-owner cache capability. It is intentionally non-Clone and its Debug
/// output reveals neither policy bytes nor filesystem identities.
pub(super) struct ManagedSessionCache {
    limit_bytes: NonZeroU64,
    binding: RuntimeConfigBinding,
    sessions: Directory,
    directories: StoreDirectories,
    lock: StoreLock,
    state: Mutex<StoreState>,
}

impl ManagedSessionCache {
    pub(in crate::setup) fn open(
        limit_bytes: NonZeroU64,
        binding: RuntimeConfigBinding,
    ) -> Result<Self, ManagedSessionCacheError> {
        binding
            .revalidate()
            .map_err(|error| ManagedSessionCacheError::StaleAuthorization(error.to_string()))?;
        let sessions = Directory::duplicate(binding.session_directory_fd())?;
        transaction::require_minimum_capacity(&sessions, limit_bytes)?;
        let complete_layout = transaction::preflight_existing_layout(&sessions)?;
        if !complete_layout {
            transaction::require_new_layout_free_space(&sessions)?;
        }
        let pending = unix::ensure_directory(&sessions, PENDING_DIR)?;
        let objects = unix::ensure_directory(&sessions, OBJECTS_DIR)?;
        let catalogs = unix::ensure_directory(&sessions, CATALOGS_DIR)?;
        let quarantine = unix::ensure_directory(&sessions, QUARANTINE_DIR)?;
        let lock = unix::acquire_lock(&sessions, LOCK_NAME)?;
        let directories = StoreDirectories {
            pending,
            objects,
            catalogs,
            quarantine,
        };
        binding
            .revalidate()
            .map_err(|error| ManagedSessionCacheError::StaleAuthorization(error.to_string()))?;
        unix::verify_lock(&sessions, LOCK_NAME, &lock)?;
        let (catalog, retained_catalogs, retained_objects) =
            transaction::recover(&binding, &directories, &sessions, &lock, limit_bytes)?;
        let store = Self {
            limit_bytes,
            binding,
            sessions,
            directories,
            lock,
            state: Mutex::new(StoreState {
                catalog,
                retained_catalogs,
                retained_objects,
                needs_recovery: false,
            }),
        };
        store.revalidate_endpoint()?;
        if store.inventory_charge()? > limit_bytes.get() {
            return Err(ManagedSessionCacheError::QuotaExceeded);
        }
        Ok(store)
    }

    pub(super) fn commit(
        &self,
        key: ManagedCheckpointKey,
        payload: &[u8],
    ) -> Result<ManagedCommitOutcome, ManagedSessionCacheError> {
        transaction::commit(self, key, payload)
    }

    pub(super) fn restore(
        &self,
        key: ManagedCheckpointKey,
    ) -> Result<Option<Vec<u8>>, ManagedSessionCacheError> {
        transaction::restore(self, key)
    }

    fn revalidate_endpoint(&self) -> Result<(), ManagedSessionCacheError> {
        self.binding
            .revalidate()
            .map_err(|error| ManagedSessionCacheError::StaleAuthorization(error.to_string()))?;
        transaction::verify_managed_root_inventory(&self.sessions)?;
        unix::verify_directory(&self.sessions, PENDING_DIR, &self.directories.pending)?;
        unix::verify_directory(&self.sessions, OBJECTS_DIR, &self.directories.objects)?;
        unix::verify_directory(&self.sessions, CATALOGS_DIR, &self.directories.catalogs)?;
        unix::verify_directory(&self.sessions, QUARANTINE_DIR, &self.directories.quarantine)?;
        unix::verify_lock(&self.sessions, LOCK_NAME, &self.lock)?;
        Ok(())
    }

    fn inventory_charge(&self) -> Result<u64, ManagedSessionCacheError> {
        transaction::inventory_charge(&self.sessions, &self.directories)
    }
}

impl fmt::Debug for ManagedSessionCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ManagedSessionCache(<redacted>)")
    }
}

struct EncodedObject<'a> {
    header: [u8; OBJECT_HEADER_BYTES],
    payload: &'a [u8],
    total_bytes: u64,
    digest: String,
}

fn encode_object<'a>(
    key: ManagedCheckpointKey,
    payload: &'a [u8],
) -> Result<EncodedObject<'a>, ManagedSessionCacheError> {
    let payload_len =
        u64::try_from(payload.len()).map_err(|_| ManagedSessionCacheError::QuotaExceeded)?;
    if payload_len == 0 || payload_len > MAX_CHECKPOINT_BYTES {
        return Err(ManagedSessionCacheError::QuotaExceeded);
    }
    let total = OBJECT_HEADER_BYTES
        .checked_add(payload.len())
        .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    let mut header = [0u8; OBJECT_HEADER_BYTES];
    header[..8].copy_from_slice(OBJECT_MAGIC);
    header[8..12].copy_from_slice(&OBJECT_VERSION.to_le_bytes());
    header[12..44].copy_from_slice(&key.0);
    header[44..52].copy_from_slice(&payload_len.to_le_bytes());
    header[52..84].copy_from_slice(&Sha256::digest(payload));
    let mut hasher = Sha256::new();
    hasher.update(header);
    hasher.update(payload);
    Ok(EncodedObject {
        header,
        payload,
        total_bytes: total as u64,
        digest: hex::encode(hasher.finalize()),
    })
}
