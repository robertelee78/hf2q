use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{ManagedSessionCacheError, MAX_CATALOG_BYTES, MAX_LIVE_ENTRIES, MAX_OBJECT_BYTES};

pub(super) const CATALOG_KIND: &str = "hf2q.managed-session-cache.catalog";
pub(super) const CATALOG_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CatalogV1 {
    kind: String,
    schema_version: u32,
    generation: u64,
    entries: Vec<CatalogEntryV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CatalogEntryV1 {
    pub(super) key_sha256: String,
    pub(super) object_sha256: String,
    pub(super) object_bytes: u64,
    pub(super) last_committed_generation: u64,
}

impl CatalogV1 {
    pub(super) fn empty() -> Self {
        Self {
            kind: CATALOG_KIND.to_owned(),
            schema_version: CATALOG_SCHEMA_VERSION,
            generation: 0,
            entries: Vec::new(),
        }
    }

    pub(super) fn parse_exact(bytes: &[u8]) -> Result<Self, ManagedSessionCacheError> {
        if bytes.is_empty() || bytes.len() > MAX_CATALOG_BYTES || !bytes.ends_with(b"\n") {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog has an invalid bounded encoding",
            ));
        }
        let parsed: Self = serde_json::from_slice(bytes).map_err(|_| {
            ManagedSessionCacheError::InvalidLayout("managed catalog is not valid JSON")
        })?;
        parsed.validate()?;
        if parsed.to_canonical_bytes()? != bytes {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog is not canonically encoded",
            ));
        }
        Ok(parsed)
    }

    pub(super) fn to_canonical_bytes(&self) -> Result<Vec<u8>, ManagedSessionCacheError> {
        self.validate()?;
        let mut bytes = serde_json::to_vec(self).map_err(|_| {
            ManagedSessionCacheError::InvalidLayout("managed catalog could not be encoded")
        })?;
        bytes.push(b'\n');
        if bytes.len() > MAX_CATALOG_BYTES {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog exceeds its byte cap",
            ));
        }
        Ok(bytes)
    }

    fn validate(&self) -> Result<(), ManagedSessionCacheError> {
        if self.kind != CATALOG_KIND || self.schema_version != CATALOG_SCHEMA_VERSION {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog has an unsupported identity",
            ));
        }
        if self.entries.len() > MAX_LIVE_ENTRIES {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog exceeds its entry cap",
            ));
        }
        let mut keys = BTreeSet::new();
        let mut previous = None;
        for entry in &self.entries {
            require_lower_hex(&entry.key_sha256)?;
            require_lower_hex(&entry.object_sha256)?;
            if entry.object_bytes == 0
                || entry.object_bytes > MAX_OBJECT_BYTES
                || entry.last_committed_generation == 0
                || entry.last_committed_generation > self.generation
            {
                return Err(ManagedSessionCacheError::InvalidLayout(
                    "managed catalog entry has an invalid size or generation",
                ));
            }
            if !keys.insert(entry.key_sha256.as_str()) {
                return Err(ManagedSessionCacheError::InvalidLayout(
                    "managed catalog contains a duplicate key",
                ));
            }
            if previous.is_some_and(|value: &str| value >= entry.key_sha256.as_str()) {
                return Err(ManagedSessionCacheError::InvalidLayout(
                    "managed catalog entries are not in canonical key order",
                ));
            }
            previous = Some(entry.key_sha256.as_str());
        }
        Ok(())
    }

    pub(super) const fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn entries(&self) -> &[CatalogEntryV1] {
        &self.entries
    }

    pub(super) fn find(&self, key: &str) -> Option<&CatalogEntryV1> {
        self.entries
            .binary_search_by(|entry| entry.key_sha256.as_str().cmp(key))
            .ok()
            .map(|index| &self.entries[index])
    }

    pub(super) fn without_keys(
        &self,
        removed: &BTreeSet<String>,
    ) -> Result<Self, ManagedSessionCacheError> {
        self.next(
            self.entries
                .iter()
                .filter(|entry| !removed.contains(&entry.key_sha256))
                .cloned()
                .collect(),
        )
    }

    pub(super) fn with_entry(
        &self,
        mut entry: CatalogEntryV1,
    ) -> Result<Self, ManagedSessionCacheError> {
        let next_generation =
            self.generation
                .checked_add(1)
                .ok_or(ManagedSessionCacheError::InvalidLayout(
                    "managed catalog generation overflowed",
                ))?;
        entry.last_committed_generation = next_generation;
        let mut entries: Vec<_> = self
            .entries
            .iter()
            .filter(|candidate| candidate.key_sha256 != entry.key_sha256)
            .cloned()
            .collect();
        entries.push(entry);
        entries.sort_by(|left, right| left.key_sha256.cmp(&right.key_sha256));
        Self {
            kind: CATALOG_KIND.to_owned(),
            schema_version: CATALOG_SCHEMA_VERSION,
            generation: next_generation,
            entries,
        }
        .validated()
    }

    fn next(&self, mut entries: Vec<CatalogEntryV1>) -> Result<Self, ManagedSessionCacheError> {
        entries.sort_by(|left, right| left.key_sha256.cmp(&right.key_sha256));
        Self {
            kind: CATALOG_KIND.to_owned(),
            schema_version: CATALOG_SCHEMA_VERSION,
            generation: self.generation.checked_add(1).ok_or(
                ManagedSessionCacheError::InvalidLayout("managed catalog generation overflowed"),
            )?,
            entries,
        }
        .validated()
    }

    fn validated(self) -> Result<Self, ManagedSessionCacheError> {
        self.validate()?;
        Ok(self)
    }
}

impl CatalogEntryV1 {
    pub(super) fn new(key_sha256: String, object_sha256: String, object_bytes: u64) -> Self {
        Self {
            key_sha256,
            object_sha256,
            object_bytes,
            last_committed_generation: 1,
        }
    }
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub(super) fn catalog_name(generation: u64, digest: &str) -> String {
    format!("{generation:020}-{digest}.catalog")
}

pub(super) fn parse_catalog_name(name: &str) -> Option<(u64, &str)> {
    if name.len() != 20 + 1 + 64 + ".catalog".len() || !name.ends_with(".catalog") {
        return None;
    }
    let (generation, remainder) = name.split_once('-')?;
    let digest = remainder.strip_suffix(".catalog")?;
    if !generation.bytes().all(|byte| byte.is_ascii_digit()) || require_lower_hex(digest).is_err() {
        return None;
    }
    Some((generation.parse().ok()?, digest))
}

fn require_lower_hex(value: &str) -> Result<(), ManagedSessionCacheError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed catalog digest is not lowercase SHA-256 hex",
        ));
    }
    Ok(())
}
