//! hf2q's bounded TUF v1 repository profile.

use std::collections::{BTreeSet, HashSet};

use jiff::Timestamp as Instant;
use sigstore_tuf::metadata::{MetaFile, Metadata, Role, Root, Snapshot, Targets, Timestamp};

use super::model::{MAX_ROOT_BYTES, MAX_SNAPSHOT_BYTES, MAX_TARGETS_BYTES, MAX_TIMESTAMP_BYTES};
use super::strict_json;
use super::TufVerifierError;

const MAX_SIGNATURES: usize = 64;
const MAX_ROOT_KEYS: usize = 64;
const MAX_KEY_IDS_PER_ROLE: usize = 64;
const MAX_SIGNATURE_HEX_BYTES: usize = 16 * 1024;
const MAX_TARGETS: usize = 4096;
const MAX_TARGET_NAME_BYTES: usize = 512;
const MAX_HASHES_PER_DESCRIPTOR: usize = 4;

pub(super) fn root(bytes: &[u8]) -> Result<Metadata<Root>, TufVerifierError> {
    strict_json::validate(bytes, MAX_ROOT_BYTES)?;
    require_exact_envelope(bytes)?;
    let metadata =
        Metadata::<Root>::from_slice(bytes).map_err(|_| TufVerifierError::MalformedMetadata)?;
    require_signatures(&metadata)?;
    let role = &metadata.signed;
    require_common(role)?;
    if role.keys.is_empty()
        || role.keys.len() > MAX_ROOT_KEYS
        || !role.extra.is_empty()
        || role.roles.keys().cloned().collect::<BTreeSet<_>>()
            != BTreeSet::from([
                "root".to_owned(),
                "snapshot".to_owned(),
                "targets".to_owned(),
                "timestamp".to_owned(),
            ])
    {
        return Err(TufVerifierError::MalformedMetadata);
    }
    for (key_id, key) in &role.keys {
        let computed_key_id = key
            .key_id()
            .map_err(|_| TufVerifierError::MalformedMetadata)?;
        if !is_lower_hex_64(key_id)
            || computed_key_id != *key_id
            || key.keytype != "ed25519"
            || key.scheme != "ed25519"
            || !is_lower_hex_64(&key.keyval.public)
            || !key.extra.is_empty()
            || !key.keyval.extra.is_empty()
        {
            return Err(TufVerifierError::MalformedMetadata);
        }
    }
    for binding in role.roles.values() {
        let unique: HashSet<_> = binding.keyids.iter().collect();
        if binding.threshold == 0
            || binding.keyids.is_empty()
            || binding.keyids.len() > MAX_KEY_IDS_PER_ROLE
            || unique.len() != binding.keyids.len()
            || binding.threshold > unique.len()
            || !binding.extra.is_empty()
            || binding
                .keyids
                .iter()
                .any(|key_id| !is_lower_hex_64(key_id) || !role.keys.contains_key(key_id))
        {
            return Err(TufVerifierError::MalformedMetadata);
        }
    }
    Ok(metadata)
}

pub(super) fn timestamp(bytes: &[u8]) -> Result<Metadata<Timestamp>, TufVerifierError> {
    strict_json::validate(bytes, MAX_TIMESTAMP_BYTES)?;
    require_exact_envelope(bytes)?;
    let metadata = Metadata::<Timestamp>::from_slice(bytes)
        .map_err(|_| TufVerifierError::MalformedMetadata)?;
    require_signatures(&metadata)?;
    require_common(&metadata.signed)?;
    if !metadata.signed.extra.is_empty()
        || metadata.signed.meta.len() != 1
        || !metadata.signed.meta.contains_key("snapshot.json")
    {
        return Err(TufVerifierError::MalformedMetadata);
    }
    require_pin(
        metadata
            .signed
            .meta
            .get("snapshot.json")
            .ok_or(TufVerifierError::MalformedMetadata)?,
        MAX_SNAPSHOT_BYTES,
    )?;
    Ok(metadata)
}

pub(super) fn snapshot(bytes: &[u8]) -> Result<Metadata<Snapshot>, TufVerifierError> {
    strict_json::validate(bytes, MAX_SNAPSHOT_BYTES)?;
    require_exact_envelope(bytes)?;
    let metadata =
        Metadata::<Snapshot>::from_slice(bytes).map_err(|_| TufVerifierError::MalformedMetadata)?;
    require_signatures(&metadata)?;
    require_common(&metadata.signed)?;
    if !metadata.signed.extra.is_empty()
        || metadata.signed.meta.len() != 1
        || !metadata.signed.meta.contains_key("targets.json")
    {
        return Err(TufVerifierError::MalformedMetadata);
    }
    require_pin(
        metadata
            .signed
            .meta
            .get("targets.json")
            .ok_or(TufVerifierError::MalformedMetadata)?,
        MAX_TARGETS_BYTES,
    )?;
    Ok(metadata)
}

pub(super) fn targets(bytes: &[u8]) -> Result<Metadata<Targets>, TufVerifierError> {
    strict_json::validate(bytes, MAX_TARGETS_BYTES)?;
    require_exact_envelope(bytes)?;
    require_exact_targets_shape(bytes)?;
    let metadata =
        Metadata::<Targets>::from_slice(bytes).map_err(|_| TufVerifierError::MalformedMetadata)?;
    require_signatures(&metadata)?;
    require_common(&metadata.signed)?;
    let role = &metadata.signed;
    if !role.extra.is_empty() || role.delegations.is_some() || role.targets.len() > MAX_TARGETS {
        return Err(TufVerifierError::MalformedMetadata);
    }
    for (name, target) in &role.targets {
        if !bounded_nonempty(name, MAX_TARGET_NAME_BYTES)
            || !target.extra.is_empty()
            || target.custom.is_some()
            || target.hashes.len() > MAX_HASHES_PER_DESCRIPTOR
        {
            return Err(TufVerifierError::MalformedMetadata);
        }
        require_sha256(&target.hashes)?;
    }
    Ok(metadata)
}

fn require_exact_targets_shape(bytes: &[u8]) -> Result<(), TufVerifierError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| TufVerifierError::MalformedMetadata)?;
    let signed = value
        .get("signed")
        .and_then(serde_json::Value::as_object)
        .ok_or(TufVerifierError::MalformedMetadata)?;
    let expected = ["_type", "expires", "spec_version", "targets", "version"];
    if signed.len() != expected.len() || expected.iter().any(|key| !signed.contains_key(*key)) {
        return Err(TufVerifierError::MalformedMetadata);
    }
    let targets = signed
        .get("targets")
        .and_then(serde_json::Value::as_object)
        .ok_or(TufVerifierError::MalformedMetadata)?;
    for target in targets.values() {
        let object = target
            .as_object()
            .ok_or(TufVerifierError::MalformedMetadata)?;
        if object.len() != 2 || !object.contains_key("length") || !object.contains_key("hashes") {
            return Err(TufVerifierError::MalformedMetadata);
        }
    }
    Ok(())
}

pub(super) fn require_fresh<T: Role>(role: &T, reference: Instant) -> Result<(), TufVerifierError> {
    let expires = expiry(role)?;
    if expires <= reference {
        return Err(TufVerifierError::ExpiredMetadata);
    }
    Ok(())
}

pub(super) fn expiry<T: Role>(role: &T) -> Result<Instant, TufVerifierError> {
    role.expires_at()
        .map_err(|_| TufVerifierError::MalformedMetadata)
}

fn require_common<T: Role>(role: &T) -> Result<(), TufVerifierError> {
    if role.version() == 0 {
        return Err(TufVerifierError::MalformedMetadata);
    }
    let expires = role
        .expires_at()
        .map_err(|_| TufVerifierError::MalformedMetadata)?;
    if expires.to_string() != role.expires() {
        return Err(TufVerifierError::MalformedMetadata);
    }
    Ok(())
}

fn require_exact_envelope(bytes: &[u8]) -> Result<(), TufVerifierError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| TufVerifierError::MalformedMetadata)?;
    let object = value
        .as_object()
        .ok_or(TufVerifierError::MalformedMetadata)?;
    if object.len() != 2 || !object.contains_key("signed") || !object.contains_key("signatures") {
        return Err(TufVerifierError::MalformedMetadata);
    }
    Ok(())
}

fn require_signatures<T: Role>(metadata: &Metadata<T>) -> Result<(), TufVerifierError> {
    if metadata.signatures.is_empty() || metadata.signatures.len() > MAX_SIGNATURES {
        return Err(TufVerifierError::MalformedMetadata);
    }
    let mut key_ids = HashSet::new();
    for signature in &metadata.signatures {
        if !is_lower_hex_64(&signature.keyid)
            || signature.sig.len() > MAX_SIGNATURE_HEX_BYTES
            || (!signature.sig.is_empty()
                && (signature.sig.len() % 2 != 0
                    || !signature.sig.bytes().all(|byte| byte.is_ascii_hexdigit())))
            || !key_ids.insert(signature.keyid.as_str())
        {
            return Err(TufVerifierError::MalformedMetadata);
        }
    }
    Ok(())
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_pin(pin: &MetaFile, maximum: usize) -> Result<(), TufVerifierError> {
    if pin.version == 0
        || pin
            .length
            .is_none_or(|length| length == 0 || length > maximum as u64)
        || !pin.extra.is_empty()
    {
        return Err(TufVerifierError::MalformedMetadata);
    }
    let hashes = pin
        .hashes
        .as_ref()
        .ok_or(TufVerifierError::MalformedMetadata)?;
    require_sha256(hashes)
}

fn require_sha256(
    hashes: &std::collections::BTreeMap<String, String>,
) -> Result<(), TufVerifierError> {
    if hashes.len() != 1 {
        return Err(TufVerifierError::MalformedMetadata);
    }
    let digest = hashes
        .get("sha256")
        .ok_or(TufVerifierError::MalformedMetadata)?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(TufVerifierError::MalformedMetadata);
    }
    Ok(())
}

fn bounded_nonempty(value: &str, maximum: usize) -> bool {
    !value.is_empty() && value.len() <= maximum
}
