//! In-memory signed repository builder used only by the spike.

use std::collections::HashMap;
use std::fmt;
use std::io::{Cursor, Write};
use std::num::NonZeroU64;

use async_trait::async_trait;
use aws_lc_rs::rand::SystemRandom;
use aws_lc_rs::signature::Ed25519KeyPair;
use jiff::Timestamp;
use tempfile::TempDir;
use tough::editor::signed::SignedRole;
use tough::editor::RepositoryEditor;
use tough::key_source::KeySource;
use tough::schema::{
    Delegations, Hashes, KeyHolder, Metafile, RoleKeys, RoleType, Root, Signed, Snapshot, Target,
    Targets, Timestamp as TufTimestamp,
};
use tough::sign::Sign;
use tough::TargetName;

use crate::application_binding::{archive_name, manifest_name, pointer_bytes, pointer_name};
use crate::capture_transport::ScriptedResponse;
use crate::model::sha256;

pub(crate) const RELEASE_VERSION: &str = "0.2.0";
pub(crate) const RELEASE_TARGET: &str = "aarch64-apple-darwin";

const FIXTURE_PAYLOADS: &[(&str, u32, &[u8])] = &[
    ("bin/hf2q", 0o755, b"hf2q-test-binary\n"),
    (
        "libexec/serve_qwen38_opencode.sh",
        0o755,
        b"#!/bin/sh\nexec hf2q serve Qwen/Qwen3.8-27B\n",
    ),
    (
        "share/doc/hf2q/README.md",
        0o644,
        b"hf2q fixture documentation\n",
    ),
    (
        "share/licenses/hf2q/LICENSE-APACHE",
        0o644,
        b"fixture Apache-2.0 license text\n",
    ),
];

#[derive(Clone)]
struct MemoryKeySource {
    seed: [u8; 32],
}

impl fmt::Debug for MemoryKeySource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("MemoryKeySource(<ephemeral-test-key>)")
    }
}

#[async_trait]
impl KeySource for MemoryKeySource {
    async fn as_sign(
        &self,
    ) -> Result<Box<dyn Sign>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        Ok(Box::new(Ed25519KeyPair::from_seed_unchecked(&self.seed)?))
    }

    async fn write(
        &self,
        _value: &str,
        _key_id_hex: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        Err("ephemeral test key source is read-only".into())
    }
}

pub(crate) struct RepositoryFixture {
    pub(crate) root: Vec<u8>,
    pub(crate) timestamp: Vec<u8>,
    pub(crate) snapshot: Vec<u8>,
    pub(crate) targets: Vec<u8>,
    pub(crate) pointer: Vec<u8>,
    pub(crate) manifest: Vec<u8>,
    pub(crate) archive: Vec<u8>,
}

impl RepositoryFixture {
    pub(crate) fn responses(&self) -> HashMap<String, ScriptedResponse> {
        HashMap::from([
            ("2.root.json".to_string(), ScriptedResponse::NotFound),
            (
                "timestamp.json".to_string(),
                ScriptedResponse::Bytes(self.timestamp.clone()),
            ),
            (
                "snapshot.json".to_string(),
                ScriptedResponse::Bytes(self.snapshot.clone()),
            ),
            (
                "targets.json".to_string(),
                ScriptedResponse::Bytes(self.targets.clone()),
            ),
        ])
    }
}

pub(crate) async fn build_static_normalization_corpus(
) -> Result<[Vec<u8>; 4], Box<dyn std::error::Error + Send + Sync>> {
    let rng = SystemRandom::new();
    let keys: Vec<Box<dyn KeySource>> = vec![Box::new(MemoryKeySource { seed: [0x45; 32] })];
    let root = root_for_key(1, keys[0].as_sign().await?.tuf_key())?;
    let signed_root =
        SignedRole::new(root.clone(), &KeyHolder::Root(root.clone()), &keys, &rng).await?;

    // Reuse the normal builder for target descriptors, then deliberately
    // rebuild the complete lower-role chain from deterministic envelope bytes.
    let fixture = build_repository(2).await?;
    let parsed_targets: Signed<Targets> = serde_json::from_slice(&fixture.targets)?;
    let signed_targets = SignedRole::new(
        parsed_targets.signed,
        &KeyHolder::Root(root.clone()),
        &keys,
        &rng,
    )
    .await?;
    let targets = deterministic_json(signed_targets.signed())?;

    let version = NonZeroU64::new(2).expect("two is nonzero");
    let mut snapshot = Snapshot::new("1.0.0".to_string(), version, future());
    snapshot
        .meta
        .insert("targets.json".to_string(), metafile(version, &targets));
    let signed_snapshot =
        SignedRole::new(snapshot, &KeyHolder::Root(root.clone()), &keys, &rng).await?;
    let snapshot = deterministic_json(signed_snapshot.signed())?;

    let mut timestamp = TufTimestamp::new("1.0.0".to_string(), version, future());
    timestamp
        .meta
        .insert("snapshot.json".to_string(), metafile(version, &snapshot));
    let signed_timestamp = SignedRole::new(timestamp, &KeyHolder::Root(root), &keys, &rng).await?;
    let timestamp_value = serde_json::to_value(signed_timestamp.signed())?;
    let mut timestamp = serde_json::to_vec_pretty(&timestamp_value)?;
    timestamp.push(b'\n');

    Ok([
        deterministic_json(signed_root.signed())?,
        timestamp,
        snapshot,
        targets,
    ])
}

fn deterministic_json<T: serde::Serialize>(
    value: &T,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    Ok(serde_json::to_vec(&serde_json::to_value(value)?)?)
}

fn metafile(version: NonZeroU64, bytes: &[u8]) -> Metafile {
    Metafile {
        length: Some(bytes.len() as u64),
        hashes: Some(Hashes {
            sha256: sha256(bytes).to_vec().into(),
            _extra: HashMap::new(),
        }),
        version,
        _extra: HashMap::new(),
    }
}

pub(crate) async fn build_repository(
    metadata_version: u64,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_repository_with_policy(metadata_version, false, future(), ArtifactPolicy::Normal).await
}

pub(crate) async fn build_pointer_manifest_mismatch_repository(
    metadata_version: u64,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_repository_with_policy(
        metadata_version,
        false,
        future(),
        ArtifactPolicy::PointerManifestMismatch,
    )
    .await
}

pub(crate) async fn build_embedded_manifest_mismatch_repository(
    metadata_version: u64,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_repository_with_policy(
        metadata_version,
        false,
        future(),
        ArtifactPolicy::EmbeddedManifestMismatch,
    )
    .await
}

pub(crate) async fn build_special_archive_mode_repository(
    metadata_version: u64,
    mode: u32,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_repository_with_policy(
        metadata_version,
        false,
        future(),
        ArtifactPolicy::ArchiveEntryMode(mode),
    )
    .await
}

pub(crate) async fn build_case_collision_manifest_repository(
    metadata_version: u64,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_repository_with_policy(
        metadata_version,
        false,
        future(),
        ArtifactPolicy::CaseCollision,
    )
    .await
}

pub(crate) async fn build_non_ascii_manifest_repository(
    metadata_version: u64,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_repository_with_policy(
        metadata_version,
        false,
        future(),
        ArtifactPolicy::NonAsciiPath,
    )
    .await
}

pub(crate) async fn build_threshold_repository(
    metadata_version: u64,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    let temp = TempDir::new()?;
    let rng = SystemRandom::new();
    let keys: Vec<Box<dyn KeySource>> = vec![
        Box::new(MemoryKeySource { seed: [0x51; 32] }),
        Box::new(MemoryKeySource { seed: [0x52; 32] }),
    ];
    let tuf_keys = vec![
        keys[0].as_sign().await?.tuf_key(),
        keys[1].as_sign().await?.tuf_key(),
    ];
    let root = root_for_threshold(1, tuf_keys, 2)?;
    let signed_root = SignedRole::new(root.clone(), &KeyHolder::Root(root), &keys, &rng).await?;
    let root_path = temp.path().join("1.root.json");
    tokio::fs::write(&root_path, signed_root.buffer()).await?;
    build_lower_repository(
        &temp,
        &root_path,
        "1.root.json",
        &signed_root,
        &keys,
        metadata_version,
        false,
        future(),
        ArtifactPolicy::Normal,
    )
    .await
}

pub(crate) async fn build_insufficient_threshold_repository(
    metadata_version: u64,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    let mut fixture = build_threshold_repository(metadata_version).await?;
    rewrite_timestamp_signatures(&mut fixture.timestamp, SignatureMutation::KeepFirst)?;
    Ok(fixture)
}

pub(crate) async fn build_duplicate_signature_repository(
    metadata_version: u64,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    let mut fixture = build_threshold_repository(metadata_version).await?;
    rewrite_timestamp_signatures(&mut fixture.timestamp, SignatureMutation::DuplicateFirst)?;
    Ok(fixture)
}

pub(crate) async fn build_wrong_role_repository_pair(
    metadata_version: u64,
) -> Result<(RepositoryFixture, RepositoryFixture), Box<dyn std::error::Error + Send + Sync>> {
    let rng = SystemRandom::new();
    let root_key: Vec<Box<dyn KeySource>> = vec![Box::new(MemoryKeySource { seed: [0x45; 32] })];
    let lower_key: Vec<Box<dyn KeySource>> = vec![Box::new(MemoryKeySource { seed: [0x59; 32] })];
    let root_tuf_key = root_key[0].as_sign().await?.tuf_key();
    let lower_tuf_key = lower_key[0].as_sign().await?.tuf_key();
    let root_key_id = root_tuf_key.key_id()?.clone();
    let lower_key_id = lower_tuf_key.key_id()?.clone();
    let root_role = RoleKeys {
        keyids: vec![root_key_id.clone()],
        threshold: NonZeroU64::new(1).expect("one is nonzero"),
        _extra: HashMap::new(),
    };
    let lower_role = RoleKeys {
        keyids: vec![lower_key_id.clone()],
        threshold: NonZeroU64::new(1).expect("one is nonzero"),
        _extra: HashMap::new(),
    };
    let trusted_root = Root {
        spec_version: "1.0.0".to_string(),
        consistent_snapshot: false,
        version: NonZeroU64::new(1).expect("one is nonzero"),
        expires: future(),
        keys: HashMap::from([(root_key_id, root_tuf_key), (lower_key_id, lower_tuf_key)]),
        roles: HashMap::from([
            (RoleType::Root, root_role),
            (RoleType::Timestamp, lower_role.clone()),
            (RoleType::Snapshot, lower_role.clone()),
            (RoleType::Targets, lower_role),
        ]),
        _extra: HashMap::new(),
    };
    let signed_trusted_root = SignedRole::new(
        trusted_root.clone(),
        &KeyHolder::Root(trusted_root),
        &root_key,
        &rng,
    )
    .await?;
    let valid_temp = TempDir::new()?;
    let valid_root_path = valid_temp.path().join("1.root.json");
    tokio::fs::write(&valid_root_path, signed_trusted_root.buffer()).await?;
    let valid = build_lower_repository(
        &valid_temp,
        &valid_root_path,
        "1.root.json",
        &signed_trusted_root,
        &lower_key,
        1,
        false,
        future(),
        ArtifactPolicy::Normal,
    )
    .await?;

    let wrong_temp = TempDir::new()?;
    let wrong_root = root_for_key(1, root_key[0].as_sign().await?.tuf_key())?;
    let signed_wrong_root = SignedRole::new(
        wrong_root.clone(),
        &KeyHolder::Root(wrong_root),
        &root_key,
        &rng,
    )
    .await?;
    let wrong_root_path = wrong_temp.path().join("1.root.json");
    tokio::fs::write(&wrong_root_path, signed_wrong_root.buffer()).await?;
    let mut wrong = build_lower_repository(
        &wrong_temp,
        &wrong_root_path,
        "1.root.json",
        &signed_wrong_root,
        &root_key,
        metadata_version,
        false,
        future(),
        ArtifactPolicy::Normal,
    )
    .await?;
    wrong.root = valid.root.clone();
    Ok((valid, wrong))
}

pub(crate) async fn build_delegated_repository(
    metadata_version: u64,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_repository_with_policy(metadata_version, true, future(), ArtifactPolicy::Normal).await
}

pub(crate) async fn build_expired_lower_repository(
    metadata_version: u64,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_repository_with_policy(metadata_version, false, past(), ArtifactPolicy::Normal).await
}

#[derive(Clone, Copy)]
enum ArtifactPolicy {
    Normal,
    PointerManifestMismatch,
    EmbeddedManifestMismatch,
    ArchiveEntryMode(u32),
    CaseCollision,
    NonAsciiPath,
}

async fn build_repository_with_policy(
    metadata_version: u64,
    include_empty_delegations: bool,
    lower_expires: Timestamp,
    artifact_policy: ArtifactPolicy,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    let temp = TempDir::new()?;
    let rng = SystemRandom::new();
    let key = MemoryKeySource {
        // Public, deterministic, fixture-only seed. It never leaves this
        // unpublished test crate and is not a release-signing credential.
        seed: [0x45; 32],
    };
    let keys: Vec<Box<dyn KeySource>> = vec![Box::new(key)];
    let tuf_key = keys[0].as_sign().await?.tuf_key();
    let key_id = tuf_key.key_id()?.clone();
    let role_keys = RoleKeys {
        keyids: vec![key_id.clone()],
        threshold: NonZeroU64::new(1).expect("one is nonzero"),
        _extra: HashMap::new(),
    };
    let roles = HashMap::from([
        (RoleType::Root, role_keys.clone()),
        (RoleType::Timestamp, role_keys.clone()),
        (RoleType::Snapshot, role_keys.clone()),
        (RoleType::Targets, role_keys),
    ]);
    let expires = future();
    let root = Root {
        spec_version: "1.0.0".to_string(),
        consistent_snapshot: false,
        version: NonZeroU64::new(1).expect("one is nonzero"),
        expires,
        keys: HashMap::from([(key_id, tuf_key)]),
        roles,
        _extra: HashMap::new(),
    };
    let signed_root = SignedRole::new(root.clone(), &KeyHolder::Root(root), &keys, &rng).await?;
    let root_path = temp.path().join("1.root.json");
    tokio::fs::write(&root_path, signed_root.buffer()).await?;
    build_lower_repository(
        &temp,
        &root_path,
        "1.root.json",
        &signed_root,
        &keys,
        metadata_version,
        include_empty_delegations,
        lower_expires,
        artifact_policy,
    )
    .await
}

pub(crate) async fn build_rotated_repository(
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_root_rotation_repository(2, RotationSignatures::Both).await
}

pub(crate) async fn build_old_only_root_rotation_repository(
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_root_rotation_repository(2, RotationSignatures::OldOnly).await
}

pub(crate) async fn build_new_only_root_rotation_repository(
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_root_rotation_repository(2, RotationSignatures::NewOnly).await
}

pub(crate) async fn build_skipped_root_rotation_repository(
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_root_rotation_repository(3, RotationSignatures::Both).await
}

pub(crate) async fn build_multi_root_rotation_repository(
) -> Result<(RepositoryFixture, Vec<u8>), Box<dyn std::error::Error + Send + Sync>> {
    let temp = TempDir::new()?;
    let rng = SystemRandom::new();
    let first_keys: Vec<Box<dyn KeySource>> = vec![Box::new(MemoryKeySource { seed: [0x45; 32] })];
    let second_keys: Vec<Box<dyn KeySource>> = vec![Box::new(MemoryKeySource { seed: [0x46; 32] })];
    let third_keys: Vec<Box<dyn KeySource>> = vec![Box::new(MemoryKeySource { seed: [0x47; 32] })];
    let first_root = root_for_key(1, first_keys[0].as_sign().await?.tuf_key())?;
    let second_root = root_for_key(2, second_keys[0].as_sign().await?.tuf_key())?;
    let third_root = root_for_key(3, third_keys[0].as_sign().await?.tuf_key())?;

    let second_old = SignedRole::new(
        second_root.clone(),
        &KeyHolder::Root(first_root),
        &first_keys,
        &rng,
    )
    .await?;
    let second_new = SignedRole::new(
        second_root.clone(),
        &KeyHolder::Root(second_root.clone()),
        &second_keys,
        &rng,
    )
    .await?;
    let signed_second = second_new.add_old_signatures(second_old.signed().signatures.clone())?;
    let second_bytes = signed_second.buffer().to_vec();

    let third_old = SignedRole::new(
        third_root.clone(),
        &KeyHolder::Root(second_root),
        &second_keys,
        &rng,
    )
    .await?;
    let third_new = SignedRole::new(
        third_root.clone(),
        &KeyHolder::Root(third_root),
        &third_keys,
        &rng,
    )
    .await?;
    let signed_third = third_new.add_old_signatures(third_old.signed().signatures.clone())?;
    let root_path = temp.path().join("3.root.json");
    tokio::fs::write(&root_path, signed_third.buffer()).await?;
    let fixture = build_lower_repository(
        &temp,
        &root_path,
        "3.root.json",
        &signed_third,
        &third_keys,
        3,
        false,
        future(),
        ArtifactPolicy::Normal,
    )
    .await?;
    Ok((fixture, second_bytes))
}

pub(crate) async fn build_multi_key_root_rotation_repository(
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_threshold_root_rotation_repository(ThresholdRotationSignatures::BothFull).await
}

pub(crate) async fn build_insufficient_old_root_threshold_repository(
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_threshold_root_rotation_repository(ThresholdRotationSignatures::InsufficientOld).await
}

pub(crate) async fn build_insufficient_new_root_threshold_repository(
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    build_threshold_root_rotation_repository(ThresholdRotationSignatures::InsufficientNew).await
}

#[derive(Clone, Copy)]
enum RotationSignatures {
    OldOnly,
    NewOnly,
    Both,
}

#[derive(Clone, Copy)]
enum ThresholdRotationSignatures {
    BothFull,
    InsufficientOld,
    InsufficientNew,
}

async fn build_threshold_root_rotation_repository(
    signatures: ThresholdRotationSignatures,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    let temp = TempDir::new()?;
    let rng = SystemRandom::new();
    let old_keys: Vec<Box<dyn KeySource>> = vec![
        Box::new(MemoryKeySource { seed: [0x51; 32] }),
        Box::new(MemoryKeySource { seed: [0x52; 32] }),
    ];
    let new_keys: Vec<Box<dyn KeySource>> = vec![
        Box::new(MemoryKeySource { seed: [0x61; 32] }),
        Box::new(MemoryKeySource { seed: [0x62; 32] }),
    ];
    let old_tuf_keys = vec![
        old_keys[0].as_sign().await?.tuf_key(),
        old_keys[1].as_sign().await?.tuf_key(),
    ];
    let new_tuf_keys = vec![
        new_keys[0].as_sign().await?.tuf_key(),
        new_keys[1].as_sign().await?.tuf_key(),
    ];
    let old_root = root_for_threshold(1, old_tuf_keys, 2)?;
    let new_root = root_for_threshold(2, new_tuf_keys, 2)?;
    let old_signers = if matches!(signatures, ThresholdRotationSignatures::InsufficientOld) {
        &old_keys[..1]
    } else {
        old_keys.as_slice()
    };
    let new_signers = if matches!(signatures, ThresholdRotationSignatures::InsufficientNew) {
        &new_keys[..1]
    } else {
        new_keys.as_slice()
    };
    let old_signature = SignedRole::new(
        new_root.clone(),
        &KeyHolder::Root(old_root),
        old_signers,
        &rng,
    )
    .await?;
    let new_signature = SignedRole::new(
        new_root.clone(),
        &KeyHolder::Root(new_root),
        new_signers,
        &rng,
    )
    .await?;
    let signed_root =
        new_signature.add_old_signatures(old_signature.signed().signatures.clone())?;
    let root_path = temp.path().join("2.root.json");
    tokio::fs::write(&root_path, signed_root.buffer()).await?;
    build_lower_repository(
        &temp,
        &root_path,
        "2.root.json",
        &signed_root,
        &new_keys,
        2,
        false,
        future(),
        ArtifactPolicy::Normal,
    )
    .await
}

async fn build_root_rotation_repository(
    root_version: u64,
    signatures: RotationSignatures,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    let temp = TempDir::new()?;
    let rng = SystemRandom::new();
    let old_keys: Vec<Box<dyn KeySource>> = vec![Box::new(MemoryKeySource { seed: [0x45; 32] })];
    let new_keys: Vec<Box<dyn KeySource>> = vec![Box::new(MemoryKeySource { seed: [0x46; 32] })];
    let old_root = root_for_key(1, old_keys[0].as_sign().await?.tuf_key())?;
    let new_root = root_for_key(root_version, new_keys[0].as_sign().await?.tuf_key())?;
    let old_signature = SignedRole::new(
        new_root.clone(),
        &KeyHolder::Root(old_root),
        &old_keys,
        &rng,
    )
    .await?;
    let new_signature = SignedRole::new(
        new_root.clone(),
        &KeyHolder::Root(new_root),
        &new_keys,
        &rng,
    )
    .await?;
    let signed_root = match signatures {
        RotationSignatures::OldOnly => old_signature,
        RotationSignatures::NewOnly => new_signature,
        RotationSignatures::Both => {
            new_signature.add_old_signatures(old_signature.signed().signatures.clone())?
        }
    };
    let root_filename = format!("{root_version}.root.json");
    let root_path = temp.path().join(&root_filename);
    tokio::fs::write(&root_path, signed_root.buffer()).await?;
    build_lower_repository(
        &temp,
        &root_path,
        &root_filename,
        &signed_root,
        &new_keys,
        2,
        false,
        future(),
        ArtifactPolicy::Normal,
    )
    .await
}

fn root_for_key(
    version: u64,
    tuf_key: tough::schema::key::Key,
) -> Result<Root, Box<dyn std::error::Error + Send + Sync>> {
    let key_id = tuf_key.key_id()?.clone();
    let role_keys = RoleKeys {
        keyids: vec![key_id.clone()],
        threshold: NonZeroU64::new(1).expect("one is nonzero"),
        _extra: HashMap::new(),
    };
    Ok(Root {
        spec_version: "1.0.0".to_string(),
        consistent_snapshot: false,
        version: NonZeroU64::new(version).ok_or("root version must be nonzero")?,
        expires: future(),
        keys: HashMap::from([(key_id, tuf_key)]),
        roles: HashMap::from([
            (RoleType::Root, role_keys.clone()),
            (RoleType::Timestamp, role_keys.clone()),
            (RoleType::Snapshot, role_keys.clone()),
            (RoleType::Targets, role_keys),
        ]),
        _extra: HashMap::new(),
    })
}

fn root_for_threshold(
    version: u64,
    tuf_keys: Vec<tough::schema::key::Key>,
    threshold: u64,
) -> Result<Root, Box<dyn std::error::Error + Send + Sync>> {
    let mut keys = HashMap::new();
    let mut keyids = Vec::new();
    for key in tuf_keys {
        let key_id = key.key_id()?.clone();
        keyids.push(key_id.clone());
        keys.insert(key_id, key);
    }
    let role_keys = RoleKeys {
        keyids,
        threshold: NonZeroU64::new(threshold).ok_or("threshold must be nonzero")?,
        _extra: HashMap::new(),
    };
    Ok(Root {
        spec_version: "1.0.0".to_string(),
        consistent_snapshot: false,
        version: NonZeroU64::new(version).ok_or("root version must be nonzero")?,
        expires: future(),
        keys,
        roles: HashMap::from([
            (RoleType::Root, role_keys.clone()),
            (RoleType::Timestamp, role_keys.clone()),
            (RoleType::Snapshot, role_keys.clone()),
            (RoleType::Targets, role_keys),
        ]),
        _extra: HashMap::new(),
    })
}

#[derive(Clone, Copy)]
enum SignatureMutation {
    KeepFirst,
    DuplicateFirst,
}

fn rewrite_timestamp_signatures(
    bytes: &mut Vec<u8>,
    mutation: SignatureMutation,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut value: serde_json::Value = serde_json::from_slice(bytes)?;
    let signatures = value
        .get_mut("signatures")
        .and_then(serde_json::Value::as_array_mut)
        .ok_or("timestamp signatures are missing")?;
    let first = signatures
        .first()
        .cloned()
        .ok_or("no timestamp signature")?;
    signatures.clear();
    signatures.push(first.clone());
    if matches!(mutation, SignatureMutation::DuplicateFirst) {
        signatures.push(first);
    }
    *bytes = serde_json::to_vec(&value)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn build_lower_repository(
    temp: &TempDir,
    root_path: &std::path::Path,
    root_filename: &str,
    signed_root: &SignedRole<Root>,
    keys: &[Box<dyn KeySource>],
    metadata_version: u64,
    include_empty_delegations: bool,
    lower_expires: Timestamp,
    artifact_policy: ArtifactPolicy,
) -> Result<RepositoryFixture, Box<dyn std::error::Error + Send + Sync>> {
    let rng = SystemRandom::new();
    let mut payloads = FIXTURE_PAYLOADS.to_vec();
    match artifact_policy {
        ArtifactPolicy::CaseCollision => payloads.push((
            "share/doc/hf2q/readme.md",
            0o644,
            b"case-colliding fixture documentation\n",
        )),
        ArtifactPolicy::NonAsciiPath => payloads.push((
            "share/doc/hf2q/Cafe\u{301}.md",
            0o644,
            b"non-ASCII fixture documentation\n",
        )),
        ArtifactPolicy::Normal
        | ArtifactPolicy::PointerManifestMismatch
        | ArtifactPolicy::EmbeddedManifestMismatch
        | ArtifactPolicy::ArchiveEntryMode(_) => {}
    }
    payloads.sort_unstable_by_key(|(path, _, _)| *path);
    let manifest = fixture_manifest(&payloads)?;
    let mut mismatched_manifest = manifest.clone();
    mismatched_manifest.push(b' ');
    let archive_manifest = match artifact_policy {
        ArtifactPolicy::EmbeddedManifestMismatch => mismatched_manifest.as_slice(),
        ArtifactPolicy::Normal
        | ArtifactPolicy::PointerManifestMismatch
        | ArtifactPolicy::ArchiveEntryMode(_)
        | ArtifactPolicy::CaseCollision
        | ArtifactPolicy::NonAsciiPath => manifest.as_slice(),
    };
    let mut archive = fixture_archive(archive_manifest, &payloads)?;
    if let ArtifactPolicy::ArchiveEntryMode(mode) = artifact_policy {
        rewrite_archive_entry_mode(&mut archive, "bin/hf2q", mode)?;
    }
    let pointer_manifest = match artifact_policy {
        ArtifactPolicy::PointerManifestMismatch => mismatched_manifest.as_slice(),
        ArtifactPolicy::Normal
        | ArtifactPolicy::EmbeddedManifestMismatch
        | ArtifactPolicy::ArchiveEntryMode(_)
        | ArtifactPolicy::CaseCollision
        | ArtifactPolicy::NonAsciiPath => manifest.as_slice(),
    };
    let pointer = pointer_bytes(pointer_manifest, &archive);
    let target_inputs = [
        (pointer_name(), pointer.as_slice()),
        (manifest_name(), manifest.as_slice()),
        (archive_name(), archive.as_slice()),
    ];

    let version = NonZeroU64::new(metadata_version).ok_or("metadata version must be nonzero")?;
    let mut target_map = HashMap::new();
    for (name, bytes) in target_inputs {
        let path = temp.path().join(name.replace('/', "__"));
        tokio::fs::write(&path, bytes).await?;
        target_map.insert(TargetName::new(name)?, Target::from_path(path).await?);
    }
    let targets = Targets {
        spec_version: "1.0.0".to_string(),
        version,
        expires: lower_expires,
        targets: target_map,
        delegations: include_empty_delegations.then(Delegations::new),
        _extra: HashMap::new(),
    };
    let signed_targets = SignedRole::new(
        targets,
        &KeyHolder::Root(signed_root.signed().signed.clone()),
        keys,
        &rng,
    )
    .await?;
    let mut editor = RepositoryEditor::new(&root_path).await?;
    editor
        .targets(signed_targets.signed().clone())?
        .targets_version(version)?
        .targets_expires(lower_expires)?
        .snapshot_version(version)
        .snapshot_expires(lower_expires)
        .timestamp_version(version)
        .timestamp_expires(lower_expires);
    let signed = editor.sign(keys).await?;
    let metadata_dir = temp.path().join("metadata");
    signed.write(&metadata_dir).await?;

    Ok(RepositoryFixture {
        root: tokio::fs::read(metadata_dir.join(root_filename)).await?,
        timestamp: tokio::fs::read(metadata_dir.join("timestamp.json")).await?,
        snapshot: tokio::fs::read(metadata_dir.join("snapshot.json")).await?,
        targets: tokio::fs::read(metadata_dir.join("targets.json")).await?,
        pointer,
        manifest,
        archive,
    })
}

fn future() -> Timestamp {
    "2999-01-01T00:00:00Z"
        .parse()
        .expect("fixed future timestamp is valid")
}

fn past() -> Timestamp {
    "2000-01-01T00:00:00Z"
        .parse()
        .expect("fixed past timestamp is valid")
}

fn fixture_manifest(
    payloads: &[(&str, u32, &[u8])],
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let files: Vec<_> = payloads
        .iter()
        .map(|(path, mode, bytes)| {
            serde_json::json!({
                "path": path,
                "type": "regular",
                "size": bytes.len(),
                "mode": format!("{mode:04o}"),
                "sha256": hex::encode(sha256(bytes)),
            })
        })
        .collect();
    Ok(serde_json::to_vec(&serde_json::json!({
        "kind": "hf2q.release-manifest",
        "schema_version": 1,
        "package": "hf2q",
        "version": RELEASE_VERSION,
        "target": RELEASE_TARGET,
        "minimum_macos": "14.0",
        "source_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "channel": "stable",
        "code_signing": {
            "team_id": "A1B2C3D4E5",
            "identifier": "us.hf2q.cli",
            "certificate_common_name": "Developer ID Application: hf2q (A1B2C3D4E5)"
        },
        "compatibility": {
            "minimum_installer_protocol": 1,
            "minimum_updater_protocol": 1,
            "launcher_registry_schema": 1
        },
        "files": files,
        "non_system_dynamic_dependencies": []
    }))?)
}

fn rewrite_archive_entry_mode(
    archive: &mut [u8],
    target_name: &str,
    mode: u32,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if mode > u16::MAX.into() {
        return Err("ZIP mode exceeds the central-directory field".into());
    }
    let mut offset = 0_usize;
    while offset
        .checked_add(46)
        .is_some_and(|end| end <= archive.len())
    {
        if archive[offset..].starts_with(b"PK\x01\x02") {
            let name_len =
                u16::from_le_bytes([archive[offset + 28], archive[offset + 29]]) as usize;
            let extra_len =
                u16::from_le_bytes([archive[offset + 30], archive[offset + 31]]) as usize;
            let comment_len =
                u16::from_le_bytes([archive[offset + 32], archive[offset + 33]]) as usize;
            let name_start = offset + 46;
            let name_end = name_start
                .checked_add(name_len)
                .ok_or("ZIP central-directory name overflow")?;
            let entry_end = name_end
                .checked_add(extra_len)
                .and_then(|value| value.checked_add(comment_len))
                .ok_or("ZIP central-directory entry overflow")?;
            if entry_end > archive.len() {
                return Err("truncated ZIP central-directory entry".into());
            }
            if &archive[name_start..name_end] == target_name.as_bytes() {
                archive[offset + 38..offset + 42].copy_from_slice(&(mode << 16).to_le_bytes());
                return Ok(());
            }
            offset = entry_end;
        } else {
            offset += 1;
        }
    }
    Err(format!("ZIP entry {target_name} was not found").into())
}

fn fixture_archive(
    manifest: &[u8],
    payloads: &[(&str, u32, &[u8])],
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let mut archive = zip::ZipWriter::new(Cursor::new(Vec::new()));
    archive.start_file(
        "release-manifest.json",
        zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .unix_permissions(0o644),
    )?;
    archive.write_all(manifest)?;
    for (path, mode, bytes) in payloads {
        archive.start_file(
            *path,
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored)
                .unix_permissions(*mode),
        )?;
        archive.write_all(bytes)?;
    }
    Ok(archive.finish()?.into_inner())
}
