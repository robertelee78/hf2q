use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::*;
use crate::distribution::schema::{ChannelPointerV1, ReleaseVersion, Sha256Digest, TargetTriple};

const MANIFEST: &[u8] = include_bytes!("../../schema/testdata/release_manifest_v1.json");
const OLD_ARCHIVE: &[u8] = b"deterministic hf2q v0.1.0 release archive fixture";
const NEW_ARCHIVE: &[u8] = b"deterministic hf2q v0.2.0 release archive fixture";

pub(in crate::distribution::update_auth) struct StableReleaseFixture {
    pub(in crate::distribution::update_auth) repository: RepositoryFixture,
    pub(in crate::distribution::update_auth) pointer: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub(in crate::distribution::update_auth) enum RetainedReleaseMutation {
    AppendOnly,
    AppendOnlySelectOld,
    RebindManifestDigest,
    RebindArchiveLength,
    RemoveManifest,
    RemovePair,
}

pub(in crate::distribution::update_auth) fn stable_release_repository(
    consistent_snapshot: bool,
) -> StableReleaseFixture {
    stable_release_repository_fixture(consistent_snapshot, EXPIRES, false)
}

pub(in crate::distribution::update_auth) fn stable_release_repository_with_expiry(
    consistent_snapshot: bool,
    expires: &str,
) -> StableReleaseFixture {
    stable_release_repository_fixture(consistent_snapshot, expires, false)
}

pub(in crate::distribution::update_auth) fn stable_release_repository_with_mismatched_pointer(
) -> StableReleaseFixture {
    stable_release_repository_fixture(true, EXPIRES, true)
}

fn stable_release_repository_fixture(
    consistent_snapshot: bool,
    expires: &str,
    mismatch_pointer_manifest: bool,
) -> StableReleaseFixture {
    let key = TestKey::seeded("stable-release-root", 0x91);
    let anchor = envelope(root_value(1, &[&key], 1, consistent_snapshot), &[&key]);
    let mut pointer = pointer("0.2.0", MANIFEST, NEW_ARCHIVE);
    if mismatch_pointer_manifest {
        let mut value: Value = serde_json::from_slice(&pointer).expect("pointer JSON");
        value["manifest"]["length"] = json!(MANIFEST.len() as u64 + 1);
        pointer = serde_json::to_vec(&value).expect("mismatched pointer JSON");
        pointer.push(b'\n');
    }
    let target_values = json!({
        "channels/stable/aarch64-apple-darwin.json": target_descriptor(&pointer),
        "releases/v0.2.0/aarch64-apple-darwin/release-manifest.json": target_descriptor(MANIFEST),
        "releases/v0.2.0/aarch64-apple-darwin/hf2q-v0.2.0-aarch64-apple-darwin.zip": target_descriptor(NEW_ARCHIVE)
    });
    let (timestamp, snapshot, targets) =
        lower_roles_with_targets(2, expires, &[&key], target_values);
    StableReleaseFixture {
        repository: RepositoryFixture {
            anchor,
            roots: Vec::new(),
            timestamp,
            snapshot,
            targets,
            consistent_snapshot,
            metadata_version: 2,
        },
        pointer,
    }
}

pub(in crate::distribution::update_auth) fn stable_release_successor_pair(
    mutation: RetainedReleaseMutation,
) -> (StableReleaseFixture, StableReleaseFixture) {
    let key = TestKey::seeded("stable-release-successor-root", 0x92);
    let anchor = envelope(root_value(1, &[&key], 1, true), &[&key]);
    let initial_pointer = pointer("0.1.0", MANIFEST, OLD_ARCHIVE);
    let selected_version = if matches!(mutation, RetainedReleaseMutation::AppendOnlySelectOld) {
        "0.1.0"
    } else {
        "0.2.0"
    };
    let successor_pointer = if selected_version == "0.1.0" {
        pointer("0.1.0", MANIFEST, OLD_ARCHIVE)
    } else {
        pointer("0.2.0", MANIFEST, NEW_ARCHIVE)
    };

    let initial_targets = json!({
        "channels/stable/aarch64-apple-darwin.json": target_descriptor(&initial_pointer),
        "releases/v0.1.0/aarch64-apple-darwin/release-manifest.json": target_descriptor(MANIFEST),
        "releases/v0.1.0/aarch64-apple-darwin/hf2q-v0.1.0-aarch64-apple-darwin.zip": target_descriptor(OLD_ARCHIVE)
    });
    let mut successor_targets = json!({
        "channels/stable/aarch64-apple-darwin.json": target_descriptor(&successor_pointer),
        "releases/v0.1.0/aarch64-apple-darwin/release-manifest.json": target_descriptor(MANIFEST),
        "releases/v0.1.0/aarch64-apple-darwin/hf2q-v0.1.0-aarch64-apple-darwin.zip": target_descriptor(OLD_ARCHIVE),
        "releases/v0.2.0/aarch64-apple-darwin/release-manifest.json": target_descriptor(MANIFEST),
        "releases/v0.2.0/aarch64-apple-darwin/hf2q-v0.2.0-aarch64-apple-darwin.zip": target_descriptor(NEW_ARCHIVE)
    });
    let targets = successor_targets
        .as_object_mut()
        .expect("successor target object");
    match mutation {
        RetainedReleaseMutation::AppendOnly | RetainedReleaseMutation::AppendOnlySelectOld => {}
        RetainedReleaseMutation::RebindManifestDigest => {
            targets
                .get_mut("releases/v0.1.0/aarch64-apple-darwin/release-manifest.json")
                .expect("old manifest")["hashes"]["sha256"] = json!("f".repeat(64));
        }
        RetainedReleaseMutation::RebindArchiveLength => {
            targets
                .get_mut(
                    "releases/v0.1.0/aarch64-apple-darwin/hf2q-v0.1.0-aarch64-apple-darwin.zip",
                )
                .expect("old archive")["length"] = json!(OLD_ARCHIVE.len() as u64 + 1);
        }
        RetainedReleaseMutation::RemoveManifest => {
            targets.remove("releases/v0.1.0/aarch64-apple-darwin/release-manifest.json");
        }
        RetainedReleaseMutation::RemovePair => {
            targets.remove("releases/v0.1.0/aarch64-apple-darwin/release-manifest.json");
            targets.remove(
                "releases/v0.1.0/aarch64-apple-darwin/hf2q-v0.1.0-aarch64-apple-darwin.zip",
            );
        }
    }

    let (first_timestamp, first_snapshot, first_targets) =
        lower_roles_with_targets(2, EXPIRES, &[&key], initial_targets);
    let (next_timestamp, next_snapshot, next_targets) =
        lower_roles_with_targets(3, EXPIRES, &[&key], successor_targets);
    (
        StableReleaseFixture {
            repository: RepositoryFixture {
                anchor: anchor.clone(),
                roots: Vec::new(),
                timestamp: first_timestamp,
                snapshot: first_snapshot,
                targets: first_targets,
                consistent_snapshot: true,
                metadata_version: 2,
            },
            pointer: initial_pointer,
        },
        StableReleaseFixture {
            repository: RepositoryFixture {
                anchor,
                roots: Vec::new(),
                timestamp: next_timestamp,
                snapshot: next_snapshot,
                targets: next_targets,
                consistent_snapshot: true,
                metadata_version: 3,
            },
            pointer: successor_pointer,
        },
    )
}

fn pointer(version: &str, manifest: &[u8], archive: &[u8]) -> Vec<u8> {
    ChannelPointerV1::new(
        ReleaseVersion::parse_stable("version", version.to_owned()).expect("version"),
        TargetTriple::Aarch64AppleDarwin,
        manifest.len() as u64,
        Sha256Digest::parse("manifest.sha256", hex::encode(Sha256::digest(manifest)))
            .expect("manifest digest"),
        archive.len() as u64,
        Sha256Digest::parse("archive.sha256", hex::encode(Sha256::digest(archive)))
            .expect("archive digest"),
    )
    .expect("pointer")
    .to_deterministic_json()
    .expect("pointer bytes")
}
