use aws_lc_rs::signature::{Ed25519KeyPair, KeyPair};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

const EXPIRES: &str = "2999-01-01T00:00:00Z";
pub(super) const STATIC_KEY_ID: &str =
    "f506cd7b600c9ae23efbe499cbe2d1cb81e5fafde8d32f40e68b86ef6f93e2a2";

pub(super) struct RepositoryFixture {
    pub(super) anchor: Vec<u8>,
    pub(super) roots: Vec<Vec<u8>>,
    pub(super) timestamp: Vec<u8>,
    pub(super) snapshot: Vec<u8>,
    pub(super) targets: Vec<u8>,
    pub(super) consistent_snapshot: bool,
    pub(super) metadata_version: u64,
}

#[derive(Clone, Copy)]
pub(super) enum RotationSignatures {
    Complete,
    MissingOld,
    MissingNew,
}

struct TestKey {
    id: String,
    pair: Ed25519KeyPair,
}

impl TestKey {
    fn seeded(id: impl Into<String>, seed: u8) -> Self {
        Self {
            id: id.into(),
            pair: Ed25519KeyPair::from_seed_unchecked(&[seed; 32])
                .expect("fixed Ed25519 test seed"),
        }
    }

    fn key_value(&self) -> Value {
        json!({
            "keytype": "ed25519",
            "scheme": "ed25519",
            "keyval": {"public": hex::encode(self.pair.public_key().as_ref())}
        })
    }
}

pub(super) fn threshold_rotation(signatures: RotationSignatures) -> RepositoryFixture {
    let old = [
        TestKey::seeded("old-a", 0x51),
        TestKey::seeded("old-b", 0x52),
    ];
    let new = [
        TestKey::seeded("new-a", 0x61),
        TestKey::seeded("new-b", 0x62),
    ];
    let old_refs: Vec<_> = old.iter().collect();
    let new_refs: Vec<_> = new.iter().collect();
    let anchor_signed = root_value(1, &old_refs, 2, false);
    let anchor = envelope(anchor_signed, &old_refs);
    let root_signed = root_value(2, &new_refs, 2, false);
    let mut root_signers = Vec::new();
    match signatures {
        RotationSignatures::Complete => root_signers.extend(old.iter()),
        RotationSignatures::MissingOld => root_signers.push(&old[0]),
        RotationSignatures::MissingNew => root_signers.extend(old.iter()),
    }
    match signatures {
        RotationSignatures::Complete | RotationSignatures::MissingOld => {
            root_signers.extend(new.iter());
        }
        RotationSignatures::MissingNew => root_signers.push(&new[0]),
    }
    let root = envelope(root_signed, &root_signers);
    let (timestamp, snapshot, targets) = lower_roles(2, EXPIRES, &new_refs);
    RepositoryFixture {
        anchor,
        roots: vec![root],
        timestamp,
        snapshot,
        targets,
        consistent_snapshot: false,
        metadata_version: 2,
    }
}

pub(super) fn multi_rotation() -> RepositoryFixture {
    let first = TestKey::seeded("root-one", 0x45);
    let second = TestKey::seeded("root-two", 0x46);
    let third = TestKey::seeded("root-three", 0x47);
    let anchor = envelope(root_value(1, &[&first], 1, false), &[&first]);
    let root_two = envelope(root_value(2, &[&second], 1, false), &[&first, &second]);
    let root_three = envelope(root_value(3, &[&third], 1, false), &[&second, &third]);
    let (timestamp, snapshot, targets) = lower_roles(3, EXPIRES, &[&third]);
    RepositoryFixture {
        anchor,
        roots: vec![root_two, root_three],
        timestamp,
        snapshot,
        targets,
        consistent_snapshot: false,
        metadata_version: 3,
    }
}

pub(super) fn successive_threshold_rotations(
) -> (RepositoryFixture, RepositoryFixture, RepositoryFixture) {
    let old = [
        TestKey::seeded("successive-old-a", 0x81),
        TestKey::seeded("successive-old-b", 0x82),
    ];
    let middle = [
        TestKey::seeded("successive-middle-a", 0x83),
        TestKey::seeded("successive-middle-b", 0x84),
    ];
    let new = [
        TestKey::seeded("successive-new-a", 0x85),
        TestKey::seeded("successive-new-b", 0x86),
    ];
    let old_refs: Vec<_> = old.iter().collect();
    let middle_refs: Vec<_> = middle.iter().collect();
    let new_refs: Vec<_> = new.iter().collect();
    let anchor = envelope(root_value(1, &old_refs, 2, false), &old_refs);
    let root_two = envelope(
        root_value(2, &middle_refs, 2, false),
        &old.iter().chain(middle.iter()).collect::<Vec<_>>(),
    );
    let root_three = envelope(
        root_value(3, &new_refs, 2, false),
        &middle.iter().chain(new.iter()).collect::<Vec<_>>(),
    );
    let (first_timestamp, first_snapshot, first_targets) = lower_roles(2, EXPIRES, &middle_refs);
    let (second_timestamp, second_snapshot, second_targets) = lower_roles(3, EXPIRES, &new_refs);
    let (rollback_timestamp, rollback_snapshot, rollback_targets) =
        lower_roles(2, EXPIRES, &new_refs);

    (
        RepositoryFixture {
            anchor: anchor.clone(),
            roots: vec![root_two.clone()],
            timestamp: first_timestamp,
            snapshot: first_snapshot,
            targets: first_targets,
            consistent_snapshot: false,
            metadata_version: 2,
        },
        RepositoryFixture {
            anchor: anchor.clone(),
            roots: vec![root_two.clone(), root_three.clone()],
            timestamp: second_timestamp,
            snapshot: second_snapshot,
            targets: second_targets,
            consistent_snapshot: false,
            metadata_version: 3,
        },
        RepositoryFixture {
            anchor,
            roots: vec![root_two, root_three],
            timestamp: rollback_timestamp,
            snapshot: rollback_snapshot,
            targets: rollback_targets,
            consistent_snapshot: false,
            metadata_version: 2,
        },
    )
}

pub(super) fn same_key_chain(rotations: usize, consistent_snapshot: bool) -> RepositoryFixture {
    same_key_chain_with_expiry(rotations, consistent_snapshot, EXPIRES)
}

pub(super) fn same_key_chain_with_expiry(
    rotations: usize,
    consistent_snapshot: bool,
    expires: &str,
) -> RepositoryFixture {
    let key = TestKey::seeded("lifetime-root", 0x71);
    let anchor = envelope(root_value(1, &[&key], 1, consistent_snapshot), &[&key]);
    let roots = (2..=rotations as u64 + 1)
        .map(|version| {
            envelope(
                root_value(version, &[&key], 1, consistent_snapshot),
                &[&key],
            )
        })
        .collect();
    let (timestamp, snapshot, targets) = lower_roles(2, expires, &[&key]);
    RepositoryFixture {
        anchor,
        roots,
        timestamp,
        snapshot,
        targets,
        consistent_snapshot,
        metadata_version: 2,
    }
}

pub(super) fn static_lower_roles(version: u64, expires: &str) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let key = TestKey::seeded(STATIC_KEY_ID, 0x45);
    // This is independent proof that the retained tough-generated fixture and
    // the deterministic signer agree about the published seed's public key.
    assert_eq!(
        hex::encode(key.pair.public_key().as_ref()),
        "6355691c178a8ff91007a7478afb955ef7352c63e7b25703984cf78b26e21a56"
    );
    lower_roles(version, expires, &[&key])
}

pub(super) fn same_key_lower_roles(version: u64, expires: &str) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let key = TestKey::seeded("lifetime-root", 0x71);
    lower_roles(version, expires, &[&key])
}

pub(super) fn anchor_at_version(version: u64) -> Vec<u8> {
    let key = TestKey::seeded("edge-root", 0x72);
    envelope(root_value(version, &[&key], 1, false), &[&key])
}

fn lower_roles(version: u64, expires: &str, signers: &[&TestKey]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let targets = envelope(
        json!({
            "_type": "targets",
            "expires": expires,
            "spec_version": "1.0.0",
            "targets": {},
            "version": version
        }),
        signers,
    );
    let snapshot = envelope(
        json!({
            "_type": "snapshot",
            "expires": expires,
            "meta": {"targets.json": descriptor(version, &targets)},
            "spec_version": "1.0.0",
            "version": version
        }),
        signers,
    );
    let timestamp = envelope(
        json!({
            "_type": "timestamp",
            "expires": expires,
            "meta": {"snapshot.json": descriptor(version, &snapshot)},
            "spec_version": "1.0.0",
            "version": version
        }),
        signers,
    );
    (timestamp, snapshot, targets)
}

fn descriptor(version: u64, bytes: &[u8]) -> Value {
    json!({
        "hashes": {"sha256": hex::encode(Sha256::digest(bytes))},
        "length": bytes.len(),
        "version": version
    })
}

fn root_value(
    version: u64,
    keys: &[&TestKey],
    threshold: usize,
    consistent_snapshot: bool,
) -> Value {
    let mut key_values = Map::new();
    let key_ids: Vec<_> = keys.iter().map(|key| key.id.clone()).collect();
    for key in keys {
        key_values.insert(key.id.clone(), key.key_value());
    }
    let binding = json!({"keyids": key_ids, "threshold": threshold});
    json!({
        "_type": "root",
        "consistent_snapshot": consistent_snapshot,
        "expires": EXPIRES,
        "keys": key_values,
        "roles": {
            "root": binding,
            "snapshot": binding,
            "targets": binding,
            "timestamp": binding
        },
        "spec_version": "1.0.0",
        "version": version
    })
}

fn envelope(signed: Value, signers: &[&TestKey]) -> Vec<u8> {
    let canonical =
        sigstore_tuf::canonical_json::to_canonical_bytes(&signed).expect("canonical test metadata");
    let signatures: Vec<_> = signers
        .iter()
        .map(|key| {
            json!({
                "keyid": key.id,
                "sig": hex::encode(key.pair.sign(&canonical).as_ref())
            })
        })
        .collect();
    serde_json::to_vec(&json!({"signatures": signatures, "signed": signed}))
        .expect("serialize signed test metadata")
}
