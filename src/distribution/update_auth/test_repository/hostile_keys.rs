use super::*;

pub(in crate::distribution::update_auth) struct HostileRootProfileCase {
    pub(in crate::distribution::update_auth) label: &'static str,
    pub(in crate::distribution::update_auth) hostile_anchor: Vec<u8>,
    pub(in crate::distribution::update_auth) valid_anchor: Vec<u8>,
    pub(in crate::distribution::update_auth) hostile_successor: Vec<u8>,
}

struct DeclaredTestKey {
    declared_id: String,
    key: sigstore_tuf::Key,
    pair: Ed25519KeyPair,
}

pub(in crate::distribution::update_auth) fn hostile_root_profile_cases(
) -> Vec<HostileRootProfileCase> {
    [
        ("aliased-key-id", HostileKeyMutation::AliasedId),
        (
            "mismatched-keytype-and-scheme",
            HostileKeyMutation::MismatchedKeytype,
        ),
        ("uppercase-public-key", HostileKeyMutation::UppercasePublic),
        (
            "whitespace-wrapped-public-key",
            HostileKeyMutation::WhitespacePublic,
        ),
    ]
    .into_iter()
    .map(|(label, mutation)| hostile_root_profile_case(label, mutation))
    .collect()
}

#[derive(Clone, Copy)]
enum HostileKeyMutation {
    AliasedId,
    MismatchedKeytype,
    UppercasePublic,
    WhitespacePublic,
}

fn hostile_root_profile_case(
    label: &'static str,
    mutation: HostileKeyMutation,
) -> HostileRootProfileCase {
    let old = TestKey::seeded("canonical-old-root", 0xe1);
    let hostile_pair =
        Ed25519KeyPair::from_seed_unchecked(&[0xe2; 32]).expect("fixed hostile key seed");
    let mut hostile_key = sigstore_tuf::Key {
        keytype: "ed25519".to_owned(),
        scheme: "ed25519".to_owned(),
        keyval: sigstore_tuf::KeyVal {
            public: hex::encode(hostile_pair.public_key().as_ref()),
            extra: BTreeMap::new(),
        },
        extra: BTreeMap::new(),
    };
    match mutation {
        HostileKeyMutation::AliasedId => {}
        HostileKeyMutation::MismatchedKeytype => hostile_key.keytype = "rsa".to_owned(),
        HostileKeyMutation::UppercasePublic => {
            hostile_key.keyval.public.make_ascii_uppercase();
        }
        HostileKeyMutation::WhitespacePublic => {
            hostile_key.keyval.public = format!(" {} ", hostile_key.keyval.public);
        }
    }
    let declared_id = if matches!(mutation, HostileKeyMutation::AliasedId) {
        "00".repeat(32)
    } else {
        hostile_key
            .key_id()
            .expect("hostile key still has canonical JSON")
    };
    let hostile = DeclaredTestKey {
        declared_id,
        key: hostile_key,
        pair: hostile_pair,
    };

    let valid_anchor_signed = root_with_declared_key(1, &old.id, &old.key);
    let valid_anchor = envelope(valid_anchor_signed, &[&old]);
    let hostile_anchor_signed = root_with_declared_key(1, &hostile.declared_id, &hostile.key);
    let hostile_anchor = envelope_with_declared_signers(
        hostile_anchor_signed,
        &[(&hostile.declared_id, &hostile.pair)],
    );
    let hostile_successor_signed = root_with_declared_key(2, &hostile.declared_id, &hostile.key);
    let hostile_successor = envelope_with_declared_signers(
        hostile_successor_signed,
        &[(&old.id, &old.pair), (&hostile.declared_id, &hostile.pair)],
    );

    HostileRootProfileCase {
        label,
        hostile_anchor,
        valid_anchor,
        hostile_successor,
    }
}

fn root_with_declared_key(version: u64, declared_id: &str, key: &sigstore_tuf::Key) -> Value {
    let binding = json!({"keyids": [declared_id], "threshold": 1});
    let mut keys = Map::new();
    keys.insert(
        declared_id.to_owned(),
        serde_json::to_value(key).expect("serialize hostile key"),
    );
    json!({
        "_type": "root",
        "consistent_snapshot": false,
        "expires": EXPIRES,
        "keys": keys,
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

fn envelope_with_declared_signers(signed: Value, signers: &[(&str, &Ed25519KeyPair)]) -> Vec<u8> {
    let canonical =
        sigstore_tuf::canonical_json::to_canonical_bytes(&signed).expect("canonical hostile root");
    let signatures: Vec<_> = signers
        .iter()
        .map(|(declared_id, pair)| {
            json!({
                "keyid": declared_id,
                "sig": hex::encode(pair.sign(&canonical).as_ref())
            })
        })
        .collect();
    serde_json::to_vec(&json!({"signatures": signatures, "signed": signed}))
        .expect("serialize hostile root")
}
