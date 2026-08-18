use crate::distribution::update_auth::tests::TARGETS;
use serde_json::json;

#[test]
fn hostile_profile_rejects_missing_pins_extra_roles_and_noncanonical_depth() {
    let (timestamp, _, _) = static_lower_roles(3, "2999-01-01T00:00:00Z");
    let mut missing_hash: serde_json::Value =
        serde_json::from_slice(&timestamp).expect("timestamp JSON");
    missing_hash["signed"]["meta"]["snapshot.json"]
        .as_object_mut()
        .expect("snapshot descriptor")
        .remove("hashes");
    assert!(matches!(
        profile::timestamp(&serde_json::to_vec(&missing_hash).expect("mutated JSON")),
        Err(TufVerifierError::MalformedMetadata)
    ));

    let mut uppercase_hash: serde_json::Value =
        serde_json::from_slice(&timestamp).expect("timestamp JSON");
    let digest = uppercase_hash["signed"]["meta"]["snapshot.json"]["hashes"]["sha256"]
        .as_str()
        .expect("SHA-256")
        .to_ascii_uppercase();
    uppercase_hash["signed"]["meta"]["snapshot.json"]["hashes"]["sha256"] = json!(digest);
    assert!(matches!(
        profile::timestamp(&serde_json::to_vec(&uppercase_hash).expect("mutated JSON")),
        Err(TufVerifierError::MalformedMetadata)
    ));

    let mut zero_length: serde_json::Value =
        serde_json::from_slice(&timestamp).expect("timestamp JSON");
    zero_length["signed"]["meta"]["snapshot.json"]["length"] = json!(0);
    assert!(matches!(
        profile::timestamp(&serde_json::to_vec(&zero_length).expect("mutated JSON")),
        Err(TufVerifierError::MalformedMetadata)
    ));

    let mut extra_meta: serde_json::Value =
        serde_json::from_slice(&timestamp).expect("timestamp JSON");
    extra_meta["signed"]["meta"]["targets.json"] = json!({
        "hashes": {"sha256": "00".repeat(32)},
        "length": 1,
        "version": 1
    });
    assert!(matches!(
        profile::timestamp(&serde_json::to_vec(&extra_meta).expect("mutated JSON")),
        Err(TufVerifierError::MalformedMetadata)
    ));

    let mut nested = String::new();
    for _ in 0..65 {
        nested.push_str("{\"x\":");
    }
    nested.push('0');
    for _ in 0..65 {
        nested.push('}');
    }
    assert!(matches!(
        strict_json::validate(nested.as_bytes(), nested.len()),
        Err(TufVerifierError::MalformedMetadata)
    ));
    assert!(matches!(
        strict_json::validate(&[0xff], 1),
        Err(TufVerifierError::MalformedMetadata)
    ));
    assert!(matches!(
        strict_json::validate(b"\xef\xbb\xbf{}", 5),
        Err(TufVerifierError::MalformedMetadata)
    ));
    assert!(matches!(
        strict_json::validate(br#"{"outer":{"key":1,"key":2}}"#, 64),
        Err(TufVerifierError::DuplicateJsonKey)
    ));
}

#[test]
fn cardinality_and_crypto_cpu_bounds_reject_maximum_plus_one() {
    let mut signatures: serde_json::Value = serde_json::from_slice(ROOT).expect("root JSON");
    signatures["signatures"] = serde_json::Value::Array(
        (0..65)
            .map(|index| json!({"keyid": format!("key-{index}"), "sig": ""}))
            .collect(),
    );
    assert!(matches!(
        profile::root(&serde_json::to_vec(&signatures).expect("signature-bound JSON")),
        Err(TufVerifierError::MalformedMetadata)
    ));

    let mut keys: serde_json::Value = serde_json::from_slice(ROOT).expect("root JSON");
    let key = keys["signed"]["keys"]
        .as_object()
        .expect("root keys")
        .values()
        .next()
        .expect("one root key")
        .clone();
    let key_map = keys["signed"]["keys"].as_object_mut().expect("root keys");
    key_map.clear();
    for index in 0..65 {
        key_map.insert(format!("key-{index}"), key.clone());
    }
    assert!(matches!(
        profile::root(&serde_json::to_vec(&keys).expect("key-bound JSON")),
        Err(TufVerifierError::MalformedMetadata)
    ));

    let mut targets: serde_json::Value = serde_json::from_slice(TARGETS).expect("targets JSON");
    let descriptor = targets["signed"]["targets"]
        .as_object()
        .expect("target inventory")
        .values()
        .next()
        .expect("one target descriptor")
        .clone();
    let inventory = targets["signed"]["targets"]
        .as_object_mut()
        .expect("target inventory");
    inventory.clear();
    for index in 0..4097 {
        inventory.insert(format!("release-{index}"), descriptor.clone());
    }
    let target_bytes = serde_json::to_vec(&targets).expect("target-bound JSON");
    assert!(target_bytes.len() < crate::distribution::update_auth::model::MAX_TARGETS_BYTES);
    assert!(matches!(
        profile::targets(&target_bytes),
        Err(TufVerifierError::MalformedMetadata)
    ));

    let mut long_name: serde_json::Value = serde_json::from_slice(TARGETS).expect("targets JSON");
    let inventory = long_name["signed"]["targets"]
        .as_object_mut()
        .expect("target inventory");
    let descriptor = inventory.values().next().expect("one descriptor").clone();
    inventory.clear();
    inventory.insert("x".repeat(513), descriptor);
    assert!(matches!(
        profile::targets(&serde_json::to_vec(&long_name).expect("long-name JSON")),
        Err(TufVerifierError::MalformedMetadata)
    ));

    let mut hashes: serde_json::Value = serde_json::from_slice(TARGETS).expect("targets JSON");
    let first = hashes["signed"]["targets"]
        .as_object_mut()
        .expect("target inventory")
        .values_mut()
        .next()
        .expect("one descriptor");
    first["hashes"] = json!({
        "sha256": "00".repeat(32),
        "sha384": "00".repeat(48),
        "sha512": "00".repeat(64),
        "blake2b": "00".repeat(64),
        "blake3": "00".repeat(32)
    });
    assert!(matches!(
        profile::targets(&serde_json::to_vec(&hashes).expect("hash-bound JSON")),
        Err(TufVerifierError::MalformedMetadata)
    ));
}

#[test]
fn wrong_role_duplicate_signature_and_mix_and_match_metadata_fail_closed() {
    let fixture = threshold_rotation(RotationSignatures::Complete);
    let (_temp, authorization) = authorization();
    let anchor = leaked_anchor(&fixture.anchor);
    let root = request(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T11:00:00Z"),
                instant("2026-08-18T11:00:01Z"),
            ],
        )
        .expect("threshold anchor starts"),
        "2.root.json",
    );
    let root_probe = request(
        root.respond(MetadataResponse::Found(
            fixture.roots[0].clone().into_boxed_slice(),
        ))
        .expect("root thresholds pass"),
        "3.root.json",
    );
    let timestamp = request(
        root_probe
            .respond(MetadataResponse::ConfirmedNotFound)
            .expect("root chain terminates"),
        "timestamp.json",
    );
    let mut wrong_role: serde_json::Value =
        serde_json::from_slice(&fixture.timestamp).expect("timestamp JSON");
    for (index, signature) in wrong_role["signatures"]
        .as_array_mut()
        .expect("timestamp signatures")
        .iter_mut()
        .enumerate()
    {
        signature["keyid"] = json!(format!("old-{}", if index == 0 { "a" } else { "b" }));
    }
    assert!(matches!(
        timestamp.respond(MetadataResponse::Found(
            serde_json::to_vec(&wrong_role)
                .expect("wrong-role JSON")
                .into_boxed_slice()
        )),
        Err(TufVerifierError::AuthenticationFailed)
    ));

    let mut duplicate: serde_json::Value =
        serde_json::from_slice(&fixture.timestamp).expect("timestamp JSON");
    let first = duplicate["signatures"][0].clone();
    duplicate["signatures"]
        .as_array_mut()
        .expect("timestamp signatures")
        .push(first);
    assert!(matches!(
        profile::timestamp(&serde_json::to_vec(&duplicate).expect("duplicate JSON")),
        Err(TufVerifierError::MalformedMetadata)
    ));

    let (_temp, authorization) = super::authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
    let root = request(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T11:00:00Z"),
                instant("2026-08-18T11:00:01Z"),
            ],
        )
        .expect("static anchor starts"),
        "2.root.json",
    );
    let timestamp = request(
        root.respond(MetadataResponse::ConfirmedNotFound)
            .expect("root chain terminates"),
        "timestamp.json",
    );
    let (timestamp_three, _, _) = static_lower_roles(3, "2999-01-01T00:00:00Z");
    let (_, snapshot_four, _) = static_lower_roles(4, "2999-01-01T00:00:00Z");
    let snapshot = request(
        timestamp
            .respond(MetadataResponse::Found(timestamp_three.into_boxed_slice()))
            .expect("timestamp three authenticates"),
        "snapshot.json",
    );
    assert!(matches!(
        snapshot.respond(MetadataResponse::Found(snapshot_four.into_boxed_slice())),
        Err(TufVerifierError::UnexpectedResponse)
    ));
}
use super::*;
