use super::*;

#[test]
fn only_root_not_found_can_advance_the_transcript() {
    let (_temp, authorization) = authorization();
    let anchor = EmbeddedTrustRoot::from_compiled(ROOT);
    let root = request(
        begin_from_anchor_for_test(
            &authorization,
            &anchor,
            [
                instant("2026-08-18T08:30:00Z"),
                instant("2026-08-18T08:30:01Z"),
            ],
        )
        .expect("verifier starts"),
        "2.root.json",
    );
    let timestamp = request(
        root.respond(MetadataResponse::ConfirmedNotFound)
            .expect("root chain terminates"),
        "timestamp.json",
    );
    assert!(matches!(
        timestamp.respond(MetadataResponse::ConfirmedNotFound),
        Err(TufVerifierError::RequiredMetadataMissing)
    ));
}

#[test]
fn hostile_json_and_size_limits_fail_before_tuf_parsing() {
    assert!(matches!(
        strict_json::validate(br#"{"signed":{},"signed":{}}"#, 128),
        Err(TufVerifierError::DuplicateJsonKey)
    ));
    assert!(matches!(
        strict_json::validate(br#"{} {}"#, 128),
        Err(TufVerifierError::MalformedMetadata)
    ));
    assert!(matches!(
        strict_json::validate(&vec![b' '; MAX_TIMESTAMP_BYTES + 1], MAX_TIMESTAMP_BYTES),
        Err(TufVerifierError::MetadataSize)
    ));
}

#[test]
fn exact_expiry_is_rejected_even_though_the_library_accepts_equality() {
    let parsed = profile::timestamp(TIMESTAMP).expect("signed timestamp profile");
    assert!(matches!(
        profile::require_fresh(&parsed.signed, instant("2999-01-01T00:00:00Z")),
        Err(TufVerifierError::ExpiredMetadata)
    ));
    profile::require_fresh(&parsed.signed, instant("2998-12-31T23:59:59.999999999Z"))
        .expect("one nanosecond before expiry remains fresh");
}

#[test]
fn even_an_empty_delegations_block_is_outside_the_v1_profile() {
    let mut value: serde_json::Value = serde_json::from_slice(TARGETS).expect("fixture JSON");
    value["signed"]["delegations"] = serde_json::json!({"keys":{},"roles":[]});
    let bytes = serde_json::to_vec(&value).expect("mutated JSON");
    assert!(matches!(
        profile::targets(&bytes),
        Err(TufVerifierError::MalformedMetadata)
    ));
}
