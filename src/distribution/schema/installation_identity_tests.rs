use sha2::{Digest, Sha256};

use super::*;

const VALID: &[u8] = include_bytes!("testdata/installation_identity_v1.json");
const VALID_SHA256: &str = "9cb4fdd5c4a2295cf1686c90ba7e0c4684e27c5b8514fe7a7cf82f7a55348496";

fn valid_value() -> serde_json::Value {
    serde_json::from_slice(VALID).expect("valid installation identity fixture")
}

fn parse_value(
    value: serde_json::Value,
) -> Result<InstallationIdentityV1, InstallationIdentityError> {
    let mut bytes = serde_json::to_vec(&value).expect("serialize hostile identity fixture");
    bytes.push(b'\n');
    InstallationIdentityV1::parse_and_validate(&bytes)
}

#[test]
fn canonical_identity_has_exact_v1_bytes() {
    assert_eq!(hex::encode(Sha256::digest(VALID)), VALID_SHA256);
    let identity = InstallationIdentityV1::parse_and_validate(VALID).expect("valid identity");
    assert_eq!(
        identity.installation_id().as_str(),
        "550e8400-e29b-41d4-a716-446655440000"
    );
    assert_eq!(identity.state_root().as_str(), "/Users/alice/.hf2q");
    assert_eq!(identity.to_deterministic_json().expect("encode"), VALID);
}

#[test]
fn constructor_emits_the_exact_golden() {
    let identity = InstallationIdentityV1::new(
        InstallationId::parse("550e8400-e29b-41d4-a716-446655440000".to_owned())
            .expect("installation ID"),
        AbsoluteInstallPath::parse("state_root", "/Users/alice/.hf2q".to_owned())
            .expect("state root"),
    );
    assert_eq!(identity.to_deterministic_json().expect("encode"), VALID);
}

#[test]
fn rejects_oversized_invalid_duplicate_unknown_trailing_and_noncanonical_json() {
    let oversized = vec![b' '; MAX_INSTALLATION_IDENTITY_BYTES + 1];
    assert!(matches!(
        InstallationIdentityV1::parse_and_validate(&oversized),
        Err(InstallationIdentityError::InputTooLarge { .. })
    ));

    for bytes in [vec![0xff], [VALID, b"{}"].concat()] {
        assert!(matches!(
            InstallationIdentityV1::parse_and_validate(&bytes),
            Err(InstallationIdentityError::Json { .. })
        ));
    }

    let duplicate = String::from_utf8(VALID.to_vec())
        .expect("UTF-8 fixture")
        .replacen(
            r#""schema_version":1,"#,
            r#""schema_version":1,"schema_version":1,"#,
            1,
        );
    assert!(matches!(
        InstallationIdentityV1::parse_and_validate(duplicate.as_bytes()),
        Err(InstallationIdentityError::Json { .. })
    ));

    let mut unknown = valid_value();
    unknown["created_at_unix_seconds"] = serde_json::json!(1);
    assert!(matches!(
        parse_value(unknown),
        Err(InstallationIdentityError::Json { .. })
    ));

    let pretty = serde_json::to_vec_pretty(&valid_value()).expect("pretty identity");
    assert!(matches!(
        InstallationIdentityV1::parse_and_validate(&pretty),
        Err(InstallationIdentityError::NonCanonicalEncoding)
    ));
    assert!(matches!(
        InstallationIdentityV1::parse_and_validate(VALID.strip_suffix(b"\n").unwrap()),
        Err(InstallationIdentityError::NonCanonicalEncoding)
    ));
}

#[test]
fn rejects_wrong_envelope_and_state_layout() {
    for (field, value) in [
        ("kind", serde_json::json!("hf2q.identity")),
        ("schema_version", serde_json::json!(2)),
        ("state_layout_schema", serde_json::json!(2)),
        ("package", serde_json::json!("other")),
    ] {
        let mut document = valid_value();
        document[field] = value;
        assert!(parse_value(document).is_err(), "accepted hostile {field}");
    }
}

#[test]
fn rejects_noncanonical_uuid_forms() {
    for installation_id in [
        "00000000-0000-0000-0000-000000000000",
        "550E8400-E29B-41D4-A716-446655440000",
        "550e8400e29b41d4a716446655440000",
        "{550e8400-e29b-41d4-a716-446655440000}",
        "urn:uuid:550e8400-e29b-41d4-a716-446655440000",
        "550e8400-e29b-11d4-a716-446655440000",
        "550e8400-e29b-41d4-c716-446655440000",
    ] {
        let mut document = valid_value();
        document["installation_id"] = serde_json::json!(installation_id);
        assert!(
            parse_value(document).is_err(),
            "accepted hostile UUID {installation_id}"
        );
    }
}

#[test]
fn rejects_noncanonical_or_cross_root_paths() {
    for root in [
        "",
        "/",
        "Users/alice/.hf2q",
        "/Users/alice/.hf2q/",
        "/Users//alice/.hf2q",
        "/Users/alice/../.hf2q",
        "/Users/alice/./.hf2q",
        "/Users\\alice\\.hf2q",
        "/Users/alice/\u{0001}.hf2q",
    ] {
        let mut document = valid_value();
        document["state_root"] = serde_json::json!(root);
        assert!(
            parse_value(document).is_err(),
            "accepted hostile root {root:?}"
        );
    }

    let mut copied = valid_value();
    copied["state_root"] = serde_json::json!("/Users/bob/.hf2q");
    let parsed = parse_value(copied).expect("other canonical roots are structurally valid");
    assert_ne!(parsed.state_root().as_str(), "/Users/alice/.hf2q");
}

#[test]
fn maximum_escaped_root_stays_inside_the_proven_wire_bound() {
    // AbsoluteInstallPath permits at most 4096 UTF-8 bytes and 255 bytes per
    // component. Quotes are the worst legal JSON escaping case.
    let mut root = String::new();
    while root.len() + 256 <= 4096 {
        root.push('/');
        root.push_str(&"\"".repeat(255));
    }
    if root.len() < 4096 {
        root.push('/');
        root.push_str(&"\"".repeat(4096 - root.len()));
    }
    assert_eq!(root.len(), 4096);
    let identity = InstallationIdentityV1::new(
        InstallationId::parse("550e8400-e29b-41d4-a716-446655440000".to_owned())
            .expect("installation ID"),
        AbsoluteInstallPath::parse("state_root", root).expect("maximum state root"),
    );
    let bytes = identity.to_deterministic_json().expect("bounded identity");
    assert!(bytes.len() <= MAX_INSTALLATION_IDENTITY_BYTES);
}
