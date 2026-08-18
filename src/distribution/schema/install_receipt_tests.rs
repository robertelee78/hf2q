use super::*;
use crate::distribution::schema::{
    InstalledVersionMarkerV2, RecordedPreparationEvidenceV2, MAX_INSTALLED_VERSION_MARKER_BYTES,
};
use sha2::{Digest, Sha256};

const VALID_RECEIPT: &[u8] = include_bytes!("testdata/install_receipt_v1_standalone.json");
const INITIAL_RECEIPT_WITH_MARKER_V2: &[u8] =
    include_bytes!("testdata/install_receipt_v1_initial_marker_v2.json");
const VALID_MARKER: &[u8] = include_bytes!("testdata/installed_version_marker_v2.json");
const PREPUBLICATION_MARKER_V1: &[u8] = include_bytes!("testdata/installed_version_marker_v1.json");
const MARKER_V1_SHA256: &str = "82947959430d40e75101148568604dc08a4301964255e2cfe82d07c13272f78d";
const MARKER_V2_SHA256: &str = "6ad66bf5ef01270bf032422086f0ae16f144b2c535b4bd7d76382dd6274a4ad0";
const INITIAL_RECEIPT_WITH_MARKER_V2_SHA256: &str =
    "3796a3f1ceea31b624d3b7230f32c6f1b0d9fc96895d5527b2951a94351b9675";

fn receipt_value() -> serde_json::Value {
    serde_json::from_slice(VALID_RECEIPT).expect("valid receipt fixture JSON")
}

fn marker_value() -> serde_json::Value {
    serde_json::from_slice(VALID_MARKER).expect("valid marker fixture JSON")
}

fn parse_receipt(value: serde_json::Value) -> Result<InstallReceiptV1, InstallReceiptError> {
    InstallReceiptV1::parse_and_validate(
        &serde_json::to_vec(&value).expect("serialize hostile receipt fixture"),
    )
}

fn parse_marker(value: serde_json::Value) -> Result<InstalledVersionMarkerV2, InstallReceiptError> {
    InstalledVersionMarkerV2::parse_and_validate(
        &serde_json::to_vec(&value).expect("serialize hostile marker fixture"),
    )
}

fn digest(bytes: &[u8]) -> Sha256Digest {
    Sha256Digest::parse("test.sha256", hex::encode(Sha256::digest(bytes))).expect("test digest")
}

fn manager_receipt(owner: &str, route: Option<&str>) -> serde_json::Value {
    let mut value = receipt_value();
    value["installation_layout_schema"] = serde_json::Value::Null;
    value["installation_root"] = serde_json::json!("/opt/package-owner/hf2q");
    value["owner_family"] = serde_json::json!(owner);
    value["update_route"] = route.map_or(serde_json::Value::Null, |route| serde_json::json!(route));
    value["active"]
        .as_object_mut()
        .expect("active object")
        .remove("bundle");
    value["last_successful_transition"] = serde_json::Value::Null;
    value
}

fn bundle(sequence: u64, digest: char) -> serde_json::Value {
    let digest: String = std::iter::repeat(digest).take(64).collect();
    serde_json::json!({
        "release_manifest_sha256": digest,
        "archive_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "installed_version_marker_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "installation_sequence": sequence
    })
}

fn manager_bundle(digest: char) -> serde_json::Value {
    let digest: String = std::iter::repeat(digest).take(64).collect();
    serde_json::json!({
        "release_manifest_sha256": digest,
        "archive_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    })
}

fn release(version: &str, sequence: u64, digest: char) -> serde_json::Value {
    serde_json::json!({
        "version": version,
        "target": "aarch64-apple-darwin",
        "bundle": bundle(sequence, digest)
    })
}

#[test]
fn golden_receipt_and_marker_are_deterministic() {
    let receipt = InstallReceiptV1::parse_and_validate(VALID_RECEIPT).expect("valid receipt");
    assert_eq!(
        receipt.installation_id().as_str(),
        "550e8400-e29b-41d4-a716-446655440000"
    );
    assert_eq!(receipt.owner_family(), OwnerFamily::Standalone);
    assert_eq!(receipt.update_route(), Some(UpdateRoute::Standalone));
    assert_eq!(receipt.active().version().as_str(), "0.2.0");
    assert!(receipt.retained().is_empty());
    assert_eq!(
        receipt.to_deterministic_json().expect("encode receipt"),
        VALID_RECEIPT
    );

    let marker = InstalledVersionMarkerV2::parse_and_validate(VALID_MARKER).expect("valid marker");
    assert_eq!(marker.release().version().as_str(), "0.2.0");
    assert_eq!(marker.prepared_from().role_versions(), (1, 1, 1, 1));
    assert_eq!(marker.installation_sequence(), 1);
    assert_eq!(
        marker.to_deterministic_json().expect("encode marker"),
        VALID_MARKER
    );
    InstalledVersionMarkerV2::parse_and_validate_exact(VALID_MARKER, &digest(VALID_MARKER))
        .expect("exact marker bytes");
}

#[test]
fn first_standalone_record_builds_exact_cross_bound_marker_and_receipt() {
    let record = crate::distribution::schema::FirstStandaloneInstallRecord::build(
        InstallationId::parse("550e8400-e29b-41d4-a716-446655440000".to_owned())
            .expect("installation id"),
        AbsoluteInstallPath::parse("installation_root", "/Users/alice/.hf2q".to_owned())
            .expect("installation root"),
        ReleaseVersion::parse_stable("version", "0.2.0".to_owned()).expect("version"),
        TargetTriple::parse("target", "aarch64-apple-darwin".to_owned()).expect("target"),
        Sha256Digest::parse(
            "manifest",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        )
        .expect("manifest digest"),
        Sha256Digest::parse(
            "archive",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned(),
        )
        .expect("archive digest"),
        RecordedPreparationEvidenceV2::verified_update_metadata(1, 1, 1, 1)
            .expect("metadata evidence"),
        1_787_011_200,
    )
    .expect("first standalone record");

    assert_eq!(record.marker_bytes(), VALID_MARKER);
    assert_eq!(hex::encode(Sha256::digest(VALID_MARKER)), MARKER_V2_SHA256);
    assert_eq!(record.marker_sha256(), &digest(VALID_MARKER));
    assert_eq!(
        std::str::from_utf8(record.receipt_bytes()).expect("generated receipt UTF-8"),
        std::str::from_utf8(INITIAL_RECEIPT_WITH_MARKER_V2).expect("golden receipt UTF-8")
    );
    assert_eq!(
        hex::encode(Sha256::digest(INITIAL_RECEIPT_WITH_MARKER_V2)),
        INITIAL_RECEIPT_WITH_MARKER_V2_SHA256
    );
    assert_eq!(
        record.marker().prepared_from().role_versions(),
        (1, 1, 1, 1)
    );
    assert_eq!(record.receipt().active().version().as_str(), "0.2.0");
    assert_eq!(
        record
            .receipt()
            .active()
            .bundle()
            .and_then(RecordedBundleIdentityV1::installed_version_marker_sha256),
        Some(record.marker_sha256())
    );

    let recovered =
        crate::distribution::schema::FirstStandaloneInstallRecord::reconstruct_from_exact_marker(
            VALID_MARKER,
        )
        .expect("restart reconstruction from durable marker");
    assert_eq!(recovered.marker_sha256(), record.marker_sha256());
    assert_eq!(recovered.receipt_bytes(), record.receipt_bytes());
}

#[test]
fn dormant_prepublication_marker_v1_is_rejected_fail_closed() {
    assert_eq!(
        hex::encode(Sha256::digest(PREPUBLICATION_MARKER_V1)),
        MARKER_V1_SHA256
    );
    assert!(matches!(
        InstalledVersionMarkerV2::parse_and_validate(PREPUBLICATION_MARKER_V1),
        Err(InstallReceiptError::UnsupportedSchema {
            document: "installed-version marker",
            actual: 1
        })
    ));
    assert!(matches!(
        crate::distribution::schema::FirstStandaloneInstallRecord::reconstruct_from_exact_marker(
            PREPUBLICATION_MARKER_V1
        ),
        Err(InstallReceiptError::UnsupportedSchema {
            document: "installed-version marker",
            actual: 1
        })
    ));
}

#[test]
fn marker_v2_requires_complete_nonzero_preparation_evidence() {
    let mut missing = marker_value();
    missing
        .as_object_mut()
        .expect("marker object")
        .remove("prepared_from");
    assert!(parse_marker(missing).is_err());

    for field in [
        "root_version",
        "timestamp_version",
        "snapshot_version",
        "targets_version",
    ] {
        let mut zero = marker_value();
        zero["prepared_from"][field] = serde_json::json!(0);
        assert!(parse_marker(zero).is_err(), "accepted zero {field}");
    }

    let mut wrong_kind = marker_value();
    wrong_kind["prepared_from"]["kind"] = serde_json::json!("manual");
    assert!(parse_marker(wrong_kind).is_err());

    let mut unknown = marker_value();
    unknown["prepared_from"]["unexpected"] = serde_json::json!(1);
    assert!(parse_marker(unknown).is_err());

    let duplicate = String::from_utf8(VALID_MARKER.to_vec())
        .expect("UTF-8 marker fixture")
        .replace(
            "\"root_version\":1",
            "\"root_version\":1,\"root_version\":2",
        );
    assert!(InstalledVersionMarkerV2::parse_and_validate(duplicate.as_bytes()).is_err());
}

#[test]
fn marker_role_versions_map_to_the_same_receipt_fields() {
    let mut value = marker_value();
    for (field, version) in [
        ("root_version", 11),
        ("timestamp_version", 12),
        ("snapshot_version", 13),
        ("targets_version", 14),
    ] {
        value["prepared_from"][field] = serde_json::json!(version);
    }
    let marker_bytes = parse_marker(value)
        .expect("asymmetric marker")
        .to_deterministic_json()
        .expect("canonical marker");
    let record =
        crate::distribution::schema::FirstStandaloneInstallRecord::reconstruct_from_exact_marker(
            &marker_bytes,
        )
        .expect("reconstructed record");
    let receipt: serde_json::Value =
        serde_json::from_slice(record.receipt_bytes()).expect("receipt JSON");
    assert_eq!(
        receipt["last_successful_transition"]["authority"],
        serde_json::json!({
            "kind": "verified-update-metadata",
            "root_version": 11,
            "timestamp_version": 12,
            "snapshot_version": 13,
            "targets_version": 14
        })
    );
}

#[test]
fn first_standalone_reconstruction_rejects_noncanonical_or_noninitial_markers() {
    assert!(matches!(
        crate::distribution::schema::FirstStandaloneInstallRecord::reconstruct_from_exact_marker(
            &vec![b' '; MAX_INSTALLED_VERSION_MARKER_BYTES + 1]
        ),
        Err(InstallReceiptError::InputTooLarge {
            document: "installed-version marker",
            ..
        })
    ));

    let noncanonical =
        serde_json::to_vec_pretty(&marker_value()).expect("pretty marker representation");
    assert!(
        crate::distribution::schema::FirstStandaloneInstallRecord::reconstruct_from_exact_marker(
            &noncanonical
        )
        .is_err()
    );

    let sequence_two = String::from_utf8(VALID_MARKER.to_vec())
        .expect("UTF-8 marker fixture")
        .replace("\"installation_sequence\":1", "\"installation_sequence\":2");
    assert!(
        crate::distribution::schema::FirstStandaloneInstallRecord::reconstruct_from_exact_marker(
            sequence_two.as_bytes()
        )
        .is_err()
    );
}

#[test]
fn exact_marker_validation_never_hashes_reserialized_json() {
    let noncanonical =
        serde_json::to_vec_pretty(&marker_value()).expect("pretty marker representation");
    assert!(InstalledVersionMarkerV2::parse_and_validate(&noncanonical).is_ok());
    assert!(matches!(
        InstalledVersionMarkerV2::parse_and_validate_exact(&noncanonical, &digest(VALID_MARKER)),
        Err(InstallReceiptError::MarkerDigestMismatch)
    ));
    assert!(matches!(
        InstalledVersionMarkerV2::parse_and_validate_exact(&noncanonical, &digest(&noncanonical)),
        Err(InstallReceiptError::NonCanonicalMarkerEncoding)
    ));
}

#[test]
fn rejects_oversized_documents_before_json_parse() {
    assert!(matches!(
        InstallReceiptV1::parse_and_validate(&vec![b' '; MAX_INSTALL_RECEIPT_BYTES + 1]),
        Err(InstallReceiptError::InputTooLarge {
            document: "install receipt",
            ..
        })
    ));
    assert!(matches!(
        InstalledVersionMarkerV2::parse_and_validate(&vec![
            b' ';
            MAX_INSTALLED_VERSION_MARKER_BYTES + 1
        ]),
        Err(InstallReceiptError::InputTooLarge {
            document: "installed-version marker",
            ..
        })
    ));
    assert!(matches!(
        InstalledVersionMarkerV2::parse_and_validate_exact(
            &vec![b' '; MAX_INSTALLED_VERSION_MARKER_BYTES + 1],
            &digest(VALID_MARKER)
        ),
        Err(InstallReceiptError::InputTooLarge {
            document: "installed-version marker",
            ..
        })
    ));
}

#[test]
fn rejects_invalid_utf8_bom_and_trailing_documents() {
    for bytes in [
        vec![0xff],
        [b"\xef\xbb\xbf".as_slice(), VALID_RECEIPT].concat(),
        [VALID_RECEIPT, b"{}".as_slice()].concat(),
    ] {
        assert!(matches!(
            InstallReceiptV1::parse_and_validate(&bytes),
            Err(InstallReceiptError::Json { .. })
        ));
    }
}

#[test]
fn rejects_duplicate_and_unknown_fields_without_terminal_injection() {
    let duplicate = String::from_utf8(VALID_RECEIPT.to_vec())
        .expect("UTF-8 fixture")
        .replacen(
            r#""schema_version":1,"#,
            r#""schema_version":1,"schema_version":1,"#,
            1,
        );
    assert!(matches!(
        InstallReceiptV1::parse_and_validate(duplicate.as_bytes()),
        Err(InstallReceiptError::Json { .. })
    ));

    let mut unknown = receipt_value();
    unknown["active"]["\u{1b}[31mPWN"] = serde_json::json!(true);
    let error = parse_receipt(unknown).expect_err("unknown field must fail");
    assert!(!format!("{error}").contains('\u{1b}'));
    assert!(!format!("{error:?}").contains('\u{1b}'));
}

#[test]
fn rejects_hostile_envelopes_and_discriminators() {
    for (field, value) in [
        ("kind", serde_json::json!("other.receipt")),
        ("schema_version", serde_json::json!(0)),
        ("schema_version", serde_json::json!(2)),
        ("package", serde_json::json!("other")),
        ("state_layout_schema", serde_json::json!(0)),
        ("state_layout_schema", serde_json::json!(2)),
    ] {
        let mut document = receipt_value();
        document[field] = value;
        assert!(parse_receipt(document).is_err(), "accepted hostile {field}");
    }

    let mut kind = receipt_value();
    kind["kind"] = serde_json::json!("\u{1b}[31mPWN");
    let error = parse_receipt(kind).expect_err("hostile kind must fail");
    assert!(!format!("{error}").contains('\u{1b}'));
    assert!(!format!("{error:?}").contains('\u{1b}'));
}

#[test]
fn rejects_noncanonical_installation_ids() {
    for installation_id in [
        "not-a-uuid",
        "550E8400-E29B-41D4-A716-446655440000",
        "550e8400-e29b-11d4-a716-446655440000",
    ] {
        let mut document = receipt_value();
        document["installation_id"] = serde_json::json!(installation_id);
        assert!(parse_receipt(document).is_err());
    }
}

#[test]
fn rejects_hostile_or_noncanonical_roots() {
    for root in [
        "",
        "/",
        "~/.hf2q",
        "relative/hf2q",
        "/Users/alice/.hf2q/",
        "/Users//alice/.hf2q",
        "/Users/alice/../bob/.hf2q",
        "/Users/alice/./.hf2q",
        "/Users/alice/.hf2q\\escape",
        "/Users/alice/.hf2q\nother",
    ] {
        let mut document = receipt_value();
        document["state_root"] = serde_json::json!(root);
        assert!(parse_receipt(document).is_err(), "accepted root {root:?}");
    }

    let mut component = receipt_value();
    component["state_root"] = serde_json::json!(format!("/Users/{}", "a".repeat(256)));
    assert!(parse_receipt(component).is_err());

    let mut under_versions = manager_receipt("homebrew", Some("brew"));
    under_versions["state_root"] = serde_json::json!("/opt/package-owner/hf2q/versions/state");
    assert!(parse_receipt(under_versions).is_err());
}

#[test]
fn enforces_owner_route_and_layout_matrix() {
    for (owner, route) in [
        ("homebrew", Some("brew")),
        ("cargo-registry", Some("cargo-install")),
        ("cargo-registry", Some("cargo-binstall")),
        ("unknown/manual", None),
    ] {
        let parsed = parse_receipt(manager_receipt(owner, route)).expect("valid owner route");
        assert_eq!(parsed.owner_family().as_str(), owner);
        assert_eq!(parsed.update_route().map(UpdateRoute::as_str), route);
    }

    for (owner, route) in [
        ("standalone", None),
        ("homebrew", Some("cargo-install")),
        ("cargo-registry", Some("brew")),
        ("unknown/manual", Some("standalone")),
    ] {
        let mut document = manager_receipt(owner, route);
        if owner == "standalone" {
            document["installation_layout_schema"] = serde_json::json!(1);
            document["installation_root"] = document["state_root"].clone();
        }
        assert!(matches!(
            parse_receipt(document),
            Err(InstallReceiptError::OwnerRouteMismatch)
        ));
    }

    let mut reusable_consent = receipt_value();
    reusable_consent["update_route"] = serde_json::json!("confirmed-migration");
    assert!(parse_receipt(reusable_consent).is_err());
}

#[test]
fn standalone_requires_equal_roots_bundle_layout_and_transition() {
    for mutate in 0..4 {
        let mut document = receipt_value();
        match mutate {
            0 => document["installation_root"] = serde_json::json!("/Users/alice/other"),
            1 => document["installation_layout_schema"] = serde_json::Value::Null,
            2 => {
                document["active"]
                    .as_object_mut()
                    .expect("active object")
                    .remove("bundle");
            }
            3 => document["last_successful_transition"] = serde_json::Value::Null,
            _ => unreachable!(),
        }
        assert!(parse_receipt(document).is_err());
    }
}

#[test]
fn manager_and_manual_receipts_can_bind_bundles_but_not_standalone_state() {
    let mut retained = manager_receipt("homebrew", Some("brew"));
    retained["retained"] = serde_json::json!([release("0.1.0", 1, 'd')]);
    assert!(parse_receipt(retained).is_err());

    let mut bundle_claim = manager_receipt("homebrew", Some("brew"));
    bundle_claim["active"]["bundle"] = manager_bundle('d');
    assert!(parse_receipt(bundle_claim).is_ok());

    let mut marker_claim = manager_receipt("homebrew", Some("brew"));
    marker_claim["active"]["bundle"] = bundle(1, 'd');
    assert!(parse_receipt(marker_claim).is_err());

    let mut partial_marker = manager_receipt("homebrew", Some("brew"));
    partial_marker["active"]["bundle"] = manager_bundle('d');
    partial_marker["active"]["bundle"]["installed_version_marker_sha256"] =
        serde_json::json!("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    assert!(parse_receipt(partial_marker).is_err());

    let mut layout_claim = manager_receipt("unknown/manual", None);
    layout_claim["installation_layout_schema"] = serde_json::json!(1);
    assert!(parse_receipt(layout_claim).is_err());
}

#[test]
fn rejects_excess_duplicate_or_cross_target_retained_releases() {
    let mut too_many = receipt_value();
    too_many["retained"] = serde_json::json!([
        release("0.1.0", 2, 'd'),
        release("0.1.1", 3, 'e'),
        release("0.1.2", 4, 'f')
    ]);
    assert!(matches!(
        parse_receipt(too_many),
        Err(InstallReceiptError::TooManyRetained { .. })
    ));

    let mut duplicate = receipt_value();
    duplicate["retained"] = serde_json::json!([release("0.2.0", 2, 'd')]);
    assert!(matches!(
        parse_receipt(duplicate),
        Err(InstallReceiptError::DuplicateVersion(_))
    ));

    let mut sequence = receipt_value();
    sequence["retained"] = serde_json::json!([release("0.1.0", 1, 'd')]);
    assert!(parse_receipt(sequence).is_err());

    let mut target = receipt_value();
    target["retained"] = serde_json::json!([release("0.1.0", 2, 'd')]);
    target["retained"][0]["target"] = serde_json::json!("x86_64-apple-darwin");
    assert!(parse_receipt(target).is_err());
}

#[test]
fn validates_rollback_transition_and_retention_order() {
    let mut document = receipt_value();
    let newly_active = release("0.1.0", 1, 'd');
    let prior_active = release("0.2.0", 2, 'e');
    document["active"] = newly_active.clone();
    document["retained"] = serde_json::json!([prior_active.clone()]);
    document["last_successful_transition"] = serde_json::json!({
        "sequence": 3,
        "type": "rollback",
        "from": {"owner_family": "standalone", "release": prior_active},
        "to": {"owner_family": "standalone", "release": newly_active},
        "authority": {
            "kind": "retained-release",
            "release_manifest_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
        },
        "completed_at_unix_seconds": 1787011300
    });
    assert!(parse_receipt(document.clone()).is_ok());

    let mut replayed_sequence = document.clone();
    replayed_sequence["last_successful_transition"]["sequence"] = serde_json::json!(2);
    assert!(parse_receipt(replayed_sequence).is_err());

    document["retained"]
        .as_array_mut()
        .expect("retained")
        .clear();
    assert!(parse_receipt(document).is_err());
}

#[test]
fn validates_confirmed_migration_as_transition_not_route() {
    let mut document = receipt_value();
    let active = document["active"].clone();
    document["last_successful_transition"] = serde_json::json!({
        "sequence": 1,
        "type": "confirmed-migration",
        "from": {
            "owner_family": "unknown/manual",
            "release": {"version": "0.2.0", "target": "aarch64-apple-darwin"}
        },
        "to": {"owner_family": "standalone", "release": active},
        "authority": {
            "kind": "verified-update-metadata",
            "root_version": 1,
            "timestamp_version": 1,
            "snapshot_version": 1,
            "targets_version": 1
        },
        "completed_at_unix_seconds": 1787011300
    });
    assert!(parse_receipt(document.clone()).is_ok());

    document["last_successful_transition"]["from"]["owner_family"] = serde_json::json!("homebrew");
    assert!(parse_receipt(document).is_err());
}

#[test]
fn validates_manager_update_without_inventing_installer_history() {
    for (owner, route) in [
        ("homebrew", "brew"),
        ("cargo-registry", "cargo-install"),
        ("cargo-registry", "cargo-binstall"),
    ] {
        let mut document = manager_receipt(owner, Some(route));
        let active = document["active"].clone();
        document["last_successful_transition"] = serde_json::json!({
            "sequence": 7,
            "type": "update",
            "from": {
                "owner_family": owner,
                "release": {"version": "0.1.0", "target": "aarch64-apple-darwin"}
            },
            "to": {"owner_family": owner, "release": active},
            "authority": {"kind": "package-manager", "route": route},
            "completed_at_unix_seconds": 1787011300
        });
        assert!(parse_receipt(document.clone()).is_ok());

        document["last_successful_transition"]["authority"]["route"] =
            serde_json::json!("standalone");
        assert!(parse_receipt(document).is_err());
    }
}

#[test]
fn validates_standalone_update_retention_and_sequence() {
    let mut document = receipt_value();
    let active = release("0.3.0", 3, 'd');
    let prior_active = release("0.2.0", 2, 'e');
    document["active"] = active.clone();
    document["retained"] = serde_json::json!([prior_active.clone()]);
    document["last_successful_transition"] = serde_json::json!({
        "sequence": 3,
        "type": "update",
        "from": {"owner_family": "standalone", "release": prior_active},
        "to": {"owner_family": "standalone", "release": active},
        "authority": {
            "kind": "verified-update-metadata",
            "root_version": 2,
            "timestamp_version": 4,
            "snapshot_version": 4,
            "targets_version": 4
        },
        "completed_at_unix_seconds": 1787011400
    });
    assert!(parse_receipt(document.clone()).is_ok());

    document["retained"]
        .as_array_mut()
        .expect("retained")
        .clear();
    assert!(parse_receipt(document.clone()).is_err());

    document["retained"] = serde_json::json!([release("0.2.0", 2, 'e')]);
    document["last_successful_transition"]["sequence"] = serde_json::json!(2);
    assert!(parse_receipt(document).is_err());
}

#[test]
fn rejects_transition_endpoint_authority_and_floor_mismatches() {
    let mut destination = receipt_value();
    destination["last_successful_transition"]["to"]["release"]["version"] =
        serde_json::json!("0.1.0");
    assert!(parse_receipt(destination).is_err());

    let mut install_from = receipt_value();
    install_from["last_successful_transition"]["from"] = serde_json::json!({
        "owner_family": "standalone",
        "release": release("0.1.0", 2, 'd')
    });
    assert!(parse_receipt(install_from).is_err());

    let mut zero_floor = receipt_value();
    zero_floor["last_successful_transition"]["authority"]["targets_version"] = serde_json::json!(0);
    assert!(parse_receipt(zero_floor).is_err());

    let mut unknown_transition = manager_receipt("unknown/manual", None);
    unknown_transition["last_successful_transition"] =
        receipt_value()["last_successful_transition"].clone();
    assert!(parse_receipt(unknown_transition).is_err());
}

#[test]
fn validates_marker_envelope_identity_and_bounds() {
    for (field, value) in [
        ("kind", serde_json::json!("other.marker")),
        ("schema_version", serde_json::json!(1)),
        ("package", serde_json::json!("other")),
        ("installation_layout_schema", serde_json::json!(0)),
        ("installation_sequence", serde_json::json!(0)),
        ("installed_at_unix_seconds", serde_json::json!(0)),
    ] {
        let mut marker = marker_value();
        marker[field] = value;
        assert!(parse_marker(marker).is_err(), "accepted hostile {field}");
    }

    let mut root = marker_value();
    root["installation_root"] = serde_json::json!("../escape");
    assert!(parse_marker(root).is_err());

    let mut digest = marker_value();
    digest["release"]["archive_sha256"] = serde_json::json!("A".repeat(64));
    assert!(parse_marker(digest).is_err());

    let mut unknown = marker_value();
    unknown["release"]["owned_paths"] = serde_json::json!(["/Users/alice"]);
    assert!(matches!(
        parse_marker(unknown),
        Err(InstallReceiptError::Json { .. })
    ));
}
