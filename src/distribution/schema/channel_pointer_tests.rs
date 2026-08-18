use super::super::target_name::MAX_TARGET_NAME_BYTES;
use super::*;

const VALID: &[u8] = include_bytes!("testdata/channel_pointer_v1.json");

fn valid_value() -> serde_json::Value {
    serde_json::from_slice(VALID).expect("valid channel pointer fixture")
}

fn parse_value(value: serde_json::Value) -> Result<ChannelPointerV1, ChannelPointerError> {
    ChannelPointerV1::parse_and_validate(
        &serde_json::to_vec(&value).expect("serialize hostile pointer fixture"),
    )
}

#[test]
fn canonical_pointer_has_exact_v1_bytes() {
    let pointer = ChannelPointerV1::parse_and_validate(VALID).expect("valid pointer");
    assert_eq!(pointer.channel().as_str(), "stable");
    assert_eq!(pointer.version().as_str(), "1.2.3");
    assert_eq!(pointer.target().as_str(), "aarch64-apple-darwin");
    assert_eq!(pointer.manifest().length(), 1234);
    assert_eq!(pointer.archive().length(), 5678);
    assert_eq!(
        pointer.to_deterministic_json().expect("encode pointer"),
        VALID
    );
}

#[test]
fn pointer_constructor_derives_names_instead_of_accepting_paths() {
    let version = ReleaseVersion::parse_stable("version", "1.2.3".into()).expect("version");
    let pointer = ChannelPointerV1::new(
        version,
        TargetTriple::Aarch64AppleDarwin,
        1234,
        Sha256Digest::parse("manifest.sha256", "a".repeat(64)).expect("digest"),
        5678,
        Sha256Digest::parse("archive.sha256", "b".repeat(64)).expect("digest"),
    )
    .expect("construct pointer");
    assert_eq!(pointer.to_deterministic_json().expect("encode"), VALID);
}

#[test]
fn rejects_oversized_invalid_duplicate_unknown_and_trailing_json() {
    let oversized = vec![b' '; MAX_CHANNEL_POINTER_BYTES + 1];
    assert!(matches!(
        ChannelPointerV1::parse_and_validate(&oversized),
        Err(ChannelPointerError::InputTooLarge { .. })
    ));

    for bytes in [vec![0xff], [VALID, b"{}"].concat()] {
        assert!(matches!(
            ChannelPointerV1::parse_and_validate(&bytes),
            Err(ChannelPointerError::Json { .. })
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
        ChannelPointerV1::parse_and_validate(duplicate.as_bytes()),
        Err(ChannelPointerError::Json { .. })
    ));

    let mut unknown = valid_value();
    unknown["manifest"]["custom"] = serde_json::json!(null);
    assert!(matches!(
        parse_value(unknown),
        Err(ChannelPointerError::Json { .. })
    ));
}

#[test]
fn rejects_v0_and_wrong_fixed_identity() {
    let v0 = br#"{"schema_version":0,"channel":"stable","version":"1.2.3","target":"aarch64-apple-darwin"}"#;
    assert!(ChannelPointerV1::parse_and_validate(v0).is_err());

    for (field, value) in [
        ("kind", serde_json::json!("hf2q.channel-pointer")),
        ("schema_version", serde_json::json!(2)),
        ("package", serde_json::json!("other")),
        ("repository_id", serde_json::json!("other")),
        ("channel", serde_json::json!("nightly")),
        ("target", serde_json::json!("x86_64-apple-darwin")),
    ] {
        let mut document = valid_value();
        document[field] = value;
        assert!(parse_value(document).is_err(), "accepted hostile {field}");
    }
}

#[test]
fn rejects_noncanonical_or_unstable_version_and_derived_name_mismatch() {
    for version in ["v1.2.3", "1.2", "01.2.3", "1.2.3-rc.1", "1.2.3+build"] {
        let mut document = valid_value();
        document["version"] = serde_json::json!(version);
        assert!(parse_value(document).is_err(), "accepted {version}");
    }

    for (section, name) in [
        (
            "manifest",
            "releases/v1.2.4/aarch64-apple-darwin/release-manifest.json",
        ),
        (
            "archive",
            "releases/v1.2.3/aarch64-apple-darwin/aaaaaaaa.hf2q-v1.2.3-aarch64-apple-darwin.zip",
        ),
    ] {
        let mut document = valid_value();
        document[section]["name"] = serde_json::json!(name);
        assert!(parse_value(document).is_err(), "accepted hostile {section}");
    }

    for name in [
        "",
        "/releases/v1.2.3/aarch64-apple-darwin/release-manifest.json",
        "../release-manifest.json",
        "https://example.invalid/release-manifest.json",
        "releases/v1.2.3/aarch64-apple-darwin/../release-manifest.json",
        "releases/v1.2.3/aarch64-apple-darwin/release\\manifest.json",
        "releases/v1.2.3/aarch64-apple-darwin/rélease-manifest.json",
    ] {
        let mut document = valid_value();
        document["manifest"]["name"] = serde_json::json!(name);
        assert!(parse_value(document).is_err(), "accepted {name:?}");
    }

    let mut overlong = valid_value();
    overlong["manifest"]["name"] = serde_json::json!("a".repeat(MAX_TARGET_NAME_BYTES + 1));
    assert!(parse_value(overlong).is_err());
}

#[test]
fn enforces_manifest_and_compressed_archive_bounds() {
    for (section, maximum) in [
        ("manifest", MAX_RELEASE_MANIFEST_BYTES as u64),
        ("archive", MAX_RELEASE_ARCHIVE_BYTES),
    ] {
        for invalid in [0, maximum + 1] {
            let mut document = valid_value();
            document[section]["length"] = serde_json::json!(invalid);
            assert!(
                parse_value(document).is_err(),
                "accepted {section}={invalid}"
            );
        }
        let mut boundary = valid_value();
        boundary[section]["length"] = serde_json::json!(maximum);
        assert!(parse_value(boundary).is_ok(), "rejected {section} boundary");
    }
}

#[test]
fn rejects_noncanonical_digests_and_sanitizes_hostile_errors() {
    for digest in ["a".to_owned(), "A".repeat(64), "g".repeat(64)] {
        let mut document = valid_value();
        document["archive"]["sha256"] = serde_json::json!(digest);
        assert!(parse_value(document).is_err());
    }

    let mut hostile = valid_value();
    hostile["kind"] = serde_json::json!("\u{1b}[31mPWN");
    let error = parse_value(hostile).expect_err("hostile kind must fail");
    assert!(!format!("{error}").contains('\u{1b}'));
    assert!(!format!("{error:?}").contains('\u{1b}'));
}

#[test]
fn logical_and_consistent_snapshot_names_are_canonical() {
    let version = ReleaseVersion::parse_stable("version", "1.2.3".into()).expect("version");
    let digest = Sha256Digest::parse("sha256", "c".repeat(64)).expect("digest");
    let pointer =
        LogicalTargetName::channel_pointer(UpdateChannel::Stable, TargetTriple::Aarch64AppleDarwin);
    let manifest = LogicalTargetName::release_manifest(&version, TargetTriple::Aarch64AppleDarwin);
    let archive = LogicalTargetName::release_archive(&version, TargetTriple::Aarch64AppleDarwin);

    assert_eq!(
        pointer.as_str(),
        "channels/stable/aarch64-apple-darwin.json"
    );
    assert_eq!(
        manifest.as_str(),
        "releases/v1.2.3/aarch64-apple-darwin/release-manifest.json"
    );
    assert_eq!(
        archive.as_str(),
        "releases/v1.2.3/aarch64-apple-darwin/hf2q-v1.2.3-aarch64-apple-darwin.zip"
    );
    assert_eq!(
        pointer.consistent_snapshot_name(&digest).as_str(),
        format!(
            "channels/stable/{}.aarch64-apple-darwin.json",
            digest.as_str()
        )
    );
    assert_eq!(
        manifest.consistent_snapshot_name(&digest).as_str(),
        format!(
            "releases/v1.2.3/aarch64-apple-darwin/{}.release-manifest.json",
            digest.as_str()
        )
    );
    assert_eq!(
        archive.consistent_snapshot_name(&digest).as_str(),
        format!(
            "releases/v1.2.3/aarch64-apple-darwin/{}.hf2q-v1.2.3-aarch64-apple-darwin.zip",
            digest.as_str()
        )
    );
    assert_eq!(
        archive.consistent_snapshot_name(&digest).basename(),
        format!("{}.hf2q-v1.2.3-aarch64-apple-darwin.zip", digest.as_str())
    );
}

#[test]
fn release_versions_have_numeric_semver_order() {
    let old = ReleaseVersion::parse_stable("version", "0.9.0".into()).expect("old version");
    let new = ReleaseVersion::parse_stable("version", "0.10.0".into()).expect("new version");
    assert!(old < new);
}
