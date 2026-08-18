use super::*;

const VALID: &[u8] = include_bytes!("testdata/release_manifest_v1.json");

fn valid_value() -> serde_json::Value {
    serde_json::from_slice(VALID).expect("valid fixture JSON")
}

fn parse_value(value: serde_json::Value) -> Result<ReleaseManifestV1, ReleaseManifestError> {
    ReleaseManifestV1::parse_and_validate(
        &serde_json::to_vec(&value).expect("serialize hostile fixture"),
    )
}

fn add_data_file(document: &mut serde_json::Value, path: &str) {
    let files = document["files"].as_array_mut().expect("files array");
    files.push(serde_json::json!({
        "path": path,
        "type": "regular",
        "size": 1,
        "mode": "0644",
        "sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    }));
    files.sort_by(|left, right| {
        left["path"]
            .as_str()
            .expect("left path")
            .cmp(right["path"].as_str().expect("right path"))
    });
}

#[test]
fn valid_fixture_has_stable_deterministic_bytes() {
    let manifest = ReleaseManifestV1::parse_and_validate(VALID).expect("valid manifest");
    assert_eq!(manifest.version().as_str(), "0.2.0");
    assert_eq!(manifest.target().as_str(), "aarch64-apple-darwin");
    assert_eq!(manifest.files().len(), 4);
    assert_eq!(manifest.payload_bytes(), 1_048_612);
    assert_eq!(manifest.code_signing().team_id(), "A1B2C3D4E5");
    assert_eq!(
        manifest.to_deterministic_json().expect("encode manifest"),
        VALID
    );
}

#[test]
fn rejects_oversized_input_before_json_parse() {
    let bytes = vec![b' '; MAX_RELEASE_MANIFEST_BYTES + 1];
    assert!(matches!(
        ReleaseManifestV1::parse_and_validate(&bytes),
        Err(ReleaseManifestError::InputTooLarge { .. })
    ));
}

#[test]
fn rejects_invalid_utf8_bom_and_trailing_documents() {
    for bytes in [
        vec![0xff],
        [b"\xef\xbb\xbf".as_slice(), VALID].concat(),
        [VALID, b"{}".as_slice()].concat(),
    ] {
        assert!(matches!(
            ReleaseManifestV1::parse_and_validate(&bytes),
            Err(ReleaseManifestError::Json { .. })
        ));
    }
}

#[test]
fn rejects_duplicate_and_unknown_json_fields() {
    let duplicate = String::from_utf8(VALID.to_vec())
        .expect("UTF-8 fixture")
        .replacen(
            r#""schema_version":1,"#,
            r#""schema_version":1,"schema_version":1,"#,
            1,
        );
    assert!(matches!(
        ReleaseManifestV1::parse_and_validate(duplicate.as_bytes()),
        Err(ReleaseManifestError::Json { .. })
    ));

    let mut unknown = valid_value();
    unknown["code_signing"]["surprise"] = serde_json::json!(true);
    assert!(matches!(
        parse_value(unknown),
        Err(ReleaseManifestError::Json { .. })
    ));
}

#[test]
fn hostile_discriminators_and_fields_cannot_inject_terminal_controls() {
    let mut kind = valid_value();
    kind["kind"] = serde_json::json!("\u{1b}[31mPWN");
    let kind_error = parse_value(kind).expect_err("hostile kind must fail");
    assert!(!format!("{kind_error}").contains('\u{1b}'));
    assert!(!format!("{kind_error:?}").contains('\u{1b}'));

    let mut unknown = valid_value();
    unknown
        .as_object_mut()
        .expect("manifest object")
        .insert("\u{1b}[31mPWN".into(), serde_json::json!(true));
    let field_error = parse_value(unknown).expect_err("hostile field must fail");
    assert!(!format!("{field_error}").contains('\u{1b}'));
    assert!(!format!("{field_error:?}").contains('\u{1b}'));
}

#[test]
fn rejects_wrong_kind_schema_and_package() {
    for (field, value) in [
        ("kind", serde_json::json!("other.release-manifest")),
        ("schema_version", serde_json::json!(2)),
        ("schema_version", serde_json::json!(0)),
        ("package", serde_json::json!("not-hf2q")),
    ] {
        let mut document = valid_value();
        document[field] = value;
        assert!(parse_value(document).is_err(), "accepted hostile {field}");
    }
}

#[test]
fn rejects_noncanonical_or_unstable_versions() {
    for version in ["1.2", "01.2.3", "1.2.3-beta.1", "1.2.3+build"] {
        let mut document = valid_value();
        document["version"] = serde_json::json!(version);
        assert!(parse_value(document).is_err(), "accepted {version}");
    }
    for minimum in ["14", "014.0", "14.", "14.0.0.1", "0.1"] {
        let mut document = valid_value();
        document["minimum_macos"] = serde_json::json!(minimum);
        assert!(parse_value(document).is_err(), "accepted {minimum}");
    }
}

#[test]
fn macos_versions_order_by_numeric_components() {
    let old = MacOsVersion::parse("minimum_macos", "9.0".into()).expect("macOS 9");
    let current = MacOsVersion::parse("minimum_macos", "14.0".into()).expect("macOS 14");
    let patch = MacOsVersion::parse("minimum_macos", "14.0.1".into()).expect("macOS 14 patch");

    assert!(old < current);
    assert!(current < patch);
    assert!(
        MacOsVersion::parse("minimum_macos", "14.0.0".into()).is_err(),
        "redundant zero patch would violate Eq/Ord consistency"
    );
    let nine = MacOsVersion::parse("minimum_macos", "14.9".into()).expect("macOS 14.9");
    let ten = MacOsVersion::parse("minimum_macos", "14.10".into()).expect("macOS 14.10");
    assert!(nine < ten);
}

#[test]
fn rejects_unsupported_target_channel_and_compatibility() {
    let mut target = valid_value();
    target["target"] = serde_json::json!("x86_64-apple-darwin");
    assert!(parse_value(target).is_err());

    let mut channel = valid_value();
    channel["channel"] = serde_json::json!("nightly");
    assert!(parse_value(channel).is_err());

    for field in [
        "minimum_installer_protocol",
        "minimum_updater_protocol",
        "launcher_registry_schema",
    ] {
        for required in [0, 2] {
            let mut compatibility = valid_value();
            compatibility["compatibility"][field] = serde_json::json!(required);
            assert!(
                parse_value(compatibility).is_err(),
                "accepted {field}={required}"
            );
        }
    }
}

#[test]
fn rejects_noncanonical_commit_and_digests() {
    for commit in ["a", &"A".repeat(40), &"g".repeat(40)] {
        let mut document = valid_value();
        document["source_commit"] = serde_json::json!(commit);
        assert!(parse_value(document).is_err());
    }
    for digest in ["a", &"A".repeat(64), &"g".repeat(64)] {
        let mut document = valid_value();
        document["files"][0]["sha256"] = serde_json::json!(digest);
        assert!(parse_value(document).is_err());
    }
}

#[test]
fn rejects_hostile_and_out_of_contract_paths() {
    for path in [
        "",
        "/bin/hf2q",
        "bin/../hf2q",
        "bin/./hf2q",
        "bin//hf2q",
        "bin\\hf2q",
        "C:/bin/hf2q",
        "release-manifest.json",
        "share/other/file",
        "share/doc/hf2q/",
        "share/doc/hf2q/has space.md",
        "share/doc/hf2q/café.md",
    ] {
        let mut document = valid_value();
        document["files"][0]["path"] = serde_json::json!(path);
        assert!(parse_value(document).is_err(), "accepted {path:?}");
    }

    let mut overlong_component = valid_value();
    overlong_component["files"][2]["path"] =
        serde_json::json!(format!("share/doc/hf2q/{}", "a".repeat(256)));
    assert!(parse_value(overlong_component).is_err());
}

#[test]
fn rejects_duplicate_or_unsorted_inventory() {
    let mut duplicate = valid_value();
    duplicate["files"][1]["path"] = duplicate["files"][0]["path"].clone();
    assert!(matches!(
        parse_value(duplicate),
        Err(ReleaseManifestError::DuplicatePath(_))
    ));

    let mut unsorted = valid_value();
    unsorted["files"]
        .as_array_mut()
        .expect("files array")
        .swap(0, 1);
    assert!(matches!(
        parse_value(unsorted),
        Err(ReleaseManifestError::UnsortedInventory(_))
    ));

    for colliding_path in ["share/doc/hf2q/readme.md", "share/doc/hf2q/README.md/child"] {
        let mut collision = valid_value();
        collision["files"]
            .as_array_mut()
            .expect("files array")
            .insert(
                3,
                serde_json::json!({
                    "path": colliding_path,
                    "type": "regular",
                    "size": 1,
                    "mode": "0644",
                    "sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
                }),
            );
        assert!(matches!(
            parse_value(collision),
            Err(ReleaseManifestError::PathCollision { .. })
        ));
    }
}

#[test]
fn rejects_case_folded_directory_and_file_directory_collisions() {
    let mut directory_alias = valid_value();
    add_data_file(&mut directory_alias, "share/doc/hf2q/A/one.txt");
    add_data_file(&mut directory_alias, "share/doc/hf2q/a/two.txt");
    assert!(matches!(
        parse_value(directory_alias),
        Err(ReleaseManifestError::PathCollision { .. })
    ));

    let mut file_directory = valid_value();
    add_data_file(&mut file_directory, "share/doc/hf2q/guide");
    add_data_file(&mut file_directory, "share/doc/hf2q/guide/readme.txt");
    assert!(matches!(
        parse_value(file_directory),
        Err(ReleaseManifestError::PathCollision { .. })
    ));
}

#[test]
fn derived_directory_inventory_has_an_exact_bound() {
    let mut at_limit = BundleTreeInventory::default();
    for index in 0..MAX_BUNDLE_DIRECTORIES - 3 {
        let name = format!("preloaded-{index:04}");
        at_limit.directories.insert(name.clone(), name);
    }
    let path = BundlePath::parse("files[].path", "share/doc/hf2q/limit.txt".into())
        .expect("valid bundle path");
    at_limit.record(&path).expect("exact directory cap");
    assert_eq!(at_limit.directories.len(), MAX_BUNDLE_DIRECTORIES);

    let mut over_limit = BundleTreeInventory::default();
    for index in 0..MAX_BUNDLE_DIRECTORIES {
        let name = format!("preloaded-{index:04}");
        over_limit.directories.insert(name.clone(), name);
    }
    assert!(matches!(
        over_limit.record(&path),
        Err(ReleaseManifestError::TooManyEntries {
            collection: "derived directories",
            limit: MAX_BUNDLE_DIRECTORIES,
            actual,
        }) if actual == MAX_BUNDLE_DIRECTORIES + 1
    ));
}

#[test]
fn parser_enforces_the_derived_directory_cap() {
    fn manifest_with_last_branch_depth(last_depth: usize) -> serde_json::Value {
        let mut document = valid_value();
        let files = document["files"].as_array_mut().expect("files array");
        files.clear();
        files.push(serde_json::json!({
            "path": "bin/hf2q",
            "type": "regular",
            "size": 1,
            "mode": "0755",
            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        }));
        for index in 0..17 {
            let depth = if index == 16 { last_depth } else { 241 };
            files.push(serde_json::json!({
                "path": format!(
                    "share/doc/hf2q/g{index:04}/{}file.txt",
                    "a/".repeat(depth)
                ),
                "type": "regular",
                "size": 1,
                "mode": "0644",
                "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            }));
        }
        files.push(serde_json::json!({
            "path": "share/licenses/hf2q/LICENSE",
            "type": "regular",
            "size": 1,
            "mode": "0644",
            "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
        }));
        files.sort_by(|left, right| {
            left["path"]
                .as_str()
                .expect("left path")
                .cmp(right["path"].as_str().expect("right path"))
        });
        document
    }

    let at_limit =
        parse_value(manifest_with_last_branch_depth(217)).expect("exact derived-directory cap");
    assert_eq!(at_limit.derived_directories().len(), MAX_BUNDLE_DIRECTORIES);
    assert!(matches!(
        parse_value(manifest_with_last_branch_depth(218)),
        Err(ReleaseManifestError::TooManyEntries {
            collection: "derived directories",
            limit: MAX_BUNDLE_DIRECTORIES,
            actual,
        }) if actual == MAX_BUNDLE_DIRECTORIES + 1
    ));
}

#[test]
fn enforces_required_payloads_and_file_modes() {
    for path in [
        "bin/hf2q",
        "share/doc/hf2q/README.md",
        "share/licenses/hf2q/LICENSE-APACHE",
    ] {
        let mut document = valid_value();
        document["files"]
            .as_array_mut()
            .expect("files array")
            .retain(|file| file["path"] != path);
        assert!(matches!(
            parse_value(document),
            Err(ReleaseManifestError::MissingRequired(_))
        ));
    }

    let mut binary_mode = valid_value();
    binary_mode["files"][0]["mode"] = serde_json::json!("0644");
    assert!(parse_value(binary_mode).is_err());

    let mut doc_mode = valid_value();
    doc_mode["files"][2]["mode"] = serde_json::json!("0755");
    assert!(parse_value(doc_mode).is_err());

    let mut empty_binary = valid_value();
    empty_binary["files"][0]["size"] = serde_json::json!(0);
    assert!(parse_value(empty_binary).is_err());

    let mut link = valid_value();
    link["files"][0]["type"] = serde_json::json!("symlink");
    assert!(matches!(
        parse_value(link),
        Err(ReleaseManifestError::Json { .. })
    ));
}

#[test]
fn rejects_payload_size_overflow_and_limit() {
    let mut overflow = valid_value();
    overflow["files"][0]["size"] = serde_json::json!(u64::MAX);
    overflow["files"][1]["size"] = serde_json::json!(1);
    assert!(matches!(
        parse_value(overflow),
        Err(ReleaseManifestError::PayloadSizeOverflow)
    ));

    let mut too_large = valid_value();
    too_large["files"][0]["size"] = serde_json::json!(MAX_BUNDLE_PAYLOAD_BYTES + 1);
    assert!(matches!(
        parse_value(too_large),
        Err(ReleaseManifestError::PayloadTooLarge { .. })
    ));
}

#[test]
fn rejects_overlong_inventory() {
    let mut document = valid_value();
    let files = document["files"].as_array_mut().expect("files array");
    files.clear();
    files.push(serde_json::json!({
        "path": "bin/hf2q",
        "type": "regular",
        "size": 1,
        "mode": "0755",
        "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    }));
    for index in 0..MAX_BUNDLE_FILES {
        files.push(serde_json::json!({
            "path": format!("share/doc/hf2q/{index:04}.md"),
            "type": "regular",
            "size": 1,
            "mode": "0644",
            "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        }));
    }
    files.push(serde_json::json!({
        "path": "share/licenses/hf2q/LICENSE",
        "type": "regular",
        "size": 1,
        "mode": "0644",
        "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    }));
    assert!(matches!(
        parse_value(document),
        Err(ReleaseManifestError::TooManyEntries {
            collection: "files",
            ..
        })
    ));
}

#[test]
fn validates_exact_code_signing_identity() {
    let mut team = valid_value();
    team["code_signing"]["team_id"] = serde_json::json!("short");
    assert!(parse_value(team).is_err());

    let mut identifier = valid_value();
    identifier["code_signing"]["identifier"] = serde_json::json!("hf2q identifier");
    assert!(parse_value(identifier).is_err());

    let mut common_name = valid_value();
    common_name["code_signing"]["certificate_common_name"] =
        serde_json::json!("Developer ID Application: hf2q (WRONGTEAM0)");
    assert!(parse_value(common_name).is_err());
}

#[test]
fn validates_dynamic_dependency_inventory() {
    let mut missing_consumer = valid_value();
    missing_consumer["non_system_dynamic_dependencies"] = serde_json::json!([{
        "consumer": "libexec/serve_qwen36_opencode.sh",
        "install_name": "@rpath/libmlx.dylib"
    }]);
    assert!(parse_value(missing_consumer).is_err());

    let mut absolute = valid_value();
    absolute["non_system_dynamic_dependencies"] = serde_json::json!([{
        "consumer": "bin/hf2q",
        "install_name": "/opt/homebrew/lib/libmlx.dylib"
    }]);
    assert!(parse_value(absolute).is_err());

    let mut space = valid_value();
    space["non_system_dynamic_dependencies"] = serde_json::json!([{
        "consumer": "bin/hf2q",
        "install_name": "@rpath/lib mlx.dylib"
    }]);
    assert!(parse_value(space).is_err());

    let mut non_executable = valid_value();
    non_executable["non_system_dynamic_dependencies"] = serde_json::json!([{
        "consumer": "share/doc/hf2q/README.md",
        "install_name": "@rpath/libmlx.dylib"
    }]);
    assert!(parse_value(non_executable).is_err());

    let mut valid = valid_value();
    valid["non_system_dynamic_dependencies"] = serde_json::json!([{
        "consumer": "bin/hf2q",
        "install_name": "@rpath/libmlx.dylib"
    }]);
    assert!(parse_value(valid).is_ok());

    for names in [
        ["@rpath/libmlx.dylib", "@rpath/libmlx.dylib"],
        ["@rpath/libz.dylib", "@rpath/liba.dylib"],
    ] {
        let mut unordered = valid_value();
        unordered["non_system_dynamic_dependencies"] = serde_json::json!([
            {"consumer": "bin/hf2q", "install_name": names[0]},
            {"consumer": "bin/hf2q", "install_name": names[1]}
        ]);
        assert!(parse_value(unordered).is_err());
    }

    let mut over_limit = valid_value();
    over_limit["non_system_dynamic_dependencies"] = serde_json::Value::Array(
        (0..=MAX_DYNAMIC_DEPENDENCIES)
            .map(|index| {
                serde_json::json!({
                    "consumer": "bin/hf2q",
                    "install_name": format!("@rpath/lib{index:03}.dylib")
                })
            })
            .collect(),
    );
    assert!(matches!(
        parse_value(over_limit),
        Err(ReleaseManifestError::TooManyEntries {
            collection: "non-system dynamic dependencies",
            ..
        })
    ));
}
