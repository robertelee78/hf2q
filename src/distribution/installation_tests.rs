use std::ffi::OsStr;
use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::installation::{
    detect, detect_with_manifest_root, reconcile_cargo_uninstall, reconcile_cargo_update,
    CargoGitSelector, CargoGitSource, CargoInstallOptions, CargoSource, Installation,
    ManagerCommand, SourceProfile,
};

const MARKER: &[u8] = b"{\"kind\":\"hf2q.install-channel\",\"schema_version\":1,\"package\":\"hf2q\",\"channel\":\"standalone\"}\n";

fn executable(path: &Path) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, b"test hf2q executable").unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o555)).unwrap();
}

fn version_executable(path: &Path) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        path,
        format!(
            "#!/bin/sh\nprintf 'hf2q {}\\n'\n",
            env!("CARGO_PKG_VERSION")
        ),
    )
    .unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o555)).unwrap();
}

fn write_receipts(root: &Path, v1_id: Option<&str>, v2_id: Option<&str>) {
    fs::create_dir_all(root.join("bin")).unwrap();
    let v1_body = match v1_id {
        Some(id) => format!("[v1]\n\"{id}\" = [\"hf2q\"]\n"),
        None => "[v1]\n".to_owned(),
    };
    let v2_body = match v2_id {
        Some(id) => format!(
            "{{\"installs\":{{\"{id}\":{{\"version_req\":null,\"bins\":[\"hf2q\"],\"features\":[],\"all_features\":false,\"no_default_features\":false,\"profile\":\"release\",\"target\":\"aarch64-apple-darwin\",\"rustc\":\"rustc test\"}}}}}}\n"
        ),
        None => "{\"installs\":{}}\n".to_owned(),
    };
    for (name, bytes) in [(".crates.toml", v1_body), (".crates2.json", v2_body)] {
        let path = root.join(name);
        fs::write(&path, bytes).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
    }
}

fn registry_id(version: &str) -> String {
    format!("hf2q {version} (registry+https://github.com/rust-lang/crates.io-index)")
}

fn receipt_options() -> CargoInstallOptions {
    CargoInstallOptions {
        version_req: None,
        features: Default::default(),
        all_features: false,
        no_default_features: false,
        profile: "release".to_owned(),
        target: Some("aarch64-apple-darwin".to_owned()),
    }
}

#[test]
fn cargo_registry_owner_is_derived_from_canonical_root_and_matching_receipts() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("cargo-root");
    let binary = root.join("bin/hf2q");
    executable(&binary);
    let id = registry_id(env!("CARGO_PKG_VERSION"));
    write_receipts(&root, Some(&id), Some(&id));

    assert_eq!(
        detect(&binary).unwrap(),
        Installation::Cargo {
            root: fs::canonicalize(root).unwrap(),
            version: semver::Version::parse(env!("CARGO_PKG_VERSION")).unwrap(),
            source: CargoSource::CratesIo,
            options: receipt_options(),
        }
    );
}

#[test]
fn cargo_path_owner_decodes_the_exact_recorded_source() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("cargo-root");
    let binary = root.join("bin/hf2q");
    executable(&binary);
    let source = temp.path().join("source with space");
    let source_url = url::Url::from_file_path(&source).unwrap();
    let id = format!("hf2q {} (path+{})", env!("CARGO_PKG_VERSION"), source_url);
    write_receipts(&root, Some(&id), Some(&id));

    assert_eq!(
        detect(&binary).unwrap(),
        Installation::Cargo {
            root: fs::canonicalize(root).unwrap(),
            version: semver::Version::parse(env!("CARGO_PKG_VERSION")).unwrap(),
            source: CargoSource::Path(source),
            options: receipt_options(),
        }
    );
}

#[test]
fn cargo_git_selectors_and_custom_registry_are_reconstructed_from_receipts() {
    for (source, expected) in [
        (
            "git+https://example.invalid/hf2q#0123456789abcdef",
            CargoSource::Git(CargoGitSource {
                repository: "https://example.invalid/hf2q".to_owned(),
                selector: None,
                resolved_revision: "0123456789abcdef".to_owned(),
            }),
        ),
        (
            "git+https://example.invalid/hf2q?branch=main#0123456789abcdef",
            CargoSource::Git(CargoGitSource {
                repository: "https://example.invalid/hf2q".to_owned(),
                selector: Some(CargoGitSelector::Branch("main".to_owned())),
                resolved_revision: "0123456789abcdef".to_owned(),
            }),
        ),
        (
            "git+https://example.invalid/hf2q?tag=v0.1.8#0123456789abcdef",
            CargoSource::Git(CargoGitSource {
                repository: "https://example.invalid/hf2q".to_owned(),
                selector: Some(CargoGitSelector::Tag("v0.1.8".to_owned())),
                resolved_revision: "0123456789abcdef".to_owned(),
            }),
        ),
        (
            "git+https://example.invalid/hf2q?rev=01234567#0123456789abcdef",
            CargoSource::Git(CargoGitSource {
                repository: "https://example.invalid/hf2q".to_owned(),
                selector: Some(CargoGitSelector::Rev("01234567".to_owned())),
                resolved_revision: "0123456789abcdef".to_owned(),
            }),
        ),
        (
            "registry+https://registry.example.invalid/index",
            CargoSource::OtherRegistry(
                "registry+https://registry.example.invalid/index".to_owned(),
            ),
        ),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("cargo-root");
        let binary = root.join("bin/hf2q");
        executable(&binary);
        let id = format!("hf2q {} ({source})", env!("CARGO_PKG_VERSION"));
        write_receipts(&root, Some(&id), Some(&id));
        let Installation::Cargo {
            source: detected, ..
        } = detect(&binary).unwrap()
        else {
            panic!("expected Cargo owner");
        };
        assert_eq!(detected, expected);
    }
}

#[test]
fn credential_bearing_cargo_source_is_owned_but_never_replayed_or_printed() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("cargo-root");
    let binary = root.join("bin/hf2q");
    executable(&binary);
    let id = format!(
        "hf2q {} (git+https://secret-token@example.invalid/hf2q#0123456789abcdef)",
        env!("CARGO_PKG_VERSION")
    );
    write_receipts(&root, Some(&id), Some(&id));
    let Installation::Cargo {
        root,
        source,
        options,
        ..
    } = detect(&binary).unwrap()
    else {
        panic!("expected Cargo owner");
    };

    assert_eq!(
        source,
        CargoSource::Other("credential-bearing Git source".to_owned())
    );
    assert!(!source.description().contains("secret-token"));
    assert!(ManagerCommand::cargo_update(&root, &source, &options).is_err());
}

#[test]
fn cargo_receipt_disagreement_missing_peer_and_version_drift_fail_closed() {
    let current = registry_id(env!("CARGO_PKG_VERSION"));
    let old = registry_id("0.0.1");

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("disagree");
    let binary = root.join("bin/hf2q");
    executable(&binary);
    write_receipts(&root, Some(&current), Some(&old));
    assert!(detect(&binary)
        .unwrap_err()
        .to_string()
        .contains("receipts disagree"));

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("missing-peer");
    let binary = root.join("bin/hf2q");
    executable(&binary);
    write_receipts(&root, Some(&current), Some(&current));
    fs::remove_file(root.join(".crates2.json")).unwrap();
    assert!(detect(&binary)
        .unwrap_err()
        .to_string()
        .contains("only one"));

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("version-drift");
    let binary = root.join("bin/hf2q");
    executable(&binary);
    write_receipts(&root, Some(&old), Some(&old));
    assert!(detect(&binary)
        .unwrap_err()
        .to_string()
        .contains("does not match running"));
}

#[test]
fn duplicate_bin_or_symlinked_receipt_is_not_accepted_as_cargo_ownership() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("wrong-bin");
    let binary = root.join("bin/hf2q");
    executable(&binary);
    let id = registry_id(env!("CARGO_PKG_VERSION"));
    write_receipts(&root, Some(&id), Some(&id));
    fs::write(
        root.join(".crates.toml"),
        format!("[v1]\n\"{id}\" = [\"hf2q\", \"other\"]\n"),
    )
    .unwrap();
    fs::set_permissions(root.join(".crates.toml"), fs::Permissions::from_mode(0o600)).unwrap();
    assert!(detect(&binary)
        .unwrap_err()
        .to_string()
        .contains("exactly the hf2q executable"));

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("symlink-receipt");
    let binary = root.join("bin/hf2q");
    executable(&binary);
    write_receipts(&root, Some(&id), Some(&id));
    let retained = root.join("retained.json");
    fs::rename(root.join(".crates2.json"), &retained).unwrap();
    symlink(&retained, root.join(".crates2.json")).unwrap();
    assert!(detect(&binary).is_err());
}

#[test]
fn standalone_and_cargo_claims_are_ambiguous_not_precedence_ordered() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("dual-owner");
    let binary = root.join("bin/hf2q");
    executable(&binary);
    fs::write(root.join("bin/.hf2q-standalone.json"), MARKER).unwrap();
    fs::set_permissions(
        root.join("bin/.hf2q-standalone.json"),
        fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    let id = registry_id(env!("CARGO_PKG_VERSION"));
    write_receipts(&root, Some(&id), Some(&id));

    assert!(detect(&binary)
        .unwrap_err()
        .to_string()
        .contains("both standalone and Cargo"));
}

fn source_fixture(relative_binary: &Path) -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("Cargo.toml"),
        b"[package]\nname = \"hf2q\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::set_permissions(
        temp.path().join("Cargo.toml"),
        fs::Permissions::from_mode(0o644),
    )
    .unwrap();
    let binary = temp.path().join(relative_binary);
    executable(&binary);
    (temp, binary)
}

#[test]
fn standard_source_debug_release_and_target_triple_layouts_are_recognized() {
    for (relative, profile) in [
        (Path::new("target/debug/hf2q"), SourceProfile::Debug),
        (Path::new("target/release/hf2q"), SourceProfile::Release),
        (
            Path::new("target/aarch64-apple-darwin/release/hf2q"),
            SourceProfile::Release,
        ),
    ] {
        let (temp, binary) = source_fixture(relative);
        assert_eq!(
            detect_with_manifest_root(&binary, temp.path()).unwrap(),
            Installation::SourceDevelopment {
                workspace_root: fs::canonicalize(temp.path()).unwrap(),
                profile,
            }
        );
    }
}

#[test]
fn copied_binary_custom_target_and_wrong_manifest_are_unmanaged() {
    let temp = tempfile::tempdir().unwrap();
    let copied = temp.path().join("hf2q");
    executable(&copied);
    assert!(matches!(
        detect(&copied).unwrap(),
        Installation::Unmanaged { .. }
    ));

    let (temp, custom) = source_fixture(Path::new("custom/release/hf2q"));
    assert!(matches!(
        detect_with_manifest_root(&custom, temp.path()).unwrap(),
        Installation::Unmanaged { .. }
    ));
    drop(temp);

    let (temp, wrong) = source_fixture(Path::new("target/release/hf2q"));
    fs::write(
        temp.path().join("Cargo.toml"),
        b"[package]\nname = \"not-hf2q\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    assert!(matches!(
        detect_with_manifest_root(&wrong, temp.path()).unwrap(),
        Installation::Unmanaged { .. }
    ));
}

#[test]
fn cargo_manager_commands_are_exact_direct_argv_with_explicit_root() {
    let root = Path::new("/tmp/hf2q root");
    let update =
        ManagerCommand::cargo_update(root, &CargoSource::CratesIo, &receipt_options()).unwrap();
    assert_eq!(
        update.argv(),
        vec![
            OsStr::new("cargo"),
            OsStr::new("install"),
            OsStr::new("--root"),
            root.as_os_str(),
            OsStr::new("--registry"),
            OsStr::new("crates-io"),
            OsStr::new("--locked"),
            OsStr::new("--bin"),
            OsStr::new("hf2q"),
            OsStr::new("--profile"),
            OsStr::new("release"),
            OsStr::new("--target"),
            OsStr::new("aarch64-apple-darwin"),
            OsStr::new("hf2q"),
        ]
    );
    assert_eq!(
        update.display(),
        "cargo install --root '/tmp/hf2q root' --registry crates-io --locked --bin hf2q --profile release --target aarch64-apple-darwin hf2q"
    );

    let mut selected_options = receipt_options();
    selected_options.version_req = Some("^0.1".to_owned());
    selected_options.features.insert("metal".to_owned());
    selected_options.features.insert("server".to_owned());
    selected_options.no_default_features = true;
    selected_options.profile = "distribution".to_owned();
    selected_options.target = None;
    assert_eq!(
        ManagerCommand::cargo_update(root, &CargoSource::CratesIo, &selected_options)
            .unwrap()
            .display(),
        "cargo install --root '/tmp/hf2q root' --registry crates-io --locked --bin hf2q --version '^0.1' --features 'metal,server' --no-default-features --profile distribution hf2q"
    );

    let path_source = CargoSource::Path(PathBuf::from("/tmp/source path"));
    let path_update = ManagerCommand::cargo_update(root, &path_source, &receipt_options()).unwrap();
    assert_eq!(
        path_update.argv(),
        vec![
            OsStr::new("cargo"),
            OsStr::new("install"),
            OsStr::new("--root"),
            root.as_os_str(),
            OsStr::new("--path"),
            OsStr::new("/tmp/source path"),
            OsStr::new("--locked"),
            OsStr::new("--bin"),
            OsStr::new("hf2q"),
            OsStr::new("--profile"),
            OsStr::new("release"),
            OsStr::new("--target"),
            OsStr::new("aarch64-apple-darwin"),
        ]
    );

    let git_source = CargoSource::Git(CargoGitSource {
        repository: "https://example.invalid/hf2q".to_owned(),
        selector: Some(CargoGitSelector::Branch("stable".to_owned())),
        resolved_revision: "0123456789abcdef".to_owned(),
    });
    let git_update = ManagerCommand::cargo_update(root, &git_source, &receipt_options()).unwrap();
    assert_eq!(
        git_update.display(),
        "cargo install --root '/tmp/hf2q root' --git https://example.invalid/hf2q --branch stable --locked --bin hf2q --profile release --target aarch64-apple-darwin hf2q"
    );

    let registry_source =
        CargoSource::OtherRegistry("registry+https://registry.example.invalid/index".to_owned());
    let registry_update =
        ManagerCommand::cargo_update(root, &registry_source, &receipt_options()).unwrap();
    assert_eq!(
        registry_update.display(),
        "cargo install --root '/tmp/hf2q root' --index https://registry.example.invalid/index --locked --bin hf2q --profile release --target aarch64-apple-darwin hf2q"
    );

    let version = semver::Version::parse(env!("CARGO_PKG_VERSION")).unwrap();
    let uninstall = ManagerCommand::cargo_uninstall(root, &version);
    assert_eq!(
        uninstall.argv(),
        vec![
            OsStr::new("cargo"),
            OsStr::new("uninstall"),
            OsStr::new("--root"),
            root.as_os_str(),
            OsStr::new("--package"),
            OsStr::new(&format!("hf2q@{version}")),
            OsStr::new("--bin"),
            OsStr::new("hf2q"),
        ]
    );
}

#[test]
fn cargo_update_reconciliation_requires_same_root_selector_and_options() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("cargo-root");
    let binary = root.join("bin/hf2q");
    version_executable(&binary);
    let new_id = format!(
        "hf2q {} (git+https://example.invalid/hf2q?branch=stable#fedcba9876543210)",
        env!("CARGO_PKG_VERSION")
    );
    write_receipts(&root, Some(&new_id), Some(&new_id));
    let root = fs::canonicalize(root).unwrap();
    let expected_source = CargoSource::Git(CargoGitSource {
        repository: "https://example.invalid/hf2q".to_owned(),
        selector: Some(CargoGitSelector::Branch("stable".to_owned())),
        resolved_revision: "0123456789abcdef".to_owned(),
    });

    assert_eq!(
        reconcile_cargo_update(&binary, &root, &expected_source, &receipt_options()).unwrap(),
        env!("CARGO_PKG_VERSION")
    );

    let wrong_source = CargoSource::Git(CargoGitSource {
        repository: "https://example.invalid/hf2q".to_owned(),
        selector: Some(CargoGitSelector::Branch("main".to_owned())),
        resolved_revision: "fedcba9876543210".to_owned(),
    });
    assert!(reconcile_cargo_update(&binary, &root, &wrong_source, &receipt_options()).is_err());

    let mut wrong_options = receipt_options();
    wrong_options.profile = "dev".to_owned();
    assert!(reconcile_cargo_update(&binary, &root, &expected_source, &wrong_options).is_err());
}

#[test]
fn cargo_uninstall_reconciliation_requires_both_receipts_to_drop_owner() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("cargo-root");
    let binary = root.join("bin/hf2q");
    fs::create_dir_all(binary.parent().unwrap()).unwrap();
    write_receipts(&root, None, None);
    reconcile_cargo_uninstall(&fs::canonicalize(&root).unwrap(), &binary).unwrap();

    let id = registry_id(env!("CARGO_PKG_VERSION"));
    write_receipts(&root, Some(&id), Some(&id));
    assert!(
        reconcile_cargo_uninstall(&fs::canonicalize(&root).unwrap(), &binary)
            .unwrap_err()
            .to_string()
            .contains("receipt remains")
    );
}

#[test]
fn real_cargo_path_install_update_and_uninstall_round_trip() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source");
    let install_root = temp.path().join("install root");
    fs::create_dir_all(source.join("src")).unwrap();
    fs::write(
        source.join("Cargo.toml"),
        format!(
            "[package]\nname = \"hf2q\"\nversion = \"{}\"\nedition = \"2021\"\n\n[[bin]]\nname = \"hf2q\"\npath = \"src/main.rs\"\n",
            env!("CARGO_PKG_VERSION")
        ),
    )
    .unwrap();
    fs::write(
        source.join("src/main.rs"),
        "fn main() { println!(\"hf2q {}\", env!(\"CARGO_PKG_VERSION\")); }\n",
    )
    .unwrap();
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let generated = Command::new(&cargo)
        .args(["generate-lockfile", "--manifest-path"])
        .arg(source.join("Cargo.toml"))
        .env("CARGO_NET_OFFLINE", "true")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "cargo generate-lockfile failed: {}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let installed = Command::new(&cargo)
        .arg("install")
        .arg("--path")
        .arg(&source)
        .arg("--root")
        .arg(&install_root)
        .args(["--locked", "--bin", "hf2q"])
        .env("CARGO_NET_OFFLINE", "true")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        installed.status.success(),
        "cargo install failed: {}",
        String::from_utf8_lossy(&installed.stderr)
    );

    let binary = install_root.join("bin/hf2q");
    let Installation::Cargo {
        root,
        version,
        source,
        options,
    } = detect(&binary).unwrap()
    else {
        panic!("real Cargo path install was not detected");
    };
    assert!(matches!(source, CargoSource::Path(_)));

    ManagerCommand::cargo_update(&root, &source, &options)
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(
        reconcile_cargo_update(&binary, &root, &source, &options).unwrap(),
        env!("CARGO_PKG_VERSION")
    );

    ManagerCommand::cargo_uninstall(&root, &version)
        .run()
        .unwrap();
    reconcile_cargo_uninstall(&root, &binary).unwrap();
    assert!(!binary.exists());
}
