use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};

use sha2::{Digest, Sha256};
use tempfile::TempDir;

use crate::cli::{Cli, Command};
use clap::Parser;

use super::{
    publish_verified_candidate, rollback, uninstall, verify_running_installation,
    CandidateExpectation, PublishOutcome, StandaloneError, ACTIVE_NAME, LOCK_NAME, MARKER_NAME,
    PREVIOUS_NAME,
};

fn fixture(bytes: &[u8]) -> (TempDir, std::path::PathBuf, CandidateExpectation) {
    let temp = TempDir::new().expect("tempdir");
    let path = temp.path().join("candidate");
    fs::write(&path, bytes).expect("write candidate");
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    let expectation = CandidateExpectation::new(bytes.len() as u64, digest).expect("expectation");
    (temp, path, expectation)
}

fn install_dir() -> (TempDir, std::path::PathBuf) {
    let temp = TempDir::new().expect("tempdir");
    let path = temp.path().canonicalize().expect("canonical temp path");
    (temp, path)
}

#[test]
fn standalone_lifecycle_installs_updates_rolls_back_and_preserves_operator_state() {
    let (_root, install) = install_dir();
    let state = TempDir::new().expect("state tempdir");
    fs::write(state.path().join("config.toml"), b"operator config").expect("config");
    fs::create_dir(state.path().join("models")).expect("models");
    fs::write(state.path().join("models/model.gguf"), b"model").expect("model");
    let (_first_temp, first, first_expectation) = fixture(b"first exact hf2q bytes");
    let (_second_temp, second, second_expectation) = fixture(b"second exact hf2q bytes");

    assert_eq!(
        publish_verified_candidate(&install, &first, &first_expectation).expect("first install"),
        PublishOutcome::Installed
    );
    assert_eq!(
        fs::read(install.join(ACTIVE_NAME)).expect("active"),
        b"first exact hf2q bytes"
    );
    assert!(install.join(MARKER_NAME).is_file());
    assert!(install.join(LOCK_NAME).is_file());
    assert!(!install.join(PREVIOUS_NAME).exists());

    assert_eq!(
        publish_verified_candidate(&install, &second, &second_expectation).expect("update"),
        PublishOutcome::Updated
    );
    assert_eq!(
        fs::read(install.join(ACTIVE_NAME)).expect("active"),
        b"second exact hf2q bytes"
    );
    assert_eq!(
        fs::read(install.join(PREVIOUS_NAME)).expect("previous"),
        b"first exact hf2q bytes"
    );
    assert_eq!(
        verify_running_installation(&install.join(ACTIVE_NAME)).expect("active binding"),
        install
    );
    assert!(verify_running_installation(&install.join(PREVIOUS_NAME)).is_err());

    rollback(&install).expect("rollback");
    assert_eq!(
        fs::read(install.join(ACTIVE_NAME)).expect("active"),
        b"first exact hf2q bytes"
    );
    assert_eq!(
        fs::read(install.join(PREVIOUS_NAME)).expect("previous"),
        b"second exact hf2q bytes"
    );

    uninstall(&install).expect("uninstall");
    for name in [ACTIVE_NAME, MARKER_NAME, PREVIOUS_NAME, LOCK_NAME] {
        assert!(!install.join(name).exists(), "{name} survived uninstall");
    }
    assert_eq!(
        fs::read(state.path().join("config.toml")).expect("config"),
        b"operator config"
    );
    assert_eq!(
        fs::read(state.path().join("models/model.gguf")).expect("model"),
        b"model"
    );
}

#[test]
fn invalid_candidate_and_unowned_existing_binary_leave_installation_unchanged() {
    let (_root, install) = install_dir();
    let (_first_temp, first, first_expectation) = fixture(b"first exact hf2q bytes");
    publish_verified_candidate(&install, &first, &first_expectation).expect("install");
    let before = fs::read(install.join(ACTIVE_NAME)).expect("active");
    let (_second_temp, second, _) = fixture(b"second exact hf2q bytes");
    let wrong = CandidateExpectation::new(23, [0_u8; 32]).expect("wrong expectation");
    assert!(matches!(
        publish_verified_candidate(&install, &second, &wrong),
        Err(StandaloneError::DigestMismatch)
    ));
    assert_eq!(fs::read(install.join(ACTIVE_NAME)).expect("active"), before);

    uninstall(&install).expect("uninstall");
    fs::write(install.join(ACTIVE_NAME), b"foreign hf2q").expect("foreign binary");
    assert!(matches!(
        publish_verified_candidate(&install, &first, &first_expectation),
        Err(StandaloneError::Invalid(_))
    ));
    assert_eq!(
        fs::read(install.join(ACTIVE_NAME)).expect("foreign"),
        b"foreign hf2q"
    );
}

#[test]
fn hostile_marker_nodes_are_rejected_without_replacement() {
    let (_root, install) = install_dir();
    let external = install.join("external-marker");
    fs::write(&external, b"external").expect("external marker");
    symlink(&external, install.join(MARKER_NAME)).expect("marker symlink");
    let (_candidate_temp, candidate, expectation) = fixture(b"candidate");
    assert!(publish_verified_candidate(&install, &candidate, &expectation).is_err());
    assert_eq!(fs::read(&external).expect("external"), b"external");
    assert_eq!(
        fs::read_link(install.join(MARKER_NAME)).expect("marker link"),
        external
    );
}

#[test]
fn marker_is_canonical_and_wrong_mode_is_preserved() {
    let (_root, install) = install_dir();
    let (_candidate_temp, candidate, expectation) = fixture(b"candidate");
    publish_verified_candidate(&install, &candidate, &expectation).expect("install");
    let expected = b"{\"kind\":\"hf2q.install-channel\",\"schema_version\":1,\"package\":\"hf2q\",\"channel\":\"standalone\"}\n";
    assert_eq!(
        fs::read(install.join(MARKER_NAME)).expect("marker"),
        expected
    );

    fs::set_permissions(install.join(MARKER_NAME), fs::Permissions::from_mode(0o644))
        .expect("wrong mode");
    let before = fs::read(install.join(ACTIVE_NAME)).expect("active");
    assert!(publish_verified_candidate(&install, &candidate, &expectation).is_err());
    assert_eq!(fs::read(install.join(ACTIVE_NAME)).expect("active"), before);
    assert_eq!(
        fs::metadata(install.join(MARKER_NAME))
            .expect("metadata")
            .permissions()
            .mode()
            & 0o7777,
        0o644
    );
}

#[test]
fn standalone_installer_bootstrap_surface_is_hidden_bounded_and_parseable() {
    let digest = "ab".repeat(32);
    let cli = Cli::try_parse_from([
        "hf2q",
        "__standalone-install",
        "--install-dir",
        "/tmp/hf2q-bin",
        "--candidate",
        "/tmp/hf2q-candidate",
        "--size",
        "42",
        "--sha256",
        &digest,
    ])
    .expect("hidden installer command parses");
    let Command::StandaloneInstall(args) = cli.command else {
        panic!("standalone installer command was not selected");
    };
    assert_eq!(args.install_dir, std::path::Path::new("/tmp/hf2q-bin"));
    assert_eq!(args.candidate, std::path::Path::new("/tmp/hf2q-candidate"));
    assert_eq!(args.size, 42);
    assert_eq!(args.sha256, digest);

    let help = Cli::try_parse_from(["hf2q", "--help"])
        .expect_err("help exits through clap")
        .to_string();
    assert!(!help.contains("__standalone-install"));
    assert!(help.contains("update"));
    assert!(help.contains("uninstall"));

    let cli =
        Cli::try_parse_from(["hf2q", "update", "--check"]).expect("standalone update check parses");
    let Command::Update(args) = cli.command else {
        panic!("update command was not selected");
    };
    assert!(args.check);
    assert!(!args.rollback);
    assert!(Cli::try_parse_from(["hf2q", "update", "--check", "--rollback"]).is_err());

    let cli =
        Cli::try_parse_from(["hf2q", "uninstall", "--yes"]).expect("uninstall command parses");
    let Command::Uninstall(args) = cli.command else {
        panic!("uninstall command was not selected");
    };
    assert!(args.yes);
}
