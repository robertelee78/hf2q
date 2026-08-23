//! Process-boundary contract for the static GGUF serving preflight.

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn info_requires_an_explicit_local_model() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .arg("info")
        .assert()
        .failure()
        .stderr(predicate::str::contains("--model"));
}

#[test]
fn info_rejects_removed_hugging_face_input_surfaces() {
    for arguments in [
        ["info", "--input", "/tmp/model"],
        ["info", "--repo", "org/model"],
    ] {
        Command::cargo_bin("hf2q")
            .unwrap()
            .args(arguments)
            .assert()
            .failure();
    }
}

#[test]
fn info_missing_model_is_a_static_preflight_rejection() {
    let assertion = Command::cargo_bin("hf2q")
        .unwrap()
        .args(["info", "--model", "/definitely/not/a/model.gguf"])
        .assert()
        .failure()
        .stdout(predicate::str::contains(
            "Validation: static preflight (tensor payloads not decoded or uploaded; Metal not initialized)",
        ))
        .stdout(predicate::str::contains("Serve support: rejected"))
        .stderr(predicate::str::is_empty());
    let stdout = String::from_utf8(assertion.get_output().stdout.clone()).unwrap();
    assert!(
        stdout
            .trim_end()
            .lines()
            .last()
            .is_some_and(|line| line.starts_with("Serve support: rejected — ")),
        "rejected static preflight must end with its serve-support verdict: {stdout}"
    );
}

#[test]
fn invalid_selected_config_still_ends_with_the_info_verdict() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap().join("state");
    std::fs::create_dir(&root).unwrap();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
    let config = root.join("config.toml");
    std::fs::write(&config, b"not valid toml = [").unwrap();
    std::fs::set_permissions(&config, std::fs::Permissions::from_mode(0o600)).unwrap();
    let assertion = Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "--state-root",
            root.to_str().unwrap(),
            "info",
            "--model",
            "/definitely/not/a/model.gguf",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::is_empty());
    let stdout = String::from_utf8(assertion.get_output().stdout.clone()).unwrap();
    assert!(stdout.contains("invalid TOML"), "{stdout}");
    assert!(
        stdout
            .trim_end()
            .lines()
            .last()
            .is_some_and(|line| line.starts_with("Serve support: rejected — ")),
        "config rejection must end with the serve-support verdict: {stdout}"
    );
}

#[test]
fn info_help_matches_the_serve_planning_surface() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .args(["info", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--model"))
        .stdout(predicate::str::contains("--mmproj"))
        .stdout(predicate::str::contains("--ctx"))
        .stdout(predicate::str::contains("--scheduler"))
        .stdout(predicate::str::contains("--max-slots"))
        .stdout(predicate::str::contains("--kv-cache-budget"))
        .stdout(predicate::str::contains("--kv-persist"))
        .stdout(predicate::str::contains("--kv-persist-budget"))
        .stdout(predicate::str::contains("--input").not())
        .stdout(predicate::str::contains("--repo").not());
}

#[test]
fn generate_kv_bits_is_a_typed_setting_not_an_environment_warning() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .env_remove("HF2Q_TQ_CODEBOOK_BITS")
        .args([
            "generate",
            "--model",
            "/definitely/not/a/model.gguf",
            "--prompt",
            "test",
            "--kv-bits",
            "5",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Model not found"))
        .stderr(predicate::str::contains("investigation-only environment variables").not())
        .stderr(predicate::str::contains("HF2Q_TQ_CODEBOOK_BITS").not());
}
