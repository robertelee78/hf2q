use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Command, Output};

use sha2::{Digest, Sha256};

fn private_dir(path: &Path) {
    fs::create_dir_all(path).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}

fn private_file(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}

fn run(binary: &Path, arguments: &[&str], cache_root: &Path) -> Output {
    let completion_home = cache_root.parent().unwrap().join("completion-home");
    Command::new(binary)
        .args(arguments)
        .env("HF2Q_CACHE_DIR", cache_root)
        .env("HOME", &completion_home)
        .env("XDG_STATE_HOME", completion_home.join("state"))
        .env("SHELL", "/bin/zsh")
        .env("BASH_COMPLETION_USER_DIR", completion_home.join("bash"))
        .env("HF2Q_ZSH_COMPLETIONS_DIR", completion_home.join("zsh"))
        .env("HF2Q_FISH_COMPLETIONS_DIR", completion_home.join("fish"))
        .env(
            "HF2Q_COMPLETION_STARTUP_FILE",
            completion_home.join(".zshrc"),
        )
        .output()
        .unwrap()
}

#[test]
fn standalone_uninstall_preview_and_explicit_purge_are_exact() {
    let source_binary = Path::new(env!("CARGO_BIN_EXE_hf2q"));
    let bytes = fs::read(source_binary).unwrap();
    let digest = hex::encode(Sha256::digest(&bytes));
    let temp = tempfile::tempdir().unwrap();
    let temp_root = temp.path().canonicalize().unwrap();
    let candidate = temp_root.join("hf2q-candidate");
    fs::copy(source_binary, &candidate).unwrap();
    fs::set_permissions(&candidate, fs::Permissions::from_mode(0o555)).unwrap();
    let install_dir = temp_root.join("bin");
    private_dir(&install_dir);

    let install = Command::new(source_binary)
        .args([
            "__standalone-install",
            "--candidate",
            candidate.to_str().unwrap(),
            "--install-dir",
            install_dir.to_str().unwrap(),
            "--size",
            &bytes.len().to_string(),
            "--sha256",
            &digest,
        ])
        .output()
        .unwrap();
    assert!(
        install.status.success(),
        "standalone activation failed: {}",
        String::from_utf8_lossy(&install.stderr)
    );
    assert!(
        source_binary.exists(),
        "the disposable candidate must protect Cargo's shared test binary"
    );

    let installed = install_dir.join("hf2q");
    let state_root = temp_root.join("state");
    private_dir(&state_root);
    private_file(&state_root.join("config.toml"), b"operator config\n");
    private_file(&state_root.join("operator-note"), b"preserve state\n");

    let cache_root = temp_root.join("cache");
    private_dir(&cache_root);
    private_dir(&cache_root.join("models/example"));
    private_dir(&cache_root.join("locks"));
    private_file(
        &cache_root.join("manifest.json"),
        b"{\"schema_version\":2,\"models\":{}}\n",
    );
    private_file(
        &cache_root.join("models/example/model.gguf"),
        b"cached model",
    );
    private_file(&cache_root.join("locks/preserve.lock"), b"lock");
    private_file(&cache_root.join("operator-note"), b"preserve cache\n");

    let completion_activation = run(&installed, &["--version"], &cache_root);
    assert!(
        completion_activation.status.success(),
        "completion activation failed: {}",
        String::from_utf8_lossy(&completion_activation.stderr)
    );
    let completion_home = temp_root.join("completion-home");
    let completion_snapshot = [
        completion_home.join("bash/completions/hf2q"),
        completion_home.join("zsh/_hf2q"),
        completion_home.join("fish/hf2q.fish"),
        completion_home.join(".zshrc"),
        completion_home.join("state/hf2q/completion-ownership-v1.json"),
    ]
    .map(|path| (path.clone(), fs::read(path).unwrap()));

    let state_arg = state_root.to_str().unwrap();
    let preview = run(
        &installed,
        &[
            "--state-root",
            state_arg,
            "uninstall",
            "--purge-config",
            "--purge-cache",
        ],
        &cache_root,
    );
    assert_eq!(preview.status.code(), Some(3));
    let preview_error = String::from_utf8_lossy(&preview.stderr);
    assert!(preview_error.contains("Config purge would remove only"));
    assert!(preview_error.contains("Cache purge would clear"));
    assert!(preview_error.contains("Completion cleanup is limited to"));
    assert!(installed.exists(), "preview removed the release");
    assert!(state_root.join("config.toml").exists());
    assert!(cache_root.join("models/example/model.gguf").exists());
    for (path, expected) in &completion_snapshot {
        assert_eq!(
            fs::read(path).unwrap(),
            *expected,
            "uninstall preview mutated {}",
            path.display()
        );
    }

    let uninstall = run(
        &installed,
        &[
            "--state-root",
            state_arg,
            "uninstall",
            "--purge-config",
            "--purge-cache",
            "--yes",
        ],
        &cache_root,
    );
    assert!(
        uninstall.status.success(),
        "explicit uninstall failed: {}",
        String::from_utf8_lossy(&uninstall.stderr)
    );

    assert!(!installed.exists());
    assert!(!install_dir.join(".hf2q-standalone.json").exists());
    assert!(!install_dir.join(".hf2q-previous").exists());
    assert!(!install_dir.join(".hf2q-standalone.lock").exists());
    assert!(!completion_home.join("bash/completions/hf2q").exists());
    assert!(!completion_home.join("zsh/_hf2q").exists());
    assert!(!completion_home.join("fish/hf2q.fish").exists());
    assert!(!completion_home.join(".zshrc").exists());
    assert!(!completion_home
        .join("state/hf2q/completion-ownership-v1.json")
        .exists());
    assert!(
        source_binary.exists(),
        "uninstall must not remove Cargo's shared test binary"
    );
    assert!(!state_root.join("config.toml").exists());
    assert_eq!(
        fs::read(state_root.join("operator-note")).unwrap(),
        b"preserve state\n"
    );
    assert!(cache_root.join("models").is_dir());
    assert!(fs::read_dir(cache_root.join("models"))
        .unwrap()
        .next()
        .is_none());
    assert_eq!(
        fs::read(cache_root.join("locks/preserve.lock")).unwrap(),
        b"lock"
    );
    assert_eq!(
        fs::read(cache_root.join("operator-note")).unwrap(),
        b"preserve cache\n"
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(cache_root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["schema_version"], 2);
    assert_eq!(manifest["models"], serde_json::json!({}));
}

#[test]
fn standalone_uninstall_preserves_and_reports_modified_completion() {
    let source_binary = Path::new(env!("CARGO_BIN_EXE_hf2q"));
    let bytes = fs::read(source_binary).unwrap();
    let digest = hex::encode(Sha256::digest(&bytes));
    let temp = tempfile::tempdir().unwrap();
    let temp_root = temp.path().canonicalize().unwrap();
    let candidate = temp_root.join("hf2q-candidate");
    fs::copy(source_binary, &candidate).unwrap();
    fs::set_permissions(&candidate, fs::Permissions::from_mode(0o555)).unwrap();
    let install_dir = temp_root.join("bin");
    private_dir(&install_dir);
    let install = Command::new(source_binary)
        .args([
            "__standalone-install",
            "--candidate",
            candidate.to_str().unwrap(),
            "--install-dir",
            install_dir.to_str().unwrap(),
            "--size",
            &bytes.len().to_string(),
            "--sha256",
            &digest,
        ])
        .output()
        .unwrap();
    assert!(install.status.success(), "{install:?}");

    let installed = install_dir.join("hf2q");
    let cache_root = temp_root.join("cache");
    private_dir(&cache_root);
    let provision = run(&installed, &["--version"], &cache_root);
    assert!(provision.status.success(), "{provision:?}");
    let completion_home = temp_root.join("completion-home");
    let bash = completion_home.join("bash/completions/hf2q");
    let mut modified = fs::read(&bash).unwrap();
    modified.extend_from_slice(b"# operator modification\n");
    fs::write(&bash, &modified).unwrap();

    let uninstall = run(&installed, &["uninstall", "--yes"], &cache_root);
    assert!(uninstall.status.success(), "{uninstall:?}");
    let stderr = String::from_utf8(uninstall.stderr).unwrap();
    assert!(
        stderr.contains("preserved completion registration"),
        "{stderr}"
    );
    assert!(stderr.contains("modified after installation"), "{stderr}");
    assert_eq!(fs::read(&bash).unwrap(), modified);
    assert!(!completion_home.join("zsh/_hf2q").exists());
    assert!(!completion_home.join("fish/hf2q.fish").exists());
    assert!(completion_home
        .join("state/hf2q/completion-ownership-v1.json")
        .exists());
    assert!(!installed.exists());
}

#[test]
fn standalone_rollback_rebuilds_completion_from_the_restored_binary() {
    let source_binary = Path::new(env!("CARGO_BIN_EXE_hf2q"));
    let bytes = fs::read(source_binary).unwrap();
    let digest = hex::encode(Sha256::digest(&bytes));
    let temp = tempfile::tempdir().unwrap();
    let temp_root = temp.path().canonicalize().unwrap();
    let install_dir = temp_root.join("bin");
    private_dir(&install_dir);

    for sequence in 1..=2 {
        let candidate = temp_root.join(format!("hf2q-candidate-{sequence}"));
        fs::copy(source_binary, &candidate).unwrap();
        fs::set_permissions(&candidate, fs::Permissions::from_mode(0o555)).unwrap();
        let install = Command::new(source_binary)
            .args([
                "__standalone-install",
                "--candidate",
                candidate.to_str().unwrap(),
                "--install-dir",
                install_dir.to_str().unwrap(),
                "--size",
                &bytes.len().to_string(),
                "--sha256",
                &digest,
            ])
            .output()
            .unwrap();
        assert!(install.status.success(), "install {sequence}: {install:?}");
    }

    let installed = install_dir.join("hf2q");
    let cache_root = temp_root.join("cache");
    private_dir(&cache_root);
    let provision = run(&installed, &["--version"], &cache_root);
    assert!(provision.status.success(), "{provision:?}");

    let completion_home = temp_root.join("completion-home");
    let registrations = [
        completion_home.join("bash/completions/hf2q"),
        completion_home.join("zsh/_hf2q"),
        completion_home.join("fish/hf2q.fish"),
    ];
    for path in &registrations {
        fs::remove_file(path).unwrap();
    }

    let rollback = run(&installed, &["update", "--rollback"], &cache_root);
    assert!(
        rollback.status.success(),
        "rollback failed: {}",
        String::from_utf8_lossy(&rollback.stderr)
    );
    for path in &registrations {
        assert!(
            path.is_file(),
            "rollback did not recreate {}",
            path.display()
        );
    }
    assert!(completion_home.join(".zshrc").is_file());
    assert!(completion_home
        .join("state/hf2q/completion-ownership-v1.json")
        .is_file());

    let uninstall = run(&installed, &["uninstall", "--yes"], &cache_root);
    assert!(uninstall.status.success(), "{uninstall:?}");
}
