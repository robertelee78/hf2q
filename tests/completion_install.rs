use std::fs;
use std::path::{Path, PathBuf};

use assert_cmd::Command;

const MARKER: &str = "# hf2q-managed shell completion";
const BEGIN: &str = "# >>> hf2q managed completion >>>";

fn run_version(home: &Path) -> std::process::Output {
    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .env("HOME", home)
        .env("SHELL", "/bin/zsh")
        .env("BASH_COMPLETION_USER_DIR", home.join("bash"))
        .env("HF2Q_ZSH_COMPLETIONS_DIR", home.join("zsh"))
        .env("HF2Q_FISH_COMPLETIONS_DIR", home.join("fish"))
        .env("HF2Q_COMPLETION_STARTUP_FILE", home.join(".zshrc"))
        .env_remove("HF2Q_NO_COMPLETION_INSTALL")
        .arg("--version")
        .output()
        .expect("run hf2q --version")
}

fn installed_paths(home: &Path) -> [PathBuf; 4] {
    [
        home.join("bash/completions/hf2q"),
        home.join("zsh/_hf2q"),
        home.join("fish/hf2q.fish"),
        home.join(".zshrc"),
    ]
}

#[test]
fn normal_invocation_installs_all_shells_and_is_idempotent() {
    let home = tempfile::tempdir().expect("isolated home");
    let first = run_version(home.path());
    assert!(first.status.success(), "{first:?}");

    let paths = installed_paths(home.path());
    for path in &paths[..3] {
        let text = fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("read generated {}: {error}", path.display()));
        assert!(
            text.contains(MARKER),
            "missing marker in {}",
            path.display()
        );
        assert!(
            text.contains("convert"),
            "missing grammar in {}",
            path.display()
        );
    }
    let startup = fs::read_to_string(&paths[3]).expect("read .zshrc");
    assert_eq!(startup.matches(BEGIN).count(), 1);
    assert!(startup.contains("zsh/_hf2q"));

    let before = paths
        .iter()
        .map(|path| fs::read(path).expect("snapshot installed file"))
        .collect::<Vec<_>>();
    let second = run_version(home.path());
    assert!(second.status.success(), "{second:?}");
    for (path, expected) in paths.iter().zip(before) {
        assert_eq!(
            fs::read(path).unwrap(),
            expected,
            "{} drifted",
            path.display()
        );
    }
}

#[test]
fn opt_out_writes_nothing() {
    let home = tempfile::tempdir().expect("isolated home");
    let output = Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .env("HOME", home.path())
        .env("SHELL", "/bin/zsh")
        .env("BASH_COMPLETION_USER_DIR", home.path().join("bash"))
        .env("HF2Q_ZSH_COMPLETIONS_DIR", home.path().join("zsh"))
        .env("HF2Q_FISH_COMPLETIONS_DIR", home.path().join("fish"))
        .env("HF2Q_COMPLETION_STARTUP_FILE", home.path().join(".zshrc"))
        .env("HF2Q_NO_COMPLETION_INSTALL", "1")
        .arg("--version")
        .output()
        .expect("run opted-out hf2q");
    assert!(output.status.success(), "{output:?}");
    for path in installed_paths(home.path()) {
        assert!(!path.exists(), "opt-out wrote {}", path.display());
    }
}

#[test]
fn one_shell_failure_does_not_fail_hf2q_or_block_other_shells() {
    let home = tempfile::tempdir().expect("isolated home");
    let blocked = home.path().join("zsh-is-a-file");
    fs::write(&blocked, b"not a directory").expect("block zsh destination");

    let output = Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .env("HOME", home.path())
        .env("SHELL", "/bin/zsh")
        .env("BASH_COMPLETION_USER_DIR", home.path().join("bash"))
        .env("HF2Q_ZSH_COMPLETIONS_DIR", &blocked)
        .env("HF2Q_FISH_COMPLETIONS_DIR", home.path().join("fish"))
        .env("HF2Q_COMPLETION_STARTUP_FILE", home.path().join(".zshrc"))
        .env_remove("HF2Q_NO_COMPLETION_INSTALL")
        .arg("--version")
        .output()
        .expect("run with blocked zsh completion");
    assert!(output.status.success(), "{output:?}");
    assert!(home.path().join("bash/completions/hf2q").is_file());
    assert!(home.path().join("fish/hf2q.fish").is_file());
    assert!(!home.path().join(".zshrc").exists());
}
