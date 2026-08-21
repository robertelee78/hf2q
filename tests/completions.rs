use std::fs;

use assert_cmd::Command;

fn assert_no_implicit_completion_files(home: &std::path::Path) {
    for path in [
        home.join("bash/completions/hf2q"),
        home.join("zsh/_hf2q"),
        home.join("fish/hf2q.fish"),
    ] {
        assert!(
            !path.exists(),
            "ordinary invocation wrote {}",
            path.display()
        );
    }
    assert_eq!(
        fs::read(home.join(".zshrc")).expect("read preserved startup file"),
        b"operator-owned startup\n"
    );
}

fn isolated_command(home: &std::path::Path) -> Command {
    let mut command = Command::cargo_bin("hf2q").expect("hf2q binary");
    command
        .env("HOME", home)
        .env("SHELL", "/bin/zsh")
        .env("BASH_COMPLETION_USER_DIR", home.join("bash"))
        .env("HF2Q_ZSH_COMPLETIONS_DIR", home.join("zsh"))
        .env("HF2Q_FISH_COMPLETIONS_DIR", home.join("fish"))
        .env("HF2Q_COMPLETION_STARTUP_FILE", home.join(".zshrc"));
    command
}

#[test]
fn ordinary_invocations_never_mutate_shell_completion_state() {
    let home = tempfile::tempdir().expect("isolated home");
    fs::write(home.path().join(".zshrc"), b"operator-owned startup\n")
        .expect("seed operator startup file");

    for args in [["--version"].as_slice(), ["--help"].as_slice()] {
        let output = isolated_command(home.path())
            .args(args)
            .output()
            .expect("run ordinary hf2q invocation");
        assert!(output.status.success(), "{output:?}");
        assert_no_implicit_completion_files(home.path());
    }
}

#[test]
fn explicit_completions_emit_grammar_to_stdout_without_writing_shell_files() {
    let home = tempfile::tempdir().expect("isolated home");
    fs::write(home.path().join(".zshrc"), b"operator-owned startup\n")
        .expect("seed operator startup file");

    let output = isolated_command(home.path())
        .args(["completions", "--shell", "zsh"])
        .output()
        .expect("generate explicit zsh completions");
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8(output.stdout).expect("completion output is UTF-8");
    assert!(stdout.starts_with("#compdef hf2q\n"));
    assert!(stdout.contains("convert"));
    assert!(stdout.contains("serve"));
    assert!(stdout.contains("chat"));
    assert!(stdout.contains("--model"));
    assert!(stdout.contains("--quant"));
    assert!(stdout.contains("--artifact"));
    assert_no_implicit_completion_files(home.path());
}
