use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use assert_cmd::Command;

const HIDDEN: &[&str] = &[
    "__standalone-install",
    "__fetch-hub-gguf",
    "__catalog-hub-gguf",
    "__verify-local-gguf",
    "source-teacher",
    "source-teacher-reference",
    "source-teacher-acceptance-verify",
    "--chat-parent-lifeline-fd",
];

struct IsolatedHome {
    root: tempfile::TempDir,
}

impl IsolatedHome {
    fn new() -> Self {
        Self {
            root: tempfile::tempdir().expect("isolated home"),
        }
    }

    fn path(&self) -> &Path {
        self.root.path()
    }

    fn command(&self) -> Command {
        let mut command = Command::cargo_bin("hf2q").expect("hf2q binary");
        command
            .env("HOME", self.path())
            .env("XDG_STATE_HOME", self.path().join("state"))
            .env("SHELL", "/bin/zsh")
            .env("BASH_COMPLETION_USER_DIR", self.path().join("bash"))
            .env("HF2Q_ZSH_COMPLETIONS_DIR", self.path().join("zsh"))
            .env("HF2Q_FISH_COMPLETIONS_DIR", self.path().join("fish"))
            .env("HF2Q_COMPLETION_STARTUP_FILE", self.path().join(".zshrc"));
        command
    }

    fn registrations(&self) -> [PathBuf; 3] {
        [
            self.path().join("bash/completions/hf2q"),
            self.path().join("zsh/_hf2q"),
            self.path().join("fish/hf2q.fish"),
        ]
    }

    fn receipt(&self) -> PathBuf {
        self.path().join("state/hf2q/completion-ownership-v1.json")
    }
}

#[test]
fn source_debug_binary_without_explicit_destinations_is_nonmutating() {
    let home = tempfile::tempdir().unwrap();
    let output = Command::cargo_bin("hf2q")
        .unwrap()
        .env("HOME", home.path())
        .env("XDG_STATE_HOME", home.path().join("state"))
        .env("SHELL", "/bin/zsh")
        .arg("--version")
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    assert!(!home.path().join(".zshrc").exists());
    assert!(!home.path().join("state").exists());
    assert!(!home.path().join(".local/share").exists());
}

#[test]
fn first_invocation_provisions_all_shells_startup_and_receipt_idempotently() {
    let home = IsolatedHome::new();
    fs::write(home.path().join(".zshrc"), b"operator-owned startup\n").unwrap();

    let first = home.command().arg("--version").output().unwrap();
    assert!(first.status.success(), "{first:?}");
    let stderr = String::from_utf8(first.stderr).unwrap();
    assert!(
        stderr.contains("installed Tab completion for bash, fish, zsh"),
        "{stderr}"
    );

    let registrations = home.registrations();
    for path in &registrations {
        let bytes = fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("hf2q-managed dynamic completion"), "{text}");
        assert!(text.contains("hf2q-completion-binding sha256:"), "{text}");
    }
    let startup = fs::read_to_string(home.path().join(".zshrc")).unwrap();
    assert!(startup.starts_with("operator-owned startup\n"));
    assert_eq!(
        startup.matches("# >>> hf2q managed completion >>>").count(),
        1
    );
    let receipt = fs::read_to_string(home.receipt()).unwrap();
    assert!(receipt.contains("\"schema_version\": 1"));
    assert!(receipt.contains("bash/completions/hf2q"));
    assert!(receipt.contains("zsh/_hf2q"));
    assert!(receipt.contains("fish/hf2q.fish"));
    assert!(receipt.contains(".zshrc"));

    let before = registrations
        .iter()
        .map(|path| fs::read(path).unwrap())
        .collect::<Vec<_>>();
    let startup_before = fs::read(home.path().join(".zshrc")).unwrap();
    let receipt_before = fs::read(home.receipt()).unwrap();
    let second = home.command().arg("--help").output().unwrap();
    assert!(second.status.success(), "{second:?}");
    assert!(
        !String::from_utf8(second.stderr)
            .unwrap()
            .contains("installed Tab completion")
    );
    for (path, expected) in registrations.iter().zip(before) {
        assert_eq!(fs::read(path).unwrap(), expected);
    }
    assert_eq!(
        fs::read(home.path().join(".zshrc")).unwrap(),
        startup_before
    );
    assert_eq!(fs::read(home.receipt()).unwrap(), receipt_before);
}

#[test]
fn opt_out_is_presence_based_and_prevents_every_write() {
    let home = IsolatedHome::new();
    for value in ["", "1"] {
        let output = home
            .command()
            .env("HF2Q_NO_COMPLETION_INSTALL", value)
            .arg("--version")
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
    }
    for path in home.registrations() {
        assert!(!path.exists(), "opt-out wrote {}", path.display());
    }
    assert!(!home.receipt().exists());
    assert!(!home.path().join(".zshrc").exists());
}

#[test]
fn static_generation_is_public_stdout_only_and_broken_pipe_safe() {
    let home = IsolatedHome::new();
    for shell in ["bash", "elvish", "fish", "powershell", "zsh"] {
        let output = home
            .command()
            .env("HF2Q_NO_COMPLETION_INSTALL", "1")
            .args(["completions", "--shell", shell])
            .output()
            .unwrap();
        assert!(output.status.success(), "{shell}: {output:?}");
        assert!(
            output.stderr.is_empty(),
            "{shell}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("convert"));
        assert!(stdout.contains("serve"));
        for hidden in HIDDEN {
            assert!(!stdout.contains(hidden), "{shell} leaked {hidden}");
        }
    }

    let binary = assert_cmd::cargo::cargo_bin("hf2q");
    let status = ProcessCommand::new("/bin/bash")
        .args(["--noprofile", "--norc", "-o", "pipefail", "-c"])
        .arg("HF2Q_NO_COMPLETION_INSTALL=1 \"$1\" completions --shell bash | /usr/bin/true")
        .arg("sh")
        .arg(binary)
        .status()
        .unwrap();
    assert!(status.success());
}

fn dynamic(home: &IsolatedHome, shell: &str, index: usize, words: &[&str]) -> String {
    let output = home
        .command()
        .env("HF2Q_NO_COMPLETION_INSTALL", "1")
        .env("HF2Q_COMPLETE", shell)
        .env("_CLAP_COMPLETE_INDEX", index.to_string())
        .env("_CLAP_COMPLETE_COMP_TYPE", "9")
        .env("_CLAP_IFS", "\n")
        .arg("--")
        .args(words)
        .output()
        .unwrap();
    assert!(output.status.success(), "{shell}: {output:?}");
    assert!(
        output.stderr.is_empty(),
        "{shell}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn dynamic_candidate_name(line: &str) -> &str {
    line.split(['\t', ':']).next().unwrap_or(line)
}

#[test]
fn dynamic_protocol_is_public_semantic_and_side_effect_free() {
    let home = IsolatedHome::new();
    for shell in ["bash", "elvish", "fish", "powershell", "zsh"] {
        let commands = dynamic(&home, shell, 1, &["hf2q", "co"]);
        assert!(
            commands
                .lines()
                .map(dynamic_candidate_name)
                .any(|line| line == "convert"),
            "{shell}: {commands}"
        );
        assert!(
            commands
                .lines()
                .map(dynamic_candidate_name)
                .any(|line| line == "completions"),
            "{shell}: {commands}"
        );
        for hidden in HIDDEN {
            assert!(
                !commands.contains(hidden),
                "{shell} dynamic protocol leaked {hidden}"
            );
        }
    }

    let quants = dynamic(
        &home,
        "bash",
        4,
        &["hf2q", "convert", "model", "--quant", "q4_"],
    );
    for quant in ["q4_0", "q4_1", "q4_k_s", "q4_k_m"] {
        assert!(quants.lines().any(|line| line == quant), "{quants}");
    }
    assert!(!quants.contains("dwq"));

    for quant in ["q2_k_s", "iq4_xs"] {
        let prefix = quant.split('_').next().unwrap();
        let candidates = dynamic(
            &home,
            "bash",
            4,
            &["hf2q", "convert", "model", "--quant", prefix],
        );
        assert!(
            candidates.lines().any(|line| line == quant),
            "{quant} was not suggested: {candidates}"
        );
    }

    let arches = dynamic(&home, "zsh", 3, &["hf2q", "smoke", "--arch", "qwen"]);
    assert!(
        arches.lines().any(|line| line.starts_with("qwen35")),
        "{arches}"
    );
    assert!(
        arches.lines().any(|line| line.starts_with("qwen35moe")),
        "{arches}"
    );

    for path in home.registrations() {
        assert!(
            !path.exists(),
            "completion protocol wrote {}",
            path.display()
        );
    }
    assert!(!home.receipt().exists());
    assert!(!home.path().join(".zshrc").exists());
}

#[test]
fn serve_model_completion_prefers_managed_models_and_keeps_explicit_paths() {
    let home = IsolatedHome::new();
    let models = home.path().join(".local/share/hf2q/models");
    let qwen36 = models.join("qwen3.6");
    let qwen38 = models.join("qwen3.8");
    fs::create_dir_all(&qwen36).unwrap();
    fs::create_dir_all(&qwen38).unwrap();

    let decoder = qwen38.join("qwen38-hf2q-q4_k_m.gguf");
    let projector = qwen38.join("qwen38-hf2q-q4_k_m-mmproj.gguf");
    fs::write(&decoder, b"decoder").unwrap();
    fs::write(&projector, b"projector").unwrap();

    let directories = dynamic(&home, "zsh", 3, &["hf2q", "serve", "--model", ""]);
    let directory_values = directories
        .lines()
        .map(dynamic_candidate_name)
        .collect::<Vec<_>>();
    assert_eq!(
        directory_values,
        [
            format!("{}/", qwen36.display()),
            format!("{}/", qwen38.display()),
        ],
        "{directories}"
    );

    let qwen38_prefix = format!("{}/", qwen38.display());
    let model_files = dynamic(
        &home,
        "bash",
        3,
        &["hf2q", "serve", "--model", &qwen38_prefix],
    );
    let decoder = decoder.display().to_string();
    let projector = projector.display().to_string();
    assert!(
        model_files.lines().any(|line| line == decoder.as_str()),
        "{model_files}"
    );
    assert!(!model_files.contains("mmproj"), "{model_files}");

    let mmproj_files = dynamic(
        &home,
        "fish",
        3,
        &["hf2q", "serve", "--mmproj", &qwen38_prefix],
    );
    assert!(
        mmproj_files.lines().any(|line| line == projector.as_str()),
        "{mmproj_files}"
    );
    assert!(!mmproj_files.lines().any(|line| line == decoder.as_str()));

    let customer_dir = home.path().join("Desktop/customer-model");
    fs::create_dir_all(&customer_dir).unwrap();
    let customer_model = customer_dir.join("customer.gguf");
    fs::write(&customer_model, b"customer").unwrap();
    let customer_prefix = format!("{}/", customer_dir.display());
    let customer_files = dynamic(
        &home,
        "powershell",
        3,
        &["hf2q", "serve", "--model", &customer_prefix],
    );
    assert!(
        customer_files
            .lines()
            .any(|line| line == customer_model.display().to_string()),
        "{customer_files}"
    );

    for path in home.registrations() {
        assert!(!path.exists(), "completion wrote {}", path.display());
    }
    assert!(!home.receipt().exists());
    assert!(!home.path().join(".zshrc").exists());
}

#[test]
fn installed_bash_and_zsh_adapters_execute_real_tab_dispatch() {
    let home = IsolatedHome::new();
    let provision = home.command().arg("--version").output().unwrap();
    assert!(provision.status.success(), "{provision:?}");

    if Path::new("/bin/bash").is_file() {
        let output = ProcessCommand::new("/bin/bash")
            .args(["--noprofile", "--norc", "-c"])
            .arg("source \"$1\"; COMP_WORDS=(hf2q co); COMP_CWORD=1; COMP_TYPE=9; _clap_complete_hf2q '' 'co'; printf '%s\\n' \"${COMPREPLY[@]}\"")
            .arg("bash")
            .arg(home.registrations()[0].clone())
            .env_remove("BASH_ENV")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.lines().any(|line| line == "convert"), "{stdout}");
        assert!(stdout.lines().any(|line| line == "completions"), "{stdout}");
    }

    if Path::new("/bin/zsh").is_file() {
        let output = ProcessCommand::new("/bin/zsh")
            .args(["-f", "-c"])
            .arg("compdef() { return 0; }; source \"$1\"; _describe() { local n=${argv[-1]}; print -rl -- \"${(@P)n}\"; }; words=(hf2q co); CURRENT=2; _clap_dynamic_completer_hf2q")
            .arg("zsh")
            .arg(home.registrations()[1].clone())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(
            stdout.lines().any(|line| line.starts_with("convert:")),
            "{stdout}"
        );
        assert!(
            stdout.lines().any(|line| line.starts_with("completions:")),
            "{stdout}"
        );
    }
}
