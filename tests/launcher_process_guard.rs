use std::path::PathBuf;
use std::process::Command;

fn repository_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

#[test]
fn process_classifier_rejects_non_server_subcommands_and_accepts_serve() {
    let helper = repository_path("scripts/hf2q_process_guard.sh");
    let script = r#"
        source "$1"
        hf2q_command_runs_serve '/opt/hf2q/target/release/hf2q serve --model model.gguf'
        hf2q_command_runs_serve './target/release/hf2q -v serve --model model.gguf'
        ! hf2q_command_runs_serve './target/release/hf2q convert owner/model --output model.gguf'
        ! hf2q_command_runs_serve './target/release/hf2q info /tmp/serve.gguf'
        ! hf2q_command_runs_serve './target/release/hf2q cache list'
    "#;
    let status = Command::new("bash")
        .arg("-c")
        .arg(script)
        .arg("launcher-process-guard-test")
        .arg(helper)
        .status()
        .expect("run launcher process-classifier contract");
    assert!(status.success());
}

#[test]
fn every_large_model_launcher_uses_the_shared_server_classifier() {
    for launcher in [
        "scripts/serve_qwen36_opencode.sh",
        "scripts/serve_gemma4_opencode.sh",
        "scripts/serve_deepseek4_opencode.sh",
    ] {
        let source = std::fs::read_to_string(repository_path(launcher)).unwrap();
        assert!(source.contains("source \"$SCRIPT_DIR/hf2q_process_guard.sh\""));
        assert!(source.contains("RUNTIME_PIDS=\"$(hf2q_active_serve_pids)\""));
        assert!(!source.contains("RUNTIME_PIDS=\"$(pgrep -x hf2q"));
    }
}

/// ADR-067 D6: the launcher rejects DFlash2 variables only where no policy
/// state accepts them. A stale or incomplete drafter path pair must not
/// prevent startup under the default policy or the kill switch.
#[test]
fn dflash2_launcher_check_leaves_policy_dependent_paths_to_hf2q() {
    let helper = repository_path("scripts/hf2q_dflash2_launcher_check.sh");
    let directory = tempfile::tempdir().expect("tempdir");
    let config = directory.path().join("config.json");
    let weights = directory.path().join("model.safetensors");
    std::fs::write(&config, b"{}").unwrap();
    std::fs::write(&weights, b"weights").unwrap();
    let stale = directory.path().join("gone.json");
    let (config, weights, stale) = (
        config.to_str().unwrap(),
        weights.to_str().unwrap(),
        stale.to_str().unwrap(),
    );
    let check = |env: &[(&str, &str)]| {
        let mut command = Command::new("bash");
        command
            .arg("-c")
            .arg(r#"source "$1" && hf2q_dflash2_launcher_check"#)
            .arg("dflash2-launcher-check-test")
            .arg(&helper)
            .env_remove("HF2Q_DFLASH2")
            .env_remove("HF2Q_DFLASH2_CONFIG")
            .env_remove("HF2Q_DFLASH2_WEIGHTS");
        for (name, value) in env {
            command.env(name, value);
        }
        command
            .output()
            .expect("run launcher DFlash2 check")
            .status
            .code()
    };
    const POLICY: &str = "HF2Q_DFLASH2";
    const CONFIG: &str = "HF2Q_DFLASH2_CONFIG";
    const WEIGHTS: &str = "HF2Q_DFLASH2_WEIGHTS";
    // Default policy and kill switch: nothing about the paths stops startup.
    for policy in [None, Some("0"), Some("off"), Some("FALSE")] {
        let with_policy = |pairs: &[(&'static str, &str)]| {
            let mut env: Vec<(&str, &str)> = pairs.to_vec();
            env.extend(policy.map(|value| (POLICY, value)));
            check(&env)
        };
        assert_eq!(with_policy(&[]), Some(0), "{policy:?}");
        assert_eq!(with_policy(&[(CONFIG, stale), (WEIGHTS, stale)]), Some(0));
        assert_eq!(with_policy(&[(CONFIG, stale)]), Some(0), "incomplete pair");
        assert_eq!(with_policy(&[(WEIGHTS, weights)]), Some(0));
        assert_eq!(
            with_policy(&[(CONFIG, config), (WEIGHTS, weights)]),
            Some(0)
        );
    }
    // Required: the pair must be complete and present, or absent.
    for required in ["1", "on", "True"] {
        assert_eq!(check(&[(POLICY, required)]), Some(0));
        assert_eq!(
            check(&[(POLICY, required), (CONFIG, config), (WEIGHTS, weights)]),
            Some(0)
        );
        assert_eq!(
            check(&[(POLICY, required), (CONFIG, stale), (WEIGHTS, weights)]),
            Some(3)
        );
        assert_eq!(
            check(&[(POLICY, required), (CONFIG, config), (WEIGHTS, stale)]),
            Some(3)
        );
        assert_eq!(check(&[(POLICY, required), (CONFIG, config)]), Some(3));
        assert_eq!(check(&[(POLICY, required), (WEIGHTS, weights)]), Some(3));
    }
    // A value hf2q itself refuses.
    assert_eq!(check(&[(POLICY, "maybe")]), Some(3));
    // Every Qwen launcher goes through this check.
    let launcher =
        std::fs::read_to_string(repository_path("scripts/serve_qwen36_opencode.sh")).unwrap();
    assert!(launcher.contains("source \"$SCRIPT_DIR/hf2q_dflash2_launcher_check.sh\""));
    assert!(launcher.contains("hf2q_dflash2_launcher_check || exit 3"));
    assert!(!launcher.contains("is set but is not a file"));
}
