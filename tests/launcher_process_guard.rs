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
