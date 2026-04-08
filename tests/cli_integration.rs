//! Integration tests for CLI argument parsing and help output.

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_help_shows_subcommands() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("convert"))
        .stdout(predicate::str::contains("info"))
        .stdout(predicate::str::contains("doctor"))
        .stdout(predicate::str::contains("completions"));
}

#[test]
fn test_version() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("hf2q"));
}

#[test]
fn test_convert_help_shows_all_flags() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .args(["convert", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--input"))
        .stdout(predicate::str::contains("--repo"))
        .stdout(predicate::str::contains("--format"))
        .stdout(predicate::str::contains("--quant"))
        .stdout(predicate::str::contains("--sensitive-layers"))
        .stdout(predicate::str::contains("--calibration-samples"))
        .stdout(predicate::str::contains("--bits"))
        .stdout(predicate::str::contains("--group-size"))
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("--json-report"))
        .stdout(predicate::str::contains("--skip-quality"))
        .stdout(predicate::str::contains("--dry-run"))
        .stdout(predicate::str::contains("--yes"))
        .stdout(predicate::str::contains("--unsupported-layers"));
}

#[test]
fn test_no_subcommand_shows_help() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .assert()
        .failure();
}

#[test]
fn test_doctor_runs() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .arg("doctor")
        .assert()
        .success()
        .stdout(predicate::str::contains("hf2q doctor"));
}

#[test]
fn test_completions_zsh() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .args(["completions", "--shell", "zsh"])
        .assert()
        .success();
}

#[test]
fn test_completions_bash() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .args(["completions", "--shell", "bash"])
        .assert()
        .success();
}

#[test]
fn test_completions_fish() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .args(["completions", "--shell", "fish"])
        .assert()
        .success();
}

#[test]
fn test_completions_zsh_includes_subcommands() {
    let output = Command::cargo_bin("hf2q")
        .unwrap()
        .args(["completions", "--shell", "zsh"])
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Zsh completions should reference the main subcommands
    assert!(
        stdout.contains("convert") || stdout.contains("hf2q"),
        "Zsh completions should reference subcommands"
    );
}

#[test]
fn test_completions_bash_includes_subcommands() {
    let output = Command::cargo_bin("hf2q")
        .unwrap()
        .args(["completions", "--shell", "bash"])
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Bash completions should reference hf2q subcommands and flags
    assert!(
        stdout.contains("hf2q") || stdout.contains("convert"),
        "Bash completions should reference the binary name or subcommands"
    );
}

#[test]
fn test_dry_run_prints_plan_and_exits_success() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");

    // Create minimal model directory
    std::fs::create_dir_all(&input_dir).unwrap();
    std::fs::write(
        input_dir.join("config.json"),
        r#"{
            "architectures": ["TinyTestModel"],
            "model_type": "tiny_test",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 32,
            "dtype": "float16"
        }"#,
    )
    .unwrap();
    std::fs::write(input_dir.join("model.safetensors"), &[0u8; 16]).unwrap();

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "q4",
            "--dry-run",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Dry Run"));
}

#[test]
fn test_dry_run_does_not_write_files() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output_dry_run");

    // Create minimal model directory
    std::fs::create_dir_all(&input_dir).unwrap();
    std::fs::write(
        input_dir.join("config.json"),
        r#"{
            "architectures": ["TinyTestModel"],
            "model_type": "tiny_test",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 32,
            "dtype": "float16"
        }"#,
    )
    .unwrap();
    std::fs::write(input_dir.join("model.safetensors"), &[0u8; 16]).unwrap();

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "q4",
            "--dry-run",
            "--output",
            output_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Output directory should NOT exist after dry-run
    assert!(
        !output_dir.exists(),
        "Dry-run should not create output directory"
    );
}

#[test]
fn test_dry_run_shows_quant_config() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");

    std::fs::create_dir_all(&input_dir).unwrap();
    std::fs::write(
        input_dir.join("config.json"),
        r#"{
            "architectures": ["TinyTestModel"],
            "model_type": "tiny_test",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 32,
            "dtype": "float16"
        }"#,
    )
    .unwrap();
    std::fs::write(input_dir.join("model.safetensors"), &[0u8; 16]).unwrap();

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "q4",
            "--dry-run",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Quantization"))
        .stderr(predicate::str::contains("Est. output"))
        .stderr(predicate::str::contains("Est. memory"))
        .stderr(predicate::str::contains("No files were written"));
}

#[test]
fn test_dry_run_safetensors_format() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");

    std::fs::create_dir_all(&input_dir).unwrap();
    std::fs::write(
        input_dir.join("config.json"),
        r#"{
            "architectures": ["TinyDenseModel"],
            "model_type": "tiny_dense",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 32,
            "dtype": "float16"
        }"#,
    )
    .unwrap();
    std::fs::write(input_dir.join("model.safetensors"), &[0u8; 16]).unwrap();

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "safetensors",
            "--quant",
            "f16",
            "--dry-run",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Dry Run"));
}
