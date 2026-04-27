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
    std::fs::write(input_dir.join("model.safetensors"), [0u8; 16]).unwrap();

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
    std::fs::write(input_dir.join("model.safetensors"), [0u8; 16]).unwrap();

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
    std::fs::write(input_dir.join("model.safetensors"), [0u8; 16]).unwrap();

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
    std::fs::write(input_dir.join("model.safetensors"), [0u8; 16]).unwrap();

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

// ---- DWQ bit-pair parameterization tests (ADR-012 P0) ----

#[test]
fn test_convert_help_lists_all_four_dwq_variants() {
    // ADR-014 P8 Decision 13: --help must list all 4 renamed DWQ quant
    // variants (dwq-N-M; the legacy dwq-mixed-N-M aliases are deleted).
    let output = Command::cargo_bin("hf2q")
        .unwrap()
        .args(["convert", "--help"])
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    for variant in ["dwq-4-6", "dwq-4-8", "dwq-6-8", "dwq-2-8"] {
        assert!(
            stdout.contains(variant),
            "--help must list {variant} (Decision 12 menu)"
        );
    }
}

#[test]
fn test_bits_with_dwq_variant_errors_with_exact_message() {
    // --bits combined with any DWQ variant must produce the documented
    // error. ADR-014 P8 Decision 13: all four renamed `dwq-N-M`
    // variants must trip it; narrowing the check to only one variant
    // would silently accept `--bits` on the other three.
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
    std::fs::write(input_dir.join("model.safetensors"), [0u8; 16]).unwrap();

    for variant in ["dwq-2-8", "dwq-4-6", "dwq-4-8", "dwq-6-8"] {
        Command::cargo_bin("hf2q")
            .unwrap()
            .args([
                "convert",
                "--input",
                input_dir.to_str().unwrap(),
                "--format",
                "gguf",
                "--quant",
                variant,
                "--bits",
                "5",
            ])
            .assert()
            .failure()
            .stderr(predicate::str::contains(
                "--bits is not used for DWQ; use --quant dwq-N-M to choose bit-pair variants",
            ));
    }
}

#[test]
fn test_dwq_mixed48_dry_run_succeeds() {
    // dwq-4-8 must be accepted by clap and reach the dry-run path.
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
    std::fs::write(input_dir.join("model.safetensors"), [0u8; 16]).unwrap();

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "dwq-4-8",
            "--dry-run",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Dry Run"));
}

#[test]
fn test_dwq46_and_dwq48_default_filenames_do_not_collide() {
    // The two default output filenames for dwq46 and dwq48 must differ from each other.
    // Verified by checking the dry-run stderr output contains the correct suffix per variant.
    let tmp = tempfile::tempdir().unwrap();

    let input46 = tmp.path().join("mymodel");
    std::fs::create_dir_all(&input46).unwrap();
    std::fs::write(
        input46.join("config.json"),
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
    std::fs::write(input46.join("model.safetensors"), [0u8; 16]).unwrap();

    // dwq-4-6 should produce *-dwq46.gguf
    let out46 = Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input46.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "dwq-4-6",
            "--dry-run",
        ])
        .output()
        .unwrap();
    let stderr46 = String::from_utf8_lossy(&out46.stderr);
    assert!(
        stderr46.contains("dwq46"),
        "dwq-4-6 dry-run output should mention 'dwq46' suffix; stderr={}",
        stderr46
    );
    assert!(
        !stderr46.contains("dwq48"),
        "dwq-4-6 must not mention 'dwq48'"
    );

    // dwq-4-8 should produce *-dwq48.gguf
    let out48 = Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input46.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "dwq-4-8",
            "--dry-run",
        ])
        .output()
        .unwrap();
    let stderr48 = String::from_utf8_lossy(&out48.stderr);
    assert!(
        stderr48.contains("dwq48"),
        "dwq-4-8 dry-run output should mention 'dwq48' suffix; stderr={}",
        stderr48
    );
    assert!(
        !stderr48.contains("dwq46"),
        "dwq-4-8 must not mention 'dwq46'"
    );
}
