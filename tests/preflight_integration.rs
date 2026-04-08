//! Integration tests for preflight validation and error handling (Epic 2).

mod create_fixture;

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use predicates::prelude::*;

/// Create a tiny test model directory with valid safetensors.
fn setup_tiny_model(dir: &Path) {
    fs::create_dir_all(dir).unwrap();

    fs::write(
        dir.join("config.json"),
        r#"{
            "architectures": ["TinyTestModel"],
            "model_type": "tiny_test",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "intermediate_size": 16,
            "dtype": "float16"
        }"#,
    )
    .unwrap();

    fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();

    let safetensors_data = create_fixture::create_tiny_safetensors();
    fs::write(dir.join("model.safetensors"), safetensors_data).unwrap();
}

/// Create a model with exotic/unsupported layer types.
fn setup_exotic_model(dir: &Path) {
    fs::create_dir_all(dir).unwrap();

    fs::write(
        dir.join("config.json"),
        r#"{
            "architectures": ["ExoticModel"],
            "model_type": "exotic",
            "hidden_size": 8,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "vocab_size": 32,
            "dtype": "float16",
            "layer_types": ["attention", "quantum_entanglement", "attention"]
        }"#,
    )
    .unwrap();

    fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();

    let safetensors_data = create_fixture::create_tiny_safetensors();
    fs::write(dir.join("model.safetensors"), safetensors_data).unwrap();
}

// --- Story 2.1: Pre-flight Validation Engine ---

#[test]
fn test_preflight_validates_input_exists() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            "/nonexistent/model/path",
            "--format",
            "coreml",
            "--quant",
            "q4",
        ])
        .assert()
        .failure()
        .code(3); // input error
}

#[test]
fn test_preflight_validates_config_json_exists() {
    let tmp = tempfile::tempdir().unwrap();
    // Directory exists but has no config.json
    fs::write(tmp.path().join("model.safetensors"), &[0u8; 16]).unwrap();

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            tmp.path().to_str().unwrap(),
            "--format",
            "coreml",
            "--quant",
            "q4",
        ])
        .assert()
        .failure()
        .code(3);
}

#[test]
fn test_preflight_validates_safetensors_exist() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("model");
    fs::create_dir_all(&input_dir).unwrap();
    // Has config.json but no safetensors
    fs::write(
        input_dir.join("config.json"),
        r#"{"architectures": ["Test"], "model_type": "test"}"#,
    )
    .unwrap();

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "coreml",
            "--quant",
            "q4",
        ])
        .assert()
        .failure()
        .code(3)
        .stderr(predicate::str::contains("safetensors"));
}

#[test]
fn test_preflight_sensitive_layers_out_of_range() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    setup_tiny_model(&input_dir);

    // Model has 2 layers, so layer 10 is out of range
    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "coreml",
            "--quant",
            "q4",
            "--sensitive-layers",
            "0-10",
        ])
        .assert()
        .failure()
        .code(3)
        .stderr(predicate::str::contains("layer"));
}

// --- Story 2.2: Unknown Layer Handling ---

#[test]
fn test_unsupported_layers_without_flag_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    setup_exotic_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "coreml",
            "--quant",
            "q4",
        ])
        .assert()
        .failure()
        .code(3)
        .stderr(predicate::str::contains("unsupported-layers=passthrough"));
}

#[test]
fn test_unsupported_layers_with_passthrough_succeeds() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output");
    setup_exotic_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "coreml",
            "--quant",
            "q4",
            "--output",
            output_dir.to_str().unwrap(),
            "--unsupported-layers",
            "passthrough",
        ])
        .assert()
        .success();

    // Verify output was created
    assert!(output_dir.join("config.json").exists());
    assert!(output_dir.join("quantization_config.json").exists());
}

#[test]
fn test_f16_handles_all_layer_types() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output");
    setup_exotic_model(&input_dir);

    // f16 should handle any layer type without needing --unsupported-layers
    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "coreml",
            "--quant",
            "f16",
            "--output",
            output_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
}

// --- Story 2.3: Exit Codes ---

#[test]
fn test_exit_code_0_on_success() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output");
    setup_tiny_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "coreml",
            "--quant",
            "q4",
            "--output",
            output_dir.to_str().unwrap(),
        ])
        .assert()
        .success()
        .code(0);
}

#[test]
fn test_exit_code_3_on_input_error() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            "/nonexistent/path",
            "--format",
            "coreml",
            "--quant",
            "q4",
        ])
        .assert()
        .failure()
        .code(3);
}

#[test]
fn test_yes_flag_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output");
    setup_tiny_model(&input_dir);

    // --yes should be accepted without error
    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "coreml",
            "--quant",
            "q4",
            "--output",
            output_dir.to_str().unwrap(),
            "--yes",
        ])
        .assert()
        .success();
}

// --- Story 2.3: --dry-run ---

#[test]
fn test_dry_run_does_not_write() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output_dry");
    setup_tiny_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "coreml",
            "--quant",
            "q4",
            "--output",
            output_dir.to_str().unwrap(),
            "--dry-run",
        ])
        .assert()
        .success()
        .code(0);

    // Output directory should NOT be created in dry-run mode
    assert!(!output_dir.exists());
}

// --- Story 2.3: Output dir conflict ---

#[test]
fn test_output_dir_conflict_detected() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output");
    setup_tiny_model(&input_dir);

    // Pre-create non-empty output dir
    fs::create_dir_all(&output_dir).unwrap();
    fs::write(output_dir.join("existing_file.txt"), "data").unwrap();

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "coreml",
            "--quant",
            "q4",
            "--output",
            output_dir.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .code(3)
        .stderr(predicate::str::contains("already exists"));
}
