//! Integration tests for the convert pipeline.

mod create_fixture;

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use predicates::prelude::*;

/// Create a tiny test model directory with valid safetensors.
fn setup_tiny_model(dir: &Path) {
    fs::create_dir_all(dir).unwrap();

    // Config
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

    // Tokenizer files
    fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();

    // Safetensors file
    let safetensors_data = create_fixture::create_tiny_safetensors();
    fs::write(dir.join("model.safetensors"), safetensors_data).unwrap();
}

#[test]
fn test_convert_q4_produces_output() {
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
            "mlx",
            "--quant",
            "q4",
            "--output",
            output_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Verify output files exist
    assert!(output_dir.join("config.json").exists());
    assert!(output_dir.join("quantization_config.json").exists());
    assert!(output_dir.join("tokenizer.json").exists());

    // Verify safetensors shards were written
    let safetensors_files: Vec<_> = fs::read_dir(&output_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();
    assert!(!safetensors_files.is_empty(), "No safetensors files in output");
}

#[test]
fn test_convert_f16_produces_output() {
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
            "mlx",
            "--quant",
            "f16",
            "--output",
            output_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert!(output_dir.join("config.json").exists());
    assert!(output_dir.join("quantization_config.json").exists());
}

#[test]
fn test_convert_q8_produces_output() {
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
            "mlx",
            "--quant",
            "q8",
            "--output",
            output_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert!(output_dir.join("config.json").exists());
}

#[test]
fn test_convert_missing_input_fails() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            "/nonexistent/path",
            "--format",
            "mlx",
            "--quant",
            "q4",
        ])
        .assert()
        .failure();
}

#[test]
fn test_convert_no_input_fails() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .args(["convert", "--format", "mlx", "--quant", "q4"])
        .assert()
        .failure();
}

#[test]
fn test_convert_auto_quant_not_implemented() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    setup_tiny_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "mlx",
            "--quant",
            "auto",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not yet implemented"));
}

#[test]
fn test_convert_coreml_not_implemented() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
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
            "f16",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not yet implemented"));
}
