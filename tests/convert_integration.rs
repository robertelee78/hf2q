//! Integration tests for the convert pipeline.

mod create_fixture;

use std::fs;
use std::path::Path;

use assert_cmd::Command;

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

/// Check that a GGUF file was produced in the output directory.
fn assert_has_gguf(output_dir: &Path) {
    let gguf_files: Vec<_> = fs::read_dir(output_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "gguf")
                .unwrap_or(false)
        })
        .collect();
    assert!(!gguf_files.is_empty(), "No GGUF files in {}", output_dir.display());
}

#[test]
fn test_convert_q4_gguf() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output");
    setup_tiny_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert", "--input", input_dir.to_str().unwrap(),
            "--format", "gguf", "--quant", "q4",
            "--output", output_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert_has_gguf(&output_dir);
}

#[test]
fn test_convert_f16_gguf() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output");
    setup_tiny_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert", "--input", input_dir.to_str().unwrap(),
            "--format", "gguf", "--quant", "f16",
            "--output", output_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert_has_gguf(&output_dir);
}

#[test]
fn test_convert_q8_gguf() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output");
    setup_tiny_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert", "--input", input_dir.to_str().unwrap(),
            "--format", "gguf", "--quant", "q8",
            "--output", output_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert_has_gguf(&output_dir);
}

#[test]
fn test_convert_missing_input_fails() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .args(["convert", "--input", "/nonexistent/path", "--format", "gguf", "--quant", "q4"])
        .assert()
        .failure();
}

#[test]
fn test_convert_no_input_fails() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .args(["convert", "--format", "gguf", "--quant", "q4"])
        .assert()
        .failure();
}

#[test]
fn test_convert_auto_quant_resolves() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output_auto");
    setup_tiny_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert", "--input", input_dir.to_str().unwrap(),
            "--format", "gguf", "--quant", "auto",
            "--output", output_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert_has_gguf(&output_dir);
}

#[test]
fn test_convert_safetensors_q4() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output_st");
    setup_tiny_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert", "--input", input_dir.to_str().unwrap(),
            "--format", "safetensors", "--quant", "q4",
            "--output", output_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert!(output_dir.join("model.safetensors").exists());
    assert!(output_dir.join("quantization_config.json").exists());
}

#[test]
fn test_convert_mixed_46_gguf() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output_mixed");
    setup_tiny_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert", "--input", input_dir.to_str().unwrap(),
            "--format", "gguf", "--quant", "mixed-4-6",
            "--sensitive-layers", "0-1",
            "--output", output_dir.to_str().unwrap(), "--skip-quality",
        ])
        .assert()
        .success();

    assert_has_gguf(&output_dir);
}

#[test]
fn test_convert_mixed_26_gguf() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output_mixed26");
    setup_tiny_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert", "--input", input_dir.to_str().unwrap(),
            "--format", "gguf", "--quant", "mixed-2-6",
            "--sensitive-layers", "1",
            "--output", output_dir.to_str().unwrap(), "--skip-quality",
        ])
        .assert()
        .success();

    assert_has_gguf(&output_dir);
}

#[test]
fn test_convert_with_json_report() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output_report");
    setup_tiny_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert", "--input", input_dir.to_str().unwrap(),
            "--format", "gguf", "--quant", "q4",
            "--output", output_dir.to_str().unwrap(),
            "--json-report", "--skip-quality",
        ])
        .assert()
        .success();

    let report_path = output_dir.join("report.json");
    assert!(report_path.exists(), "report.json should exist");

    let content = fs::read_to_string(&report_path).unwrap();
    let report: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert_eq!(report["schema_version"], "1");
    assert_eq!(report["quantization"]["method"], "q4");
    assert_eq!(report["quantization"]["bits"], 4);
}

#[test]
fn test_convert_json_report_to_stdout() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output_stdout_report");
    setup_tiny_model(&input_dir);

    let output = Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert", "--input", input_dir.to_str().unwrap(),
            "--format", "gguf", "--quant", "q4",
            "--output", output_dir.to_str().unwrap(),
            "--json-report", "--yes", "--skip-quality",
        ])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: serde_json::Value = serde_json::from_str(&stdout).expect("stdout should be valid JSON");
    assert_eq!(report["schema_version"], "1");
}

#[test]
fn test_convert_skip_quality_flag() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("output_skipq");
    setup_tiny_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert", "--input", input_dir.to_str().unwrap(),
            "--format", "gguf", "--quant", "q4",
            "--output", output_dir.to_str().unwrap(), "--skip-quality",
        ])
        .assert()
        .success();

    assert_has_gguf(&output_dir);
}
