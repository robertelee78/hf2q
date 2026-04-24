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

/// ADR-012 Decision 15 second AC bullet: "Test that Gemma-4 conversion
/// still produces the same sidecar set it did before." We cover the
/// qwen35 + qwen35moe paths in their own integration suites; this test
/// anchors the non-qwen35 path so any future edit that accidentally
/// arch-gates `copy_sidecars` (making it a qwen-only behaviour) trips
/// immediately. TinyTestModel is the generic, non-qwen35 carrier.
#[test]
fn test_convert_sidecars_preserved_on_non_qwen35_path() {
    use std::fs as stdfs;
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input_sidecars");
    let output_dir = tmp.path().join("output_sidecars");
    setup_tiny_model(&input_dir);

    // Overwrite the minimal sidecars setup_tiny_model wrote + add the rest
    // with distinctive content so byte-identity is meaningfully checked.
    let sidecar_contents: &[(&str, &str)] = &[
        ("chat_template.jinja", "{% for m in messages %}USER:{{ m.content }}{% endfor %}"),
        ("tokenizer.json", "{\"version\":\"1.0\",\"model\":{\"type\":\"BPE\"}}"),
        ("tokenizer_config.json", "{\"model_max_length\":4096,\"bos_token\":\"<s>\"}"),
        ("config.json", "OVERWRITE_ME_PRESERVE_CHECK"), // overwritten below
        ("generation_config.json", "{\"temperature\":0.7,\"top_p\":0.9}"),
        ("special_tokens_map.json", "{\"bos_token\":\"<s>\",\"eos_token\":\"</s>\"}"),
    ];
    for (name, body) in sidecar_contents {
        if *name == "config.json" {
            // Preserve the valid config.json setup_tiny_model wrote.
            continue;
        }
        stdfs::write(input_dir.join(name), body).unwrap();
    }

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

    // Every sidecar we wrote to input must appear in output with
    // byte-identical content. config.json included — its content was
    // written by setup_tiny_model.
    let expected_files = [
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
    ];
    for name in expected_files {
        let src = input_dir.join(name);
        let dst = output_dir.join(name);
        assert!(
            dst.exists(),
            "sidecar '{name}' missing from non-qwen35 output — \
             copy_sidecars may have been arch-gated (Decision 15 regression)"
        );
        let src_bytes = stdfs::read(&src).unwrap();
        let dst_bytes = stdfs::read(&dst).unwrap();
        assert_eq!(
            src_bytes, dst_bytes,
            "sidecar '{name}' content is not byte-identical \
             (Decision 15 byte-preservation AC broken)"
        );
    }
}

/// Missing-sidecar silent-skip AC — Decision 15 says "If any are missing
/// in the HF repo, skip silently (not all models ship all sidecars)."
/// Proves convert succeeds and emits whatever sidecars exist, without
/// whinging about the absent ones.
#[test]
fn test_convert_sidecars_missing_silently_skipped_on_non_qwen35_path() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("input_no_sidecars");
    let output_dir = tmp.path().join("output_no_sidecars");
    setup_tiny_model(&input_dir);
    // setup_tiny_model wrote only tokenizer.json + tokenizer_config.json —
    // the other 4 sidecars are deliberately absent.

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

    // The 4 sidecars that were never in input must NOT appear in output.
    for name in ["chat_template.jinja", "generation_config.json", "special_tokens_map.json"] {
        assert!(
            !output_dir.join(name).exists(),
            "sidecar '{name}' appeared in output when not present in input \
             — copy_sidecars must not fabricate files"
        );
    }
}
