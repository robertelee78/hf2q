//! Integration tests for the info subcommand.

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use predicates::prelude::*;

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
            "vocab_size": 32,
            "dtype": "float16"
        }"#,
    )
    .unwrap();
}

#[test]
fn test_info_displays_metadata() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("model");
    setup_tiny_model(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args(["info", "--input", input_dir.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("TinyTestModel"))
        .stdout(predicate::str::contains("tiny_test"))
        .stdout(predicate::str::contains("8"))  // hidden_size
        .stdout(predicate::str::contains("2"));  // layers
}

#[test]
fn test_info_missing_directory() {
    Command::cargo_bin("hf2q")
        .unwrap()
        .args(["info", "--input", "/nonexistent/path"])
        .assert()
        .failure();
}

#[test]
fn test_info_no_config_json() {
    let tmp = tempfile::tempdir().unwrap();

    Command::cargo_bin("hf2q")
        .unwrap()
        .args(["info", "--input", tmp.path().to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("config.json"));
}

#[test]
fn test_info_moe_model() {
    let tmp = tempfile::tempdir().unwrap();
    let input_dir = tmp.path().join("model");
    fs::create_dir_all(&input_dir).unwrap();

    fs::write(
        input_dir.join("config.json"),
        r#"{
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "dtype": "bfloat16",
            "text_config": {
                "hidden_size": 2816,
                "num_hidden_layers": 30,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "vocab_size": 262144,
                "num_experts": 128,
                "top_k_experts": 8,
                "intermediate_size": 2112,
                "layer_types": ["sliding_attention", "full_attention"]
            }
        }"#,
    )
    .unwrap();

    Command::cargo_bin("hf2q")
        .unwrap()
        .args(["info", "--input", input_dir.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Gemma4ForConditionalGeneration"))
        .stdout(predicate::str::contains("128"))  // experts
        .stdout(predicate::str::contains("MoE"));
}
