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
