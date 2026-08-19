//! Build-time converter provenance without invoking `git` or another helper.
//!
//! Registry packages carry Cargo's `.cargo_vcs_info.json`; source builds can
//! provide one of the explicit CI/release SHA variables. Dirty or unversioned
//! source trees deliberately receive no inferred commit and remote conversion
//! fails closed instead of claiming a provenance identity it cannot prove.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn exact_git_sha(value: &str) -> Option<String> {
    let value = value.trim();
    (value.len() == 40 && value.chars().all(|character| character.is_ascii_hexdigit()))
        .then(|| value.to_ascii_lowercase())
}

fn packaged_vcs_info_path(manifest_dir: &Path) -> Option<PathBuf> {
    let path = manifest_dir.join(".cargo_vcs_info.json");
    path.is_file().then_some(path)
}

fn packaged_vcs_sha(path: &Path) -> Option<String> {
    let contents = fs::read_to_string(path).ok()?;
    let after_key = contents.split_once("\"sha1\"")?.1;
    let after_colon = after_key.split_once(':')?.1;
    let quoted = after_colon.split_once('"')?.1;
    let value = quoted.split_once('"')?.0;
    exact_git_sha(value)
}

fn main() {
    for name in ["GIT_COMMIT_SHA", "VERGEN_GIT_SHA", "GITHUB_SHA"] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    let explicit = ["GIT_COMMIT_SHA", "VERGEN_GIT_SHA", "GITHUB_SHA"]
        .into_iter()
        .filter_map(|name| env::var(name).ok())
        .find_map(|value| exact_git_sha(&value));
    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR").map(std::path::PathBuf::from);
    let packaged_vcs_info = manifest_dir.as_deref().and_then(packaged_vcs_info_path);

    // Cargo treats a missing `rerun-if-changed` input as perpetually stale.
    // `.cargo_vcs_info.json` exists in a packaged crate but not in a normal Git
    // checkout, so only register the file dependency when Cargo supplied it.
    if let Some(path) = packaged_vcs_info.as_deref() {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let commit = explicit.or_else(|| packaged_vcs_info.as_deref().and_then(packaged_vcs_sha));

    if let Some(commit) = commit {
        println!("cargo:rustc-env=HF2Q_BUILD_GIT_SHA={commit}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_sha_validation_is_fail_closed() {
        assert_eq!(exact_git_sha(&"A".repeat(40)), Some("a".repeat(40)));
        assert_eq!(exact_git_sha("abc"), None);
        assert_eq!(exact_git_sha(&"g".repeat(40)), None);
    }

    #[test]
    fn packaged_vcs_info_supplies_registry_commit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".cargo_vcs_info.json");
        fs::write(
            &path,
            format!("{{\"git\":{{\"sha1\":\"{}\"}}}}", "D".repeat(40)),
        )
        .unwrap();
        assert_eq!(packaged_vcs_info_path(dir.path()), Some(path.clone()));
        assert_eq!(packaged_vcs_sha(&path), Some("d".repeat(40)));
    }

    #[test]
    fn missing_packaged_vcs_info_is_not_a_cargo_file_dependency() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(packaged_vcs_info_path(dir.path()), None);
    }
}
