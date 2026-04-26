//! ADR-005 Phase 3 iter-205 (AC line 5351) — `hf2q cache clear` CLI smoke.
//!
//! `src/serve/cache.rs` carries the in-binary unit tests for the
//! `invalidate` / `invalidate_repo` / `purge` API (run via
//! `cargo test --release --bin hf2q -- serve::cache::`).  Those cover
//! the happy path + error contracts at the library layer.
//!
//! This integration suite exercises the operator-facing surface across
//! the binary boundary by spawning `hf2q cache <action>` as a
//! subprocess and asserting the observable stdout / stderr / exit
//! code.  Because hf2q is a binary crate (no `[lib]` target — see
//! `tests/auto_pipeline_smoke.rs:11-13`), the cross-process route is
//! the only way an external test can reach `cmd_cache`.
//!
//! # Two scopes
//!
//! 1. **Default**: skip with a diagnostic (matches `auto_pipeline_smoke`'s
//!    discipline — keeps `cargo test --release` cheap on dev machines).
//! 2. **`HF2Q_CACHE_CLEAR_E2E=1`**: runs the cache-list / cache-size /
//!    cache-clear subprocess assertions.  Each subprocess uses an
//!    isolated tempdir cache so runs are deterministic; no network.
//!
//! ```bash
//! cargo test --release --test cache_clear
//!   # → all skipped, exit 0
//!
//! HF2Q_CACHE_CLEAR_E2E=1 cargo test --release --test cache_clear -- --test-threads=1
//!   # → list + size + clear paths PASS
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

const ENV_GATE: &str = "HF2Q_CACHE_CLEAR_E2E";

fn skip_unless_gated(name: &str) -> bool {
    if std::env::var(ENV_GATE).as_deref() == Ok("1") {
        return false;
    }
    eprintln!(
        "[skip] {} — set {}=1 to run the cache-clear subprocess smoke",
        name, ENV_GATE
    );
    true
}

fn hf2q_bin() -> PathBuf {
    if let Ok(s) = std::env::var("CARGO_BIN_EXE_hf2q") {
        return PathBuf::from(s);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/release/hf2q")
}

/// Spawn `hf2q cache <args...>` with `HF2Q_CACHE_DIR=cache_dir`.
/// Returns (exit_code, stdout, stderr).  No deadline — the subcommand
/// is in-and-out (no listener bind, no warmup).
fn run_cache(cache_dir: &Path, args: &[&str]) -> (Option<i32>, String, String) {
    let out = Command::new(hf2q_bin())
        .arg("cache")
        .args(args)
        .env("HF2Q_CACHE_DIR", cache_dir)
        .output()
        .expect("spawn hf2q cache");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    (out.status.code(), stdout, stderr)
}

/// Pre-populate a tempdir cache with one fake-but-valid quantized
/// GGUF + manifest entry for `repo_id` at `quant`.  Mirrors the
/// fixture pattern from `auto_pipeline_smoke.rs:215-291` but stays
/// self-contained — no dependency on the binary's library surface
/// (the binary has no `[lib]` target).
fn fab_fixture(cache_dir: &Path, repo_id: &str, quant: &str) -> PathBuf {
    std::fs::create_dir_all(cache_dir).unwrap();
    std::fs::create_dir_all(cache_dir.join("models")).unwrap();
    std::fs::create_dir_all(cache_dir.join("locks")).unwrap();
    let slug = repo_id.replace('/', "__");
    let quant_dir = cache_dir
        .join("models")
        .join(&slug)
        .join("quantized")
        .join(quant);
    std::fs::create_dir_all(&quant_dir).unwrap();
    let gguf_path = quant_dir.join("model.gguf");
    let gguf_bytes = build_minimal_gguf();
    std::fs::write(&gguf_path, &gguf_bytes).unwrap();
    let sha = sha256_hex(&gguf_bytes);
    let bytes = gguf_bytes.len() as u64;

    let manifest_json = format!(
        r#"{{
  "schema_version": 2,
  "models": {{
    "{repo_id}": {{
      "repo_id": "{repo_id}",
      "revision": "deadbeef00000000000000000000000000000000",
      "source": {{ "kind": "local", "path": "{src}", "sha256": "{src_sha}" }},
      "quantizations": {{
        "{quant}": {{
          "quant_type": "{quant}",
          "gguf_path": "{gguf}",
          "mmproj_path": null,
          "bytes": {bytes},
          "sha256": "{sha}",
          "quantized_at_secs": 12345,
          "quantized_by_version": "test-fixture"
        }}
      }},
      "last_accessed_secs": 12345,
      "source_shards": []
    }}
  }}
}}"#,
        src = cache_dir.join("source").display(),
        src_sha = "0".repeat(64),
        gguf = gguf_path.display(),
    );

    // Append (vs overwrite) is wrong here — we always start with a
    // fresh fixture per test, so write straight over.
    std::fs::write(cache_dir.join("manifest.json"), &manifest_json).unwrap();
    gguf_path
}

// ---------------------------------------------------------------------
// Default-on sanity — confirms the binary documents the surface under
// clap.  These are cheap; no `--release` build product gets exercised
// beyond `cache --help` / `cache clear --help` parsing.
// ---------------------------------------------------------------------

#[test]
fn binary_cache_help_documents_subcommands() {
    let out = Command::new(hf2q_bin())
        .args(["cache", "--help"])
        .output()
        .expect("spawn hf2q cache --help");
    assert!(out.status.success(), "cache --help should succeed");
    let body = String::from_utf8_lossy(&out.stdout);
    assert!(body.contains("list"), "cache --help must list `list`");
    assert!(body.contains("size"), "cache --help must list `size`");
    assert!(body.contains("clear"), "cache --help must list `clear`");
}

#[test]
fn binary_cache_clear_help_documents_flags() {
    let out = Command::new(hf2q_bin())
        .args(["cache", "clear", "--help"])
        .output()
        .expect("spawn hf2q cache clear --help");
    assert!(out.status.success());
    let body = String::from_utf8_lossy(&out.stdout);
    assert!(body.contains("--model"), "must document --model");
    assert!(body.contains("--quant"), "must document --quant");
    assert!(body.contains("--all"), "must document --all");
    assert!(body.contains("--yes"), "must document --yes");
}

// ---------------------------------------------------------------------
// E2E #1 — cache list / size on empty cache (env-gated)
// ---------------------------------------------------------------------

#[test]
fn e2e_list_and_size_on_empty_cache() {
    if skip_unless_gated("e2e_list_and_size_on_empty_cache") {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let cache_dir = tmp.path().join("hf2q_cache");

    let (code, stdout, stderr) = run_cache(&cache_dir, &["list"]);
    assert_eq!(code, Some(0), "list must succeed; stderr: {stderr}");
    assert!(
        stdout.contains("cache empty") || stdout.contains("(cache empty"),
        "empty list must say so; got:\n{stdout}"
    );

    let (code, stdout, stderr) = run_cache(&cache_dir, &["size"]);
    assert_eq!(code, Some(0), "size must succeed; stderr: {stderr}");
    assert!(stdout.contains("0 bytes"), "empty size must be 0; got:\n{stdout}");
}

// ---------------------------------------------------------------------
// E2E #2 — cache clear --model <repo> --quant <q> removes one entry
// ---------------------------------------------------------------------

#[test]
fn e2e_clear_specific_quant_removes_only_that_entry() {
    if skip_unless_gated("e2e_clear_specific_quant_removes_only_that_entry") {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let cache_dir = tmp.path().join("hf2q_cache");
    let repo = "smoke-fixture/repo-205";
    let gguf = fab_fixture(&cache_dir, repo, "Q8_0");
    assert!(gguf.is_file(), "fixture must land");

    let (code, stdout, stderr) =
        run_cache(&cache_dir, &["clear", "--model", repo, "--quant", "Q8_0"]);
    assert_eq!(code, Some(0), "clear must succeed; stderr: {stderr}");
    assert!(
        stdout.contains("cleared") && stdout.contains("Q8_0"),
        "stdout must report what was cleared; got:\n{stdout}"
    );
    assert!(
        !gguf.exists(),
        "GGUF must be removed after clear --quant Q8_0"
    );

    // Manifest must still parse and now have an empty `quantizations`
    // map for the repo (the model entry survives — only the quant
    // was removed).
    let manifest =
        std::fs::read_to_string(cache_dir.join("manifest.json")).unwrap();
    assert!(manifest.contains(repo));
    assert!(!manifest.contains("\"Q8_0\""));
}

// ---------------------------------------------------------------------
// E2E #3 — cache clear --model <repo> (no --quant) removes the repo
// ---------------------------------------------------------------------

#[test]
fn e2e_clear_repo_removes_model_dir() {
    if skip_unless_gated("e2e_clear_repo_removes_model_dir") {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let cache_dir = tmp.path().join("hf2q_cache");
    let repo = "smoke-fixture/repo-205-all";
    let gguf = fab_fixture(&cache_dir, repo, "Q8_0");
    let slug = repo.replace('/', "__");
    let model_dir = cache_dir.join("models").join(&slug);
    assert!(model_dir.is_dir());

    let (code, stdout, stderr) = run_cache(&cache_dir, &["clear", "--model", repo]);
    assert_eq!(code, Some(0), "clear must succeed; stderr: {stderr}");
    assert!(
        stdout.contains("cleared") && stdout.contains(repo),
        "stdout must name the repo; got:\n{stdout}"
    );

    assert!(!gguf.exists());
    assert!(!model_dir.exists(), "model dir must be removed");
    let manifest =
        std::fs::read_to_string(cache_dir.join("manifest.json")).unwrap();
    assert!(
        !manifest.contains(repo),
        "manifest must no longer reference the repo"
    );
}

// ---------------------------------------------------------------------
// E2E #4 — cache clear --all without --yes refuses with named error
// ---------------------------------------------------------------------

#[test]
fn e2e_clear_all_without_yes_refuses() {
    if skip_unless_gated("e2e_clear_all_without_yes_refuses") {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let cache_dir = tmp.path().join("hf2q_cache");
    let repo = "smoke-fixture/repo-205-refuse";
    let gguf = fab_fixture(&cache_dir, repo, "Q8_0");

    let (code, _stdout, stderr) = run_cache(&cache_dir, &["clear", "--all"]);
    assert_ne!(code, Some(0), "must NOT exit zero without --yes");
    assert!(
        stderr.contains("--yes"),
        "stderr must explain the missing --yes; got:\n{stderr}"
    );
    // GGUF survives — refusal must not delete anything.
    assert!(gguf.exists(), "fixture must NOT have been removed");
}

// ---------------------------------------------------------------------
// E2E #5 — cache clear --all --yes purges everything
// ---------------------------------------------------------------------

#[test]
fn e2e_clear_all_with_yes_purges_everything() {
    if skip_unless_gated("e2e_clear_all_with_yes_purges_everything") {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let cache_dir = tmp.path().join("hf2q_cache");
    // Two repos so we can prove the purge nuked more than one.
    let a_gguf = fab_fixture(&cache_dir, "smoke-fixture/repo-a", "Q8_0");
    // fab_fixture overwrites manifest.json — second call would clobber
    // the first repo. Hand-craft a multi-model manifest instead.
    let b_repo = "smoke-fixture/repo-b";
    let b_slug = b_repo.replace('/', "__");
    let b_quant_dir = cache_dir
        .join("models")
        .join(&b_slug)
        .join("quantized")
        .join("Q4_K_M");
    std::fs::create_dir_all(&b_quant_dir).unwrap();
    let b_gguf = b_quant_dir.join("model.gguf");
    std::fs::write(&b_gguf, b"second-model-bytes").unwrap();
    let b_sha = sha256_hex(b"second-model-bytes");
    let multi = format!(
        r#"{{
  "schema_version": 2,
  "models": {{
    "smoke-fixture/repo-a": {{
      "repo_id": "smoke-fixture/repo-a",
      "revision": "rev-a",
      "source": {{ "kind": "local", "path": "/p", "sha256": "{zeros}" }},
      "quantizations": {{
        "Q8_0": {{
          "quant_type": "Q8_0",
          "gguf_path": "{a}",
          "mmproj_path": null,
          "bytes": {a_bytes},
          "sha256": "{a_sha}",
          "quantized_at_secs": 1,
          "quantized_by_version": "test"
        }}
      }},
      "last_accessed_secs": 1,
      "source_shards": []
    }},
    "{b_repo}": {{
      "repo_id": "{b_repo}",
      "revision": "rev-b",
      "source": {{ "kind": "local", "path": "/p", "sha256": "{zeros}" }},
      "quantizations": {{
        "Q4_K_M": {{
          "quant_type": "Q4_K_M",
          "gguf_path": "{b}",
          "mmproj_path": null,
          "bytes": 18,
          "sha256": "{b_sha}",
          "quantized_at_secs": 2,
          "quantized_by_version": "test"
        }}
      }},
      "last_accessed_secs": 2,
      "source_shards": []
    }}
  }}
}}"#,
        zeros = "0".repeat(64),
        a = a_gguf.display(),
        a_bytes = std::fs::metadata(&a_gguf).unwrap().len(),
        a_sha = sha256_hex(&std::fs::read(&a_gguf).unwrap()),
        b_repo = b_repo,
        b = b_gguf.display(),
        b_sha = b_sha,
    );
    std::fs::write(cache_dir.join("manifest.json"), &multi).unwrap();

    let (code, stdout, stderr) =
        run_cache(&cache_dir, &["clear", "--all", "--yes"]);
    assert_eq!(code, Some(0), "purge must succeed; stderr: {stderr}");
    assert!(
        stdout.contains("purged"),
        "stdout must say purged; got:\n{stdout}"
    );

    // Both GGUFs are gone.
    assert!(!a_gguf.exists());
    assert!(!b_gguf.exists());
    // models/ is empty.
    let count = std::fs::read_dir(cache_dir.join("models")).unwrap().count();
    assert_eq!(count, 0, "models/ must be empty");
    // Manifest reset to schema-v2 empty.
    let manifest =
        std::fs::read_to_string(cache_dir.join("manifest.json")).unwrap();
    assert!(manifest.contains("\"schema_version\""));
    assert!(!manifest.contains("smoke-fixture/repo-a"));
    assert!(!manifest.contains("smoke-fixture/repo-b"));
}

// ---------------------------------------------------------------------
// E2E #6 — cache clear with no flags refuses with usage error
// ---------------------------------------------------------------------

#[test]
fn e2e_clear_without_flags_refuses_with_usage() {
    if skip_unless_gated("e2e_clear_without_flags_refuses_with_usage") {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let cache_dir = tmp.path().join("hf2q_cache");

    let (code, _stdout, stderr) = run_cache(&cache_dir, &["clear"]);
    assert_ne!(code, Some(0), "must NOT exit zero with no flags");
    assert!(
        stderr.contains("--model") || stderr.contains("--all"),
        "stderr must point operator at the right flags; got:\n{stderr}"
    );
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

/// Smallest legal GGUF header — magic + v3 + zero tensors + zero
/// metadata.  Won't parse downstream, but the cache APIs only care
/// about file existence + the SHA matching the manifest entry.
fn build_minimal_gguf() -> Vec<u8> {
    let mut buf = Vec::with_capacity(24);
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf
}

/// SHA-256 of a byte slice as lowercase hex.  Standalone so the smoke
/// has zero dependency on the binary's `cache::sha256_file`.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut s = String::with_capacity(64);
    for b in out {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
