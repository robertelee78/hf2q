//! ADR-005 Phase 3 (lines 901–917) item 4/4 — auto-pipeline closer.
//!
//! `hf2q serve --model <repo-or-path>` end-to-end glue that chains:
//!
//! - W51 iter-201 [`crate::serve::quant_select`] (hardware → quant type)
//! - W70 iter-202 [`crate::serve::cache`] (manifest + locks + atomic writes)
//! - W71 iter-203 [`crate::input::integrity`] (per-shard SHA-256 verify)
//!
//! into one entry point: [`resolve_or_prepare_model`].  Given either a
//! filesystem GGUF path or a HuggingFace repo-id, returns a path to a
//! ready-to-load `.gguf` file — downloading + integrity-checking +
//! quantizing as needed, with no manual operator steps.
//!
//! # Design decisions (Chesterton's fence)
//!
//! - **CLI surface unchanged**: `ServeArgs::model` stays `Option<PathBuf>`.
//!   clap's `PathBuf` parser accepts any string, so `--model
//!   google/gemma-4-27b-it` still parses cleanly; we classify in code.
//! - **HF detection heuristic**: a string is a repo-id iff (a) does not
//!   exist on disk AND (b) matches `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$`.
//!   This is the same shape HF uses (`org/repo-name`) and excludes
//!   absolute paths (start `/`), relative paths (start `./` or `../`),
//!   Windows-shaped paths, and anything with an extension.
//! - **Quantize via subprocess**: per the iter-204 brief, we `Command`-spawn
//!   `hf2q convert` rather than calling the convert library directly. The
//!   convert/ tree is the OOM session's territory (see CLAUDE.md fence
//!   list), and a subprocess boundary keeps the auto-pipeline's failure
//!   modes — non-zero exit, stderr — uniformly observable.
//! - **K-quant emit gap (ADR-014 P7)**: the W51 selection table returns
//!   `Q8_0 / Q6_K / Q4_K_M / Q3_K_M`, but the convert CLI surface today
//!   only exposes `q4` / `q8` / etc.  K-quant emit is mid-port (ADR-014
//!   P7).  Until P7 closes, [`map_quant_to_cli`] degrades K-quant table
//!   outputs to the closest available legacy quant and logs the choice
//!   verbatim so an operator can see it in `info` logs.
//! - **HF cache reuse**: when the source has already been downloaded by
//!   `hf-hub`, the auto-pipeline detects the snapshot via
//!   [`crate::serve::cache::ModelCache::detect_hf_hub_source`] and skips
//!   the re-download.  Bytes never leave `~/.cache/huggingface/hub/`.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};

use super::cache::{
    cache_model_path, sha256_file, ModelCache, QuantEntry, SourcePointer,
};
use super::quant_select::{select_quant, GpuInfo, QuantType};
use crate::input::integrity::verify_repo;
use crate::intelligence::hardware::HardwareProfile;

/// Classify a `--model` argument into one of the two supported shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelInput {
    /// Filesystem path that exists; pass through unchanged.
    Path(PathBuf),
    /// HuggingFace repo-id (`org/repo-name`); needs the auto-pipeline.
    HfRepoId(String),
}

/// Classify a `--model` arg as a path or a HF repo-id.
///
/// Decision order (cheapest first):
/// 1. If the arg, treated as a path, exists on disk → `Path`.
/// 2. Else if the arg matches the HF repo-id shape → `HfRepoId`.
/// 3. Else → `Err` with both checks named so the user can fix the input.
///
/// The HF repo-id shape: `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$`.  Single `/`
/// separator; ASCII alphanumerics + `.`/`_`/`-` only (HF practice; the
/// canonical character class is documented at huggingface.co).  Multi-`/`
/// inputs (subpaths within a repo, anything with a directory component)
/// are rejected — the auto-pipeline operates on whole repos, not files
/// within them.
pub fn classify_model_input(arg: &str) -> Result<ModelInput> {
    if arg.is_empty() {
        return Err(anyhow!("--model is empty"));
    }
    let p = Path::new(arg);
    if p.exists() {
        return Ok(ModelInput::Path(p.to_path_buf()));
    }
    if looks_like_hf_repo_id(arg) {
        return Ok(ModelInput::HfRepoId(arg.to_string()));
    }
    Err(anyhow!(
        "--model={arg} does not exist on disk and is not a valid \
         HuggingFace repo-id (expected `org/repo-name` with only \
         ASCII alphanumerics, '.', '_', '-')"
    ))
}

/// Pure-text check for the HF repo-id shape.  Public for unit tests.
pub fn looks_like_hf_repo_id(arg: &str) -> bool {
    // Reject leading separators / dots so absolute / relative paths
    // never match.  Also reject backslash so Windows-shaped paths get
    // a clean error message rather than an attempted hub fetch.
    if arg.is_empty()
        || arg.starts_with('/')
        || arg.starts_with('.')
        || arg.starts_with('\\')
        || arg.contains('\\')
    {
        return false;
    }
    let parts: Vec<&str> = arg.split('/').collect();
    if parts.len() != 2 {
        return false;
    }
    let valid_part = |s: &str| {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
    };
    valid_part(parts[0]) && valid_part(parts[1])
}

/// Map the W51 selection table's `QuantType` to the convert CLI's
/// `--quant <name>` argument.
///
/// Until ADR-014 P7 closes (K-quant emit on CLI), K-quant table outputs
/// degrade to the closest available legacy quant.  The choice is logged
/// at `info` so operators can see what's happening and we keep a record
/// for the cutover when P7 lands.
///
/// Returned `&'static str` matches the values clap's `QuantMethod`
/// `value_enum` accepts (or its `alias =` short form): `q4` (= Q4_0),
/// `q8` (= Q8_0).
fn map_quant_to_cli(quant: QuantType) -> &'static str {
    match quant {
        // Q8_0 maps cleanly — same byte layout, same name.
        QuantType::Q8_0 => "q8",
        // Q6_K is not on the convert CLI yet (ADR-014 P7); Q8_0 is the
        // closest fidelity available today.  Bigger files than the
        // table assumes, but never lower fidelity than the operator
        // would tolerate.
        QuantType::Q6_K => "q8",
        // Q4_K_M is the K-quant trajectory for Q4_0 today.  Same 4 bpw
        // ballpark, same dispatch in candle/llama.cpp readers.
        QuantType::Q4_K_M => "q4",
        // Q3_K_M is < 4 bpw — there is no legacy 3-bit quant on the
        // convert CLI.  Q4_0 is the safe minimum until P7 lands Q3_K
        // emit.
        QuantType::Q3_K_M => "q4",
    }
}

/// Returned by [`resolve_or_prepare_model`].  Carries the path the serve
/// loader needs plus enough metadata for the caller to log + observe
/// what the pipeline did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    /// Final filesystem path to a GGUF ready for `mlx_native::gguf::GgufFile::open`.
    pub gguf_path: PathBuf,
    /// `Some` when the auto-pipeline ran (HF input); `None` for a path passthrough.
    pub repo_id: Option<String>,
    /// `Some` when the auto-pipeline ran; `None` for a path passthrough.
    pub quant: Option<QuantType>,
    /// `true` if the cached entry was used as-is (lookup + verify hit);
    /// `false` if a download/quantize ran (or N/A for a path passthrough).
    pub from_cache: bool,
}

/// End-to-end resolution.  Either passes a filesystem GGUF through, or
/// runs the full auto-pipeline (download → integrity → quantize → record
/// → return) for a HF repo-id.
///
/// `model_arg` accepts either:
/// - A path string (absolute or relative) that exists on disk; returned as-is.
/// - A HF repo-id like `google/gemma-4-27b-it`; downloads + quantizes if
///   not already cached.
///
/// The `cache` argument is `&mut` because the success path mutates the
/// manifest (records source / records quantized / touches LRU).  Pass
/// `&mut ModelCache::open()?` from the caller.
///
/// `no_integrity == true` mirrors the `--no-integrity` CLI flag (off by
/// default, ON only when the operator explicitly opts out).
pub fn resolve_or_prepare_model(
    model_arg: &str,
    cache: &mut ModelCache,
    hw: &HardwareProfile,
    no_integrity: bool,
) -> Result<ResolvedModel> {
    let input = classify_model_input(model_arg)?;
    match input {
        ModelInput::Path(p) => Ok(ResolvedModel {
            gguf_path: p,
            repo_id: None,
            quant: None,
            from_cache: false,
        }),
        ModelInput::HfRepoId(repo_id) => {
            run_auto_pipeline(&repo_id, cache, hw, no_integrity)
        }
    }
}

/// HF-repo-id branch — extracted so unit tests can cover classification +
/// pipeline separately.
fn run_auto_pipeline(
    repo_id: &str,
    cache: &mut ModelCache,
    hw: &HardwareProfile,
    no_integrity: bool,
) -> Result<ResolvedModel> {
    let info = GpuInfo::from_hardware_profile(hw);
    let quant = select_quant(&info)
        .with_context(|| format!("hardware → quant selection for {repo_id}"))?;

    tracing::info!(
        repo = repo_id,
        memory_gib = info.memory_gib_floor(),
        quant = quant.as_str(),
        "auto-pipeline: hardware → quant selected"
    );

    // Pre-lock fast path: if cache hit + verify PASS, return immediately.
    if let Some(hit) = lookup_and_verify(cache, repo_id, quant, no_integrity) {
        cache.touch(repo_id).ok(); // best-effort LRU bump
        return Ok(hit);
    }

    // Cache miss (or verify-fail / corruption).  Acquire write lock and
    // re-check after; another concurrent process may have populated it.
    let _lock = cache
        .lock_quant(repo_id, quant)
        .with_context(|| format!("acquire cache write lock for {repo_id}@{}", quant.as_str()))?;

    if let Some(hit) = lookup_and_verify(cache, repo_id, quant, no_integrity) {
        cache.touch(repo_id).ok();
        return Ok(hit);
    }

    // Step 1: source bytes.  Reuse hf-hub cache if it already has them;
    // otherwise download.
    let snapshot = ensure_source_present(cache, repo_id, no_integrity)?;

    // Step 2: invoke convert subprocess to produce the GGUF.
    let target_gguf = cache_model_path(cache.root(), repo_id, quant)?;
    if let Some(parent) = target_gguf.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create quant dir: {}", parent.display()))?;
    }
    run_convert_subprocess(&snapshot.local_dir, &target_gguf, quant, no_integrity)?;

    // Step 3: hash + record + flush manifest.
    let bytes = std::fs::metadata(&target_gguf)
        .with_context(|| format!("stat produced GGUF: {}", target_gguf.display()))?
        .len();
    let sha256 = sha256_file(&target_gguf)?;
    let entry = QuantEntry {
        quant_type: quant.as_str().to_string(),
        gguf_path: target_gguf.clone(),
        mmproj_path: None,
        bytes,
        sha256,
        quantized_at_secs: secs_since_epoch(),
        quantized_by_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    cache
        .record_quantized(repo_id, entry)
        .with_context(|| format!("record_quantized for {repo_id}@{}", quant.as_str()))?;

    tracing::info!(
        repo = repo_id,
        quant = quant.as_str(),
        path = %target_gguf.display(),
        bytes,
        "auto-pipeline: cache populated"
    );

    Ok(ResolvedModel {
        gguf_path: target_gguf,
        repo_id: Some(repo_id.to_string()),
        quant: Some(quant),
        from_cache: false,
    })
}

/// Try the cache; return `Some(hit)` only when the manifest entry exists
/// AND the on-disk SHA-256 still matches (or `--no-integrity` is set, in
/// which case the verify is skipped with a warn).
fn lookup_and_verify(
    cache: &ModelCache,
    repo_id: &str,
    quant: QuantType,
    no_integrity: bool,
) -> Option<ResolvedModel> {
    let entry = cache.lookup(repo_id, quant)?;
    let path = entry.gguf_path.clone();
    if !path.exists() {
        tracing::warn!(
            repo = repo_id,
            quant = quant.as_str(),
            path = %path.display(),
            "cache manifest entry references missing GGUF; re-quantizing"
        );
        return None;
    }
    if no_integrity {
        tracing::warn!(
            repo = repo_id,
            quant = quant.as_str(),
            "auto-pipeline: --no-integrity set; skipping cached SHA-256 verify (NOT recommended)"
        );
    } else if let Err(e) = cache.verify_quantized(repo_id, quant) {
        tracing::warn!(
            repo = repo_id,
            quant = quant.as_str(),
            error = %e,
            "cached GGUF failed integrity check; re-quantizing"
        );
        return None;
    }
    tracing::info!(
        repo = repo_id,
        quant = quant.as_str(),
        path = %path.display(),
        "auto-pipeline: cache hit"
    );
    Some(ResolvedModel {
        gguf_path: path,
        repo_id: Some(repo_id.to_string()),
        quant: Some(quant),
        from_cache: true,
    })
}

/// Source-bytes invariant: after this returns, `<snapshot.local_dir>`
/// holds the unquantized HF snapshot AND the cache manifest carries a
/// matching `SourcePointer::HfHub` entry (with per-shard integrity
/// records when integrity is on).
struct SnapshotInfo {
    local_dir: PathBuf,
}

fn ensure_source_present(
    cache: &mut ModelCache,
    repo_id: &str,
    no_integrity: bool,
) -> Result<SnapshotInfo> {
    // Cheap pre-check — skip download if hf-hub already has the snapshot.
    let detected = ModelCache::detect_hf_hub_source(repo_id);
    let (local_dir, revision) = if let Some(snap) = detected {
        tracing::info!(
            repo = repo_id,
            path = %snap.path.display(),
            revision = %snap.revision,
            "auto-pipeline: hf-hub snapshot already present; skipping download"
        );
        (snap.path, snap.revision)
    } else {
        tracing::info!(repo = repo_id, "auto-pipeline: downloading from HF Hub");
        let progress = crate::progress::ProgressReporter::new();
        let dir = crate::input::hf_download::download_model(repo_id, &progress)
            .map_err(|e| anyhow!("HF download for {repo_id}: {e}"))?;
        // Revision is the snapshot dir's name when it looks like a 40-hex
        // commit SHA — same lift as `cli::resolve_convert_config`.
        let revision = dir
            .file_name()
            .and_then(|n| n.to_str())
            .filter(|s| s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit()))
            .unwrap_or("main")
            .to_string();
        (dir, revision)
    };

    // Integrity verify (W71) + record into manifest.
    let source = SourcePointer::HfHub {
        path: local_dir.clone(),
        revision: revision.clone(),
    };
    if no_integrity {
        tracing::warn!(
            repo = repo_id,
            "auto-pipeline: --no-integrity set; skipping HF integrity verify (NOT recommended)"
        );
        cache
            .record_source(repo_id, &revision, source)
            .with_context(|| format!("record_source for {repo_id}"))?;
    } else {
        let shards = verify_repo(repo_id, &revision, &local_dir)
            .map_err(|e| anyhow!("HF integrity check for {repo_id}@{revision}: {e}"))?;
        cache
            .record_source_with_shards(repo_id, &revision, source, shards)
            .with_context(|| format!("record_source_with_shards for {repo_id}"))?;
    }

    Ok(SnapshotInfo { local_dir })
}

/// Spawn `hf2q convert` to produce the cached GGUF.
///
/// Subprocess boundary preserves the convert/ tree's fence (ADR-014 P7
/// is actively editing src/quantize/, src/backends/gguf.rs, etc.).  Failure
/// is the subprocess returning non-zero: the auto-pipeline propagates
/// stderr verbatim so an operator sees the convert-side error message.
fn run_convert_subprocess(
    snapshot_dir: &Path,
    target_gguf: &Path,
    quant: QuantType,
    no_integrity: bool,
) -> Result<()> {
    let bin = std::env::var("CARGO_BIN_EXE_hf2q").unwrap_or_else(|_| {
        // Production fallback: same-binary self-spawn.  `current_exe` is
        // the canonical lookup outside of cargo.
        std::env::current_exe()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "hf2q".to_string())
    });
    let cli_quant = map_quant_to_cli(quant);
    if cli_quant_was_degraded(quant) {
        tracing::info!(
            table_quant = quant.as_str(),
            cli_quant,
            "auto-pipeline: K-quant emit not yet on CLI (ADR-014 P7); \
             degrading to closest available legacy quant"
        );
    }
    let mut cmd = Command::new(&bin);
    cmd.arg("convert")
        .arg("--input")
        .arg(snapshot_dir)
        .arg("--format")
        .arg("gguf")
        .arg("--quant")
        .arg(cli_quant)
        .arg("--output")
        .arg(target_gguf)
        .arg("--yes")
        // The subprocess re-downloads nothing — `--input` is local.  But
        // it does run quality measurement by default; skip for the
        // auto-pipeline path because (a) we already verified the source
        // bytes via integrity and (b) quality measurement allocates an
        // extra ~F32 round-trip that doubles peak memory (see
        // `project_phase45_quality_oom.md`).
        .arg("--skip-quality");
    if no_integrity {
        cmd.arg("--no-integrity");
    }

    tracing::info!(
        bin = %bin,
        snapshot = %snapshot_dir.display(),
        target = %target_gguf.display(),
        cli_quant,
        "auto-pipeline: spawning convert subprocess"
    );
    let started = std::time::Instant::now();
    let output = cmd
        .output()
        .with_context(|| format!("spawn convert subprocess: {bin}"))?;
    let elapsed_ms = started.elapsed().as_millis();
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        return Err(anyhow!(
            "convert subprocess exited with {} (elapsed {}ms)\n\
             --- stdout ---\n{}\n--- stderr ---\n{}",
            output.status,
            elapsed_ms,
            stdout.trim_end(),
            stderr.trim_end(),
        ));
    }
    if !target_gguf.exists() {
        return Err(anyhow!(
            "convert subprocess returned 0 but target GGUF is missing at {}",
            target_gguf.display()
        ));
    }
    tracing::info!(
        target = %target_gguf.display(),
        elapsed_ms = elapsed_ms as u64,
        "auto-pipeline: convert subprocess complete"
    );
    Ok(())
}

/// Did the static map send a K-quant table output to a legacy CLI quant?
/// Used to emit the cutover-tracking log line; flips to all-false once
/// ADR-014 P7 wires K-quant emit through clap.
fn cli_quant_was_degraded(quant: QuantType) -> bool {
    match quant {
        QuantType::Q8_0 => false,         // Q8 maps clean
        QuantType::Q6_K => true,          // → q8 (Q8_0)
        QuantType::Q4_K_M => true,        // → q4 (Q4_0)
        QuantType::Q3_K_M => true,        // → q4 (Q4_0)
    }
}

fn secs_since_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── classify_model_input ────────────────────────────────────────────

    #[test]
    fn classify_existing_path() {
        // /tmp exists on every macOS / Linux dev box; using `/` as a
        // test path is portable.
        let m = classify_model_input("/").unwrap();
        assert_eq!(m, ModelInput::Path(PathBuf::from("/")));
    }

    #[test]
    fn classify_existing_file_under_tempdir() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("my.gguf");
        std::fs::write(&p, b"x").unwrap();
        let m = classify_model_input(p.to_str().unwrap()).unwrap();
        assert_eq!(m, ModelInput::Path(p));
    }

    #[test]
    fn classify_hf_repo_id_basic() {
        let m = classify_model_input("google/gemma-4-27b-it").unwrap();
        assert_eq!(m, ModelInput::HfRepoId("google/gemma-4-27b-it".into()));
    }

    #[test]
    fn classify_hf_repo_id_with_dots_and_underscores() {
        let m = classify_model_input("Org_1.x/repo-name_v2.0").unwrap();
        assert!(matches!(m, ModelInput::HfRepoId(_)));
    }

    #[test]
    fn classify_rejects_nonexistent_absolute_path() {
        let err = classify_model_input("/this/does/not/exist.gguf").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("does not exist") && msg.contains("repo-id"),
            "expected guidance in error: {msg}"
        );
    }

    #[test]
    fn classify_rejects_nonexistent_relative_path() {
        let err = classify_model_input("./missing.gguf").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("does not exist"), "{msg}");
    }

    #[test]
    fn classify_rejects_multi_slash() {
        // Triple-component is not a HF repo-id (HF doesn't have
        // sub-path concept here).
        let err = classify_model_input("org/sub/repo").unwrap_err();
        assert!(format!("{err}").contains("repo-id"));
    }

    #[test]
    fn classify_rejects_empty_string() {
        let err = classify_model_input("").unwrap_err();
        assert!(format!("{err}").contains("empty"));
    }

    #[test]
    fn classify_rejects_no_slash() {
        let err = classify_model_input("just-a-name").unwrap_err();
        assert!(format!("{err}").contains("repo-id"));
    }

    #[test]
    fn classify_rejects_backslash() {
        let err = classify_model_input("org\\repo").unwrap_err();
        assert!(format!("{err}").contains("repo-id"));
    }

    #[test]
    fn classify_rejects_special_chars() {
        let err = classify_model_input("org/repo with space").unwrap_err();
        assert!(format!("{err}").contains("repo-id"));
    }

    // ── looks_like_hf_repo_id ───────────────────────────────────────────

    #[test]
    fn looks_like_hf_accepts_canonical_shapes() {
        assert!(looks_like_hf_repo_id("google/gemma-4-27b-it"));
        assert!(looks_like_hf_repo_id("Qwen/Qwen3-MoE-A35B"));
        assert!(looks_like_hf_repo_id("a/b"));
        assert!(looks_like_hf_repo_id("Org_1.x/repo-name_v2.0"));
    }

    #[test]
    fn looks_like_hf_rejects_paths() {
        assert!(!looks_like_hf_repo_id(""));
        assert!(!looks_like_hf_repo_id("/abs/path/file"));
        assert!(!looks_like_hf_repo_id("./rel"));
        assert!(!looks_like_hf_repo_id("../up"));
        assert!(!looks_like_hf_repo_id("\\bad\\win"));
        assert!(!looks_like_hf_repo_id("org\\repo"));
        assert!(!looks_like_hf_repo_id("a/b/c"));
        assert!(!looks_like_hf_repo_id("only-org"));
        assert!(!looks_like_hf_repo_id("org/"));
        assert!(!looks_like_hf_repo_id("/repo"));
        assert!(!looks_like_hf_repo_id("org/repo with spaces"));
        assert!(!looks_like_hf_repo_id("org/r$pecial"));
    }

    // ── map_quant_to_cli ────────────────────────────────────────────────

    #[test]
    fn map_quant_q8_clean() {
        assert_eq!(map_quant_to_cli(QuantType::Q8_0), "q8");
        assert!(!cli_quant_was_degraded(QuantType::Q8_0));
    }

    #[test]
    fn map_quant_kquants_degrade_until_p7() {
        assert_eq!(map_quant_to_cli(QuantType::Q6_K), "q8");
        assert_eq!(map_quant_to_cli(QuantType::Q4_K_M), "q4");
        assert_eq!(map_quant_to_cli(QuantType::Q3_K_M), "q4");
        assert!(cli_quant_was_degraded(QuantType::Q6_K));
        assert!(cli_quant_was_degraded(QuantType::Q4_K_M));
        assert!(cli_quant_was_degraded(QuantType::Q3_K_M));
    }

    // ── resolve_or_prepare_model — pass-through for filesystem paths ────

    #[test]
    fn resolve_passthrough_existing_path() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("model.gguf");
        std::fs::write(&p, b"x").unwrap();
        let cache_dir = dir.path().join("hf2q");
        let mut cache = ModelCache::open_at(&cache_dir).unwrap();
        let hw = HardwareProfile {
            chip_model: "test".into(),
            total_memory_bytes: 64u64 << 30,
            available_memory_bytes: 64u64 << 30,
            total_cores: 16,
            performance_cores: 12,
            efficiency_cores: 4,
            memory_bandwidth_gbs: 400.0,
        };
        let r = resolve_or_prepare_model(p.to_str().unwrap(), &mut cache, &hw, false).unwrap();
        assert_eq!(r.gguf_path, p);
        assert_eq!(r.repo_id, None);
        assert_eq!(r.quant, None);
        assert!(!r.from_cache);
    }

    // ── resolve_or_prepare_model — cache-hit fast path ──────────────────
    //
    // Pre-populate the cache with a fake-but-valid quantized GGUF entry
    // (manifest + on-disk file with matching SHA-256), then assert that
    // resolve returns it without any network or subprocess action.

    #[test]
    fn resolve_cache_hit_returns_cached_path_without_network() {
        let tmp = tempfile::tempdir().unwrap();
        let cache_root = tmp.path().join("hf2q");
        let mut cache = ModelCache::open_at(&cache_root).unwrap();

        // Hardware fixture → forces Q8_0 (≥64 GiB).
        let hw = HardwareProfile {
            chip_model: "M5 Max".into(),
            total_memory_bytes: 128u64 << 30,
            available_memory_bytes: 128u64 << 30,
            total_cores: 16,
            performance_cores: 12,
            efficiency_cores: 4,
            memory_bandwidth_gbs: 400.0,
        };
        let info = GpuInfo::from_hardware_profile(&hw);
        let quant = select_quant(&info).unwrap();
        assert_eq!(quant, QuantType::Q8_0, "fixture: 128 GiB → Q8_0");

        let repo_id = "test-org/test-repo";

        // Drop a fake source entry so `record_quantized` has a parent
        // ModelEntry to attach to.
        cache
            .record_source(
                repo_id,
                "abcdef",
                SourcePointer::Local {
                    path: tmp.path().join("source"),
                    sha256: "deadbeef".to_string(),
                },
            )
            .unwrap();

        // Drop the cached GGUF on disk, hash it, record it.
        let gguf = cache_model_path(cache.root(), repo_id, quant).unwrap();
        std::fs::create_dir_all(gguf.parent().unwrap()).unwrap();
        std::fs::write(&gguf, b"FAKE GGUF BYTES - only the SHA matters here").unwrap();
        let sha = sha256_file(&gguf).unwrap();
        let bytes = std::fs::metadata(&gguf).unwrap().len();
        cache
            .record_quantized(
                repo_id,
                QuantEntry {
                    quant_type: quant.as_str().to_string(),
                    gguf_path: gguf.clone(),
                    mmproj_path: None,
                    bytes,
                    sha256: sha,
                    quantized_at_secs: secs_since_epoch(),
                    quantized_by_version: env!("CARGO_PKG_VERSION").to_string(),
                },
            )
            .unwrap();

        // Resolve — must hit the cache (no network, no subprocess).
        let r = resolve_or_prepare_model(repo_id, &mut cache, &hw, false).unwrap();
        assert!(r.from_cache, "expected cache-hit path");
        assert_eq!(r.gguf_path, gguf);
        assert_eq!(r.repo_id.as_deref(), Some(repo_id));
        assert_eq!(r.quant, Some(QuantType::Q8_0));
    }

    // ── resolve_or_prepare_model — cache-corruption fallthrough ─────────
    //
    // When the on-disk SHA-256 mismatches the manifest, the lookup-and-
    // verify helper must reject the entry and let the caller proceed to
    // the (in this test) unreachable network path.  We don't drive the
    // full miss path here (would require a real subprocess + HF) — we
    // just assert `lookup_and_verify` returns None on corruption.

    #[test]
    fn lookup_rejects_corrupted_cache_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let cache_root = tmp.path().join("hf2q");
        let mut cache = ModelCache::open_at(&cache_root).unwrap();
        let repo_id = "x/y";
        let quant = QuantType::Q8_0;

        cache
            .record_source(
                repo_id,
                "rev",
                SourcePointer::Local {
                    path: tmp.path().join("src"),
                    sha256: "n/a".into(),
                },
            )
            .unwrap();
        let gguf = cache_model_path(cache.root(), repo_id, quant).unwrap();
        std::fs::create_dir_all(gguf.parent().unwrap()).unwrap();
        std::fs::write(&gguf, b"original").unwrap();
        let real_sha = sha256_file(&gguf).unwrap();
        cache
            .record_quantized(
                repo_id,
                QuantEntry {
                    quant_type: quant.as_str().into(),
                    gguf_path: gguf.clone(),
                    mmproj_path: None,
                    bytes: 8,
                    sha256: real_sha,
                    quantized_at_secs: 0,
                    quantized_by_version: "test".into(),
                },
            )
            .unwrap();
        // Corrupt the on-disk file.
        std::fs::write(&gguf, b"CORRUPTED").unwrap();

        // With integrity ON, verify must fail and lookup_and_verify must
        // return None.
        let hit = lookup_and_verify(&cache, repo_id, quant, false);
        assert!(hit.is_none(), "corrupted cache must fall through");

        // With --no-integrity, the same call returns the (unsafe!) hit.
        let hit_unsafe = lookup_and_verify(&cache, repo_id, quant, true);
        assert!(hit_unsafe.is_some(), "--no-integrity must skip the SHA check");
    }
}
