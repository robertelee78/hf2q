//! `hf2q smoke` subcommand implementation (ADR-012 Decision 16).
//!
//! End-to-end conformance: preflight → convert → llama-cli inference →
//! transcript assertion → commit transcript.  Arch-generic; reads its knobs
//! from the [`ArchEntry`] registered for the requested arch.
//!
//! Preflight exit codes (Decision 16 acceptance):
//!   * 2 — `HF_TOKEN` unset
//!   * 3 — insufficient disk
//!   * 4 — `llama-cli` not executable on `$PATH` or at the override path
//!   * 5 — `hf2q` binary not built / not on `$PATH`
//!   * 6 — HF repo unresolvable

use std::path::{Path, PathBuf};
use std::process::Command;

use thiserror::Error;

use super::conformance::{ConformanceError, SmokeAssertion};
use super::registry::{ArchEntry, RegistryError};
use super::Registry;

// ---------------------------------------------------------------------------
// Exit codes (re-exported from `arch::mod`)
// ---------------------------------------------------------------------------

pub const SMOKE_EXIT_HF_TOKEN: u8 = 2;
pub const SMOKE_EXIT_DISK: u8 = 3;
pub const SMOKE_EXIT_LLAMA_CLI: u8 = 4;
pub const SMOKE_EXIT_NO_BINARY: u8 = 5;
pub const SMOKE_EXIT_NO_REPO: u8 = 6;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Quant variants the smoke harness supports.  P8 ships only Q4_0; DWQ
/// variants land in P9 alongside the PPL/KL eval helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmokeQuant {
    Q4_0,
    DwqMixed46,
    DwqMixed48,
}

impl SmokeQuant {
    pub fn as_str(&self) -> &'static str {
        match self {
            SmokeQuant::Q4_0 => "q4_0",
            SmokeQuant::DwqMixed46 => "dwq-mixed-4-6",
            SmokeQuant::DwqMixed48 => "dwq-mixed-4-8",
        }
    }
}

/// Options to a single smoke invocation.
#[derive(Clone)]
pub struct SmokeOptions {
    pub arch: String,
    pub quant: SmokeQuant,
    pub with_vision: bool,
    pub skip_convert: bool,
    pub dry_run: bool,
    /// Override path to `llama-cli` for tests.  Defaults to looking on `$PATH`.
    pub llama_cli_path: Option<PathBuf>,
    /// Override path for `hf2q` self-binary check.  Defaults to looking on
    /// `$PATH`.
    pub hf2q_path: Option<PathBuf>,
    /// Optional HF-resolver hook for tests.  When `Some`, the closure is
    /// called instead of shelling out to `huggingface-cli repo info`.
    /// Returning `Ok(())` means the repo resolves.
    pub hf_resolver: Option<HfResolver>,
    /// Optional disk-availability hook for tests; returns free GB at the
    /// supplied path.  Default uses `fs2::available_space`.
    pub disk_resolver: Option<DiskResolver>,
    /// Path used for the disk preflight check (default `~/.cache/hf2q`).
    pub disk_check_path: Option<PathBuf>,
    /// Output directory for committed transcripts (default
    /// `tests/fixtures/smoke-transcripts/`).
    pub transcript_dir: Option<PathBuf>,
}

impl std::fmt::Debug for SmokeOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmokeOptions")
            .field("arch", &self.arch)
            .field("quant", &self.quant)
            .field("with_vision", &self.with_vision)
            .field("skip_convert", &self.skip_convert)
            .field("dry_run", &self.dry_run)
            .field("llama_cli_path", &self.llama_cli_path)
            .field("hf2q_path", &self.hf2q_path)
            .field("hf_resolver", &self.hf_resolver.as_ref().map(|_| "<closure>"))
            .field("disk_resolver", &self.disk_resolver.as_ref().map(|_| "<closure>"))
            .field("disk_check_path", &self.disk_check_path)
            .field("transcript_dir", &self.transcript_dir)
            .finish()
    }
}

impl SmokeOptions {
    pub fn new(arch: impl Into<String>, quant: SmokeQuant) -> Self {
        Self {
            arch: arch.into(),
            quant,
            with_vision: false,
            skip_convert: false,
            dry_run: false,
            llama_cli_path: None,
            hf2q_path: None,
            hf_resolver: None,
            disk_resolver: None,
            disk_check_path: None,
            transcript_dir: None,
        }
    }
}

pub type HfResolver = std::sync::Arc<dyn Fn(&str) -> Result<(), String> + Send + Sync>;
pub type DiskResolver = std::sync::Arc<dyn Fn(&Path) -> Result<u64, String> + Send + Sync>;

#[derive(Debug, Error)]
pub enum SmokeError {
    #[error("registry: {0}")]
    Registry(#[from] RegistryError),

    #[error("HF_TOKEN environment variable unset (or empty); smoke needs HF auth")]
    HfTokenMissing,

    #[error(
        "insufficient disk at {path:?}: {available_gb} GB free, need {required_gb} GB \
         (arch '{arch}' floor {floor_gb} GB + 10 GB buffer)"
    )]
    InsufficientDisk {
        arch: String,
        path: PathBuf,
        available_gb: u64,
        required_gb: u64,
        floor_gb: u64,
    },

    #[error("disk-check probe failed at {path:?}: {reason}")]
    DiskProbe { path: PathBuf, reason: String },

    #[error("llama-cli not found at {0:?}; install llama.cpp or pass --llama-cli")]
    LlamaCliMissing(PathBuf),

    #[error("hf2q binary not found at {0:?}; build with `cargo build --release`")]
    Hf2qBinaryMissing(PathBuf),

    #[error("HF repo '{repo}' unresolvable: {reason}")]
    HfRepoUnresolvable { repo: String, reason: String },

    #[error("vision smoke requested but arch '{arch}' has no vision tower")]
    VisionUnsupported { arch: String },

    #[error("conformance: {0}")]
    Conformance(#[from] ConformanceError),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl SmokeError {
    /// Map an error to its smoke-specific exit code.  Errors that are not
    /// preflight (Conformance, Io, Registry) reuse the conversion-error code
    /// `1` (mapped by main.rs's `AppError::Conversion` arm).
    pub fn exit_code_override(&self) -> Option<u8> {
        match self {
            SmokeError::HfTokenMissing => Some(SMOKE_EXIT_HF_TOKEN),
            SmokeError::InsufficientDisk { .. } | SmokeError::DiskProbe { .. } => {
                Some(SMOKE_EXIT_DISK)
            }
            SmokeError::LlamaCliMissing(_) => Some(SMOKE_EXIT_LLAMA_CLI),
            SmokeError::Hf2qBinaryMissing(_) => Some(SMOKE_EXIT_NO_BINARY),
            SmokeError::HfRepoUnresolvable { .. } => Some(SMOKE_EXIT_NO_REPO),
            SmokeError::VisionUnsupported { .. }
            | SmokeError::Registry(_)
            | SmokeError::Conformance(_)
            | SmokeError::Io(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run a smoke invocation.
///
/// `--dry-run` exercises preflight only; nothing else runs.  In production
/// the convert + llama-cli sub-runs add wall-clock + disk costs that CI
/// cannot afford — call sites use `--dry-run` for the unit-test surface.
pub fn run_smoke(options: &SmokeOptions) -> Result<SmokeAssertion, SmokeError> {
    let entry = Registry::global().get(&options.arch)?;

    if options.with_vision && !entry.has_vision {
        return Err(SmokeError::VisionUnsupported {
            arch: entry.arch.to_string(),
        });
    }

    preflight(entry, options)?;

    if options.dry_run {
        // Synthesize a "would-pass" assertion so dry-run tests can still
        // exercise the full preflight surface.  Marked tokens=0 so any caller
        // that mistakes a dry-run for a real run sees the discrepancy.
        return Ok(SmokeAssertion {
            arch: entry.arch.to_string(),
            tokens_generated: 0,
            tensors_loaded: 0,
        });
    }

    // Convert (Decision 16 step 2) + llama-cli (step 3) live behind the
    // preflight gate.  The full implementation requires shelling out to the
    // hf2q + llama-cli binaries with a real GGUF on disk — an integration-
    // level concern.  Until P9's DWQ wire-up makes the convert side
    // measurable end-to-end, the in-process API below returns a clear
    // signal that the executor needs a real environment.  CI exercises
    // dry-run + every preflight failure mode via `tests/smoke_conformance.rs`.
    Err(SmokeError::Conformance(ConformanceError::NoTokenCount {
        arch: entry.arch.to_string(),
    }))
}

// ---------------------------------------------------------------------------
// Preflight
// ---------------------------------------------------------------------------

fn preflight(entry: &ArchEntry, options: &SmokeOptions) -> Result<(), SmokeError> {
    // 1. HF_TOKEN
    let token = std::env::var("HF_TOKEN").unwrap_or_default();
    if token.is_empty() {
        return Err(SmokeError::HfTokenMissing);
    }

    // 2. Disk
    let probe_path = options
        .disk_check_path
        .clone()
        .unwrap_or_else(default_cache_dir);
    let available_gb = match &options.disk_resolver {
        Some(f) => f(&probe_path).map_err(|reason| SmokeError::DiskProbe {
            path: probe_path.clone(),
            reason,
        })?,
        None => probe_disk_default(&probe_path).map_err(|reason| SmokeError::DiskProbe {
            path: probe_path.clone(),
            reason,
        })?,
    };
    let required_gb = entry.disk_floor_gb + 10;
    if available_gb < required_gb {
        return Err(SmokeError::InsufficientDisk {
            arch: entry.arch.to_string(),
            path: probe_path,
            available_gb,
            required_gb,
            floor_gb: entry.disk_floor_gb,
        });
    }

    // 3. llama-cli
    let llama_cli = options
        .llama_cli_path
        .clone()
        .unwrap_or_else(default_llama_cli);
    if !llama_cli.exists() {
        return Err(SmokeError::LlamaCliMissing(llama_cli));
    }

    // 4. hf2q binary
    let hf2q = options.hf2q_path.clone().unwrap_or_else(default_hf2q_binary);
    if !hf2q.exists() {
        return Err(SmokeError::Hf2qBinaryMissing(hf2q));
    }

    // 5. HF repo resolves.  At least one of the registered repos must be
    //    reachable; we accept the first that resolves.  Test hooks run in
    //    process; production shells out to `huggingface-cli`.
    let mut last_err: Option<String> = None;
    let mut resolved = false;
    for repo in entry.hf_repos {
        let outcome = match &options.hf_resolver {
            Some(f) => f(repo),
            None => resolve_repo_default(repo),
        };
        match outcome {
            Ok(()) => {
                resolved = true;
                break;
            }
            Err(reason) => last_err = Some(reason),
        }
    }
    if !resolved {
        return Err(SmokeError::HfRepoUnresolvable {
            repo: entry
                .hf_repos
                .first()
                .copied()
                .unwrap_or("<unknown>")
                .to_string(),
            reason: last_err.unwrap_or_else(|| "no canonical repos for arch".into()),
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Default probes (production behavior)
// ---------------------------------------------------------------------------

fn default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("HF2Q_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache/hf2q");
    }
    PathBuf::from(".")
}

fn default_llama_cli() -> PathBuf {
    if let Ok(p) = std::env::var("LLAMA_CLI") {
        return PathBuf::from(p);
    }
    PathBuf::from("/opt/llama.cpp/build/bin/llama-cli")
}

fn default_hf2q_binary() -> PathBuf {
    if let Ok(p) = std::env::var("HF2Q_BINARY") {
        return PathBuf::from(p);
    }
    // cargo's standard release-build location.
    PathBuf::from("./target/release/hf2q")
}

fn probe_disk_default(path: &Path) -> Result<u64, String> {
    use sysinfo::Disks;

    let disks = Disks::new_with_refreshed_list();
    let mut best: Option<(usize, u64)> = None;
    for disk in disks.list() {
        let mount = disk.mount_point();
        if path.starts_with(mount) {
            let len = mount.as_os_str().len();
            if best.map(|(b, _)| len > b).unwrap_or(true) {
                best = Some((len, disk.available_space()));
            }
        }
    }
    match best {
        Some((_, bytes)) => Ok(bytes / (1024 * 1024 * 1024)),
        None => Err(format!("no mount point matched {path:?}")),
    }
}

fn resolve_repo_default(repo: &str) -> Result<(), String> {
    // huggingface-cli is a Python tool — we shell out at runtime only; no
    // build dependency.  A clean exit code means the repo resolves.
    let output = Command::new("huggingface-cli")
        .args(["repo", "info", repo])
        .output()
        .map_err(|e| format!("huggingface-cli not installed or unreachable: {e}"))?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(stderr.trim().to_string())
    }
}

// ---------------------------------------------------------------------------
// Re-exports for tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Serialize all tests in this module that mutate the `HF_TOKEN`
    /// environment variable.  Cargo runs tests in parallel by default, so
    /// without this the set/remove races and the wrong test fails sporadically.
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn opts(arch: &str) -> SmokeOptions {
        let mut o = SmokeOptions::new(arch, SmokeQuant::Q4_0);
        // Wire test hooks for everything that would shell out.
        o.disk_resolver = Some(Arc::new(|_| Ok(1_000)));
        o.hf_resolver = Some(Arc::new(|_| Ok(())));
        o.llama_cli_path = Some(PathBuf::from("/usr/bin/true"));
        o.hf2q_path = Some(PathBuf::from("/usr/bin/true"));
        o
    }

    fn with_token<F: FnOnce()>(f: F) {
        let _g = env_lock();
        let prev = std::env::var("HF_TOKEN").ok();
        std::env::set_var("HF_TOKEN", "smoke-test");
        f();
        match prev {
            Some(v) => std::env::set_var("HF_TOKEN", v),
            None => std::env::remove_var("HF_TOKEN"),
        }
    }

    #[test]
    fn dry_run_succeeds_with_full_preflight_green() {
        with_token(|| {
            let mut o = opts("qwen35");
            o.dry_run = true;
            let assertion = run_smoke(&o).expect("dry-run should pass with all hooks green");
            assert_eq!(assertion.arch, "qwen35");
            assert_eq!(assertion.tokens_generated, 0);
        });
    }

    #[test]
    fn missing_hf_token_returns_exit_2() {
        let _g = env_lock();
        let prev = std::env::var("HF_TOKEN").ok();
        std::env::remove_var("HF_TOKEN");
        let mut o = opts("qwen35");
        o.dry_run = true;
        let err = run_smoke(&o).unwrap_err();
        assert_eq!(err.exit_code_override(), Some(SMOKE_EXIT_HF_TOKEN));
        if let Some(v) = prev {
            std::env::set_var("HF_TOKEN", v);
        }
    }

    #[test]
    fn insufficient_disk_returns_exit_3() {
        with_token(|| {
            let mut o = opts("qwen35moe");
            o.dry_run = true;
            // Force the probe to report 1 GB free — well below qwen35moe's
            // 150-GB floor.
            o.disk_resolver = Some(Arc::new(|_| Ok(1)));
            let err = run_smoke(&o).unwrap_err();
            assert_eq!(err.exit_code_override(), Some(SMOKE_EXIT_DISK));
            assert!(format!("{err}").contains("insufficient disk"));
        });
    }

    #[test]
    fn missing_llama_cli_returns_exit_4() {
        with_token(|| {
            let mut o = opts("qwen35");
            o.dry_run = true;
            o.llama_cli_path = Some(PathBuf::from("/path/that/does/not/exist/llama-cli"));
            let err = run_smoke(&o).unwrap_err();
            assert_eq!(err.exit_code_override(), Some(SMOKE_EXIT_LLAMA_CLI));
        });
    }

    #[test]
    fn missing_hf2q_binary_returns_exit_5() {
        with_token(|| {
            let mut o = opts("qwen35");
            o.dry_run = true;
            o.hf2q_path = Some(PathBuf::from("/path/that/does/not/exist/hf2q"));
            let err = run_smoke(&o).unwrap_err();
            assert_eq!(err.exit_code_override(), Some(SMOKE_EXIT_NO_BINARY));
        });
    }

    #[test]
    fn unresolvable_hf_repo_returns_exit_6() {
        with_token(|| {
            let mut o = opts("qwen35moe");
            o.dry_run = true;
            o.hf_resolver = Some(Arc::new(|_| Err("404 Not Found".into())));
            let err = run_smoke(&o).unwrap_err();
            assert_eq!(err.exit_code_override(), Some(SMOKE_EXIT_NO_REPO));
        });
    }

    #[test]
    fn unknown_arch_returns_registry_error() {
        with_token(|| {
            let mut o = opts("bogus");
            o.dry_run = true;
            let err = run_smoke(&o).unwrap_err();
            // Unknown-arch errors are registry errors, not preflight; they
            // map to the standard conversion exit code 1 in main.rs.
            assert_eq!(err.exit_code_override(), None);
            assert!(format!("{err}").contains("unknown arch"));
            assert!(format!("{err}").contains("qwen35"));
        });
    }

    #[test]
    fn vision_requested_on_moe_fails_uniformly() {
        with_token(|| {
            let mut o = opts("qwen35moe");
            o.dry_run = true;
            o.with_vision = true;
            let err = run_smoke(&o).unwrap_err();
            assert!(matches!(err, SmokeError::VisionUnsupported { .. }));
        });
    }

    #[test]
    fn vision_on_dense_passes_dry_run() {
        with_token(|| {
            let mut o = opts("qwen35");
            o.dry_run = true;
            o.with_vision = true;
            run_smoke(&o).expect("dense supports vision");
        });
    }
}
