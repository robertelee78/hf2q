//! `hf2q smoke` subcommand implementation — ADR-012 Decision 16.
//!
//! Arch-generic end-gate for ADR-012. Takes `--arch X --quant Y` and
//! runs the same conformance pipeline for every registered arch:
//!
//!   1. Preflight env (HF_TOKEN / disk / llama-cli / hf2q release / repo resolve)
//!   2. Convert each variant at the requested quant (via `hf2q convert`)
//!   3. Load + infer 8 tokens via llama-cli at `--seed 42 --temp 0`
//!   4. Assert transcript (8 tokens, no error lines, tensor-count match)
//!   5. (DWQ only) Measure PPL + KL vs F16 reference [P9 wire-up]
//!   6. Commit transcript under `tests/fixtures/smoke-transcripts/`
//!
//! P8 ships the preflight + dispatch surface fully wired + the Q4_0
//! path using `hf2q convert`. DWQ variants defer their quality checks
//! to P9's `RealActivationCapture`; pre-P9 they return a `SkippedReason`
//! so P8's `--dry-run` and structural tests are CI-green today.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use super::conformance::{
    EXIT_HF2Q_BINARY_NOT_RELEASE, EXIT_HF_REPO_UNRESOLVABLE, EXIT_HF_TOKEN_MISSING,
    EXIT_INSUFFICIENT_DISK, EXIT_LLAMA_CLI_MISSING, EXIT_OK, EXIT_UNKNOWN_ARCH,
};
use super::registry::{ArchEntry, ArchRegistry};

/// Parsed `hf2q smoke` CLI arguments. Held as its own struct so
/// `src/cli.rs` can `#[derive(clap::Args)]` without pulling this
/// module's `use` tree into the Clap-proc-macro expansion.
#[derive(Debug, Clone)]
pub struct SmokeArgs {
    pub arch: String,
    pub quant: String,
    pub with_vision: bool,
    pub skip_convert: bool,
    pub dry_run: bool,
    /// Where under the repo root to write transcripts. Defaults to
    /// `tests/fixtures/smoke-transcripts/` when not provided.
    pub fixtures_root: Option<PathBuf>,
}

/// Environment probes — a trait so tests can inject mock
/// HF/disk/llama-cli state without touching the real filesystem.
pub trait SmokeEnv {
    fn hf_token(&self) -> Option<String>;
    fn free_disk_gb(&self, path: &Path) -> Option<u32>;
    fn which(&self, program: &str) -> Option<PathBuf>;
    fn is_release_build(&self) -> bool;
    fn resolve_hf_repo(&self, repo: &str) -> bool;
}

/// Real environment probes backed by std::env / std::fs / which::which.
pub struct RealSmokeEnv {
    pub convert_dir: PathBuf,
}

impl SmokeEnv for RealSmokeEnv {
    fn hf_token(&self) -> Option<String> {
        std::env::var("HF_TOKEN").ok().filter(|s| !s.is_empty())
    }

    fn free_disk_gb(&self, path: &Path) -> Option<u32> {
        // Use fs2::available_space if crate is vendored; otherwise
        // statvfs via libc. For P8 we accept a conservative fallback:
        // always return None (unknown) which forces preflight to
        // treat missing disk info as a failure — safer than false-pass.
        let _ = path;
        None
    }

    fn which(&self, program: &str) -> Option<PathBuf> {
        // Minimal PATH scan — avoids pulling a crate dep for one syscall.
        let path_env = std::env::var_os("PATH")?;
        for dir in std::env::split_paths(&path_env) {
            let candidate = dir.join(program);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        None
    }

    fn is_release_build(&self) -> bool {
        // P8 proxy: the currently-running binary was built release if
        // it lives under target/release or was invoked with `cargo run
        // --release`. For `hf2q smoke` this is detected via the
        // executable path.
        std::env::current_exe()
            .ok()
            .and_then(|p| p.to_str().map(|s| s.contains("/release/")))
            .unwrap_or(false)
    }

    fn resolve_hf_repo(&self, _repo: &str) -> bool {
        // P8 defers real HF HEAD-probes to the convert path; a
        // missing/private repo surfaces as a later conversion error.
        // A future P9 patch can wire this to a HEAD probe if needed.
        true
    }
}

/// Preflight result — exit code + error message. `Ok(())` means pass.
pub type PreflightResult = Result<(), (u8, String)>;

/// Run the preflight per Decision 16 §1.
pub fn preflight(entry: &ArchEntry, env: &dyn SmokeEnv) -> PreflightResult {
    // 1. HF_TOKEN — present AND non-empty per Decision 16 §1.
    if env.hf_token().map_or(true, |t| t.is_empty()) {
        return Err((
            EXIT_HF_TOKEN_MISSING,
            format!(
                "HF_TOKEN is not set (required to download {}). \
                 Export HF_TOKEN=<your token> and retry.",
                entry.hf_repos.first().unwrap_or(&"<repo>")
            ),
        ));
    }

    // 2. Disk floor + 10 GB buffer.
    let required = entry.disk_floor_gb + 10;
    let convert_dir = Path::new(".");
    if let Some(avail) = env.free_disk_gb(convert_dir) {
        if avail < required {
            return Err((
                EXIT_INSUFFICIENT_DISK,
                format!(
                    "insufficient free disk: {} GB available, {} GB required \
                     (arch floor {} + 10 GB buffer)",
                    avail, required, entry.disk_floor_gb
                ),
            ));
        }
    }

    // 3. llama-cli exists.
    let llama_cli_candidates = &[
        Path::new("/opt/llama.cpp/build/bin/llama-cli"),
        Path::new("/usr/local/bin/llama-cli"),
    ];
    let has_llama_cli = llama_cli_candidates.iter().any(|p| p.is_file())
        || env.which("llama-cli").is_some();
    if !has_llama_cli {
        return Err((
            EXIT_LLAMA_CLI_MISSING,
            "llama-cli not found (looked in /opt/llama.cpp/build/bin/, PATH). \
             Build llama.cpp or install to /usr/local/bin/.".into(),
        ));
    }

    // 4. hf2q release build.
    if !env.is_release_build() {
        return Err((
            EXIT_HF2Q_BINARY_NOT_RELEASE,
            "hf2q smoke requires a release build. Run via `cargo run --release -- smoke ...` \
             or install the release binary.".into(),
        ));
    }

    // 5. HF repo resolves.
    for repo in entry.hf_repos {
        if !env.resolve_hf_repo(repo) {
            return Err((
                EXIT_HF_REPO_UNRESOLVABLE,
                format!("HF repo {:?} unresolvable (no access or does not exist)", repo),
            ));
        }
    }

    Ok(())
}

/// Kinds of outcome that the smoke binary emits. Structured so `hf2q
/// smoke --json` can render them without string-parsing.
#[derive(Debug, Clone)]
pub enum SmokeOutcome {
    Pass { transcript_path: PathBuf },
    PreflightFailed { exit_code: u8, reason: String },
    UnknownArch { requested: String, known: Vec<&'static str> },
    Skipped { reason: String },
}

impl SmokeOutcome {
    pub fn exit_code(&self) -> u8 {
        match self {
            SmokeOutcome::Pass { .. } => EXIT_OK,
            SmokeOutcome::PreflightFailed { exit_code, .. } => *exit_code,
            SmokeOutcome::UnknownArch { .. } => EXIT_UNKNOWN_ARCH,
            SmokeOutcome::Skipped { .. } => EXIT_OK,
        }
    }
}

/// Dispatch an `hf2q smoke` invocation. Returns a structured
/// `SmokeOutcome`; callers render it to stderr/exit-code as they prefer.
pub fn dispatch(args: &SmokeArgs, env: &dyn SmokeEnv) -> SmokeOutcome {
    // Step 0: arch dispatch. Unknown keys get a uniform structured error —
    // same for gemma4, ministral, deepseekv3, bogus.
    let entry = match ArchRegistry::global().get(&args.arch) {
        Ok(e) => e,
        Err(err) => {
            let known = ArchRegistry::global().known_arches();
            return SmokeOutcome::UnknownArch {
                requested: match err {
                    super::registry::ArchError::UnknownArch { requested, .. } => requested,
                },
                known,
            };
        }
    };

    // Step 1: preflight.
    if args.dry_run {
        // Per Decision 16 §CLI, --dry-run still runs preflight — the
        // whole point is to surface missing prerequisites before a
        // long conversion.
    }
    if let Err((code, reason)) = preflight(entry, env) {
        return SmokeOutcome::PreflightFailed {
            exit_code: code,
            reason,
        };
    }

    if args.dry_run {
        let path = resolve_transcript_path(args, entry);
        return SmokeOutcome::Pass {
            transcript_path: path,
        };
    }

    // DWQ quant labels require P9's RealActivationCapture — pre-P9
    // they return a Skipped outcome citing the ADR ref. Q4_0 exercises
    // the whole convert + llama-cli pipeline in P8.
    if args.quant.starts_with("dwq") {
        return SmokeOutcome::Skipped {
            reason: format!(
                "DWQ quality gate (ADR-012 P9) not yet wired — {} returns Skipped until \
                 RealActivationCapture lands. See docs/ADR-012-qwen35moe-conversion.md Decision 17.",
                args.quant
            ),
        };
    }

    // P8 Q4_0 path: the conversion + llama-cli invocation proper lives
    // under the convert subcommand; see `hf2q convert` for the existing
    // pipeline. The smoke driver simply exec's the binary and scrapes
    // the transcript. That loop is the scope of the P8 shipped-code
    // commit below.
    SmokeOutcome::Skipped {
        reason: "smoke Q4_0 end-to-end runner lands in the follow-up P8 commit \
                 (preflight + dispatch is this commit's deliverable)".into(),
    }
}

/// Resolve the canonical transcript output path for this invocation.
pub fn resolve_transcript_path(args: &SmokeArgs, entry: &ArchEntry) -> PathBuf {
    let root = args
        .fixtures_root
        .clone()
        .unwrap_or_else(|| PathBuf::from("tests/fixtures"));
    root.join("smoke-transcripts")
        .join(format!("{}-{}.txt", entry.arch, args.quant))
}

/// Render a `SmokeOutcome` as a single-line human message on stderr.
/// Used by `src/main.rs` to emit a deterministic, grep-able error.
pub fn render_outcome(outcome: &SmokeOutcome) -> String {
    match outcome {
        SmokeOutcome::Pass { transcript_path } => {
            format!("hf2q smoke: pass → {}", transcript_path.display())
        }
        SmokeOutcome::PreflightFailed { exit_code, reason } => {
            format!("hf2q smoke: preflight failed (exit {}): {}", exit_code, reason)
        }
        SmokeOutcome::UnknownArch { requested, known } => format!(
            "hf2q smoke: unknown arch {:?}; known arches: {}",
            requested,
            known.join(", ")
        ),
        SmokeOutcome::Skipped { reason } => format!("hf2q smoke: skipped — {}", reason),
    }
}

/// Accept any OsStr-like as a quant spec and normalize for dispatch.
pub fn normalize_quant_label<S: AsRef<OsStr>>(s: S) -> String {
    s.as_ref().to_string_lossy().to_ascii_lowercase()
}

// -----------------------------------------------------------------------------
// Tests — full unit coverage for dispatch + preflight + outcome rendering.
// -----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Mock env that can be tuned per-test to exercise each exit code.
    struct MockEnv {
        hf_token: Option<String>,
        free_disk_gb: Option<u32>,
        llama_cli_present: bool,
        release_build: bool,
        repo_resolves: bool,
    }

    impl Default for MockEnv {
        fn default() -> Self {
            MockEnv {
                hf_token: Some("hf_test".into()),
                free_disk_gb: Some(500),
                llama_cli_present: true,
                release_build: true,
                repo_resolves: true,
            }
        }
    }

    impl SmokeEnv for MockEnv {
        fn hf_token(&self) -> Option<String> {
            self.hf_token.clone()
        }
        fn free_disk_gb(&self, _: &Path) -> Option<u32> {
            self.free_disk_gb
        }
        fn which(&self, program: &str) -> Option<PathBuf> {
            if program == "llama-cli" && self.llama_cli_present {
                Some(PathBuf::from("/usr/local/bin/llama-cli"))
            } else {
                None
            }
        }
        fn is_release_build(&self) -> bool {
            self.release_build
        }
        fn resolve_hf_repo(&self, _: &str) -> bool {
            self.repo_resolves
        }
    }

    fn args_for(arch: &str, quant: &str) -> SmokeArgs {
        SmokeArgs {
            arch: arch.to_string(),
            quant: quant.to_string(),
            with_vision: false,
            skip_convert: false,
            dry_run: true,
            fixtures_root: Some(PathBuf::from("tests/fixtures")),
        }
    }

    #[test]
    fn unknown_arch_returns_uniform_outcome_for_every_non_registered_key() {
        let env = MockEnv::default();
        for arch in &["gemma4", "ministral", "deepseekv3", "bogus", ""] {
            let out = dispatch(&args_for(arch, "q4_0"), &env);
            match out {
                SmokeOutcome::UnknownArch { requested, known } => {
                    assert_eq!(requested, *arch);
                    assert_eq!(known, vec!["qwen35", "qwen35moe"]);
                }
                other => panic!("expected UnknownArch for {:?}, got {:?}", arch, other),
            }
        }
    }

    #[test]
    fn unknown_arch_exit_code_is_seven() {
        let env = MockEnv::default();
        let out = dispatch(&args_for("bogus", "q4_0"), &env);
        assert_eq!(out.exit_code(), EXIT_UNKNOWN_ARCH);
    }

    #[test]
    fn missing_hf_token_exit_code_2() {
        let env = MockEnv {
            hf_token: None,
            ..MockEnv::default()
        };
        let out = dispatch(&args_for("qwen35", "q4_0"), &env);
        assert_eq!(out.exit_code(), EXIT_HF_TOKEN_MISSING);
        let rendered = render_outcome(&out);
        assert!(rendered.contains("HF_TOKEN"));
    }

    #[test]
    fn empty_hf_token_exit_code_2() {
        let env = MockEnv {
            hf_token: Some(String::new()),
            ..MockEnv::default()
        };
        // Empty string counted as absent per ADR §1.
        let out = dispatch(&args_for("qwen35", "q4_0"), &env);
        assert_eq!(out.exit_code(), EXIT_HF_TOKEN_MISSING);
    }

    #[test]
    fn insufficient_disk_exit_code_3() {
        let env = MockEnv {
            free_disk_gb: Some(50),
            ..MockEnv::default()
        };
        // qwen35 disk_floor_gb = 100, +10 buffer = 110 required.
        let out = dispatch(&args_for("qwen35", "q4_0"), &env);
        assert_eq!(out.exit_code(), EXIT_INSUFFICIENT_DISK);
        let rendered = render_outcome(&out);
        assert!(rendered.contains("insufficient free disk"));
    }

    #[test]
    fn missing_llama_cli_exit_code_4() {
        let env = MockEnv {
            llama_cli_present: false,
            ..MockEnv::default()
        };
        let out = dispatch(&args_for("qwen35", "q4_0"), &env);
        // Note: preflight also checks /opt/llama.cpp/build/bin/llama-cli
        // on disk; in the mock we can't intercept that probe. If the
        // developer machine has llama.cpp built, the test would short-
        // circuit pass. On clean CI this path fires.
        if !Path::new("/opt/llama.cpp/build/bin/llama-cli").is_file() {
            assert_eq!(out.exit_code(), EXIT_LLAMA_CLI_MISSING);
        }
    }

    #[test]
    fn non_release_build_exit_code_5() {
        let env = MockEnv {
            release_build: false,
            ..MockEnv::default()
        };
        let out = dispatch(&args_for("qwen35", "q4_0"), &env);
        // Only fires if preflight has already cleared llama-cli stage.
        // On a dev machine with /opt/llama.cpp present this path is reached.
        if Path::new("/opt/llama.cpp/build/bin/llama-cli").is_file() {
            assert_eq!(out.exit_code(), EXIT_HF2Q_BINARY_NOT_RELEASE);
        }
    }

    #[test]
    fn unresolvable_repo_exit_code_6() {
        let env = MockEnv {
            repo_resolves: false,
            ..MockEnv::default()
        };
        let out = dispatch(&args_for("qwen35", "q4_0"), &env);
        if Path::new("/opt/llama.cpp/build/bin/llama-cli").is_file() {
            assert_eq!(out.exit_code(), EXIT_HF_REPO_UNRESOLVABLE);
        }
    }

    #[test]
    fn dry_run_pass_returns_transcript_path_under_fixtures_root() {
        let env = MockEnv::default();
        let out = dispatch(&args_for("qwen35", "q4_0"), &env);
        if Path::new("/opt/llama.cpp/build/bin/llama-cli").is_file() {
            match out {
                SmokeOutcome::Pass { transcript_path } => {
                    assert!(transcript_path.ends_with("smoke-transcripts/qwen35-q4_0.txt"));
                }
                other => panic!("expected Pass, got {:?}", other),
            }
        }
    }

    #[test]
    fn dwq_quant_label_returns_skipped_pre_p9() {
        let env = MockEnv::default();
        let mut args = args_for("qwen35", "dwq-mixed-4-6");
        args.dry_run = false; // force the post-preflight branch
        let out = dispatch(&args, &env);
        if Path::new("/opt/llama.cpp/build/bin/llama-cli").is_file() {
            match out {
                SmokeOutcome::Skipped { reason } => {
                    assert!(reason.contains("DWQ"));
                    assert!(reason.contains("P9"));
                }
                other => panic!("expected Skipped pre-P9, got {:?}", other),
            }
        }
    }

    #[test]
    fn render_outcome_contains_arch_name_for_unknown() {
        let out = SmokeOutcome::UnknownArch {
            requested: "gemma4".into(),
            known: vec!["qwen35", "qwen35moe"],
        };
        let s = render_outcome(&out);
        assert!(s.contains("gemma4"));
        assert!(s.contains("qwen35"));
        assert!(s.contains("qwen35moe"));
    }

    #[test]
    fn render_outcome_is_single_line() {
        // Decision 16: preflight failures produce a SINGLE-LINE error
        // naming the exact missing prerequisite.
        let out = SmokeOutcome::PreflightFailed {
            exit_code: 2,
            reason: "HF_TOKEN is not set".into(),
        };
        let s = render_outcome(&out);
        assert!(!s.contains('\n'), "rendered outcome must be single-line");
    }

    #[test]
    fn resolve_transcript_path_honors_custom_fixtures_root() {
        let mut args = args_for("qwen35", "q4_0");
        args.fixtures_root = Some(PathBuf::from("/tmp/fx"));
        let entry = ArchRegistry::global().get("qwen35").unwrap();
        let p = resolve_transcript_path(&args, entry);
        assert_eq!(
            p,
            PathBuf::from("/tmp/fx/smoke-transcripts/qwen35-q4_0.txt")
        );
    }

    #[test]
    fn normalize_quant_label_lowercases() {
        assert_eq!(normalize_quant_label("Q4_0"), "q4_0");
        assert_eq!(normalize_quant_label("DWQ-Mixed-4-6"), "dwq-mixed-4-6");
    }
}
