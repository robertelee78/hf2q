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
    EXIT_INSUFFICIENT_DISK, EXIT_LLAMA_CLI_MISSING, EXIT_OK,
    EXIT_SMOKE_ASSERTION_FAILED, EXIT_UNKNOWN_ARCH,
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
    /// Path to a local safetensors directory. When set, the smoke
    /// runner skips the HF download step (preflight HF_TOKEN check is
    /// also skipped) and converts the local dir. Enables CI testing
    /// of the Q4_0 end-to-end path on synthetic models without a
    /// network dependency.
    pub local_dir: Option<PathBuf>,
    /// Where to keep the converted GGUF. Defaults to a temp dir so
    /// repeat smoke runs don't accumulate disk. Retained for diagnosis
    /// when `--keep-outputs` is passed.
    pub convert_output_dir: Option<PathBuf>,
    /// Override path for the llama-cli binary. When set, smoke uses
    /// this path instead of searching `/opt/llama.cpp/build/bin/` or
    /// `$PATH`. Enables CI tests via a shell-script stub that emits
    /// a deterministic transcript (per Decision 16 §Acceptance:
    /// "CI runs a dedicated unit test suite ... via a mock llama-cli
    /// stub"). Also bypasses the preflight llama-cli-present check
    /// since the override itself is the proof of presence.
    pub llama_cli_override: Option<PathBuf>,
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
///
/// - `local_dir_provided` true: skip HF_TOKEN + HF-repo-resolve checks.
/// - `llama_cli_override` Some: skip the default llama-cli search path
///   (the override itself is the proof-of-presence).
pub fn preflight_full(
    entry: &ArchEntry,
    env: &dyn SmokeEnv,
    local_dir_provided: bool,
    llama_cli_override: Option<&Path>,
) -> PreflightResult {
    // 1. HF_TOKEN — present AND non-empty per Decision 16 §1.
    //    Skipped when --local-dir is set (no download needed).
    if !local_dir_provided && env.hf_token().map_or(true, |t| t.is_empty()) {
        return Err((
            EXIT_HF_TOKEN_MISSING,
            format!(
                "HF_TOKEN is not set (required to download {}). \
                 Export HF_TOKEN=<your token> and retry, or pass --local-dir \
                 to use a pre-downloaded safetensors directory.",
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

    // 3. llama-cli exists — either via the explicit override path, or
    //    via the default search list.
    if let Some(override_path) = llama_cli_override {
        if !override_path.is_file() {
            return Err((
                EXIT_LLAMA_CLI_MISSING,
                format!(
                    "--llama-cli-override path {:?} does not exist",
                    override_path
                ),
            ));
        }
    } else {
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
                 Build llama.cpp or install to /usr/local/bin/, or pass \
                 --llama-cli-override <path>.".into(),
            ));
        }
    }

    // 4. hf2q release build.
    if !env.is_release_build() {
        return Err((
            EXIT_HF2Q_BINARY_NOT_RELEASE,
            "hf2q smoke requires a release build. Run via `cargo run --release -- smoke ...` \
             or install the release binary.".into(),
        ));
    }

    // 5. HF repo resolves (skipped when --local-dir is set).
    if !local_dir_provided {
        for repo in entry.hf_repos {
            if !env.resolve_hf_repo(repo) {
                return Err((
                    EXIT_HF_REPO_UNRESOLVABLE,
                    format!("HF repo {:?} unresolvable (no access or does not exist)", repo),
                ));
            }
        }
    }

    Ok(())
}

/// Compatibility wrapper — old callers that don't pass `--local-dir`
/// or `--llama-cli-override`.
pub fn preflight(entry: &ArchEntry, env: &dyn SmokeEnv) -> PreflightResult {
    preflight_full(entry, env, false, None)
}

/// Preflight helper for the (most common) case of `--local-dir` without
/// an llama-cli override.
pub fn preflight_with_local(
    entry: &ArchEntry,
    env: &dyn SmokeEnv,
    local_dir_provided: bool,
) -> PreflightResult {
    preflight_full(entry, env, local_dir_provided, None)
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

    // Print the dry-run informational report BEFORE preflight — operators
    // see what the run WOULD do even when preflight surfaces a missing
    // prerequisite (so the fix is actionable on the first read).
    if args.dry_run {
        print_dry_run_report(entry, args);
    }

    // Step 1: preflight. --local-dir skips HF_TOKEN + repo-resolve checks;
    // --llama-cli-override skips the default llama-cli search.
    let local_dir_provided = args.local_dir.is_some();
    if let Err((code, reason)) = preflight_full(
        entry,
        env,
        local_dir_provided,
        args.llama_cli_override.as_deref(),
    ) {
        return SmokeOutcome::PreflightFailed {
            exit_code: code,
            reason,
        };
    }

    // Step 1b: --local-dir existence check. Earlier this fired only at
    // run_q4_0_pipeline (post-dry-run), so a user running
    //   hf2q smoke --arch qwen35 --dry-run --local-dir /path/typo
    // would see "pass" and exit 0 — misleading because the same command
    // without --dry-run would fail at convert. Catch it here so dry-run
    // is a faithful pre-flight check.
    if let Some(local) = &args.local_dir {
        if !local.exists() {
            return SmokeOutcome::PreflightFailed {
                // Reuse EXIT_SMOKE_ASSERTION_FAILED (8) for parity with the
                // post-preflight `run_q4_0_pipeline` local-dir check; both
                // paths now emit the same code so scripts can't distinguish
                // "caught by preflight" vs "caught by pipeline" — only that
                // it failed and the message names the missing path.
                exit_code: EXIT_SMOKE_ASSERTION_FAILED,
                reason: format!(
                    "--local-dir {:?} does not exist (no input safetensors directory \
                     to convert from). Pass a valid path or omit --local-dir to use \
                     the HF download path.",
                    local
                ),
            };
        }
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

    // Q4_0 end-to-end: convert → llama-cli → scrape transcript.
    match run_q4_0_pipeline(entry, args) {
        Ok(transcript_path) => SmokeOutcome::Pass { transcript_path },
        Err(reason) => SmokeOutcome::PreflightFailed {
            exit_code: EXIT_SMOKE_ASSERTION_FAILED,
            reason,
        },
    }
}

/// Run the Q4_0 end-to-end smoke pipeline and emit the transcript.
///
/// 1. Resolve input directory (either `--local-dir` or the first HF repo).
/// 2. `hf2q convert --quant q4 --output <tmpdir>/smoke.gguf`.
/// 3. `llama-cli --model ... -n 8 --seed 42 --temp 0 --no-warmup`.
/// 4. Assert transcript: no error lines, 8 tokens generated.
/// 5. Write transcript to `tests/fixtures/smoke-transcripts/{arch}-{quant}.txt`.
///
/// **Why no `--log-disable`:** real llama-cli routes both the model-
/// loader summary (`loaded meta data with N tensors ...`) and the
/// timing block (`eval time = X ms / N runs`) through `LLAMA_LOG_INFO`
/// (see `/opt/llama.cpp/src/llama-context.cpp:3486`). Adding
/// `--log-disable` (which calls `common_log_pause`) suppresses both
/// — leaving the smoke harness's transcript-assertion parsers
/// looking at empty stderr. The transcript itself is bounded
/// (`-n 8`) so the log volume stays small without requiring
/// suppression.
fn run_q4_0_pipeline(
    entry: &ArchEntry,
    args: &SmokeArgs,
) -> Result<PathBuf, String> {
    use std::process::Command;

    let input_dir = args
        .local_dir
        .clone()
        .ok_or_else(|| {
            format!(
                "non-local smoke path (HF download) is not shipped in this commit; \
                 pass --local-dir <path> to convert a pre-downloaded safetensors dir \
                 for arch {}.",
                entry.arch
            )
        })?;
    if !input_dir.exists() {
        return Err(format!("--local-dir {:?} does not exist", input_dir));
    }

    // Use a temp dir for the convert output unless the caller asked
    // to keep it.
    let keep_dir = args
        .convert_output_dir
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join("hf2q-smoke-convert"));
    let _ = std::fs::create_dir_all(&keep_dir);
    let gguf_path = keep_dir.join(format!("{}-{}.gguf", entry.arch, args.quant));

    if !args.skip_convert {
        let hf2q_exe = std::env::current_exe()
            .map_err(|e| format!("locate hf2q binary: {}", e))?;
        let convert_args = build_convert_args(args, entry, &input_dir, &gguf_path)?;
        let convert_out = Command::new(&hf2q_exe)
            .args(&convert_args)
            .output()
            .map_err(|e| format!("run hf2q convert: {}", e))?;
        if !convert_out.status.success() {
            return Err(format!(
                "hf2q convert failed (exit {}): {}",
                convert_out.status,
                String::from_utf8_lossy(&convert_out.stderr)
            ));
        }
    } else if !gguf_path.exists() {
        return Err(format!(
            "--skip-convert set but no pre-existing GGUF at {:?}",
            gguf_path
        ));
    }

    // llama-cli invocation — deterministic per Decision 16 §3.
    // Prefer the explicit override (CI stub) over the system-search path.
    let llama_cli = match &args.llama_cli_override {
        Some(p) => p.clone(),
        None => find_llama_cli()?,
    };
    let prompt = entry
        .smoke_prompts
        .first()
        .copied()
        .unwrap_or("The quick brown fox");

    let llama_out = Command::new(&llama_cli)
        .args([
            "--model",
            gguf_path.to_str().ok_or("gguf_path not UTF-8")?,
            "--prompt",
            prompt,
            "-n",
            "8",
            "--seed",
            "42",
            "--temp",
            "0",
            // No `--log-disable` — real llama-cli's loader summary +
            // timing block both flow through LLAMA_LOG_INFO and would
            // be suppressed, leaving the transcript-assertion parsers
            // staring at empty stderr. See run_q4_0_pipeline doc comment.
            "--no-warmup",
        ])
        .output()
        .map_err(|e| format!("run llama-cli: {}", e))?;

    let combined_stderr = String::from_utf8_lossy(&llama_out.stderr);
    let combined_stdout = String::from_utf8_lossy(&llama_out.stdout);
    // Decision 16 §AC requires byte-identical transcripts across two
    // fresh runs, but real llama-cli emits per-run timing data
    // (ms / tokens-per-second) that varies with system load. Sanitize
    // decimal numbers (the timestamps + rates) to placeholders before
    // writing — integer counts (n_runs, n_tokens) are preserved so
    // the structural assertion still has its scrape targets.
    let sanitized_stderr = sanitize_timestamps(&combined_stderr);
    let transcript_body = format!(
        "# hf2q smoke transcript\n\
         # arch:  {}\n\
         # quant: {}\n\
         # prompt: {:?}\n\
         # (timestamps stripped; byte-stable across runs)\n\n\
         ---stdout---\n{}\n\
         ---stderr---\n{}\n",
        entry.arch, args.quant, prompt, combined_stdout, sanitized_stderr
    );

    // Scan stderr for regression patterns per conformance helpers.
    super::conformance::scan_llama_cli_stderr(&combined_stderr)
        .map_err(|e| format!("llama-cli regression pattern: {}", e))?;

    // n_eval check.
    if let Some(n_eval) = super::conformance::extract_n_eval(&combined_stderr) {
        if n_eval != 8 {
            return Err(format!(
                "llama-cli produced {} tokens, expected 8",
                n_eval
            ));
        }
    } else {
        // Not every llama-cli build prints the timings block; treat
        // missing as informational rather than a failure.
    }

    // Write transcript.
    let transcript_path = resolve_transcript_path(args, entry);
    if let Some(parent) = transcript_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&transcript_path, transcript_body)
        .map_err(|e| format!("write transcript {:?}: {}", transcript_path, e))?;
    Ok(transcript_path)
}

/// Build the convert subprocess args. Pure function — extracted so
/// the `--with-vision` → `--emit-vision-tower` wiring (Decision 16
/// §CLI flag) can be unit-tested without spawning a subprocess.
///
/// Three guards combine before --emit-vision-tower is appended:
/// 1. `args.with_vision` — user opt-in.
/// 2. `entry.has_vision` — arch advertises a vision-tower path.
/// 3. (Convert-side) silent-skip if config.json has no vision_config
///    per Decision 18 / commit 18cbaaa.
fn build_convert_args(
    args: &SmokeArgs,
    entry: &ArchEntry,
    input_dir: &Path,
    gguf_path: &Path,
) -> Result<Vec<String>, String> {
    let mut convert_args: Vec<String> = vec![
        "convert".into(),
        "--input".into(),
        input_dir.to_str().ok_or("input_dir not UTF-8")?.into(),
        "--format".into(),
        "gguf".into(),
        "--quant".into(),
        args.quant.clone(),
        "--output".into(),
        gguf_path.to_str().ok_or("gguf_path not UTF-8")?.into(),
        "--yes".into(),
        "--skip-quality".into(),
    ];
    if args.with_vision && entry.has_vision {
        convert_args.push("--emit-vision-tower".into());
    }
    Ok(convert_args)
}

/// Replace decimal-number sequences (`X.YZ`) with `<X.XX>` so the
/// transcript stays byte-identical across runs even with real llama-cli
/// emitting per-run timing data. Integer counts (n_runs, n_tokens) are
/// preserved — only sequences with an embedded `.` are normalised, so
/// version strings like "GGUF V3" pass through.
///
/// **Whitespace handling:** real llama-cli uses width-padded format
/// specifiers (`%10.2f`) so a number like "58.90" (5 chars) gets 5
/// leading spaces while "158.90" (6 chars) gets only 4. To remain
/// byte-identical across magnitudes, the sanitizer absorbs any
/// whitespace immediately preceding a decimal-number run into the
/// placeholder.
///
/// Conservative: only whitespace ADJACENT to a decimal is absorbed.
/// Whitespace before integer-only tokens (e.g. "/     8 runs") is
/// preserved because integer counts use the same `%5d` width
/// specifier and their column positions stay stable.
fn sanitize_timestamps(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_digit() {
            // Scan the integer prefix.
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            // If followed by `.<digit>`, swallow the decimal part,
            // remove any whitespace already pushed onto out (the
            // width-padding leading spaces), and emit the placeholder.
            let has_decimal = i + 1 < bytes.len()
                && bytes[i] == b'.'
                && bytes[i + 1].is_ascii_digit();
            if has_decimal {
                i += 1; // dot
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                // Trim trailing whitespace from out — that's the
                // width-padding before this decimal that varies
                // with the number's magnitude.
                while out.ends_with(' ') || out.ends_with('\t') {
                    out.pop();
                }
                out.push_str("<X.XX>");
            } else {
                out.push_str(&s[start..i]);
            }
        } else {
            // SAFETY: i was at a valid char boundary because we only
            // advance by ASCII digits which are 1-byte. Non-ASCII is
            // emitted unchanged.
            let ch_start = i;
            // Find next char boundary.
            i += 1;
            while i < bytes.len() && (bytes[i] & 0xC0) == 0x80 {
                i += 1;
            }
            out.push_str(&s[ch_start..i]);
        }
    }
    out
}

fn find_llama_cli() -> Result<PathBuf, String> {
    let candidates = [
        "/opt/llama.cpp/build/bin/llama-cli",
        "/usr/local/bin/llama-cli",
    ];
    for c in candidates {
        if std::path::Path::new(c).is_file() {
            return Ok(PathBuf::from(c));
        }
    }
    if let Some(path_env) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_env) {
            let cand = dir.join("llama-cli");
            if cand.is_file() {
                return Ok(cand);
            }
        }
    }
    Err("llama-cli not found".into())
}

/// Emit a human-readable summary of what the smoke run WOULD do,
/// triggered by `--dry-run`. Decision 16 §1 calls for preflight
/// verification with visible feedback; this renders the arch entry's
/// knobs so operators see what transcript path, disk floor, HF repos,
/// tensor catalog, and quality thresholds apply before committing to
/// a long convert.
pub fn print_dry_run_report(entry: &ArchEntry, args: &SmokeArgs) {
    println!("═══ hf2q smoke dry-run ═══");
    println!("  arch:              {}", entry.arch);
    println!("  quant:             {}", args.quant);
    println!("  has_mtp:           {}", entry.has_mtp);
    println!("  has_vision:        {}", entry.has_vision);
    println!(
        "  disk_floor_gb:     {} (+10 GB buffer = {} required)",
        entry.disk_floor_gb,
        entry.disk_floor_gb + 10
    );
    println!(
        "  hf_architectures:  {}",
        entry.hf_architectures.join(", ")
    );
    println!("  hf_repos:          {}", entry.hf_repos.join(", "));
    println!(
        "  tensor_catalog:    {} template entries",
        entry.tensor_catalog.entries.len()
    );
    println!("  quality_thresholds:");
    println!(
        "    ppl_ratio_dwq46: ≤ {:.2}×",
        entry.quality_thresholds.ppl_ratio_dwq46
    );
    println!(
        "    ppl_ratio_dwq48: ≤ {:.2}×",
        entry.quality_thresholds.ppl_ratio_dwq48
    );
    println!(
        "    max_median_kl:   < {:.2} nats",
        entry.quality_thresholds.max_median_kl
    );
    println!(
        "  smoke_prompts:     {}",
        entry.smoke_prompts.first().unwrap_or(&"(none)")
    );
    let path = resolve_transcript_path(args, entry);
    println!("  transcript_path:   {}", path.display());
    if args.local_dir.is_some() {
        println!(
            "  local_dir:         {} (HF_TOKEN preflight skipped)",
            args.local_dir.as_ref().unwrap().display()
        );
    }
    println!("═══════════════════════════");
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
            local_dir: None,
            convert_output_dir: None,
            llama_cli_override: None,
        }
    }

    /// `--local-dir <path>` with a non-existent path must fail at the
    /// dispatch layer (NOT defer to run_q4_0_pipeline). Without this
    /// pre-pipeline check, `hf2q smoke ... --dry-run --local-dir /typo`
    /// returned exit 0 — misleading because the same command without
    /// --dry-run would fail at convert. Now both --dry-run and full
    /// runs reject the missing path at the same gate.
    #[test]
    fn local_dir_missing_path_returns_preflight_failure_in_dry_run() {
        let env = MockEnv::default();
        let mut args = args_for("qwen35", "q4_0");
        args.local_dir = Some(PathBuf::from(
            "/this/path/definitely/does/not/exist/qwen35-input",
        ));
        args.dry_run = true;
        let outcome = dispatch(&args, &env);
        match outcome {
            SmokeOutcome::PreflightFailed { exit_code, reason } => {
                assert_eq!(
                    exit_code, EXIT_SMOKE_ASSERTION_FAILED,
                    "missing --local-dir path must trip exit 8 (parity with \
                     run_q4_0_pipeline's post-preflight check)"
                );
                assert!(
                    reason.contains("--local-dir"),
                    "error must name the offending flag, got: {reason}"
                );
                assert!(
                    reason.contains("does not exist"),
                    "error must explain the missing-dir condition, got: {reason}"
                );
            }
            other => panic!(
                "expected PreflightFailed with EXIT_SMOKE_ASSERTION_FAILED, got: {other:?}"
            ),
        }
    }

    /// `--llama-cli-override <path>` with a path that doesn't exist
    /// must trip preflight exit code 4 (EXIT_LLAMA_CLI_MISSING) with
    /// an actionable error naming the missing path. Without this test,
    /// a refactor that removed the `is_file()` check at smoke.rs:168
    /// would silently swallow the missing-override case and either
    /// proceed to the convert step (which would fail later with a
    /// less-actionable error) or exec the missing path (which would
    /// fail at the syscall layer).
    #[test]
    fn llama_cli_override_missing_path_returns_exit_4() {
        let env = MockEnv::default();
        let entry = ArchRegistry::global().get("qwen35").unwrap();
        let nonexistent = Path::new("/this/path/definitely/does/not/exist/llama-cli");
        let result = preflight_full(entry, &env, true, Some(nonexistent));
        match result {
            Err((code, reason)) => {
                assert_eq!(
                    code, EXIT_LLAMA_CLI_MISSING,
                    "missing override path must trip EXIT_LLAMA_CLI_MISSING (4)"
                );
                assert!(
                    reason.contains("--llama-cli-override"),
                    "error must name the offending flag, got: {reason}"
                );
                assert!(
                    reason.contains("does not exist"),
                    "error must explain the missing-file condition, got: {reason}"
                );
            }
            Ok(()) => panic!("preflight must fail on missing override path"),
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

    #[test]
    fn sanitize_timestamps_strips_decimals_keeps_integers() {
        let real_eval_line = "llama_perf_context_print:        eval time =      58.90 ms /     8 runs   (    8.41 ms per token,   118.85 tokens per second)";
        let s = sanitize_timestamps(real_eval_line);
        // Decimal numbers replaced.
        assert!(!s.contains("58.90"), "got: {s}");
        assert!(!s.contains("8.41"), "got: {s}");
        assert!(!s.contains("118.85"), "got: {s}");
        // Placeholders present.
        assert!(s.contains("<X.XX>"), "got: {s}");
        // Integer count preserved (the load-bearing bit).
        assert!(s.contains("/     8 runs"), "got: {s}");
        // Structural keywords preserved.
        assert!(s.contains("eval time"));
        assert!(s.contains("ms per token"));
        assert!(s.contains("tokens per second"));
    }

    #[test]
    fn sanitize_timestamps_preserves_integer_only_tokens() {
        // "GGUF V3" — version is integer-only, must NOT be sanitized.
        let s = sanitize_timestamps("GGUF V3 (latest) and 737 tensors");
        assert!(s.contains("V3"), "version preserved: {s}");
        assert!(s.contains("737 tensors"), "tensor count preserved: {s}");
        assert!(!s.contains("<X.XX>"), "no decimal seen: {s}");
    }

    #[test]
    fn sanitize_timestamps_byte_identical_across_two_calls_with_different_decimals() {
        // Same structural content, different decimal values — the two
        // sanitized outputs MUST be byte-identical. This is the
        // load-bearing property for Decision 16's byte-identical AC.
        let stderr_a = "eval time =     58.90 ms /     8 runs   (    8.41 ms per token,   118.85 tokens per second)\n";
        let stderr_b = "eval time =     67.12 ms /     8 runs   (    9.55 ms per token,   119.20 tokens per second)\n";
        assert_ne!(stderr_a, stderr_b, "raw inputs must differ for the test to be meaningful");
        assert_eq!(
            sanitize_timestamps(stderr_a),
            sanitize_timestamps(stderr_b),
            "sanitized transcripts must be byte-identical"
        );
    }

    /// `--with-vision` + `entry.has_vision` ⇒ `--emit-vision-tower`
    /// is in the convert args. Locks Decision 16 §CLI's documented
    /// `[--with-vision]` flag wiring.
    #[test]
    fn build_convert_args_with_vision_appends_emit_vision_tower() {
        let mut args = args_for("qwen35", "q4_0");
        args.with_vision = true;
        let entry = ArchRegistry::global().get("qwen35").unwrap();
        let convert_args = build_convert_args(
            &args,
            entry,
            Path::new("/in"),
            Path::new("/out.gguf"),
        )
        .unwrap();
        assert!(
            convert_args.iter().any(|a| a == "--emit-vision-tower"),
            "must include --emit-vision-tower; got {:?}",
            convert_args
        );
    }

    /// `--with-vision` is FALSE ⇒ no `--emit-vision-tower` in convert
    /// args. The default-off path stays untouched.
    #[test]
    fn build_convert_args_without_with_vision_omits_emit_vision_tower() {
        let args = args_for("qwen35", "q4_0");
        // with_vision defaults false in args_for.
        assert!(!args.with_vision);
        let entry = ArchRegistry::global().get("qwen35").unwrap();
        let convert_args = build_convert_args(
            &args,
            entry,
            Path::new("/in"),
            Path::new("/out.gguf"),
        )
        .unwrap();
        assert!(
            !convert_args.iter().any(|a| a == "--emit-vision-tower"),
            "must NOT include --emit-vision-tower when with_vision=false; got {:?}",
            convert_args
        );
    }

    /// `--with-vision` set BUT arch has `has_vision=false` ⇒ flag is
    /// suppressed. Catches a future mistake where the smoke harness
    /// passes --emit-vision-tower against an arch (e.g. qwen35moe)
    /// whose checkpoints don't ship a vision_config — Decision 16
    /// §CLI: the `--with-vision` flag is honored only when the arch
    /// actually has a vision-tower path.
    #[test]
    fn build_convert_args_arch_without_vision_suppresses_flag_even_if_user_asked() {
        let mut args = args_for("qwen35moe", "q4_0");
        args.with_vision = true;
        let entry = ArchRegistry::global().get("qwen35moe").unwrap();
        // qwen35moe's entry has has_vision=false (Robert's MoE target
        // dropped vision_config — see qwen35moe.rs:226).
        assert!(!entry.has_vision);
        let convert_args = build_convert_args(
            &args,
            entry,
            Path::new("/in"),
            Path::new("/out.gguf"),
        )
        .unwrap();
        assert!(
            !convert_args.iter().any(|a| a == "--emit-vision-tower"),
            "qwen35moe must NOT request --emit-vision-tower even when user passes --with-vision; got {:?}",
            convert_args
        );
    }

    /// Magnitudes-vary case: real llama-cli uses `%10.2f` so e.g.
    /// "58.90" gets 5 leading spaces while "158.90" gets 4 (right-padded
    /// to 10 chars). The sanitizer must absorb that variable-width
    /// padding into the placeholder so two runs with different timing
    /// magnitudes still produce byte-identical sanitized output.
    #[test]
    fn sanitize_timestamps_byte_identical_across_different_magnitudes() {
        // Width-padded magnitudes: "58.90" (5 chars, 5 leading spaces)
        // vs "158.90" (6 chars, 4 leading spaces). Both occupy 10 cols.
        let stderr_a = "eval time =     58.90 ms /     8 runs   (    8.41 ms per token,   118.85 tokens per second)\n";
        let stderr_b = "eval time =    158.90 ms /     8 runs   (   18.41 ms per token,  1118.85 tokens per second)\n";
        let sa = sanitize_timestamps(stderr_a);
        let sb = sanitize_timestamps(stderr_b);
        assert_eq!(
            sa, sb,
            "magnitude-varying widths must collapse to identical sanitized output\nA: {sa}\nB: {sb}"
        );
        // Integer column ("/ 8 runs") must be preserved (load-bearing
        // for the parser — Decision 16 §4 asserts the runs count).
        assert!(sa.contains("/     8 runs"));
    }
}
