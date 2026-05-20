//! hf2q — Pure Rust CLI for quantizing HuggingFace models to GGUF and safetensors.
//!
//! Entry point: dispatches clap subcommands to appropriate handlers.
//!
//! Exit codes:
//!   0 = success
//!   1 = conversion error
//!   2 = quality threshold exceeded
//!   3 = input/validation error

pub mod arch;
pub mod backends;
pub mod cli;
// `core` is the in-place precursor to the planned `hf2q-core` crate
// (workspace v0.1.0 split). See `src/core/mod.rs` for the boundary
// rule and the planned submodule layout.
pub mod core;
pub mod convert;
mod debug;
mod doctor;
pub mod models;
pub mod gguf_patch;
pub mod inference;
pub mod input;
pub mod intelligence;
pub mod ir;
pub mod progress;
pub mod quantize;
mod serve;

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::error;

use cli::{Cli, Command};

/// Exit codes.
///
/// 0/1/3 are the long-standing convert codes.  4–6 are added by
/// ADR-012 P8's `hf2q smoke` subcommand for distinct preflight failure modes
/// (per Decision 16 acceptance: each preflight failure surfaces a unique
/// non-zero code so a CI runner can tell "missing token" from "missing disk").
const EXIT_SUCCESS: u8 = 0;
const EXIT_CONVERSION_ERROR: u8 = 1;
const EXIT_INPUT_ERROR: u8 = 3;

/// Error types for exit code classification.
#[derive(Debug)]
enum AppError {
    Input(anyhow::Error),
    Conversion(anyhow::Error),
    /// Smoke-subcommand exit codes per ADR-012 Decision 16 §preflight (2-8).
    /// Carries the smoke-specific code so the process exits with the
    /// documented value rather than the generic `EXIT_CONVERSION_ERROR=1`
    /// AppError default. Without this variant, every distinct smoke
    /// failure mode collapses to exit 1 — defeating Decision 16's
    /// "distinct non-zero code" contract at the OS-process level.
    Smoke { code: u8, msg: anyhow::Error },
}

impl AppError {
    fn exit_code(&self) -> u8 {
        match self {
            AppError::Input(_) => EXIT_INPUT_ERROR,
            AppError::Conversion(_) => EXIT_CONVERSION_ERROR,
            AppError::Smoke { code, .. } => *code,
        }
    }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::Input(e) => write!(f, "{:#}", e),
            AppError::Conversion(e) => write!(f, "{:#}", e),
            AppError::Smoke { msg, .. } => write!(f, "{:#}", msg),
        }
    }
}

fn main() -> ExitCode {
    // Emit one-shot warning / ack-gate summary for any investigation-only
    // env vars that are set. Uses direct eprintln! (not tracing), so it
    // runs correctly before the subscriber is installed. Placed before
    // Cli::parse so the warning appears even when clap exits early on
    // --help or --version.
    debug::INVESTIGATION_ENV.activate();

    let cli = Cli::parse();

    // Logging subscriber init. Priority:
    //   1. --log-level (explicit) overrides everything.
    //   2. -v/-vv/-vvv bumps verbosity.
    //   3. RUST_LOG env var.
    //   4. Default: hf2q=warn (silent on the generate boot path).
    // Log format (text/json) comes from --log-format (Decision #11).
    // Stderr writer: logs never touch stdout, keeping the generation
    // stream unpolluted. ANSI colors only when stderr is a TTY for
    // text format; JSON format is always ANSI-free.
    use std::io::IsTerminal;
    use tracing_subscriber::EnvFilter;
    let filter = if let Some(lvl) = cli.log_level {
        EnvFilter::new(format!("hf2q={lvl},mlx_native={lvl}", lvl = lvl.as_str()))
    } else {
        match cli.verbose {
            0 => EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("hf2q=warn")),
            1 => EnvFilter::new("hf2q=info,mlx_native=info"),
            2 => EnvFilter::new("hf2q=debug,mlx_native=debug"),
            _ => EnvFilter::new("hf2q=trace,mlx_native=trace"),
        }
    };
    let stderr_is_tty = std::io::stderr().is_terminal();
    match cli.log_format {
        cli::LogFormat::Text => {
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_writer(std::io::stderr)
                .with_ansi(stderr_is_tty)
                .without_time()
                .init();
        }
        cli::LogFormat::Json => {
            tracing_subscriber::fmt()
                .json()
                .with_env_filter(filter)
                .with_writer(std::io::stderr)
                .with_current_span(false)
                .with_span_list(false)
                .init();
        }
    }

    match run(cli) {
        Ok(()) => ExitCode::from(EXIT_SUCCESS),
        Err(app_err) => {
            let exit_code = app_err.exit_code();
            error!("{}", app_err);
            eprintln!("Error: {}", app_err);
            ExitCode::from(exit_code)
        }
    }
}

fn run(cli: Cli) -> Result<(), AppError> {
    match cli.command {
        Command::GgufPatch(args) => cmd_gguf_patch(args),
        Command::Info(args) => cmd_info(args).map_err(AppError::Input),
        Command::Doctor => doctor::run_doctor().map_err(AppError::Conversion),
        Command::Completions(args) => cmd_completions(args).map_err(AppError::Input),
        Command::Generate(args) => serve::cmd_generate(args).map_err(AppError::Conversion),
        Command::Serve(args) => serve::cmd_serve(args).map_err(AppError::Conversion),
        Command::Parity(args) => serve::cmd_parity(args).map_err(AppError::Conversion),
        Command::Smoke(args) => cmd_smoke(args),
        // ADR-005 Phase 3 iter-205 (AC line 5351): operator-facing
        // cache management.  Errors map to AppError::Input because
        // every failure surface (unknown_repo, unknown_quant, missing
        // --yes, mutually-exclusive-flags) is a user-input mistake;
        // exit-3 is the documented signal.
        Command::Cache(args) => serve::cmd_cache(args).map_err(AppError::Input),
        Command::Convert(args) => cmd_convert(args),
    }
}

/// ADR-033 P4 — drive the convert pipeline.
///
/// Parses `--quant <name>` via `QuantSelector::from_name`, resolves the
/// HF input directory (positional `<hf_dir>` OR auto-download via
/// `--repo <hf_repo>`; mutually exclusive — B1), hands the result to
/// [`crate::convert::run_convert`], and maps the typed `ConvertError`
/// onto `AppError::Input` (parse / arch / missing tensor — operator-input
/// issues) vs `AppError::Conversion` (source read, orchestrator, IO —
/// pipeline-internal issues).
fn cmd_convert(args: cli::ConvertCliArgs) -> Result<(), AppError> {
    use crate::convert::{run_convert, ConvertArgs, ConvertError, QuantSelector};

    // QuantSelector parses both standard ftypes (`q5_k_m`, `q8_0`, ...)
    // and Apex tiers (`apex-balanced`, `apex-i-quality`, ...). Reserved
    // names (`dwq`, bare `apex`, `tq1_0`, `tq2_0`) surface as typed
    // errors per ADR §6 reserved-name stubs.
    let selector = QuantSelector::from_name(&args.quant)
        .map_err(|e| AppError::Input(anyhow::anyhow!("{e}")))?;

    // ----- B1: resolve HF input directory ---------------------------------
    // Exactly one of {positional <hf_dir>, --repo <hf_repo>} must be set.
    // clap's `conflicts_with` rejects the "both set" case at parse time;
    // we still guard here as defense-in-depth so the typed error variant
    // survives any future plumbing change that bypasses clap.
    let hf_dir = match (args.hf_dir, args.repo) {
        (Some(_), Some(_)) => {
            return Err(AppError::Input(anyhow::anyhow!(
                "{}",
                ConvertError::RepoAndDirMutuallyExclusive
            )));
        }
        (Some(path), None) => path,
        (None, Some(repo)) => {
            download_repo_via_hf_cli(&repo).map_err(|e| AppError::Conversion(anyhow::anyhow!("{e}")))?
        }
        (None, None) => {
            return Err(AppError::Input(anyhow::anyhow!(
                "convert: either positional `<hf_dir>` or `--repo <hf_repo>` is required"
            )));
        }
    };

    let resolved = ConvertArgs {
        hf_dir,
        selector,
        output: args.output,
        imatrix: args.imatrix,
        imatrix_corpus: args.imatrix_corpus,
        imatrix_out: args.imatrix_out,
        imatrix_n_ctx: args.imatrix_n_ctx,
    };
    run_convert(resolved).map_err(|e| match e {
        ConvertError::UnsupportedArch { .. }
        | ConvertError::UnmappedTensor { .. }
        | ConvertError::MissingHparam { .. }
        | ConvertError::IncompleteExpertGroup { .. }
        | ConvertError::DuplicateExpertIndex { .. }
        | ConvertError::ApexMissingLayerCount
        | ConvertError::ApexCustomOutOfScope { .. }
        | ConvertError::Apex(_)
        | ConvertError::Tokenizer(_)
        | ConvertError::Imatrix(_)
        | ConvertError::ImatrixRequiredForITier { .. }
        | ConvertError::ImatrixNCtxInvalid { .. }
        | ConvertError::RepoAndDirMutuallyExclusive => {
            AppError::Input(anyhow::anyhow!("{e}"))
        }
        ConvertError::Source(_)
        | ConvertError::Orchestrator(_)
        | ConvertError::Io(_)
        | ConvertError::HfDownload { .. } => {
            AppError::Conversion(anyhow::anyhow!("{e}"))
        }
    })
}

/// B1 — sanitize an HF repo id for filesystem use.
///
/// Replaces every `/` with `__` so `google/gemma-4-26b-a4b-it` becomes
/// `google__gemma-4-26b-a4b-it`. Other characters pass through.
/// Centralized as a pure function so the unit test can pin the contract
/// without invoking the download subprocess.
fn sanitize_repo_for_cache_dir(repo: &str) -> String {
    repo.replace('/', "__")
}

/// B1 — shell out to `huggingface-cli download <repo> --local-dir <cache>`
/// and return the cache directory on success.
///
/// `<cache>` = `~/.cache/hf2q/repos/<sanitize_repo_for_cache_dir(repo)>/`.
/// The directory is created if missing; existing partial downloads are
/// resumed by `huggingface-cli`'s own logic. Stdout/stderr stream
/// through to the operator's terminal so download progress is visible;
/// on non-zero exit we capture the tail of stderr into the typed
/// `ConvertError::HfDownload` variant.
fn download_repo_via_hf_cli(repo: &str) -> Result<PathBuf, crate::convert::ConvertError> {
    use crate::convert::ConvertError;

    // Resolve cache root: ~/.cache/hf2q/repos/<sanitized>/.
    // `home::home_dir()` is unavailable in std; fall back to $HOME env.
    let home = std::env::var("HOME").map_err(|_| ConvertError::HfDownload {
        repo: repo.to_string(),
        exit_code: None,
        stderr: "HOME env var not set — cannot resolve ~/.cache/hf2q/repos/".to_string(),
    })?;
    let cache_dir = PathBuf::from(home)
        .join(".cache")
        .join("hf2q")
        .join("repos")
        .join(sanitize_repo_for_cache_dir(repo));
    std::fs::create_dir_all(&cache_dir).map_err(|e| ConvertError::HfDownload {
        repo: repo.to_string(),
        exit_code: None,
        stderr: format!("failed to create cache dir `{}`: {e}", cache_dir.display()),
    })?;

    eprintln!(
        "[hf2q convert --repo] downloading {repo} → {} via huggingface-cli",
        cache_dir.display()
    );

    let output = std::process::Command::new("huggingface-cli")
        .arg("download")
        .arg(repo)
        .arg("--local-dir")
        .arg(&cache_dir)
        .output();

    let output = match output {
        Ok(o) => o,
        Err(e) => {
            return Err(ConvertError::HfDownload {
                repo: repo.to_string(),
                exit_code: None,
                stderr: format!(
                    "failed to spawn `huggingface-cli`: {e} \
                     (is huggingface-cli on PATH? `pip install -U huggingface_hub[cli]`)"
                ),
            });
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        return Err(ConvertError::HfDownload {
            repo: repo.to_string(),
            exit_code: output.status.code(),
            stderr,
        });
    }

    Ok(cache_dir)
}

fn cmd_gguf_patch(args: cli::GgufPatchArgs) -> Result<(), AppError> {
    if !args.dry_run && !args.in_place && args.output.is_none() {
        return Err(AppError::Input(anyhow::anyhow!(
            "gguf-patch requires --output <out> or --in-place unless --dry-run is set"
        )));
    }

    gguf_patch::patch_chat_template_from_arch(gguf_patch::GgufPatchOptions {
        input: args.input,
        output: args.output,
        in_place: args.in_place,
        dry_run: args.dry_run,
    })
    .map(|_| ())
    .map_err(AppError::Conversion)
}

/// Handle the `smoke` subcommand — ADR-012 Decision 16.
///
/// Dispatches via `ArchRegistry::get(arch)` — unknown arches (including
/// gemma4, ministral, deepseekv3, bogus) return a uniform structured
/// error. Preflight failures map to the documented exit codes 2-6.
fn cmd_smoke(args: cli::SmokeArgs) -> Result<(), AppError> {
    let smoke_args = arch::smoke::SmokeArgs {
        arch: args.arch,
        quant: arch::smoke::normalize_quant_label(&args.quant),
        with_vision: args.with_vision,
        skip_convert: args.skip_convert,
        dry_run: args.dry_run,
        fixtures_root: args.fixtures_root,
        local_dir: args.local_dir,
        convert_output_dir: args.convert_output_dir,
        llama_cli_override: args.llama_cli_override,
    };
    let env = arch::smoke::RealSmokeEnv {
        convert_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    };
    let outcome = arch::smoke::dispatch(&smoke_args, &env);
    let code = outcome.exit_code();
    let rendered = arch::smoke::render_outcome(&outcome);
    if matches!(outcome, arch::smoke::SmokeOutcome::Pass { .. }
                        | arch::smoke::SmokeOutcome::Skipped { .. })
    {
        println!("{}", rendered);
        Ok(())
    } else {
        // Preflight / unknown-arch — propagate the smoke-specific exit
        // code (Decision 16 §preflight: 2-8 distinct non-zero codes)
        // rather than collapsing to AppError::Conversion's exit 1.
        // Without `AppError::Smoke`, the documented exit codes were
        // shadowed at the process boundary — fixed in this commit.
        eprintln!("{}", rendered);
        Err(AppError::Smoke {
            code,
            msg: anyhow::anyhow!("{}", rendered),
        })
    }
}


/// Handle the `info` subcommand.
fn cmd_info(args: cli::InfoArgs) -> Result<()> {
    let input_dir = resolve_info_input(&args)?;

    let config_path = input_dir.join("config.json");
    if !config_path.exists() {
        anyhow::bail!(
            "No config.json found in {}. Is this a HuggingFace model directory?",
            input_dir.display()
        );
    }

    let metadata = input::config_parser::parse_config(&config_path)
        .context("Failed to parse model config")?;

    println!();
    println!(
        "{}",
        console::style("Model Information").bold().green()
    );
    println!("{}", input::config_parser::format_info(&metadata));
    println!();

    Ok(())
}

/// Resolve the input directory for the info subcommand.
fn resolve_info_input(args: &cli::InfoArgs) -> Result<PathBuf> {
    match (&args.input, &args.repo) {
        (Some(path), None) => {
            if !path.exists() {
                anyhow::bail!("Input directory does not exist: {}", path.display());
            }
            Ok(path.clone())
        }
        (None, Some(repo_id)) => {
            let progress = progress::ProgressReporter::new();
            let download_dir = input::hf_download::download_model(repo_id, &progress)
                .context("Failed to download model from HuggingFace Hub")?;
            Ok(download_dir)
        }
        (None, None) => {
            anyhow::bail!("Either --input or --repo must be specified");
        }
        (Some(_), Some(_)) => {
            anyhow::bail!("--input and --repo are mutually exclusive");
        }
    }
}

/// Handle the `completions` subcommand.
fn cmd_completions(args: cli::CompletionsArgs) -> Result<()> {
    use clap::CommandFactory;
    use clap_complete::generate;

    let mut cmd = Cli::command();
    generate(args.shell, &mut cmd, "hf2q", &mut std::io::stdout());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// B1 — `/` in the HF repo id is replaced with `__` so the
    /// resulting string is filesystem-safe. Pins the exact spec from
    /// the rename/B1 mission.
    #[test]
    fn sanitize_repo_for_cache_dir_replaces_slash_with_double_underscore() {
        assert_eq!(
            sanitize_repo_for_cache_dir("google/gemma-4-26b-a4b-it"),
            "google__gemma-4-26b-a4b-it"
        );
    }

    /// B1 — repo without `/` passes through unchanged (degenerate input
    /// from a custom local URL alias; we still want it filesystem-safe).
    #[test]
    fn sanitize_repo_for_cache_dir_passes_through_when_no_slash() {
        assert_eq!(sanitize_repo_for_cache_dir("local-only"), "local-only");
    }

    /// B1 — repo with multiple `/` (uncommon HF nested-org pattern)
    /// replaces every separator, not just the first.
    #[test]
    fn sanitize_repo_for_cache_dir_replaces_every_slash() {
        assert_eq!(
            sanitize_repo_for_cache_dir("org/sub/model"),
            "org__sub__model"
        );
    }

    /// B1 — `cmd_convert` rejects "both `<hf_dir>` and `--repo`" with
    /// the typed `RepoAndDirMutuallyExclusive` variant, mapped to
    /// `AppError::Input` (exit code 3). clap's `conflicts_with` should
    /// catch this at parse time; this test pins the defense-in-depth
    /// path in case clap's rules change.
    #[test]
    fn cmd_convert_rejects_repo_and_dir_both_set() {
        let args = cli::ConvertCliArgs {
            hf_dir: Some(PathBuf::from("/tmp/example")),
            repo: Some("org/repo".to_string()),
            quant: "q8_0".to_string(),
            output: PathBuf::from("/tmp/out.gguf"),
            imatrix: None,
            imatrix_corpus: None,
            imatrix_out: None,
            imatrix_n_ctx: None,
        };
        let err = cmd_convert(args).expect_err("must error");
        match err {
            AppError::Input(e) => {
                let s = format!("{e:#}");
                assert!(
                    s.contains("mutually exclusive"),
                    "expected mutually-exclusive diagnostic, got `{s}`"
                );
            }
            other => panic!("expected AppError::Input, got {other:?}"),
        }
    }
}

