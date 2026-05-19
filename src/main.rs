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
        Command::ConvertV2(args) => cmd_convert_v2(args),
    }
}

/// ADR-033 P4 — drive the new convert pipeline.
///
/// Parses `--quant <name>` via `LlamaFtype::from_name`, hands the
/// resolved args to [`crate::convert::run_convert_v2`], and maps the
/// typed `ConvertV2Error` onto `AppError::Input` (parse / arch / missing
/// tensor — operator-input issues) vs `AppError::Conversion` (source
/// read, orchestrator, IO — pipeline-internal issues). Mirrors the
/// existing legacy `cmd_convert` exit-code convention.
fn cmd_convert_v2(args: cli::ConvertV2CliArgs) -> Result<(), AppError> {
    use crate::convert::{run_convert_v2, ConvertV2Args, ConvertV2Error, QuantSelector};

    // QuantSelector parses both standard ftypes (`q5_k_m`, `q8_0`, ...)
    // and Apex tiers (`apex-balanced`, `apex-i-quality`, ...). Reserved
    // names (`dwq`, bare `apex`, `tq1_0`, `tq2_0`) surface as typed
    // errors per ADR §6 reserved-name stubs.
    let selector = QuantSelector::from_name(&args.quant)
        .map_err(|e| AppError::Input(anyhow::anyhow!("{e}")))?;
    let resolved = ConvertV2Args {
        hf_dir: args.hf_dir,
        selector,
        output: args.output,
    };
    run_convert_v2(resolved).map_err(|e| match e {
        ConvertV2Error::UnsupportedArch { .. }
        | ConvertV2Error::UnmappedTensor { .. }
        | ConvertV2Error::MissingHparam { .. }
        | ConvertV2Error::IncompleteExpertGroup { .. }
        | ConvertV2Error::DuplicateExpertIndex { .. }
        | ConvertV2Error::ApexMissingLayerCount
        | ConvertV2Error::ApexCustomOutOfScope { .. }
        | ConvertV2Error::Apex(_)
        | ConvertV2Error::Tokenizer(_) => AppError::Input(anyhow::anyhow!("{e}")),
        ConvertV2Error::Source(_) | ConvertV2Error::Orchestrator(_) | ConvertV2Error::Io(_) => {
            AppError::Conversion(anyhow::anyhow!("{e}"))
        }
    })
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

