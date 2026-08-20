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
pub mod convert;
pub mod core;
mod debug;
pub mod distribution;
mod doctor;
pub mod gguf_patch;
pub mod inference;
pub mod input;
pub mod intelligence;
pub mod ir;
pub mod models;
pub mod progress;
pub mod quantize;
mod serve;
mod setup;

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
    Smoke {
        code: u8,
        msg: anyhow::Error,
    },
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
    // Best-effort zero-config tab completion. Release builds reconcile
    // hf2q-owned Bash, Zsh, and Fish registrations in their per-user loader
    // locations; debug/test builds require explicit isolated destinations.
    // This normally precedes clap parsing so --help/--version and parse errors
    // can complete first-run registration. `setup` and the standalone
    // installer/updater/uninstaller are closed exceptions: even help or malformed
    // input must not mutate shell integration.
    let raw_args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    if !invocation_suppresses_completion_reconciliation(&raw_args) {
        cli::completion_install::reconcile();
    }

    // Emit one-shot warning / ack-gate summary for any investigation-only
    // env vars that are set. Uses direct eprintln! (not tracing), so it
    // runs correctly before the subscriber is installed. Placed before
    // Cli::parse so the warning appears even when clap exits early on
    // --help or --version.
    debug::INVESTIGATION_ENV.activate();

    let cli = Cli::parse_from(raw_args);

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
            0 => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("hf2q=warn")),
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
                .with_writer(serve::operator_ui::LogMakeWriter)
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

fn invocation_suppresses_completion_reconciliation(args: &[std::ffi::OsString]) -> bool {
    let mut arguments = args.iter().skip(1);
    while let Some(argument) = arguments.next() {
        let Some(argument) = argument.to_str() else {
            continue;
        };
        if argument == "--" {
            return arguments.next().is_some_and(|value| {
                matches!(
                    value.to_str(),
                    Some("setup" | "__standalone-install" | "update" | "uninstall")
                )
            });
        }
        if matches!(argument, "--log-format" | "--log-level" | "--state-root") {
            let _ = arguments.next();
            continue;
        }
        if argument.starts_with('-') {
            continue;
        }
        return matches!(
            argument,
            "setup" | "__standalone-install" | "update" | "uninstall"
        );
    }
    false
}

fn run(cli: Cli) -> Result<(), AppError> {
    let log_format = cli.log_format;
    let state_root = cli.state_root;
    match cli.command {
        Command::StandaloneInstall(args) => cmd_standalone_install(args),
        Command::Update(args) => cmd_update(args),
        Command::Uninstall(args) => cmd_uninstall(args),
        Command::Setup(args) => setup::run(args, state_root.as_deref()).map_err(|error| {
            if error.is_input() {
                AppError::Input(anyhow::Error::from(error))
            } else {
                AppError::Conversion(anyhow::Error::from(error))
            }
        }),
        Command::GgufPatch(args) => cmd_gguf_patch(args),
        Command::Info(args) => cmd_info(args).map_err(AppError::Input),
        Command::Doctor => doctor::run_doctor().map_err(AppError::Conversion),
        Command::Completions(args) => cmd_completions(args).map_err(AppError::Input),
        Command::Generate(args) => serve::cmd_generate(args).map_err(AppError::Conversion),
        Command::Serve(args) => {
            let operator_config = load_operator_config(state_root.as_deref())?;
            serve::cmd_serve(
                args,
                log_format,
                operator_config.as_ref().map(|config| &config.serve),
            )
            .map_err(AppError::Conversion)
        }
        Command::Parity(args) => serve::cmd_parity(args).map_err(AppError::Conversion),
        Command::Smoke(args) => cmd_smoke(args),
        // ADR-005 Phase 3 iter-205 (AC line 5351): operator-facing
        // cache management.  Errors map to AppError::Input because
        // every failure surface (unknown_repo, unknown_quant, missing
        // --yes, mutually-exclusive-flags) is a user-input mistake;
        // exit-3 is the documented signal.
        Command::Cache(args) => serve::cmd_cache(args).map_err(AppError::Input),
        Command::Convert(args) => {
            let operator_config = load_operator_config(state_root.as_deref())?;
            cmd_convert(args, operator_config.as_ref())
        }
        Command::Tokenizer(args) => cmd_tokenizer(args),
    }
}

fn running_executable() -> Result<std::path::PathBuf, AppError> {
    std::env::current_exe()
        .context("resolve the running hf2q executable")
        .and_then(|path| {
            std::fs::canonicalize(path).context("canonicalize the running hf2q executable")
        })
        .map_err(AppError::Conversion)
}

fn cmd_update(args: cli::UpdateArgs) -> Result<(), AppError> {
    let executable = running_executable()?;
    if args.rollback {
        let install_dir = distribution::standalone::verify_running_installation(&executable)
            .map_err(|error| AppError::Conversion(anyhow::Error::from(error)))?;
        distribution::standalone::rollback(&install_dir)
            .map_err(|error| AppError::Conversion(anyhow::Error::from(error)))?;
        println!(
            "Restored the previous standalone hf2q in {}. Run `hf2q --version` to inspect it.",
            install_dir.display()
        );
        return Ok(());
    }
    let outcome = distribution::standalone::run_update(&executable, args.check)
        .map_err(|error| AppError::Conversion(anyhow::Error::from(error)))?;
    match outcome {
        distribution::standalone::UpdateOutcome::Current { version } => {
            println!("hf2q {version} is already the current stable standalone release.");
        }
        distribution::standalone::UpdateOutcome::Available { current, latest } => {
            println!("Standalone update available: {current} -> {latest}.");
        }
        distribution::standalone::UpdateOutcome::Updated { previous, current } => {
            println!(
                "Updated standalone hf2q {previous} -> {current}. Roll back with `hf2q update --rollback`."
            );
        }
    }
    Ok(())
}

fn cmd_uninstall(args: cli::UninstallArgs) -> Result<(), AppError> {
    let executable = running_executable()?;
    let install_dir = distribution::standalone::verify_running_installation(&executable)
        .map_err(|error| AppError::Conversion(anyhow::Error::from(error)))?;
    if !args.yes {
        return Err(AppError::Input(anyhow::anyhow!(
            "standalone uninstall would remove only {}, .hf2q-standalone.json, .hf2q-previous, and .hf2q-standalone.lock; configuration and models are preserved. Rerun `hf2q uninstall --yes` to confirm",
            executable.display()
        )));
    }
    distribution::standalone::uninstall(&install_dir)
        .map_err(|error| AppError::Conversion(anyhow::Error::from(error)))?;
    println!(
        "Removed standalone hf2q from {}. Configuration and models were preserved.",
        install_dir.display()
    );
    Ok(())
}

fn cmd_standalone_install(args: cli::StandaloneInstallArgs) -> Result<(), AppError> {
    let expectation =
        distribution::standalone::CandidateExpectation::from_hex(args.size, &args.sha256)
            .map_err(|error| AppError::Input(anyhow::Error::from(error)))?;
    let outcome = distribution::standalone::publish_verified_candidate(
        &args.install_dir,
        &args.candidate,
        &expectation,
    )
    .map_err(|error| AppError::Conversion(anyhow::Error::from(error)))?;
    let action = match outcome {
        distribution::standalone::PublishOutcome::Installed => "Installed",
        distribution::standalone::PublishOutcome::Updated => "Updated",
    };
    println!(
        "{action} hf2q at {}",
        args.install_dir.join("hf2q").display()
    );
    Ok(())
}

fn load_operator_config(
    state_root: Option<&std::path::Path>,
) -> Result<Option<setup::OperatorConfigV2>, AppError> {
    setup::load_operator_config(state_root).map_err(|error| {
        if error.is_input() {
            AppError::Input(anyhow::Error::from(error))
        } else {
            AppError::Conversion(anyhow::Error::from(error))
        }
    })
}

/// ADR-038 G4-CFA-5e — operator-facing tokenizer.json patching.
fn cmd_tokenizer(args: cli::TokenizerArgs) -> Result<(), AppError> {
    use cli::TokenizerAction;
    match args.action {
        TokenizerAction::FixBos {
            path,
            gguf,
            bos_id,
            bos_text,
        } => {
            // If a sibling GGUF is provided, read BOS metadata from it
            // (matches the runtime adapter's resolution path).
            let (resolved_id, resolved_text) = if let Some(gguf_path) = gguf {
                let g = mlx_native::gguf::GgufFile::open(&gguf_path).map_err(|e| {
                    AppError::Input(anyhow::anyhow!("open GGUF {}: {e}", gguf_path.display()))
                })?;
                let id = g
                    .metadata_u32("tokenizer.ggml.bos_token_id")
                    .ok_or_else(|| {
                        AppError::Input(anyhow::anyhow!(
                            "GGUF {} has no tokenizer.ggml.bos_token_id metadata",
                            gguf_path.display()
                        ))
                    })?;
                // The runtime adapter resolves BOS *text* via the
                // tokenizer's vocab. Here we don't have the tokenizer
                // loaded yet (we're about to patch its file), so fall
                // back to the operator-supplied `--bos-text` default.
                (id, bos_text)
            } else {
                (bos_id, bos_text)
            };

            let mutated =
                core::tokenizer_adapter::fix_tokenizer_json_bos(&path, &resolved_text, resolved_id)
                    .map_err(|e| {
                        AppError::Input(anyhow::anyhow!(
                            "fix_tokenizer_json_bos {}: {e}",
                            path.display()
                        ))
                    })?;
            if mutated {
                println!(
                    "Patched {}: prepended BOS SpecialToken {:?} (id={}) to post_processor.single",
                    path.display(),
                    resolved_text,
                    resolved_id,
                );
            } else {
                println!(
                    "No change to {}: post_processor.single already starts with BOS SpecialToken {:?}",
                    path.display(), resolved_text,
                );
            }
            Ok(())
        }
    }
}

/// ADR-033 P4 — drive the convert pipeline.
///
/// Parses `--quant <name>` via `QuantSelector::from_name`, resolves the
/// HF input directory (an explicit local path or a positional/`--repo`
/// canonical Hub reference; mutually exclusive — B1), resolves remote input
/// natively to one immutable commit, and hands the result to
/// [`crate::convert::run_convert`], and maps the typed `ConvertError`
/// onto `AppError::Input` (parse / arch / missing tensor — operator-input
/// issues) vs `AppError::Conversion` (source read, orchestrator, IO —
/// pipeline-internal issues).
fn cmd_convert(
    args: cli::ConvertCliArgs,
    operator_config: Option<&setup::OperatorConfigV2>,
) -> Result<(), AppError> {
    use crate::convert::{
        run_convert, ConvertArgs, ConvertError, QuantSelector, RemoteConversionSource,
    };

    // QuantSelector parses both standard ftypes (`q5_k_m`, `q8_0`, ...)
    // and Apex tiers (`apex-balanced`, `apex-i-quality`, ...). Reserved
    // names (`dwq`, bare `apex`, `tq1_0`, `tq2_0`) surface as typed
    // errors per ADR §6 reserved-name stubs.
    let quant = resolve_convert_quant(args.quant.as_deref(), operator_config)?;
    let selector =
        QuantSelector::from_name(quant).map_err(|e| AppError::Input(anyhow::anyhow!("{e}")))?;
    let source_repo = args.source_repo.clone();
    let source_revision = args.source_revision.clone();

    // ----- B1: resolve HF input directory ---------------------------------
    // Exactly one of {positional <hf_dir>, --repo <hf_repo>} must be set.
    // clap's `conflicts_with` rejects the "both set" case at parse time;
    // we still guard here as defense-in-depth so the typed error variant
    // survives any future plumbing change that bypasses clap.
    let input = classify_convert_input(args.hf_dir, args.repo, args.revision.as_deref())
        .map_err(|error| AppError::Input(anyhow::anyhow!("{error}")))?;
    let (hf_dir, mut remote_source) = match input {
        ConvertInput::Local(path) => (path, None),
        ConvertInput::Remote(reference) => {
            let progress = crate::progress::ProgressReporter::new();
            let downloaded =
                crate::input::hf_download::download_model_reference(reference, &progress)
                    .map_err(|error| AppError::Conversion(anyhow::anyhow!("{error}")))?;
            let (path, resolved, manifest) = downloaded.into_parts();
            let verified = crate::input::integrity::verify_conversion_manifest(
                resolved.repo_id(),
                resolved.revision(),
                &path,
                manifest,
            )
            .map_err(|error| AppError::Conversion(anyhow::anyhow!("{error}")))?;
            let source = RemoteConversionSource::from_verified(resolved, &path, &verified)
                .map_err(|error| AppError::Conversion(anyhow::anyhow!("{error}")))?;
            (path, Some(source))
        }
    };

    if let Some(repo) = source_repo {
        let reference =
            crate::input::hf_reference::HfModelReference::parse(&repo, source_revision.as_deref())
                .map_err(|error| AppError::Input(anyhow::anyhow!("{error}")))?;
        let revision = reference.requested_revision().ok_or_else(|| {
            AppError::Input(anyhow::anyhow!(
                "convert: --source-repo requires an exact immutable --source-revision"
            ))
        })?;
        let resolved = reference
            .clone()
            .resolve(revision)
            .map_err(|error| AppError::Input(anyhow::anyhow!("{error}")))?;
        let verified = crate::input::integrity::verify_remote_conversion_source(
            resolved.repo_id(),
            resolved.revision(),
            &hf_dir,
        )
        .map_err(|e| AppError::Conversion(anyhow::anyhow!("{e}")))?;
        remote_source = Some(
            RemoteConversionSource::from_verified(resolved, &hf_dir, &verified)
                .map_err(|e| AppError::Conversion(anyhow::anyhow!("{e}")))?,
        );
    }

    let resolved = ConvertArgs {
        hf_dir,
        selector,
        output: args.output,
        dry_run: args.dry_run,
        imatrix: args.imatrix,
        imatrix_corpus: args.imatrix_corpus,
        imatrix_out: args.imatrix_out,
        imatrix_n_ctx: args.imatrix_n_ctx,
        mmproj: args.mmproj,
        remote_source,
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
        | ConvertError::RepoAndDirMutuallyExclusive
        | ConvertError::MissingInput
        | ConvertError::RevisionRequiresRemote
        | ConvertError::HfReference(_) => AppError::Input(anyhow::anyhow!("{e}")),
        ConvertError::Source(_)
        | ConvertError::Orchestrator(_)
        | ConvertError::Io(_)
        | ConvertError::Integrity(_)
        | ConvertError::Receipt(_)
        | ConvertError::Vision(_)
        | ConvertError::HfDownload(_) => AppError::Conversion(anyhow::anyhow!("{e}")),
    })
}

fn resolve_convert_quant<'a>(
    explicit: Option<&'a str>,
    operator_config: Option<&'a setup::OperatorConfigV2>,
) -> Result<&'a str, AppError> {
    explicit
        .or_else(|| operator_config.map(|config| config.convert.quant.as_str()))
        .ok_or_else(|| {
            AppError::Input(anyhow::anyhow!(
                "convert requires --quant unless a default was recorded by `hf2q setup`"
            ))
        })
}

#[derive(Debug)]
enum ConvertInput {
    Local(PathBuf),
    Remote(crate::input::hf_reference::HfModelReference),
}

fn classify_convert_input(
    positional: Option<PathBuf>,
    repo: Option<String>,
    revision: Option<&str>,
) -> Result<ConvertInput, crate::convert::ConvertError> {
    match (positional, repo) {
        (Some(_), Some(_)) => Err(crate::convert::ConvertError::RepoAndDirMutuallyExclusive),
        (None, None) => Err(crate::convert::ConvertError::MissingInput),
        (None, Some(reference)) => Ok(ConvertInput::Remote(
            crate::input::hf_reference::HfModelReference::parse(&reference, revision)?,
        )),
        (Some(path), None) if is_explicit_local_path(&path) => {
            if revision.is_some() {
                Err(crate::convert::ConvertError::RevisionRequiresRemote)
            } else {
                Ok(ConvertInput::Local(path))
            }
        }
        (Some(path), None) => {
            let input = path.to_str().ok_or_else(|| {
                crate::convert::ConvertError::HfReference(
                    crate::input::hf_reference::HfReferenceError::InvalidRepoId {
                        repo: path.to_string_lossy().into_owned(),
                    },
                )
            })?;
            Ok(ConvertInput::Remote(
                crate::input::hf_reference::HfModelReference::parse(input, revision)?,
            ))
        }
    }
}

fn is_explicit_local_path(path: &std::path::Path) -> bool {
    if path.exists() || path.is_absolute() {
        return true;
    }
    let rendered = path.as_os_str().to_string_lossy();
    matches!(rendered.as_ref(), "." | "..")
        || rendered.starts_with("./")
        || rendered.starts_with("../")
        || rendered.starts_with('~')
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
    if matches!(
        outcome,
        arch::smoke::SmokeOutcome::Pass { .. } | arch::smoke::SmokeOutcome::Skipped { .. }
    ) {
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

    let metadata =
        input::config_parser::parse_config(&config_path).context("Failed to parse model config")?;

    println!();
    println!("{}", console::style("Model Information").bold().green());
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
    use clap::Parser;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn convert_source_classifier_preserves_explicit_local_paths() {
        for local in ["/tmp/example", "./models/example", "../models/example"] {
            let classified =
                classify_convert_input(Some(PathBuf::from(local)), None, None).unwrap();
            assert!(
                matches!(classified, ConvertInput::Local(path) if path == PathBuf::from(local))
            );
        }
        assert!(matches!(
            classify_convert_input(Some(PathBuf::from("./models/example")), None, Some("main")),
            Err(crate::convert::ConvertError::RevisionRequiresRemote)
        ));
    }

    #[test]
    fn convert_source_classifier_accepts_positional_repo_and_url() {
        for remote in [
            "Qwen/Qwen3.8-27B",
            "https://huggingface.co/Qwen/Qwen3.8-27B/tree/main",
        ] {
            let classified =
                classify_convert_input(Some(PathBuf::from(remote)), None, None).unwrap();
            let ConvertInput::Remote(reference) = classified else {
                panic!("expected remote reference");
            };
            assert_eq!(reference.repo_id(), "Qwen/Qwen3.8-27B");
        }
    }

    #[test]
    fn convert_source_classifier_reconciles_revision_and_rejects_ambiguity() {
        let classified = classify_convert_input(
            Some(PathBuf::from(
                "https://huggingface.co/Qwen/Qwen3.8-27B/tree/main",
            )),
            None,
            Some("main"),
        )
        .unwrap();
        assert!(matches!(classified, ConvertInput::Remote(_)));
        assert!(classify_convert_input(
            Some(PathBuf::from(
                "https://huggingface.co/Qwen/Qwen3.8-27B/tree/main",
            )),
            None,
            Some("other"),
        )
        .is_err());
        assert!(matches!(
            classify_convert_input(
                Some(PathBuf::from("Qwen/Qwen3.8-27B")),
                Some("Qwen/Qwen3.8-27B".to_owned()),
                None,
            ),
            Err(crate::convert::ConvertError::RepoAndDirMutuallyExclusive)
        ));
    }

    #[test]
    fn convert_config_quant_is_used_and_explicit_quant_wins() {
        let config = setup::OperatorConfigV2::guide_defaults().unwrap();
        assert_eq!(
            resolve_convert_quant(None, Some(&config)).unwrap(),
            "q4_k_m"
        );
        assert_eq!(
            resolve_convert_quant(Some("q5_k_m"), Some(&config)).unwrap(),
            "q5_k_m"
        );
        assert!(resolve_convert_quant(None, None).is_err());
    }

    #[test]
    fn invalid_operator_config_fails_before_convert_source_or_serve_bind() {
        let temp = tempfile::TempDir::new().unwrap();
        let root = temp.path().canonicalize().unwrap().join("state");
        std::fs::create_dir(&root).unwrap();
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::write(root.join("config.toml"), b"not = [valid").unwrap();
        std::fs::set_permissions(
            root.join("config.toml"),
            std::fs::Permissions::from_mode(0o600),
        )
        .unwrap();

        let output = temp.path().join("never-created-by-invalid-config.gguf");
        let convert = Cli::try_parse_from([
            "hf2q",
            "--state-root",
            root.to_str().unwrap(),
            "convert",
            "definitely-not-contacted/remote-model",
            "--output",
            output.to_str().unwrap(),
        ])
        .unwrap();
        assert!(matches!(run(convert), Err(AppError::Input(_))));
        assert!(!output.exists());

        let model = temp.path().join("definitely-not-opened.gguf");
        let serve = Cli::try_parse_from([
            "hf2q",
            "--state-root",
            root.to_str().unwrap(),
            "serve",
            "--model",
            model.to_str().unwrap(),
            "--port",
            "65534",
        ])
        .unwrap();
        assert!(matches!(run(serve), Err(AppError::Input(_))));
    }
}
