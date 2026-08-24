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
mod chat;
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
mod model_spec;
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
    /// The command already emitted its complete operator-facing diagnostic.
    /// Preserve the nonzero classification without appending a second line.
    ReportedInput,
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
            AppError::ReportedInput => EXIT_INPUT_ERROR,
            AppError::Smoke { code, .. } => *code,
        }
    }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::Input(e) => write!(f, "{:#}", e),
            AppError::Conversion(e) => write!(f, "{:#}", e),
            AppError::ReportedInput => Ok(()),
            AppError::Smoke { msg, .. } => write!(f, "{:#}", msg),
        }
    }
}

fn main() -> ExitCode {
    // Dynamic completion is the first operational branch. A completion
    // request exits before diagnostics, logging, configuration, cache access,
    // downloads, or model/runtime initialization.
    cli::completion_install::complete_env();

    let raw_args: Vec<std::ffi::OsString> = std::env::args_os().collect();

    // Proven standalone/Cargo release installations keep their owned Bash,
    // Zsh, and Fish adapters synchronized with this exact binary. Other
    // binaries require explicit isolated destinations; protocol calls never
    // reconcile.
    cli::completion_install::reconcile(&raw_args);
    cli::completion_install::report_outcome();

    let cli = match Cli::try_parse_from(raw_args) {
        Ok(cli) => cli,
        Err(error) => {
            // Preserve the established diagnostics contract for Clap's early
            // --help/--version/error exits. Valid commands initialize typed
            // overrides below before this read-once snapshot is activated.
            debug::INVESTIGATION_ENV.activate();
            error.exit();
        }
    };

    // Apply the typed generate override before the read-once investigation
    // snapshot initializes. Clap has already validated the value set.
    if let Command::Generate(args) = &cli.command {
        if let Some(bits) = args.kv_bits.as_deref() {
            debug::investigation_env::set_cli_tq_codebook_bits(bits)
                .expect("Clap-validated --kv-bits must initialize once");
        }
    }

    // Emit one-shot warning / ack-gate summary for development-only env
    // controls. Real operator flags are already typed above.
    debug::INVESTIGATION_ENV.activate();

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
        Err(AppError::ReportedInput) => ExitCode::from(EXIT_INPUT_ERROR),
        Err(app_err) => {
            let exit_code = app_err.exit_code();
            error!("{}", app_err);
            eprintln!("Error: {}", app_err);
            ExitCode::from(exit_code)
        }
    }
}

fn run(cli: Cli) -> Result<(), AppError> {
    let log_format = cli.log_format;
    let state_root = cli.state_root;
    match cli.command {
        Command::StandaloneInstall(args) => cmd_standalone_install(args),
        Command::FetchHubGguf(args) => cmd_fetch_hub_gguf(args),
        Command::CatalogHubGguf(args) => cmd_catalog_hub_gguf(args),
        Command::VerifyLocalGguf(args) => cmd_verify_local_gguf(args),
        Command::Update(args) => cmd_update(args),
        Command::Uninstall(args) => cmd_uninstall(args, state_root.as_deref()),
        Command::Setup(args) => setup::run(args, state_root.as_deref()).map_err(|error| {
            if error.is_input() {
                AppError::Input(anyhow::Error::from(error))
            } else {
                AppError::Conversion(anyhow::Error::from(error))
            }
        }),
        Command::GgufPatch(args) => cmd_gguf_patch(args),
        Command::Info(args) => {
            let operator_config = match load_operator_config(state_root.as_deref()) {
                Ok(config) => config,
                Err(error) => {
                    serve::print_info_early_rejection(&args.model, &error.to_string());
                    return Err(AppError::ReportedInput);
                }
            };
            serve::cmd_info(args, operator_config.as_ref().map(|config| &config.serve))
                .map_err(|_| AppError::ReportedInput)
        }
        Command::SourceTeacher(args) => cmd_source_teacher(args),
        Command::SourceTeacherReference(args) => cmd_source_teacher_reference(args),
        Command::SourceTeacherAcceptanceVerify(args) => cmd_source_teacher_acceptance_verify(args),
        Command::Doctor => doctor::run_doctor().map_err(AppError::Conversion),
        Command::Completions(args) => cmd_completions(args).map_err(AppError::Input),
        Command::Generate(args) => serve::cmd_generate(args).map_err(AppError::Conversion),
        Command::Chat(args) => {
            chat::cmd_chat(args, state_root.as_deref()).map_err(AppError::Conversion)
        }
        Command::Serve(args) => {
            if args.target.as_deref().is_some_and(|target| {
                matches!(
                    crate::model_spec::parse_model_spec(target),
                    Ok(crate::model_spec::ModelSpec::List)
                )
            }) {
                return serve::managed_artifacts::print_inventory(&args.model_dirs)
                    .map_err(AppError::Conversion);
            }
            let operator_config = load_operator_config(state_root.as_deref())?;
            serve::cmd_serve(
                args,
                log_format,
                operator_config.as_ref().map(|config| &config.serve),
                operator_config
                    .as_ref()
                    .map(|config| config.convert.quant.as_str()),
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

fn cmd_fetch_hub_gguf(args: cli::FetchHubGgufArgs) -> Result<(), AppError> {
    let artifact = input::hf_download::HubGgufArtifact {
        repository: args.repository,
        revision: args.revision,
        filename: args.artifact,
        bytes: args.bytes,
        sha256: args.sha256,
        quant_hint: Some(args.quant),
        role: "text_model".to_owned(),
        selectable: true,
        unavailable_reason: None,
    };
    let path = input::hf_download::download_hub_gguf(&artifact)
        .map_err(|error| AppError::Conversion(anyhow::Error::from(error)))?;
    println!("{}", path.display());
    Ok(())
}

fn cmd_catalog_hub_gguf(args: cli::CatalogHubGgufArgs) -> Result<(), AppError> {
    let reference = input::hf_reference::HfModelReference::parse(&args.repository, None)
        .map_err(|error| AppError::Input(anyhow::Error::from(error)))?;
    let catalog = input::hf_download::resolve_hub_gguf_catalog(reference)
        .map_err(|error| AppError::Conversion(anyhow::Error::from(error)))?;
    serde_json::to_writer(std::io::stdout().lock(), &catalog)
        .map_err(|error| AppError::Conversion(anyhow::Error::from(error)))?;
    println!();
    Ok(())
}

fn cmd_verify_local_gguf(args: cli::VerifyLocalGgufArgs) -> Result<(), AppError> {
    let quant = serve::quant_select::QuantType::from_canonical_str(&args.quant)
        .map_err(|error| AppError::Input(anyhow::anyhow!(error)))?;
    let receipt = serve::api::local_artifacts::verify_local_artifact(
        serve::api::local_artifacts::LocalVerificationRequest {
            root: &args.root,
            artifact: &args.artifact,
            bytes: args.bytes,
            sha256: &args.sha256,
            quant,
        },
    )
    .map_err(AppError::Input)?;
    serde_json::to_writer(std::io::stdout().lock(), &receipt)
        .map_err(|error| AppError::Conversion(anyhow::Error::from(error)))?;
    println!();
    Ok(())
}

fn cmd_source_teacher(args: cli::SourceTeacherArgs) -> Result<(), AppError> {
    use crate::inference::models::qwen35::source_precision::{
        preflight_official_qwen38_source_teacher, run_official_qwen38_source_teacher,
        OfficialQwen38EvaluationSplitV1, OfficialQwen38SourceTeacherRequestV1,
    };

    let evaluation_split = match args.evaluation_split {
        cli::SourceTeacherEvaluationSplitArg::Calibration => {
            OfficialQwen38EvaluationSplitV1::Calibration
        }
        cli::SourceTeacherEvaluationSplitArg::PolicyValidation => {
            OfficialQwen38EvaluationSplitV1::PolicyValidation
        }
    };

    let request = OfficialQwen38SourceTeacherRequestV1 {
        model_dir: args.model_dir,
        output: args.output,
        evaluation_split,
    };
    let summary = if args.execute {
        run_official_qwen38_source_teacher(request)
    } else {
        preflight_official_qwen38_source_teacher(&request)
    }
    .map_err(AppError::Conversion)?;
    println!(
        "{}",
        serde_json::to_string(&summary).map_err(|error| {
            AppError::Conversion(anyhow::anyhow!(
                "serialize source-teacher evidence summary: {error}"
            ))
        })?
    );
    Ok(())
}

fn cmd_source_teacher_acceptance_verify(
    args: cli::SourceTeacherAcceptanceVerifyArgs,
) -> Result<(), AppError> {
    use crate::inference::models::qwen35::source_precision::verify_official_qwen38_acceptance_evidence;

    let receipt = verify_official_qwen38_acceptance_evidence(&args.model_dir)
        .map_err(AppError::Conversion)?;
    println!(
        "{}",
        serde_json::to_string(&receipt).map_err(|error| {
            AppError::Conversion(anyhow::anyhow!(
                "serialize closed acceptance quality-gate receipt: {error}"
            ))
        })?
    );
    Ok(())
}

fn cmd_source_teacher_reference(args: cli::SourceTeacherReferenceArgs) -> Result<(), AppError> {
    use crate::inference::models::qwen35::source_precision::{
        compare_official_qwen38_source_reference, OfficialQwen38SourceReferenceRequestV1,
    };

    let receipt =
        compare_official_qwen38_source_reference(&OfficialQwen38SourceReferenceRequestV1 {
            model_dir: args.model_dir,
            native_summary: args.native_summary,
            native_target: args.native_target,
            external_evidence: args.external_evidence,
            external_target: args.external_target,
        })
        .map_err(AppError::Conversion)?;
    println!(
        "{}",
        serde_json::to_string(&receipt).map_err(|error| {
            AppError::Conversion(anyhow::anyhow!(
                "serialize source-reference comparison receipt: {error}"
            ))
        })?
    );
    Ok(())
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
    distribution::commands::update(args, &executable).map_err(lifecycle_app_error)
}

fn cmd_uninstall(
    args: cli::UninstallArgs,
    state_root: Option<&std::path::Path>,
) -> Result<(), AppError> {
    let executable = running_executable()?;
    distribution::commands::uninstall(args, state_root, &executable).map_err(lifecycle_app_error)
}

fn lifecycle_app_error(error: distribution::commands::LifecycleError) -> AppError {
    let is_input = error.is_input();
    let error = anyhow::Error::from(error);
    if is_input {
        AppError::Input(error)
    } else {
        AppError::Conversion(error)
    }
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
fn plan_remote_native_product_bytes(
    hf_dir: &std::path::Path,
    selector: &crate::convert::QuantSelector,
    reference: crate::input::hf_reference::ResolvedHfModelReference,
    source_sha256: &str,
    requires_projector: bool,
    text_only: bool,
    projector_only: bool,
) -> anyhow::Result<Option<u64>> {
    let crate::convert::QuantSelector::Standard(ftype) = selector else {
        return Ok(None);
    };
    let projector_planned = projector_only || (requires_projector && !text_only);
    let text = if projector_only {
        0
    } else {
        crate::convert::cli_driver::plan_standard_text_output_bytes(
            hf_dir,
            *ftype,
            reference,
            source_sha256.to_owned(),
            projector_planned,
        )?
    };
    let projector = if projector_planned {
        crate::models::vit::planned_vision_tower_output_bytes(
            hf_dir,
            Some(source_sha256),
            (!projector_only).then_some("00000000-0000-0000-0000-000000000000"),
        )?
    } else {
        0
    };
    Ok(Some(text.checked_add(projector).ok_or_else(|| {
        anyhow::anyhow!("native conversion product plan overflowed u64")
    })?))
}

fn cmd_convert(
    args: cli::ConvertCliArgs,
    operator_config: Option<&setup::OperatorConfigV2>,
) -> Result<(), AppError> {
    use crate::convert::{
        run_convert, ConvertArgs, ConvertError, ConvertMode, RemoteConversionSource,
    };

    // QuantSelector parses both standard ftypes (`q5_k_m`, `q8_0`, ...)
    // and Apex tiers (`apex-balanced`, `apex-i-quality`, ...). Reserved
    // names (`dwq`, bare `apex`, `tq1_0`, `tq2_0`) surface as typed
    // errors per ADR §6 reserved-name stubs.
    let source_repo = args.source_repo.clone();
    let source_revision = args.source_revision.clone();

    // ----- B1: resolve HF input directory ---------------------------------
    // Exactly one of {positional <hf_dir>, --repo <hf_repo>} must be set.
    // clap's `conflicts_with` rejects the "both set" case at parse time;
    // we still guard here as defense-in-depth so the typed error variant
    // survives any future plumbing change that bypasses clap.
    let input = classify_convert_input(args.hf_dir, args.repo, args.revision.as_deref())
        .map_err(|error| AppError::Input(anyhow::anyhow!("{error}")))?;
    let operand_quant = match &input {
        ConvertInput::Local(_) => None,
        ConvertInput::Remote { operand_quant, .. } => operand_quant.as_deref(),
    };
    let selector = resolve_convert_selector(args.quant.as_deref(), operand_quant, operator_config)?;
    let quant_name = selector.receipt_name();
    let implicit_output = args.output.is_none();
    let no_clobber = implicit_output || args.no_clobber;
    let mut remote_conversion_lease = None;
    if source_repo.is_none() {
        if let ConvertInput::Remote { reference, .. } = &input {
            match prepare_remote_conversion_lease(
                reference,
                &quant_name,
                args.output.as_deref(),
                no_clobber,
                implicit_output,
                args.text_only,
                args.mmproj,
                args.mmproj_output.as_deref(),
            )? {
                RemoteConversionDecision::Reuse(output) => {
                    println!(
                        "Using existing verified hf2q conversion: {}",
                        output.display()
                    );
                    return Ok(());
                }
                RemoteConversionDecision::Proceed(lease) => remote_conversion_lease = Some(lease),
            }
        }
    }
    let (hf_dir, mut remote_source) = match input {
        ConvertInput::Local(path) => (path, None),
        ConvertInput::Remote { reference, .. } => {
            let prepared =
                crate::input::hf_download::prepare_native_planning_source(reference.clone())
                    .map_err(|error| AppError::Conversion(anyhow::anyhow!("{error}")))?;
            let source_plan = prepared.source_plan();
            let lease = remote_conversion_lease.as_ref().ok_or_else(|| {
                AppError::Conversion(anyhow::anyhow!(
                    "remote conversion reached source planning without its operation lease"
                ))
            })?;
            if source_plan.repository != reference.repo_id()
                || !source_plan.revision.eq_ignore_ascii_case(&lease.revision)
            {
                return Err(AppError::Conversion(anyhow::anyhow!(
                    "remote conversion source revision changed after operation-lock planning"
                )));
            }
            let planned_output = lease.output.clone();
            let planning_reference = crate::input::hf_reference::HfModelReference::parse(
                &source_plan.repository,
                Some(&source_plan.revision),
            )
            .map_err(|error| AppError::Input(anyhow::anyhow!("{error}")))?
            .resolve(&source_plan.revision)
            .map_err(|error| AppError::Input(anyhow::anyhow!("{error}")))?;
            let planned_output_bytes = plan_remote_native_product_bytes(
                prepared.path(),
                &selector,
                planning_reference,
                prepared.source_bundle_sha256(),
                source_plan.requires_projector,
                args.text_only,
                args.mmproj,
            )
            .map_err(AppError::Conversion)?
            .unwrap_or_else(|| {
                crate::serve::managed_artifacts::planned_native_product_bytes(
                    source_plan.total_weight_bytes,
                    source_plan.output_upper_bound_bytes,
                    source_plan.requires_projector,
                    args.text_only,
                    args.mmproj,
                )
            });
            crate::input::hf_download::check_native_source_conversion_plan(
                &source_plan,
                &planned_output,
                planned_output_bytes,
            )
            .map_err(|error| AppError::Conversion(anyhow::anyhow!("{error}")))?;
            let progress = crate::progress::ProgressReporter::new();
            let pinned = crate::input::hf_reference::HfModelReference::parse(
                &source_plan.repository,
                Some(&source_plan.revision),
            )
            .map_err(|error| AppError::Input(anyhow::anyhow!("{error}")))?;
            let downloaded = crate::input::hf_download::download_model_reference(pinned, &progress)
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
            if let Some(actual_plan) = plan_remote_native_product_bytes(
                &path,
                &selector,
                source.reference().clone(),
                source.source_sha256(),
                source_plan.requires_projector,
                args.text_only,
                args.mmproj,
            )
            .map_err(AppError::Conversion)?
            {
                if actual_plan != planned_output_bytes {
                    return Err(AppError::Conversion(anyhow::anyhow!(
                        "native conversion plan changed after source download (pretransfer={planned_output_bytes}, materialized={actual_plan})"
                    )));
                }
            }
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

    let output = match remote_source.as_ref() {
        Some(source) => match remote_conversion_lease.as_ref() {
            Some(lease) => lease.output.clone(),
            None => {
                let reference = source.reference();
                let default = crate::model_spec::default_convert_output(
                    &crate::model_spec::managed_model_root().map_err(AppError::Input)?,
                    reference.repo_id(),
                    reference.revision(),
                    &quant_name,
                )
                .map_err(AppError::Input)?;
                crate::model_spec::resolve_output_path(args.output.as_deref(), default)
                    .map_err(AppError::Input)?
            }
        },
        None => {
            let explicit = args.output.ok_or_else(|| {
                AppError::Input(anyhow::anyhow!(
                    "convert of a local source directory requires --output FILE_OR_DIR"
                ))
            })?;
            if explicit.is_dir() {
                let source_name = hf_dir
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| {
                        AppError::Input(anyhow::anyhow!(
                            "local source directory has no UTF-8 model name"
                        ))
                    })?;
                explicit.join(format!(
                    "{source_name}-hf2q-{}.gguf",
                    quant_name.to_ascii_lowercase()
                ))
            } else {
                explicit
            }
        }
    };

    if let Some(source) = remote_source.as_ref() {
        if no_clobber {
            crate::convert::recover_conversion_publication(&output)
                .map_err(|error| AppError::Conversion(anyhow::anyhow!("{error}")))?;
        }
        if remote_conversion_lease.is_none() {
            if let Some(snapshot) = verified_conversion_snapshot_matches(
                &output,
                source.reference().repo_id(),
                source.reference().revision(),
                &quant_name,
            )
            .map_err(AppError::Conversion)?
            {
                if existing_conversion_mode_complete(
                    &snapshot,
                    args.text_only,
                    args.mmproj,
                    args.mmproj_output.as_deref(),
                )
                .map_err(AppError::Conversion)?
                {
                    println!(
                        "Using existing verified hf2q conversion: {}",
                        output.display()
                    );
                    return Ok(());
                }
            }
        }
        if no_clobber {
            if let Some(conflict) = first_conversion_destination_conflict(
                &output,
                args.text_only,
                args.mmproj,
                args.mmproj_output.as_deref(),
            )
            .map_err(AppError::Conversion)?
            {
                return Err(AppError::Input(anyhow::anyhow!(
                    "no-clobber conversion destination already exists without a matching verified hf2q receipt: {}",
                    conflict.display()
                )));
            }
        }
    }

    let mode = if args.mmproj {
        ConvertMode::ProjectorOnly
    } else if args.text_only {
        ConvertMode::TextOnly
    } else {
        ConvertMode::Paired {
            projector_output: args.mmproj_output,
        }
    };
    let resolved = ConvertArgs {
        hf_dir,
        selector,
        output,
        no_clobber,
        dry_run: args.dry_run,
        imatrix: args.imatrix,
        imatrix_corpus: args.imatrix_corpus,
        imatrix_out: args.imatrix_out,
        imatrix_n_ctx: args.imatrix_n_ctx,
        mode,
        remote_source,
    };
    let _local_operation_locks = if remote_conversion_lease.is_none() && !resolved.dry_run {
        let destinations = conversion_operation_lock_destinations(&resolved.output, &resolved.mode);
        Some(
            crate::core::paired_artifact::ConversionOperationLocks::exclusive(destinations)
                .map_err(|error| AppError::Conversion(anyhow::anyhow!("{error}")))?,
        )
    } else {
        None
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
        | ConvertError::Pair { .. }
        | ConvertError::HfDownload(_) => AppError::Conversion(anyhow::anyhow!("{e}")),
    })
}

fn resolve_convert_selector(
    explicit: Option<&str>,
    operand: Option<&str>,
    operator_config: Option<&setup::OperatorConfigV2>,
) -> Result<crate::convert::QuantSelector, AppError> {
    let parse = |name: &str| {
        crate::convert::QuantSelector::from_name(name)
            .or_else(|_| crate::convert::QuantSelector::from_name(&name.to_ascii_lowercase()))
            .map_err(|error| AppError::Input(anyhow::anyhow!("{error}")))
    };
    let explicit = explicit.map(parse).transpose()?;
    let operand = operand.map(parse).transpose()?;
    if let (Some(flag), Some(suffix)) = (&explicit, &operand) {
        if flag != suffix {
            return Err(AppError::Input(anyhow::anyhow!(
                "model quant suffix {} conflicts with --quant {}",
                suffix.receipt_name(),
                flag.receipt_name()
            )));
        }
    }
    if let Some(selected) = explicit.or(operand) {
        return Ok(selected);
    }
    if let Some(config) = operator_config {
        return parse(&config.convert.quant);
    }
    let hardware = crate::core::hardware::HardwareProfiler::detect()
        .map_err(|error| AppError::Conversion(anyhow::anyhow!("detect hardware: {error}")))?;
    let quant = crate::serve::quant_select::select_quant(
        &crate::serve::quant_select::GpuInfo::from_hardware_profile(&hardware),
    )
    .map_err(AppError::Input)?;
    parse(quant.as_str())
}

#[derive(Debug)]
enum ConvertInput {
    Local(PathBuf),
    Remote {
        reference: crate::input::hf_reference::HfModelReference,
        operand_quant: Option<String>,
    },
}

fn classify_convert_input(
    positional: Option<PathBuf>,
    repo: Option<String>,
    revision: Option<&str>,
) -> Result<ConvertInput, crate::convert::ConvertError> {
    match (positional, repo) {
        (Some(_), Some(_)) => Err(crate::convert::ConvertError::RepoAndDirMutuallyExclusive),
        (None, None) => Err(crate::convert::ConvertError::MissingInput),
        (None, Some(raw)) => {
            let (reference, quant) = crate::model_spec::split_repository_quant_suffix(&raw);
            Ok(ConvertInput::Remote {
                reference: crate::input::hf_reference::HfModelReference::parse(
                    reference, revision,
                )?,
                operand_quant: quant.map(str::to_owned),
            })
        }
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
            let (reference, quant) = crate::model_spec::split_repository_quant_suffix(input);
            Ok(ConvertInput::Remote {
                reference: crate::input::hf_reference::HfModelReference::parse(
                    reference, revision,
                )?,
                operand_quant: quant.map(str::to_owned),
            })
        }
    }
}

#[cfg(test)]
fn verified_conversion_identity_matches(
    output: &std::path::Path,
    repository: &str,
    revision: &str,
    quant_name: &str,
) -> anyhow::Result<bool> {
    Ok(verified_conversion_snapshot_matches(output, repository, revision, quant_name)?.is_some())
}

struct VerifiedConversionSnapshot {
    path: PathBuf,
    receipt: crate::convert::receipt::ConversionReceipt,
    sha256: String,
    artifact: crate::core::bounded_file::StableRegularFile,
}

#[cfg(test)]
thread_local! {
    static VERIFIED_CONVERSION_HASH_PATHS: std::cell::RefCell<Vec<PathBuf>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

#[cfg(test)]
fn reset_verified_conversion_hash_paths() {
    VERIFIED_CONVERSION_HASH_PATHS.with(|paths| paths.borrow_mut().clear());
}

#[cfg(test)]
fn verified_conversion_hash_count(path: &std::path::Path) -> usize {
    VERIFIED_CONVERSION_HASH_PATHS.with(|paths| {
        paths
            .borrow()
            .iter()
            .filter(|candidate| candidate.as_path() == path)
            .count()
    })
}

fn verified_conversion_snapshot_matches(
    output: &std::path::Path,
    repository: &str,
    revision: &str,
    quant_name: &str,
) -> anyhow::Result<Option<VerifiedConversionSnapshot>> {
    let Some(snapshot) = read_verified_conversion_snapshot(output)? else {
        return Ok(None);
    };
    let receipt = &snapshot.receipt;
    Ok((receipt.source.repository_id == repository
        && receipt.source.revision.eq_ignore_ascii_case(revision)
        && receipt.quant_selector.eq_ignore_ascii_case(quant_name))
    .then_some(snapshot))
}

fn remote_conversion_output_for_noop(
    reference: &crate::input::hf_reference::HfModelReference,
    quant_name: &str,
    explicit_output: Option<&std::path::Path>,
) -> anyhow::Result<(PathBuf, String)> {
    let revision = match reference.requested_revision() {
        Some(requested)
            if requested.len() == 40 && requested.bytes().all(|byte| byte.is_ascii_hexdigit()) =>
        {
            requested.to_ascii_lowercase()
        }
        _ => crate::input::hf_download::resolve_model_reference(reference.clone())?
            .reference()
            .revision()
            .to_owned(),
    };
    let default = crate::model_spec::default_convert_output(
        &crate::model_spec::managed_model_root()?,
        reference.repo_id(),
        &revision,
        quant_name,
    )?;
    Ok((
        crate::model_spec::resolve_output_path(explicit_output, default)?,
        revision,
    ))
}

struct RemoteConversionLease {
    output: PathBuf,
    revision: String,
    _locks: crate::core::paired_artifact::ConversionOperationLocks,
}

enum RemoteConversionDecision {
    Reuse(PathBuf),
    Proceed(RemoteConversionLease),
}

#[allow(clippy::too_many_arguments)]
fn prepare_remote_conversion_lease(
    reference: &crate::input::hf_reference::HfModelReference,
    quant_name: &str,
    explicit_output: Option<&std::path::Path>,
    no_clobber: bool,
    _implicit_output: bool,
    text_only: bool,
    projector_only: bool,
    projector_override: Option<&std::path::Path>,
) -> Result<RemoteConversionDecision, AppError> {
    let (output, revision) =
        remote_conversion_output_for_noop(reference, quant_name, explicit_output)
            .map_err(AppError::Conversion)?;
    let projector = if !text_only && !projector_only {
        match projector_override
            .map(std::path::Path::to_path_buf)
            .or_else(|| default_conversion_projector_output(&output).ok())
        {
            Some(projector) => {
                validate_remote_pair_output_paths(&output, &projector).map_err(AppError::Input)?;
                Some(projector)
            }
            None => None,
        }
    } else {
        None
    };
    let locks = crate::core::paired_artifact::ConversionOperationLocks::exclusive(
        std::iter::once(output.clone()).chain(projector),
    )
    .map_err(|error| AppError::Conversion(anyhow::anyhow!("{error}")))?;
    if no_clobber {
        crate::convert::recover_conversion_publication(&output)
            .map_err(|error| AppError::Conversion(anyhow::anyhow!("{error}")))?;
    }
    let matching_snapshot =
        verified_conversion_snapshot_matches(&output, reference.repo_id(), &revision, quant_name)
            .map_err(AppError::Conversion)?;
    if let Some(snapshot) = matching_snapshot.as_ref() {
        if existing_conversion_mode_complete(
            snapshot,
            text_only,
            projector_only,
            projector_override,
        )
        .map_err(AppError::Conversion)?
        {
            return Ok(RemoteConversionDecision::Reuse(output));
        }
    }
    if no_clobber {
        if let Some(conflict) = first_conversion_destination_conflict(
            &output,
            text_only,
            projector_only,
            projector_override,
        )
        .map_err(AppError::Conversion)?
        {
            return Err(AppError::Input(anyhow::anyhow!(
                "no-clobber conversion destination already exists without a matching verified hf2q receipt: {}",
                conflict.display()
            )));
        }
    }
    Ok(RemoteConversionDecision::Proceed(RemoteConversionLease {
        output,
        revision,
        _locks: locks,
    }))
}

fn existing_conversion_mode_complete(
    text_snapshot: &VerifiedConversionSnapshot,
    text_only: bool,
    projector_only: bool,
    projector_override: Option<&std::path::Path>,
) -> anyhow::Result<bool> {
    let text_output = text_snapshot.path.as_path();
    if projector_only {
        return Ok(false);
    }
    if text_only {
        return Ok(true);
    }
    let text_gguf = mlx_native::gguf::GgufFile::from_file(text_snapshot.artifact.try_clone()?)?;
    if !crate::serve::managed_artifacts::text_gguf_requires_projector(&text_gguf) {
        return Ok(projector_override.is_none());
    }
    let text_receipt = &text_snapshot.receipt;
    let projector = match projector_override {
        Some(path) => path.to_path_buf(),
        None => default_conversion_projector_output(text_output)?,
    };
    let projector_snapshot = match read_verified_conversion_snapshot(&projector)? {
        Some(snapshot) => snapshot,
        None => return Ok(false),
    };
    let projector_receipt = &projector_snapshot.receipt;
    if projector_receipt.source.repository_id != text_receipt.source.repository_id
        || projector_receipt.source.revision != text_receipt.source.revision
        || !projector_receipt
            .quant_selector
            .eq_ignore_ascii_case("f16-mmproj")
    {
        return Ok(false);
    }
    let guard = match crate::core::paired_artifact::PairReadGuard::acquire_read_only(
        text_output,
        &projector,
    ) {
        Ok(guard) => guard,
        Err(_) => return Ok(false),
    };
    let projector_gguf =
        mlx_native::gguf::GgufFile::from_file(projector_snapshot.artifact.try_clone()?)?;
    let valid = guard
        .validate(&text_gguf, &projector_gguf, &projector_snapshot.sha256)
        .is_ok();
    Ok(valid && text_snapshot.artifact.is_stable()? && projector_snapshot.artifact.is_stable()?)
}

fn read_verified_conversion_snapshot(
    output: &std::path::Path,
) -> anyhow::Result<Option<VerifiedConversionSnapshot>> {
    use crate::convert::receipt::{
        receipt_path, ConversionReceipt, CONVERSION_RECEIPT_SCHEMA_VERSION,
    };

    let Some(receipt_bytes) = read_bounded_regular_nofollow(&receipt_path(output), 1024 * 1024)?
    else {
        return Ok(None);
    };
    let receipt: ConversionReceipt = serde_json::from_slice(&receipt_bytes)?;
    if receipt.schema_version != CONVERSION_RECEIPT_SCHEMA_VERSION
        || receipt.converter.package != "hf2q"
    {
        return Ok(None);
    }
    let Some(mut artifact) =
        crate::core::bounded_file::StableRegularFile::open_exact(output, receipt.output.size)?
    else {
        return Ok(None);
    };
    #[cfg(test)]
    VERIFIED_CONVERSION_HASH_PATHS.with(|paths| paths.borrow_mut().push(output.to_path_buf()));
    let Some(sha256) = artifact.sha256()? else {
        return Ok(None);
    };
    if !sha256.eq_ignore_ascii_case(&receipt.output.sha256) {
        return Ok(None);
    }
    Ok(Some(VerifiedConversionSnapshot {
        path: output.to_path_buf(),
        receipt,
        sha256,
        artifact,
    }))
}

fn open_regular_nofollow(path: &std::path::Path) -> anyhow::Result<Option<std::fs::File>> {
    use std::os::unix::fs::OpenOptionsExt;

    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
    {
        Ok(file) if file.metadata()?.is_file() => Ok(Some(file)),
        Ok(_) => Ok(None),
        Err(error)
            if error.kind() == std::io::ErrorKind::NotFound
                || error.raw_os_error() == Some(libc::ELOOP) =>
        {
            Ok(None)
        }
        Err(error) => Err(error.into()),
    }
}

fn read_bounded_regular_nofollow(
    path: &std::path::Path,
    max_bytes: u64,
) -> anyhow::Result<Option<Vec<u8>>> {
    use std::io::Read;

    let Some(file) = open_regular_nofollow(path)? else {
        return Ok(None);
    };
    let before = open_file_snapshot(&file)?;
    if before.len > max_bytes {
        return Ok(None);
    }
    let mut bytes = Vec::new();
    (&file).take(max_bytes + 1).read_to_end(&mut bytes)?;
    let after = open_file_snapshot(&file)?;
    if bytes.len() as u64 > max_bytes || bytes.len() as u64 != before.len || before != after {
        return Ok(None);
    }
    Ok(Some(bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OpenFileSnapshot {
    dev: u64,
    ino: u64,
    len: u64,
    mtime: i64,
    mtime_nsec: i64,
    ctime: i64,
    ctime_nsec: i64,
}

fn open_file_snapshot(file: &std::fs::File) -> anyhow::Result<OpenFileSnapshot> {
    use std::os::unix::fs::MetadataExt;

    let metadata = file.metadata()?;
    Ok(OpenFileSnapshot {
        dev: metadata.dev(),
        ino: metadata.ino(),
        len: metadata.len(),
        mtime: metadata.mtime(),
        mtime_nsec: metadata.mtime_nsec(),
        ctime: metadata.ctime(),
        ctime_nsec: metadata.ctime_nsec(),
    })
}

#[cfg(test)]
fn compute_open_file_sha256_stable(
    file: &mut std::fs::File,
) -> anyhow::Result<Option<(String, u64)>> {
    compute_open_file_sha256_stable_with_hook(file, || {})
}

#[cfg(test)]
fn compute_open_file_sha256_stable_with_hook(
    file: &mut std::fs::File,
    after_first_chunk: impl FnOnce(),
) -> anyhow::Result<Option<(String, u64)>> {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Seek};

    let before = open_file_snapshot(file)?;
    file.rewind()?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut total = 0_u64;
    let mut hook = Some(after_first_chunk);
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        total = total.saturating_add(read as u64);
        if let Some(hook) = hook.take() {
            hook();
        }
    }
    let after = open_file_snapshot(file)?;
    if before != after || total != before.len {
        return Ok(None);
    }
    Ok(Some((hex::encode(hasher.finalize()), total)))
}

fn default_conversion_projector_output(text_output: &std::path::Path) -> anyhow::Result<PathBuf> {
    let file_name = text_output
        .file_name()
        .and_then(|name| name.to_str())
        .context("text conversion output must have a UTF-8 filename")?;
    let stem = file_name
        .strip_suffix(".gguf")
        .or_else(|| file_name.strip_suffix(".GGUF"))
        .context("automatic paired conversion output must end in .gguf")?;
    Ok(text_output.with_file_name(format!("{stem}-mmproj.gguf")))
}

fn conversion_operation_lock_destinations(
    output: &std::path::Path,
    mode: &crate::convert::ConvertMode,
) -> Vec<PathBuf> {
    let mut destinations = vec![output.to_path_buf()];
    if let crate::convert::ConvertMode::Paired { projector_output } = mode {
        if let Some(projector) = projector_output
            .clone()
            .or_else(|| default_conversion_projector_output(output).ok())
        {
            destinations.push(projector);
        }
    }
    destinations
}

fn validate_remote_pair_output_paths(
    text: &std::path::Path,
    projector: &std::path::Path,
) -> anyhow::Result<()> {
    let normalized_parent = |path: &std::path::Path| {
        path.parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_path_buf()
    };
    if text == projector {
        anyhow::bail!("text and projector outputs must be different files");
    }
    if normalized_parent(text) != normalized_parent(projector) {
        anyhow::bail!("paired text and projector outputs must share one directory");
    }
    let destinations = [
        text.to_path_buf(),
        crate::convert::receipt::receipt_path(text),
        crate::convert::tensor_lineage::tensor_conversion_receipt_path(text),
        projector.to_path_buf(),
        crate::convert::receipt::receipt_path(projector),
        crate::convert::tensor_lineage::tensor_conversion_receipt_path(projector),
    ];
    for (index, path) in destinations.iter().enumerate() {
        if destinations[index + 1..].contains(path) {
            anyhow::bail!(
                "paired conversion destination collision at {}",
                path.display()
            );
        }
    }
    Ok(())
}

fn first_conversion_destination_conflict(
    output: &std::path::Path,
    text_only: bool,
    projector_only: bool,
    projector_override: Option<&std::path::Path>,
) -> anyhow::Result<Option<PathBuf>> {
    let mut artifacts = vec![output.to_path_buf()];
    if !text_only && !projector_only {
        if let Some(projector) = projector_override
            .map(std::path::Path::to_path_buf)
            .or_else(|| default_conversion_projector_output(output).ok())
        {
            artifacts.push(projector);
        }
    }
    for artifact in artifacts {
        for path in [
            artifact.clone(),
            crate::convert::receipt::receipt_path(&artifact),
            crate::convert::tensor_lineage::tensor_conversion_receipt_path(&artifact),
        ] {
            match std::fs::symlink_metadata(&path) {
                Ok(_) => return Ok(Some(path)),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
    }
    Ok(None)
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

/// Handle the `completions` subcommand.
fn cmd_completions(args: cli::CompletionsArgs) -> Result<()> {
    use std::io::Write as _;

    let mut command = cli::complete::public_completion_command();
    let mut generated = Vec::new();
    clap_complete::generate(args.shell, &mut command, "hf2q", &mut generated);
    match std::io::stdout().lock().write_all(&generated) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => return Ok(()),
        Err(error) => return Err(error).context("write generated completion script"),
    }

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
            let ConvertInput::Remote { reference, .. } = classified else {
                panic!("expected remote reference");
            };
            assert_eq!(reference.repo_id(), "Qwen/Qwen3.8-27B");
        }
    }

    #[test]
    fn convert_source_classifier_extracts_exact_quant_suffix() {
        let classified =
            classify_convert_input(Some(PathBuf::from("owner/model:Q8_0")), None, None).unwrap();
        let ConvertInput::Remote {
            reference,
            operand_quant,
        } = classified
        else {
            panic!("expected remote reference");
        };
        assert_eq!(reference.repo_id(), "owner/model");
        assert_eq!(operand_quant.as_deref(), Some("Q8_0"));
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
        assert!(matches!(classified, ConvertInput::Remote { .. }));
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
    fn convert_quant_suffix_flag_config_and_conflict_are_deterministic() {
        let config = setup::OperatorConfigV2::guide_defaults().unwrap();
        assert_eq!(
            resolve_convert_selector(None, None, Some(&config))
                .unwrap()
                .receipt_name(),
            "q4_k_m"
        );
        assert_eq!(
            resolve_convert_selector(Some("q5_k_m"), Some("Q5_K_M"), Some(&config))
                .unwrap()
                .receipt_name(),
            "q5_k_m"
        );
        assert!(resolve_convert_selector(Some("q5_k_m"), Some("Q8_0"), Some(&config)).is_err());
    }

    #[test]
    fn convert_noop_uses_exact_revision_without_network_resolution() {
        let reference = crate::input::hf_reference::HfModelReference::parse(
            "owner/model",
            Some("0123456789abcdef0123456789abcdef01234567"),
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let (output, revision) =
            remote_conversion_output_for_noop(&reference, "q6_k", Some(directory.path())).unwrap();
        assert_eq!(revision, "0123456789abcdef0123456789abcdef01234567");
        assert_eq!(output, directory.path().join("model-hf2q-q6_k.gguf"));
    }

    #[test]
    fn convert_noop_requires_a_digest_verified_hf2q_receipt() {
        use crate::convert::receipt::{
            receipt_path, ConversionReceipt, ConverterReceipt, ExcludedDsparkReceipt,
            OutputReceipt, PeakChunkBoundReceipt, SourceReceipt, CONVERSION_RECEIPT_SCHEMA_VERSION,
        };

        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("model.gguf");
        std::fs::write(&output, b"verified-native-conversion").unwrap();
        let digest = crate::core::sha256::compute_file_sha256(&output).unwrap();
        let revision = "a".repeat(40);
        let receipt = ConversionReceipt {
            schema_version: CONVERSION_RECEIPT_SCHEMA_VERSION,
            source: SourceReceipt {
                original_reference: "owner/model".into(),
                repository_id: "owner/model".into(),
                repository_type: "model".into(),
                canonical_url: "https://huggingface.co/owner/model".into(),
                revision: revision.clone(),
                filename: None,
                bundle_sha256: "b".repeat(64),
                files: Vec::new(),
            },
            converter: ConverterReceipt {
                package: "hf2q".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                git_commit: "c".repeat(40),
            },
            quant_selector: "q4_k_m".into(),
            output: OutputReceipt {
                path: output.display().to_string(),
                size: std::fs::metadata(&output).unwrap().len(),
                sha256: digest,
            },
            excluded_dspark: ExcludedDsparkReceipt {
                tensor_count: 0,
                status: "none_detected".into(),
            },
            peak_chunk_bound: PeakChunkBoundReceipt::default(),
        };
        std::fs::write(receipt_path(&output), serde_json::to_vec(&receipt).unwrap()).unwrap();
        reset_verified_conversion_hash_paths();
        assert!(
            verified_conversion_identity_matches(&output, "owner/model", &revision, "q4_k_m")
                .unwrap()
        );
        assert_eq!(verified_conversion_hash_count(&output), 1);
        reset_verified_conversion_hash_paths();
        assert!(
            !verified_conversion_identity_matches(&output, "other/model", &revision, "q4_k_m")
                .unwrap()
        );
        assert_eq!(verified_conversion_hash_count(&output), 1);
    }

    #[test]
    fn concurrent_remote_convert_waiter_reuses_winner_without_entering_conversion() {
        use crate::convert::receipt::{
            receipt_path, ConversionReceipt, ConverterReceipt, ExcludedDsparkReceipt,
            OutputReceipt, PeakChunkBoundReceipt, SourceReceipt, CONVERSION_RECEIPT_SCHEMA_VERSION,
        };
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{mpsc, Arc};

        fn publish_test_winner(output: &std::path::Path, revision: &str) {
            std::fs::write(output, b"winner-native-conversion").unwrap();
            let receipt = ConversionReceipt {
                schema_version: CONVERSION_RECEIPT_SCHEMA_VERSION,
                source: SourceReceipt {
                    original_reference: "owner/model".into(),
                    repository_id: "owner/model".into(),
                    repository_type: "model".into(),
                    canonical_url: "https://huggingface.co/owner/model".into(),
                    revision: revision.into(),
                    filename: None,
                    bundle_sha256: "b".repeat(64),
                    files: Vec::new(),
                },
                converter: ConverterReceipt {
                    package: "hf2q".into(),
                    version: env!("CARGO_PKG_VERSION").into(),
                    git_commit: "c".repeat(40),
                },
                quant_selector: "q4_k_m".into(),
                output: OutputReceipt {
                    path: output.display().to_string(),
                    size: std::fs::metadata(output).unwrap().len(),
                    sha256: crate::core::sha256::compute_file_sha256(output).unwrap(),
                },
                excluded_dspark: ExcludedDsparkReceipt {
                    tensor_count: 0,
                    status: "none_detected".into(),
                },
                peak_chunk_bound: PeakChunkBoundReceipt::default(),
            };
            std::fs::write(receipt_path(output), serde_json::to_vec(&receipt).unwrap()).unwrap();
        }

        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("model-q4_k_m.gguf");
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let reference =
            crate::input::hf_reference::HfModelReference::parse("owner/model", Some(revision))
                .unwrap();
        let conversions = Arc::new(AtomicUsize::new(0));
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel();
        let (started_tx, started_rx) = mpsc::channel();

        std::thread::scope(|scope| {
            let first_output = output.clone();
            let first_reference = reference.clone();
            let first_conversions = Arc::clone(&conversions);
            let first = scope.spawn(move || {
                let decision = prepare_remote_conversion_lease(
                    &first_reference,
                    "q4_k_m",
                    Some(&first_output),
                    true,
                    false,
                    true,
                    false,
                    None,
                )
                .unwrap();
                let RemoteConversionDecision::Proceed(lease) = decision else {
                    panic!("first caller must own conversion");
                };
                first_conversions.fetch_add(1, Ordering::SeqCst);
                acquired_tx.send(()).unwrap();
                finish_rx.recv().unwrap();
                publish_test_winner(&first_output, revision);
                drop(lease);
            });

            acquired_rx.recv().unwrap();
            let second_output = output.clone();
            let second_reference = reference.clone();
            let second_conversions = Arc::clone(&conversions);
            let second = scope.spawn(move || {
                started_tx.send(()).unwrap();
                match prepare_remote_conversion_lease(
                    &second_reference,
                    "q4_k_m",
                    Some(&second_output),
                    true,
                    false,
                    true,
                    false,
                    None,
                )
                .unwrap()
                {
                    RemoteConversionDecision::Reuse(path) => path,
                    RemoteConversionDecision::Proceed(_) => {
                        second_conversions.fetch_add(1, Ordering::SeqCst);
                        panic!("waiter must recheck and reuse the winner")
                    }
                }
            });
            started_rx.recv().unwrap();
            assert_eq!(conversions.load(Ordering::SeqCst), 1);
            finish_tx.send(()).unwrap();
            first.join().unwrap();
            assert_eq!(second.join().unwrap(), output);
        });
        assert_eq!(conversions.load(Ordering::SeqCst), 1);
    }

    #[cfg(unix)]
    #[test]
    fn conversion_noop_receipt_reads_are_bounded_nofollow_snapshots() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.json");
        std::fs::write(&target, b"{}").unwrap();
        let link = directory.path().join("receipt.json");
        symlink(&target, &link).unwrap();
        assert!(read_bounded_regular_nofollow(&link, 1024)
            .unwrap()
            .is_none());

        let oversized = directory.path().join("oversized.json");
        std::fs::write(&oversized, vec![b'x'; 1025]).unwrap();
        assert!(read_bounded_regular_nofollow(&oversized, 1024)
            .unwrap()
            .is_none());

        let artifact = directory.path().join("artifact.gguf");
        std::fs::write(&artifact, b"old-snapshot").unwrap();
        let mut opened = open_regular_nofollow(&artifact).unwrap().unwrap();
        let prior = directory.path().join("prior.gguf");
        std::fs::rename(&artifact, &prior).unwrap();
        std::fs::write(&artifact, b"replacement").unwrap();
        assert_eq!(
            compute_open_file_sha256_stable(&mut opened)
                .unwrap()
                .unwrap()
                .0,
            crate::core::sha256::compute_file_sha256(&prior).unwrap()
        );
        assert_ne!(
            compute_open_file_sha256_stable(&mut opened)
                .unwrap()
                .unwrap()
                .0,
            crate::core::sha256::compute_file_sha256(&artifact).unwrap()
        );

        let mutating = directory.path().join("mutating.gguf");
        std::fs::write(&mutating, vec![b'a'; 2 * 1024 * 1024]).unwrap();
        let mut opened = open_regular_nofollow(&mutating).unwrap().unwrap();
        let stable = compute_open_file_sha256_stable_with_hook(&mut opened, || {
            std::fs::write(&mutating, b"same-inode replacement").unwrap();
        })
        .unwrap();
        assert!(stable.is_none(), "same-inode mutation must fail closed");
    }

    #[test]
    fn convert_default_projector_output_is_a_stable_sibling() {
        assert_eq!(
            default_conversion_projector_output(std::path::Path::new("/models/model-q6_k.gguf"))
                .unwrap(),
            PathBuf::from("/models/model-q6_k-mmproj.gguf")
        );
        assert!(
            default_conversion_projector_output(std::path::Path::new("/models/model.bin")).is_err()
        );
    }

    #[test]
    fn automatic_pair_lock_plan_defers_extension_validation_until_multimodal_detection() {
        use crate::convert::ConvertMode;

        let extensionless = std::path::Path::new("/models/operator-output");
        assert_eq!(
            conversion_operation_lock_destinations(
                extensionless,
                &ConvertMode::Paired {
                    projector_output: None,
                },
            ),
            vec![extensionless.to_path_buf()]
        );

        let text = std::path::Path::new("/models/model-q4_k_m.gguf");
        assert_eq!(
            conversion_operation_lock_destinations(
                text,
                &ConvertMode::Paired {
                    projector_output: None,
                },
            ),
            vec![
                text.to_path_buf(),
                PathBuf::from("/models/model-q4_k_m-mmproj.gguf"),
            ]
        );

        let explicit = PathBuf::from("/models/operator-projector");
        assert_eq!(
            conversion_operation_lock_destinations(
                extensionless,
                &ConvertMode::Paired {
                    projector_output: Some(explicit.clone()),
                },
            ),
            vec![extensionless.to_path_buf(), explicit]
        );
    }

    #[test]
    fn extensionless_conditional_pair_conflict_checks_text_without_inventing_a_projector() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("operator-output");
        assert!(
            first_conversion_destination_conflict(&output, false, false, None)
                .unwrap()
                .is_none()
        );
        std::fs::write(&output, b"operator bytes").unwrap();
        assert_eq!(
            first_conversion_destination_conflict(&output, false, false, None).unwrap(),
            Some(output)
        );
    }

    #[test]
    fn implicit_pair_conflict_detects_projector_or_receipt_without_text() {
        let directory = tempfile::tempdir().unwrap();
        let text = directory.path().join("model-q4_k_m.gguf");
        let projector = default_conversion_projector_output(&text).unwrap();
        std::fs::write(&projector, b"operator-projector").unwrap();
        assert_eq!(
            first_conversion_destination_conflict(&text, false, false, None).unwrap(),
            Some(projector.clone())
        );
        std::fs::remove_file(&projector).unwrap();
        let receipt = crate::convert::receipt::receipt_path(&projector);
        std::fs::write(&receipt, b"operator-receipt").unwrap();
        assert_eq!(
            first_conversion_destination_conflict(&text, false, false, None).unwrap(),
            Some(receipt)
        );
        std::fs::remove_file(crate::convert::receipt::receipt_path(&projector)).unwrap();
        let tensor_receipt =
            crate::convert::tensor_lineage::tensor_conversion_receipt_path(&projector);
        std::fs::write(&tensor_receipt, b"operator-tensor-receipt").unwrap();
        assert_eq!(
            first_conversion_destination_conflict(&text, false, false, None).unwrap(),
            Some(tensor_receipt)
        );
        assert!(!text.exists());
    }

    #[test]
    fn remote_pair_paths_and_explicit_no_clobber_conflicts_fail_before_source_planning() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let text = first.path().join("model-q4_k_m.gguf");
        let cross_parent = second.path().join("model-mmproj.gguf");
        assert!(validate_remote_pair_output_paths(&text, &cross_parent).is_err());

        let projector = default_conversion_projector_output(&text).unwrap();
        std::fs::write(&projector, b"operator projector").unwrap();
        let reference = crate::input::hf_reference::HfModelReference::parse(
            "owner/model",
            Some("0123456789abcdef0123456789abcdef01234567"),
        )
        .unwrap();
        let error = prepare_remote_conversion_lease(
            &reference,
            "q4_k_m",
            Some(&text),
            true,
            false,
            false,
            false,
            None,
        )
        .err()
        .expect("explicit no-clobber projector conflict must refuse")
        .to_string();
        assert!(
            error.contains("no-clobber conversion destination"),
            "{error}"
        );
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
