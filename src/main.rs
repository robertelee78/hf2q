//! hf2q — Pure Rust CLI for converting HuggingFace models to hardware-optimized formats.
//!
//! Entry point: dispatches clap subcommands to appropriate handlers.
//! anyhow is used at this level for top-level error handling.

mod backends;
mod cli;
mod doctor;
#[allow(dead_code)]
mod inference;
mod input;
#[allow(dead_code)]
mod intelligence;
mod ir;
#[allow(dead_code)]
mod preflight;
mod progress;
#[allow(dead_code)]
mod quality;
mod quantize;
#[allow(dead_code)]
mod report;

use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::error;

use cli::{Cli, Command};

fn main() -> ExitCode {
    // Initialize tracing subscriber for structured logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    // Adjust log level based on -v flags
    if cli.verbose > 0 {
        let level = match cli.verbose {
            1 => "info",
            2 => "debug",
            _ => "trace",
        };
        // Re-initialize would fail since tracing is already init'd, so we use env filter
        // The user can also set RUST_LOG=debug for fine-grained control
        tracing::debug!("Verbosity level: {}", level);
    }

    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("{:#}", e);
            eprintln!("Error: {:#}", e);
            // Map error types to exit codes (FR39)
            // 1 = conversion error, 3 = input error
            // (exit code 2 = quality threshold exceeded — Epic 4)
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Convert(args) => cmd_convert(args),
        Command::Info(args) => cmd_info(args),
        Command::Doctor => doctor::run_doctor(),
        Command::Completions(args) => cmd_completions(args),
    }
}

/// Handle the `convert` subcommand.
fn cmd_convert(args: cli::ConvertArgs) -> Result<()> {
    use backends::mlx::MlxBackend;
    use backends::OutputBackend;
    use progress::ProgressReporter;
    use quantize::static_quant::StaticQuantizer;

    let config = cli::resolve_convert_config(&args)
        .context("Failed to resolve conversion configuration")?;

    let progress = ProgressReporter::new();

    tracing::info!(
        input = %config.input_dir.display(),
        format = %config.format,
        quant = %config.quant,
        "Starting conversion"
    );

    // Phase 1: Read model metadata and tensors
    let (metadata, mut tensor_map) =
        input::read_model(&config.input_dir, &progress)
            .context("Failed to read model")?;

    let input_size = tensor_map.total_size_bytes() as u64;

    tracing::info!(
        architecture = %metadata.architecture,
        params = metadata.param_count,
        tensors = tensor_map.len(),
        "Model loaded"
    );

    // Phase 2: Convert bf16 → f16
    let bf16_count = tensor_map
        .convert_bf16_to_f16()
        .context("bf16 to f16 conversion failed")?;
    if bf16_count > 0 {
        tracing::info!(converted = bf16_count, "Converted bf16 tensors to f16");
    }

    // Phase 3: Quantize
    let quant_method_str = config.quant.to_string();
    let quantizer = StaticQuantizer::new(&quant_method_str)
        .context("Failed to create quantizer")?;

    let bits = config.bits.unwrap_or(quantizer_default_bits(&config.quant));

    let quantized_model = quantize::quantize_model(
        &tensor_map,
        &metadata,
        &quantizer,
        bits,
        config.group_size,
        &progress,
    )
    .context("Quantization failed")?;

    // Phase 4: Validate and write output
    let backend = MlxBackend::new();

    let warnings = backend
        .validate(&quantized_model)
        .context("Output validation failed")?;

    for w in &warnings {
        tracing::warn!("{}", w.message);
    }

    let manifest = backend
        .write(
            &quantized_model,
            &config.input_dir,
            &config.output_dir,
            &progress,
        )
        .context("Failed to write output")?;

    // Phase 5: Print summary
    let model_name = config
        .input_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string());

    progress::print_summary(
        &model_name,
        &metadata.architecture,
        metadata.param_count,
        &quant_method_str,
        input_size,
        manifest.total_size_bytes,
        &manifest.output_dir,
        &progress.elapsed_display(),
    );

    Ok(())
}

/// Get the default bit width for a quantization method.
fn quantizer_default_bits(method: &cli::QuantMethod) -> u8 {
    match method {
        cli::QuantMethod::F16 => 16,
        cli::QuantMethod::Q8 => 8,
        cli::QuantMethod::Q4 => 4,
        cli::QuantMethod::Q2 => 2,
        // These are not reachable for Epic 1 (resolved in cli.rs),
        // but provide sensible defaults for future use.
        cli::QuantMethod::Auto => 4,
        cli::QuantMethod::Q4Mxfp => 4,
        cli::QuantMethod::Mixed26 => 4,
        cli::QuantMethod::Mixed36 => 4,
        cli::QuantMethod::Mixed46 => 4,
        cli::QuantMethod::DwqMixed46 => 4,
    }
}

/// Handle the `info` subcommand.
fn cmd_info(args: cli::InfoArgs) -> Result<()> {
    let input_dir = match (&args.input, &args.repo) {
        (Some(path), None) => {
            if !path.exists() {
                anyhow::bail!("Input directory does not exist: {}", path.display());
            }
            path.clone()
        }
        (None, Some(_repo)) => {
            anyhow::bail!(
                "HuggingFace Hub download (--repo) is not yet implemented. \
                 Use --input with a local model directory."
            );
        }
        (None, None) => {
            anyhow::bail!("Either --input or --repo must be specified");
        }
        (Some(_), Some(_)) => {
            anyhow::bail!("--input and --repo are mutually exclusive");
        }
    };

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

/// Handle the `completions` subcommand.
fn cmd_completions(args: cli::CompletionsArgs) -> Result<()> {
    use clap::CommandFactory;
    use clap_complete::generate;

    let mut cmd = Cli::command();
    generate(args.shell, &mut cmd, "hf2q", &mut std::io::stdout());

    Ok(())
}
