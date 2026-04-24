//! hf2q — Pure Rust CLI for quantizing HuggingFace models to GGUF and safetensors.
//!
//! Entry point: dispatches clap subcommands to appropriate handlers.
//!
//! Exit codes:
//!   0 = success
//!   1 = conversion error
//!   2 = quality threshold exceeded
//!   3 = input/validation error

#[allow(dead_code)]
mod arch;
#[allow(dead_code)]
mod backends;
mod cli;
mod debug;
mod doctor;
#[allow(dead_code)]
mod models;
#[allow(dead_code)]
mod gpu;
#[allow(dead_code)]
mod inference;
#[allow(dead_code)]
mod input;
#[allow(dead_code)]
mod intelligence;
#[allow(dead_code)]
mod ir;
#[allow(dead_code)]
mod preflight;
#[allow(dead_code)]
mod progress;
#[allow(dead_code)]
mod quality;
mod quantize;
#[allow(dead_code)]
mod report;
mod serve;

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::{error, warn};

use cli::{Cli, Command};

/// Global flag set when SIGINT (Ctrl+C) is received.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Exit codes.
const EXIT_SUCCESS: u8 = 0;
const EXIT_CONVERSION_ERROR: u8 = 1;
const EXIT_QUALITY_EXCEEDED: u8 = 2;
const EXIT_INPUT_ERROR: u8 = 3;

/// Error types for exit code classification.
#[derive(Debug)]
enum AppError {
    Input(anyhow::Error),
    Conversion(anyhow::Error),
    #[allow(dead_code)]
    QualityExceeded(anyhow::Error),
    #[allow(dead_code)]
    Interrupted,
}

impl AppError {
    fn exit_code(&self) -> u8 {
        match self {
            AppError::Input(_) => EXIT_INPUT_ERROR,
            AppError::Conversion(_) => EXIT_CONVERSION_ERROR,
            AppError::QualityExceeded(_) => EXIT_QUALITY_EXCEEDED,
            AppError::Interrupted => EXIT_CONVERSION_ERROR,
        }
    }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::Input(e) => write!(f, "{:#}", e),
            AppError::Conversion(e) => write!(f, "{:#}", e),
            AppError::QualityExceeded(e) => write!(f, "{:#}", e),
            AppError::Interrupted => write!(f, "Conversion interrupted by user"),
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
            match &app_err {
                AppError::Interrupted => {}
                _ => {
                    error!("{}", app_err);
                    eprintln!("Error: {}", app_err);
                }
            }
            ExitCode::from(exit_code)
        }
    }
}

fn run(cli: Cli) -> Result<(), AppError> {
    match cli.command {
        Command::Convert(args) => cmd_convert(args),
        Command::Info(args) => cmd_info(args).map_err(AppError::Input),
        Command::Validate(args) => cmd_validate(args),
        Command::Doctor => doctor::run_doctor().map_err(AppError::Conversion),
        Command::Completions(args) => cmd_completions(args).map_err(AppError::Input),
        Command::Generate(args) => serve::cmd_generate(args).map_err(AppError::Conversion),
        Command::Serve(args) => serve::cmd_serve(args).map_err(AppError::Conversion),
        Command::Parity(args) => serve::cmd_parity(args).map_err(AppError::Conversion),
        Command::Smoke(args) => cmd_smoke(args),
    }
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
        // Preflight / unknown-arch — non-zero exit with the documented code.
        eprintln!("{}", rendered);
        if code == arch::conformance::EXIT_UNKNOWN_ARCH {
            Err(AppError::Input(anyhow::anyhow!("{}", rendered)))
        } else {
            Err(AppError::Conversion(anyhow::anyhow!("{}", rendered)))
        }
    }
}

/// Handle the `convert` subcommand.
fn cmd_convert(args: cli::ConvertArgs) -> Result<(), AppError> {
    use backends::gguf::GgufBackend;
    use backends::safetensors_out::SafetensorsBackend;
    use backends::OutputBackend;
    use intelligence::fingerprint::ModelFingerprint;
    use intelligence::hardware::HardwareProfiler;
    use intelligence::ruvector::{QualityMetrics, RuVectorDb};
    use intelligence::AutoResolver;
    use progress::ProgressReporter;
    use quantize::static_quant::StaticQuantizer;
    use quantize::Quantizer;

    let mut config = cli::resolve_convert_config(&args)
        .context("Failed to resolve conversion configuration")
        .map_err(AppError::Input)?;

    let progress = ProgressReporter::new();

    tracing::info!(
        input = %config.input_dir.display(),
        format = %config.format,
        quant = %config.quant,
        "Starting conversion"
    );

    // Phase 0: Read model metadata for preflight
    let config_path = config.input_dir.join("config.json");
    let metadata = if config_path.exists() {
        input::config_parser::parse_config(&config_path)
            .context("Failed to parse model config")
            .map_err(AppError::Input)?
    } else {
        return Err(AppError::Input(anyhow::anyhow!(
            "No config.json found in {}. Is this a HuggingFace model directory?",
            config.input_dir.display()
        )));
    };

    // Phase 0.25: Initialize RuVector
    let mut ruvector_db = match RuVectorDb::open_default() {
        Ok(db) => db,
        Err(e) => {
            return Err(AppError::Input(anyhow::anyhow!(
                "RuVector not accessible: {}. Required to store learnings. Run `hf2q doctor` to diagnose.",
                e
            )));
        }
    };

    if let Err(e) = ruvector_db.flag_version_changes() {
        warn!("Failed to check RuVector version flags: {}", e);
    }

    // Phase 0.3: Hardware profiling and model fingerprinting
    let hardware = HardwareProfiler::detect()
        .context("Hardware profiling failed")
        .map_err(AppError::Conversion)?;

    let fingerprint = ModelFingerprint::from_metadata(&metadata)
        .context("Model fingerprinting failed")
        .map_err(AppError::Conversion)?;

    // Phase 0.4: Auto mode resolution
    let mut auto_plan: Option<intelligence::auto_quant::AutoQuantPlan> = None;
    let resolved_auto = if config.quant == cli::QuantMethod::Auto {
        {
            // Determine format hint for format-aware auto selection
            let format_hint = match config.format {
                cli::OutputFormat::Gguf => {
                    Some(intelligence::OutputFormatHint::Gguf)
                }
                cli::OutputFormat::Safetensors => {
                    Some(intelligence::OutputFormatHint::Safetensors)
                }
            };

            let resolved = AutoResolver::resolve_with_format(
                &hardware,
                &fingerprint,
                Some(&ruvector_db),
                format_hint,
            )
            .context("Auto mode resolution failed")
            .map_err(AppError::Conversion)?;

            intelligence::display_resolved_config(&resolved);

            // Generate the detailed auto-quant plan with component overrides
            auto_plan = AutoResolver::generate_plan(&hardware, &fingerprint);

            let resolved_quant = match resolved.quant_method.as_str() {
                "f16" => cli::QuantMethod::F16,
                "q8" => cli::QuantMethod::Q8,
                "q4" => cli::QuantMethod::Q4,
                "q2" => cli::QuantMethod::Q2,
                "mixed-4-6" => cli::QuantMethod::Mixed46,
                "mixed-3-6" => cli::QuantMethod::Mixed36,
                "mixed-2-6" => cli::QuantMethod::Mixed26,
                "apex" => cli::QuantMethod::Apex,
                "dwq-mixed-4-6" => cli::QuantMethod::DwqMixed46,
                "dwq-mixed-4-8" => cli::QuantMethod::DwqMixed48,
                "dwq-mixed-6-8" => cli::QuantMethod::DwqMixed68,
                "dwq-mixed-2-8" => cli::QuantMethod::DwqMixed28,
                other => {
                    warn!(
                        "Auto mode recommended '{}' which is not yet implemented. Falling back to q4.",
                        other
                    );
                    cli::QuantMethod::Q4
                }
            };

            config.quant = resolved_quant;
            if config.bits.is_none() {
                config.bits = Some(resolved.bits);
            }
            if resolved.group_size > 0 {
                config.group_size = resolved.group_size;
            }

            if args.output.is_none() {
                let model_name = config
                    .input_dir
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model".to_string());
                let suffix = config.quant.default_filename_suffix();
                config.output_dir = match config.format {
                    crate::cli::OutputFormat::Gguf => std::path::PathBuf::from(format!(
                        "{}-{}.gguf",
                        model_name, suffix
                    )),
                    _ => std::path::PathBuf::from(format!(
                        "{}-{}-{}",
                        model_name, config.format, suffix
                    )),
                };
            }

            Some(resolved)
        }
    } else {
        None
    };

    // Validate that --bits is not combined with a DWQ variant.
    // DWQ bit selection is encoded in the --quant variant name itself;
    // --bits has no effect and would silently mislead users.
    if config.bits.is_some() && config.quant.dwq_bit_pair().is_some() {
        return Err(AppError::Conversion(anyhow::anyhow!(
            "--bits is not used for DWQ; use --quant dwq-mixed-N-M to choose bit-pair variants"
        )));
    }

    // Phase 0.5: Pre-flight validation
    let preflight_report = preflight::validate(&config, &metadata)
        .map_err(|e| AppError::Input(anyhow::anyhow!("{}", e)))?;

    for warning in &preflight_report.warnings {
        warn!("{}", warning);
    }

    for pt in &preflight_report.passthrough_layers {
        warn!(
            "Layer {} (type '{}') will be passed through at f16",
            pt.layer_index, pt.layer_type
        );
    }

    if config.dry_run {
        if let Some(ref resolved) = resolved_auto {
            tracing::info!(
                "Auto mode resolved to {} (source: {}, confidence: {:.0}%)",
                resolved.quant_method,
                resolved.source,
                resolved.confidence * 100.0,
            );
        }
        print_dry_run_plan(&config, &metadata, &preflight_report);
        return Ok(());
    }

    // Set up Ctrl+C handler
    let output_dir_created_by_us = !config.output_dir.exists();
    let output_dir_for_cleanup = Arc::new(config.output_dir.clone());
    let cleanup_dir = output_dir_for_cleanup.clone();

    ctrlc::set_handler(move || {
        INTERRUPTED.store(true, Ordering::SeqCst);

        if output_dir_created_by_us {
            let path = cleanup_dir.as_ref();
            if path.exists() {
                // For .gguf file paths, remove the partial file; for directories, remove the whole dir
                let result = if path.is_file() {
                    std::fs::remove_file(path)
                } else {
                    std::fs::remove_dir_all(path)
                };
                if let Err(e) = result {
                    eprintln!(
                        "Warning: Failed to clean up partial output '{}': {}",
                        path.display(),
                        e
                    );
                } else {
                    eprintln!(
                        "Conversion interrupted. Partial output cleaned up: {}",
                        path.display()
                    );
                }
            } else {
                eprintln!("Conversion interrupted.");
            }
        } else {
            eprintln!("Conversion interrupted. Pre-existing output was not modified.");
        }
    })
    .ok();

    // Phase 1: Read model tensors
    let (_, mut tensor_map) = input::read_model(&config.input_dir, &progress)
        .context("Failed to read model")
        .map_err(AppError::Conversion)?;

    check_interrupted()?;

    // Phase 1.5: ADR-012 Decision 9 / P5 — qwen35moe expert merge.
    //
    // Must run BEFORE quantization: `merge_moe_experts_in_tensor_map`
    // consumes pre-quantization F16/BF16 bytes where shape × dtype.size
    // matches data.len(). Running post-quant on Q4_0 bytes trips the
    // expected-bytes check in `merge_expert_tensors`.
    //
    // Without this call, qwen35moe GGUFs emit N_experts × 3 × N_layers
    // separate per-expert tensors instead of 3 × N_layers merged ones,
    // and llama.cpp's loader rejects the file (silent P4→P5 wire-up
    // gap caught by ADR-012 P11 round-trip test).
    //
    // No-op for dense arches (guarded by the arch string check inside
    // the function itself).
    models::qwen35::moe::merge_moe_experts_in_tensor_map(&mut tensor_map, &metadata)
        .context("qwen35moe expert merge failed")
        .map_err(AppError::Conversion)?;

    check_interrupted()?;

    let input_size = tensor_map.total_size_bytes() as u64;

    tracing::info!(
        architecture = %metadata.architecture,
        params = metadata.param_count,
        tensors = tensor_map.len(),
        "Model loaded"
    );

    // Phase 2: Create output backend
    let backend: Box<dyn OutputBackend> = match config.format {
        cli::OutputFormat::Gguf => Box::new(GgufBackend::new()),
        cli::OutputFormat::Safetensors => Box::new(SafetensorsBackend::new()),
    };

    if !backend.requires_native_quantization() {
        let bf16_count = tensor_map
            .convert_bf16_to_f16()
            .context("bf16 to f16 conversion failed")
            .map_err(AppError::Conversion)?;
        if bf16_count > 0 {
            tracing::info!(converted = bf16_count, "Converted bf16 tensors to f16");
        }
    } else {
        tracing::info!("Preserving bf16 tensors for native quantization backend");
    }

    check_interrupted()?;

    // Phase 3: Quantize
    let quant_method_str = config.quant.to_string();
    let bits = config.bits.unwrap_or(quantizer_default_bits(&config.quant));

    let quantized_model = if backend.requires_native_quantization() {
        tracing::info!("Skipping IR quantization (native backend handles quantization)");
        None
    } else {
        Some(match config.quant {
            cli::QuantMethod::Mixed26 | cli::QuantMethod::Mixed36 | cli::QuantMethod::Mixed46 => {
                let mixed_quantizer = quantize::mixed::MixedBitQuantizer::new(
                    &quant_method_str,
                    &config.sensitive_layers,
                    config.group_size,
                )
                .context("Failed to create mixed-bit quantizer")
                .map_err(AppError::Conversion)?;

                if mixed_quantizer.requires_calibration() {
                    tracing::info!("Mixed-bit quantizer requires calibration data");
                }

                let tensor_names: Vec<String> = tensor_map.tensors.keys().cloned().collect();
                let bits_map =
                    quantize::mixed::build_per_layer_bits_map(&mixed_quantizer, &tensor_names);
                tracing::debug!(
                    layer_count = bits_map.len(),
                    "Built per-layer bit allocation map for mixed-bit quantization"
                );

                quantize::quantize_model(
                    &tensor_map,
                    &metadata,
                    &mixed_quantizer,
                    bits,
                    config.group_size,
                    &progress,
                )
                .context("Mixed-bit quantization failed")
                .map_err(AppError::Conversion)?
            }
            cli::QuantMethod::DwqMixed46
            | cli::QuantMethod::DwqMixed48
            | cli::QuantMethod::DwqMixed68
            | cli::QuantMethod::DwqMixed28 => {
                // Derive (base_bits, sensitive_bits) from the chosen variant.
                // dwq_bit_pair() is guaranteed Some(_) for all DWQ variants.
                let (base_bits, sensitive_bits) = config
                    .quant
                    .dwq_bit_pair()
                    .expect("DWQ variant must have a valid bit pair");

                // Try loading tokenizer for activation-based calibration
                let has_tokenizer = config.input_dir.join("tokenizer.json").exists();

                // Resolve the DWQ architecture for routing (ADR-012 Decision 13).
                // qwen35 / qwen35moe MUST use ActivationCapture — no weight-space fallback.
                let dwq_arch = quantize::dwq::DwqArch::from_hf_architecture(
                    &metadata.architecture,
                    metadata.is_moe(),
                );

                // ADR-012 Decision 13 no-fallback guard:
                // If the model is qwen35 or qwen35moe, we MUST have an ActivationCapture
                // implementation from the inference session.  At present (pre-ADR-013 real
                // impl), we error clearly rather than silently running weight-space DWQ.
                // When ADR-013 P12 real ActivationCapture lands, this block is replaced by
                // the real capture injection.
                if dwq_arch.requires_activation_capture() {
                    // No real ActivationCapture impl available yet — error clearly.
                    // Per ADR-012 Decision 13 / no-fallback mantra: fix the blocker upstream,
                    // do not route around it with weight-space fallback.
                    return Err(AppError::Conversion(anyhow::anyhow!(
                        "{}",
                        quantize::dwq::DwqError::NoActivationCapture
                    )));
                }

                let dwq_config = quantize::dwq::DwqConfig {
                    calibration_samples: config.calibration_samples,
                    sensitive_layers: config.sensitive_layers.clone(),
                    group_size: config.group_size,
                    base_bits,
                    sensitive_bits,
                    use_activations: has_tokenizer,
                    arch: dwq_arch,
                    ..quantize::dwq::DwqConfig::default()
                };

                let dwq_quantizer = quantize::dwq::DwqQuantizer::new(dwq_config.clone())
                    .context("Failed to create DWQ quantizer")
                    .map_err(AppError::Conversion)?;
                tracing::info!(
                    calibration = dwq_quantizer.requires_calibration(),
                    max_iterations = dwq_quantizer.config().max_iterations,
                    use_activations = dwq_config.use_activations,
                    "DWQ quantizer initialized: {}",
                    dwq_quantizer.name()
                );

                // Activation-based calibration path (non-qwen35 archs with tokenizer).
                // For qwen35/qwen35moe the guard above has already returned an error,
                // so this branch is unreachable for those archs.
                if dwq_config.use_activations {
                    tracing::info!("Tokenizer found, using activation-based DWQ calibration");
                    match quantize::dwq_activation::run_dwq_activation_calibration(
                        &tensor_map,
                        &metadata,
                        &dwq_config,
                        &config.input_dir,
                        &progress,
                    ) {
                        Ok(model) => model,
                        Err(e) => {
                            // ADR-012 Decision 13: the weight-space fallback below is ONLY
                            // reachable for non-qwen35 archs (guard above enforces this).
                            // For qwen35/qwen35moe we never reach here.
                            warn!(
                                "Activation-based DWQ failed ({}), falling back to weight-space calibration",
                                e
                            );
                            quantize::dwq::run_dwq_calibration(
                                &tensor_map,
                                &metadata,
                                &dwq_config,
                                &progress,
                            )
                            .context("DWQ weight-space calibration failed")
                            .map_err(AppError::Conversion)?
                        }
                    }
                } else {
                    tracing::info!("No tokenizer found, using weight-space DWQ calibration");
                    quantize::dwq::run_dwq_calibration(
                        &tensor_map,
                        &metadata,
                        &dwq_config,
                        &progress,
                    )
                    .context("DWQ calibration failed")
                    .map_err(AppError::Conversion)?
                }
            }
            cli::QuantMethod::Apex => {
                let apex_config = quantize::apex::ApexConfig {
                    calibration_tokens: config.calibration_samples as usize,
                    target_bpw: args.target_bpw,
                    min_type: "Q3_K_S".to_string(),
                    max_type: "Q6_K".to_string(),
                };

                quantize::apex::run_apex_quantization(
                    &tensor_map,
                    &metadata,
                    &config.input_dir,
                    &apex_config,
                    &progress,
                )
                .context("Apex quantization failed")
                .map_err(AppError::Conversion)?
            }
            _ => {
                let quantizer = StaticQuantizer::new(&quant_method_str)
                    .context("Failed to create quantizer")
                    .map_err(AppError::Conversion)?;

                quantize::quantize_model(
                    &tensor_map,
                    &metadata,
                    &quantizer,
                    bits,
                    config.group_size,
                    &progress,
                )
                .context("Quantization failed")
                .map_err(AppError::Conversion)?
            }
        })
    };

    check_interrupted()?;

    // Phase 4: Write output
    let mut bit_overrides: Option<HashMap<String, u8>> = if let Some(ref plan) = auto_plan {
        let mut map = HashMap::new();
        for co in &plan.component_overrides {
            for tensor_name in tensor_map.tensors.keys() {
                if tensor_name.contains(&co.pattern) {
                    map.insert(tensor_name.clone(), co.bits);
                }
            }
        }
        if map.is_empty() {
            None
        } else {
            tracing::info!(
                overrides = map.len(),
                "Built per-tensor bit override map from auto-quant plan"
            );
            Some(map)
        }
    } else {
        None
    };

    if !config.sensitive_layers.is_empty() && bit_overrides.is_none() {
        let mut map = HashMap::new();
        for range in &config.sensitive_layers {
            for layer_idx in range.clone() {
                let prefix = format!(".layers.{}.", layer_idx);
                for tensor_name in tensor_map.tensors.keys() {
                    if tensor_name.contains(&prefix) {
                        let elevated = (bits + 2).min(8);
                        map.insert(tensor_name.clone(), elevated);
                    }
                }
            }
        }
        if !map.is_empty() {
            tracing::info!(
                overrides = map.len(),
                "Built per-tensor bit override map from --sensitive-layers"
            );
            bit_overrides = Some(map);
        }
    }

    // Phase 4.5: Quality measurement (before write, while quantized_model is available)
    let quality_report = if config.skip_quality {
        tracing::info!("Quality measurement skipped (--skip-quality)");
        quality::QualityReport::empty()
    } else if let Some(ref qm) = quantized_model {
        // Build a dequantized TensorMap from the quantized output for comparison
        let quantized_tensor_map = quality::dequantize_to_tensor_map(qm);
        match quality::measure_quality(
            &tensor_map,
            &quantized_tensor_map,
            &metadata,
            &config.input_dir,
            &progress,
        ) {
            Ok(qr) => {
                quality::print_quality_summary(&qr);
                qr
            }
            Err(e) => {
                warn!("Quality measurement failed: {}. Continuing.", e);
                quality::QualityReport::empty()
            }
        }
    } else {
        // Native backend: quality measurement not available inline
        tracing::info!("Quality measurement not available for native quantization backend");
        tracing::info!("Use 'hf2q validate' after conversion to measure quality");
        quality::QualityReport::empty()
    };

    check_interrupted()?;

    // Phase 4.6: Write output
    let manifest = if backend.requires_native_quantization() {
        let native_bits = if let Some(ref plan) = auto_plan {
            plan.base_bits
        } else {
            config.bits.unwrap_or(quantizer_default_bits(&config.quant))
        };

        let quantize_config = backends::QuantizeConfig {
            bits: native_bits,
            group_size: config.group_size,
            bit_overrides: bit_overrides.as_ref(),
        };
        backend
            .quantize_and_write(
                &tensor_map,
                &metadata,
                &quantize_config,
                &config.input_dir,
                &config.output_dir,
                &progress,
            )
            .context("Native quantization and write failed")
            .map_err(AppError::Conversion)?
    } else {
        let quantized = quantized_model.expect(
            "BUG: quantized_model should be Some for non-native backends",
        );

        let warnings = backend
            .validate(&quantized)
            .context("Output validation failed")
            .map_err(AppError::Conversion)?;

        for w in &warnings {
            tracing::warn!("{}", w.message);
        }

        backend
            .write(
                &quantized,
                &config.input_dir,
                &config.output_dir,
                &progress,
            )
            .context("Failed to write output")
            .map_err(AppError::Conversion)?
    };

    // Phase 4.7: Copy sidecar files into output directory (Decision 15).
    // manifest.output_dir is the canonical parent directory that contains
    // the written .gguf or safetensors files.
    copy_sidecars(&config.input_dir, std::path::Path::new(&manifest.output_dir));

    // Phase 4.8: ADR-012 P10 / Decision 18 — emit pure-Rust mmproj when
    // --emit-vision-tower is set AND the HF config has a vision_config.
    // Silent skip for Gemma4 (no vision_config) and Qwen3.6-35B-A3B MoE
    // (publisher dropped vision_config). Only Qwen3.6-27B dense emits.
    //
    // Sovereignty: when the P10 emitter succeeds, delete the backend's
    // auto-emitted legacy `<text>-mmproj.gguf` so there's ONE
    // authoritative mmproj per convert invocation — the pure-Rust P10
    // output produced from clip.cpp/clip-model.h spec.
    if args.emit_vision_tower {
        let output_parent = std::path::Path::new(&manifest.output_dir);
        match models::vit::convert_vision_tower(&config.input_dir, output_parent) {
            Ok(Some(path)) => {
                tracing::info!(
                    "ADR-012 P10: mmproj emitted at {}",
                    path.display()
                );
                // Remove the backend's legacy auto-emitted mmproj if
                // present. The legacy path fires whenever the model
                // has `vision_tower.*` tensors (see src/backends/gguf.rs
                // line 449 `has_vision` check). With P10 in play the
                // legacy file is now redundant duplication; keeping
                // both invites loader confusion.
                for entry in std::fs::read_dir(output_parent)
                    .into_iter()
                    .flatten()
                    .flatten()
                {
                    let p = entry.path();
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        if name.ends_with("-mmproj.gguf")
                            && !name.starts_with("mmproj-")
                            && p != path
                        {
                            if let Err(e) = std::fs::remove_file(&p) {
                                warn!(
                                    "failed to remove legacy mmproj {:?}: {}",
                                    p, e
                                );
                            } else {
                                tracing::info!(
                                    "ADR-012 P10 sovereignty: removed legacy backend mmproj {:?}",
                                    p
                                );
                            }
                        }
                    }
                }
            }
            Ok(None) => {
                tracing::info!(
                    "--emit-vision-tower requested but {} has no vision_config — skipping",
                    config.input_dir.display()
                );
            }
            Err(e) => {
                warn!("vision-tower emission failed: {}", e);
            }
        }
    }

    // Phase 5.1: Regression detection against previous baseline
    let quality_gate_result = if quality_report.has_metrics() {
        let thresholds = quality::QualityThresholds::default();
        let gate = quality::regression::build_quality_gate(&quality_report, &thresholds);

        // Query RuVector for previous best result for this model
        if let Ok(Some(baseline_resolved)) = ruvector_db.query_best_config(&hardware, &fingerprint)
        {
            let baseline_report =
                extract_baseline_from_reasoning(&baseline_resolved.reasoning);
            if baseline_report.has_metrics() {
                let regression_warnings =
                    quality::regression::detect_regression(&quality_report, &baseline_report, 0.1);
                if !regression_warnings.is_empty() {
                    eprintln!();
                    for w in &regression_warnings {
                        let severity_icon = match w.severity {
                            quality::regression::RegressionSeverity::Error => "ERROR",
                            quality::regression::RegressionSeverity::Warning => "WARN ",
                            quality::regression::RegressionSeverity::Info => "INFO ",
                        };
                        eprintln!(
                            "  [{}] Regression on {}: {:.6} -> {:.6} ({:+.1}%)",
                            severity_icon,
                            w.metric,
                            w.baseline_value,
                            w.current_value,
                            w.degradation_pct
                        );
                    }
                    eprintln!();
                }
                quality::regression::print_comparison(&baseline_report, &quality_report);
            }
        }

        // Enforce quality gate if --quality-gate flag is set
        if config.quality_gate && !gate.passed {
            eprintln!(
                "{}",
                console::style("QUALITY GATE FAILED:").red().bold()
            );
            for v in &gate.violations {
                eprintln!("  - {}", v);
            }
            eprintln!();

            // Generate JSON report before exiting, if requested
            if config.json_report {
                let json_report = report::ReportBuilder::new(
                    config.input_dir.display().to_string(),
                    config.output_dir.display().to_string(),
                    metadata.clone(),
                    quant_method_str.clone(),
                    bits,
                    config.group_size,
                )
                .with_input_size(input_size)
                .with_quality(quality_report.clone())
                .with_quality_gate(gate.clone())
                .with_hardware(report::HardwareSummary {
                    chip_model: hardware.chip_model.clone(),
                    total_memory_bytes: hardware.total_memory_bytes,
                    available_memory_bytes: hardware.available_memory_bytes,
                    total_cores: hardware.total_cores,
                })
                .with_timing(progress.elapsed().as_secs_f64(), None)
                .build();

                if config.yes {
                    let _ = report::write_to_stdout(&json_report);
                } else {
                    let report_path = config.output_dir.join("report.json");
                    let _ = report::write_to_file(&json_report, &report_path);
                }
            }

            return Err(AppError::QualityExceeded(anyhow::anyhow!(
                "{} quality threshold(s) exceeded",
                gate.violations.len()
            )));
        }

        Some(gate)
    } else {
        None
    };

    // Phase 5.5: Store conversion result in RuVector
    if let Err(e) = ruvector_db.store_conversion(
        &hardware,
        &fingerprint,
        &quant_method_str,
        bits,
        config.group_size,
        QualityMetrics {
            kl_divergence: quality_report.kl_divergence,
            perplexity_delta: quality_report.perplexity_delta,
            cosine_similarity: quality_report.cosine_sim_average,
        },
    ) {
        warn!("Failed to store conversion result in RuVector: {}", e);
    }

    // Phase 6: JSON report generation
    if config.json_report {
        let mut report_builder = report::ReportBuilder::new(
            config.input_dir.display().to_string(),
            config.output_dir.display().to_string(),
            metadata.clone(),
            quant_method_str.clone(),
            bits,
            config.group_size,
        )
        .with_input_size(input_size)
        .with_manifest(manifest.clone())
        .with_quality(quality_report.clone())
        .with_hardware(report::HardwareSummary {
            chip_model: hardware.chip_model.clone(),
            total_memory_bytes: hardware.total_memory_bytes,
            available_memory_bytes: hardware.available_memory_bytes,
            total_cores: hardware.total_cores,
        })
        .with_timing(progress.elapsed().as_secs_f64(), None);

        if let Some(gate) = quality_gate_result {
            report_builder = report_builder.with_quality_gate(gate);
        }

        let json_report = report_builder.build();

        if config.yes {
            report::write_to_stdout(&json_report)
                .context("Failed to write JSON report to stdout")
                .map_err(AppError::Conversion)?;
        } else {
            let report_path = config.output_dir.join("report.json");
            report::write_to_file(&json_report, &report_path)
                .context("Failed to write JSON report")
                .map_err(AppError::Conversion)?;
            tracing::info!(path = %report_path.display(), "JSON report written");
        }
    }

    // Phase 7: Print summary
    if !(config.json_report && config.yes) {
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
    }

    Ok(())
}

/// Handle the `validate` subcommand.
fn cmd_validate(args: cli::ValidateArgs) -> Result<(), AppError> {
    use progress::ProgressReporter;

    // Validate directories exist
    if !args.original.exists() {
        return Err(AppError::Input(anyhow::anyhow!(
            "Original model directory does not exist: {}",
            args.original.display()
        )));
    }
    if !args.quantized.exists() {
        return Err(AppError::Input(anyhow::anyhow!(
            "Quantized model directory does not exist: {}",
            args.quantized.display()
        )));
    }

    let progress = ProgressReporter::new();

    // Read metadata from original model
    let config_path = args.original.join("config.json");
    if !config_path.exists() {
        return Err(AppError::Input(anyhow::anyhow!(
            "No config.json found in {}. Is this a HuggingFace model directory?",
            args.original.display()
        )));
    }

    let metadata = input::config_parser::parse_config(&config_path)
        .context("Failed to parse model config")
        .map_err(AppError::Input)?;

    // Read original model tensors
    let (_, original_tensors) = input::read_model(&args.original, &progress)
        .context("Failed to read original model")
        .map_err(AppError::Conversion)?;

    // Read quantized model tensors
    let (_, quantized_tensors) = input::read_model(&args.quantized, &progress)
        .context("Failed to read quantized model")
        .map_err(AppError::Conversion)?;

    // Run quality measurement
    let quality_report = quality::measure_quality(
        &original_tensors,
        &quantized_tensors,
        &metadata,
        &args.original,
        &progress,
    )
    .map_err(|e| AppError::Conversion(anyhow::anyhow!("{}", e)))?;

    // Update RuVector with quality metrics from validation
    if quality_report.has_metrics() {
        if let Ok(mut ruvector_db) = intelligence::ruvector::RuVectorDb::open_default() {
            let hw = intelligence::hardware::HardwareProfiler::detect()
                .ok()
                .unwrap_or_else(|| intelligence::hardware::HardwareProfile {
                    chip_model: "unknown".to_string(),
                    total_memory_bytes: 0,
                    available_memory_bytes: 0,
                    performance_cores: 0,
                    efficiency_cores: 0,
                    total_cores: 0,
                    memory_bandwidth_gbs: 0.0,
                });
            if let Ok(fp) = intelligence::fingerprint::ModelFingerprint::from_metadata(&metadata) {
                let quant_method = detect_quant_method_from_path(&args.quantized);
                let metrics = intelligence::ruvector::QualityMetrics {
                    kl_divergence: quality_report.kl_divergence,
                    perplexity_delta: quality_report.perplexity_delta,
                    cosine_similarity: quality_report.cosine_sim_average,
                };
                if let Err(e) = ruvector_db.update_quality(
                    &hw.stable_id(),
                    &fp.stable_id(),
                    &quant_method,
                    metrics,
                ) {
                    warn!("Could not update RuVector quality metrics: {}", e);
                }
            }
        }
    }

    // Check thresholds
    let thresholds = quality::QualityThresholds {
        max_kl_divergence: args.max_kl,
        max_perplexity_delta: args.max_ppl_delta,
        min_cosine_similarity: args.min_cosine,
    };

    let violations = quality::check_thresholds(&quality_report, &thresholds);

    // Build quality gate for CI output
    let quality_gate = quality::regression::build_quality_gate(&quality_report, &thresholds);

    // Output
    if args.json {
        let json_output = serde_json::json!({
            "report": quality_report,
            "thresholds": {
                "max_kl_divergence": thresholds.max_kl_divergence,
                "max_perplexity_delta": thresholds.max_perplexity_delta,
                "min_cosine_similarity": thresholds.min_cosine_similarity,
            },
            "violations": violations,
            "pass": violations.is_empty(),
            "quality_gate": quality_gate,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json_output)
                .unwrap_or_else(|_| "{}".to_string())
        );
    } else {
        quality::print_quality_summary(&quality_report);

        if violations.is_empty() {
            eprintln!(
                "{}",
                console::style("PASS: All quality thresholds met.").green().bold()
            );
        } else {
            eprintln!(
                "{}",
                console::style("FAIL: Quality thresholds exceeded:").red().bold()
            );
            for v in &violations {
                eprintln!("  - {}", v);
            }
        }
        eprintln!();
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(AppError::QualityExceeded(anyhow::anyhow!(
            "{} quality threshold(s) exceeded",
            violations.len()
        )))
    }
}

/// Check if the process has been interrupted by SIGINT.
fn check_interrupted() -> Result<(), AppError> {
    if INTERRUPTED.load(Ordering::SeqCst) {
        Err(AppError::Interrupted)
    } else {
        Ok(())
    }
}

/// Print the dry-run conversion plan.
fn print_dry_run_plan(
    config: &cli::ConvertConfig,
    metadata: &ir::ModelMetadata,
    preflight: &preflight::PreflightReport,
) {
    use console::style;

    let bits = config.bits.unwrap_or(quantizer_default_bits(&config.quant));
    let estimated_memory = preflight.estimated_output_bytes + (metadata.param_count * 2);

    eprintln!();
    eprintln!("{}", style("Dry Run -- Conversion Plan").bold().cyan());
    eprintln!("{}", style("============================").bold().cyan());
    eprintln!();
    eprintln!("  Input:        {}", config.input_dir.display());
    eprintln!("  Output:       {}", config.output_dir.display());
    eprintln!("  Format:       {}", config.format);
    eprintln!("  Quantization: {}", config.quant);
    eprintln!("  Bit width:    {}", bits);
    eprintln!("  Group size:   {}", config.group_size);
    eprintln!();
    eprintln!("  Model:        {}", metadata.architecture);
    eprintln!("  Parameters:   {}", progress::format_param_count(metadata.param_count));
    eprintln!("  Layers:       {}", metadata.num_layers);
    if metadata.is_moe() {
        eprintln!(
            "  MoE:          {} experts, top-k={}",
            metadata.num_experts.unwrap_or(0),
            metadata.top_k_experts.unwrap_or(0)
        );
    }
    eprintln!();
    eprintln!(
        "  Est. output:  {}",
        progress::format_bytes(preflight.estimated_output_bytes)
    );
    eprintln!(
        "  Est. memory:  {}",
        progress::format_bytes(estimated_memory)
    );
    if preflight.available_disk_bytes > 0 {
        eprintln!(
            "  Avail. disk:  {}",
            progress::format_bytes(preflight.available_disk_bytes)
        );
    }

    if !preflight.passthrough_layers.is_empty() {
        eprintln!();
        eprintln!(
            "  {} layers will be passed through at f16:",
            preflight.passthrough_layers.len()
        );
        for pt in &preflight.passthrough_layers {
            eprintln!("    - Layer {} (type '{}')", pt.layer_index, pt.layer_type);
        }
    }

    for warning in &preflight.warnings {
        eprintln!("  Warning: {}", style(warning).yellow());
    }

    eprintln!();
    eprintln!("{}", style("No files were written (dry run).").dim());
    eprintln!();
}

/// Sidecar files to copy from the HF source directory into the output
/// directory alongside the produced `.gguf` (Decision 15).
///
/// Files are copied byte-identically. If a file is missing from the
/// source directory it is silently skipped — not all models ship all
/// sidecars.
const SIDECAR_FILES: &[&str] = &[
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
];

/// Copy sidecar files from `src_dir` to `dst_dir` (Decision 15).
///
/// - Byte-identical content; no transformation.
/// - Missing source files are silently skipped.
/// - The destination directory must already exist.
/// - Errors on individual copies are logged as warnings but do not fail
///   the overall convert operation.
fn copy_sidecars(src_dir: &std::path::Path, dst_dir: &std::path::Path) {
    for filename in SIDECAR_FILES {
        let src = src_dir.join(filename);
        if !src.exists() {
            // Silent skip — not an error.
            continue;
        }
        let dst = dst_dir.join(filename);
        match std::fs::copy(&src, &dst) {
            Ok(bytes) => {
                tracing::info!(
                    file = %filename,
                    bytes = bytes,
                    "Sidecar file copied"
                );
            }
            Err(e) => {
                warn!(
                    "Failed to copy sidecar '{}' to output: {}",
                    filename, e
                );
            }
        }
    }
}

/// Try to detect the quantization method from an output directory path.
fn detect_quant_method_from_path(path: &std::path::Path) -> String {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    // Long forms before short to avoid substring collisions (e.g. "dwq-mixed-4-6" before "mixed-4-6").
    let known = [
        "dwq-mixed-4-8", "dwq-mixed-6-8", "dwq-mixed-2-8", "dwq-mixed-4-6",
        "dwq48", "dwq68", "dwq28", "dwq46",
        "mixed-2-6", "mixed-3-6", "mixed-4-6",
        "q2", "q4", "q8", "f16",
        "apex",
    ];
    for method in &known {
        if name.contains(method) {
            return method.to_string();
        }
    }
    "unknown".to_string()
}

/// Get the default bit width for a quantization method.
fn quantizer_default_bits(method: &cli::QuantMethod) -> u8 {
    match method {
        cli::QuantMethod::F16 => 16,
        cli::QuantMethod::Q8 => 8,
        cli::QuantMethod::Q4 => 4,
        cli::QuantMethod::Q2 => 2,
        cli::QuantMethod::Auto => 4,
        cli::QuantMethod::Mixed26 => 4,
        cli::QuantMethod::Mixed36 => 4,
        cli::QuantMethod::Mixed46 => 4,
        cli::QuantMethod::DwqMixed46 => 4,
        cli::QuantMethod::DwqMixed48 => 4,
        cli::QuantMethod::DwqMixed68 => 6,
        cli::QuantMethod::DwqMixed28 => 2,
        cli::QuantMethod::Apex => 4,
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

/// Extract a baseline QualityReport from a RuVector resolved config reasoning string.
///
/// The reasoning string has the format:
///   "Based on stored conversion from <timestamp> (KL divergence: <val>, perplexity delta: <val>)"
fn extract_baseline_from_reasoning(reasoning: &str) -> quality::QualityReport {
    let mut report = quality::QualityReport::empty();

    // Parse KL divergence
    if let Some(start) = reasoning.find("KL divergence: ") {
        let rest = &reasoning[start + "KL divergence: ".len()..];
        if let Some(end) = rest.find(',').or_else(|| rest.find(')')) {
            if let Ok(val) = rest[..end].trim().parse::<f64>() {
                report.kl_divergence = Some(val);
            }
        }
    }

    // Parse perplexity delta
    if let Some(start) = reasoning.find("perplexity delta: ") {
        let rest = &reasoning[start + "perplexity delta: ".len()..];
        if let Some(end) = rest.find(')').or_else(|| rest.find(',')) {
            if let Ok(val) = rest[..end].trim().parse::<f64>() {
                report.perplexity_delta = Some(val);
            }
        }
    }

    report
}
