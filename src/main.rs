//! hf2q — Pure Rust CLI for converting HuggingFace models to hardware-optimized formats.
//!
//! Entry point: dispatches clap subcommands to appropriate handlers.
//! anyhow is used at this level for top-level error handling.
//!
//! Exit codes (FR39):
//!   0 = success
//!   1 = conversion error
//!   2 = quality threshold exceeded
//!   3 = input/validation error

mod backends;
mod cli;
mod doctor;
#[cfg(feature = "mlx-native")]
mod hub;
#[allow(dead_code)]
mod inference;
mod input;
mod intelligence;
mod ir;
mod preflight;
mod progress;
#[allow(dead_code)]
mod quality;
mod quantize;
#[allow(dead_code)]
mod report;
#[cfg(feature = "serve")]
mod serve;
#[cfg(feature = "mlx-native")]
#[allow(dead_code)]
mod tokenizer;

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

/// Exit codes per FR39.
const EXIT_SUCCESS: u8 = 0;
const EXIT_CONVERSION_ERROR: u8 = 1;
const EXIT_QUALITY_EXCEEDED: u8 = 2;
const EXIT_INPUT_ERROR: u8 = 3;

/// Error types for exit code classification.
#[derive(Debug)]
enum AppError {
    /// Input or validation error (exit code 3)
    Input(anyhow::Error),
    /// Conversion error (exit code 1)
    Conversion(anyhow::Error),
    /// Quality threshold exceeded (exit code 2)
    #[allow(dead_code)]
    QualityExceeded(anyhow::Error),
    /// Interrupted by signal (exit code 1)
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
    // Initialize tracing subscriber for structured logging.
    // All progress/warnings go to stderr so stdout stays clean for --json-report.
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
        tracing::debug!("Verbosity level: {}", level);
    }

    match run(cli) {
        Ok(()) => ExitCode::from(EXIT_SUCCESS),
        Err(app_err) => {
            let exit_code = app_err.exit_code();
            match &app_err {
                AppError::Interrupted => {
                    // Message already printed by the signal handler
                }
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
        Command::Doctor => doctor::run_doctor().map_err(AppError::Conversion),
        Command::Completions(args) => cmd_completions(args).map_err(AppError::Input),
        #[cfg(feature = "mlx-native")]
        Command::Infer(args) => cmd_infer(args),
        #[cfg(feature = "serve")]
        Command::Serve(args) => cmd_serve(args),
    }
}

/// Handle the `infer` subcommand (requires mlx-native feature).
#[cfg(feature = "mlx-native")]
fn cmd_infer(args: cli::InferArgs) -> Result<(), AppError> {
    use std::io::Write;

    use console::style;
    use inference::engine::{EngineConfig, InferenceEngine};
    use inference::sampler::SamplerConfig;

    // Validate sampling parameters
    if args.temperature < 0.0 {
        return Err(AppError::Input(anyhow::anyhow!(
            "Temperature must be >= 0.0, got {}",
            args.temperature
        )));
    }
    if args.top_p <= 0.0 || args.top_p > 1.0 {
        return Err(AppError::Input(anyhow::anyhow!(
            "Top-p must be in (0.0, 1.0], got {}",
            args.top_p
        )));
    }
    if args.repetition_penalty < 1.0 {
        return Err(AppError::Input(anyhow::anyhow!(
            "Repetition penalty must be >= 1.0, got {}",
            args.repetition_penalty
        )));
    }

    // Require a prompt for CLI mode
    let prompt = args.prompt.as_deref().ok_or_else(|| {
        AppError::Input(anyhow::anyhow!(
            "No prompt provided. Use --prompt \"your text here\""
        ))
    })?;

    // Step 1: Resolve model path (local or Hub download)
    let model_dir = hub::download::resolve_model_path(&args.model)
        .map_err(|e| AppError::Input(anyhow::anyhow!("{}", e)))?;

    // Step 2: Detect architecture (for memory estimation display)
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Err(AppError::Input(anyhow::anyhow!(
            "No config.json found in {}. Is this a valid model directory?",
            model_dir.display()
        )));
    }

    let model_config = inference::models::registry::detect_architecture(&config_path)
        .map_err(|e| AppError::Input(anyhow::anyhow!("{}", e)))?;

    eprintln!(
        "{} {}",
        style("Architecture:").bold(),
        model_config.architecture_str
    );

    // Step 3: Memory estimation and fail-fast check
    let max_tokens = args.max_tokens.unwrap_or(512) as usize;
    let max_seq_len = (max_tokens as u64).max(4096);
    let mem_estimate = inference::memory_estimate::estimate_memory(
        &model_config,
        &model_dir,
        max_seq_len,
    )
    .map_err(|e| AppError::Input(anyhow::anyhow!("{}", e)))?;

    eprintln!(
        "{} {:.1} GB (weights: {:.1} GB, KV cache: {:.1} GB)",
        style("Memory estimate:").bold(),
        mem_estimate.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        mem_estimate.weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        mem_estimate.kv_cache_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    eprintln!(
        "{} {:.1} GB ({:.1}% usage)",
        style("Available memory:").bold(),
        mem_estimate.available_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        mem_estimate.usage_percent(),
    );

    inference::memory_estimate::check_memory(&mem_estimate)
        .map_err(|e| AppError::Input(anyhow::anyhow!("{}", e)))?;

    // Step 4: Build engine config
    let sampler_config = SamplerConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k as usize,
        repetition_penalty: args.repetition_penalty,
    };

    let engine_config = EngineConfig {
        max_tokens,
        sampler: sampler_config,
        stop_sequences: Vec::new(),
    };

    // Step 5: Create inference engine (loads model, tokenizer, chat template)
    eprintln!("{}", style("Loading model...").dim());
    let chat_template_override = args.chat_template.as_deref();

    let mut engine = InferenceEngine::new(&model_dir, engine_config, chat_template_override)
        .map_err(|e| {
            // Map chat template not found to a user-friendly message
            match &e {
                inference::engine::EngineError::ChatTemplate(
                    crate::tokenizer::chat_template::ChatTemplateError::NotFound,
                ) => AppError::Input(anyhow::anyhow!(
                    "No chat template found. Provide one with --chat-template"
                )),
                _ => AppError::Conversion(anyhow::anyhow!("{}", e)),
            }
        })?;

    eprintln!("{}", style("Generating...").dim());
    eprintln!();

    // Step 6: Generate with streaming output
    let stdout = std::io::stdout();
    let (text, stats) = engine
        .generate(prompt, |token_text| {
            // Stream each token to stdout as it is generated
            let mut out = stdout.lock();
            let _ = out.write_all(token_text.as_bytes());
            let _ = out.flush();
            true // continue generating
        })
        .map_err(|e| AppError::Conversion(anyhow::anyhow!("{}", e)))?;

    // Ensure we end on a newline
    if !text.ends_with('\n') {
        println!();
    }

    // Step 7: Print generation statistics
    eprintln!();
    eprintln!(
        "{} {} prompt tokens, {} generated tokens",
        style("Stats:").bold(),
        stats.prompt_tokens,
        stats.generated_tokens,
    );
    eprintln!(
        "{} prefill {:.1} tok/s, decode {:.1} tok/s, total {:.2}s",
        style("Speed:").bold(),
        stats.prefill_tokens_per_sec(),
        stats.decode_tokens_per_sec(),
        stats.total_time_secs,
    );

    Ok(())
}

/// Handle the `serve` subcommand (requires serve feature).
#[cfg(feature = "serve")]
fn cmd_serve(args: cli::ServeArgs) -> Result<(), AppError> {
    use console::style;
    use inference::engine::{EngineConfig, InferenceEngine};
    use inference::sampler::SamplerConfig;

    // Step 1: Resolve model path
    let model_dir = hub::download::resolve_model_path(&args.model)
        .map_err(|e| AppError::Input(anyhow::anyhow!("{}", e)))?;

    // Step 2: Validate model directory
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Err(AppError::Input(anyhow::anyhow!(
            "No config.json found in {}. Is this a valid model directory?",
            model_dir.display()
        )));
    }

    let model_config = inference::models::registry::detect_architecture(&config_path)
        .map_err(|e| AppError::Input(anyhow::anyhow!("{}", e)))?;

    eprintln!(
        "{} {}",
        style("Architecture:").bold(),
        model_config.architecture_str
    );

    // Step 3: Memory estimation
    let max_seq_len = 4096u64; // Default for serve mode
    let mem_estimate =
        inference::memory_estimate::estimate_memory(&model_config, &model_dir, max_seq_len)
            .map_err(|e| AppError::Input(anyhow::anyhow!("{}", e)))?;

    eprintln!(
        "{} {:.1} GB (weights: {:.1} GB, KV cache: {:.1} GB)",
        style("Memory estimate:").bold(),
        mem_estimate.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        mem_estimate.weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        mem_estimate.kv_cache_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    inference::memory_estimate::check_memory(&mem_estimate)
        .map_err(|e| AppError::Input(anyhow::anyhow!("{}", e)))?;

    // Step 4: Build engine config (defaults for serve mode)
    let engine_config = EngineConfig {
        max_tokens: 4096,
        sampler: SamplerConfig::default(),
        stop_sequences: Vec::new(),
    };

    // Step 5: Load model
    eprintln!("{}", style("Loading model...").dim());
    let chat_template_override = args.chat_template.as_deref();

    let prompt_cache_config = inference::prompt_cache::PromptCacheConfig {
        enabled: !args.no_prompt_cache,
        ..Default::default()
    };

    let engine = InferenceEngine::new_with_prompt_cache(
        &model_dir,
        engine_config,
        chat_template_override,
        prompt_cache_config,
    )
    .map_err(|e| match &e {
        inference::engine::EngineError::ChatTemplate(
            crate::tokenizer::chat_template::ChatTemplateError::NotFound,
        ) => AppError::Input(anyhow::anyhow!(
            "No chat template found. Provide one with --chat-template"
        )),
        _ => AppError::Conversion(anyhow::anyhow!("{}", e)),
    })?;

    let engine_max_seq_len = engine.max_seq_len();

    // Step 6: Load tokenizer and chat template separately for the server
    // (so validation can happen without locking the engine Mutex)
    let tokenizer = tokenizer::HfTokenizer::from_dir(&model_dir)
        .map_err(|e| AppError::Conversion(anyhow::anyhow!("{}", e)))?;

    let chat_template = if let Some(override_path) = chat_template_override {
        tokenizer::chat_template::ChatTemplate::from_file(override_path)
            .map_err(|e| AppError::Conversion(anyhow::anyhow!("{}", e)))?
    } else {
        tokenizer::chat_template::ChatTemplate::from_model_dir(&model_dir)
            .map_err(|e| AppError::Conversion(anyhow::anyhow!("{}", e)))?
    };

    // Derive model name from directory basename
    let model_name = model_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown-model".to_string());

    eprintln!(
        "{} {}",
        style("Prompt cache:").bold(),
        if args.no_prompt_cache { "disabled" } else { "enabled" }
    );

    eprintln!(
        "{} {}",
        style("Model:").bold(),
        model_name
    );

    // Step 7: Start the tokio runtime and run the server
    let serve_config = serve::ServeConfig {
        host: args.host,
        port: args.port,
        queue_depth: args.queue_depth,
        embedding_concurrency: args.embedding_concurrency,
    };

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| AppError::Conversion(anyhow::anyhow!("Failed to create tokio runtime: {}", e)))?;

    // Pass the raw model config for vision encoder initialization
    let raw_config: Option<serde_json::Value> = std::fs::read_to_string(&config_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());

    rt.block_on(async {
        serve::run(
            engine,
            tokenizer,
            chat_template,
            model_name,
            engine_max_seq_len,
            serve_config,
            raw_config.as_ref(),
        )
        .await
        .map_err(|e| AppError::Conversion(e))
    })
}

/// Handle the `convert` subcommand.
fn cmd_convert(args: cli::ConvertArgs) -> Result<(), AppError> {
    use backends::coreml::CoremlBackend;
    use backends::mlx::MlxBackend;
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

    // Phase 0.25: Initialize RuVector (required for all conversions)
    let mut ruvector_db = match RuVectorDb::open_default() {
        Ok(db) => db,
        Err(e) => {
            return Err(AppError::Input(anyhow::anyhow!(
                "RuVector not accessible: {}. Required to store learnings. Run `hf2q doctor` to diagnose.",
                e
            )));
        }
    };

    // Flag any records from previous hf2q versions for re-calibration (FR29)
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

    // Phase 0.4: Auto mode resolution (if --quant auto or default)
    //
    // For MLX output, we use the intelligent auto_quant algorithm which produces
    // per-tensor bit overrides. For other formats, we use the existing AutoResolver
    // which returns a simple ResolvedConfig.
    use intelligence::auto_quant::{resolve_auto_plan, AutoQuantConstraints, QualityPreference};

    let mut auto_plan: Option<intelligence::auto_quant::AutoQuantPlan> = None;
    let resolved_auto = if config.quant == cli::QuantMethod::Auto {
        if config.format == cli::OutputFormat::Mlx {
            // MLX path: use intelligent auto_quant for per-tensor bit allocation
            let constraints = AutoQuantConstraints {
                min_tok_per_sec: 80.0,
                quality_preference: QualityPreference::Balanced,
                ..AutoQuantConstraints::default()
            };

            let plan = resolve_auto_plan(&hardware, &fingerprint, &constraints)
                .context("Auto-quant resolution failed")
                .map_err(AppError::Conversion)?;

            // Log the auto plan
            tracing::info!(
                base_bits = plan.base_bits,
                estimated_tok_s = plan.estimated_tok_per_sec,
                overrides = plan.component_overrides.len(),
                "Auto-quant resolved: {}",
                plan.reasoning
            );

            // Map the plan's quant_method to our CLI enum
            let resolved_quant = match plan.quant_method.as_str() {
                "f16" => cli::QuantMethod::F16,
                "q8" => cli::QuantMethod::Q8,
                "q4" => cli::QuantMethod::Q4,
                "q2" => cli::QuantMethod::Q2,
                "mixed-4-6" => cli::QuantMethod::Mixed46,
                "mixed-3-6" => cli::QuantMethod::Mixed36,
                "mixed-2-6" => cli::QuantMethod::Mixed26,
                other => {
                    warn!(
                        "Auto-quant recommended '{}' which is not a known method. Using q4 as base.",
                        other
                    );
                    cli::QuantMethod::Q4
                }
            };

            config.quant = resolved_quant;
            config.bits = Some(plan.base_bits);
            if plan.group_size > 0 {
                config.group_size = plan.group_size;
            }

            // Build a ResolvedConfig for display and dry-run compatibility
            let resolved = intelligence::ResolvedConfig {
                quant_method: plan.quant_method.clone(),
                bits: plan.base_bits,
                group_size: plan.group_size,
                confidence: plan.confidence,
                source: intelligence::ResolvedSource::Heuristic,
                reasoning: plan.reasoning.clone(),
                hardware: hardware.clone(),
                fingerprint: fingerprint.clone(),
            };
            intelligence::display_resolved_config(&resolved);

            // Update output dir if it was auto-generated with "auto" in the name
            if args.output.is_none() {
                let model_name = config
                    .input_dir
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model".to_string());
                config.output_dir = std::path::PathBuf::from(format!(
                    "{}-{}-{}",
                    model_name, config.format, config.quant
                ));
            }

            auto_plan = Some(plan);
            Some(resolved)
        } else {
            // Non-MLX path: use existing AutoResolver (works for CoreML/GGUF)
            let resolved = AutoResolver::resolve(&hardware, &fingerprint, Some(&ruvector_db))
                .context("Auto mode resolution failed")
                .map_err(AppError::Conversion)?;

            // Display the resolved config
            intelligence::display_resolved_config(&resolved);

            // Apply the resolved config to the ConvertConfig
            let resolved_quant = match resolved.quant_method.as_str() {
                "f16" => cli::QuantMethod::F16,
                "q8" => cli::QuantMethod::Q8,
                "q4" => cli::QuantMethod::Q4,
                "q2" => cli::QuantMethod::Q2,
                "mixed-4-6" => cli::QuantMethod::Mixed46,
                "mixed-3-6" => cli::QuantMethod::Mixed36,
                "mixed-2-6" => cli::QuantMethod::Mixed26,
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

            // Update output dir if it was auto-generated with "auto" in the name
            if args.output.is_none() {
                let model_name = config
                    .input_dir
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model".to_string());
                config.output_dir = std::path::PathBuf::from(format!(
                    "{}-{}-{}",
                    model_name, config.format, config.quant
                ));
            }

            Some(resolved)
        }
    } else {
        None
    };

    // Phase 0.5: Pre-flight validation (Story 2.1)
    let preflight_report = preflight::validate(&config, &metadata)
        .map_err(|e| AppError::Input(anyhow::anyhow!("{}", e)))?;

    // Log preflight warnings
    for warning in &preflight_report.warnings {
        warn!("{}", warning);
    }

    // Log passthrough layers
    for pt in &preflight_report.passthrough_layers {
        warn!(
            "Layer {} (type '{}') will be passed through at f16",
            pt.layer_index, pt.layer_type
        );
    }

    // If dry run, print plan and exit
    if config.dry_run {
        // Also show auto mode resolution in dry run
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

    // Set up Ctrl+C handler (Story 2.4)
    // Track whether the output directory was created by us (not pre-existing)
    let output_dir_created_by_us = !config.output_dir.exists();
    let output_dir_for_cleanup = Arc::new(config.output_dir.clone());
    let cleanup_dir = output_dir_for_cleanup.clone();

    ctrlc::set_handler(move || {
        INTERRUPTED.store(true, Ordering::SeqCst);

        // Only clean up directories we created
        if output_dir_created_by_us {
            let dir = cleanup_dir.as_ref();
            if dir.exists() {
                if let Err(e) = std::fs::remove_dir_all(dir) {
                    eprintln!(
                        "Warning: Failed to clean up partial output directory '{}': {}",
                        dir.display(),
                        e
                    );
                } else {
                    eprintln!(
                        "Conversion interrupted. Partial output cleaned up: {}",
                        dir.display()
                    );
                }
            } else {
                eprintln!("Conversion interrupted.");
            }
        } else {
            eprintln!("Conversion interrupted. Pre-existing output directory was not modified.");
        }

        // The ctrlc crate with "termination" feature will terminate the process after this handler
    })
    .ok(); // Ignore error if handler was already set (e.g., in tests)

    // Phase 1: Read model tensors
    let (_, mut tensor_map) = input::read_model(&config.input_dir, &progress)
        .context("Failed to read model")
        .map_err(AppError::Conversion)?;

    check_interrupted()?;

    let input_size = tensor_map.total_size_bytes() as u64;

    tracing::info!(
        architecture = %metadata.architecture,
        params = metadata.param_count,
        tensors = tensor_map.len(),
        "Model loaded"
    );

    // Phase 2: Convert bf16 -> f16 (skip for MLX native path which preserves bf16)
    let backend: Box<dyn OutputBackend> = match config.format {
        cli::OutputFormat::Mlx => Box::new(MlxBackend::new()),
        cli::OutputFormat::Coreml => Box::new(CoremlBackend::new()),
        other => {
            return Err(AppError::Conversion(anyhow::anyhow!(
                "Output format '{}' is not yet implemented",
                other
            )));
        }
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

    // Phase 3: Quantize (skipped for native backends like MLX)
    let quant_method_str = config.quant.to_string();
    let bits = config.bits.unwrap_or(quantizer_default_bits(&config.quant));

    let quantized_model = if backend.requires_native_quantization() {
        // Skip IR quantization for native backends — they quantize from original tensors
        // directly in Phase 4. Running IR quantization here would be wasted work since
        // the native path re-quantizes from the original tensor_map.
        tracing::info!("Skipping IR quantization (native backend handles quantization)");
        None
    } else {
        Some(match config.quant {
            cli::QuantMethod::Mixed26 | cli::QuantMethod::Mixed36 | cli::QuantMethod::Mixed46 => {
                // Mixed-bit quantization (Story 5.1)
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

                // Build per-layer bit allocation map for logging
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
            cli::QuantMethod::DwqMixed46 => {
                // DWQ calibration (Story 5.2) — requires InferenceRunner
                let mut runner = inference::create_runner();
                if !runner.is_available() {
                    return Err(AppError::Conversion(anyhow::anyhow!(
                        "DWQ quantization requires the mlx-backend feature. \
                         Rebuild with: cargo build --features mlx-backend"
                    )));
                }

                let dwq_config = quantize::dwq::DwqConfig {
                    calibration_samples: config.calibration_samples,
                    sensitive_layers: config.sensitive_layers.clone(),
                    group_size: config.group_size,
                    base_bits: 4,
                    sensitive_bits: 6,
                    ..quantize::dwq::DwqConfig::default()
                };

                // Validate DWQ config by constructing the quantizer
                let dwq_quantizer = quantize::dwq::DwqQuantizer::new(dwq_config.clone())
                    .context("Failed to create DWQ quantizer")
                    .map_err(AppError::Conversion)?;
                tracing::info!(
                    calibration = dwq_quantizer.requires_calibration(),
                    max_iterations = dwq_quantizer.config().max_iterations,
                    "DWQ quantizer initialized: {}",
                    dwq_quantizer.name()
                );

                quantize::dwq::run_dwq_calibration(
                    runner.as_mut(),
                    &tensor_map,
                    &metadata,
                    &dwq_config,
                    &progress,
                )
                .context("DWQ calibration failed")
                .map_err(AppError::Conversion)?
            }
            _ => {
                // Static quantization (f16, q8, q4, q2)
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

    // Phase 4: Write output (backend was created in Phase 2)

    // Build per-tensor bit overrides from auto_quant plan or sensitive_layers
    let mut bit_overrides: Option<HashMap<String, u8>> = if let Some(ref plan) = auto_plan {
        // Auto-quant plan: convert component_overrides to tensor-name -> bits map
        let mut map = HashMap::new();
        for co in &plan.component_overrides {
            // Match each pattern against actual tensor names
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

    // Wire --sensitive-layers for non-auto modes: elevate sensitive layers by +2 bits
    if !config.sensitive_layers.is_empty() && bit_overrides.is_none() {
        let mut map = HashMap::new();
        for range in &config.sensitive_layers {
            for layer_idx in range.clone() {
                let prefix = format!(".layers.{}.", layer_idx);
                for tensor_name in tensor_map.tensors.keys() {
                    if tensor_name.contains(&prefix) {
                        // Sensitive layers get +2 bits, capped at 8
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

    let manifest = if backend.requires_native_quantization() {
        // Native path (MLX): backend quantizes directly from original tensors.
        // Pass auto_quant base_bits (or the configured bits) plus per-tensor overrides.
        let native_bits = if let Some(ref plan) = auto_plan {
            plan.base_bits
        } else {
            config.bits.unwrap_or(quantizer_default_bits(&config.quant))
        };

        backend
            .quantize_and_write(
                &tensor_map,
                &metadata,
                native_bits,
                config.group_size,
                bit_overrides.as_ref(),
                &config.input_dir,
                &config.output_dir,
                &progress,
            )
            .context("Native quantization and write failed")
            .map_err(AppError::Conversion)?
    } else {
        // Standard path: validate pre-quantized model and write
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

    check_interrupted()?;

    // Phase 5: Quality measurement (Story 4.2, 4.3)
    let quality_report = if config.skip_quality {
        tracing::info!("Quality measurement skipped (--skip-quality)");
        quality::QualityReport::empty()
    } else {
        let mut runner = inference::create_runner();
        if runner.is_available() {
            match quality::measure_quality(
                runner.as_mut(),
                &tensor_map,
                &tensor_map, // Use original tensors for comparison
                &metadata,
                &progress,
            ) {
                Ok(qr) => {
                    quality::print_quality_summary(&qr);
                    qr
                }
                Err(e) => {
                    warn!("Quality measurement failed: {}. Continuing without quality metrics.", e);
                    quality::QualityReport::empty()
                }
            }
        } else {
            tracing::info!(
                "Quality measurement skipped: mlx-backend feature not enabled. \
                 Rebuild with --features mlx-backend to enable quality measurement."
            );
            quality::QualityReport::empty()
        }
    };

    // Phase 5.5: Store conversion result in RuVector (Story 7.2)
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
        // Storage failure is not fatal — warn but continue
        warn!("Failed to store conversion result in RuVector: {}", e);
    }

    // Phase 6: JSON report generation (Story 4.4)
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
        .with_manifest(manifest.clone())
        .with_quality(quality_report.clone())
        .with_hardware(report::HardwareSummary {
            chip_model: hardware.chip_model.clone(),
            total_memory_bytes: hardware.total_memory_bytes,
            available_memory_bytes: hardware.available_memory_bytes,
            total_cores: hardware.total_cores,
        })
        .with_timing(progress.elapsed().as_secs_f64(), None)
        .build();

        if config.yes {
            // Write to stdout when --json-report + --yes
            report::write_to_stdout(&json_report)
                .context("Failed to write JSON report to stdout")
                .map_err(AppError::Conversion)?;
        } else {
            // Write to file in output directory
            let report_path = config.output_dir.join("report.json");
            report::write_to_file(&json_report, &report_path)
                .context("Failed to write JSON report")
                .map_err(AppError::Conversion)?;
            tracing::info!(path = %report_path.display(), "JSON report written");
        }
    }

    // Phase 7: Print summary (skip when writing JSON to stdout to keep stdout clean)
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

    // CoreML-specific validation in dry-run
    if config.format == cli::OutputFormat::Coreml {
        match backends::coreml::validate_metadata_for_coreml(metadata) {
            Ok(coreml_warnings) => {
                for w in &coreml_warnings {
                    eprintln!("  Warning: {}", style(&w.message).yellow());
                }
            }
            Err(e) => {
                eprintln!("  {}: {}", style("ERROR").bold().red(), e);
            }
        }
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

/// Get the default bit width for a quantization method.
fn quantizer_default_bits(method: &cli::QuantMethod) -> u8 {
    match method {
        cli::QuantMethod::F16 => 16,
        cli::QuantMethod::Q8 => 8,
        cli::QuantMethod::Q4 => 4,
        cli::QuantMethod::Q2 => 2,
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

    // Use eprintln for the header (keeps stdout clean for piping)
    // But for info command, stdout is the expected output
    println!();
    println!(
        "{}",
        console::style("Model Information").bold().green()
    );
    println!("{}", input::config_parser::format_info(&metadata));
    println!();

    Ok(())
}

/// Resolve the input directory for the info subcommand, supporting --repo via HF download.
fn resolve_info_input(args: &cli::InfoArgs) -> Result<PathBuf> {
    match (&args.input, &args.repo) {
        (Some(path), None) => {
            if !path.exists() {
                anyhow::bail!("Input directory does not exist: {}", path.display());
            }
            Ok(path.clone())
        }
        (None, Some(repo_id)) => {
            // Story 3.2: hf2q info --repo org/model for remote inspection
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
