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
///
/// 0/1/2/3 are the long-standing convert/validate codes.  4–6 are added by
/// ADR-012 P8's `hf2q smoke` subcommand for distinct preflight failure modes
/// (per Decision 16 acceptance: each preflight failure surfaces a unique
/// non-zero code so a CI runner can tell "missing token" from "missing disk").
const EXIT_SUCCESS: u8 = 0;
const EXIT_CONVERSION_ERROR: u8 = 1;
const EXIT_QUALITY_EXCEEDED: u8 = 2;
const EXIT_INPUT_ERROR: u8 = 3;
// Smoke preflight (mirrored from `arch::smoke` so callers don't need to
// import the inner constants).
const EXIT_SMOKE_LLAMA_CLI_MISSING: u8 = 4;
const EXIT_SMOKE_BINARY_MISSING: u8 = 5;
const EXIT_SMOKE_REPO_MISSING: u8 = 6;

/// Error types for exit code classification.
#[derive(Debug)]
enum AppError {
    Input(anyhow::Error),
    Conversion(anyhow::Error),
    #[allow(dead_code)]
    QualityExceeded(anyhow::Error),
    #[allow(dead_code)]
    Interrupted,
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
            AppError::QualityExceeded(_) => EXIT_QUALITY_EXCEEDED,
            AppError::Interrupted => EXIT_CONVERSION_ERROR,
            AppError::Smoke { code, .. } => *code,
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
    //
    // `metadata` is `mut` so Phase 1.8 (vocab-pad de-pad, ADR-012) can
    // overwrite `metadata.vocab_size` with the de-padded count derived from
    // tokenizer.json — keeps the downstream metadata emission consistent
    // with the truncated embedding tensors.
    let config_path = config.input_dir.join("config.json");
    let mut metadata = if config_path.exists() {
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

    // Phase 1: Read model tensors (ADR-014 P0/P1 lazy entry).
    //
    // `read_tensors_lazy` returns a LazyTensorMap whose entries each
    // carry an Arc<Mmap> + closure. Tensor bytes do not materialise
    // until each Phase 1.x transform / quantize consumer invokes
    // `LazyTensor::materialize`. Phase 1.4 + 1.42 are pure-metadata
    // key rewrites and run on the LazyTensorMap directly; Phase 1.45+
    // are not yet lifted (later P1 iters), so the lazy map is bridged
    // back to an eager TensorMap via `materialize_all` immediately
    // before Phase 1.45 — see the bridge call below.
    let mut lazy_map = input::safetensors::read_tensors_lazy(&config.input_dir, &progress)
        .context("Failed to index model tensors lazily")
        .map_err(AppError::Conversion)?;

    check_interrupted()?;

    // Phase 1.4: ADR-012 P9b real-model finding — strip `language_model.`
    // prefix from all tensor names before the qwen35 transforms run.
    //
    // Real Qwen3.6 safetensors use the nested
    //   `model.language_model.layers.N.*`
    // namespace (mirroring the `Qwen3_5{,Moe}ForConditionalGeneration` HF
    // class hierarchy). The qwen35 transforms downstream were written for
    // the simpler `model.layers.N.*` namespace:
    //   - `apply_qwen35_linear_attn_transforms_in_tensor_map` uses
    //     `.contains()` substring matches (incidentally robust to either
    //     prefix).
    //   - `merge_moe_experts_in_tensor_map` uses exact-format matches
    //     `model.layers.{N}.mlp.experts.{E}.{proj}.weight` (silently
    //     no-ops on `language_model.` variants — caught by the qwen35moe
    //     real-model run on jenerallee78/Qwen3.6-35B-A3B-...).
    //
    // Normalizing globally here matches what `hf_name_to_gguf` does at
    // GGUF-write time (line ~1599: `replace("language_model.", "")`), so
    // the input-side and output-side namespaces stay consistent for the
    // entire pipeline.
    //
    // Vision tensors (`model.vision_tower.*`, `model.embed_vision.*`) and
    // root-level non-language_model paths are unaffected (the strip is a
    // simple `.replace` and only fires on tensors that actually contain
    // the literal `language_model.` segment).
    //
    // ADR-014 P1: lifted onto LazyTensorMap. Pure metadata operation; no
    // tensor bytes touched.
    {
        let renames: Vec<(String, String)> = lazy_map
            .names()
            .filter(|n| n.contains("language_model."))
            .map(|n| (n.clone(), n.replace("language_model.", "")))
            .collect();
        if !renames.is_empty() {
            tracing::info!(
                count = renames.len(),
                "qwen35 P9b real-model finding: stripping `language_model.` prefix from {} tensors",
                renames.len()
            );
            for (old_name, new_name) in renames {
                if let Some(old_lazy) = lazy_map.remove(&old_name) {
                    let new_meta = crate::ir::lazy::LazyMeta::new(
                        new_name.clone(),
                        old_lazy.shape().to_vec(),
                        old_lazy.dtype(),
                    );
                    let new_name_for_closure = new_name.clone();
                    let renamed_lazy = old_lazy.map_with_meta(new_meta, move |mut tref| {
                        tref.name = new_name_for_closure.clone();
                        Ok(tref)
                    });
                    lazy_map.insert(renamed_lazy);
                }
            }
        }
    }

    // Phase 1.42: ADR-013 P14 — MTP tensor handling.
    //
    // Emit Qwen3.5 MTP tensors by default. HF names use `mtp.*`; the helper
    // rewrites them into `model.layers.{num_hidden_layers + mtp_idx}.*` so the
    // GGUF mapper emits:
    //   - inner transformer-block tensors at `blk.{N}.*`
    //   - wrapper NextN tensors at `blk.{N}.nextn.*`
    //
    // Temporary escape hatch:
    //   HF2Q_QWEN35_DROP_MTP=1
    //
    // REMOVE when llama.cpp adds qwen35 MTP loading OR by 2026-Q4 if upstream
    // lags. The default must remain emit-by-default so hf2q-produced GGUFs
    // carry the complete MTP block for native speculative decoding.
    //
    // ADR-014 P1: lifted onto LazyTensorMap via `_lazy` variant. Pure
    // metadata key rewrite; no tensor bytes touched.
    {
        let qwen35_family = models::qwen35::is_qwen35_family_architecture(
            &metadata.architecture,
            &metadata.model_type,
        );
        let drop_mtp = qwen35_family
            && std::env::var("HF2Q_QWEN35_DROP_MTP")
                .map(|v| v == "1")
                .unwrap_or(false);

        if drop_mtp {
            let mtp_keys: Vec<String> = lazy_map
                .names()
                .filter(|n| n.starts_with("mtp."))
                .cloned()
                .collect();
            if !mtp_keys.is_empty() {
                tracing::warn!(
                    dropped = mtp_keys.len(),
                    "HF2Q_QWEN35_DROP_MTP=1: dropping {} mtp.* tensors; escape hatch expires when llama.cpp adds qwen35 MTP loading OR by 2026-Q4 if upstream lags",
                    mtp_keys.len()
                );
                for key in mtp_keys {
                    lazy_map.remove(&key);
                }
            }
        } else {
            let renamed = models::qwen35::rename_mtp_tensors_to_layer_form_lazy(
                &mut lazy_map,
                &metadata,
            )
            .context("qwen35 MTP tensor rename failed")
            .map_err(AppError::Conversion)?;
            if renamed > 0 {
                tracing::info!(
                    renamed,
                    "Phase 1.42: emitting {} qwen35 MTP tensors by default",
                    renamed
                );
            }
        }
    }

    // ADR-014 P1 bridge: subsequent Phase 1.x transforms (1.45 / 1.5 /
    // 1.6 / 1.7 / 1.8) are still eager — they take `&mut TensorMap` and
    // operate on materialised bytes. Bridge through `materialize_all`
    // here. Each Phase 1.x lifted in subsequent P1 iters moves this
    // bridge later in the pipeline; once the streaming quantize loop
    // (P2) lands the bridge becomes test-only.
    let mut tensor_map = lazy_map
        .materialize_all()
        .context("Failed to materialise lazy tensor map (P0→P1 bridge)")
        .map_err(AppError::Conversion)?;

    // Phase 1.45: ADR-012 P9b real-model finding (jenerallee78 abliterated apex):
    // some qwen35moe checkpoints ship a FUSED gate+up tensor named
    //   model.layers.N.mlp.experts.gate_up_proj   (shape [N_exp, 2*moe_inter, hidden])
    // plus
    //   model.layers.N.mlp.experts.down_proj      (shape [N_exp, hidden, moe_inter])
    // — both WITHOUT the `.weight` suffix llama.cpp / hf_name_to_gguf
    // expect. Split the fused gate_up along axis 1 into separate
    // gate_proj/up_proj and add `.weight` to all three. After this phase
    // the tensors are in the canonical pre-merged form expected by
    // `merge_moe_experts_in_tensor_map`'s pre-merged-skip path
    // (commit 9045d04).
    models::qwen35::moe::split_and_rename_fused_gate_up_in_tensor_map(
        &mut tensor_map,
        &metadata,
    )
    .context("qwen35moe fused gate_up split + .weight rename failed")
    .map_err(AppError::Conversion)?;

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

    // Phase 1.6: ADR-012 P3 Decision 6 — RMS norm +1 bias (Qwen3.5 family
    // `gamma + 1` convention, py:4794-4795). Baked into the stored weights
    // at convert time. Silent wire-up gap caught by spec-source audit
    // 2026-04-24 — P3 shipped the per-tensor transform but never called
    // it; before this fix, converted Qwen3.5 GGUFs shipped RMS norm
    // weights WITHOUT the +1 bias, producing silent logit skew at
    // inference. No-op for non-Qwen3.5 arches (Gemma4, LLaMA).
    models::qwen35::apply_rms_norm_plus_one_in_tensor_map(&mut tensor_map, &metadata)
        .context("qwen35 RMS norm +1 bias application failed")
        .map_err(AppError::Conversion)?;

    // Phase 1.7: ADR-012 P2/P3 Decision 5 + R2 — linear-attention transforms.
    // A_log negation, conv1d squeeze, in_proj_qkvz split into qkv + z,
    // and the 6-case V-head grouped→tiled reorder per py:5367-5424.
    //
    // 4th silent wire-up gap caught by the 2026-04-24 audit. ADR-012 R2
    // was the named "plausible-looking nonsense" failure mode — pre-fix
    // every Qwen3.5 GGUF we produced shipped HF-grouped V-head layout
    // instead of ggml-tiled. Enabled here after the fixture shapes in
    // tests/convert_qwen35{,moe}_integration.rs were corrected to match
    // real Qwen3.5 spec ([linear_num_value_heads, hidden] for A_log /
    // in_proj_a etc., not [num_heads, hidden]).
    //
    // Spec sources: convert_hf_to_gguf.py:5367-5424 (6-case dispatcher)
    // + Qwen3NextModel.modify_tensors:4786-4830 (A_log negation, dt_bias
    // rename, conv1d squeeze). No-op for non-Qwen3.5 arches.
    models::qwen35::apply_qwen35_linear_attn_transforms_in_tensor_map(
        &mut tensor_map,
        &metadata,
    )
    .context("qwen35 linear-attention transforms failed")
    .map_err(AppError::Conversion)?;

    // Phase 1.8: ADR-012 vocab-pad de-pad — surfaced 2026-04-25 by
    // `hf2q smoke --arch qwen35 --quant q4_0 --local-dir <hf-cache-snap>`.
    //
    // HF safetensors pads `model.embed_tokens.weight` and `lm_head.weight` to
    // an aligned vocab dimension (Qwen3.6-27B emits 248320 rows even though
    // the tokenizer owns 248044 unique ids).  llama.cpp's loader compares the
    // tensor shape against the emitted `tokenizer.ggml.tokens` array length
    // and rejects a mismatch with
    //   tensor 'token_embd.weight' has wrong shape; expected H, T, got H, P
    // where T is de-padded and P is padded.
    //
    // This phase truncates both embedding tensors to the de-padded count
    // computed from `tokenizer.json` (the same ground truth that drives
    // tokenizer.ggml.tokens emission downstream).  No-op when the model has
    // no `tokenizer.json`, when no padding is present, or when the embed
    // tensor doesn't match the original metadata.vocab_size.  Updates
    // `metadata.vocab_size` to the de-padded value so downstream metadata
    // emission is consistent.
    if let Some(true_vocab) =
        input::detect_padded_vocab(&config.input_dir, &metadata)
            .context("vocab-pad detection failed")
            .map_err(AppError::Conversion)?
    {
        let truncated = input::truncate_padded_vocab(&mut tensor_map, &mut metadata, true_vocab);
        if truncated > 0 {
            tracing::info!(
                truncated_tensors = truncated,
                de_padded_vocab = true_vocab,
                "ADR-012 Phase 1.8: vocab-pad fix applied"
            );
        }
    }

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

                if dwq_arch.requires_activation_capture() {
                    // ADR-012 P9b.2/3b — qwen35 / qwen35moe two-pass conversion.
                    //
                    // Decision 13 mandates ActivationCapture for these arches; no
                    // weight-space fallback. Steps:
                    //   1. Tokenizer.json is mandatory (capture tokenization
                    //      contract) — error if missing.
                    //   2. Emit an intermediate F16 GGUF from the in-memory
                    //      tensor_map via `emit_gguf_from_tensor_map` (P9b.1).
                    //      `tensor_map` is held in scope for the subsequent
                    //      DWQ pass — no double-read from safetensors (P9b.4).
                    //   3. Construct a `RealActivationCapture` from the
                    //      intermediate GGUF + tokenizer (P9b.3b). This loads
                    //      the model into RAM; the file is no longer needed
                    //      after the constructor returns.
                    //   4. Run activation-aware DWQ via
                    //      `run_dwq_activation_calibration` (P9b.3a — the real
                    //      impl that consumes a `&mut dyn ActivationCapture`).
                    //   5. The `tempfile::TempDir` is dropped at the end of
                    //      this scope, removing the intermediate GGUF — RAII
                    //      cleanup (P9b.5).
                    let tokenizer_json = config.input_dir.join("tokenizer.json");
                    if !tokenizer_json.exists() {
                        return Err(AppError::Conversion(anyhow::anyhow!(
                            "ADR-012 P9b: qwen35/qwen35moe DWQ requires \
                             tokenizer.json in {}; not found. No weight-space \
                             fallback per Decision 13.",
                            config.input_dir.display()
                        )));
                    }

                    let intermediate_dir = tempfile::tempdir()
                        .context("ADR-012 P9b: failed to create tempdir for intermediate GGUF")
                        .map_err(AppError::Conversion)?;
                    let intermediate_path =
                        intermediate_dir.path().join("intermediate-f16.gguf");

                    tracing::info!(
                        path = %intermediate_path.display(),
                        "ADR-012 P9b: emitting intermediate F16 GGUF for activation capture"
                    );
                    backends::gguf::emit_gguf_from_tensor_map(
                        &tensor_map,
                        &metadata,
                        &config.input_dir,
                        &intermediate_path,
                        &progress,
                    )
                    .context("ADR-012 P9b: intermediate F16 GGUF emission failed")
                    .map_err(AppError::Conversion)?;

                    // ADR-012 P9b apex MoE OOM mitigation (cost-model
                    // candidate #3, 2026-04-25): swap tensor_map out for an
                    // empty placeholder across the capture call so the
                    // F32-expanded Qwen35Model (~128 GB MoE params) doesn't
                    // have to coexist with tensor_map (~70 GB BF16) on a
                    // 128 GB system. Re-read the safetensors after capture
                    // (~30s I/O cost) and re-apply Phases 1.4 + 1.45 + 1.6 +
                    // 1.7 transforms (idempotent — same input produces same
                    // tensor_map). The placeholder lets `tensor_map` stay
                    // in-scope for the downstream Phase 4.x usage.
                    let _drop_for_capture = std::mem::replace(
                        &mut tensor_map,
                        ir::TensorMap::new(),
                    );
                    drop(_drop_for_capture);

                    let mut capture =
                        inference::models::qwen35::activation_capture_real::RealActivationCapture::new(
                            &intermediate_path,
                            &tokenizer_json,
                        )
                        .map_err(|e| {
                            AppError::Conversion(anyhow::anyhow!(
                                "ADR-012 P9b: RealActivationCapture::new failed: {e}"
                            ))
                        })?;

                    tracing::info!(
                        "ADR-012 P9b: capturing activations from intermediate F16 GGUF"
                    );
                    let derived_sensitive_ranges =
                        quantize::dwq_activation::capture_activations_to_sensitive_ranges(
                            &metadata,
                            &dwq_config,
                            &mut capture,
                        )
                        .context(
                            "ADR-012 P9b: activation capture / sensitivity derivation failed",
                        )
                        .map_err(AppError::Conversion)?;

                    drop(capture); // release Qwen35Model (~128 GB for apex MoE) ASAP
                    drop(intermediate_dir); // RAII cleanup of tempdir

                    // Re-read tensor_map from safetensors and re-apply the
                    // qwen35 normalization + transform phases. All
                    // transforms are idempotent (input safetensors haven't
                    // changed), so the result is byte-equivalent to the
                    // tensor_map we just dropped.
                    tracing::info!(
                        "ADR-012 P9b: re-reading tensor_map for DWQ pass after capture"
                    );
                    let (_, mut tensor_map_re) =
                        input::read_model(&config.input_dir, &progress)
                            .context("ADR-012 P9b: re-read of model after capture failed")
                            .map_err(AppError::Conversion)?;

                    // Re-apply the same qwen35 normalization the original
                    // tensor_map went through (Phases 1.4 + 1.45 + 1.6 + 1.7).
                    {
                        let renames: Vec<(String, String)> = tensor_map_re
                            .tensors
                            .keys()
                            .filter(|n| n.contains("language_model."))
                            .map(|n| (n.clone(), n.replace("language_model.", "")))
                            .collect();
                        for (old_name, new_name) in renames {
                            if let Some(mut tensor) = tensor_map_re.tensors.remove(&old_name) {
                                tensor.name = new_name.clone();
                                tensor_map_re.tensors.insert(new_name, tensor);
                            }
                        }
                    }
                    models::qwen35::moe::split_and_rename_fused_gate_up_in_tensor_map(
                        &mut tensor_map_re,
                        &metadata,
                    )
                    .context("ADR-012 P9b: re-read fused gate_up split failed")
                    .map_err(AppError::Conversion)?;
                    models::qwen35::moe::merge_moe_experts_in_tensor_map(
                        &mut tensor_map_re,
                        &metadata,
                    )
                    .context("ADR-012 P9b: re-read MoE expert merge failed")
                    .map_err(AppError::Conversion)?;
                    models::qwen35::apply_rms_norm_plus_one_in_tensor_map(
                        &mut tensor_map_re,
                        &metadata,
                    )
                    .context("ADR-012 P9b: re-read RMS+1 transform failed")
                    .map_err(AppError::Conversion)?;
                    models::qwen35::apply_qwen35_linear_attn_transforms_in_tensor_map(
                        &mut tensor_map_re,
                        &metadata,
                    )
                    .context("ADR-012 P9b: re-read linear-attn transforms failed")
                    .map_err(AppError::Conversion)?;

                    // ADR-012 Bug 7 (2026-04-26): re-apply HF2Q_QWEN35_DROP_MTP=1
                    // MTP-tensor drop to the re-read tensor_map.  The first-pass
                    // call (main.rs:543-559) dropped 15 `mtp.*` tensors from the
                    // initial tensor_map, but the freshly-re-read safetensors
                    // still contain them.  Without this re-application, DWQ
                    // calibrates + emits 866 tensors total (851 base + 15 MTP)
                    // while the metadata's `block_count = 64` (no MTP) tells
                    // llama.cpp to expect 851; load fails with
                    //   done_getting_tensors: wrong number of tensors;
                    //   expected 866, got 851
                    // because the 15 MTP tensors have names llama.cpp doesn't
                    // recognize for a 64-block qwen35 model.  Mirror the
                    // first-pass drop here for re-read symmetry.  Same env-var
                    // gate as the first-pass branch; same removal condition
                    // (when llama.cpp gains qwen35 MTP loading OR 2026-Q4).
                    {
                        let drop_mtp = std::env::var("HF2Q_QWEN35_DROP_MTP")
                            .map(|v| v == "1")
                            .unwrap_or(false);
                        if drop_mtp {
                            let mtp_keys: Vec<String> = tensor_map_re
                                .tensors
                                .keys()
                                .filter(|n| n.starts_with("mtp."))
                                .cloned()
                                .collect();
                            if !mtp_keys.is_empty() {
                                tracing::warn!(
                                    dropped = mtp_keys.len(),
                                    "ADR-012 P9b Bug 7: HF2Q_QWEN35_DROP_MTP=1 re-applied on re-read; dropping {} mtp.* tensors",
                                    mtp_keys.len()
                                );
                                for key in mtp_keys {
                                    tensor_map_re.tensors.remove(&key);
                                }
                            }
                        }
                    }

                    // ADR-012 Bug 6 (2026-04-26): re-apply Phase 1.8 vocab-pad
                    // de-pad to the re-read tensor_map.  The first-pass call
                    // already updated `metadata.vocab_size` to the de-padded
                    // value (e.g. 248044 for Qwen3.6-27B), but the freshly-
                    // re-read safetensors still have the padded
                    // model.embed_tokens.weight + lm_head.weight rows
                    // (248320).  Without this re-application, the final DWQ
                    // GGUF embeds at 248320 rows and llama.cpp rejects with
                    //   tensor 'token_embd.weight' has wrong shape;
                    //   expected H, 248044, got H, 248320
                    // The refactored `truncate_padded_vocab` detects padded
                    // rows from `tensor_map` itself, so calling with
                    // `true_vocab = metadata.vocab_size` is a no-op when the
                    // tensor_map is already aligned and an active truncate
                    // when the re-read produced padded rows.
                    let de_padded_vocab = metadata.vocab_size;
                    let truncated = input::truncate_padded_vocab(
                        &mut tensor_map_re,
                        &mut metadata,
                        de_padded_vocab,
                    );
                    if truncated > 0 {
                        tracing::info!(
                            truncated_tensors = truncated,
                            de_padded_vocab,
                            "ADR-012 P9b Bug 6: re-applied vocab-pad fix to re-read tensor_map"
                        );
                    }

                    tracing::info!(
                        "ADR-012 P9b: running DWQ scale calibration with {} activation-derived sensitive layer ranges",
                        derived_sensitive_ranges.len()
                    );
                    let model = quantize::dwq_activation::run_dwq_with_sensitive_ranges(
                        &tensor_map_re,
                        &metadata,
                        &dwq_config,
                        derived_sensitive_ranges,
                        &progress,
                    )
                    .context("ADR-012 P9b: DWQ scale calibration failed")
                    .map_err(AppError::Conversion)?;
                    // Re-bind tensor_map to the re-read version for the
                    // downstream Phase 4.x bit-overrides + write path.
                    tensor_map = tensor_map_re;
                    model
                } else {
                    tracing::info!(
                        "Using DWQ weight-space calibration (non-qwen35 arch)"
                    );
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

    // Phase 4.5: Quality measurement (before write, while quantized_model is available).
    //
    // ADR-012 / Task #13: use the streaming variant which dequantizes and compares
    // one tensor at a time, bounding peak memory to ~max_tensor_F32 × 2 instead of
    // whole_model × 2.  The pre-Task-#13 path held both the original `tensor_map`
    // (~108 GB F32 for 27B) and a fully-dequantized clone (~108 GB F16) at once,
    // peaking near 170 GB on a 128 GB M5 Max and OOM-killing the convert process.
    let quality_report = if config.skip_quality {
        tracing::info!("Quality measurement skipped (--skip-quality)");
        quality::QualityReport::empty()
    } else if let Some(ref qm) = quantized_model {
        match quality::measure_quality_streaming(
            &tensor_map,
            qm,
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
