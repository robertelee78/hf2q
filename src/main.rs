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
pub mod calibrate;
pub mod backends;
pub mod cli;
mod debug;
mod doctor;
pub mod models;
pub mod gguf_patch;
pub mod gpu;
pub mod inference;
pub mod input;
pub mod intelligence;
pub mod ir;
pub mod preflight;
pub mod progress;
pub mod quality;
pub mod quantize;
pub mod report;
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
        Command::GgufPatch(args) => cmd_gguf_patch(args),
        Command::Info(args) => cmd_info(args).map_err(AppError::Input),
        Command::Validate(args) => cmd_validate(args),
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
    }
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

/// How to source the [`crate::inference::models::qwen35::activation_capture::ActivationCapture`]
/// for calibrators that need a forward pass.
///
/// `None` is the explicit "no capture" state — used by uncalibrated
/// variants and by `DwqArch::Other` (weight-space DWQ).
///
/// `Eager` carries an already-built capture (used by tests and by
/// callers that share a model with a live inference session).
///
/// `Lazy` carries the tokenizer needed by the calibrator to build a
/// `RealActivationCapture` directly from the `LazyTensorMap` it receives
/// inside `calibrate(...)`.
enum CaptureSpec {
    None,
    #[allow(dead_code)] // exercised by test-only callers that inject a capture
    Eager(Box<dyn crate::inference::models::qwen35::activation_capture::ActivationCapture + Send + Sync>),
    Lazy { tokenizer: Box<tokenizers::Tokenizer> },
}

/// Select the [`Calibrator`] impl for a given `--quant` variant
/// (ADR-014 P8 iter-1 — Decision 12 + Decision 13; P4 iter-1 — lazy capture build).
///
/// ## Mapping (exhaustive over the 17-variant menu)
///
/// | Variant family                | Calibrator                                    |
/// |-------------------------------|-----------------------------------------------|
/// | `Dwq46` / `Dwq48` / `Dwq68` / `Dwq28` | [`DwqCalibrator`]                     |
/// | `ImatrixQ4KM` / `ImatrixQ5KM` / `ImatrixQ6K` / `ImatrixAdaptive` | [`ImatrixCalibrator`] (requires capture for forward-pass arches; returns `Err(ForwardPassUnavailable)` otherwise — no NoneCalibrator silent fallback) |
/// | `Auto` / `F16` / `Bf16` / `Q2` / `Q4` / `Q8` / `Q4KM` / `Q5KM` / `Q6K` | [`NoneCalibrator`] (uncalibrated path) |
///
/// ## No-silent-fallback contract
///
/// For Imatrix variants, when `capture` is `None`, this helper returns
/// `Err(CalibrationError::ForwardPassUnavailable)` rather than silently
/// downgrading to `NoneCalibrator`. This mirrors the [`DwqCalibrator`]
/// no-silent-fallback contract from ADR-013 D13: a missing forward-pass
/// driver is a typed error, never a quality regression.
///
/// ## Decision 12 lock — exhaustive match
///
/// The match is exhaustive (no `_` arm). Adding a new variant to
/// [`cli::QuantMethod`] without updating this dispatch fails the build,
/// which is the intended Decision 12 lock.
///
/// `dwq_arch` is the routing decision driving
/// [`calibrate::dwq_calibrator::DwqCalibrator::requires_forward_pass`].
/// `num_layers` and `hidden_size` are passed to [`ImatrixCalibrator::new`]
/// so its calibrate-time shape validation can fire (mismatch → typed
/// error, not silent corruption).
///
/// [`DwqCalibrator`]: crate::calibrate::dwq_calibrator::DwqCalibrator
/// [`ImatrixCalibrator`]: crate::calibrate::imatrix_calibrator::ImatrixCalibrator
/// [`NoneCalibrator`]: crate::calibrate::calibrator::NoneCalibrator
#[allow(clippy::too_many_arguments)]
fn select_calibrator(
    method: cli::QuantMethod,
    dwq_arch: calibrate::dwq::DwqArch,
    capture: CaptureSpec,
    base_bits: u8,
    sensitive_bits: u8,
    calibration_samples: u32,
    num_layers: u32,
    hidden_size: u32,
) -> Result<Box<dyn calibrate::calibrator::Calibrator>, calibrate::calibrator::CalibrationError> {
    use cli::QuantMethod::*;
    match method {
        Dwq46 | Dwq48 | Dwq68 | Dwq28 => {
            // P4 iter-1: route qwen35 DWQ through the lazy build path so
            // the calibrator constructs RealActivationCapture from the
            // resident tensor map.
            match capture {
                CaptureSpec::Lazy { tokenizer } => Ok(Box::new(
                    calibrate::dwq_calibrator::DwqCalibrator::with_activation_capture_lazy(
                        dwq_arch,
                        *tokenizer,
                        base_bits,
                        sensitive_bits,
                        calibration_samples,
                    ),
                )),
                CaptureSpec::Eager(c) => {
                    Ok(Box::new(calibrate::dwq_calibrator::DwqCalibrator::new(
                        dwq_arch,
                        Some(c),
                        base_bits,
                        sensitive_bits,
                        calibration_samples,
                    )))
                }
                CaptureSpec::None => {
                    Ok(Box::new(calibrate::dwq_calibrator::DwqCalibrator::new(
                        dwq_arch,
                        None,
                        base_bits,
                        sensitive_bits,
                        calibration_samples,
                    )))
                }
            }
        }
        ImatrixQ2KS | ImatrixQ2K | ImatrixQ3KS | ImatrixQ3KM | ImatrixQ3KL | ImatrixQ4KS | ImatrixQ4KM | ImatrixQ5KS | ImatrixQ5KM | ImatrixQ6K | ImatrixAdaptive => {
            // ADR-014 Decision 13 + ADR-013 D13: no NoneCalibrator silent
            // fallback when the forward-pass driver is absent. Surface a
            // typed error so the caller can either supply a capture or
            // pick a non-imatrix variant. ImatrixCalibrator itself only
            // accepts an eagerly-built capture today; the lazy qwen35 path
            // is DWQ-only.
            let real_capture = match capture {
                CaptureSpec::Eager(c) => c,
                CaptureSpec::None | CaptureSpec::Lazy { .. } => {
                    return Err(calibrate::calibrator::CalibrationError::ForwardPassUnavailable {
                        arch: format!("{dwq_arch:?}"),
                    });
                }
            };
            Ok(Box::new(
                calibrate::imatrix_calibrator::ImatrixCalibrator::new(
                    real_capture,
                    num_layers,
                    hidden_size,
                ),
            ))
        }
        Auto | F16 | Bf16 | Q2 | Q4 | Q8 | Q2KS | Q2K | Q3KS | Q3KM | Q3KL | Q4KS | Q4KM | Q5KS | Q5KM | Q6K => {
            Ok(Box::new(calibrate::calibrator::NoneCalibrator::new()))
        }
    }
}

/// Build a [`crate::calibrate::calibrator::CalibrationCorpus`] for the
/// `cmd_convert` Phase 2 dispatch (ADR-014 P2 iter-2 §S1).
///
/// **Why colocated in main.rs**: this helper exists to bridge the
/// converter's `metadata.vocab_size + calibration_samples` configuration
/// into the [`crate::calibrate::calibrator::Calibrator`] trait's
/// arch-agnostic corpus shape. It is *not* a general-purpose corpus
/// builder — the per-arch tokenizer-driven corpus (e.g. wikitext-2 for
/// imatrix) belongs with the calibrator itself once it lands.
///
/// The returned corpus is a single chunk of synthetic tokens — the
/// exact same shape that
/// [`crate::calibrate::dwq_activation::capture_activations_to_sensitive_ranges`]
/// generates internally via `generate_calibration_tokens`. The
/// [`crate::calibrate::dwq_calibrator::DwqCalibrator`] consumes the
/// corpus only as a cache-key contributor; the real tokens are derived
/// by `capture_activations_to_sensitive_ranges` so this corpus shape is
/// informational for DWQ. For `ImatrixCalibrator`, the corpus chunks
/// drive the forward-pass loop directly (one chunk = one capture call).
fn build_calibration_corpus(
    samples: u32,
    vocab_size: u64,
    name: &str,
) -> calibrate::calibrator::CalibrationCorpus {
    // Mirror `dwq_activation::generate_calibration_tokens` — deterministic
    // synthetic tokens drawn from `[0, vocab_size)` at sample positions.
    // The exact distribution is unimportant because DwqCalibrator's cache
    // key consumes corpus_sha (the SHA over the produced tokens), and
    // ImatrixCalibrator's mock-test path uses MockActivationCapture's
    // monotonic formula. Real wikitext-2 corpora live one layer up and
    // pre-tokenize before reaching this builder.
    let chunk_len = samples as usize;
    let v = vocab_size.max(1) as u32;
    let mut chunk = Vec::with_capacity(chunk_len);
    for i in 0..chunk_len {
        chunk.push((i as u32) % v);
    }
    calibrate::calibrator::CalibrationCorpus {
        chunks: if chunk_len == 0 { vec![] } else { vec![chunk] },
        name: name.to_string(),
    }
}

/// Convert a [`crate::calibrate::calibrator::CalibrationData::Dwq`]
/// per-layer flag map (the [`crate::calibrate::dwq_calibrator::DwqCalibrator`]
/// output shape) back into the
/// [`crate::calibrate::dwq::DwqConfig::sensitive_layers`] range list
/// shape that the downstream byte-emit consumes (ADR-014 P2 iter-2 §S2).
///
/// The map's keys are `blk.<i>.sensitivity` synthetic tensor names; each
/// value is a single-element `Vec<f32>` carrying `1.0` (sensitive) or
/// `0.0` (base). Adjacent sensitive indices are coalesced into ranges
/// for compactness — the same minimisation
/// [`crate::calibrate::dwq_activation::indices_to_ranges`] applies.
///
/// Returns an empty `Vec` for `CalibrationData::Dwq(empty)` (the
/// `DwqArch::Other` weight-space contract) and for `CalibrationData::None`.
fn dwq_calibration_to_sensitive_ranges(
    data: &calibrate::calibrator::CalibrationData,
) -> Vec<std::ops::RangeInclusive<usize>> {
    let map = match data {
        calibrate::calibrator::CalibrationData::Dwq(m) => m,
        _ => return Vec::new(),
    };
    let prefix = calibrate::dwq_calibrator::SENSITIVITY_TENSOR_PREFIX;
    let suffix = calibrate::dwq_calibrator::SENSITIVITY_TENSOR_SUFFIX;
    let mut indices: Vec<usize> = Vec::new();
    for (key, value) in map.iter() {
        let is_sensitive = value.first().map(|v| *v >= 0.5).unwrap_or(false);
        if !is_sensitive {
            continue;
        }
        // Parse "blk.<N>.sensitivity" → N
        let stripped = match key.strip_prefix(&format!("{prefix}.")) {
            Some(s) => s,
            None => continue,
        };
        let idx_str = match stripped.strip_suffix(&format!(".{suffix}")) {
            Some(s) => s,
            None => continue,
        };
        if let Ok(idx) = idx_str.parse::<usize>() {
            indices.push(idx);
        }
    }
    if indices.is_empty() {
        return Vec::new();
    }
    indices.sort_unstable();
    indices.dedup();
    let mut ranges = Vec::new();
    let mut start = indices[0];
    let mut end = indices[0];
    for &i in &indices[1..] {
        if i == end + 1 {
            end = i;
        } else {
            ranges.push(start..=end);
            start = i;
            end = i;
        }
    }
    ranges.push(start..=end);
    ranges
}

fn clone_tensor_map_to_lazy(tensor_map: &crate::ir::TensorMap) -> crate::ir::lazy::LazyTensorMap {
    let mut out = crate::ir::lazy::LazyTensorMap::new();
    for (_, tensor) in tensor_map.iter() {
        let meta = crate::ir::lazy::LazyMeta::new(
            tensor.name.clone(),
            tensor.shape.clone(),
            tensor.dtype,
        );
        // ADR-020 P13 step 4: from_arc_bytes + Arc::clone is a pointer-bump
        // share, NOT a deep byte clone. Pre-2026-05-06 this was
        // `from_bytes(meta, tensor.data.clone())` — the 52 GB clone for
        // Qwen3.6-27B that dominated the 199 GB DWQ peak.
        out.insert(crate::ir::lazy::LazyTensor::from_arc_bytes(
            meta,
            std::sync::Arc::clone(&tensor.data),
        ));
    }
    out
}

/// ADR-014 P7 iter-90 — extract the duplicated 3-arm Phase 3 dispatch
/// from each `cli::QuantMethod` arm of `cmd_convert`. Reads the two
/// env-flag toggles in fixed precedence order (MUT > legacy > eager)
/// and dispatches to the matching `quantize` entry point.
///
/// Precedence:
///   - `HF2Q_STREAMING_PHASE3_MUT=1` → zero-byte-copy consuming-mut
///     wedge (iter-83); drains `tensor_map` via `take_data_as_arc`.
///   - `HF2Q_STREAMING_PHASE3=1`     → borrowed wedge (iter-77);
///     wraps each `t.data.clone()` into `Arc<Vec<u8>>`.
///   - else                          → eager `quantize_model`.
///
/// Consolidates the four arms (K-quant codec / ImatrixAdaptive / DwqK /
/// StaticQuantizer) which previously each duplicated the same if/else
/// block (iter-84/85/87/88 inline). Byte-identity across all 4 arms ×
/// all 3 paths is locked by the `cmd_convert_dispatch` SHA matrix gates
/// (iter-61/62/64/66/84/89).
///
/// `arm_label` appears in the routing log line so traces stay
/// attributable to the originating dispatch site.
fn dispatch_phase3_quantize(
    tensor_map: &mut crate::ir::TensorMap,
    metadata: &crate::ir::ModelMetadata,
    quantizer: &dyn quantize::Quantizer,
    bits: u8,
    group_size: usize,
    progress: &progress::ProgressReporter,
    arm_label: &'static str,
) -> Result<crate::ir::QuantizedModel, quantize::QuantizeError> {
    let phase3_mut = std::env::var("HF2Q_STREAMING_PHASE3_MUT").as_deref() == Ok("1");
    let phase3 = std::env::var("HF2Q_STREAMING_PHASE3").as_deref() == Ok("1");
    if phase3_mut {
        tracing::info!(
            arm = arm_label,
            "ADR-014 P7 iter-90 helper: HF2Q_STREAMING_PHASE3_MUT=1 → quantize_via_streaming_consuming_mut (zero-byte-copy)"
        );
        quantize::quantize_via_streaming_consuming_mut(
            tensor_map, metadata, quantizer, bits, group_size, progress,
        )
    } else if phase3 {
        tracing::info!(
            arm = arm_label,
            "ADR-014 P7 iter-90 helper: HF2Q_STREAMING_PHASE3=1 → quantize_via_streaming_borrowed"
        );
        quantize::quantize_via_streaming_borrowed(
            tensor_map, metadata, quantizer, bits, group_size, progress,
        )
    } else {
        quantize::quantize_model(
            tensor_map, metadata, quantizer, bits, group_size, progress,
        )
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

    let mut config = cli::resolve_convert_config(&args)
        .context("Failed to resolve conversion configuration")
        .map_err(AppError::Input)?;

    // ADR-014 P8 Decision 12 §off-diagonal — validate the orthogonal
    // (--calibration, --output-format) selector with the
    // HF2Q_UNSAFE_EXPERIMENTS dev gate. Diagonal cells (e.g. imatrix +
    // k-quant-q4_k_m) are always accepted; off-diagonal cells (e.g.
    // imatrix + bit-pair-4-6) require the env gate to surface
    // accidental misconfigurations.
    let orthogonal_pair = {
        let env_unsafe = std::env::var(cli::HF2Q_UNSAFE_EXPERIMENTS_ENV).ok();
        cli::validate_off_diagonal_selector(
            config.calibration,
            config.output_format,
            env_unsafe.as_deref(),
        )
        .map_err(|e| AppError::Input(anyhow::anyhow!("{e}")))?
    };
    if let Some((cal, fmt)) = orthogonal_pair {
        // Future P2 iter-2 work will route this orthogonal pair into
        // its own (Calibrator, OutputFormat) dispatch path. For this
        // iter we accept the pair, surface a tracing event, and
        // continue with the existing --quant dispatch (the diagonal
        // cells are reachable via --quant today; off-diagonal cells
        // produce a tracing breadcrumb for the dev-gate user).
        tracing::info!(
            calibration = %cal,
            output_format = %fmt,
            diagonal = cli::is_diagonal_cell(cal, fmt),
            "ADR-014 P8 §off-diagonal: orthogonal selector accepted (P2 iter-2 wires the live dispatch)"
        );
    }

    let progress = ProgressReporter::new();

    tracing::info!(
        input = %config.input_dir.display(),
        format = %config.format,
        quant = %config.quant,
        "Starting conversion"
    );

    // ADR-014 P7 iter-91 (2026-04-28): defensive guard against _MUT × Phase 4.5
    // mismatch. HF2Q_STREAMING_PHASE3_MUT drains tensor_map mid-Phase-3 (via
    // TensorRef::take_data_as_arc → mem::take on the inner Vec<u8>), leaving
    // Phase 4.5 quality measurement with empty bytes. Until the lazy-from-disk
    // Phase 4.5 path lands (iter-3 wholesale work), the _MUT wedge must be
    // paired with --skip-quality. Fail fast with a clear message rather than
    // silently producing a degenerate quality report from drained tensors.
    //
    // Two supported workarounds in the error message:
    //   1. --skip-quality (use `hf2q validate` post-conversion)
    //   2. HF2Q_STREAMING_PHASE3=1 (borrowed wedge — shares via Arc, keeps
    //      tensor_map populated, preserves Phase 4.5 quality measurement)
    let phase3_mut = std::env::var("HF2Q_STREAMING_PHASE3_MUT").as_deref() == Ok("1");
    if phase3_mut && !config.skip_quality {
        return Err(AppError::Conversion(anyhow::anyhow!(
            "HF2Q_STREAMING_PHASE3_MUT=1 (zero-byte-copy wedge) requires \
             --skip-quality. The _MUT path drains the in-memory tensor_map \
             during Phase 3 dispatch, so Phase 4.5 quality measurement would \
             operate on empty bytes. Workarounds:\n  \
             1. Re-run with --skip-quality (recommended for memory-constrained \
                DWQ re-emit; measure quality via `hf2q validate` post-conversion).\n  \
             2. Use HF2Q_STREAMING_PHASE3=1 instead (borrowed wedge — shares \
                bytes via Arc, keeps tensor_map populated, preserves Phase 4.5 \
                quality)."
        )));
    }

    // Phase 0: Read model metadata for preflight
    //
    // 2026-05-02: `metadata` previously needed `mut` for Phase 1.8
    // (vocab-pad de-pad, ADR-012) which overwrote `metadata.vocab_size`
    // post-depad. Now that depad is removed (commit 505b5b8 + this one),
    // metadata.vocab_size stays at config.json's authoritative value
    // throughout — no mutation required.
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

    // ADR-014 P7 iter-8 — Calibrator dispatch seam.
    //
    // `select_calibrator` returns a `Box<dyn Calibrator>` based on the
    // `--quant` variant. This iter logs the selection but leaves the
    // existing `DwqQuantizer` / `KQuantCodecQuantizer` /
    // `VariantKQuantizer` dispatch unchanged — the full
    // `(Calibrator, OutputFormat)` restructure is P2 iter-2 (separate
    // task; do not preempt). The seam exists so that future iter can
    // pull on it without re-deriving the variant → calibrator map.
    {
        let dwq_arch = calibrate::dwq::DwqArch::from_hf_architecture(
            &metadata.architecture,
            metadata.is_moe(),
        );
        let (base_bits, sensitive_bits) = config
            .quant
            .dwq_bit_pair()
            .unwrap_or((4, 6));
        // ADR-014 P8: select_calibrator now routes Imatrix variants to
        // ImatrixCalibrator. The diagnostic seam at this iter still
        // passes `capture: None` because the live capture is built
        // later at the DWQ / imatrix dispatch site (see below). For
        // Imatrix variants, that surfaces ForwardPassUnavailable here
        // — we *log* the typed error and continue (the live dispatch
        // site below builds its own capture and re-runs select_calibrator
        // with `Some(_)`). Suppressing here is correct: this block is
        // the diagnostic preview, not the live path.
        match select_calibrator(
            config.quant,
            dwq_arch,
            CaptureSpec::None, // capture is attached at the live DWQ dispatch site below
            base_bits,
            sensitive_bits,
            config.calibration_samples,
            metadata.num_layers,
            metadata.hidden_size as u32,
        ) {
            Ok(calibrator) => {
                tracing::info!(
                    calibrator = %calibrator.name(),
                    requires_forward_pass = calibrator.requires_forward_pass(),
                    quant = %config.quant,
                    arch = ?dwq_arch,
                    "selected calibrator"
                );
            }
            Err(e) => {
                tracing::info!(
                    quant = %config.quant,
                    arch = ?dwq_arch,
                    err = %e,
                    "calibrator preview skipped (live dispatch will attach capture below)"
                );
            }
        }
    }

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

            // ADR-014 P8 Decision 18: AutoResolver returns the new
            // 17-variant menu strings. The map below covers every
            // string the auto path can emit; an unrecognised string is
            // a defect (resolver and CLI are out of sync), so we
            // surface a typed conversion error rather than silently
            // falling back to q4 (no-shortcut contract).
            let resolved_quant = match resolved.quant_method.as_str() {
                "auto" => cli::QuantMethod::Auto,
                "f16" => cli::QuantMethod::F16,
                "bf16" => cli::QuantMethod::Bf16,
                "q2" => cli::QuantMethod::Q2,
                "q4" => cli::QuantMethod::Q4,
                "q8" => cli::QuantMethod::Q8,
                "q2_k_s" => cli::QuantMethod::Q2KS,
                "q2_k" => cli::QuantMethod::Q2K,
                "q3_k_s" => cli::QuantMethod::Q3KS,
                "q3_k_m" => cli::QuantMethod::Q3KM,
                "q3_k_l" => cli::QuantMethod::Q3KL,
                "q4_k_s" => cli::QuantMethod::Q4KS,
                "q4_k_m" => cli::QuantMethod::Q4KM,
                "q5_k_s" => cli::QuantMethod::Q5KS,
                "q5_k_m" => cli::QuantMethod::Q5KM,
                "q6_k" => cli::QuantMethod::Q6K,
                "imatrix-q2_k_s" => cli::QuantMethod::ImatrixQ2KS,
                "imatrix-q2_k" => cli::QuantMethod::ImatrixQ2K,
                "imatrix-q3_k_s" => cli::QuantMethod::ImatrixQ3KS,
                "imatrix-q3_k_m" => cli::QuantMethod::ImatrixQ3KM,
                "imatrix-q3_k_l" => cli::QuantMethod::ImatrixQ3KL,
                "imatrix-q4_k_s" => cli::QuantMethod::ImatrixQ4KS,
                "imatrix-q4_k_m" => cli::QuantMethod::ImatrixQ4KM,
                "imatrix-q5_k_s" => cli::QuantMethod::ImatrixQ5KS,
                "imatrix-q5_k_m" => cli::QuantMethod::ImatrixQ5KM,
                "imatrix-q6_k" => cli::QuantMethod::ImatrixQ6K,
                "imatrix-adaptive" => cli::QuantMethod::ImatrixAdaptive,
                "dwq-4-6" => cli::QuantMethod::Dwq46,
                "dwq-4-8" => cli::QuantMethod::Dwq48,
                "dwq-6-8" => cli::QuantMethod::Dwq68,
                "dwq-2-8" => cli::QuantMethod::Dwq28,
                other => {
                    return Err(AppError::Conversion(anyhow::anyhow!(
                        "Auto mode recommended '{}' which is not in the \
                         Decision-12 17-variant menu. AutoResolver and CLI \
                         are out of sync — this is a defect, not a runtime \
                         fallback path.",
                        other
                    )));
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
            "--bits is not used for DWQ; use --quant dwq-N-M to choose \
             bit-pair variants (e.g. dwq-4-6, dwq-4-8, dwq-6-8, dwq-2-8)"
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
    //
    // ADR-014 P1: lifted onto LazyTensorMap. The split materialises the
    // fused parent tensor (1.5 GB peak per layer for apex MoE), splits
    // into halves, drops the parent. The down_proj rename is pure
    // metadata. Streaming benefit at this phase is bounded — Decision 7
    // (Phase 1.5) provides the layer-streaming MoE merge that picks up
    // these split tensors.
    models::qwen35::moe::split_and_rename_fused_gate_up_in_lazy_map(
        &mut lazy_map,
        &metadata,
    )
    .context("qwen35moe fused gate_up split + .weight rename (lazy) failed")
    .map_err(AppError::Conversion)?;

    // Phase 1.5: ADR-012 Decision 9 / P5 — qwen35moe expert merge.
    //
    // Must run BEFORE quantization: the merge consumes pre-quantization
    // F16/BF16 bytes where shape × dtype.size matches data.len().
    // Without this call, qwen35moe GGUFs emit N_experts × 3 × N_layers
    // separate per-expert tensors instead of 3 × N_layers merged ones,
    // and llama.cpp's loader rejects the file.
    //
    // ADR-014 P1 + Decision 7 layer-streaming: lifted onto
    // LazyTensorMap. Each merged (layer, projection) is one LazyTensor
    // whose closure stacks 256 experts at materialise time — the
    // streaming quantize loop (P2) materialises one merged tile,
    // quantises it, writes it, drops it, then proceeds to the next.
    // Peak resident bytes stay bounded by one merged tile (~750 MB
    // apex BF16) instead of the eager-pipeline ~80 GB stack of all
    // merged tiles at once.
    //
    // No-op for dense arches and pre-merged inputs (apex MoE post
    // Phase 1.45's split is already pre-merged).
    models::qwen35::moe::merge_moe_experts_in_lazy_map(&mut lazy_map, &metadata)
        .context("qwen35moe expert merge (lazy) failed")
        .map_err(AppError::Conversion)?;

    // Phase 1.6: ADR-012 P3 Decision 6 — RMS norm +1 bias (Qwen3.5 family
    // `gamma + 1` convention, py:4794-4795). Baked into the stored weights
    // at convert time. Silent wire-up gap caught by spec-source audit
    // 2026-04-24 — P3 shipped the per-tensor transform but never called
    // it; before this fix, converted Qwen3.5 GGUFs shipped RMS norm
    // weights WITHOUT the +1 bias, producing silent logit skew at
    // inference. No-op for non-Qwen3.5 arches (Gemma4, LLaMA).
    //
    // ADR-014 P1: lifted onto LazyTensorMap. Per-tensor `.map()` —
    // closure adds 1.0 to each F32/F16/BF16 RMS-norm weight at
    // materialise time. Shape and dtype preserved.
    models::qwen35::apply_rms_norm_plus_one_in_lazy_map(&mut lazy_map, &metadata)
        .context("qwen35 RMS norm +1 bias application (lazy) failed")
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
    //
    // ADR-014 P1: lifted onto LazyTensorMap. Per-tensor materialise +
    // transform + re-insert as LazyTensor::from_bytes (the transforms
    // slice and permute element data, can't compose via .map()).
    // Streaming benefit bounded — linear-attn projections are MB-scale
    // per layer, orders of magnitude smaller than the GB MoE tiles.
    models::qwen35::apply_qwen35_linear_attn_transforms_in_lazy_map(
        &mut lazy_map,
        &metadata,
    )
    .context("qwen35 linear-attention transforms (lazy) failed")
    .map_err(AppError::Conversion)?;

    // Phase 1.8 (vocab-pad de-pad) — REMOVED 2026-05-02.
    //
    // The depad was added 2026-04-25 as a workaround for a converter bug
    // in `gguf.rs::load_tokenizer_metadata`: at the time the emit code
    // built the GGUF tokens array from `tokenizer.json[model][vocab]`
    // ONLY (the base BPE = 248044 entries for Qwen3.6-27B), missing
    // `added_tokens` whose ids land beyond the base range. The HF
    // embedding matrix has 248320 rows (per `text_config.vocab_size`),
    // and llama.cpp's loader compares tokens.length to embed.rows and
    // rejects a mismatch. The 2026-04-25 depad "fixed" the loader
    // rejection by truncating the embedding from 248320 → 248044,
    // matching the (broken) emit.
    //
    // The cost of this workaround was the user's currently-broken
    // GGUF: rows 248044..248069 (the 26 TRAINED special-token rows
    // including `<|im_end|>`, `<|endoftext|>`, `<think>`, `</think>`)
    // got dropped along with the 250 reserved/PAD rows. The model
    // could no longer emit any of those special tokens as atomic ids,
    // turn-end never fired cleanly, the user observed "stops after
    // thinking with no answer".
    //
    // The 2026-04-25 author DOCUMENTED at `input/mod.rs:127-133` that
    // this was wrong:
    //   "extending the GGUF emit to also include added_tokens (and
    //    bumping the embedding to match, 248070 rows) is the strictly-
    //    correct end state — it would preserve the trained embeddings
    //    for the 26 chat special tokens (currently those rows in
    //    safetensors are dropped during truncation). Tracked separately."
    //
    // 2026-05-02 commit `505b5b8` fixed the converter to mirror
    // llama.cpp's `get_vocab_base`: emits the full `text_config.vocab_size`
    // (= 248320) as the tokens array length, gap-fills [PAD{i}] / UNUSED
    // for reserved slots. With that fix the depad is a NET-NEGATIVE
    // workaround — it strips trained data from a now-correct pipeline.
    // Removing here.
    //
    // The `input::detect_padded_vocab` and `input::truncate_padded_vocab`
    // helpers remain for future use against genuinely malformed sources
    // (HF embedding rows different from config.json's vocab_size); the
    // CALL is removed here because well-formed HF sources don't need it.

    // ADR-014 P7 iter-60 (2026-04-28): move backend creation BEFORE
    // the materialize_all() bridge.  Backend selection is just a
    // `match config.format` that constructs a struct — no tensor work.
    // Moving it earlier sets up the iter-61+ conditional bridge:
    // skip materialize_all when streaming + non-native; materialize
    // only for the native quantize_and_write path (the trait contract
    // requires &TensorMap).
    //
    // Order pre-iter-60: lazy_map → materialize_all → backend creation
    // Order post-iter-60: lazy_map → backend creation → materialize_all
    //
    // Pure ordering change.  No behavior change this iter.
    // ADR-005 Phase 4 iter-211 — derive source-bundle SHA from the
    // verified shards (when available) and stamp three
    // `hf2q.provenance.*` metadata keys at GGUF emit time.  A serve-time
    // auto-pipeline cache hit will short-circuit the per-load 30 GB
    // SHA-256 integrity re-check on a match (iter-207 reader half).
    //
    // `compute_source_bundle_sha256` returns `None` when the shard list
    // has no LFS records (local-source `--input` runs, all-non-LFS
    // shard manifests).  In that case we don't stamp provenance keys —
    // the emitted GGUF parses as `External` and `verify_quantized`
    // runs exactly as pre-iter-211 (byte-identical fall-through).
    //
    // Provenance is mandatory for hf2q-emitted GGUFs WHEN the binding
    // can be computed; there is intentionally NO operator opt-out
    // (`--no-provenance` rejected by design — a half-stamped GGUF is
    // worse than an external one).  Operators who want to skip the
    // bundle-binding entirely use `--no-integrity` at download time,
    // which keeps `source_shards == None` here.
    // ADR-005 iter-211b — close the production mmproj_sha256 stub.
    //
    // BEFORE iter-211b: this site set `mmproj_sha256: None` even when
    // the model carried vision tensors and would emit a sibling mmproj
    // GGUF, because the mmproj writes AFTER main-GGUF metadata is
    // finalized (legacy: `GgufBackend::write` post-body; ADR-012 P10:
    // `cmd_convert` Phase 4.8). That `None` was a "todo later" stub
    // forbidden by the Engineering Mantra; it left auto-pipeline mmproj
    // integrity unbound from the cache manifest.
    //
    // AFTER iter-211b: when vision tensors are present in the source
    // `LazyTensorMap` (so a sibling mmproj GGUF will land), we emit
    // `Some(MMPROJ_SHA256_PLACEHOLDER)` — a 64-byte ASCII-zero string
    // that passes the iter-207 reader's hex validation but is byte-
    // stable for in-place overwrite. After the mmproj sibling lands on
    // disk (Phase 4.6 legacy write or Phase 4.8 ADR-012 P10 emit),
    // Phase 4.85 below computes the sibling's SHA via
    // `compute_file_sha256` and calls `patch_mmproj_sha256_in_gguf`
    // to overwrite the 64 placeholder bytes in place — no kv_count
    // change, no offset shifts, no metadata block re-emit.
    let mmproj_will_emit = lazy_map.iter().any(|(name, _)| {
        let n = name.as_str();
        n.contains("vision_tower") || n.contains("embed_vision")
    });
    let provenance = config.source_shards.as_ref().and_then(|shards| {
        let cache_shards: Vec<serve::cache::SourceShard> =
            shards.iter().map(serve::cache::SourceShard::from_integrity).collect();
        serve::cache::compute_source_bundle_sha256(&cache_shards).map(|sha| {
            backends::gguf::Hf2qProvenance {
                producer_version: format!("hf2q {}", env!("CARGO_PKG_VERSION")),
                source_sha256: sha,
                mmproj_sha256: if mmproj_will_emit {
                    Some(backends::gguf::MMPROJ_SHA256_PLACEHOLDER.to_string())
                } else {
                    None
                },
            }
        })
    });
    let placeholder_emitted = provenance
        .as_ref()
        .and_then(|p| p.mmproj_sha256.as_deref())
        == Some(backends::gguf::MMPROJ_SHA256_PLACEHOLDER);
    if let Some(p) = &provenance {
        tracing::info!(
            producer_version = %p.producer_version,
            source_sha256 = %p.source_sha256,
            "ADR-005 iter-211: stamping hf2q.provenance.* keys at GGUF emit"
        );
    }

    let backend: Box<dyn OutputBackend> = match config.format {
        cli::OutputFormat::Gguf => {
            let mut b = GgufBackend::new();
            if let Some(p) = provenance {
                b = b.with_provenance(p);
            }
            Box::new(b)
        }
        cli::OutputFormat::Safetensors => {
            Box::new(SafetensorsBackend::with_shard_size_gb(config.shard_size_gb))
        }
    };

    // ADR-014 P1 bridge: every Phase 1.x transform now lifted. The
    // streaming quantize loop (P2) will consume from `lazy_map`
    // directly; until P2 lands, bridge through `materialize_all` once,
    // immediately before Phase 2's backend dispatch. After P2 the
    // bridge becomes test-only.
    //
    // **ADR-014 P7 iter-53 survey (2026-04-28)** — remaining tensor_map
    // consumers downstream of this bridge, in order of dependency:
    //
    //   ✅ Phase 3 dispatch (4 arms)            iter-48-51 wired (env-flag)
    //   ✅ Phase 4.5 quality measurement        iter-52 wired (env-flag)
    //
    //   ⏳ Telemetry (cheap):
    //         - tensor_map.total_size_bytes()   needs `LazyTensorMap::total_size_bytes`
    //         - tensor_map.len()                LazyTensorMap::len() exists
    //
    //   ⏳ bf16→f16 in-place conversion at line ~1188:
    //         - needs `LazyTensorMap::convert_bf16_to_f16`
    //         - quantize_streaming already handles per-tensor bf16→f16
    //           via its `bf16_to_f16: bool` arg; the upfront mutation
    //           here is for non-streaming paths.
    //
    //   ⏳ DWQ calibrator path (line 1275-1277):
    //         - clone_tensor_map_to_lazy(&tensor_map) / from_eager_borrowed
    //         - already builds lazy map from tensor_map; trivially flips
    //           when the source is already a LazyTensorMap
    //
    //   ⏳ bit_overrides scan (lines 1765, 1789):
    //         - tensor_map.tensors.keys() → swap to lazy_map.names()
    //
    //   🔴 Phase 4.6 native backend (line 1898):
    //         - backend.quantize_and_write(&tensor_map, ...) — trait
    //           contract requires &TensorMap
    //         - Two options: (A) add quantize_and_write_lazy variant
    //           to OutputBackend trait, (B) keep materialize_all but
    //           move it INSIDE the native arm.  Option B simpler.
    //
    // Plan for the wholesale skip (queued for iter-54+):
    //   1. Replace this `materialize_all()` with conditional bridge:
    //        if backend.requires_native_quantization() {
    //            // native path needs eager tensor_map; materialize here
    //            tensor_map = lazy_map.materialize_all()?;
    //        } else {
    //            // non-native path stays lazy through Phase 3 + 4.5;
    //            // tensor_map is never materialised
    //        }
    //   2. Add LazyTensorMap::total_size_bytes + convert_bf16_to_f16.
    //   3. Carry lazy_map through Phase 3 + 4.5 + bit_overrides + telemetry.
    //
    // Until then, this bridge runs upfront and `HF2Q_STREAMING_PHASE3=1`
    // (iter-48 onwards) exercises the streaming code path through the
    // resident eager tensor_map (TEST INTEGRATION not memory win).
    let mut tensor_map = lazy_map
        .materialize_all()
        .context("Failed to materialise lazy tensor map (P1→P2 bridge)")
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
    //
    // ADR-014 P9 iter-1 §S5 — propagate `--shard-size-gb` (default 5.0)
    // into the safetensors emit path. The flag is ignored by the GGUF
    // backend (which writes a single .gguf file regardless of size).
    //
    // ADR-014 P7 iter-60 (2026-04-28): backend creation moved BEFORE
    // the materialize_all() bridge (above) — see iter-60 ordering note.
    // This is now a no-op marker comment kept for git-blame continuity.

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

    // ──────────────────────────────────────────────────────────────────
    // ADR-014 P4 iter-1 — Calibrator-first dispatch without the temporary
    // qwen35 capture artifact. DWQ on qwen35/qwen35moe now keeps the
    // transformed tensor_map resident and lets DwqCalibrator build
    // RealActivationCapture directly from a LazyTensorMap view.
    // ──────────────────────────────────────────────────────────────────
    let dwq_arch_for_dispatch = calibrate::dwq::DwqArch::from_hf_architecture(
        &metadata.architecture,
        metadata.is_moe(),
    );
    let (cal_base_bits, cal_sensitive_bits) = config.quant.dwq_bit_pair().unwrap_or((4, 6));

    let needs_dwq_lazy_capture = matches!(
        config.quant,
        cli::QuantMethod::Dwq46
            | cli::QuantMethod::Dwq48
            | cli::QuantMethod::Dwq68
            | cli::QuantMethod::Dwq28
    ) && dwq_arch_for_dispatch.requires_activation_capture()
        && !backend.requires_native_quantization();

    let capture_spec = if needs_dwq_lazy_capture {
        let tokenizer_json = config.input_dir.join("tokenizer.json");
        if !tokenizer_json.exists() {
            return Err(AppError::Conversion(anyhow::anyhow!(
                "ADR-014 P4: qwen35/qwen35moe DWQ requires tokenizer.json \
                 in {}; not found. No weight-space fallback per Decision 13.",
                config.input_dir.display()
            )));
        }
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_json).map_err(|e| {
            AppError::Conversion(anyhow::anyhow!(
                "ADR-014 P4: failed to load tokenizer.json for qwen35/qwen35moe DWQ \
                 at {}: {e}",
                tokenizer_json.display()
            ))
        })?;
        CaptureSpec::Lazy {
            tokenizer: Box::new(tokenizer),
        }
    } else {
        CaptureSpec::None
    };

    let mut calibrator = if backend.requires_native_quantization() {
        // Native-backend path skips IR quantization entirely; no
        // calibrator needed. Use a NoneCalibrator stub to keep the type
        // uniform — it is never invoked.
        Box::new(calibrate::calibrator::NoneCalibrator::new()) as Box<dyn calibrate::calibrator::Calibrator>
    } else {
        select_calibrator(
            config.quant,
            dwq_arch_for_dispatch,
            capture_spec,
            cal_base_bits,
            cal_sensitive_bits,
            config.calibration_samples,
            metadata.num_layers,
            metadata.hidden_size as u32,
        )
        .map_err(|e| {
            AppError::Conversion(anyhow::anyhow!(
                "ADR-014 P4: select_calibrator failed for --quant {}: {e:#}",
                config.quant
            ))
        })?
    };

    let calibration_data = if calibrator.requires_forward_pass() {
        let corpus = build_calibration_corpus(
            config.calibration_samples,
            metadata.vocab_size,
            "hf2q-cmd-convert-synthetic",
        );

        // ADR-014 P4 iter-103 (2026-04-29): hoist cache HIT short-circuit
        // ABOVE `clone_tensor_map_to_lazy`.  iter-95 hoisted the cache
        // check above the Qwen35Model build *inside* `calibrator.calibrate()`,
        // but the dense + MoE wedge sites here still pay a 70 GB clone in
        // `clone_tensor_map_to_lazy` BEFORE calibrate() is invoked. For
        // Qwen3.6 35B-A3B the clone alone pushes peak to ~140 GB (tensor_map
        // 70 GB + lazy_view clone 70 GB) — jetsam-killed before calibrate()
        // ever runs. The cache key only depends on tensor metadata
        // (name/shape/dtype) which `LazyTensorMap::from_eager_borrowed`
        // exposes without cloning bytes; `model_fingerprint` never calls
        // `materialize()`. Compute the key from the borrowed view, look up
        // the cache, and skip the clone + calibrate entirely on HIT.
        //
        // 35BMOE re-emit unblocked under this hoist + the
        // iter-95/iter-100b cache priming (ADR-014 P11 closure path).
        let borrowed_view = ir::lazy::LazyTensorMap::from_eager_borrowed(&tensor_map);
        let cache_key = calibrate::cache::SensitivityCacheKey::with_algorithm_version(
            calibrate::calibrator::model_fingerprint(&borrowed_view, &metadata),
            calibrate::calibrator::corpus_sha(&corpus),
            calibrate::cache::SENSITIVITY_ALGORITHM_VERSION,
        );
        tracing::warn!(
            arch = ?dwq_arch_for_dispatch,
            cache_key = %cache_key.hash(),
            algorithm_version = calibrate::cache::SENSITIVITY_ALGORITHM_VERSION,
            "DWQ cache key (iter-103 P11 cache-priming aid; pre-clone hoist)"
        );
        let pre_clone_cache_hit: Option<calibrate::calibrator::CalibrationData> =
            match calibrate::cache::cache_file_path(&cache_key) {
                Ok(path) => match calibrate::cache::load_dwq_from_path(&path, &cache_key) {
                    Ok(Some(map)) => {
                        tracing::info!(
                            cache_key = %cache_key.hash(),
                            path = %path.display(),
                            entries = map.len(),
                            "ADR-014 P4 iter-103: DWQ cache HIT pre-clone — skipping \
                             clone_tensor_map_to_lazy + calibrate (saves ~tensor_map bytes \
                             of peak memory on dense + MoE qwen35 sources)"
                        );
                        Some(calibrate::calibrator::CalibrationData::Dwq(map))
                    }
                    _ => None,
                },
                Err(_) => None,
            };

        match pre_clone_cache_hit {
            Some(data) => data,
            None => {
                // Cache MISS — fall through to the original clone + calibrate
                // flow. This path still pays the 70 GB clone for 35BMOE on
                // first-time conversion; the OOM mitigation lives in iter-104+
                // (per-tensor Arc-shared lazy_view) but cache HIT post-prime
                // makes that a one-time cost.
                let lazy_view = if needs_dwq_lazy_capture {
                    clone_tensor_map_to_lazy(&tensor_map)
                } else {
                    ir::lazy::LazyTensorMap::from_eager_borrowed(&tensor_map)
                };
                tracing::info!(
                    calibrator = %calibrator.name(),
                    n_chunks = corpus.n_chunks(),
                    tokens = corpus.total_tokens(),
                    "ADR-014 P4: running calibrator.calibrate() (cache MISS path)"
                );
                calibrator
                    .calibrate(&lazy_view, &metadata, &corpus, &progress)
                    .map_err(|e| {
                        AppError::Conversion(anyhow::anyhow!(
                            "ADR-014 P4: calibrator.calibrate() failed for --quant {}: {e:#}",
                            config.quant
                        ))
                    })?
            }
        }
    } else {
        calibrate::calibrator::CalibrationData::None
    };

    // ADR-014 P4 iter-94 (2026-04-28): explicit Phase 2 → Phase 3 memory hand-off.
    //
    // The DWQ qwen35/qwen35moe path holds a `RealActivationCapture` (boxed
    // `Qwen35Model` with all weights uploaded to GPU + F32-expanded host
    // copies via `load_lazy_dense_ffn`'s `load_lazy_f32`) inside `calibrator`.
    // For Qwen3.6 27B that's ~52 GB of resident state on top of the 52 GB
    // `tensor_map` Phase 3 still needs.
    //
    // Without this drop, the implicit drop happens at function-end (after
    // Phase 3 + Phase 4 + Phase 4.5 + Phase 4.6 all run), so the calibrator's
    // Qwen35Model + GPU buffers stay alive across the entire quantize phase.
    // iter-93 measured peak memory = 158 GB (jetsam SIGKILL on M5 Max 128 GB)
    // for `27b-dwq46`; ADR-012's reference emission Apr 26 recorded ≤ 64 GB.
    // The 2.5× regression is dominated by this `calibrator` lifetime — the
    // model is dead state after `calibrate()` returns `CalibrationData`.
    //
    // Explicit `drop(calibrator)` here releases the GPU+host weights at the
    // exact phase boundary they become unreachable from the convert pipeline.
    // Saves ~52 GB peak on 27B dense, ~70 GB on 35B-MoE.
    drop(calibrator);

    let quantized_model = if backend.requires_native_quantization() {
        tracing::info!("Skipping IR quantization (native backend handles quantization)");
        None
    } else {
        Some(match config.quant {
            // ADR-014 P8 Decision 12: K-quant variants (uncalibrated and
            // imatrix-calibrated) route through the unified
            // KQuantCodecQuantizer — final GGUF block bytes in one pass,
            // no IR-quantize → repack indirection.
            cli::QuantMethod::Q2KS
            | cli::QuantMethod::Q2K
            | cli::QuantMethod::Q3KS
            | cli::QuantMethod::Q3KM
            | cli::QuantMethod::Q3KL
            | cli::QuantMethod::Q4KS
            | cli::QuantMethod::Q4KM
            | cli::QuantMethod::Q5KS
            | cli::QuantMethod::Q5KM
            | cli::QuantMethod::Q6K
            | cli::QuantMethod::ImatrixQ2KS
            | cli::QuantMethod::ImatrixQ2K
            | cli::QuantMethod::ImatrixQ3KS
            | cli::QuantMethod::ImatrixQ3KM
            | cli::QuantMethod::ImatrixQ3KL
            | cli::QuantMethod::ImatrixQ4KS
            | cli::QuantMethod::ImatrixQ4KM
            | cli::QuantMethod::ImatrixQ5KS
            | cli::QuantMethod::ImatrixQ5KM
            | cli::QuantMethod::ImatrixQ6K => {
                let target = match config.quant {
                    cli::QuantMethod::Q2KS
                    | cli::QuantMethod::Q2K
                    | cli::QuantMethod::ImatrixQ2KS
                    | cli::QuantMethod::ImatrixQ2K => {
                        quantize::k_quant_codec::KQuantTarget::Q2K
                    }
                    cli::QuantMethod::Q3KS
                    | cli::QuantMethod::Q3KM
                    | cli::QuantMethod::Q3KL
                    | cli::QuantMethod::ImatrixQ3KS
                    | cli::QuantMethod::ImatrixQ3KM
                    | cli::QuantMethod::ImatrixQ3KL => {
                        quantize::k_quant_codec::KQuantTarget::Q3K
                    }
                    cli::QuantMethod::Q4KS | cli::QuantMethod::Q4KM
                    | cli::QuantMethod::ImatrixQ4KS | cli::QuantMethod::ImatrixQ4KM => {
                        quantize::k_quant_codec::KQuantTarget::Q4K
                    }
                    cli::QuantMethod::Q5KS | cli::QuantMethod::Q5KM
                    | cli::QuantMethod::ImatrixQ5KS | cli::QuantMethod::ImatrixQ5KM => {
                        quantize::k_quant_codec::KQuantTarget::Q5K
                    }
                    cli::QuantMethod::Q6K | cli::QuantMethod::ImatrixQ6K => {
                        quantize::k_quant_codec::KQuantTarget::Q6K
                    }
                    _ => {
                        // Outer match arm guarantees one of the cases
                        // above. Use a typed AppError rather than
                        // `unreachable!()` to honour the no-panic
                        // contract added by ADR-014 P2 iter-2.
                        return Err(AppError::Conversion(anyhow::anyhow!(
                            "ADR-014 P2 iter-2: K-quant codec target dispatch \
                             reached unreachable arm for --quant {}",
                            config.quant
                        )));
                    }
                };

                // ADR-014 P2 iter-2 §S2: feed real CalibrationData from the
                // calibrator into the codec. For uncalibrated K-quant
                // variants the calibrator was NoneCalibrator, returning
                // CalibrationData::None — preserving the uncalibrated
                // codec path (byte-identical to iter-1 for Q4KM/Q5KM/Q6K
                // — see §T6 byte-identity gate).
                let kq = quantize::k_quant_codec_quantizer::KQuantCodecQuantizer::new(
                    quant_method_str.clone(),
                    target,
                    calibration_data.clone(),
                );

                tracing::info!(
                    quant = %config.quant,
                    target = ?target,
                    calibration = ?calibration_data.is_some(),
                    "K-quant codec quantizer dispatched (ADR-014 P8 Decision 12 + P2 iter-2)"
                );

                // ADR-014 P7 iter-48 (2026-04-28): env-flag-gated
                // exercise of the streaming code path in production.
                // `HF2Q_STREAMING_PHASE3=1` routes through
                // `quantize_via_streaming_borrowed` which is
                // byte-identical to `quantize_model` per iter-47/48
                // gates.  This is a TEST INTEGRATION not a memory win
                // — the borrowed wedge holds 2× peak briefly during
                // the per-tensor clone.  The actual memory savings
                // land when iter-3 surgery removes the downstream
                // `tensor_map` dependency.  Default OFF.
                //
                // ADR-014 P7 iter-84 (2026-04-28): NEW env flag
                // `HF2Q_STREAMING_PHASE3_MUT=1` routes through
                // `quantize_via_streaming_consuming_mut` (iter-83) —
                // ACTUAL zero-byte-copy: drains tensor_map's bytes via
                // `take_data_as_arc` (iter-82), no clone, no peak
                // inflation.  Caveat: post-call tensor_map data is
                // empty.  Phase 4.5 quality (iter-52) builds its own
                // lazy_map upfront from `tensor_map` — when
                // HF2Q_STREAMING_PHASE3_MUT is on, Phase 4.5 must
                // ALSO run to completion BEFORE Phase 3 wedge fires,
                // OR fall back to skip-quality mode.  This iter wires
                // the dispatch only on the K-quant codec arm; iter-85+
                // will handle the Phase 4.5 ordering.
                //
                // Precedence: STREAMING_PHASE3_MUT > STREAMING_PHASE3 > eager.
                // ADR-014 P7 iter-90: extracted into dispatch_phase3_quantize.
                dispatch_phase3_quantize(
                    &mut tensor_map,
                    &metadata,
                    &kq,
                    bits,
                    config.group_size,
                    &progress,
                    "kquant_codec",
                )
                .context("K-quant codec quantization failed")
                .map_err(AppError::Conversion)?
            }
            // ADR-014 P8 Decision 12: imatrix-adaptive routes through
            // VariantKQuantizer with per-tensor target dispatch via
            // layer_mix::target_for. Preserves the per-tensor optimal
            // precision behavior previously exposed as `--quant apex`.
            cli::QuantMethod::ImatrixAdaptive => {
                let n_layers = metadata.num_layers as usize;
                let vq = quantize::variant_quantizer::VariantKQuantizer::new(
                    quantize::layer_mix::KQuantVariant::Q4_K_M,
                    calibration_data.clone(),
                    n_layers,
                );

                tracing::info!(
                    quant = %config.quant,
                    n_layers,
                    calibration = ?calibration_data.is_some(),
                    "imatrix-adaptive: VariantKQuantizer dispatched (preserves Apex per-tensor optimal precision)"
                );

                // ADR-014 P7 iter-49 (2026-04-28): same env-flag-gated
                // streaming wire-up as iter-48's K-quant codec arm.
                // Byte-identity to `quantize_model` proven by iter-47/48
                // wedge gates; VariantKQuantizer dispatch is a strict
                // superset of KQuantCodecQuantizer dispatch (per-tensor
                // routing via layer_mix), so the same wedge applies.
                //
                // ADR-014 P7 iter-90: extracted into dispatch_phase3_quantize
                // (precedence: MUT > legacy > eager).
                dispatch_phase3_quantize(
                    &mut tensor_map,
                    &metadata,
                    &vq,
                    bits,
                    config.group_size,
                    &progress,
                    "imatrix_adaptive",
                )
                .context("imatrix-adaptive (variant K-quant) quantization failed")
                .map_err(AppError::Conversion)?
            }
            cli::QuantMethod::Dwq46
            | cli::QuantMethod::Dwq48
            | cli::QuantMethod::Dwq68
            | cli::QuantMethod::Dwq28 => {
                // ADR-014 P4: DWQ capture runs above through
                // DwqCalibrator::with_activation_capture_lazy, directly
                // from the resident tensor map. Convert the
                // calibrator's `CalibrationData::Dwq` per-layer flag map
                // into the `DwqConfig.sensitive_layers` shape so the
                // downstream byte-emit knows which layers are sensitive.
                let (base_bits, sensitive_bits) =
                    config.quant.dwq_bit_pair().ok_or_else(|| {
                        AppError::Conversion(anyhow::anyhow!(
                            "ADR-014 P2 iter-2: dwq_bit_pair() returned None for \
                             DWQ variant {}; cli::QuantMethod and dwq_bit_pair are \
                             out of sync (defect, not a runtime fallback path)",
                            config.quant
                        ))
                    })?;
                let derived_ranges = dwq_calibration_to_sensitive_ranges(&calibration_data);
                // Activation-derived ranges win over CLI --sensitive-layers
                // when the calibrator returned non-empty data (qwen35*
                // path); otherwise (DwqArch::Other / weight-space) we
                // honour the CLI selection so existing behaviour for
                // non-qwen35 DWQ is preserved bit-for-bit.
                let final_sensitive_ranges = if !derived_ranges.is_empty() {
                    derived_ranges
                } else {
                    config.sensitive_layers.clone()
                };

                // ADR-014 P11-prereq Iter C (2026-04-27): default DWQ
                // emit switches from the legacy `MixedBitQuantizer`
                // (Q4_0-family bytes) to `DwqKQuantizer` (Q4_K_M-family
                // bytes via `KQuantCodecQuantizer`).  The legacy path
                // remains available via `HF2Q_USE_LEGACY_DWQ_Q4_0=1`
                // for benchmarking comparison ONLY — no silent
                // fallback; the env var is the single switch and is
                // logged at INFO when set.  The replacement is
                // in-place: the dispatch arm below preserves the
                // resulting `QuantizedModel.quant_method` tag
                // (`dwq-mixed-{base}-{sens}`) so downstream
                // discriminators (`safetensors_out::is_dwq_method`,
                // ADR-012 D22 ship list) keep matching exactly.
                let use_legacy_dwq = std::env::var("HF2Q_USE_LEGACY_DWQ_Q4_0")
                    .map(|v| v == "1")
                    .unwrap_or(false);

                let legacy_quant_method =
                    format!("dwq-mixed-{}-{}", base_bits, sensitive_bits);

                if use_legacy_dwq {
                    let dwq_config = calibrate::dwq::DwqConfig {
                        calibration_samples: config.calibration_samples,
                        sensitive_layers: final_sensitive_ranges.clone(),
                        group_size: config.group_size,
                        base_bits,
                        sensitive_bits,
                        use_activations: dwq_arch_for_dispatch
                            .requires_activation_capture(),
                        arch: dwq_arch_for_dispatch,
                        ..calibrate::dwq::DwqConfig::default()
                    };
                    tracing::info!(
                        quant = %config.quant,
                        arch = ?dwq_arch_for_dispatch,
                        sensitive_ranges = final_sensitive_ranges.len(),
                        base_bits,
                        sensitive_bits,
                        "DWQ byte-emit via LEGACY MixedBitQuantizer (Q4_0-family) — \
                         HF2Q_USE_LEGACY_DWQ_Q4_0=1 (ADR-014 P11-prereq Iter C \
                         escape hatch; default switched to DwqKQuantizer \
                         Q4_K_M-family)"
                    );
                    calibrate::dwq_activation::run_dwq_with_sensitive_ranges(
                        &tensor_map,
                        &metadata,
                        &dwq_config,
                        final_sensitive_ranges,
                        &progress,
                    )
                    .map_err(|e| {
                        AppError::Conversion(anyhow::anyhow!(
                            "ADR-014 P2 iter-2: legacy DWQ byte-emit failed: {e:#}"
                        ))
                    })?
                } else {
                    // Default path (ADR-014 P11-prereq Iter C): route
                    // through `DwqKQuantizer` so each tensor emits final
                    // K-quant block bytes via `KQuantCodecQuantizer`.
                    // The variant maps to (base, sensitive) targets
                    // exactly per Iter B's `DwqKVariant`:
                    //
                    //   dwq46 → (Q4_K, Q6_K)
                    //   dwq48 → (Q4_K, Q8_0)
                    //   dwq68 → (Q6_K, Q8_0)
                    //   dwq28 → (Q2_K [deferred], Q8_0)
                    //
                    // `dwq28` base-bucket tensors surface a typed
                    // `QuantizeError::TensorQuantizeFailed` from the
                    // codec until Q2_K codec lands; this propagates up
                    // as `AppError::Conversion` — no panic, no silent
                    // fallback (per Iter B's contract; `dwq28` users
                    // wait for the codec port).
                    let variant = match config.quant {
                        cli::QuantMethod::Dwq46 => {
                            quantize::dwq_k_quantizer::DwqKVariant::P46
                        }
                        cli::QuantMethod::Dwq48 => {
                            quantize::dwq_k_quantizer::DwqKVariant::P48
                        }
                        cli::QuantMethod::Dwq68 => {
                            quantize::dwq_k_quantizer::DwqKVariant::P68
                        }
                        cli::QuantMethod::Dwq28 => {
                            quantize::dwq_k_quantizer::DwqKVariant::P28
                        }
                        // Outer match arm guarantees one of the four
                        // DWQ variants. Surface a typed error rather
                        // than `unreachable!()` to honour the no-panic
                        // contract added by ADR-014 P2 iter-2.
                        _ => {
                            return Err(AppError::Conversion(anyhow::anyhow!(
                                "ADR-014 P11-prereq Iter C: DwqKVariant \
                                 dispatch reached unreachable arm for \
                                 --quant {}",
                                config.quant
                            )));
                        }
                    };

                    // ADR-014 P11-prereq Iter C — Sensitivity → Imatrix
                    // wiring DEFERRED (documented in
                    // `dwq_k_quantizer.rs` module doc + ADR-014 Iter C
                    // closure section).  Reason: `DwqConfig` exposes
                    // only a per-layer scalar sensitivity, while
                    // `CalibrationData::Imatrix` requires per-column
                    // vectors of length `row_len` — the two shapes are
                    // dimensionally incompatible.  Bridging requires
                    // either (a) a separate `ImatrixCollector` pass
                    // alongside DWQ, or (b) upstream broadcast (which
                    // degenerates to `_ref` since uniform weights ≡ no
                    // weights).  Both push past the 250 LOC HARD-STOP
                    // for this iter.  Passing `CalibrationData::None`
                    // routes the codebook search through the `_ref`
                    // (non-imatrix-weighted) path — same algorithmic
                    // surface as today's `--quant q4_k_m` (uncalibrated
                    // `KQuantCodecQuantizer` cell), which already
                    // ships and is gate-locked by Iter A's regression
                    // tests.  A future iter lands path (a).
                    let dwq_k = quantize::dwq_k_quantizer::DwqKQuantizer::new(
                        variant,
                        &final_sensitive_ranges,
                        calibrate::calibrator::CalibrationData::None,
                    );

                    tracing::info!(
                        quant = %config.quant,
                        arch = ?dwq_arch_for_dispatch,
                        sensitive_ranges = final_sensitive_ranges.len(),
                        base_bits,
                        sensitive_bits,
                        codec = "DwqKQuantizer (K-quant codec-direct)",
                        "DWQ byte-emit via DwqKQuantizer dispatch \
                         (ADR-014 P11-prereq Iter C — Q4_K_M-family default)"
                    );

                    // Drive `quantize::quantize_model` so per-tensor
                    // preserve flags + vision-tensor handling stay
                    // consistent with the K-quant uncalibrated cell
                    // above.  Override the resulting `quant_method`
                    // tag back to the legacy `dwq-mixed-{b}-{s}` form
                    // so downstream consumers
                    // (`safetensors_out::is_dwq_method`, ADR-012 D22
                    // ship-list filename suffix at
                    // `default_filename_suffix`) keep matching exactly
                    // (Chesterton's fence — we replace the codec, not
                    // the discriminator surface).
                    //
                    // ADR-014 P7 iter-51 (2026-04-28): same env-flag-gated
                    // streaming wire-up as iter-48/49/50 — completes the
                    // four-arm Phase 3 migration.  DwqKQuantizer routes
                    // each tensor through `target_for(name)` then
                    // `KQuantCodecQuantizer`, so the streaming dispatch
                    // is byte-identical (proven by the iter-47/48 wedge
                    // gates).  legacy_quant_method override is preserved
                    // post-quantize regardless of which path emitted the
                    // bytes.
                    // ADR-014 P7 iter-90: extracted into dispatch_phase3_quantize
                    // (precedence: MUT > legacy > eager). legacy_quant_method
                    // override applies regardless of path.
                    let mut quantized = dispatch_phase3_quantize(
                        &mut tensor_map,
                        &metadata,
                        &dwq_k,
                        bits,
                        config.group_size,
                        &progress,
                        "dwq_k",
                    )
                    .map_err(|e| {
                        AppError::Conversion(anyhow::anyhow!(
                            "ADR-014 P11-prereq Iter C: DwqKQuantizer \
                             byte-emit failed: {e:#}"
                        ))
                    })?;
                    quantized.quant_method = legacy_quant_method;
                    quantized
                }
            }
            // ADR-014 P8 Decision 12: flat / legacy block-quant variants
            // (f16, bf16, q2, q4, q8) and Auto (post-resolution) flow
            // through the existing StaticQuantizer dispatch. Auto is
            // included because resolve_convert_config rewrites
            // `config.quant` to a non-Auto value before this match runs;
            // listing it explicitly preserves the exhaustive-match
            // contract (Decision 12 lock).
            cli::QuantMethod::Auto
            | cli::QuantMethod::F16
            | cli::QuantMethod::Bf16
            | cli::QuantMethod::Q2
            | cli::QuantMethod::Q4
            | cli::QuantMethod::Q8 => {
                let quantizer = StaticQuantizer::new(&quant_method_str)
                    .context("Failed to create quantizer")
                    .map_err(AppError::Conversion)?;

                // ADR-014 P7 iter-50 (2026-04-28): same env-flag-gated
                // streaming wire-up as iter-48/49.  StaticQuantizer is
                // the catch-all for non-K-quant non-DWQ methods (F16,
                // BF16, Q2/Q4/Q8 legacy).  All paths through the
                // StaticQuantizer's `quantize_tensor` impl are
                // deterministic + Send+Sync, so streaming dispatch is
                // byte-identical (proven by iter-47's
                // `quantize_streaming_byte_identical_to_quantize_model`
                // which uses StaticQuantizer in the smoke fixture).
                // ADR-014 P7 iter-90: extracted into dispatch_phase3_quantize
                // (precedence: MUT > legacy > eager).
                dispatch_phase3_quantize(
                    &mut tensor_map,
                    &metadata,
                    &quantizer,
                    bits,
                    config.group_size,
                    &progress,
                    "static_quantizer",
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
        // ADR-014 P7 iter-52 (2026-04-28): env-flag-gated wire-up of
        // `measure_quality_streaming_lazy` (iter-43).  Builds a parallel
        // LazyTensorMap from the resident `tensor_map` for the quality
        // measurement.  Byte-identity to the eager variant proven by
        // iter-43's `measure_quality_streaming_lazy_matches_eager`
        // gate.  Default OFF.  Memory profile during this iter:
        // unchanged from eager (the lazy_map clones bytes per-tensor).
        // The actual memory-budget win lands when iter-3 wholesale
        // removes `materialize_all()` and sources `lazy_map` directly
        // from `read_tensors_lazy`, bypassing eager materialise.
        let streaming_phase3 =
            std::env::var("HF2Q_STREAMING_PHASE3").as_deref() == Ok("1");
        if streaming_phase3 {
            tracing::info!("ADR-014 P7 iter-52: HF2Q_STREAMING_PHASE3=1 → measure_quality_streaming_lazy");
            // Build a LazyTensorMap parallel view from the eager
            // tensor_map (per-tensor `from_bytes` clone — same memory
            // profile as the iter-48 wedge).
            let mut lazy_map = ir::lazy::LazyTensorMap::new();
            for (_, t) in tensor_map.tensors.iter() {
                let meta =
                    ir::lazy::LazyMeta::new(t.name.clone(), t.shape.clone(), t.dtype);
                lazy_map.insert(ir::lazy::LazyTensor::from_arc_bytes(meta, std::sync::Arc::clone(&t.data)));
            }
            match quality::measure_quality_streaming_lazy(
                &lazy_map,
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
                    warn!("Quality measurement (lazy) failed: {}. Continuing.", e);
                    quality::QualityReport::empty()
                }
            }
        } else {
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

    // Phase 4.85: ADR-005 iter-211b — patch the mmproj_sha256 placeholder
    // in the main GGUF with the real SHA of the sibling mmproj GGUF.
    //
    // The placeholder was emitted during Phase 4.6 when vision tensors
    // were detected in the source LazyTensorMap (see iter-211b plumbing
    // above where `provenance.mmproj_sha256` is set). The mmproj sibling
    // is now on disk (either via the legacy `GgufBackend::write` path or
    // the ADR-012 P10 path above). Compute its SHA, patch the main GGUF
    // in place, eliminating the iter-211 "todo later" stub.
    //
    // Path resolution: scan the output dir for files ending in
    // `-mmproj.gguf` or `mmproj-*.gguf` (P10 alternate naming). The
    // P10 path deleted the legacy file above, so at most one mmproj
    // remains per the sovereignty rule. The main GGUF is the single
    // text file in the manifest.
    if placeholder_emitted {
        let output_parent = std::path::Path::new(&manifest.output_dir);

        // Find the main GGUF (the one that does NOT end in -mmproj.gguf).
        let main_gguf_path = manifest
            .files
            .iter()
            .find(|f| {
                let n = &f.filename;
                n.ends_with(".gguf")
                    && !n.ends_with("-mmproj.gguf")
                    && !n.starts_with("mmproj-")
            })
            .map(|f| output_parent.join(&f.filename));

        // Find the mmproj GGUF on disk (legacy or P10 naming).
        let mmproj_path: Option<std::path::PathBuf> = std::fs::read_dir(output_parent)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| {
                        (n.ends_with("-mmproj.gguf") || n.starts_with("mmproj-"))
                            && n.ends_with(".gguf")
                    })
                    .unwrap_or(false)
            });

        match (main_gguf_path, mmproj_path) {
            (Some(main_p), Some(mmproj_p)) => {
                let mmproj_sha = serve::cache::compute_file_sha256(&mmproj_p)
                    .with_context(|| {
                        format!(
                            "iter-211b: failed to compute SHA-256 of mmproj at {}",
                            mmproj_p.display()
                        )
                    })
                    .map_err(AppError::Conversion)?;
                backends::gguf::patch_mmproj_sha256_in_gguf(&main_p, &mmproj_sha)
                    .with_context(|| {
                        format!(
                            "iter-211b: failed to patch hf2q.mmproj_sha256 placeholder in {}",
                            main_p.display()
                        )
                    })
                    .map_err(AppError::Conversion)?;
                tracing::info!(
                    "ADR-005 iter-211b: patched hf2q.mmproj_sha256 in {} with mmproj SHA {} (from {})",
                    main_p.display(),
                    mmproj_sha,
                    mmproj_p.display()
                );
            }
            (None, _) => {
                return Err(AppError::Conversion(anyhow::anyhow!(
                    "iter-211b: placeholder was emitted but no main GGUF found in manifest \
                     (output_dir={}); cannot resolve mmproj_sha256",
                    manifest.output_dir
                )));
            }
            (Some(_), None) => {
                // Vision tensors were present at metadata-write time but
                // no mmproj landed on disk — refuse the silent leftover
                // placeholder. This is a structural inconsistency
                // (vision-tensor detection vs actual mmproj emission)
                // and the operator should see it loudly per the
                // Engineering Mantra "no fallback".
                return Err(AppError::Conversion(anyhow::anyhow!(
                    "iter-211b: hf2q.mmproj_sha256 placeholder was emitted (vision tensors \
                     detected in LazyTensorMap) but no mmproj GGUF was produced in {}. \
                     This indicates a mismatch between vision-tensor detection at metadata-write \
                     time and the actual emit path. The placeholder MUST not survive in the \
                     emitted GGUF — either fix the detector or fix the emitter.",
                    manifest.output_dir
                )));
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
        // ADR-014 P9 iter-1 §S2 — the safetensors backend owns the
        // mutation of `config.json` (mlx-lm `quantization` injection
        // per `mlx_lm.utils.save_config:912-913`). When the backend
        // already produced a file at this path, do not clobber it
        // with a byte-copy from the source — the injected config is
        // load-bearing for downstream mlx-lm consumers.
        if dst.exists() {
            tracing::debug!(
                file = %filename,
                "Sidecar already present at destination — skipping copy \
                 (backend-written file takes precedence)"
            );
            continue;
        }
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

    // ADR-014 P8 Decision 12: detect the 17-variant menu strings.
    // Long forms before short to avoid substring collisions (e.g.
    // "imatrix-q4_k_m" matched before "q4_k_m"; "dwq-4-6" matched
    // before "dwq46"; "q4_k_m" matched before "q4").
    let known = [
        // Imatrix-calibrated K-quant (longest first)
        "imatrix-q4_k_m", "imatrix-q5_k_m", "imatrix-q6_k", "imatrix-adaptive",
        // Uncalibrated K-quant
        "q4_k_m", "q5_k_m", "q6_k",
        // DWQ kebab forms
        "dwq-4-6", "dwq-4-8", "dwq-6-8", "dwq-2-8",
        // DWQ filename suffixes
        "dwq46", "dwq48", "dwq68", "dwq28",
        // Flat / legacy block-quant variants
        "bf16", "f16",
        "q2", "q4", "q8",
    ];
    for method in &known {
        if name.contains(method) {
            return method.to_string();
        }
    }
    "unknown".to_string()
}

/// Get the default bit width for a quantization method.
///
/// Exhaustive over the 17-variant menu (Decision 12 lock).  K-quant
/// variants report the *base* target bit width (Q4_K=4, Q5_K=5,
/// Q6_K=6); per-tensor `_M` upgrades and imatrix-adaptive routing are
/// applied inside the quantizer, not at this dispatch level.
fn quantizer_default_bits(method: &cli::QuantMethod) -> u8 {
    match method {
        cli::QuantMethod::Auto => 4, // post-resolution this is overwritten
        cli::QuantMethod::F16 | cli::QuantMethod::Bf16 => 16,
        cli::QuantMethod::Q8 => 8,
        cli::QuantMethod::Q4 => 4,
        cli::QuantMethod::Q2 => 2,
        cli::QuantMethod::Q2KS | cli::QuantMethod::ImatrixQ2KS => 2,
        cli::QuantMethod::Q2K | cli::QuantMethod::ImatrixQ2K => 2,
        cli::QuantMethod::Q3KS | cli::QuantMethod::ImatrixQ3KS => 3,
        cli::QuantMethod::Q3KM | cli::QuantMethod::ImatrixQ3KM => 3,
        cli::QuantMethod::Q3KL | cli::QuantMethod::ImatrixQ3KL => 3,
        cli::QuantMethod::Q4KS | cli::QuantMethod::ImatrixQ4KS => 4,
        cli::QuantMethod::Q4KM | cli::QuantMethod::ImatrixQ4KM => 4,
        cli::QuantMethod::Q5KS | cli::QuantMethod::ImatrixQ5KS => 5,
        cli::QuantMethod::Q5KM | cli::QuantMethod::ImatrixQ5KM => 5,
        cli::QuantMethod::Q6K | cli::QuantMethod::ImatrixQ6K => 6,
        cli::QuantMethod::ImatrixAdaptive => 4, // Q4_K_M base
        cli::QuantMethod::Dwq46 => 4,
        cli::QuantMethod::Dwq48 => 4,
        cli::QuantMethod::Dwq68 => 6,
        cli::QuantMethod::Dwq28 => 2,
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
