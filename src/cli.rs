//! CLI argument parsing and validation via clap derive API.
//!
//! All environment variable resolution happens here at startup.
//! No global state — CLI args produce a `ConvertConfig` that flows through the pipeline.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use clap_complete::Shell;

/// hf2q — Pure Rust CLI for converting HuggingFace models to hardware-optimized formats.
#[derive(Parser, Debug)]
#[command(name = "hf2q", version, about, long_about = None)]
pub struct Cli {
    /// Increase logging verbosity (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Convert a HuggingFace model to a hardware-optimized format
    Convert(ConvertArgs),

    /// Inspect model metadata before converting
    Info(InfoArgs),

    /// Validate quality of a quantized model against its original
    Validate(ValidateArgs),

    /// Diagnose RuVector, hardware detection, and disk space
    Doctor,

    /// Generate shell completions
    Completions(CompletionsArgs),

    /// Run text generation from a GGUF model
    Generate(GenerateArgs),

    /// Serve a GGUF model via OpenAI-compatible HTTP API
    Serve(ServeArgs),
}

#[derive(clap::Args, Debug)]
pub struct ConvertArgs {
    /// Local safetensors directory
    #[arg(long, conflicts_with = "repo")]
    pub input: Option<PathBuf>,

    /// HuggingFace repo ID (downloads automatically)
    #[arg(long, conflicts_with = "input")]
    pub repo: Option<String>,

    /// Output target format
    #[arg(long, value_enum)]
    pub format: OutputFormat,

    /// Quantization method (default: auto)
    #[arg(long, value_enum, default_value = "auto")]
    pub quant: QuantMethod,

    /// Layer ranges to protect at higher precision (e.g., "13-24", "1,5,13-24")
    #[arg(long)]
    pub sensitive_layers: Option<String>,

    /// Sample count for DWQ calibration
    #[arg(long, default_value = "1024")]
    pub calibration_samples: u32,

    /// Target average bits per weight for Apex quantization (e.g., 4.5)
    #[arg(long, default_value = "4.5")]
    pub target_bpw: f32,

    /// Custom bit width (2-8)
    #[arg(long, value_parser = clap::value_parser!(u8).range(2..=8))]
    pub bits: Option<u8>,

    /// Custom group size
    #[arg(long, value_enum)]
    pub group_size: Option<GroupSize>,

    /// Output path: .gguf file for GGUF format, directory for safetensors
    #[arg(long, short)]
    pub output: Option<PathBuf>,

    /// Emit structured JSON report for CI/automation
    #[arg(long)]
    pub json_report: bool,

    /// Skip KL divergence / perplexity measurement
    #[arg(long)]
    pub skip_quality: bool,

    /// Fail with exit code 2 if quality thresholds are exceeded
    #[arg(long)]
    pub quality_gate: bool,

    /// Run preflight + auto resolution, print plan, exit without converting
    #[arg(long)]
    pub dry_run: bool,

    /// Non-interactive mode — skip all confirmation prompts
    #[arg(long)]
    pub yes: bool,

    /// How to handle unsupported layer types
    #[arg(long, value_enum)]
    pub unsupported_layers: Option<UnsupportedLayerPolicy>,
}

#[derive(clap::Args, Debug)]
pub struct InfoArgs {
    /// Local safetensors directory
    #[arg(long, conflicts_with = "repo")]
    pub input: Option<PathBuf>,

    /// HuggingFace repo ID
    #[arg(long, conflicts_with = "input")]
    pub repo: Option<String>,
}

#[derive(clap::Args, Debug)]
pub struct CompletionsArgs {
    /// Shell to generate completions for
    #[arg(long, value_enum)]
    pub shell: Shell,
}

#[derive(clap::Args, Debug)]
pub struct ValidateArgs {
    /// Directory containing the original model
    #[arg(long)]
    pub original: PathBuf,

    /// Directory containing the quantized model
    #[arg(long)]
    pub quantized: PathBuf,

    /// Maximum KL divergence threshold
    #[arg(long, default_value = "0.1")]
    pub max_kl: f64,

    /// Maximum perplexity delta threshold
    #[arg(long, default_value = "2.0")]
    pub max_ppl_delta: f64,

    /// Minimum cosine similarity threshold
    #[arg(long, default_value = "0.95")]
    pub min_cosine: f64,

    /// Emit JSON output
    #[arg(long)]
    pub json: bool,
}

#[derive(clap::Args, Debug)]
pub struct GenerateArgs {
    /// Path to GGUF model file
    #[arg(long)]
    pub model: PathBuf,

    /// Path to tokenizer.json (if not alongside GGUF)
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

    /// Path to config.json (if not alongside GGUF)
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Prompt text (required unless --prompt-file is given)
    #[arg(long, required_unless_present = "prompt_file")]
    pub prompt: Option<String>,

    /// Path to file containing prompt text
    #[arg(long, conflicts_with = "prompt")]
    pub prompt_file: Option<PathBuf>,

    /// Run benchmark mode: 5 consecutive runs, report median and p95 tok/s
    #[arg(long)]
    pub benchmark: bool,

    /// Maximum tokens to generate
    #[arg(long, default_value = "256")]
    pub max_tokens: usize,

    /// Sampling temperature (0.0 = greedy)
    #[arg(long, default_value = "0.7")]
    pub temperature: f64,

    /// Top-p nucleus sampling
    #[arg(long, default_value = "0.9")]
    pub top_p: f64,

    /// Top-k sampling (0 = disabled)
    #[arg(long, default_value = "50")]
    pub top_k: usize,

    /// Repetition penalty (1.0 = disabled)
    #[arg(long, default_value = "1.0")]
    pub repetition_penalty: f64,

    /// Override chat template with a Jinja2 string
    ///
    /// Priority order (per ADR-005 Phase 1): this flag > --chat-template-file >
    /// GGUF `tokenizer.chat_template` metadata > hardcoded fallback.
    #[arg(long)]
    pub chat_template: Option<String>,

    /// Override chat template by reading from a file containing a Jinja2 template
    #[arg(long, conflicts_with = "chat_template")]
    pub chat_template_file: Option<PathBuf>,

    /// MoE expert dispatch mode. `loop` (default) keeps the Phase-1 baseline
    /// per-expert `QMatMul::forward` loop with two forced `to_vec2()` routing
    /// syncs per layer. `fused` routes layer 0 only through the fused
    /// `kernel_mul_mv_id_*` path behind the ADR-005 1bNEW.1 Phase B gate; once
    /// Phase C lands the same path sweeps every layer. Default stays on
    /// `loop` until 1bNEW.1 completes all four phases.
    #[arg(long, value_enum, default_value = "loop")]
    pub moe_kernel: MoeKernelMode,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeKernelMode {
    /// Phase-1 baseline — per-expert `QMatMul::forward` loop (slow, correct).
    Loop,
    /// ADR-005 1bNEW.1 — fused `kernel_mul_mv_id_*` dispatch.
    Fused,
}

impl std::fmt::Display for MoeKernelMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loop => write!(f, "loop"),
            Self::Fused => write!(f, "fused"),
        }
    }
}

#[derive(clap::Args, Debug)]
pub struct ServeArgs {
    /// Path to GGUF model file
    #[arg(long)]
    pub model: PathBuf,

    /// Path to tokenizer.json (if not alongside GGUF)
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

    /// Path to config.json (if not alongside GGUF)
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to listen on
    #[arg(long, default_value = "8080")]
    pub port: u16,

    /// Maximum sequence length
    #[arg(long, default_value = "4096")]
    pub max_seq_len: usize,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// GGUF format for llama.cpp / Ollama
    Gguf,
    /// Quantized safetensors for inferrs / Candle / vLLM
    Safetensors,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf => write!(f, "gguf"),
            Self::Safetensors => write!(f, "safetensors"),
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantMethod {
    Auto,
    F16,
    Q8,
    Q4,
    Q2,
    #[value(name = "mixed-2-6")]
    Mixed26,
    #[value(name = "mixed-3-6")]
    Mixed36,
    #[value(name = "mixed-4-6")]
    Mixed46,
    #[value(name = "dwq-mixed-4-6")]
    DwqMixed46,
    /// Apex: imatrix-calibrated, per-tensor optimal precision (requires Phase 2 GPU support)
    Apex,
}

impl std::fmt::Display for QuantMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::F16 => write!(f, "f16"),
            Self::Q8 => write!(f, "q8"),
            Self::Q4 => write!(f, "q4"),
            Self::Q2 => write!(f, "q2"),
            Self::Mixed26 => write!(f, "mixed-2-6"),
            Self::Mixed36 => write!(f, "mixed-3-6"),
            Self::Mixed46 => write!(f, "mixed-4-6"),
            Self::DwqMixed46 => write!(f, "dwq-mixed-4-6"),
            Self::Apex => write!(f, "apex"),
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupSize {
    #[value(name = "32")]
    G32,
    #[value(name = "64")]
    G64,
    #[value(name = "128")]
    G128,
}

impl GroupSize {
    pub fn as_usize(self) -> usize {
        match self {
            Self::G32 => 32,
            Self::G64 => 64,
            Self::G128 => 128,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnsupportedLayerPolicy {
    /// Pass unsupported layers through at f16
    Passthrough,
}

/// Resolved configuration for a conversion run.
/// Constructed from CLI args; flows through the entire pipeline.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ConvertConfig {
    /// Path to local model directory (resolved from --input or --repo download)
    pub input_dir: PathBuf,
    /// Output format
    pub format: OutputFormat,
    /// Quantization method
    pub quant: QuantMethod,
    /// Sensitive layer ranges (parsed from "13-24" style strings)
    pub sensitive_layers: Vec<std::ops::RangeInclusive<usize>>,
    /// DWQ calibration sample count
    pub calibration_samples: u32,
    /// Explicit bit width override (None = use quant method default)
    pub bits: Option<u8>,
    /// Group size for quantization
    pub group_size: usize,
    /// Output directory
    pub output_dir: PathBuf,
    /// Whether to emit JSON report
    pub json_report: bool,
    /// Whether to skip quality measurement
    pub skip_quality: bool,
    /// Whether to enforce quality gate (exit code 2 on threshold violation)
    pub quality_gate: bool,
    /// Whether this is a dry run
    pub dry_run: bool,
    /// Non-interactive mode
    pub yes: bool,
    /// How to handle unsupported layer types
    pub unsupported_layers: Option<UnsupportedLayerPolicy>,
}

/// Default group size for quantization.
pub const DEFAULT_GROUP_SIZE: usize = 64;

/// Parse a sensitive layers specification like "13-24" or "1,5,13-24" into ranges.
pub fn parse_sensitive_layers(spec: &str) -> anyhow::Result<Vec<std::ops::RangeInclusive<usize>>> {
    let mut ranges = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start_str, end_str)) = part.split_once('-') {
            let start: usize = start_str.trim().parse().map_err(|_| {
                anyhow::anyhow!("Invalid layer number in range '{}': '{}'", part, start_str.trim())
            })?;
            let end: usize = end_str.trim().parse().map_err(|_| {
                anyhow::anyhow!("Invalid layer number in range '{}': '{}'", part, end_str.trim())
            })?;
            if start > end {
                anyhow::bail!(
                    "Invalid layer range '{}': start ({}) must be <= end ({})",
                    part,
                    start,
                    end
                );
            }
            ranges.push(start..=end);
        } else {
            let layer: usize = part.parse().map_err(|_| {
                anyhow::anyhow!("Invalid layer number: '{}'", part)
            })?;
            ranges.push(layer..=layer);
        }
    }
    if ranges.is_empty() {
        anyhow::bail!("Empty sensitive layers specification");
    }
    Ok(ranges)
}

/// Build a ConvertConfig from parsed CLI ConvertArgs.
pub fn resolve_convert_config(args: &ConvertArgs) -> anyhow::Result<ConvertConfig> {
    let input_dir = match (&args.input, &args.repo) {
        (Some(path), None) => {
            if !path.exists() {
                anyhow::bail!("Input directory does not exist: {}", path.display());
            }
            if !path.is_dir() {
                anyhow::bail!("Input path is not a directory: {}", path.display());
            }
            path.clone()
        }
        (None, Some(repo_id)) => {
            // HF Hub download (Epic 3, Story 3.1)
            let progress = crate::progress::ProgressReporter::new();
            crate::input::hf_download::download_model(repo_id, &progress)
                .map_err(|e| anyhow::anyhow!("{}", e))?
        }
        (None, None) => {
            anyhow::bail!("Either --input or --repo must be specified");
        }
        (Some(_), Some(_)) => {
            // clap's conflicts_with should prevent this, but be explicit
            anyhow::bail!("--input and --repo are mutually exclusive");
        }
    };

    let sensitive_layers = match &args.sensitive_layers {
        Some(spec) => parse_sensitive_layers(spec)?,
        None => Vec::new(),
    };

    let group_size = match args.group_size {
        Some(gs) => gs.as_usize(),
        None => DEFAULT_GROUP_SIZE,
    };

    let output_dir = match &args.output {
        Some(p) => p.clone(),
        None => {
            let model_name = input_dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "model".to_string());
            match args.format {
                OutputFormat::Gguf => PathBuf::from(format!("{}-{}.gguf", model_name, args.quant)),
                _ => PathBuf::from(format!("{}-{}-{}", model_name, args.format, args.quant)),
            }
        }
    };

    // Validate quant method is implemented
    match args.quant {
        QuantMethod::Auto => {
            // Auto mode — resolve at conversion time
        }
        QuantMethod::Mixed26 | QuantMethod::Mixed36 | QuantMethod::Mixed46 => {
            // Mixed-bit quantization
        }
        QuantMethod::DwqMixed46 => {
            // DWQ weight-space calibration (no inference needed)
        }
        QuantMethod::Apex => {
            // Apex: imatrix-calibrated, per-tensor optimal precision quantization
        }
        QuantMethod::F16 | QuantMethod::Q8 | QuantMethod::Q4 | QuantMethod::Q2 => {}
    }

    // Both output formats are implemented
    match args.format {
        OutputFormat::Gguf | OutputFormat::Safetensors => {}
    }

    Ok(ConvertConfig {
        input_dir,
        format: args.format,
        quant: args.quant,
        sensitive_layers,
        calibration_samples: args.calibration_samples,
        bits: args.bits,
        group_size,
        output_dir,
        json_report: args.json_report,
        skip_quality: args.skip_quality,
        quality_gate: args.quality_gate,
        dry_run: args.dry_run,
        yes: args.yes,
        unsupported_layers: args.unsupported_layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sensitive_layers_single_range() {
        let ranges = parse_sensitive_layers("13-24").unwrap();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 13..=24);
    }

    #[test]
    fn test_parse_sensitive_layers_multiple() {
        let ranges = parse_sensitive_layers("1,5,13-24").unwrap();
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], 1..=1);
        assert_eq!(ranges[1], 5..=5);
        assert_eq!(ranges[2], 13..=24);
    }

    #[test]
    fn test_parse_sensitive_layers_invalid_range() {
        let result = parse_sensitive_layers("24-13");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_sensitive_layers_empty() {
        let result = parse_sensitive_layers("");
        assert!(result.is_err());
    }

    #[test]
    fn test_group_size_values() {
        assert_eq!(GroupSize::G32.as_usize(), 32);
        assert_eq!(GroupSize::G64.as_usize(), 64);
        assert_eq!(GroupSize::G128.as_usize(), 128);
    }
}
