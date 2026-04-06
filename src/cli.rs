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

    /// Diagnose RuVector, hardware detection, mlx-rs, and disk space
    Doctor,

    /// Generate shell completions
    Completions(CompletionsArgs),
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

    /// Custom bit width (2-8)
    #[arg(long, value_parser = clap::value_parser!(u8).range(2..=8))]
    pub bits: Option<u8>,

    /// Custom group size
    #[arg(long, value_enum)]
    pub group_size: Option<GroupSize>,

    /// Output directory (default: ./model-{format}-{quant}/)
    #[arg(long, short)]
    pub output: Option<PathBuf>,

    /// Emit structured JSON report for CI/automation
    #[arg(long)]
    pub json_report: bool,

    /// Skip KL divergence / perplexity measurement
    #[arg(long)]
    pub skip_quality: bool,

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

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Mlx,
    Coreml,
    // Future targets — defined now so CLI schema is complete
    Gguf,
    Nvfp4,
    Gptq,
    Awq,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mlx => write!(f, "mlx"),
            Self::Coreml => write!(f, "coreml"),
            Self::Gguf => write!(f, "gguf"),
            Self::Nvfp4 => write!(f, "nvfp4"),
            Self::Gptq => write!(f, "gptq"),
            Self::Awq => write!(f, "awq"),
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
    Q4Mxfp,
    #[value(name = "mixed-2-6")]
    Mixed26,
    #[value(name = "mixed-3-6")]
    Mixed36,
    #[value(name = "mixed-4-6")]
    Mixed46,
    #[value(name = "dwq-mixed-4-6")]
    DwqMixed46,
}

impl std::fmt::Display for QuantMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::F16 => write!(f, "f16"),
            Self::Q8 => write!(f, "q8"),
            Self::Q4 => write!(f, "q4"),
            Self::Q2 => write!(f, "q2"),
            Self::Q4Mxfp => write!(f, "q4-mxfp"),
            Self::Mixed26 => write!(f, "mixed-2-6"),
            Self::Mixed36 => write!(f, "mixed-3-6"),
            Self::Mixed46 => write!(f, "mixed-4-6"),
            Self::DwqMixed46 => write!(f, "dwq-mixed-4-6"),
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
    /// Whether this is a dry run
    pub dry_run: bool,
    /// Non-interactive mode
    pub yes: bool,
    /// How to handle unsupported layer types
    pub unsupported_layers: Option<UnsupportedLayerPolicy>,
}

/// Default group size for quantization.
pub const DEFAULT_GROUP_SIZE: usize = 64;

/// Default number of output shards for MLX format.
pub const DEFAULT_OUTPUT_SHARDS: usize = 4;

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
            PathBuf::from(format!("{}-{}-{}", model_name, args.format, args.quant))
        }
    };

    // Validate quant method is implemented
    match args.quant {
        QuantMethod::Auto => {
            // Auto mode is now implemented (Epic 6) — resolve at conversion time
        }
        QuantMethod::Q4Mxfp => {
            anyhow::bail!(
                "Quantization method '{}' is not yet implemented. \
                 Available methods: auto, f16, q8, q4, q2, mixed-2-6, mixed-3-6, mixed-4-6, dwq-mixed-4-6",
                args.quant
            );
        }
        QuantMethod::Mixed26 | QuantMethod::Mixed36 | QuantMethod::Mixed46 => {
            // Mixed-bit quantization (Epic 5, Story 5.1)
        }
        QuantMethod::DwqMixed46 => {
            // DWQ calibration (Epic 5, Story 5.2) — requires mlx-backend for inference
        }
        QuantMethod::F16 | QuantMethod::Q8 | QuantMethod::Q4 | QuantMethod::Q2 => {}
    }

    // Validate output format is implemented
    match args.format {
        OutputFormat::Mlx | OutputFormat::Coreml => {}
        OutputFormat::Gguf | OutputFormat::Nvfp4 | OutputFormat::Gptq | OutputFormat::Awq => {
            anyhow::bail!(
                "Output format '{}' is not yet implemented. \
                 Available formats: mlx, coreml",
                args.format
            );
        }
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
