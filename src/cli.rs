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

    /// Log output format. `text` (default) emits human-readable colored
    /// output on stderr; `json` emits one JSON object per line (structured
    /// ingest for Loki / Datadog / etc.). ADR-005 Decision #11.
    #[arg(long, value_enum, global = true, default_value = "text")]
    pub log_format: LogFormat,

    /// Explicit log level override. When unset, verbosity (`-v`) controls
    /// the level. When set, this wins and `-v` is ignored. ADR-005
    /// Decision #11.
    #[arg(long, value_enum, global = true)]
    pub log_level: Option<LogLevel>,

    #[command(subcommand)]
    pub command: Command,
}

/// Log output format (Decision #11).
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum LogFormat {
    /// Human-readable colored stderr (ANSI only when stderr is a TTY).
    Text,
    /// One JSON object per log event (structured ingest).
    Json,
}

/// Log level override (Decision #11). When `None`, `-v` flag controls it.
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
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

    /// ADR-009 parity validation against locked references
    Parity(ParityArgs),

    /// ADR-012 Decision 16 — end-gate smoke test for a registered arch
    Smoke(SmokeArgs),

    /// Manage the local model cache (`~/.cache/hf2q/`).
    ///
    /// `hf2q cache list` enumerates cached models. `hf2q cache size`
    /// totals on-disk bytes. `hf2q cache clear` invalidates entries
    /// (single quant, all quants for one model, or the entire cache).
    /// All clear operations atomically rewrite the cache manifest;
    /// concurrent serves observe coherent before/after state.
    Cache(CacheArgs),
}

/// Top-level args for `hf2q cache`. The actual surface is split into
/// `CacheAction` subcommands (clap convention shared with `parity`).
#[derive(clap::Args, Debug)]
pub struct CacheArgs {
    #[command(subcommand)]
    pub action: CacheAction,
}

/// `hf2q cache` subcommands. ADR-005 Phase 3 iter-205 (AC line 5351).
#[derive(Subcommand, Debug)]
pub enum CacheAction {
    /// List cached models with their quantizations and on-disk sizes.
    List,

    /// Print the total on-disk size of the cache.
    Size,

    /// Clear cache entries — single (model, quant), all quants for
    /// one model, or the entire cache (`--all --yes`).
    Clear {
        /// HF repo-id of the model whose cache entry to clear
        /// (e.g. `google/gemma-4-27b-it`). Required unless `--all`.
        #[arg(long)]
        model: Option<String>,

        /// Limit removal to a single quantization (`Q8_0`, `Q6_K`,
        /// `Q4_K_M`, `Q3_K_M`). When omitted with `--model`, every
        /// quant cached for that model is removed (plus the model dir
        /// `source/` and `repo_meta.json`).
        #[arg(long)]
        quant: Option<String>,

        /// Clear every cached model. Refuses without `--yes` to
        /// prevent accidental wipes (the entire on-disk cache is
        /// removed; manifest is reset to empty).
        #[arg(long, default_value_t = false)]
        all: bool,

        /// Confirm a destructive operation. Required by `--all`.
        #[arg(long, default_value_t = false)]
        yes: bool,
    },
}

#[derive(clap::Args, Debug, Clone)]
pub struct SmokeArgs {
    /// Arch key as registered in `src/arch/` (qwen35, qwen35moe)
    #[arg(long)]
    pub arch: String,

    /// Quant method to smoke-test
    #[arg(long, default_value = "q4_0")]
    pub quant: String,

    /// Also exercise the --emit-vision-tower path (dense variants with vision_config)
    #[arg(long, default_value_t = false)]
    pub with_vision: bool,

    /// Skip the convert step and reuse an existing GGUF on disk
    #[arg(long, default_value_t = false)]
    pub skip_convert: bool,

    /// Run preflight + dispatch, skip convert/inference, emit transcript path
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,

    /// Fixtures root (defaults to `tests/fixtures/`)
    #[arg(long)]
    pub fixtures_root: Option<PathBuf>,

    /// Use a local safetensors directory instead of downloading from HF.
    /// When set, preflight skips HF_TOKEN + repo-resolution checks.
    /// Enables CI testing of the Q4_0 end-to-end path on synthetic models.
    #[arg(long)]
    pub local_dir: Option<PathBuf>,

    /// Keep converted GGUF(s) in this directory. Defaults to a temp dir
    /// so repeat smoke runs don't accumulate disk.
    #[arg(long)]
    pub convert_output_dir: Option<PathBuf>,

    /// Override path for llama-cli. When set, preflight skips the default
    /// llama-cli search (/opt/llama.cpp/build/bin/ + PATH) and the smoke
    /// runner uses this path directly. Enables CI tests to inject a
    /// mock stub emitting a deterministic transcript.
    #[arg(long)]
    pub llama_cli_override: Option<PathBuf>,
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

    /// ADR-012 P10 (Decision 18) — emit mmproj-<slug>-F16.gguf alongside
    /// the text GGUF when the HF repo has a vision_config. Silently skipped
    /// when no vision_config is present (Gemma4, Qwen3.6-35B-A3B MoE).
    #[arg(long, default_value_t = false)]
    pub emit_vision_tower: bool,

    /// ADR-005 Phase 3 item 3/4 — skip post-download per-shard SHA-256
    /// integrity verification against HuggingFace's `x-linked-etag`.
    ///
    /// **NOT recommended.** Integrity is on by default; this opt-out
    /// exists for development workflows + air-gapped setups.  With this
    /// flag, downloaded shards are accepted as-is — corruption, MITM,
    /// and silent force-push of the source repo will not be detected.
    #[arg(long, default_value_t = false)]
    pub no_integrity: bool,
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

    /// Enable Qwen3.5 MTP speculative decoding when MTP weights are present.
    /// Also enabled by HF2Q_SPEC_DECODE=1; Qwen3.5 GGUFs with MTP default on.
    #[arg(long)]
    pub speculative: bool,

    /// Override chat template with a Jinja2 string
    ///
    /// Priority order (per ADR-005 Phase 1): this flag > --chat-template-file >
    /// GGUF `tokenizer.chat_template` metadata > hardcoded fallback.
    #[arg(long)]
    pub chat_template: Option<String>,

    /// Override chat template by reading from a file containing a Jinja2 template
    #[arg(long, conflicts_with = "chat_template")]
    pub chat_template_file: Option<PathBuf>,

    // ADR-008: candle-era kernel mode flags removed.
    // The mlx-native backend handles all dispatch internally.
}

#[derive(clap::Args, Debug)]
pub struct ServeArgs {
    /// Path to GGUF model file. Optional in the iter-2 backbone which only
    /// exposes /health, /readyz, /v1/models; required once /v1/chat and
    /// /v1/embeddings route. Fail-fast on bad weights is preserved: if
    /// --model is supplied, the GGUF header is validated at startup.
    #[arg(long)]
    pub model: Option<PathBuf>,

    /// Path to tokenizer.json (if not alongside GGUF).
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

    /// Path to config.json (if not alongside GGUF).
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Host to bind to. Defaults to 127.0.0.1 (localhost only) per ADR-005
    /// Decision #7. Use `--host 0.0.0.0` to expose on the LAN; public-internet
    /// is NOT a supported deployment target (reverse-proxy assumption,
    /// Decision #13).
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port to listen on.
    #[arg(long, default_value = "8080")]
    pub port: u16,

    /// Maximum sequence length (reserved for future iterations — context
    /// length is read from GGUF metadata).
    #[arg(long, default_value = "4096")]
    pub max_seq_len: usize,

    /// Bearer token required on every request (Decision #8). Unset = no auth.
    /// Also read from `HF2Q_AUTH_TOKEN` env var if --auth-token is absent
    /// (handled inside `cmd_serve`, not via clap's `env` feature to keep the
    /// `clap` feature set minimal).
    #[arg(long)]
    pub auth_token: Option<String>,

    /// CORS allowed origin. Repeatable. Empty = allow any (localhost dev);
    /// non-empty = restrictive allowlist (Decision #9).
    #[arg(long = "cors-origin", value_name = "ORIGIN")]
    pub cors_origins: Vec<String>,

    /// Hard cap on the FIFO generation queue. Overflow returns 429 +
    /// Retry-After (Decision #19).
    #[arg(long, default_value = "32")]
    pub queue_capacity: usize,

    /// Cache directory to scan for /v1/models listings. Defaults to
    /// $HOME/.cache/hf2q (Decision #26).
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,

    /// Default context-overflow policy (Decision #23). Per-request
    /// `hf2q_overflow_policy` overrides this.
    #[arg(long, value_enum, default_value = "summarize")]
    pub overflow_policy: OverflowPolicyArg,

    /// Path to a dedicated embedding GGUF (BERT family — nomic-embed-text,
    /// mxbai-embed-large, bge-small-en, etc.). When supplied, the server
    /// validates the file's GGUF header + parses the BERT config at
    /// startup and surfaces it via `/v1/models` with extension fields
    /// (`pooling`, `context_length`, `hidden_size`). The forward pass
    /// that backs `/v1/embeddings` requests lands in a later iter
    /// (ADR-005 Phase 2b, Task #13).
    #[arg(long)]
    pub embedding_model: Option<PathBuf>,

    /// Path to a multimodal projector GGUF (mmproj). When supplied, the
    /// server validates the file's GGUF header + parses the
    /// `MmprojConfig` at startup and surfaces it via `/v1/models`.
    /// Required for `image_url` content parts in `/v1/chat/completions`.
    /// Without it, the server is text-only and rejects image parts with
    /// 400 `no_mmproj_loaded` (ADR-005 Phase 2c Task #14).
    #[arg(long)]
    pub mmproj: Option<PathBuf>,

    /// ADR-005 Phase 3 item 3/4 — skip pre-load cache integrity
    /// verification of the cached GGUF on disk.
    ///
    /// **NOT recommended.** When the model came from `hf2q convert`'s
    /// auto-pipeline (iter-204), the cache manifest carries the SHA-256
    /// recorded at quantize time; serve re-hashes the file on load to
    /// catch disk bit-rot, partial writes, or manual edits. This flag
    /// disables that check; corruption silently passes through.
    #[arg(long, default_value_t = false)]
    pub no_integrity: bool,
}

/// CLI-facing copy of `serve::api::schema::OverflowPolicy`. Kept local to
/// avoid leaking the schema module into the CLI arg parser.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum OverflowPolicyArg {
    Reject,
    TruncateLeft,
    Summarize,
}

#[derive(Subcommand, Debug)]
pub enum ParityCommand {
    /// Check hf2q output against locked reference fixtures.
    ///
    /// Default mode: byte-prefix comparison vs `*_llama.txt` (or vs the
    /// frozen hf2q self-baseline when `--self-baseline` is set).  Pass
    /// `--tq-quality` to switch to the ADR-007 §853-866 Gate H envelope
    /// check instead (cosine / argmax / PPL Δ vs a frozen
    /// `<prompt>_tq_quality.json` fixture).
    Check {
        /// Path to GGUF model file
        #[arg(long)]
        model: PathBuf,

        /// Eval prompt name (sourdough, short_hello, sliding_wrap)
        #[arg(long, default_value = "sourdough")]
        prompt: String,

        /// Minimum common byte prefix required to pass
        #[arg(long)]
        min_prefix: Option<usize>,

        /// Maximum tokens to generate
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Compare against the frozen hf2q self-baseline (*_hf2q.txt)
        /// instead of the llama.cpp reference (*_llama.txt). Encodes
        /// ADR-005 Closeout Amendment Gate D (hf2q-self bisect-safety
        /// when math deliberately changes and temporary llama.cpp drift
        /// is expected). Pass requires byte-identical match (not a
        /// min-prefix floor).
        #[arg(long)]
        self_baseline: bool,

        /// ADR-007 Gate H — switch this invocation to the TQ-active
        /// quality envelope check (cosine on SDPA outputs + argmax flip
        /// rate + PPL Δ vs the frozen `<prompt>_tq_quality.json`
        /// fixture). Requires `--fixture` to point at a fixture produced
        /// by `parity capture --tq-quality`.  All other flags below
        /// (`--cosine-mean-floor`, etc.) only take effect when this is
        /// set.
        #[arg(long)]
        tq_quality: bool,

        /// Path to the frozen `<prompt>_tq_quality.json` fixture (Gate H
        /// only). Required when `--tq-quality` is set; the comparison
        /// is meaningless without a frozen reference, so `parity check
        /// --tq-quality` errors clearly when this is absent.
        #[arg(long)]
        fixture: Option<PathBuf>,

        /// Cosine-similarity mean floor for Gate H (industry std 0.999;
        /// day-of-close envelope measured 0.9998).  ADR-007 §853.
        #[arg(long, default_value_t = 0.999)]
        cosine_mean_floor: f32,

        /// Cosine-similarity p1 floor for Gate H (industry std 0.99; day-
        /// of-close envelope measured 0.9986).  ADR-007 §853.
        #[arg(long, default_value_t = 0.99)]
        cosine_p1_floor: f32,

        /// Argmax-flip-rate ceiling for Gate H (W12 1.5%; day-of-close
        /// envelope measured 0.8% — variance baked in).  ADR-007 §853.
        #[arg(long, default_value_t = 0.015)]
        argmax_max: f32,

        /// PPL-delta ceiling for Gate H expressed as a fraction (W12
        /// 0.02 = 2.0%; day-of-close envelope measured 1.24% — variance
        /// baked in).  ADR-007 §866.
        #[arg(long, default_value_t = 0.02)]
        ppl_delta_max: f32,
    },

    /// Capture fresh reference outputs (requires model).
    ///
    /// Default mode: writes the byte-prefix anchor (`<prompt>_hf2q.txt`)
    /// for Gates C/D/E/F.  Pass `--tq-quality` to instead capture the
    /// frozen Gate H envelope fixture (`<prompt>_tq_quality.json`)
    /// described in ADR-007 §853-866.
    Capture {
        /// Path to GGUF model file
        #[arg(long)]
        model: PathBuf,

        /// Output directory for captured references
        #[arg(long, default_value = "tests/evals/reference")]
        output: PathBuf,

        /// Eval prompt name (sourdough, short_hello, sliding_wrap, all)
        #[arg(long, default_value = "all")]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long)]
        max_tokens: Option<usize>,

        /// ADR-007 Gate H — capture the frozen `<prompt>_tq_quality.json`
        /// fixture instead of the byte-prefix anchor.  Single prompt only
        /// (`all` is rejected); the in-process two-regime decode loop
        /// runs dense pass 1 + TQ pass 2 on one model load and emits the
        /// dense token stream + dense per-step NLL + the day-of-capture
        /// envelope (cosine / argmax / PPL Δ) into the fixture JSON.
        #[arg(long)]
        tq_quality: bool,
    },
}

#[derive(clap::Args, Debug)]
pub struct ParityArgs {
    #[command(subcommand)]
    pub command: ParityCommand,
}

// (legacy ADR-012 P8 SmokeArgs removed during worktree merge — the canonical
// definition lives at line 95 with the worktree's superset surface.)

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
    #[value(alias = "q8_0")]
    Q8,
    /// `q4_0` is accepted as an alias — the smoke harness's default
    /// (`hf2q smoke --arch ... --quant q4_0` per Decision 16 §3) was
    /// previously rejected by clap because only the bare `q4` form
    /// was registered. Both names refer to the same bit-identical
    /// 4-bit emission.
    #[value(alias = "q4_0")]
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
    #[value(name = "dwq-mixed-4-8")]
    DwqMixed48,
    #[value(name = "dwq-mixed-6-8")]
    DwqMixed68,
    #[value(name = "dwq-mixed-2-8")]
    DwqMixed28,
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
            Self::DwqMixed48 => write!(f, "dwq-mixed-4-8"),
            Self::DwqMixed68 => write!(f, "dwq-mixed-6-8"),
            Self::DwqMixed28 => write!(f, "dwq-mixed-2-8"),
            Self::Apex => write!(f, "apex"),
        }
    }
}

impl QuantMethod {
    /// Return the (base_bits, sensitive_bits) pair for DWQ variants.
    /// Returns None for non-DWQ variants.
    pub fn dwq_bit_pair(self) -> Option<(u8, u8)> {
        match self {
            Self::DwqMixed46 => Some((4, 6)),
            Self::DwqMixed48 => Some((4, 8)),
            Self::DwqMixed68 => Some((6, 8)),
            Self::DwqMixed28 => Some((2, 8)),
            _ => None,
        }
    }

    /// Return the default output filename suffix for this quant method.
    ///
    /// DWQ variants produce compact suffixes like "dwq46", "dwq48", etc.
    /// All other variants return their Display string unchanged (e.g. "mixed-4-6", "q4").
    pub fn default_filename_suffix(self) -> String {
        match self {
            Self::DwqMixed46 => "dwq46".to_string(),
            Self::DwqMixed48 => "dwq48".to_string(),
            Self::DwqMixed68 => "dwq68".to_string(),
            Self::DwqMixed28 => "dwq28".to_string(),
            other => other.to_string(),
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

#[allow(dead_code)]
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
    /// Skip post-download SHA-256 integrity verification against HF's
    /// `x-linked-etag`. Default `false` (integrity ON). ADR-005 Phase 3
    /// item 3/4.
    pub no_integrity: bool,
}

#[allow(dead_code)]
/// Default group size for quantization.
pub const DEFAULT_GROUP_SIZE: usize = 64;

#[allow(dead_code)]
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

#[allow(dead_code)]
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
            let dir = crate::input::hf_download::download_model(repo_id, &progress)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            // ADR-005 Phase 3 item 3/4 — verify per-shard integrity
            // against HF's x-linked-etag immediately after download.
            // On by default; --no-integrity bypasses with a stern warn.
            if args.no_integrity {
                tracing::warn!(
                    repo = %repo_id,
                    "integrity check skipped per --no-integrity (NOT recommended)"
                );
            } else {
                // Use the snapshot dir's name as the revision when it
                // looks like a 40-hex commit SHA; otherwise fall back to
                // "main" (the rolling tip).  hf-hub's snapshot layout
                // names directories by the commit hash, so the dir's
                // file_name() is the authoritative pin.
                let revision = dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .filter(|s| s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit()))
                    .unwrap_or("main")
                    .to_string();
                crate::input::integrity::verify_repo(repo_id, &revision, &dir)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
            }
            dir
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
            let suffix = args.quant.default_filename_suffix();
            match args.format {
                OutputFormat::Gguf => PathBuf::from(format!("{}-{}.gguf", model_name, suffix)),
                _ => PathBuf::from(format!("{}-{}-{}", model_name, args.format, suffix)),
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
        QuantMethod::DwqMixed46
        | QuantMethod::DwqMixed48
        | QuantMethod::DwqMixed68
        | QuantMethod::DwqMixed28 => {
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
        no_integrity: args.no_integrity,
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

    // ---- DWQ bit-pair dispatch table ----

    #[test]
    fn test_dwq_bit_pair_dispatch_table() {
        assert_eq!(QuantMethod::DwqMixed46.dwq_bit_pair(), Some((4, 6)));
        assert_eq!(QuantMethod::DwqMixed48.dwq_bit_pair(), Some((4, 8)));
        assert_eq!(QuantMethod::DwqMixed68.dwq_bit_pair(), Some((6, 8)));
        assert_eq!(QuantMethod::DwqMixed28.dwq_bit_pair(), Some((2, 8)));
    }

    #[test]
    fn test_non_dwq_variants_return_none_for_bit_pair() {
        assert_eq!(QuantMethod::Auto.dwq_bit_pair(), None);
        assert_eq!(QuantMethod::F16.dwq_bit_pair(), None);
        assert_eq!(QuantMethod::Q8.dwq_bit_pair(), None);
        assert_eq!(QuantMethod::Q4.dwq_bit_pair(), None);
        assert_eq!(QuantMethod::Q2.dwq_bit_pair(), None);
        assert_eq!(QuantMethod::Mixed26.dwq_bit_pair(), None);
        assert_eq!(QuantMethod::Mixed36.dwq_bit_pair(), None);
        assert_eq!(QuantMethod::Mixed46.dwq_bit_pair(), None);
        assert_eq!(QuantMethod::Apex.dwq_bit_pair(), None);
    }

    // ---- Default filename suffix helper ----

    #[test]
    fn test_dwq_default_filename_suffix() {
        assert_eq!(QuantMethod::DwqMixed46.default_filename_suffix(), "dwq46");
        assert_eq!(QuantMethod::DwqMixed48.default_filename_suffix(), "dwq48");
        assert_eq!(QuantMethod::DwqMixed68.default_filename_suffix(), "dwq68");
        assert_eq!(QuantMethod::DwqMixed28.default_filename_suffix(), "dwq28");
    }

    #[test]
    fn test_non_dwq_filename_suffix_byte_identical_to_display() {
        // All non-DWQ variants must return their Display string unchanged.
        for m in [
            QuantMethod::Auto,
            QuantMethod::F16,
            QuantMethod::Q8,
            QuantMethod::Q4,
            QuantMethod::Q2,
            QuantMethod::Mixed26,
            QuantMethod::Mixed36,
            QuantMethod::Mixed46,
            QuantMethod::Apex,
        ] {
            assert_eq!(
                m.default_filename_suffix(),
                m.to_string(),
                "non-DWQ suffix diverged from Display for {:?}",
                m
            );
        }
    }

    // ---- Display strings ----

    #[test]
    fn test_dwq_display_strings() {
        assert_eq!(QuantMethod::DwqMixed46.to_string(), "dwq-mixed-4-6");
        assert_eq!(QuantMethod::DwqMixed48.to_string(), "dwq-mixed-4-8");
        assert_eq!(QuantMethod::DwqMixed68.to_string(), "dwq-mixed-6-8");
        assert_eq!(QuantMethod::DwqMixed28.to_string(), "dwq-mixed-2-8");
    }

    // ---- Gemma-4 regression snapshot guard ----
    // Asserts that the DwqMixed46 dispatch path continues to produce
    // (base_bits=4, sensitive_bits=6) — byte-identical to pre-change behaviour.
    // This guard will fail if any future refactor silently changes the Gemma-4
    // quantization parameters.

    #[test]
    fn test_gemma4_regression_dwq_mixed46_dispatch_unchanged() {
        // Dispatch table entry must still be (4, 6).
        assert_eq!(
            QuantMethod::DwqMixed46.dwq_bit_pair(),
            Some((4, 6)),
            "Gemma-4 regression: DwqMixed46 must dispatch to (base=4, sensitive=6)"
        );
        // Default filename suffix (new behaviour per ADR Decision 10(c)).
        assert_eq!(
            QuantMethod::DwqMixed46.default_filename_suffix(),
            "dwq46",
            "Gemma-4 regression: DwqMixed46 filename suffix must be 'dwq46'"
        );
    }

    // ── ADR-005 Phase 3 iter-203 — --no-integrity flag parsing ──────────

    use clap::Parser;

    #[test]
    fn convert_no_integrity_default_false() {
        // Without the flag, integrity is ON (no_integrity == false).
        let cli =
            Cli::parse_from(["hf2q", "convert", "--input", "/tmp/x", "--format", "gguf"]);
        let Command::Convert(args) = cli.command else {
            panic!("expected Convert");
        };
        assert!(!args.no_integrity, "default must be --no-integrity OFF");
    }

    #[test]
    fn convert_no_integrity_flag_parses_to_true() {
        let cli = Cli::parse_from([
            "hf2q",
            "convert",
            "--input",
            "/tmp/x",
            "--format",
            "gguf",
            "--no-integrity",
        ]);
        let Command::Convert(args) = cli.command else {
            panic!("expected Convert");
        };
        assert!(args.no_integrity, "--no-integrity must set the flag");
    }

    #[test]
    fn serve_no_integrity_default_false() {
        let cli = Cli::parse_from(["hf2q", "serve"]);
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve");
        };
        assert!(!args.no_integrity);
    }

    #[test]
    fn serve_no_integrity_flag_parses_to_true() {
        let cli = Cli::parse_from(["hf2q", "serve", "--no-integrity"]);
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve");
        };
        assert!(args.no_integrity);
    }

    #[test]
    fn convert_no_integrity_propagates_into_resolved_config() {
        // resolve_convert_config is the canonical lift from ConvertArgs to
        // ConvertConfig.  Verify the boolean copy makes it through.  Use
        // an --input path so the network leg never runs.
        let tmp = tempfile::tempdir().unwrap();
        let args = ConvertArgs {
            input: Some(tmp.path().to_path_buf()),
            repo: None,
            format: OutputFormat::Gguf,
            quant: QuantMethod::Q4,
            sensitive_layers: None,
            calibration_samples: 1024,
            target_bpw: 4.5,
            bits: None,
            group_size: None,
            output: None,
            json_report: false,
            skip_quality: false,
            quality_gate: false,
            dry_run: false,
            yes: false,
            unsupported_layers: None,
            emit_vision_tower: false,
            no_integrity: true,
        };
        let cfg = resolve_convert_config(&args).unwrap();
        assert!(cfg.no_integrity);
    }

    #[test]
    fn convert_no_integrity_default_propagates_false() {
        let tmp = tempfile::tempdir().unwrap();
        let args = ConvertArgs {
            input: Some(tmp.path().to_path_buf()),
            repo: None,
            format: OutputFormat::Gguf,
            quant: QuantMethod::Q4,
            sensitive_layers: None,
            calibration_samples: 1024,
            target_bpw: 4.5,
            bits: None,
            group_size: None,
            output: None,
            json_report: false,
            skip_quality: false,
            quality_gate: false,
            dry_run: false,
            yes: false,
            unsupported_layers: None,
            emit_vision_tower: false,
            no_integrity: false,
        };
        let cfg = resolve_convert_config(&args).unwrap();
        assert!(!cfg.no_integrity);
    }
}
