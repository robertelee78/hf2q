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

    /// Patch an existing GGUF file's metadata without changing tensor bytes
    GgufPatch(GgufPatchArgs),

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

#[derive(clap::Args, Debug)]
pub struct GgufPatchArgs {
    /// Input GGUF file
    pub input: PathBuf,

    /// Output GGUF path. Required unless --in-place or --dry-run is set.
    #[arg(long, conflicts_with = "in_place")]
    pub output: Option<PathBuf>,

    /// Rewrite the input file atomically in place
    #[arg(long, conflicts_with = "output", default_value_t = false)]
    pub in_place: bool,

    /// Add tokenizer.chat_template from the architecture default.
    ///
    /// This is currently the default and only patch operation; the flag is
    /// accepted for explicit operator intent and forward-compatible scripts.
    #[arg(long, default_value_t = false)]
    pub add_chat_template_from_arch: bool,

    /// Report the planned action without writing
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,
}

/// Top-level args for `hf2q cache`. The actual surface is split into
/// `CacheAction` subcommands (clap convention shared with `parity`).
#[derive(clap::Args, Debug)]
pub struct CacheArgs {
    #[command(subcommand)]
    pub action: CacheAction,
}

/// `hf2q cache` subcommands. ADR-005 Phase 3 iter-205 (AC line 5351).
///
/// ADR-017 §R-F6 extension: the `--kv-namespace` flag (orthogonal to
/// the weights-side surface) flips the action's target from the
/// model-weights cache to the on-disk persistent KV-cache subtree
/// (`<kv-persist>/models/<fp_short>/`). Path discovery order when
/// `--kv-namespace` is set:
///
/// 1. `--kv-path PATH` (explicit; first to win).
/// 2. `HF2Q_KV_PERSIST_PATH` env var.
///
/// We do NOT silently fall back to the weights-cache root or any
/// implicit default — see `docs/operating-kv-cache.md` §3 / ADR-017
/// §R-F1: the kv-persist path is operator-supplied at `cmd_serve`
/// startup with no default, and guessing here could clear an
/// unintended directory.
#[derive(Subcommand, Debug)]
pub enum CacheAction {
    /// List cached models with their quantizations and on-disk sizes.
    ///
    /// With `--kv-namespace`: enumerates per-model directories under
    /// `<kv-persist>/models/<fp_short>/` (one row per `<fp_short>`)
    /// with bytes-on-disk + block count under that subtree. Closes
    /// `docs/operating-kv-cache.md` §11 #4 (ADR-017 §R-F6).
    List {
        /// Switch this action to the persistent KV-cache subtree
        /// instead of the model-weights cache. ADR-017 §R-F6.
        #[arg(long, default_value_t = false)]
        kv_namespace: bool,

        /// KV-persist root override. Required (or via
        /// `HF2Q_KV_PERSIST_PATH`) when `--kv-namespace` is set.
        #[arg(long, value_name = "PATH", requires = "kv_namespace")]
        kv_path: Option<PathBuf>,
    },

    /// Print the total on-disk size of the cache.
    ///
    /// With `--kv-namespace`: total bytes-on-disk under
    /// `<kv-persist>/models/`.
    Size {
        /// Switch this action to the persistent KV-cache subtree.
        /// ADR-017 §R-F6.
        #[arg(long, default_value_t = false)]
        kv_namespace: bool,

        /// KV-persist root override. Required (or via
        /// `HF2Q_KV_PERSIST_PATH`) when `--kv-namespace` is set.
        #[arg(long, value_name = "PATH", requires = "kv_namespace")]
        kv_path: Option<PathBuf>,
    },

    /// Clear cache entries — single (model, quant), all quants for
    /// one model, or the entire cache (`--all --yes`).
    ///
    /// With `--kv-namespace --model <repo>`: removes the per-`(repo,
    /// quant)` directory at `<kv-persist>/models/<fp_short>/kv/`.
    /// `--model` is REQUIRED for kv-namespace clear (no `--all`
    /// shortcut — operator runbook §11 #4 specifically calls for
    /// per-repo scope; whole-cache wipe stays as `rm -rf
    /// <kv-persist>` on a stopped serve). Does NOT touch
    /// `<kv-persist>/locks/` (those are session-scoped) or other
    /// repos' subtrees.
    Clear {
        /// HF repo-id of the model whose cache entry to clear
        /// (e.g. `google/gemma-4-27b-it`). Required unless `--all`
        /// (weights-side) or always-required for `--kv-namespace`.
        #[arg(long)]
        model: Option<String>,

        /// Limit removal to a single quantization (`Q8_0`, `Q6_K`,
        /// `Q4_K_M`, `Q3_K_M`). When omitted with `--model`, every
        /// quant cached for that model is removed (plus the model dir
        /// `source/` and `repo_meta.json`).
        ///
        /// For `--kv-namespace` the per-`(repo, quant)` fingerprint
        /// only varies by `(repo, quant)` at this commit (see
        /// `docs/operating-kv-cache.md` §3); omitting `--quant`
        /// removes every quant variant cached for the repo.
        #[arg(long)]
        quant: Option<String>,

        /// Clear every cached model. Refuses without `--yes` to
        /// prevent accidental wipes (the entire on-disk cache is
        /// removed; manifest is reset to empty). Refused under
        /// `--kv-namespace` — see variant docstring.
        #[arg(long, default_value_t = false)]
        all: bool,

        /// Confirm a destructive operation. Required by `--all`.
        #[arg(long, default_value_t = false)]
        yes: bool,

        /// Switch this action to the persistent KV-cache subtree.
        /// ADR-017 §R-F6.
        #[arg(long, default_value_t = false)]
        kv_namespace: bool,

        /// KV-persist root override. Required (or via
        /// `HF2Q_KV_PERSIST_PATH`) when `--kv-namespace` is set.
        #[arg(long, value_name = "PATH", requires = "kv_namespace")]
        kv_path: Option<PathBuf>,

        /// Override the active-serve sentinel-flock guard. Use only
        /// when you know no `hf2q serve --kv-persist=SAME_PATH` is
        /// running against this cache root (a stopped serve can
        /// leave a stale sentinel under abnormal exit; this flag
        /// re-enables clear in that case).
        #[arg(long, default_value_t = false)]
        force: bool,
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

    /// Quantization method (default: auto). Mutually exclusive with the
    /// orthogonal `--calibration` / `--output-format` pair (Decision 12
    /// §off-diagonal).
    #[arg(
        long,
        value_enum,
        default_value = "auto",
        conflicts_with = "calibration",
        conflicts_with = "output_format",
    )]
    pub quant: QuantMethod,

    /// Calibrator axis for the orthogonal off-diagonal selector
    /// (ADR-014 Decision 12 §off-diagonal). Must be supplied together
    /// with `--output-format`. Off-diagonal pairs (e.g. `imatrix` +
    /// `bit-pair-4-6`) require `HF2Q_UNSAFE_EXPERIMENTS=1`.
    #[arg(
        long,
        value_enum,
        requires = "output_format",
    )]
    pub calibration: Option<CalibrationFlag>,

    /// OutputFormat axis for the orthogonal off-diagonal selector
    /// (ADR-014 Decision 12 §off-diagonal). Must be supplied together
    /// with `--calibration`. Off-diagonal pairs (e.g. `imatrix` +
    /// `bit-pair-4-6`) require `HF2Q_UNSAFE_EXPERIMENTS=1`.
    #[arg(
        long = "output-format",
        value_enum,
        requires = "calibration",
    )]
    pub output_format: Option<OutputFormatFlag>,

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

    /// Target shard size in GB for the safetensors directory layout
    /// (ADR-014 P9 iter-1 §S5). Default 5.0 GB matches the mlx-lm /
    /// HuggingFace community convention. Range: 0.5..=50.0. Ignored
    /// for `--format gguf` and for the single-file `--quant f16` /
    /// `--quant bf16` safetensors path (Decision 17 byte-identity).
    #[arg(
        long = "shard-size-gb",
        default_value_t = 5.0,
        value_parser = parse_shard_size_gb,
    )]
    pub shard_size_gb: f64,

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
    #[arg(long, default_value = "0.8")]
    pub temperature: f64,

    /// Top-p nucleus sampling
    #[arg(long, default_value = "0.95")]
    pub top_p: f64,

    /// Top-k sampling (0 = disabled)
    #[arg(long, default_value = "40")]
    pub top_k: usize,

    /// Min-p sampling (0.0 = disabled)
    #[arg(long, default_value = "0.05")]
    pub min_p: f64,

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

    /// Force thinking-mode rendering on: pass `enable_thinking=true` to the
    /// chat template's Jinja context. For Qwen3-thinking / QwQ / GPT-OSS-
    /// reasoning checkpoints, this opens an unfilled `<think>\n` block at
    /// the end of the rendered prompt — the model emits reasoning content
    /// + `</think>` + the answer (the "both" outcome).
    ///
    /// **Default behavior (NEITHER flag set) is auto-detect via the
    /// canonical render-and-diff signal**: hf2q renders the resolved chat
    /// template TWICE — once with `enable_thinking=true`, once with
    /// `=false` — and checks if the bytes differ. If they do, the
    /// template branches on the variable → the model class supports
    /// thinking-mode → default-on (model produces BOTH reasoning trace
    /// AND answer). If the bytes are identical (or no template can be
    /// resolved), default is `false` (safe fallback).
    ///
    /// Mirrors llama.cpp's `--reasoning auto` decision logic at
    /// `/opt/llama.cpp/common/chat-diff-analyzer.cpp:319-401` (the
    /// `compare_thinking_enabled` function) and the user-facing decision
    /// at `/opt/llama.cpp/tools/server/server-context.cpp:1050`.
    ///
    /// Pass `--enable-thinking` to override auto-detect ON (e.g. a custom
    /// template that doesn't probe as thinking-capable but you know the
    /// model supports it). Pass `--no-thinking` to override auto-detect
    /// OFF (e.g. a template that probes as thinking-capable but the
    /// model's actual checkpoint doesn't know how to close `</think>`,
    /// improvising `<|end|>` instead). Mutually exclusive.
    ///
    /// 2026-05-02: added (commit `8c110f5`, plumb-only) → re-defaulted to
    /// false (regression-fix iter 2) → name-substring heuristic (REJECTED
    /// by user) → render-and-diff canonical signal (peer audit:
    /// `/tmp/cfa-thinking-detect/peer-detection-report.md`). H2 audit:
    /// `docs/research/decode-test-gap-2026-05-02.md`.
    #[arg(long)]
    pub enable_thinking: bool,

    /// Force thinking-mode rendering off. See `--enable-thinking` for the
    /// full rationale (auto-detect default + override semantics). Use this
    /// when a model whose name LOOKS thinking-capable (e.g. `qwen-thinking-
    /// distill`) actually breaks with the open-`<think>` prompt cue.
    /// Mutually exclusive with `--enable-thinking`.
    #[arg(long, conflicts_with = "enable_thinking")]
    pub no_thinking: bool,

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

    /// Suppress the human-readable serve load banner on stdout.
    #[arg(long, default_value_t = false)]
    pub quiet: bool,

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

    /// ADR-017 Phase C.1 — enable persistent block-prefix KV cache to
    /// disk. The argument is the cache directory (e.g.
    /// `/tmp/hf2q-kv-persist` or `$HOME/.cache/hf2q/kv-persist`). The
    /// directory is created if missing; the recovery scan runs at
    /// startup to rebuild the in-memory `BlockIndex` from any
    /// previously-written envelopes.
    ///
    /// When unset (default), the engine wires `NoopKvSpiller` and
    /// behaves byte-identical to the pre-ADR-017 path. When set, the
    /// engine wires `BlockPrefixCacheSpiller` + a per-loaded-family
    /// `EngineBindable` registration so the spiller's `pre_evict` /
    /// `post_admit` triggers route through the on-disk lifecycle.
    ///
    /// C.1 ships the WIRING substrate. The actual sourdough byte-exact
    /// coherence run + perf-validation matrix lands in Phase D after
    /// B-dense.2's round-trip parity matrix on real GGUF; until then,
    /// the C.1 default registration uses `StubGemma4Spill` (always
    /// `Skipped` on snapshot/restore) so the on-path is observable
    /// but functionally inert.
    #[arg(long = "kv-persist", value_name = "PATH")]
    pub kv_persist_path: Option<PathBuf>,
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

/// Quantization method — the 17-variant Decision-12 menu (ADR-014 P8).
///
/// ## Cells
///
/// The menu spans the diagonal of the (Calibrator × OutputFormat) matrix
/// from ADR-014 Decision 9.  Off-diagonal cells (e.g. `dwq` calibrator
/// with K-quant output) are reachable only via the orthogonal
/// `--calibration` + `--output-format` flags, gated by
/// `HF2Q_UNSAFE_EXPERIMENTS=1` (Decision 12 §off-diagonal).
///
/// | Variant            | Calibrator   | OutputFormat   | Notes                                 |
/// |--------------------|--------------|----------------|---------------------------------------|
/// | `auto`             | (resolved)   | (resolved)     | AutoResolver Decision 18 routing      |
/// | `f16` / `bf16`     | none         | flat           | Passthrough float                     |
/// | `q2` / `q4` / `q8` | none         | flat           | Legacy IR-quantize (Q2_K/Q4_0/Q8_0)   |
/// | `q4_k_m`           | none         | k-quant Q4_K   | Uncalibrated K-quant (community-std)  |
/// | `q5_k_m`           | none         | k-quant Q5_K   |                                       |
/// | `q6_k`             | none         | k-quant Q6_K   |                                       |
/// | `imatrix-q4_k_m`   | imatrix      | k-quant Q4_K   | llama.cpp-style importance matrix     |
/// | `imatrix-q5_k_m`   | imatrix      | k-quant Q5_K   |                                       |
/// | `imatrix-q6_k`     | imatrix      | k-quant Q6_K   |                                       |
/// | `imatrix-adaptive` | imatrix      | k-quant adapt  | Per-tensor optimal precision (apex)   |
/// | `dwq-4-6`          | dwq          | bit-pair (4,6) | Apple/MLX distilled weight quant      |
/// | `dwq-4-8`          | dwq          | bit-pair (4,8) |                                       |
/// | `dwq-6-8`          | dwq          | bit-pair (6,8) |                                       |
/// | `dwq-2-8`          | dwq          | bit-pair (2,8) |                                       |
///
/// ## Decision 13 — clean cut, no aliases
///
/// The pre-P8 variants (`apex`, `mixed-2-6`, `mixed-3-6`, `mixed-4-6`,
/// `dwq-mixed-N-M`) are deleted with no aliases.  Old user scripts MUST
/// fail with a helpful "did you mean" error mapped via
/// [`map_deleted_quant_hint`].  The renames are:
///
/// - `apex` → `imatrix-adaptive` (preserves per-tensor optimal precision)
/// - `mixed-N-M` → `q4_k_m` (uncalibrated K-quant is the modern equivalent)
/// - `dwq-mixed-N-M` → `dwq-N-M` (cosmetic rename, same algorithm)
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantMethod {
    /// `auto` — AutoResolver Decision 18 routing table picks the cell
    /// based on dense/MoE classification + hardware (memory, bandwidth).
    Auto,
    /// `f16` — flat float-16 passthrough.
    F16,
    /// `bf16` — flat bfloat-16 passthrough (preserves the source dtype
    /// for HF models exported as bf16; no precision loss vs f16 cast).
    Bf16,
    /// `q2` — flat 2-bit (Q2_K block format).
    Q2,
    /// `q4` — flat 4-bit (Q4_0 / Q4_K block format depending on output).
    /// `q4_0` is accepted as an alias — the smoke harness's default
    /// (`hf2q smoke --arch ... --quant q4_0` per Decision 16 §3) was
    /// previously rejected by clap because only the bare `q4` form was
    /// registered. Both names refer to the same bit-identical 4-bit
    /// emission.
    #[value(alias = "q4_0")]
    Q4,
    /// `q8` — flat 8-bit (Q8_0).
    #[value(alias = "q8_0")]
    Q8,
    /// `q2_k_s` — Q2_K "small" variant. 2.625 bpw base. Output → Q6_K,
    /// attn_v → Q4_K, ffn_down i<n/8 → Q4_K else base.
    #[value(name = "q2_k_s")]
    Q2KS,
    /// `q2_k` — default Q2_K variant. Output → Q6_K, attn_v → Q4_K,
    /// ffn_down → Q3_K always.
    #[value(name = "q2_k")]
    Q2K,
    /// `q3_k_s` — uncalibrated K-quant Q3_K, "small" variant. Base Q3_K
    /// everywhere except output (→ Q6_K). 3.4375 bpw base.
    #[value(name = "q3_k_s")]
    Q3KS,
    /// `q3_k_m` — Q3_K "medium" with attn_v Q5_K (i_layer<2) / Q4_K
    /// upgrade per `llama-quant.cpp:527-528`, ffn_down Q5_K
    /// (i_layer<n/16) / Q4_K (use_more_bits) / Q3_K per `:578-582`.
    #[value(name = "q3_k_m")]
    Q3KM,
    /// `q3_k_l` — Q3_K "large" with attn_v → Q5_K (`:530`) and
    /// ffn_down → Q5_K (`:587-588`).  Output bumps to Q6_K.
    #[value(name = "q3_k_l")]
    Q3KL,
    /// `q4_k_s` — Q4_K "small" variant (minimal upgrades; attn_v
    /// `i_layer<4 → Q5_K`, otherwise base; ffn_down stays at base).
    #[value(name = "q4_k_s")]
    Q4KS,
    /// `q4_k_m` — uncalibrated K-quant Q4_K with per-tensor `_M`
    /// upgrades (output/token_embd → Q6_K, attn_v / ffn_down on
    /// `use_more_bits` layers).
    #[value(name = "q4_k_m")]
    Q4KM,
    /// `q5_k_s` — Q5_K "small" variant (minimal upgrades).
    #[value(name = "q5_k_s")]
    Q5KS,
    /// `q5_k_m` — uncalibrated K-quant Q5_K (`_M` variant).
    #[value(name = "q5_k_m")]
    Q5KM,
    /// `q6_k` — uncalibrated K-quant Q6_K.
    #[value(name = "q6_k")]
    Q6K,
    /// `imatrix-q2_k_s` — Q2_K_S with imatrix-weighted codebook search.
    #[value(name = "imatrix-q2_k_s")]
    ImatrixQ2KS,
    /// `imatrix-q2_k` — default Q2_K with imatrix-weighted codebook search.
    #[value(name = "imatrix-q2_k")]
    ImatrixQ2K,
    /// `imatrix-q3_k_s` — Q3_K_S with imatrix-weighted codebook search.
    #[value(name = "imatrix-q3_k_s")]
    ImatrixQ3KS,
    /// `imatrix-q3_k_m` — Q3_K_M with imatrix-weighted codebook search.
    #[value(name = "imatrix-q3_k_m")]
    ImatrixQ3KM,
    /// `imatrix-q3_k_l` — Q3_K_L with imatrix-weighted codebook search.
    #[value(name = "imatrix-q3_k_l")]
    ImatrixQ3KL,
    /// `imatrix-q4_k_s` — Q4_K_S with imatrix-weighted codebook search.
    #[value(name = "imatrix-q4_k_s")]
    ImatrixQ4KS,
    /// `imatrix-q4_k_m` — K-quant Q4_K (`_M` variant) with imatrix-
    /// weighted codebook search (llama.cpp PR #4861 / commit `ec893798`).
    #[value(name = "imatrix-q4_k_m")]
    ImatrixQ4KM,
    /// `imatrix-q5_k_s` — Q5_K_S with imatrix-weighted codebook search.
    #[value(name = "imatrix-q5_k_s")]
    ImatrixQ5KS,
    /// `imatrix-q5_k_m` — K-quant Q5_K (`_M` variant) with imatrix-
    /// weighted codebook search.
    #[value(name = "imatrix-q5_k_m")]
    ImatrixQ5KM,
    /// `imatrix-q6_k` — K-quant Q6_K with imatrix-weighted codebook
    /// search.
    #[value(name = "imatrix-q6_k")]
    ImatrixQ6K,
    /// `imatrix-adaptive` — imatrix-calibrated, per-tensor optimal
    /// precision K-quant (replaces the deleted `apex` variant).
    /// Routes through [`crate::quantize::variant_quantizer::VariantKQuantizer`]
    /// with `layer_mix::target_for` per-tensor target dispatch.
    #[value(name = "imatrix-adaptive")]
    ImatrixAdaptive,
    /// `dwq-4-6` — Apple/MLX distilled weight quantization with
    /// (base=4, sensitive=6) bit pair.
    #[value(name = "dwq-4-6")]
    Dwq46,
    /// `dwq-4-8` — DWQ with (base=4, sensitive=8) bit pair.
    #[value(name = "dwq-4-8")]
    Dwq48,
    /// `dwq-6-8` — DWQ with (base=6, sensitive=8) bit pair.
    #[value(name = "dwq-6-8")]
    Dwq68,
    /// `dwq-2-8` — DWQ with (base=2, sensitive=8) bit pair.
    #[value(name = "dwq-2-8")]
    Dwq28,
}

impl std::fmt::Display for QuantMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::F16 => write!(f, "f16"),
            Self::Bf16 => write!(f, "bf16"),
            Self::Q2 => write!(f, "q2"),
            Self::Q4 => write!(f, "q4"),
            Self::Q8 => write!(f, "q8"),
            Self::Q2KS => write!(f, "q2_k_s"),
            Self::Q2K => write!(f, "q2_k"),
            Self::Q3KS => write!(f, "q3_k_s"),
            Self::Q3KM => write!(f, "q3_k_m"),
            Self::Q3KL => write!(f, "q3_k_l"),
            Self::Q4KS => write!(f, "q4_k_s"),
            Self::Q4KM => write!(f, "q4_k_m"),
            Self::Q5KS => write!(f, "q5_k_s"),
            Self::Q5KM => write!(f, "q5_k_m"),
            Self::Q6K => write!(f, "q6_k"),
            Self::ImatrixQ2KS => write!(f, "imatrix-q2_k_s"),
            Self::ImatrixQ2K => write!(f, "imatrix-q2_k"),
            Self::ImatrixQ3KS => write!(f, "imatrix-q3_k_s"),
            Self::ImatrixQ3KM => write!(f, "imatrix-q3_k_m"),
            Self::ImatrixQ3KL => write!(f, "imatrix-q3_k_l"),
            Self::ImatrixQ4KS => write!(f, "imatrix-q4_k_s"),
            Self::ImatrixQ4KM => write!(f, "imatrix-q4_k_m"),
            Self::ImatrixQ5KS => write!(f, "imatrix-q5_k_s"),
            Self::ImatrixQ5KM => write!(f, "imatrix-q5_k_m"),
            Self::ImatrixQ6K => write!(f, "imatrix-q6_k"),
            Self::ImatrixAdaptive => write!(f, "imatrix-adaptive"),
            Self::Dwq46 => write!(f, "dwq-4-6"),
            Self::Dwq48 => write!(f, "dwq-4-8"),
            Self::Dwq68 => write!(f, "dwq-6-8"),
            Self::Dwq28 => write!(f, "dwq-2-8"),
        }
    }
}

impl QuantMethod {
    /// Return the (base_bits, sensitive_bits) pair for DWQ variants.
    /// Returns None for non-DWQ variants.
    pub fn dwq_bit_pair(self) -> Option<(u8, u8)> {
        match self {
            Self::Dwq46 => Some((4, 6)),
            Self::Dwq48 => Some((4, 8)),
            Self::Dwq68 => Some((6, 8)),
            Self::Dwq28 => Some((2, 8)),
            _ => None,
        }
    }

    /// True when this method actually quantizes weights (anything other
    /// than the float-passthrough variants `f16` / `bf16`).
    ///
    /// Used by the safetensors backend (ADR-014 P9 iter-1 §S2) to pick
    /// between the single-file float emission and the mlx-lm-style
    /// directory layout with sharded `model-NNNNN-of-MMMMM.safetensors`
    /// + `model.safetensors.index.json` + `config.json` injection.
    ///
    /// `Auto` returns `true` because once `resolve_convert_config`
    /// resolves it to a concrete variant the dispatch needs the same
    /// quantized-path treatment; the resolver replaces `Auto` with a
    /// concrete variant before this predicate is consulted by the
    /// safetensors emit path, but defaulting to `true` here is the
    /// conservative no-shortcut answer (mantra).
    pub fn is_quantized(self) -> bool {
        match self {
            Self::F16 | Self::Bf16 => false,
            Self::Auto
            | Self::Q2
            | Self::Q4
            | Self::Q8
            | Self::Q2KS
            | Self::Q2K
            | Self::Q3KS
            | Self::Q3KM
            | Self::Q3KL
            | Self::Q4KS
            | Self::Q4KM
            | Self::Q5KS
            | Self::Q5KM
            | Self::Q6K
            | Self::ImatrixQ2KS
            | Self::ImatrixQ2K
            | Self::ImatrixQ3KS
            | Self::ImatrixQ3KM
            | Self::ImatrixQ3KL
            | Self::ImatrixQ4KS
            | Self::ImatrixQ4KM
            | Self::ImatrixQ5KS
            | Self::ImatrixQ5KM
            | Self::ImatrixQ6K
            | Self::ImatrixAdaptive
            | Self::Dwq46
            | Self::Dwq48
            | Self::Dwq68
            | Self::Dwq28 => true,
        }
    }

    /// Return the default output filename suffix for this quant method.
    ///
    /// DWQ variants produce compact suffixes like "dwq46", "dwq48", etc.
    /// All other variants return their Display string unchanged (e.g.
    /// "imatrix-q4_k_m", "q4_k_m", "q4").
    pub fn default_filename_suffix(self) -> String {
        match self {
            Self::Dwq46 => "dwq46".to_string(),
            Self::Dwq48 => "dwq48".to_string(),
            Self::Dwq68 => "dwq68".to_string(),
            Self::Dwq28 => "dwq28".to_string(),
            other => other.to_string(),
        }
    }
}

/// Map a deleted (Decision 13) `--quant` string to a "did you mean"
/// hint for the user-facing error message.
///
/// Returns `Some(hint)` for the deleted variants (`apex`, `mixed-N-M`,
/// `dwq-mixed-N-M`) and `None` for any other unknown string (in which
/// case clap's default `possible values` rendering is sufficient).
///
/// This is consulted at convert-dispatch time after `Cli::try_parse`
/// fails on a deleted variant — the helper is also exposed publicly so
/// the integration tests can assert the mapping table without re-deriving
/// it.
pub fn map_deleted_quant_hint(raw: &str) -> Option<String> {
    let lowered = raw.trim().to_lowercase();
    match lowered.as_str() {
        "apex" => Some(
            "`--quant apex` was removed in ADR-014 P8 (Decision 13). \
             Use `--quant imatrix-adaptive` for imatrix-calibrated \
             per-tensor optimal precision (preserves apex's behavior)."
                .to_string(),
        ),
        "mixed-2-6" | "mixed-3-6" | "mixed-4-6" => Some(format!(
            "`--quant {raw}` was removed in ADR-014 P8 (Decision 13). \
             Use `--quant q4_k_m` for uncalibrated K-quant (the modern \
             equivalent), or `--quant imatrix-q4_k_m` for the \
             imatrix-calibrated path."
        )),
        "dwq-mixed-4-6" => Some(
            "`--quant dwq-mixed-4-6` was renamed in ADR-014 P8 \
             (Decision 13). Use `--quant dwq-4-6` (same algorithm, \
             cosmetic rename)."
                .to_string(),
        ),
        "dwq-mixed-4-8" => Some(
            "`--quant dwq-mixed-4-8` was renamed in ADR-014 P8 \
             (Decision 13). Use `--quant dwq-4-8` (same algorithm, \
             cosmetic rename)."
                .to_string(),
        ),
        "dwq-mixed-6-8" => Some(
            "`--quant dwq-mixed-6-8` was renamed in ADR-014 P8 \
             (Decision 13). Use `--quant dwq-6-8` (same algorithm, \
             cosmetic rename)."
                .to_string(),
        ),
        "dwq-mixed-2-8" => Some(
            "`--quant dwq-mixed-2-8` was renamed in ADR-014 P8 \
             (Decision 13). Use `--quant dwq-2-8` (same algorithm, \
             cosmetic rename)."
                .to_string(),
        ),
        _ => None,
    }
}

/// Calibrator axis for the orthogonal `--calibration` flag (ADR-014
/// Decision 12 §off-diagonal). Pairs with [`OutputFormatFlag`].
///
/// When both flags are supplied, the convert dispatch builds the
/// corresponding (Calibrator, OutputFormat) pair — including off-
/// diagonal cells reachable only when `HF2Q_UNSAFE_EXPERIMENTS=1`.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationFlag {
    /// `none` — no calibration (NoneCalibrator).
    None,
    /// `imatrix` — llama.cpp-style importance matrix
    /// (ImatrixCalibrator).
    Imatrix,
    /// `dwq` — Apple/MLX distilled weight quantization
    /// (DwqCalibrator).
    Dwq,
}

impl std::fmt::Display for CalibrationFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Imatrix => write!(f, "imatrix"),
            Self::Dwq => write!(f, "dwq"),
        }
    }
}

/// OutputFormat axis for the orthogonal `--output-format` flag
/// (ADR-014 Decision 12 §off-diagonal). Pairs with [`CalibrationFlag`].
///
/// The variant naming uses kebab-form `flat-*` / `k-quant-*` /
/// `bit-pair-N-M` so the on-disk codec is unambiguous.
///
/// **Note**: This is distinct from the existing [`OutputFormat`]
/// (gguf / safetensors), which selects the *container* format. This
/// enum picks the per-tensor *codec* used inside that container.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormatFlag {
    /// `flat-f16` — flat float-16 (no quantization).
    #[value(name = "flat-f16")]
    FlatF16,
    /// `flat-bf16` — flat bfloat-16.
    #[value(name = "flat-bf16")]
    FlatBf16,
    /// `flat-q2` — flat 2-bit (Q2_K).
    #[value(name = "flat-q2")]
    FlatQ2,
    /// `flat-q4` — flat 4-bit (Q4_0).
    #[value(name = "flat-q4")]
    FlatQ4,
    /// `flat-q8` — flat 8-bit (Q8_0).
    #[value(name = "flat-q8")]
    FlatQ8,
    /// `k-quant-q4_k_m` — K-quant Q4_K (`_M` variant).
    #[value(name = "k-quant-q4_k_m")]
    KQuantQ4KM,
    /// `k-quant-q5_k_m` — K-quant Q5_K (`_M` variant).
    #[value(name = "k-quant-q5_k_m")]
    KQuantQ5KM,
    /// `k-quant-q6_k` — K-quant Q6_K.
    #[value(name = "k-quant-q6_k")]
    KQuantQ6K,
    /// `k-quant-adaptive` — per-tensor optimal precision K-quant
    /// (apex's behavior).
    #[value(name = "k-quant-adaptive")]
    KQuantAdaptive,
    /// `bit-pair-4-6` — DWQ-style (base=4, sensitive=6) bit pair.
    #[value(name = "bit-pair-4-6")]
    BitPair46,
    /// `bit-pair-4-8` — DWQ-style (base=4, sensitive=8) bit pair.
    #[value(name = "bit-pair-4-8")]
    BitPair48,
    /// `bit-pair-6-8` — DWQ-style (base=6, sensitive=8) bit pair.
    #[value(name = "bit-pair-6-8")]
    BitPair68,
    /// `bit-pair-2-8` — DWQ-style (base=2, sensitive=8) bit pair.
    #[value(name = "bit-pair-2-8")]
    BitPair28,
}

impl std::fmt::Display for OutputFormatFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FlatF16 => write!(f, "flat-f16"),
            Self::FlatBf16 => write!(f, "flat-bf16"),
            Self::FlatQ2 => write!(f, "flat-q2"),
            Self::FlatQ4 => write!(f, "flat-q4"),
            Self::FlatQ8 => write!(f, "flat-q8"),
            Self::KQuantQ4KM => write!(f, "k-quant-q4_k_m"),
            Self::KQuantQ5KM => write!(f, "k-quant-q5_k_m"),
            Self::KQuantQ6K => write!(f, "k-quant-q6_k"),
            Self::KQuantAdaptive => write!(f, "k-quant-adaptive"),
            Self::BitPair46 => write!(f, "bit-pair-4-6"),
            Self::BitPair48 => write!(f, "bit-pair-4-8"),
            Self::BitPair68 => write!(f, "bit-pair-6-8"),
            Self::BitPair28 => write!(f, "bit-pair-2-8"),
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
    /// Orthogonal Calibrator selector (ADR-014 Decision 12 §off-diagonal).
    /// `Some` → caller supplied `--calibration X --output-format Y`;
    /// `None` → caller used `--quant`.
    pub calibration: Option<CalibrationFlag>,
    /// Orthogonal OutputFormat selector (ADR-014 Decision 12 §off-diagonal).
    pub output_format: Option<OutputFormatFlag>,
    /// ADR-014 P9 iter-1 §S5 — target shard size in GB for the
    /// safetensors directory layout. Default 5.0; range 0.5..=50.0.
    pub shard_size_gb: f64,
    /// ADR-005 Phase 4 iter-211 — per-shard integrity records returned
    /// by [`crate::input::integrity::verify_repo`] during the
    /// `--repo` resolution path.  When `Some`, `cmd_convert` derives a
    /// canonical source-bundle SHA via
    /// [`crate::serve::cache::compute_source_bundle_sha256`] and the
    /// GGUF backend stamps it into `hf2q.source_sha256` so the
    /// auto-pipeline short-circuit can fire on the next cache hit.
    /// `None` for `--input` (local) runs and `--no-integrity` runs —
    /// no source bundle binding is available, GGUF parses as External,
    /// auto-pipeline runs the W71 30 GB SHA-256 re-hash exactly as
    /// before.
    pub source_shards: Option<Vec<crate::input::integrity::ShardIntegrity>>,
}

/// Custom value parser for `--shard-size-gb` (ADR-014 P9 iter-1 §S5).
/// Validates the 0.5..=50.0 GB range up front so an out-of-range flag
/// fails clap parsing rather than producing a malformed shard layout
/// downstream.
fn parse_shard_size_gb(raw: &str) -> Result<f64, String> {
    let value: f64 = raw
        .parse()
        .map_err(|e| format!("`--shard-size-gb` must be a decimal number: {e}"))?;
    if !value.is_finite() {
        return Err("`--shard-size-gb` must be finite (no NaN / Infinity)".to_string());
    }
    if !(0.5..=50.0).contains(&value) {
        return Err(format!(
            "`--shard-size-gb` must be in the range 0.5..=50.0 (got {value})"
        ));
    }
    Ok(value)
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
    // ADR-005 Phase 4 iter-211 — capture per-shard integrity records
    // from verify_repo and surface them on `ConvertConfig` so the
    // GGUF backend can stamp `hf2q.source_sha256` at write time.
    // `None` for the local-path / `--no-integrity` paths (no source
    // bundle binding is available; GGUF parses as External post-write).
    let mut source_shards: Option<Vec<crate::input::integrity::ShardIntegrity>> = None;
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
                let shards = crate::input::integrity::verify_repo(repo_id, &revision, &dir)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                source_shards = Some(shards);
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

    // Validate quant method is implemented (exhaustive — Decision 12 lock).
    match args.quant {
        QuantMethod::Auto => {
            // Auto mode — resolve at conversion time via AutoResolver Decision 18.
        }
        QuantMethod::F16
        | QuantMethod::Bf16
        | QuantMethod::Q2
        | QuantMethod::Q4
        | QuantMethod::Q8 => {
            // Flat float / legacy block-quant variants.
        }
        QuantMethod::Q2KS
        | QuantMethod::Q2K
        | QuantMethod::Q3KS
        | QuantMethod::Q3KM
        | QuantMethod::Q3KL
        | QuantMethod::Q4KS
        | QuantMethod::Q4KM
        | QuantMethod::Q5KS
        | QuantMethod::Q5KM
        | QuantMethod::Q6K => {
            // Uncalibrated K-quant (NoneCalibrator + KQuantCodecQuantizer).
        }
        QuantMethod::ImatrixQ2KS
        | QuantMethod::ImatrixQ2K
        | QuantMethod::ImatrixQ3KS
        | QuantMethod::ImatrixQ3KM
        | QuantMethod::ImatrixQ3KL
        | QuantMethod::ImatrixQ4KS
        | QuantMethod::ImatrixQ4KM
        | QuantMethod::ImatrixQ5KS
        | QuantMethod::ImatrixQ5KM
        | QuantMethod::ImatrixQ6K => {
            // imatrix-calibrated K-quant (ImatrixCalibrator + KQuantCodecQuantizer).
        }
        QuantMethod::ImatrixAdaptive => {
            // imatrix-calibrated per-tensor optimal precision (replaces former Apex).
        }
        QuantMethod::Dwq46
        | QuantMethod::Dwq48
        | QuantMethod::Dwq68
        | QuantMethod::Dwq28 => {
            // DWQ weight/activation calibration → bit-pair output.
        }
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
        calibration: args.calibration,
        output_format: args.output_format,
        shard_size_gb: args.shard_size_gb,
        source_shards,
    })
}

/// Environment-variable name for the off-diagonal dev-gate
/// (ADR-014 Decision 12 §off-diagonal). Setting `HF2Q_UNSAFE_EXPERIMENTS=1`
/// unlocks `--calibration X --output-format Y` cells that aren't part
/// of the validated 17-variant menu.
pub const HF2Q_UNSAFE_EXPERIMENTS_ENV: &str = "HF2Q_UNSAFE_EXPERIMENTS";

/// Errors from the off-diagonal `--calibration` / `--output-format`
/// validator.
#[derive(Debug, thiserror::Error)]
pub enum OffDiagonalError {
    /// Both `--calibration` and `--output-format` were supplied but
    /// `HF2Q_UNSAFE_EXPERIMENTS` is unset (or not equal to `"1"`).
    #[error(
        "off-diagonal cells require {env}=1; the requested pair \
         (--calibration {cal} --output-format {fmt}) is not a validated \
         cell in the Decision 12 menu. See docs/converting-a-model.md \
         §maintainers, or use one of the 17 `--quant` variants instead."
    )]
    DevGateRequired {
        env: &'static str,
        cal: String,
        fmt: String,
    },

    /// One of the orthogonal flags was supplied without the other.
    /// clap's `requires` already catches this, but the helper has its
    /// own pure check so library callers get the typed error too.
    #[error(
        "--calibration and --output-format must be supplied together \
         (got --calibration={cal:?} --output-format={fmt:?}). \
         Use --quant for the validated 17-variant menu, or supply \
         both flags for off-diagonal cells (with {env}=1)."
    )]
    PartialOrthogonalSelector {
        env: &'static str,
        cal: Option<CalibrationFlag>,
        fmt: Option<OutputFormatFlag>,
    },
}

/// Decide whether a given `(--calibration, --output-format)` pair is
/// "diagonal" (matches one of the 17 `--quant` variants — no dev-gate
/// needed) or "off-diagonal" (requires `HF2Q_UNSAFE_EXPERIMENTS=1`).
///
/// Returns `true` for diagonal cells (e.g. `imatrix` + `k-quant-q4_k_m`
/// is the same as `--quant imatrix-q4_k_m`).
pub fn is_diagonal_cell(cal: CalibrationFlag, fmt: OutputFormatFlag) -> bool {
    use CalibrationFlag::*;
    use OutputFormatFlag::*;
    matches!(
        (cal, fmt),
        // none × flat-* / k-quant-* (uncalibrated cells)
        (None, FlatF16)
            | (None, FlatBf16)
            | (None, FlatQ2)
            | (None, FlatQ4)
            | (None, FlatQ8)
            | (None, KQuantQ4KM)
            | (None, KQuantQ5KM)
            | (None, KQuantQ6K)
            // imatrix × k-quant-* (calibrated K-quant cells)
            | (Imatrix, KQuantQ4KM)
            | (Imatrix, KQuantQ5KM)
            | (Imatrix, KQuantQ6K)
            | (Imatrix, KQuantAdaptive)
            // dwq × bit-pair-* (DWQ cells)
            | (Dwq, BitPair46)
            | (Dwq, BitPair48)
            | (Dwq, BitPair68)
            | (Dwq, BitPair28)
    )
}

/// Validate the orthogonal `(--calibration, --output-format)` selector
/// per ADR-014 Decision 12 §off-diagonal.
///
/// - Both flags `None` → caller is using `--quant`; returns `Ok(None)`.
/// - One flag `Some`, the other `None` → returns
///   [`OffDiagonalError::PartialOrthogonalSelector`] (also caught by
///   clap's `requires`, but the typed error here is the library
///   contract).
/// - Both flags `Some` AND diagonal cell → returns `Ok(Some((cal, fmt)))`
///   with no env-gate check (a diagonal cell is just a verbose `--quant`).
/// - Both flags `Some` AND off-diagonal cell AND env unset → returns
///   [`OffDiagonalError::DevGateRequired`].
/// - Both flags `Some` AND off-diagonal cell AND `HF2Q_UNSAFE_EXPERIMENTS=1`
///   → returns `Ok(Some((cal, fmt)))`.
pub fn validate_off_diagonal_selector(
    cal: Option<CalibrationFlag>,
    fmt: Option<OutputFormatFlag>,
    env_unsafe_experiments: Option<&str>,
) -> Result<Option<(CalibrationFlag, OutputFormatFlag)>, OffDiagonalError> {
    match (cal, fmt) {
        (None, None) => Ok(None),
        (Some(_), None) | (None, Some(_)) => {
            Err(OffDiagonalError::PartialOrthogonalSelector {
                env: HF2Q_UNSAFE_EXPERIMENTS_ENV,
                cal,
                fmt,
            })
        }
        (Some(c), Some(f)) => {
            if is_diagonal_cell(c, f) {
                return Ok(Some((c, f)));
            }
            // Off-diagonal cell — env gate required.
            if env_unsafe_experiments == Some("1") {
                Ok(Some((c, f)))
            } else {
                Err(OffDiagonalError::DevGateRequired {
                    env: HF2Q_UNSAFE_EXPERIMENTS_ENV,
                    cal: c.to_string(),
                    fmt: f.to_string(),
                })
            }
        }
    }
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
        assert_eq!(QuantMethod::Dwq46.dwq_bit_pair(), Some((4, 6)));
        assert_eq!(QuantMethod::Dwq48.dwq_bit_pair(), Some((4, 8)));
        assert_eq!(QuantMethod::Dwq68.dwq_bit_pair(), Some((6, 8)));
        assert_eq!(QuantMethod::Dwq28.dwq_bit_pair(), Some((2, 8)));
    }

    #[test]
    fn test_non_dwq_variants_return_none_for_bit_pair() {
        // Every non-DWQ variant in the 17-variant menu returns None.
        for m in [
            QuantMethod::Auto,
            QuantMethod::F16,
            QuantMethod::Bf16,
            QuantMethod::Q2,
            QuantMethod::Q4,
            QuantMethod::Q8,
            QuantMethod::Q4KM,
            QuantMethod::Q5KM,
            QuantMethod::Q6K,
            QuantMethod::ImatrixQ4KM,
            QuantMethod::ImatrixQ5KM,
            QuantMethod::ImatrixQ6K,
            QuantMethod::ImatrixAdaptive,
        ] {
            assert_eq!(
                m.dwq_bit_pair(),
                None,
                "non-DWQ variant {m:?} must return None for dwq_bit_pair"
            );
        }
    }

    // ---- Default filename suffix helper ----

    #[test]
    fn test_dwq_default_filename_suffix() {
        assert_eq!(QuantMethod::Dwq46.default_filename_suffix(), "dwq46");
        assert_eq!(QuantMethod::Dwq48.default_filename_suffix(), "dwq48");
        assert_eq!(QuantMethod::Dwq68.default_filename_suffix(), "dwq68");
        assert_eq!(QuantMethod::Dwq28.default_filename_suffix(), "dwq28");
    }

    #[test]
    fn test_non_dwq_filename_suffix_byte_identical_to_display() {
        // All non-DWQ variants must return their Display string unchanged.
        for m in [
            QuantMethod::Auto,
            QuantMethod::F16,
            QuantMethod::Bf16,
            QuantMethod::Q2,
            QuantMethod::Q4,
            QuantMethod::Q8,
            QuantMethod::Q4KM,
            QuantMethod::Q5KM,
            QuantMethod::Q6K,
            QuantMethod::ImatrixQ4KM,
            QuantMethod::ImatrixQ5KM,
            QuantMethod::ImatrixQ6K,
            QuantMethod::ImatrixAdaptive,
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
        assert_eq!(QuantMethod::Dwq46.to_string(), "dwq-4-6");
        assert_eq!(QuantMethod::Dwq48.to_string(), "dwq-4-8");
        assert_eq!(QuantMethod::Dwq68.to_string(), "dwq-6-8");
        assert_eq!(QuantMethod::Dwq28.to_string(), "dwq-2-8");
    }

    // ---- Gemma-4 regression snapshot guard ----
    // Asserts that the Dwq46 dispatch path continues to produce
    // (base_bits=4, sensitive_bits=6) — byte-identical to pre-change behaviour.
    // This guard will fail if any future refactor silently changes the Gemma-4
    // quantization parameters.

    /// ADR-014 P8 iter-15: Q4_K_S + Q5_K_S CLI surface — uncalibrated
    /// + imatrix variants must Display correctly and report
    /// is_quantized=true + dwq_bit_pair=None.  Mirrors iter-10's
    /// Q3 surface lock for the new "small" K-variant CLI exposures.
    #[test]
    fn test_q4ks_q5ks_cli_surface_iter15() {
        // Display strings match canonical wire names.
        assert_eq!(QuantMethod::Q4KS.to_string(), "q4_k_s");
        assert_eq!(QuantMethod::Q5KS.to_string(), "q5_k_s");
        assert_eq!(QuantMethod::ImatrixQ4KS.to_string(), "imatrix-q4_k_s");
        assert_eq!(QuantMethod::ImatrixQ5KS.to_string(), "imatrix-q5_k_s");

        for v in [
            QuantMethod::Q4KS,
            QuantMethod::Q5KS,
            QuantMethod::ImatrixQ4KS,
            QuantMethod::ImatrixQ5KS,
        ] {
            assert_eq!(v.dwq_bit_pair(), None, "{v:?} should not be DWQ");
            assert!(v.is_quantized(), "{v:?} should be quantized");
        }

        // Filename suffix follows Display (no DWQ-style compaction).
        assert_eq!(QuantMethod::Q4KS.default_filename_suffix(), "q4_k_s");
        assert_eq!(QuantMethod::Q5KS.default_filename_suffix(), "q5_k_s");
        assert_eq!(QuantMethod::ImatrixQ4KS.default_filename_suffix(), "imatrix-q4_k_s");
        assert_eq!(QuantMethod::ImatrixQ5KS.default_filename_suffix(), "imatrix-q5_k_s");
    }

    /// ADR-014 P8 iter-10: Q3_K family must round-trip through Display
    /// + dwq_bit_pair (None for non-DWQ) + filename suffix.  Locks the
    /// CLI surface for `q3_k_s`/`q3_k_m`/`q3_k_l` (and their imatrix
    /// counterparts) so a future refactor can't silently rename them.
    #[test]
    fn test_q3_k_family_cli_surface_iter10() {
        // Display strings match the canonical wire names.
        assert_eq!(QuantMethod::Q3KS.to_string(), "q3_k_s");
        assert_eq!(QuantMethod::Q3KM.to_string(), "q3_k_m");
        assert_eq!(QuantMethod::Q3KL.to_string(), "q3_k_l");
        assert_eq!(QuantMethod::ImatrixQ3KS.to_string(), "imatrix-q3_k_s");
        assert_eq!(QuantMethod::ImatrixQ3KM.to_string(), "imatrix-q3_k_m");
        assert_eq!(QuantMethod::ImatrixQ3KL.to_string(), "imatrix-q3_k_l");

        // Q3 family is K-quant, not DWQ.
        for v in [
            QuantMethod::Q3KS,
            QuantMethod::Q3KM,
            QuantMethod::Q3KL,
            QuantMethod::ImatrixQ3KS,
            QuantMethod::ImatrixQ3KM,
            QuantMethod::ImatrixQ3KL,
        ] {
            assert_eq!(v.dwq_bit_pair(), None, "{v:?} should not be DWQ");
            assert!(v.is_quantized(), "{v:?} should report is_quantized");
        }

        // Default filename suffix matches Display (no special-case for K-quant).
        assert_eq!(QuantMethod::Q3KS.default_filename_suffix(), "q3_k_s");
        assert_eq!(QuantMethod::Q3KM.default_filename_suffix(), "q3_k_m");
        assert_eq!(QuantMethod::Q3KL.default_filename_suffix(), "q3_k_l");
        assert_eq!(QuantMethod::ImatrixQ3KS.default_filename_suffix(), "imatrix-q3_k_s");
    }

    #[test]
    fn test_gemma4_regression_dwq46_dispatch_unchanged() {
        // Dispatch table entry must still be (4, 6).
        assert_eq!(
            QuantMethod::Dwq46.dwq_bit_pair(),
            Some((4, 6)),
            "Gemma-4 regression: Dwq46 must dispatch to (base=4, sensitive=6)"
        );
        // Default filename suffix (new behaviour per ADR Decision 10(c)).
        assert_eq!(
            QuantMethod::Dwq46.default_filename_suffix(),
            "dwq46",
            "Gemma-4 regression: Dwq46 filename suffix must be 'dwq46'"
        );
    }

    // ---- ADR-014 P8 Decision 12: 17-variant menu unit tests ----

    /// S1 test (a): every Decision-12 variant string parses through
    /// `Cli::parse_from` and resolves to the expected `QuantMethod`
    /// variant. Drives the clap value-parser table for every menu cell.
    #[test]
    fn test_every_decision12_variant_parses() {
        // (CLI string, expected variant)
        let cases: &[(&str, QuantMethod)] = &[
            ("auto", QuantMethod::Auto),
            ("f16", QuantMethod::F16),
            ("bf16", QuantMethod::Bf16),
            ("q2", QuantMethod::Q2),
            ("q4", QuantMethod::Q4),
            ("q8", QuantMethod::Q8),
            ("q4_k_m", QuantMethod::Q4KM),
            ("q5_k_m", QuantMethod::Q5KM),
            ("q6_k", QuantMethod::Q6K),
            ("imatrix-q4_k_m", QuantMethod::ImatrixQ4KM),
            ("imatrix-q5_k_m", QuantMethod::ImatrixQ5KM),
            ("imatrix-q6_k", QuantMethod::ImatrixQ6K),
            ("imatrix-adaptive", QuantMethod::ImatrixAdaptive),
            ("dwq-4-6", QuantMethod::Dwq46),
            ("dwq-4-8", QuantMethod::Dwq48),
            ("dwq-6-8", QuantMethod::Dwq68),
            ("dwq-2-8", QuantMethod::Dwq28),
        ];
        assert_eq!(
            cases.len(),
            17,
            "Decision-12 menu must enumerate exactly 17 variants"
        );
        for (s, expected) in cases {
            let cli = Cli::try_parse_from([
                "hf2q",
                "convert",
                "--input",
                "/tmp/x",
                "--format",
                "gguf",
                "--quant",
                s,
            ])
            .unwrap_or_else(|e| {
                panic!("variant {s} must parse, but clap rejected it: {e}")
            });
            let Command::Convert(args) = cli.command else {
                panic!("expected Convert subcommand");
            };
            assert_eq!(
                args.quant, *expected,
                "variant {s} parsed to {:?} but expected {:?}",
                args.quant, expected,
            );
        }
    }

    /// S1 test (b): `apex` is rejected by clap; the deletion-hint mapper
    /// produces a useful "did you mean imatrix-adaptive" message.
    #[test]
    fn test_apex_string_rejected_with_hint() {
        let result = Cli::try_parse_from([
            "hf2q",
            "convert",
            "--input",
            "/tmp/x",
            "--format",
            "gguf",
            "--quant",
            "apex",
        ]);
        assert!(result.is_err(), "apex must be rejected by clap");

        let hint = map_deleted_quant_hint("apex").expect("apex must have a hint");
        assert!(
            hint.contains("imatrix-adaptive"),
            "apex hint must suggest imatrix-adaptive, got: {hint}"
        );
    }

    /// S1 test (c): each of the deleted `mixed-N-M` variants is rejected
    /// by clap and the hint mapper points at `q4_k_m` / `imatrix-q4_k_m`.
    #[test]
    fn test_mixed_4_6_rejected_with_hint() {
        for raw in ["mixed-2-6", "mixed-3-6", "mixed-4-6"] {
            let result = Cli::try_parse_from([
                "hf2q",
                "convert",
                "--input",
                "/tmp/x",
                "--format",
                "gguf",
                "--quant",
                raw,
            ]);
            assert!(result.is_err(), "{raw} must be rejected by clap");

            let hint = map_deleted_quant_hint(raw)
                .unwrap_or_else(|| panic!("{raw} must have a hint"));
            assert!(
                hint.contains("q4_k_m"),
                "{raw} hint must suggest q4_k_m / imatrix-q4_k_m, got: {hint}"
            );
        }
    }

    /// S1 test (d): each of the renamed `dwq-mixed-N-M` variants is
    /// rejected by clap and the hint mapper points at `dwq-N-M`.
    #[test]
    fn test_dwq_mixed_4_6_rejected_with_hint() {
        let cases = [
            ("dwq-mixed-4-6", "dwq-4-6"),
            ("dwq-mixed-4-8", "dwq-4-8"),
            ("dwq-mixed-6-8", "dwq-6-8"),
            ("dwq-mixed-2-8", "dwq-2-8"),
        ];
        for (raw, expected_suggest) in cases {
            let result = Cli::try_parse_from([
                "hf2q",
                "convert",
                "--input",
                "/tmp/x",
                "--format",
                "gguf",
                "--quant",
                raw,
            ]);
            assert!(result.is_err(), "{raw} must be rejected by clap");

            let hint = map_deleted_quant_hint(raw)
                .unwrap_or_else(|| panic!("{raw} must have a hint"));
            assert!(
                hint.contains(expected_suggest),
                "{raw} hint must suggest {expected_suggest}, got: {hint}"
            );
        }
    }

    /// S1 test (e): every variant Display string round-trips through
    /// the clap value parser back to the same enum variant. Catches any
    /// drift between `#[value(name = ...)]` and `Display`.
    #[test]
    fn test_display_round_trip_for_all_17() {
        let all_variants = [
            QuantMethod::Auto,
            QuantMethod::F16,
            QuantMethod::Bf16,
            QuantMethod::Q2,
            QuantMethod::Q4,
            QuantMethod::Q8,
            QuantMethod::Q4KM,
            QuantMethod::Q5KM,
            QuantMethod::Q6K,
            QuantMethod::ImatrixQ4KM,
            QuantMethod::ImatrixQ5KM,
            QuantMethod::ImatrixQ6K,
            QuantMethod::ImatrixAdaptive,
            QuantMethod::Dwq46,
            QuantMethod::Dwq48,
            QuantMethod::Dwq68,
            QuantMethod::Dwq28,
        ];
        assert_eq!(all_variants.len(), 17, "expected 17 variants");
        for v in all_variants {
            let s = v.to_string();
            let cli = Cli::try_parse_from([
                "hf2q",
                "convert",
                "--input",
                "/tmp/x",
                "--format",
                "gguf",
                "--quant",
                &s,
            ])
            .unwrap_or_else(|e| {
                panic!(
                    "Display->parse round trip failed for {v:?} (\"{s}\"): {e}"
                )
            });
            let Command::Convert(args) = cli.command else {
                panic!("expected Convert");
            };
            assert_eq!(
                args.quant, v,
                "Display->parse round trip resolved \"{s}\" to {:?}, expected {v:?}",
                args.quant,
            );
        }
    }

    /// ADR-014 P8 iter-111 (2026-04-29) — extend the iter-1 17-variant
    /// round-trip gate to cover the full menu the QuantMethod enum
    /// actually exposes today (32 variants). Pre-iter-111 the round-trip
    /// guard was stuck at the "Decision 12 final 17" set; the K-quant
    /// `_S` family (Q2_K_S, Q3_K_S/M/L, Q4_K_S, Q5_K_S, Q6_K) + the
    /// matching `imatrix-*_S` variants + the alias `dwq` (Decision 12
    /// alias for `dwq-mixed-4-6`) were never round-trip-checked. iter-111
    /// closes that hole so a future Display ↔ parse drift fails the
    /// CI gate immediately.
    #[test]
    fn test_display_round_trip_for_all_quant_variants_iter111() {
        let all_variants: Vec<QuantMethod> = vec![
            // Auto + base + legacy (6).
            QuantMethod::Auto,
            QuantMethod::F16,
            QuantMethod::Bf16,
            QuantMethod::Q2,
            QuantMethod::Q4,
            QuantMethod::Q8,
            // K-quant family (10): all S/M/L sub-variants for Q2_K..Q5_K + Q6_K.
            QuantMethod::Q2KS,
            QuantMethod::Q2K,
            QuantMethod::Q3KS,
            QuantMethod::Q3KM,
            QuantMethod::Q3KL,
            QuantMethod::Q4KS,
            QuantMethod::Q4KM,
            QuantMethod::Q5KS,
            QuantMethod::Q5KM,
            QuantMethod::Q6K,
            // Imatrix-K family (10): same shape as K-quant family.
            QuantMethod::ImatrixQ2KS,
            QuantMethod::ImatrixQ2K,
            QuantMethod::ImatrixQ3KS,
            QuantMethod::ImatrixQ3KM,
            QuantMethod::ImatrixQ3KL,
            QuantMethod::ImatrixQ4KS,
            QuantMethod::ImatrixQ4KM,
            QuantMethod::ImatrixQ5KS,
            QuantMethod::ImatrixQ5KM,
            QuantMethod::ImatrixQ6K,
            // Imatrix-adaptive (Apex per-tensor optimal precision) (1).
            QuantMethod::ImatrixAdaptive,
            // DWQ family (4): four bit-pair variants. The unparameterized
            // `Dwq` alias lives in `CalibrationFlag` (the orthogonal
            // `--calibration` axis from Decision 12 §off-diagonal), not
            // in `QuantMethod`, so it doesn't round-trip through `--quant`.
            QuantMethod::Dwq46,
            QuantMethod::Dwq48,
            QuantMethod::Dwq68,
            QuantMethod::Dwq28,
        ];
        assert_eq!(
            all_variants.len(),
            31,
            "expected the full 31-variant menu (6 base + 10 K-quant + 10 imatrix-K + ImatrixAdaptive + 4 DWQ)"
        );
        for v in all_variants {
            let s = v.to_string();
            let cli = Cli::try_parse_from([
                "hf2q",
                "convert",
                "--input",
                "/tmp/x",
                "--format",
                "gguf",
                "--quant",
                &s,
            ])
            .unwrap_or_else(|e| {
                panic!("iter-111 Display→parse round trip failed for {v:?} (\"{s}\"): {e}")
            });
            let Command::Convert(args) = cli.command else {
                panic!("iter-111: expected Convert subcommand for variant {v:?}");
            };
            assert_eq!(
                args.quant, v,
                "iter-111 Display→parse round trip resolved \"{s}\" to {:?}, expected {v:?}",
                args.quant,
            );
        }
    }

    // ---- ADR-014 P8 Decision 12 §off-diagonal: --calibration / --output-format ----

    /// S4 test (a): supplying `--calibration` alone (without
    /// `--output-format`) is rejected by clap's `requires`.
    #[test]
    fn calibration_flag_alone_without_env_rejected() {
        let result = Cli::try_parse_from([
            "hf2q",
            "convert",
            "--input",
            "/tmp/x",
            "--format",
            "gguf",
            "--calibration",
            "imatrix",
        ]);
        assert!(
            result.is_err(),
            "--calibration without --output-format must be rejected"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("output-format") || err_msg.contains("output_format"),
            "error must cite --output-format requirement, got: {err_msg}"
        );
    }

    /// S4 test (b): both flags supplied with an off-diagonal cell and
    /// no env var is rejected with a doc pointer.
    #[test]
    fn both_flags_without_env_rejected_with_doc_pointer() {
        // imatrix + bit-pair-4-6 is OFF-diagonal (DWQ codec with imatrix calibrator).
        let err = validate_off_diagonal_selector(
            Some(CalibrationFlag::Imatrix),
            Some(OutputFormatFlag::BitPair46),
            None, // env unset
        )
        .expect_err("off-diagonal cell without env must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("HF2Q_UNSAFE_EXPERIMENTS"),
            "error must mention HF2Q_UNSAFE_EXPERIMENTS, got: {msg}"
        );
        assert!(
            msg.contains("docs/converting-a-model.md"),
            "error must include doc pointer, got: {msg}"
        );
    }

    /// S4 test (c): both flags supplied with an off-diagonal cell and
    /// `HF2Q_UNSAFE_EXPERIMENTS=1` is accepted.
    #[test]
    fn both_flags_with_env_accepted_off_diagonal() {
        // Off-diagonal: dwq calibrator with k-quant-q4_k_m output.
        let pair = validate_off_diagonal_selector(
            Some(CalibrationFlag::Dwq),
            Some(OutputFormatFlag::KQuantQ4KM),
            Some("1"),
        )
        .expect("off-diagonal cell with env=1 must be accepted");
        assert_eq!(
            pair,
            Some((CalibrationFlag::Dwq, OutputFormatFlag::KQuantQ4KM))
        );

        // Diagonal cells don't need the env at all.
        let pair2 = validate_off_diagonal_selector(
            Some(CalibrationFlag::Imatrix),
            Some(OutputFormatFlag::KQuantQ4KM),
            None,
        )
        .expect("diagonal cell must be accepted without env");
        assert_eq!(
            pair2,
            Some((CalibrationFlag::Imatrix, OutputFormatFlag::KQuantQ4KM))
        );
    }

    /// S4 test (d): supplying `--quant` with `--calibration` is
    /// mutually exclusive (clap's conflicts_with). Note: clap rejects
    /// the combo even when `--quant` is just defaulted, because
    /// `conflicts_with` triggers when both are SUPPLIED via flags;
    /// hence the test passes both explicitly.
    #[test]
    fn calibration_conflicts_with_quant_clap_rejects() {
        let result = Cli::try_parse_from([
            "hf2q",
            "convert",
            "--input",
            "/tmp/x",
            "--format",
            "gguf",
            "--quant",
            "q4_k_m",
            "--calibration",
            "imatrix",
            "--output-format",
            "k-quant-q4_k_m",
        ]);
        assert!(
            result.is_err(),
            "--quant + --calibration must be mutually exclusive"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.to_lowercase().contains("conflict")
                || err_msg.contains("cannot be used"),
            "error must indicate conflict, got: {err_msg}"
        );
    }

    /// Diagonal-cell helper coverage: every diagonal cell in
    /// [`is_diagonal_cell`] must be reachable without the env gate.
    #[test]
    fn is_diagonal_cell_covers_documented_cells() {
        use CalibrationFlag::*;
        use OutputFormatFlag::*;
        let diagonal_cells: &[(CalibrationFlag, OutputFormatFlag)] = &[
            // none × flat / k-quant
            (None, FlatF16),
            (None, FlatBf16),
            (None, FlatQ2),
            (None, FlatQ4),
            (None, FlatQ8),
            (None, KQuantQ4KM),
            (None, KQuantQ5KM),
            (None, KQuantQ6K),
            // imatrix × k-quant
            (Imatrix, KQuantQ4KM),
            (Imatrix, KQuantQ5KM),
            (Imatrix, KQuantQ6K),
            (Imatrix, KQuantAdaptive),
            // dwq × bit-pair
            (Dwq, BitPair46),
            (Dwq, BitPair48),
            (Dwq, BitPair68),
            (Dwq, BitPair28),
        ];
        for (c, f) in diagonal_cells {
            assert!(
                is_diagonal_cell(*c, *f),
                "diagonal cell ({c:?}, {f:?}) must be recognised"
            );
        }
        // A representative off-diagonal cell.
        assert!(
            !is_diagonal_cell(Imatrix, BitPair46),
            "imatrix + bit-pair-4-6 must be off-diagonal"
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
            calibration: None,
            output_format: None,
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
            shard_size_gb: 5.0,
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
            calibration: None,
            output_format: None,
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
            shard_size_gb: 5.0,
        };
        let cfg = resolve_convert_config(&args).unwrap();
        assert!(!cfg.no_integrity);
    }

    /// ADR-014 P9 iter-1 §S5 — `--shard-size-gb` parses, validates the
    /// 0.5..=50.0 range, and propagates into ConvertConfig.
    #[test]
    fn shard_size_gb_flag_parses_and_propagates() {
        // Default value when not set on the CLI surface.
        let tmp = tempfile::tempdir().unwrap();
        let args = ConvertArgs {
            input: Some(tmp.path().to_path_buf()),
            repo: None,
            format: OutputFormat::Safetensors,
            quant: QuantMethod::Dwq46,
            calibration: None,
            output_format: None,
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
            shard_size_gb: 2.5,
        };
        let cfg = resolve_convert_config(&args).unwrap();
        assert!((cfg.shard_size_gb - 2.5).abs() < 1e-9);

        // Explicit parser exercise — out-of-range must error.
        assert!(parse_shard_size_gb("0.4").is_err());
        assert!(parse_shard_size_gb("50.5").is_err());
        assert!(parse_shard_size_gb("not-a-number").is_err());
        assert!(parse_shard_size_gb("inf").is_err());

        // Boundary values must parse.
        assert!((parse_shard_size_gb("0.5").unwrap() - 0.5).abs() < 1e-9);
        assert!((parse_shard_size_gb("50").unwrap() - 50.0).abs() < 1e-9);
        assert!((parse_shard_size_gb("5.0").unwrap() - 5.0).abs() < 1e-9);
    }
}
