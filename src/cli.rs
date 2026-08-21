//! CLI argument parsing and validation via clap derive API.
//!
//! All environment variable resolution happens here at startup.
//! No global state — CLI args produce a `ConvertConfig` that flows through the pipeline.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
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

    /// Absolute hf2q state root containing config.toml. Defaults to
    /// `$HOME/.hf2q`. Setup writes this file; convert and serve read it.
    #[arg(long, value_name = "ABSOLUTE_PATH", global = true)]
    pub state_root: Option<PathBuf>,

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
    /// Internal bootstrap used only by the reviewed standalone installer.
    #[command(name = "__standalone-install", hide = true)]
    StandaloneInstall(StandaloneInstallArgs),

    /// Update a standalone installation, or restore its retained previous version.
    Update(UpdateArgs),

    /// Remove a standalone hf2q executable while preserving config and models.
    Uninstall(UninstallArgs),

    /// Learn this Mac and record defaults used by convert and serve.
    Setup(SetupArgs),

    /// Patch an existing GGUF file's metadata without changing tensor bytes
    GgufPatch(GgufPatchArgs),

    /// Inspect model metadata before converting
    Info(InfoArgs),

    /// Diagnose RuVector, hardware detection, and disk space
    Doctor,

    /// Generate shell completions
    Completions(CompletionsArgs),

    /// Run text generation from a GGUF model
    Generate(GenerateArgs),

    /// Chat with a served model using a diagnostic scrollback interface
    Chat(ChatArgs),

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

    /// ADR-033 — convert a HuggingFace model directory to GGUF via
    /// the unified policy/quantizer/writer pipeline (the only convert
    /// path post-P6). Per [[feedback-no-backwards-compat-2026-05-18]]:
    /// no migration shims; no alias for legacy `--quant` values; no
    /// `convert-v2` alias (the historical name retired 2026-05-19 via
    /// B4 rename).
    Convert(ConvertCliArgs),

    /// Operate on bundled `tokenizer.json` files (ADR-038 G4-CFA-5e).
    ///
    /// Currently provides `fix-bos`, which patches a `tokenizer.json`'s
    /// `post_processor.single` template + `special_tokens` map so that
    /// `tokenizer.encode(text, add_special_tokens=true)` prepends BOS —
    /// matching how most modern Gemma / Qwen tokenizers ship by default.
    ///
    /// The runtime adapter (`core::tokenizer_adapter::tokenize_with_bos_eos_from_gguf`)
    /// is the load-bearing fix for hf2q's own dispatch paths; this CLI
    /// surface is a belt-and-suspenders option for operators who feed
    /// hf2q's bundled `tokenizer.json` files to other tools.
    Tokenizer(TokenizerArgs),
}

#[derive(clap::Args, Debug, Clone)]
pub struct StandaloneInstallArgs {
    /// Existing canonical user-owned directory that will contain hf2q.
    #[arg(long, value_name = "ABSOLUTE_PATH")]
    pub install_dir: PathBuf,

    /// Downloaded candidate executable already accepted by release policy.
    #[arg(long, value_name = "ABSOLUTE_PATH")]
    pub candidate: PathBuf,

    /// Exact candidate byte length from the immutable release record.
    #[arg(long)]
    pub size: u64,

    /// Exact lowercase SHA-256 from the immutable release record.
    #[arg(long, value_name = "LOWERCASE_HEX")]
    pub sha256: String,
}

#[derive(clap::Args, Debug, Clone)]
pub struct UninstallArgs {
    /// Confirm removal of the standalone executable and its channel files.
    #[arg(long)]
    pub yes: bool,
}

#[derive(clap::Args, Debug, Clone)]
pub struct UpdateArgs {
    /// Report whether a newer stable standalone release exists without installing it.
    #[arg(long, conflicts_with = "rollback")]
    pub check: bool,

    /// Atomically restore the one previous standalone executable.
    #[arg(long, conflicts_with = "check")]
    pub rollback: bool,
}

#[derive(clap::Args, Debug, Clone)]
pub struct SetupArgs {
    /// Default quant selector used by `hf2q convert` when `--quant` is omitted.
    #[arg(long, value_name = "QUANT")]
    pub default_quant: Option<String>,

    /// Default serve bind host. Setup accepts only 127.0.0.1 or 0.0.0.0.
    #[arg(long, value_name = "HOST")]
    pub serve_host: Option<String>,

    /// Default serve port.
    #[arg(long, value_name = "PORT")]
    pub serve_port: Option<u16>,

    /// Default serve scheduler.
    #[arg(long, value_enum, value_name = "POLICY")]
    pub serve_scheduler: Option<SchedulerArg>,

    /// Default active-request slots for inflight-batched serving.
    #[arg(long, value_name = "N")]
    pub serve_max_slots: Option<u32>,

    /// Accept the displayed values without prompting. A fresh config uses the
    /// canonical Qwen3.8 guide defaults; a rerun retains current values.
    /// Explicit setup flags override individual values.
    #[arg(long, default_value_t = false)]
    pub accept_defaults: bool,
}

#[derive(clap::Args, Debug)]
pub struct TokenizerArgs {
    #[command(subcommand)]
    pub action: TokenizerAction,
}

#[derive(Subcommand, Debug)]
pub enum TokenizerAction {
    /// Patch `<path>` in place so its `post_processor.single` template
    /// prepends BOS. Idempotent: re-runs are no-ops. Reads BOS metadata
    /// (token id + text) from a sibling GGUF when `--gguf <path>` is
    /// given; otherwise the operator must supply `--bos-id` + `--bos-text`.
    FixBos {
        /// Path to the `tokenizer.json` file to patch.
        #[arg(value_name = "TOKENIZER_JSON")]
        path: PathBuf,

        /// Optional sibling GGUF — when supplied, reads
        /// `tokenizer.ggml.bos_token_id` and resolves the token text
        /// via the same logic as the runtime adapter.
        #[arg(long, value_name = "GGUF")]
        gguf: Option<PathBuf>,

        /// Override the BOS token id (when no `--gguf` is given).
        /// Default `2` matches Gemma family.
        #[arg(long, default_value_t = 2)]
        bos_id: u32,

        /// Override the BOS token text. Default `<bos>`.
        #[arg(long, default_value = "<bos>")]
        bos_text: String,
    },
}

/// `hf2q convert <hf-dir> --quant <name> -o <out.gguf>` clap args.
///
/// Resolved to [`crate::convert::ConvertArgs`] in `main.rs::cmd_convert`.
#[derive(clap::Args, Debug, Clone)]
pub struct ConvertCliArgs {
    /// Local Hugging Face model directory or public model reference.
    /// Existing paths and explicit `./`, `../`, or absolute paths remain
    /// local. A non-path repository ID or canonical huggingface.co model,
    /// tree, blob, or resolve URL is downloaded natively after its revision
    /// is resolved to an immutable commit. Mutually exclusive with `--repo`.
    #[arg(conflicts_with = "repo")]
    pub hf_dir: Option<PathBuf>,

    /// Compatibility spelling for a Hugging Face model reference. Downloads
    /// use hf2q's in-process `hf-hub` client and standard Hub cache/token
    /// discovery; no Python or external Hub CLI is required.
    #[arg(long, conflicts_with = "hf_dir")]
    pub repo: Option<String>,

    /// Optional branch, tag, or full commit for a remote positional reference
    /// or `--repo`. It must match any URL-embedded revision and is resolved to
    /// a full immutable commit before transfer. Invalid for local paths.
    #[arg(long, value_name = "REVISION")]
    pub revision: Option<String>,

    /// Provenance repo for an already-downloaded positional `<hf_dir>`.
    /// This does not download anything; hf2q verifies the local bytes against
    /// the exact Hugging Face revision before conversion and writes the same
    /// immutable receipt produced by `--repo` conversions.
    #[arg(
        long,
        requires = "hf_dir",
        conflicts_with = "repo",
        value_name = "HF_REPO"
    )]
    pub source_repo: Option<String>,

    /// Exact 40-hex Hugging Face commit for `--source-repo`.
    #[arg(
        long,
        requires = "source_repo",
        conflicts_with = "repo",
        value_name = "REVISION"
    )]
    pub source_revision: Option<String>,

    /// File-type to quantize to. Accepts:
    ///   - Standard llama.cpp ftypes: `f32`, `f16`, `bf16`, `q4_0`,
    ///     `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2_k`, `q3_k_s/m/l`,
    ///     `q4_k_s/m`, `q5_k_s/m`, `q6_k`, `iq4_nl`.
    ///   - Apex algorithmic tiers (MoE arches only): `apex-quality`,
    ///     `apex-i-quality`, `apex-balanced`, `apex-i-balanced`,
    ///     `apex-compact`, `apex-i-compact`, `apex-mini`.
    ///   - DeepSeek-V4 agent profile: `deepseek4-agentic-q2` (Q2_K
    ///     expert gate/up, Q3_K expert down, Q8_0 context path).
    ///
    /// Reserved / out-of-scope names (per ADR §P7 AC#1 + Decision §6).
    /// Each surfaces a typed `QuantSelectorError` variant with an
    /// operator-actionable hint pointing at the supported alternative:
    ///   - `dwq` → `DwqReserved` (reserved until ADR-046's native
    ///     affine artifact, trainer, quality, and runtime gates land).
    ///   - `tq1_0`, `tq2_0` → `TqOutOfV1Scope` (recognized ftypes
    ///     without a Quantizer impl in v1).
    ///   - bare `apex` → `ApexUnqualified` (must pick an explicit
    ///     tier; no implicit default).
    ///   - `apex-custom` → `ApexCustomRequiresTensorTypeFile`
    ///     (requires `--tensor-type-file <path>`).
    ///   - `apex-nano`, `apex-i-nano`, `apex-micro`, `apex-i-micro` →
    ///     `ApexTierOutOfScope` (mudler experimental tiers dropped
    ///     from v1; reach via `apex-custom`).
    ///   - any other `apex-<x>` → `UnknownApexTier` (lists the
    ///     supported tier set in the message).
    ///
    /// Parsed via `QuantSelector::from_name`; unrecognized values
    /// surface as input errors. When omitted, the value recorded by
    /// `hf2q setup` is used; without either source, conversion fails and
    /// asks the operator to choose. Per
    /// [[feedback-no-backwards-compat-2026-05-18]]: no legacy aliases.
    #[arg(long)]
    pub quant: Option<String>,

    /// Destination GGUF file. Existing files are overwritten.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Build and report the exact tensor/type/byte plan without creating the
    /// output GGUF. Source download or provenance hashing may still read the
    /// complete input; no tensor payload is materialized by the converter.
    #[arg(
        long,
        default_value_t = false,
        conflicts_with_all = ["imatrix_corpus", "imatrix_out"]
    )]
    pub dry_run: bool,

    /// Pre-computed imatrix file (`.imatrix.gguf`). Required for I-tier
    /// APEX variants (`apex-i-quality`, `apex-i-balanced`, `apex-i-compact`)
    /// per ADR-033 §Pi. Mutually exclusive with `--imatrix-corpus`.
    ///
    /// For supported architectures, `--imatrix-corpus cdv3` is the
    /// recommended alternative. It drives hf2q's own forward-pass-based
    /// generator (ADR-033 §Pi Phase B). A pre-computed file remains useful
    /// for architectures that are not yet wired for in-tree generation.
    #[arg(long, conflicts_with = "imatrix_corpus")]
    pub imatrix: Option<PathBuf>,

    /// Auto-generate an imatrix in-memory during convert by running the
    /// hf2q decoder over the named calibration corpus. ADR-033 §Pi
    /// "Phase B" — SHIPPED 2026-05-19 (Stage 3c). The driver converts
    /// the source `<hf_dir>` to a temporary F16 GGUF, loads it via the
    /// per-arch inference path, tokenizes the corpus, and runs
    /// `forward_prefill` over `--imatrix-n-ctx`-sized chunks (default
    /// 512 while intercepting per-tensor activations.
    ///
    /// **Supported arches: Gemma 4 plus dense and MoE Qwen3.5-family
    /// decoders.** Other arches (MiniMax-M2, BERT, etc.) surface
    /// `ImatrixError::UnsupportedArchForDriver`. For those, use the
    /// `--imatrix <file>` flag with a pre-computed `.imatrix.gguf`.
    ///
    /// Accepted values: `cdv3` (bartowski's default, baked at compile
    /// time), `mudler` (selector parses but the corpus itself is
    /// not yet bundled — typed CorpusRead error), or
    /// `user-file:<path>` for an operator-supplied `.txt` corpus.
    ///
    /// Wall time: roughly seconds per chunk × ~100 chunks on a 26B-A4B
    /// Gemma 4 — operator-coffee-time, not CI-time.
    #[arg(long, conflicts_with = "imatrix")]
    pub imatrix_corpus: Option<String>,

    /// Optional side-effect: write the imatrix used by this convert
    /// run to the given path. Useful both for caching in-tree-computed
    /// imatrices (`--imatrix-corpus`) and for round-tripping
    /// pre-computed ones (`--imatrix <file>`). The on-disk format
    /// matches stock `llama-imatrix --output-format gguf`.
    #[arg(long)]
    pub imatrix_out: Option<PathBuf>,

    /// Context length for in-tree imatrix collection (ADR-033 §Pi
    /// Stage 3c). Only consulted when `--imatrix-corpus` is set;
    /// ignored on the `--imatrix <file>` load path (the loaded file's
    /// `imatrix.chunk_size` KV is authoritative there). Defaults to
    /// 512 to match stock `llama-imatrix -c 512`. Must be > 0; the
    /// corpus must tokenize to at least this many tokens or the
    /// driver surfaces `ImatrixError::CorpusTooShort`.
    ///
    /// Larger `n_ctx` means fewer, longer chunks per forward-pass
    /// loop — useful for matching imatrices produced by stock
    /// llama-imatrix with non-default `-c` values, but at the cost
    /// of more memory per forward pass.
    #[arg(long)]
    pub imatrix_n_ctx: Option<u32>,

    /// Export the multimodal projector (mmproj) sidecar instead of
    /// the text decoder. Mirrors canonical
    /// `python3 convert_hf_to_gguf.py --mmproj` at
    /// `/opt/llama.cpp/convert_hf_to_gguf.py:117-118`.
    ///
    /// When set, hf2q routes the input model through the per-arch
    /// mmproj mapper (currently Gemma 4 vision: SigLIP encoder +
    /// Gemma-3 MLP projector). The output GGUF carries the vision
    /// tower (`v.*`) + projector (`mm.*`) tensors and the
    /// `clip.*` metadata schema; text-decoder tensors are silently
    /// dropped (they're written separately by a non-`--mmproj` run).
    /// Conventional output filename: `<basename>-mmproj.gguf` —
    /// operator picks via `--output`.
    ///
    /// Currently supported arches: Gemma 4 multimodal
    /// (`Gemma4ForConditionalGeneration` with `vision_config`).
    /// Other multimodal arches surface
    /// `ConvertError::UnsupportedArch`.
    #[arg(long, default_value_t = false)]
    pub mmproj: bool,
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

    /// Ignore end-of-sequence token; generate the full max_tokens regardless of EOS.
    /// Used for benchmarks that need a fixed-length generation (e.g., true tg5000 vs peer).
    #[arg(long)]
    pub ignore_eos: bool,

    /// Path to a multimodal projector GGUF (mmproj). When supplied,
    /// `hf2q generate` validates the GGUF + parses `MmprojConfig` +
    /// loads the projector weights onto the Metal device, and surfaces
    /// the file in the load banner (`vision = <path> (sha256 ...)`).
    /// Mirror of `serve --mmproj`.
    #[arg(long)]
    pub mmproj: Option<PathBuf>,

    /// Path to an input image (PNG / JPEG / WebP). Requires `--mmproj`.
    /// When supplied, `hf2q generate` runs the ViT GPU forward, splices
    /// the vision embeddings into the prompt at the chat template's
    /// image markers, and decodes against the augmented sequence.
    #[arg(long)]
    pub image: Option<PathBuf>,

    /// Sampling temperature (0.0 = greedy, deterministic).
    ///
    /// Default `0.0` mirrors `--temp 0` in llama-cli's deterministic mode and
    /// gives the user predictable, complete, byte-reproducible output on every
    /// run.  Pass any positive value (e.g. `--temperature 0.8`) to opt into
    /// stochastic sampling — useful for creative-writing prompts where output
    /// diversity matters more than reproducibility, but the cost is occasional
    /// early-`<|im_end|>` stops on prompts where the model's distribution
    /// has non-trivial mass on EOS at any step.
    #[arg(long, default_value = "0.0")]
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

    /// TurboQuant KV cache bit-width (ADR-007 Path C F-6.1).
    ///
    /// Selects the Lloyd-Max codebook used to compress the KV cache:
    ///   - **8** (default): production-grade 8-bit Lloyd-Max HB SDPA. 256
    ///     centroids, ±5.0652659 range. Gate A cosine 0.9998, Gate C
    ///     1.24% PPL delta (intrinsic distortion floor — see ADR §F-2).
    ///   - **6**: opt-in research mode. 64 centroids. Wider cosine spread.
    ///   - **5**: opt-in research mode. 32 centroids. Wider PPL gap.
    ///   - **4**: legacy 4-bit nibble-packed `flash_attn_vec_tq` path
    ///     (close-section §1117 — 127-byte sourdough ceiling, 5.3% argmax
    ///     divergence, 1.55% PPL — NOT shippable as default).
    ///
    /// 16-bit TQ is intentionally **NOT supported** — at 2 bytes/element
    /// it is structurally redundant with F16 dense (`HF2Q_USE_DENSE=1`)
    /// and adds no compression benefit. See ADR §F-2 finding.
    ///
    /// When unset, falls back to `HF2Q_TQ_CODEBOOK_BITS` env var (default 8).
    /// When set, this flag wins and pre-populates the env var before the
    /// engine initializes.
    #[arg(long, value_parser = clap::builder::PossibleValuesParser::new(["4", "5", "6", "8"]))]
    pub kv_bits: Option<String>,

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

#[derive(clap::Args, Debug, Clone)]
pub struct ChatArgs {
    /// OpenAI-compatible server base URL. Without this flag, the automatic
    /// local discovery integration selects a registered hf2q server.
    #[arg(long, value_name = "URL")]
    pub url: Option<String>,

    /// Model id to request. If omitted, chat selects from the endpoint's
    /// advertised models.
    #[arg(long, value_name = "MODEL")]
    pub model: Option<String>,

    /// System message kept for this TUI session only.
    #[arg(long)]
    pub system: Option<String>,

    /// Sampling temperature. Omitted from requests unless explicitly set.
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Nucleus-sampling probability. Omitted unless explicitly set.
    #[arg(long)]
    pub top_p: Option<f32>,

    /// Maximum completion tokens. Omitted unless explicitly set.
    #[arg(long)]
    pub max_tokens: Option<usize>,

    /// Sampling seed. Omitted unless explicitly set.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Reasoning effort for endpoints that support it.
    #[arg(long, value_parser = ["low", "high", "max"])]
    pub reasoning_effort: Option<String>,

    /// Leave a server started by this chat session running on exit.
    #[arg(long, default_value_t = false)]
    pub keep_serving: bool,
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

    /// Host to bind to. Explicit CLI wins over the setup config; without
    /// either, defaults to 127.0.0.1 (localhost only) per ADR-005 Decision
    /// #7. Use `--host 0.0.0.0` to expose on the LAN; public-internet is NOT
    /// a supported deployment target (reverse-proxy assumption, Decision
    /// #13).
    #[arg(long)]
    pub host: Option<String>,

    /// Port to listen on. Explicit CLI wins over the setup config; without
    /// either, the built-in default remains 8080.
    #[arg(long)]
    pub port: Option<u16>,

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
    /// non-empty = restrictive whitelist (Decision #9).
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

    /// Legacy ADR-020 experimental overlay for a compatible mlx-affine
    /// safetensors file on top of the GGUF-loaded weights. For each
    /// trained Linear stem (`blk.{i}.attn_q`, `attn_k`, `attn_v`,
    /// `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`), the matching
    /// slot is replaced with an affine-mode `MlxQWeight` that
    /// dispatches through `qmm_affine_t_packed_simd4_b4` (mlx-native
    /// d0de92a).  Currently only dense families (Gemma 4) honor the
    /// overlay; qwen35moe MoE-expert tensors are skipped with a warning
    /// pending a production full-model affine path. hf2q currently has no
    /// DWQ producer; this narrow consumer is not `--quant dwq` support.
    /// See ADR-046.
    #[arg(long)]
    pub dwq_overlay: Option<PathBuf>,

    /// Suppress the human-readable serve load banner on stdout.
    #[arg(long, default_value_t = false)]
    pub quiet: bool,

    /// Interactive operator surface for a foreground server. `auto` uses a
    /// stable live dashboard when stderr is a terminal and preserves plain
    /// logs for pipes, services, CI, and JSON logging. `plain` always emits
    /// normal logs; `dashboard` requests the terminal UI explicitly.
    #[arg(long, value_enum, default_value = "auto")]
    pub operator_ui: OperatorUiArg,

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

    /// ADR-040 Phase C iter-4 (C4) — scheduler-policy selection for the
    /// in-process [`serve::api::engine::Engine`].
    ///
    /// `fifo_serial` (default) preserves the ADR-005 Decision #2 + #19
    /// production contract byte-for-byte: one mpsc channel, one worker
    /// thread, serialized FIFO dispatch, 429 + Retry-After on queue
    /// overflow. This is the value the operator gets when the flag and
    /// the `HF2Q_SCHEDULER` env var are BOTH unset (ADR-040 §3.6
    /// backward-compat pledge).
    ///
    /// `inflight_batched` activates ADR-040's slot-aware path
    /// ([`EngineMode::SlotAware`]). Supported families provision independent
    /// full-context cache sessions and interleave active requests. Unsupported
    /// families fail at spawn rather than silently sharing cache state.
    ///
    /// Also read from `HF2Q_SCHEDULER` if `--scheduler` is absent.
    /// CLI flag wins on conflict (mirrors `--auth-token` semantics).
    #[arg(long = "scheduler", value_name = "POLICY", value_enum)]
    pub scheduler: Option<SchedulerArg>,

    /// ADR-040 Phase C iter-4 (C4) — `max_slots` for
    /// [`EngineMode::SlotAware`].
    ///
    /// Honored ONLY when `--scheduler inflight_batched` is selected; with
    /// the default `fifo_serial` policy the engine's `max_slots` is
    /// hard-pinned to `1` (one worker, one in-flight request). Default
    /// `4` per ADR-040 §3.4 — half the reopen-trigger threshold of `8`
    /// to ship the first ramp with headroom.
    ///
    /// `0` is rejected with a clear operator-facing error at parse time
    /// (mirrors ADR-040 iter-2.5 F3a `.max(1)` discipline applied
    /// upstream of the scheduler — refuse the misconfiguration loudly
    /// rather than silently coerce). Range: `1..=u32::MAX`.
    ///
    /// Also read from `HF2Q_MAX_SLOTS` if `--max-slots` is absent.
    #[arg(long = "max-slots", value_name = "N")]
    pub max_slots: Option<u32>,

    /// Aggregate physical KV-cache residency budget shared by every
    /// full-context SlotAware agent. This value never divides or truncates a
    /// slot's logical context; admission charges each slot's retained
    /// high-water growth against the shared total. `0` or omission disables
    /// the physical-budget gate. Also read from
    /// `HF2Q_KV_CACHE_BUDGET_BYTES`; the CLI flag wins.
    #[arg(long = "kv-cache-budget-bytes", value_name = "BYTES")]
    pub kv_cache_budget_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OperatorUiArg {
    Auto,
    Dashboard,
    Plain,
}

/// CLI-facing copy of `serve::api::schema::OverflowPolicy`. Kept local to
/// avoid leaking the schema module into the CLI arg parser.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum OverflowPolicyArg {
    Reject,
    TruncateLeft,
    Summarize,
}

/// ADR-040 Phase C iter-4 (C4) — CLI-facing scheduler-policy selector.
///
/// Mirrors the on-disk env-var values `fifo_serial` and `inflight_batched`
/// — both are accepted case-insensitively by the env-var parser (per
/// `serve::mod::parse_scheduler_config_from_env`). clap's `ValueEnum` is
/// case-insensitive by default (`rename_all = "snake_case"` is the
/// implicit default for snake-case Rust variant names), so the operator
/// can type any of `fifo_serial`, `FIFO_SERIAL`, `inflight_batched`,
/// `INFLIGHT_BATCHED` at the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SchedulerArg {
    /// ADR-005 Decision #2 + #19 production path. One mpsc channel, one
    /// worker thread, serialized FIFO dispatch. Default; byte-equivalent
    /// to pre-ADR-040 (per ADR-040 §3.6).
    FifoSerial,
    /// ADR-040 slot-aware path: independent full-context sessions with a
    /// shared physical KV residency ceiling.
    InflightBatched,
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn serve_no_integrity_default_false() {
        let cli = Cli::parse_from(["hf2q", "serve"]);
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve");
        };
        assert!(!args.no_integrity);
    }

    #[test]
    fn chat_defaults_preserve_zero_hidden_request_policy() {
        let cli = Cli::parse_from(["hf2q", "chat", "--url", "http://127.0.0.1:9123"]);
        let Command::Chat(args) = cli.command else {
            panic!("expected Chat");
        };
        assert_eq!(args.url.as_deref(), Some("http://127.0.0.1:9123"));
        assert!(args.model.is_none());
        assert!(args.system.is_none());
        assert!(args.temperature.is_none());
        assert!(args.top_p.is_none());
        assert!(args.max_tokens.is_none());
        assert!(args.seed.is_none());
        assert!(args.reasoning_effort.is_none());
        assert!(!args.keep_serving);
    }

    #[test]
    fn chat_accepts_explicit_diagnostic_controls() {
        let cli = Cli::parse_from([
            "hf2q",
            "chat",
            "--url",
            "https://localhost:9443",
            "--model",
            "model-a",
            "--system",
            "diagnose only",
            "--temperature",
            "0.2",
            "--top-p",
            "0.8",
            "--max-tokens",
            "512",
            "--seed",
            "7",
            "--reasoning-effort",
            "high",
            "--keep-serving",
        ]);
        let Command::Chat(args) = cli.command else {
            panic!("expected Chat");
        };
        assert_eq!(args.model.as_deref(), Some("model-a"));
        assert_eq!(args.system.as_deref(), Some("diagnose only"));
        assert_eq!(args.temperature, Some(0.2));
        assert_eq!(args.top_p, Some(0.8));
        assert_eq!(args.max_tokens, Some(512));
        assert_eq!(args.seed, Some(7));
        assert_eq!(args.reasoning_effort.as_deref(), Some("high"));
        assert!(args.keep_serving);
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
    fn serve_operator_ui_defaults_to_auto_and_accepts_plain() {
        let cli = Cli::parse_from(["hf2q", "serve"]);
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve");
        };
        assert_eq!(args.operator_ui, OperatorUiArg::Auto);

        let cli = Cli::parse_from(["hf2q", "serve", "--operator-ui", "plain"]);
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve");
        };
        assert_eq!(args.operator_ui, OperatorUiArg::Plain);
    }

    #[test]
    fn convert_revision_is_bound_to_repo_download() {
        let cli = Cli::parse_from([
            "hf2q",
            "convert",
            "--repo",
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            "--revision",
            "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            "--quant",
            "q2_k_s",
            "-o",
            "/tmp/deepseek4.gguf",
        ]);
        let Command::Convert(args) = cli.command else {
            panic!("expected Convert");
        };
        assert_eq!(
            args.revision.as_deref(),
            Some("7872f01b1d1fe23eabc4c98b48bffcef5a386062")
        );
    }

    #[test]
    fn convert_revision_is_accepted_for_positional_remote_reference() {
        let cli = Cli::parse_from([
            "hf2q",
            "convert",
            "Qwen/Qwen3.8-27B",
            "--revision",
            "main",
            "--quant",
            "q2_k_s",
            "-o",
            "/tmp/deepseek4.gguf",
        ]);
        let Command::Convert(args) = cli.command else {
            panic!("expected Convert");
        };
        assert_eq!(args.revision.as_deref(), Some("main"));
        assert_eq!(args.hf_dir, Some(PathBuf::from("Qwen/Qwen3.8-27B")));
    }

    #[test]
    fn getting_started_qwen38_commands_parse_exactly() {
        let revision = "08c2f075b43bc06456382db6b918a3dcabdcf4dd";
        let model = "/tmp/Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf";
        let cli = Cli::parse_from(["hf2q", "setup", "--accept-defaults"]);
        let Command::Setup(args) = cli.command else {
            panic!("expected Setup");
        };
        assert!(args.accept_defaults);

        let cli = Cli::parse_from([
            "hf2q",
            "convert",
            "jenerallee78/Qwen3.8-27B-Abliterated-SFT",
            "--revision",
            revision,
            "--output",
            model,
        ]);
        let Command::Convert(args) = cli.command else {
            panic!("expected Convert");
        };
        assert_eq!(
            args.hf_dir,
            Some(PathBuf::from("jenerallee78/Qwen3.8-27B-Abliterated-SFT"))
        );
        assert_eq!(args.revision.as_deref(), Some(revision));
        assert!(args.quant.is_none());
        assert_eq!(args.output, PathBuf::from(model));

        let cli = Cli::parse_from(["hf2q", "serve", "--model", model]);
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve");
        };
        assert_eq!(args.model, Some(PathBuf::from(model)));
        assert!(args.host.is_none());
        assert!(args.port.is_none());
        assert!(args.scheduler.is_none());
        assert!(args.max_slots.is_none());
        assert!(args.mmproj.is_none());
    }
}
