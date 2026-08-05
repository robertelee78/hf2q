//! `hf2q convert <hf-dir> --quant <name> -o <out.gguf>` driver.
//!
//! Historically introduced as `convert-v2` (ADR-033 P4); B4 retired the
//! `-v2` suffix on 2026-05-19 once P6 deleted the legacy pipeline.
//! Per [[feedback-no-backwards-compat-2026-05-18]] no alias is kept —
//! the historical name fails loudly.
//!
//! First operator-facing entry point for the ADR-033 convert pipeline.
//! Composes [`HfModelSource::open`] → per-arch `map_tensor_name` +
//! `build_metadata` → [`ConvertOrchestrator`] into a single end-to-end
//! run. Streaming throughout: the source reader mmaps each safetensors
//! shard and yields F32 tensors lazily; the orchestrator quantizes +
//! writes one tensor at a time. Per ADR-033 §"Open Issues / Real-Model
//! Findings" 2026-05-18, this fixed the 4× SIGKILL-137 on a 26B-param
//! real-model convert.
//!
//! Per ADR-033 §P0-§P3: this driver does NOT introduce any new
//! quantization or write logic — every byte emitted comes from the
//! orchestrator. Per [[feedback-no-loop-suppression-2026-05-17]]: an
//! unsupported arch / unmapped tensor / missing expert surfaces as a
//! typed [`ConvertError`]; the orchestrator already rejects shape
//! misalignments at the policy/quantizer layer.
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: no migration shims,
//! no `--quant` aliases for legacy names — `LlamaFtype::from_name` is
//! the single source of truth.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use crate::backends::gguf::types::MetaValue;
use crate::convert::arch::bake::BakeOp;
use crate::convert::arch::qwen35moe_full::Qwen35MoeFullCtx;
use crate::convert::arch::{
    bake, bert, deepseek4, deepseek4_metadata, gemma4, gemma4_mmproj, llama3, minimax_m2,
    nomic_bert, qwen35moe, qwen35moe_full, qwen3vl_text,
};
use crate::convert::arch::gemma4::MappedTensor as Gemma4Mapped;
use crate::convert::arch::minimax_m2::{ExpertRole, MappedTensor as MiniMaxMapped};
use crate::convert::arch::qwen35moe::{ExpertKind, MappedTensor as QwenMapped};
use crate::convert::quant_selector::{approximate_for_apex, QuantSelector};
use crate::convert::orchestrator::PlanEntry;
use crate::convert::receipt::{
    clear_stale_receipt, write_success_receipt, PeakChunkBoundReceipt,
    RemoteConversionSource, ReceiptError,
};
use crate::convert::source_reader::SourceError;
use crate::convert::tokenizer::TokenizerError;
use crate::convert::{
    build_tokenizer_metadata, ConvertOrchestrator, HfModelSource, HfTensor, OrchestratorError,
};
use crate::quantize::ggml_quants::SourceDtype;
use crate::quantize::ggml_quants::apex::{
    detect_apex_config, load_mudler_config, ApexError, ApexPolicy, FingerprintHParams,
};
use crate::quantize::ggml_quants::standard_policy::HParams;
use crate::quantize::ggml_quants::ArchName;
use crate::core::provenance::{KEY_PRODUCER_VERSION, KEY_SOURCE_SHA256};

// ============================================================================
// Public API
// ============================================================================

/// Arguments for [`run_convert`]. Mirrors the
/// `hf2q convert <hf-dir> --quant <name> -o <out.gguf>` CLI surface
/// but is constructible directly from integration tests (the `--quant`
/// string is already resolved to a [`QuantSelector`] here).
///
/// `selector` is the unified `--quant <name>` parse result — either a
/// standard llama.cpp ftype, an Apex algorithmic tier, or (out of v1
/// scope) an `apex-custom` tensor-type-file path. See
/// [`crate::convert::quant_selector::QuantSelector`].
#[derive(Debug, Clone)]
pub struct ConvertArgs {
    /// HuggingFace model directory — must contain `config.json` plus
    /// either `model.safetensors` or `model.safetensors.index.json` +
    /// shards.
    pub hf_dir: PathBuf,
    /// Resolved `--quant <name>` selector. Standard ftypes route through
    /// `StandardPolicy`; Apex tiers route through `ApexPolicy`.
    pub selector: QuantSelector,
    /// Destination GGUF path. Existing files are overwritten.
    pub output: PathBuf,
    /// ADR-033 §Pi: pre-computed imatrix file (`.imatrix.gguf`). Required
    /// for I-tier APEX (`apex-i-*`) variants. Mutually exclusive with
    /// `imatrix_corpus`. Phase A's load path; round-trip-tested against
    /// the writer in `src/quantize/imatrix/gguf_loader.rs`.
    pub imatrix: Option<PathBuf>,
    /// ADR-033 §Pi: in-tree imatrix generation via named calibration
    /// corpus. Phase B Stage 3c SHIPPED 2026-05-19 — runs the
    /// `compute_imatrix` driver (HF dir → tempfile F16 GGUF → load
    /// → tokenize → chunk × `forward_prefill` → ImatrixData). Stage
    /// 3.0 wires Gemma 4 only; other arches surface
    /// `UnsupportedArchForDriver`.
    pub imatrix_corpus: Option<String>,
    /// ADR-033 §Pi: optional side-effect — write the imatrix used by
    /// this run to the given path. Useful for caching in-tree
    /// generations and for round-trip tests.
    pub imatrix_out: Option<PathBuf>,
    /// ADR-033 §Pi: context length for in-tree imatrix collection (only
    /// honored when `imatrix_corpus` is set; ignored on the `--imatrix
    /// <file>` load path). `None` ⇒ default 512 tokens per chunk
    /// matching stock `llama-imatrix -c 512`. Must be > 0; the driver
    /// surfaces `ImatrixError::CorpusTooShort` if the tokenized corpus
    /// can't fill even one chunk of size `n_ctx`.
    pub imatrix_n_ctx: Option<u32>,
    /// `--mmproj` flag: export the vision projector (mmproj) sidecar
    /// GGUF instead of the text decoder. See `ConvertCliArgs::mmproj`.
    pub mmproj: bool,
    /// Verified exact-revision identity for `--repo` conversion.
    pub remote_source: Option<RemoteConversionSource>,
}

/// Errors raised by [`run_convert`]. Wraps the typed errors from the
/// source reader + orchestrator + filesystem layers, and adds two
/// driver-only variants:
///
/// - [`ConvertError::UnsupportedArch`] — `config.json::model_type` /
///   `architectures` did not match any of the 8 supported arches.
/// - [`ConvertError::UnmappedTensor`] — a safetensors tensor name was
///   not recognized by the selected arch's `map_tensor_name`.
///
/// Per [[feedback-no-loop-suppression-2026-05-17]] both surface as
/// typed errors — never silently skipped.
#[derive(Debug)]
pub enum ConvertError {
    /// `HfModelSource::open` / `iter_tensors` / `materialize_tensor`
    /// failure (missing config, malformed safetensors, unsupported
    /// source dtype, missing FP8 sibling-scale, etc.).
    Source(SourceError),
    /// `ConvertOrchestrator::write` failure (policy reject, quantizer
    /// reject, writer I/O failure).
    Orchestrator(OrchestratorError),
    /// Filesystem I/O failed (e.g. could not create the output file).
    Io(std::io::Error),
    /// Remote source integrity or manifest verification failed.
    Integrity(crate::core::integrity::IntegrityError),
    /// Success-receipt construction or atomic persistence failed.
    Receipt(ReceiptError),
    /// `config.json` did not name one of the 8 supported architectures.
    /// `arch_name` carries the offending raw string (from `model_type`
    /// or `architectures[0]`).
    UnsupportedArch { arch_name: String },
    /// A safetensors tensor name was not recognized by the selected
    /// arch's `map_tensor_name`. Per the no-loop-suppression rule, this
    /// errors instead of being silently dropped.
    UnmappedTensor { hf_name: String, arch: String },
    /// One or more experts of an MoE group never showed up in the
    /// safetensors. `present` carries the expert indices that DID
    /// appear so the operator can diagnose which checkpoint shard is
    /// incomplete.
    IncompleteExpertGroup {
        gguf_name: String,
        layer: usize,
        kind_label: &'static str,
        present_count: usize,
        n_experts_config: usize,
    },
    /// Two HF tensors mapped to the same `(layer, kind, expert_index)`
    /// triple. Per [[feedback-no-loop-suppression-2026-05-17]]: this is
    /// a checkpoint corruption / mapper bug, not silent overwrite.
    DuplicateExpertIndex {
        gguf_name: String,
        layer: usize,
        kind_label: &'static str,
        expert_index: usize,
    },
    /// `config.json` was missing a required hparam the orchestrator
    /// needs for [`HParams`] (specifically `num_attention_heads`).
    /// Other arch-specific required keys still panic from
    /// `build_metadata`'s `[]` indexing — that contract is the per-arch
    /// mapper's, not the driver's, to enforce.
    MissingHparam { key: &'static str },
    /// `--quant apex-<tier>` was selected but `config.json` is missing
    /// `num_hidden_layers` (Apex needs it for the EDGE/NEAR/MID per-layer
    /// gradient).
    ApexMissingLayerCount,
    /// `--quant apex-custom --tensor-type-file <path>` is the reserved
    /// per-tensor override path per ADR Decision §"Per-model APEX config
    /// override". Out of v1 convert-v2 scope; surfaces here as a typed
    /// error stub for the future P4b wiring. `path` carries the
    /// operator-supplied tensor-type-file (preserved for diagnostics).
    ApexCustomOutOfScope { path: PathBuf },
    /// `ApexPolicy::new` rejected the source arch / hparams (unsupported
    /// arch, dense model, etc.). Wraps the typed `ApexError` so callers
    /// see the canonical mudler-aligned diagnostic.
    Apex(ApexError),
    /// `tokenizer::build_tokenizer_metadata` failed — missing /
    /// malformed `tokenizer.json`, unresolvable EOS token, etc. Per
    /// [[feedback-no-loop-suppression-2026-05-17]] this surfaces here
    /// rather than producing a GGUF that llama.cpp rejects with
    /// `key not found in model: tokenizer.ggml.model`.
    Tokenizer(TokenizerError),
    /// ADR-033 §Pi: an imatrix-subsystem failure surfaced. Wraps the
    /// typed [`crate::quantize::imatrix::ImatrixError`] so the operator
    /// sees the same diagnostic regardless of whether the failure
    /// happened in the loader, the writer, or the (Phase B) forward
    /// driver.
    Imatrix(crate::quantize::imatrix::ImatrixError),
    /// ADR-033 §Pi: an I-tier APEX (`apex-i-*`) variant was requested
    /// but neither `--imatrix <file>` nor `--imatrix-corpus <name>` was
    /// provided. Per the no-silent-fallback rule we refuse to silently
    /// degrade to the non-I sibling tier.
    ImatrixRequiredForITier { tier: &'static str },
    /// ADR-033 §Pi: `--imatrix-n-ctx 0` was passed. Per the
    /// no-loop-suppression rule we refuse rather than silently
    /// defaulting; the operator gave an explicit invalid value.
    ImatrixNCtxInvalid { n_ctx: u32 },
    /// B1 — operator supplied BOTH a positional `<hf_dir>` AND `--repo`.
    /// Exactly one input source is required. Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: refuse rather than
    /// silently pick one.
    RepoAndDirMutuallyExclusive,
    /// Remote conversion must pin the exact immutable Hub commit.
    ImmutableRevisionRequired { supplied: Option<String> },
    /// `--revision` has no meaning for an already-local source directory.
    RevisionRequiresRepo,
    /// Repo id is unsafe or outside HuggingFace's path-shaped id grammar.
    InvalidRepoId { repo: String },
    /// B1 — the allowed `hf download <repo>` source fetch exited non-zero. Captures
    /// the exit code (`None` if the process was killed by a signal)
    /// plus the captured stderr so the operator can diagnose auth /
    /// network / missing-binary failures.
    HfDownload {
        repo: String,
        exit_code: Option<i32>,
        stderr: String,
    },
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::Source(e) => write!(f, "convert/source: {e}"),
            ConvertError::Orchestrator(e) => write!(f, "convert/orchestrator: {e}"),
            ConvertError::Io(e) => write!(f, "convert/io: {e}"),
            ConvertError::Integrity(e) => write!(f, "convert/integrity: {e}"),
            ConvertError::Receipt(e) => write!(f, "convert/receipt: {e}"),
            ConvertError::UnsupportedArch { arch_name } => {
                write!(
                    f,
                    "convert: unsupported architecture `{arch_name}` \
                     (supported: llama, gemma3, bert, nomic_bert, qwen3_moe, qwen3_5_moe, \
                     qwen3_vl, minimax_m2)"
                )
            }
            ConvertError::UnmappedTensor { hf_name, arch } => write!(
                f,
                "convert: tensor `{hf_name}` not recognized by `{arch}` mapper"
            ),
            ConvertError::IncompleteExpertGroup {
                gguf_name,
                layer,
                kind_label,
                present_count,
                n_experts_config,
            } => write!(
                f,
                "convert: expert group `{gguf_name}` (layer={layer}, kind={kind_label}) \
                 only saw {present_count}/{n_experts_config} experts"
            ),
            ConvertError::DuplicateExpertIndex {
                gguf_name,
                layer,
                kind_label,
                expert_index,
            } => write!(
                f,
                "convert: duplicate expert index {expert_index} for \
                 `{gguf_name}` (layer={layer}, kind={kind_label})"
            ),
            ConvertError::MissingHparam { key } => write!(
                f,
                "convert: config.json is missing required hparam `{key}`"
            ),
            ConvertError::ApexMissingLayerCount => write!(
                f,
                "convert: --quant apex-<tier> requires `num_hidden_layers` in config.json"
            ),
            ConvertError::ApexCustomOutOfScope { path } => write!(
                f,
                "convert: --quant apex-custom --tensor-type-file `{}` is reserved \
                 (out of v1 scope)",
                path.display()
            ),
            ConvertError::Apex(e) => write!(f, "convert/apex: {e}"),
            ConvertError::Tokenizer(e) => write!(f, "convert/tokenizer: {e}"),
            ConvertError::Imatrix(e) => write!(f, "convert/imatrix: {e}"),
            ConvertError::ImatrixRequiredForITier { tier } => write!(
                f,
                "convert: --quant apex-{tier} requires `--imatrix <file>` \
                 or `--imatrix-corpus <name>` (ADR-033 §Pi Phase B SHIPPED 2026-05-19)"
            ),
            ConvertError::ImatrixNCtxInvalid { n_ctx } => write!(
                f,
                "convert: --imatrix-n-ctx {n_ctx} is invalid; \
                 must be > 0 (default 512 matches stock `llama-imatrix -c 512`)"
            ),
            ConvertError::RepoAndDirMutuallyExclusive => write!(
                f,
                "convert: `--repo <hf_repo>` and positional `<hf_dir>` are mutually exclusive — \
                 pass exactly one"
            ),
            ConvertError::ImmutableRevisionRequired { supplied } => write!(
                f,
                "convert: `--repo` requires `--revision <40-hex-commit>`; got {}",
                supplied.as_deref().unwrap_or("<missing>")
            ),
            ConvertError::RevisionRequiresRepo => write!(
                f,
                "convert: `--revision` is valid only with `--repo`; local directories are used as supplied"
            ),
            ConvertError::InvalidRepoId { repo } => write!(
                f,
                "convert: invalid HuggingFace repo id `{repo}`; expected slash-separated ASCII name components"
            ),
            ConvertError::HfDownload {
                repo,
                exit_code,
                stderr,
            } => write!(
                f,
                "convert: HuggingFace download for {repo} exited with status {} — stderr:\n{}",
                exit_code
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "<signal>".to_string()),
                stderr.trim_end()
            ),
        }
    }
}

impl std::error::Error for ConvertError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConvertError::Source(e) => Some(e),
            ConvertError::Orchestrator(e) => Some(e),
            ConvertError::Io(e) => Some(e),
            ConvertError::Integrity(e) => Some(e),
            ConvertError::Receipt(e) => Some(e),
            ConvertError::Apex(e) => Some(e),
            ConvertError::Tokenizer(e) => Some(e),
            ConvertError::Imatrix(e) => Some(e),
            _ => None,
        }
    }
}

impl From<crate::quantize::imatrix::ImatrixError> for ConvertError {
    fn from(e: crate::quantize::imatrix::ImatrixError) -> Self {
        ConvertError::Imatrix(e)
    }
}

impl From<SourceError> for ConvertError {
    fn from(e: SourceError) -> Self {
        ConvertError::Source(e)
    }
}

impl From<OrchestratorError> for ConvertError {
    fn from(e: OrchestratorError) -> Self {
        ConvertError::Orchestrator(e)
    }
}

impl From<std::io::Error> for ConvertError {
    fn from(e: std::io::Error) -> Self {
        ConvertError::Io(e)
    }
}

impl From<crate::core::integrity::IntegrityError> for ConvertError {
    fn from(e: crate::core::integrity::IntegrityError) -> Self {
        ConvertError::Integrity(e)
    }
}

impl From<ReceiptError> for ConvertError {
    fn from(e: ReceiptError) -> Self {
        ConvertError::Receipt(e)
    }
}

impl From<ApexError> for ConvertError {
    fn from(e: ApexError) -> Self {
        ConvertError::Apex(e)
    }
}

impl From<TokenizerError> for ConvertError {
    fn from(e: TokenizerError) -> Self {
        ConvertError::Tokenizer(e)
    }
}

// ============================================================================
// Driver entry point
// ============================================================================

/// Run the ADR-033 convert pipeline end-to-end on a HuggingFace model
/// directory.
///
/// Flow:
/// 1. [`HfModelSource::open`] mmaps safetensors + reads `config.json`. Tensor metadata only.
/// 2. Detect arch from `config["model_type"]` / `config["architectures"]`.
/// 3. Build a [`ConvertOrchestrator`] pinned to that arch + ftype +
///    [`HParams`] from config.
/// 4. Per HF tensor: dispatch via the arch's `map_tensor_name`.
///    - `Direct(gguf_name)` → reverse shape to GGUF order + push to
///      orchestrator.
///    - `ExpertGroup{..}` → buffer in the expert accumulator.
///    - `Drop` → discard (the arch mapper signed off explicitly).
///    - `None` → typed [`ConvertError::UnmappedTensor`].
/// 5. Drain the expert accumulator: assert every group has exactly
///    `n_experts` slices, sort by `expert_index`, then decode and
///    quantize one expert at a time into the fused 3-D GGUF payload.
/// 6. Emit metadata via the arch's `build_metadata`.
/// 7. [`ConvertOrchestrator::write`] → BufWriter over the output file.
pub fn run_convert(args: ConvertArgs) -> Result<(), ConvertError> {
    // ----- 1. Open source (mmap, metadata-only) ---------------------------
    // Per ADR-033 §"Open Issues / Real-Model Findings" 2026-05-18: the
    // source reader does NOT load every safetensors shard into RAM. It
    // mmaps each shard and records a flat tensor index. Payload bytes are
    // read one tensor at a time in the streaming stage below.
    let src = HfModelSource::open(&args.hf_dir)?;
    let excluded_dspark_count = src.excluded_mtp_tensor_count();
    if excluded_dspark_count > 0 {
        tracing::warn!(
            target: "convert",
            excluded = excluded_dspark_count,
            "DeepSeek-V4 MTP/DSpark tensors excluded from base GGUF; separate draft artifact remains required"
        );
    }

    // ----- 2. Detect arch ---------------------------------------------------
    let detected_arch = detect_arch(&src.config)?;
    // `--mmproj` overrides arch routing to the vision-projector sidecar
    // mapper. Mirrors canonical `convert_hf_to_gguf.py:223,229,233` —
    // when `--mmproj` is set, the script swaps `TEXT_MODEL_MAP` for
    // `MMPROJ_MODEL_MAP`. Requires a `vision_config` sub-object in the
    // root config.json (Gemma 4 / Gemma 3 ForConditionalGeneration).
    let arch = if args.mmproj {
        if src.config.get("vision_config").is_none() {
            return Err(ConvertError::UnsupportedArch {
                arch_name: format!(
                    "{detected_arch:?} (--mmproj requires a `vision_config` sub-object in config.json; not present)"
                ),
            });
        }
        match detected_arch {
            ArchName::Gemma4 => {
                // Two Gemma vision variants ship under the same
                // `model_type=gemma4` umbrella:
                //   - Gemma 3 vision (SigLIP-style): tensor names
                //     `model.vision_tower.vision_model.embeddings.*` +
                //     `.encoder.layers.<N>.*`. Canonical handler is
                //     `Gemma3VisionModel` (`gemma.py:251-302`).
                //   - Gemma 4 vision: tensor names
                //     `model.vision_tower.encoder.layers.<N>.*` (no
                //     `vision_model` infix) + audio sibling at
                //     `model.audio_tower.*`. Canonical handler is
                //     `Gemma4VisionAudioModel` (`gemma.py:768+`).
                //     The patch embedder needs a 2-D → 4-D reshape +
                //     permute; the projector type is `GEMMA4V` not
                //     `gemma3`.
                //
                // hf2q's existing `src/convert/arch/gemma4_mmproj.rs`
                // implements the FIRST variant (Gemma 3 vision); the
                // Gemma 4 variant requires a separate arch port that
                // is not yet shipped. Detect from a known marker
                // tensor and fail fast with a typed error.
                let has_gemma4_vision = src
                    .tensor_metas()
                    .any(|m| m.name.starts_with("model.vision_tower.encoder.layers."));
                if has_gemma4_vision {
                    // Gemma 4 26B-A4B-IT transformer-style vision tower.
                    // Mapper at src/convert/arch/gemma4_vision_mmproj.rs.
                    ArchName::Gemma4VisionMmproj
                } else {
                    ArchName::Gemma4Mmproj
                }
            }
            other => {
                return Err(ConvertError::UnsupportedArch {
                    arch_name: format!(
                        "--mmproj not supported for {other:?} yet (only Gemma 3 vision shipped)"
                    ),
                });
            }
        }
    } else {
        detected_arch
    };

    // ----- 3. Build orchestrator -------------------------------------------
    // The selector branches the policy:
    //   - Standard(ftype): orchestrator runs StandardPolicy.
    //   - Apex(tier): build ApexPolicy { tier, n_layers, n_expert }
    //     from config.json + the detected arch; orchestrator routes
    //     per-tensor decisions through it. The `general.file_type` byte
    //     carries `approximate_for_apex(tier)` as the closest standard
    //     ftype (purely cosmetic — per-tensor ggml_types are recorded
    //     on each tensor info entry).
    //   - ApexCustom(path): out of v1 convert-v2 scope; typed error.
    let hparams = build_hparams(&src.config)?;
    let (mut orch, ftype_for_metadata) = match &args.selector {
        QuantSelector::Standard(ftype) => {
            let orch = ConvertOrchestrator::new(*ftype, arch, hparams);
            (orch, *ftype)
        }
        QuantSelector::Apex(tier) => {
            let n_layers = config_n_layers(&src.config)
                .ok_or(ConvertError::ApexMissingLayerCount)?;
            let n_expert = hparams.n_expert;

            // ADR-033 §Pi: resolve the imatrix surface. Two paths:
            //   - `--imatrix <path>`: loads a pre-computed
            //     `.imatrix.gguf` (Phase A; useful when the target
            //     arch isn't yet wired for in-tree generation, or
            //     when re-using a stock `llama-imatrix` output).
            //   - `--imatrix-corpus <name>`: drives Stage 3c's
            //     in-tree `compute_imatrix` (Gemma 4 only at
            //     Stage 3.0; other arches surface
            //     `UnsupportedArchForDriver`).
            //
            // The policy constructor choice depends on whether imatrix data
            // is present:
            //   - tier is I-tier + imatrix data present → new_with_imatrix
            //   - tier is I-tier + no imatrix data → typed reject (the §Pi
            //     no-silent-fallback rule).
            //   - tier is non-I → new (imatrix data, if present, is still
            //     respected; it's optional for non-I tiers).
            let imatrix_data = resolve_imatrix_input(
                tier,
                args.imatrix.as_deref(),
                args.imatrix_corpus.as_deref(),
                &args.hf_dir,
                arch,
                args.imatrix_n_ctx.unwrap_or(512),
            )?;

            let mut apex_policy = if imatrix_data.is_some() {
                ApexPolicy::new_with_imatrix(*tier, arch, n_layers, n_expert)?
            } else {
                ApexPolicy::new(*tier, arch, n_layers, n_expert)?
            };

            // Side-effect: write the imatrix used by this run to disk if
            // requested. Idempotent for the loaded-from-file case (round-trip
            // re-emits the same bytes).
            if let (Some(out_path), Some(data)) = (&args.imatrix_out, imatrix_data.as_ref()) {
                let label = data
                    .loaded
                    .datasets
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "user-file".to_string());
                data.write_gguf(out_path, &[label])?;
                eprintln!(
                    "[hf2q imatrix] wrote {} ({} tensor pairs)",
                    out_path.display(),
                    data.tensor_pair_count()
                );
            }
            // ADR-033 §P4b wiring (SHIPPED 2026-05-22): the loaded imatrix
            // is threaded through to `quantizer.quantize(..., Some(imatrix))`
            // via the orchestrator's `with_imatrix` setter. Pre-P4b, this
            // line read `let _ = imatrix_data;` (silently dropped), so
            // apex-i-quality produced byte-identical output to apex-quality.
            // See `[[project-adr033-p4b-unimplemented-2026-05-22]]` for the
            // empirical SHA256 A/B that surfaced the gap.
            let imatrix_for_orch = imatrix_data;

            // ADR-033 §9 — per-model APEX config override.
            //
            // Hash the source config.json's 9-tuple of identifying
            // hparams; if it matches a vendored mudler config in
            // `data/apex-references/manifest.json`, attach the
            // per-tensor overlay to the policy. The override wins
            // over the algorithmic generator silently (per ADR §9
            // line 104 "fingerprint match is invisible to the
            // user"), but we log the match to stderr so the operator
            // can audit which config fired — mitigates the
            // "surprising override" risk called out in the ADR.
            let effective = effective_config(&src.config);
            if let Some(fp_hparams) = FingerprintHParams::from_config(effective) {
                if let Some(entry) = detect_apex_config(&fp_hparams, *tier) {
                    let mudler = load_mudler_config(entry)?;
                    apex_policy = apex_policy.with_mudler_override(mudler);
                    eprintln!(
                        "[hf2q apex] auto-detected APEX config: {} \
                         (fingerprint={}, tier={}, arch={})",
                        entry.mudler_config_path,
                        &entry.fingerprint[..16],
                        entry.tier,
                        entry.arch,
                    );
                }
            }
            let ftype = approximate_for_apex(*tier);
            let orch = ConvertOrchestrator::new_with_apex(ftype, arch, hparams, apex_policy)
                .with_imatrix(imatrix_for_orch);
            (orch, ftype)
        }
        QuantSelector::ApexCustom(path) => {
            return Err(ConvertError::ApexCustomOutOfScope {
                path: path.clone(),
            });
        }
    };

    // ----- 4. Emit metadata (orchestrator buffers it for begin_write) ----
    // Read README.md YAML frontmatter (HF model card) up-front so the
    // arch's `build_metadata` can emit `general.{license, tags,
    // languages, base_model.*}` per canonical's
    // `gguf-py/gguf/metadata.py::Metadata.load`. `None` if there's no
    // README.md or the frontmatter block is absent — arches that don't
    // consume the model card ignore the parameter.
    let model_card = crate::convert::model_card::parse_readme_frontmatter(&args.hf_dir);
    let sampling = crate::convert::model_card::parse_generation_config(&args.hf_dir);
    let dir_basename: Option<String> = args
        .hf_dir
        .file_name()
        .and_then(|s| s.to_str())
        .map(String::from);
    // Pre-compute `general.size_label` for MoE arches by walking the
    // source tensors. Mirrors canonical's
    // `gguf_writer.get_total_parameter_count()` + `gguf.size_label()`
    // pipeline at `gguf-py/gguf/utility.py:44-52`. Per-arch expert
    // detection: nomic_bert v2-moe uses HF name pattern
    // `mlp.experts.mlp.w` to flag expert tensors (vs the canonical
    // GGUF-side `_exps.` check — equivalent results).
    let size_label = compute_size_label_for_arch(arch, &src, &src.config);
    let ftype_u32 = ftype_for_metadata as u32;
    // BERT: resolve pooling type via canonical's `_try_set_pooling_type`
    // path (modules.json → "Pooling" mod's `path` key → 1_Pooling/
    // config.json → pooling_mode_*_token bools). Returned as the
    // canonical PoolingType enum u32 (0=NONE/RANK/MEAN/CLS/LAST per
    // gguf.PoolingType).
    let bert_pooling_override = if matches!(arch, ArchName::Bert) {
        resolve_bert_pooling_type(&args.hf_dir)
    } else {
        None
    };
    // Qwen3-VL deepstack count comes from vision_config (sibling to
    // text_config at root). After `effective_config()` unwraps to
    // text_config the vision_config is invisible — so we read it from
    // the original src.config here. Mirrors canonical
    // /opt/llama.cpp/conversion/qwen3vl.py:255-258 path.
    let qwen3vl_n_deepstack = if matches!(arch, ArchName::Qwen3VlText) {
        let vc = src
            .config
            .get("thinker_config")
            .and_then(|tc| tc.get("vision_config"))
            .or_else(|| src.config.get("vision_config"));
        vc.and_then(|v| v.get("deepstack_visual_indexes"))
            .and_then(|a| a.as_array())
            .map(|a| a.len() as u32)
    } else {
        None
    };
    let arch_metadata = build_metadata_for_arch(
        arch,
        &src.config,
        ftype_u32,
        model_card.as_ref(),
        size_label.as_deref(),
        sampling.as_ref(),
        dir_basename.as_deref(),
        bert_pooling_override,
        qwen3vl_n_deepstack,
    );

    // Canonical emits `general.quantization_version` and
    // `general.file_type` AFTER the tokenizer block (positions 50-51 of
    // the canonical Q8_0 GGUF dump for nomic v2-moe). Split them out
    // of the arch metadata into a postlude that emits last. Other
    // arches' build_metadata may also place these at the end of their
    // vec; we pull them out by exact key match so non-MoE paths still
    // emit in the same canonical-tail position.
    const POSTLUDE_KEYS: &[&str] = &[
        "general.quantization_version",
        "general.file_type",
    ];
    // mmproj sidecars use a DIFFERENT KV order than text decoders:
    // general.file_type lives EARLY (after general.size_label, before
    // clip.*), while general.quantization_version remains last. The
    // generic postlude split (file_type → end) breaks this. For mmproj
    // we trust the build_metadata's emit order — only quantization_version
    // is pulled to postlude.
    let postlude_keys: &[&str] = if matches!(
        arch,
        ArchName::Gemma4Mmproj | ArchName::Gemma4VisionMmproj
    ) {
        &["general.quantization_version"]
    } else {
        POSTLUDE_KEYS
    };
    let mut prelude: Vec<(String, MetaValue)> = Vec::with_capacity(arch_metadata.len());
    let mut postlude: Vec<(String, MetaValue)> = Vec::with_capacity(postlude_keys.len());
    for (k, v) in arch_metadata {
        if postlude_keys.contains(&k.as_str()) {
            postlude.push((k, v));
        } else {
            prelude.push((k, v));
        }
    }
    for (k, v) in prelude {
        orch.add_metadata(k, v);
    }

    // ----- 4b. Emit tokenizer metadata --------------------------------
    // llama.cpp's vocab loader rejects any GGUF that is missing
    // `tokenizer.ggml.model` — failure mode reported 2026-05-18 by the
    // real-model convert-v2 smoke test on
    // /opt/hf2q/models/google-gemma-4-26b-a4b-it. Per
    // [[feedback-no-loop-suppression-2026-05-17]] we surface every
    // tokenizer-parse failure as a typed `ConvertError::Tokenizer`
    // variant rather than skipping silently — that exact silent skip
    // is what produced the bug.
    // mmproj sidecars do NOT carry tokenizer metadata. Canonical's
    // MmprojModel base (base.py:2152+) explicitly skips set_vocab. The
    // text-decoder GGUF (written separately) owns the tokenizer; the
    // mmproj sidecar is consumed alongside it by the runtime.
    if !matches!(arch, ArchName::Gemma4Mmproj | ArchName::Gemma4VisionMmproj) {
        for (k, v) in build_tokenizer_metadata(&args.hf_dir, arch)? {
            orch.add_metadata(k, v);
        }
    }
    // Postlude: general.quantization_version + general.file_type
    // emitted last (per canonical order).
    for (k, v) in postlude {
        orch.add_metadata(k, v);
    }
    if let Some(remote) = args.remote_source.as_ref() {
        orch.add_metadata(
            KEY_PRODUCER_VERSION.to_string(),
            MetaValue::String(format!("hf2q {}", env!("CARGO_PKG_VERSION"))),
        );
        orch.add_metadata(
            KEY_SOURCE_SHA256.to_string(),
            MetaValue::String(remote.source_sha256.clone()),
        );
    }

    // ----- 5. Plan + stream tensors (with MoE expert fusion) -------------
    //
    // Some arches require tensors that are NOT in the safetensors but
    // ARE part of the canonical GGUF (e.g. Gemma 4's `rope_freqs.weight`
    // proportional-rope mask, synthesized at convert time per
    // `gemma.py::Gemma4Model::generate_extra_tensors`). We synthesize
    // them here as fully-materialized F32 `HfTensor`s and flow them
    // through the same map/plan/stream path as on-disk tensors.
    let synthesized: Vec<HfTensor> = synthesized_tensors_for_arch(arch, &src.config);
    let plan = build_convert_plan(arch, &src, &synthesized)?;

    // 5a. Orchestrator plan-phase: feed every tensor's metadata, no
    // payload bytes.
    let plan_entries: Vec<PlanEntry> = plan.steps.iter().map(|s| s.plan_entry()).collect();
    orch.plan_tensors(plan_entries)?;

    // 5b. Begin writing — header + KVs + tensor-info reservations.
    if args.remote_source.is_some() {
        clear_stale_receipt(&args.output)?;
    }
    let f = File::create(&args.output)?;
    let bw = BufWriter::new(f);
    let mut sw = orch.begin_write(bw)?;
    let mut peak_chunk_bound = PeakChunkBoundReceipt::default();

    // 5c. Stream every tensor's data in plan order. A fused MoE tensor
    // is emitted one expert at a time: decode → F16 roundtrip → quantize
    // → exact payload chunk. The expert Vec drops before the next source
    // tensor is opened, so 256-expert models never allocate the 3-D F32
    // stack.
    for (idx, step) in plan.steps.iter().enumerate() {
        match step {
            PlanStep::Fused {
                gguf_name,
                member_hf_names,
                per_expert_py_shape,
                ..
            } => {
                let per_expert_elems = per_expert_py_shape.iter().try_fold(
                    1usize,
                    |n, &d| n.checked_mul(d),
                ).ok_or_else(|| {
                    ConvertError::Source(SourceError::Safetensors(format!(
                        "fused expert tensor `{gguf_name}` shape product overflow: {per_expert_py_shape:?}"
                    )))
                })?;
                sw.begin_tensor_chunks(idx)?;
                for name in member_hf_names {
                    let ht = src.materialize_tensor(name)?;
                    if ht.shape != *per_expert_py_shape || ht.data.len() != per_expert_elems {
                        return Err(ConvertError::Source(SourceError::Safetensors(format!(
                            "fused expert slice `{name}`: shape {:?} / data len {} != expected {:?} / {}",
                            ht.shape,
                            ht.data.len(),
                            per_expert_py_shape,
                            per_expert_elems
                        ))));
                    }
                    sw.stream_tensor_chunk(idx, &ht.data)?;
                }
                let stats = sw.finish_tensor_chunks(idx)?;
                peak_chunk_bound.observe(stats);
                debug_assert_eq!(stats.chunk_count, member_hf_names.len());
                debug_assert_eq!(stats.max_chunk_elements, per_expert_elems);
                tracing::debug!(
                    target: "convert",
                    tensor = gguf_name,
                    experts = stats.chunk_count,
                    max_live_f32_elements = stats.max_chunk_elements,
                    "streamed fused expert tensor with bounded input chunks"
                );
            }
            _ => {
                let data: Vec<f32> = step.materialize(&src, &synthesized)?;
                let stats = sw.stream_tensor(idx, &data)?;
                peak_chunk_bound.observe(stats);
            }
        }
    }

    // 5d. Finalize — seek-back to fill tensor offsets, flush.
    sw.finalize()?;
    if let Some(remote) = args.remote_source.as_ref() {
        write_success_receipt(
            &args.output,
            remote,
            &args.selector.receipt_name(),
            excluded_dspark_count,
            peak_chunk_bound,
        )?;
    }
    Ok(())
}

/// ADR-033 §Pi: resolve the imatrix-CLI surface into an
/// [`ImatrixData`] value (or `None` for runs that don't need one).
///
/// Resolution rules (in priority order):
///   - `--imatrix <file>` set → [`ImatrixData::load_from_path`].
///   - `--imatrix-corpus <name>` set → drive Stage 3
///     [`compute_imatrix`] over the corpus, returning the produced
///     `ImatrixData { provenance: Computed }`. Driver-side failures
///     surface as typed `ImatrixError` (ConvertFailed,
///     ModelLoadFailed, UnsupportedArchForDriver, etc.) wrapped via
///     `ConvertError::Imatrix`.
///   - Neither set + tier is I-tier → typed `ImatrixRequiredForITier`.
///   - Neither set + tier is non-I → `Ok(None)` (the run proceeds
///     without imatrix data).
///
/// `hf_dir` and `arch` are required for the corpus-driven path
/// (Stage 3 driver needs them to convert + load the model
/// in-tree). Unused for the `--imatrix <file>` and tier-only paths.
fn resolve_imatrix_input(
    tier: &crate::quantize::ggml_quants::apex::ApexTier,
    imatrix_path: Option<&std::path::Path>,
    imatrix_corpus: Option<&str>,
    hf_dir: &std::path::Path,
    arch: crate::quantize::ggml_quants::ArchName,
    n_ctx: u32,
) -> Result<Option<crate::quantize::imatrix::ImatrixData>, ConvertError> {
    use crate::quantize::imatrix::{
        compute_imatrix, ComputeImatrixParams, CorpusBytes, CorpusSource, ImatrixData,
    };

    if let Some(path) = imatrix_path {
        let data = ImatrixData::load_from_path(path)?;
        eprintln!(
            "[hf2q imatrix] loaded {} ({} tensor pairs, chunks={}, chunk_size={})",
            path.display(),
            data.tensor_pair_count(),
            data.loaded.chunk_count,
            data.loaded.chunk_size,
        );
        return Ok(Some(data));
    }
    if let Some(corpus_name) = imatrix_corpus {
        if n_ctx == 0 {
            return Err(ConvertError::ImatrixNCtxInvalid { n_ctx });
        }
        // Stage 3c.2 (ADR-033 §Pi Phase B): in-tree forward-pass
        // driver. Loads the source HF dir → F16 GGUF tempfile, runs
        // the per-arch decoder forward pass over `corpus_name`'s
        // tokenized chunks, returns the computed imatrix.
        //
        // Stage 3.0 wires Gemma 4 only; other arches surface a typed
        // `UnsupportedArchForDriver` error (NOT a silent fallback to
        // the workaround). Operators with Qwen 3.5/3.6 etc. should
        // continue using stock `llama-imatrix` + `--imatrix <path>`
        // until Stage 3b.4 adds Qwen35Moe driver wiring.
        let source = CorpusSource::from_cli(corpus_name)?;
        let corpus = CorpusBytes::load(&source)?;
        let label = source.dataset_label();
        eprintln!(
            "[hf2q imatrix] computing in-tree on corpus `{label}` \
             ({} bytes, ~{} words, n_ctx={n_ctx})",
            corpus.byte_count(),
            corpus.approx_word_count(),
        );
        // ADR-033 §Pi: `n_ctx` is operator-settable via
        // `--imatrix-n-ctx <N>`; defaults to 512 to match stock
        // `llama-imatrix -c 512`. Validated > 0 above. Larger
        // values mean fewer, longer chunks per forward-pass loop.
        let params = ComputeImatrixParams {
            hf_dir: hf_dir.to_path_buf(),
            corpus,
            n_ctx,
            arch,
        };
        let data = compute_imatrix(&params)?;
        eprintln!(
            "[hf2q imatrix] computed {} tensor pairs, chunks={}, chunk_size={}",
            data.tensor_pair_count(),
            data.loaded.chunk_count,
            data.loaded.chunk_size,
        );
        return Ok(Some(data));
    }
    if tier.requires_imatrix() {
        return Err(ConvertError::ImatrixRequiredForITier {
            tier: tier.cli_name(),
        });
    }
    Ok(None)
}

/// Extract the effective layer count from the HF config — equals
/// `num_hidden_layers + mtp_num_hidden_layers` (the latter defaults
/// to 0 when not present). Required by `ApexPolicy::new` and the
/// orchestrator's policy walk for the per-layer EDGE/NEAR/MID
/// gradient.
///
/// Per canonical `_Qwen35MtpMixin.__init__` (qwen.py:550-555),
/// `block_count = num_hidden_layers + mtp_num_hidden_layers` when
/// MTP is present — so a Qwen 3.5 model with 40 transformer layers
/// + 1 MTP block has 41 GGUF blocks (the MTP block lives at
/// `blk.{num_hidden_layers}`).
///
/// Returns `None` if `num_hidden_layers` is missing or non-positive —
/// surfaces as [`ConvertError::ApexMissingLayerCount`] at the caller.
fn config_n_layers(config: &serde_json::Value) -> Option<u32> {
    let cfg = effective_config(config);
    let base = cfg
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .filter(|&x| x > 0)?;
    let mtp = cfg
        .get("mtp_num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    Some((base + mtp) as u32)
}

// ============================================================================
// Arch detection
// ============================================================================

/// Detect [`ArchName`] from `config.json`.
///
/// Strategy: inspect `model_type` (string) first since it's the
/// canonical HF discriminant; fall back to `architectures` (array)
/// when present. The two are checked independently per HF convention
/// — older configs sometimes ship one without the other.
///
/// Per ADR-033 the supported arches are a closed set (8 entries); any
/// other arch surfaces as [`ConvertError::UnsupportedArch`].
fn detect_arch(config: &serde_json::Value) -> Result<ArchName, ConvertError> {
    let model_type = config.get("model_type").and_then(|v| v.as_str());
    let architectures: Vec<&str> = config
        .get("architectures")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();

    // Detect from model_type first.
    if let Some(mt) = model_type {
        match mt {
            "llama" => return Ok(ArchName::Llama3),
            // gemma3 (Gemma 3 architecture) + gemma4 / gemma4_text (Gemma 4
            // release strings — operator's google-gemma-4-26b-a4b-it has
            // model_type="gemma4" with nested text_config.model_type=
            // "gemma4_text"). Surfaced 2026-05-18 by real-model convert
            // smoke test against /opt/hf2q/models/google-gemma-4-26b-a4b-it.
            "gemma3" | "gemma" | "gemma4" | "gemma4_text" => return Ok(ArchName::Gemma4),
            "bert" => return Ok(ArchName::Bert),
            "nomic_bert" => return Ok(ArchName::NomicBert),
            // qwen3_moe (canonical Qwen 3.6 dense MoE). The Qwen 3.5/3.6
            // linear-attention + MTP variants route to Qwen35MoeFull
            // (the qwen35moe canonical arch) below.
            "qwen3_moe" => return Ok(ArchName::Qwen35Moe),
            // Qwen 3.5/3.6 with linear-attention + MTP. Top-level
            // `qwen3_5_moe` is the multimodal-VLM
            // `Qwen3_5MoeForConditionalGeneration` config (operator's
            // /opt/hf2q/models/Qwen-Qwen3.5-35B-A3B has this at config.json:6).
            // The `_text` variant is the nested text_config.model_type
            // that text-only `Qwen3_5MoeForCausalLM` checkpoints expose.
            // Note: "Qwen 3.6" is a model VERSION name; all locally-
            // available qwen3.6-* models use Qwen3_5* arch strings
            // (canonical does NOT define a Qwen3_6Moe* arch — verified
            // via grep on /opt/llama.cpp/conversion).
            "qwen3_5_moe" | "qwen3_5_moe_text" => return Ok(ArchName::Qwen35MoeFull),
            "qwen3_vl" | "qwen3_vl_moe" | "qwen3_vl_text" => return Ok(ArchName::Qwen3VlText),
            "minimax_m2" => return Ok(ArchName::MiniMaxM2),
            "deepseek_v4" => return Ok(ArchName::Deepseek4),
            _ => {}
        }
    }

    // Fall back to the architectures[] array — HF's older convention.
    // We probe the well-known class names mapper.
    for cls in &architectures {
        match *cls {
            "LlamaForCausalLM" => return Ok(ArchName::Llama3),
            // Both Gemma3*ForCausalLM and Gemma3ForConditionalGeneration
            // are produced by HF for the same gemma-3 family; we accept
            // the prefix.
            // Gemma3*/Gemma2*/GemmaForCausalLM + Gemma4*ForConditionalGeneration
            // / Gemma4ForCausalLM. The operator's gemma-4-26b release uses
            // "Gemma4ForConditionalGeneration" (multimodal config wrapping
            // the text decoder). Prefix-match covers both -ForCausalLM and
            // -ForConditionalGeneration suffixes.
            s if s.starts_with("Gemma3")
                || s.starts_with("Gemma2")
                || s.starts_with("Gemma4")
                || s == "GemmaForCausalLM" =>
            {
                return Ok(ArchName::Gemma4);
            }
            "BertForMaskedLM" | "BertModel" => return Ok(ArchName::Bert),
            "NomicBertModel" => return Ok(ArchName::NomicBert),
            // Qwen3MoeForCausalLM (canonical) — older dense MoE.
            "Qwen3MoeForCausalLM" => return Ok(ArchName::Qwen35Moe),
            // Qwen 3.5 (and the "3.6" model versions that use the same
            // arch strings) with linear-attention + MTP. Includes both
            // text-only ForCausalLM and multimodal-VLM
            // ForConditionalGeneration releases (the latter is the
            // operator's locally-downloaded
            // /opt/hf2q/models/Qwen-Qwen3.5-35B-A3B variant — config
            // has architectures=["Qwen3_5MoeForConditionalGeneration"]).
            // Canonical at /opt/llama.cpp/conversion/qwen.py:626 only
            // registers Qwen3_5MoeFor* (no Qwen3_6Moe* arch exists).
            "Qwen3_5MoeForCausalLM" | "Qwen3_5MoeForConditionalGeneration" => {
                return Ok(ArchName::Qwen35MoeFull);
            }
            "Qwen3VLForConditionalGeneration"
            | "Qwen3VLMoeForConditionalGeneration"
            | "Qwen3VLTextForCausalLM" => {
                return Ok(ArchName::Qwen3VlText);
            }
            "MiniMaxM2ForCausalLM" => return Ok(ArchName::MiniMaxM2),
            "DeepseekV4ForCausalLM" => return Ok(ArchName::Deepseek4),
            _ => {}
        }
    }

    // Nothing matched — typed error per the no-fallback rule. We carry
    // the most-specific name we observed for diagnostics.
    let observed = model_type
        .map(|s| s.to_string())
        .or_else(|| architectures.first().map(|s| s.to_string()))
        .unwrap_or_else(|| "<missing model_type and architectures>".into());
    Err(ConvertError::UnsupportedArch {
        arch_name: observed,
    })
}

// ============================================================================
// HParams
// ============================================================================

/// Multimodal-config flatten: when the top-level config is a multimodal
/// wrapper (Gemma 4 mmproj-bundle, Qwen3-VL omni, etc.) the text-decoder
/// hparams live in `config["text_config"]`, not at the top level. This
/// helper returns the inner text-config when present, else the outer
/// config unchanged.
///
/// Real-world bug surfaced 2026-05-18 by `hf2q convert-v2
/// /opt/hf2q/models/google-gemma-4-26b-a4b-it --quant q5_k_m`: the
/// outer config has only the multimodal scaffolding
/// (architectures / model_type / vision_config / text_config), and
/// `build_hparams` was reading `num_attention_heads` from the outer
/// config → MissingHparam error.
/// Resolve BERT pooling type by walking `modules.json` to find a
/// `*Pooling` entry's `path`, then reading that subdir's
/// `config.json` for `pooling_mode_*_token` booleans. Port of
/// canonical's `TextModel._try_set_pooling_type` at
/// /opt/llama.cpp/conversion/base.py:1883-1915.
///
/// Returns `Some(PoolingType as u32)` if a Pooling module is found
/// AND its config carries a recognized mode. Otherwise `None` (BERT
/// build_metadata then falls back to `config["pooling"]` or MEAN).
///
/// PoolingType enum values match canonical's `gguf.PoolingType`:
///   0 = NONE, 1 = MEAN, 2 = CLS, 3 = LAST, 4 = RANK.
fn resolve_bert_pooling_type(model_dir: &std::path::Path) -> Option<u32> {
    let modules_path = model_dir.join("modules.json");
    let modules_raw = std::fs::read_to_string(&modules_path).ok()?;
    let modules: serde_json::Value = serde_json::from_str(&modules_raw).ok()?;
    let pooling_subdir = modules
        .as_array()?
        .iter()
        .find_map(|m| {
            let ty = m.get("type")?.as_str()?;
            if ty.ends_with("Pooling") {
                m.get("path")?.as_str().map(String::from)
            } else {
                None
            }
        })?;
    let pooling_config_path = model_dir.join(pooling_subdir).join("config.json");
    let pooling_raw = std::fs::read_to_string(&pooling_config_path).ok()?;
    let pooling: serde_json::Value = serde_json::from_str(&pooling_raw).ok()?;
    if pooling.get("pooling_mode_mean_tokens").and_then(|v| v.as_bool()) == Some(true) {
        Some(1) // MEAN
    } else if pooling.get("pooling_mode_cls_token").and_then(|v| v.as_bool()) == Some(true) {
        Some(2) // CLS
    } else if pooling.get("pooling_mode_lasttoken").and_then(|v| v.as_bool()) == Some(true) {
        Some(3) // LAST
    } else if let Some(mode) = pooling.get("pooling_mode").and_then(|v| v.as_str()) {
        match mode {
            "mean" => Some(1),
            "cls" => Some(2),
            "lasttoken" => Some(3),
            _ => None,
        }
    } else {
        None
    }
}

pub fn effective_config(config: &serde_json::Value) -> &serde_json::Value {
    // Gemma 4 / Qwen3-VL-omni pattern: `text_config` is the text decoder.
    if let Some(text) = config.get("text_config") {
        return text;
    }
    // Future multimodal wrappers may use other key names; keep this
    // single-source so per-arch mappers don't each have to handle it.
    config
}

/// Extract the [`HParams`] block the orchestrator needs for
/// `target_for`'s GQA + counter-walk branches. Mirrors the convention
/// in the per-arch `build_metadata` mappers (default `n_head_kv` to
/// `n_head`, `n_expert` to zero when absent).
fn build_hparams(config: &serde_json::Value) -> Result<HParams, ConvertError> {
    let config = effective_config(config);
    // Accept BOTH HF naming conventions: most arches use
    // `num_attention_heads`; nomic-bert / older HF variants use `n_head`.
    // Mirrors canonical `find_hparam(["n_heads", "num_attention_heads"])`
    // in `/opt/llama.cpp/conversion/llama.py:131` and similar in bert.py.
    let n_head = config
        .get("num_attention_heads")
        .or_else(|| config.get("n_head"))
        .or_else(|| config.get("n_heads"))
        .and_then(|v| v.as_u64())
        .ok_or(ConvertError::MissingHparam {
            key: "num_attention_heads",
        })? as u32;
    let n_head_kv = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(n_head);
    // MoE expert count — different HF keys depending on arch. We accept
    // any of the canonical names; defaulting to 0 for dense models.
    let n_expert = config
        .get("num_experts")
        .or_else(|| config.get("num_local_experts"))
        .or_else(|| config.get("n_routed_experts"))
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(0);
    // Accept BOTH conventions: standard HF uses `num_hidden_layers`;
    // nomic-bert v2-moe uses bare `n_layer`. Same find_hparam pattern
    // as canonical at `/opt/llama.cpp/conversion/base.py`.
    let n_hidden = config
        .get("num_hidden_layers")
        .or_else(|| config.get("n_layer"))
        .and_then(|v| v.as_u64())
        .ok_or(ConvertError::MissingHparam {
            key: "num_hidden_layers",
        })?;
    // MTP-aware: include nextn-block layer count in HParams.n_layer
    // so the policy walks 0..(n_hidden+mtp) and recognizes the MTP
    // block at index n_hidden. See [`config_n_layers`] for the
    // canonical rationale.
    let n_mtp = config
        .get("mtp_num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let n_layer = (n_hidden + n_mtp) as u32;

    Ok(HParams {
        n_expert,
        n_head,
        n_head_kv,
        n_layer,
        n_mtp_layers: n_mtp as u32,
    })
}

// ============================================================================
// Per-arch dispatchers (map_tensor_name + build_metadata)
// ============================================================================

/// Per-arch synthesized-tensor dispatcher.
///
/// Returns the list of `HfTensor`s that need to be appended to the
/// safetensors-derived tensor list before staging. Currently only
/// Gemma 4 has synthesized tensors (the `rope_freqs.weight`
/// proportional-rope mask — `gemma.py:702-718`).
///
/// The synthesized tensors are pushed through the SAME mapping +
/// staging path as on-disk tensors: the per-arch mapper must recognize
/// the synthesized tensor's name and the orchestrator's F32-keep gate
/// (`orchestrator::is_f32_keep_tensor`) must emit them raw.
fn synthesized_tensors_for_arch(
    arch: ArchName,
    config: &serde_json::Value,
) -> Vec<HfTensor> {
    let config = effective_config(config);
    match arch {
        ArchName::Gemma4 => gemma4::build_synthesized_tensors(config),
        // Other arches have no synthesized tensors at v1 (Qwen3MoE,
        // MiniMaxM2, Llama3, Bert, NomicBert, Qwen3VlText, Gemma4Mmproj
        // — all read every tensor straight from safetensors).
        _ => Vec::new(),
    }
}

/// Per-arch `build_metadata` dispatcher.
///
/// Gemma4Mmproj's mapper takes a `vision_config` sub-object rather
/// than the full config; we don't drive the mmproj sidecar from
/// convert-v2 (it has its own input directory convention), so the
/// driver only ever lands on the 7 text/encoder mappers. If a caller
/// somehow routes `ArchName::Gemma4Mmproj` through here, we fall back
/// to feeding the full config — the mapper will surface a missing-key
/// panic if its required fields aren't present.
fn build_metadata_for_arch(
    arch: ArchName,
    config: &serde_json::Value,
    ftype: u32,
    model_card: Option<&crate::convert::model_card::ModelCard>,
    size_label: Option<&str>,
    sampling: Option<&crate::convert::model_card::SamplingConfig>,
    model_dir_basename: Option<&str>,
    bert_pooling_override: Option<u32>,
    qwen3vl_n_deepstack: Option<u32>,
) -> Vec<(String, MetaValue)> {
    // Multimodal-wrapper flatten: text-decoder hparams live in
    // config["text_config"] for Gemma 4 / Qwen3-VL omni-shape configs.
    // Per-arch mappers don't each have to handle this; we resolve at the
    // driver boundary. Surfaced 2026-05-18 by the operator's
    // google-gemma-4-26b-a4b-it real-model convert smoke test.
    //
    // Mmproj exception: `Gemma4Mmproj::build_metadata` expects the
    // `vision_config` SUB-OBJECT (not the text_config); we extract it
    // explicitly. Canonical's `convert_hf_to_gguf.py --mmproj` does
    // the equivalent via `MmprojModel.__init__` reading
    // `hparams["vision_config"]`.
    if matches!(arch, ArchName::Gemma4Mmproj) {
        let vision = config
            .get("vision_config")
            .expect("--mmproj routing requires vision_config (validated in run_convert)");
        return gemma4_mmproj::build_metadata(vision, ftype);
    }
    if matches!(arch, ArchName::Gemma4VisionMmproj) {
        let vision = config
            .get("vision_config")
            .expect("--mmproj routing requires vision_config (validated in run_convert)");
        // Gemma 4 text decoder hidden_size lives at root + text_config
        // (multimodal wrapper). Read root first, then text_config sub-object.
        let text_hidden = config
            .get("text_config")
            .and_then(|tc| tc.get("hidden_size"))
            .or_else(|| config.get("hidden_size"))
            .and_then(|v| v.as_u64())
            .expect("--mmproj Gemma4Vision requires text_config.hidden_size") as u32;
        return crate::convert::arch::gemma4_vision_mmproj::build_metadata(
            vision,
            text_hidden,
            ftype,
            model_card,
            sampling,
            model_dir_basename,
        );
    }
    let config = effective_config(config);
    match arch {
        ArchName::Llama3 => llama3::build_metadata(
            config,
            ftype,
            model_card,
            sampling,
            model_dir_basename,
        ),
        ArchName::Gemma4 => gemma4::build_metadata(
            config,
            ftype,
            model_card,
            sampling,
            model_dir_basename,
        ),
        ArchName::Gemma4Mmproj => unreachable!("handled above"),
        ArchName::Gemma4VisionMmproj => unreachable!("handled above"),
        ArchName::Bert => bert::build_metadata(
            config,
            ftype,
            model_card,
            sampling,
            model_dir_basename,
            bert_pooling_override,
        ),
        ArchName::NomicBert => nomic_bert::build_metadata(config, ftype, model_card, size_label),
        ArchName::Qwen35Moe => qwen35moe::build_metadata(config, ftype),
        ArchName::Qwen35MoeFull => match build_qwen35moe_full_ctx(config) {
            Some(ctx) => qwen35moe_full::build_metadata(
                &ctx,
                config,
                ftype,
                model_card,
                sampling,
                model_dir_basename,
                size_label,
            ),
            // Config missing required hparams. Fall back to the older
            // qwen3moe metadata layout (which uses `general.architecture
            // = "qwen3moe"`) so at minimum SOME metadata is written;
            // this case is also caught at map_tensor → UnmappedTensor
            // for every tensor when ctx is missing.
            None => qwen35moe::build_metadata(config, ftype),
        },
        ArchName::Qwen3VlText => qwen3vl_text::build_metadata(
            config,
            ftype,
            model_card,
            sampling,
            model_dir_basename,
            qwen3vl_n_deepstack,
        ),
        ArchName::MiniMaxM2 => minimax_m2::build_metadata(
            config,
            ftype,
            model_card,
            sampling,
            model_dir_basename,
            size_label,
        ),
        ArchName::Deepseek4 => deepseek4_metadata::build_metadata(
            config,
            ftype,
            model_card,
            sampling,
            model_dir_basename,
        ),
        // Falcon is a placeholder in ArchName for target_for's branch
        // expression; it is NOT a convert-v2 supported arch. Reaching
        // this arm means detect_arch returned Falcon, which it
        // currently never does.
        ArchName::Falcon => unreachable!(
            "ArchName::Falcon is a target_for placeholder, not a convert-v2 supported arch"
        ),
    }
}

/// What one HF tensor maps to under the selected arch's mapper.
///
/// Unifies the two mapper signatures:
///  - Dense arches expose `map_tensor_name(&str) -> Option<String>`
///    (Llama3, Gemma4, Gemma4Mmproj, Bert, NomicBert, Qwen3VlText).
///  - MoE arches expose `map_tensor_name(&str) -> Option<MappedTensor>`
///    (Qwen35Moe, MiniMaxM2).
///
/// The driver lifts both into a single shape so the staging loop has
/// one match-arm per outcome.
enum MapOutcome {
    Direct(String),
    /// 1:1 rename plus a post-load data transform applied inside
    /// `PlanStep::materialize`. Used by per-arch mappers that need to
    /// add 1.0 to norm.weight, negate-exp an SSM A_log, reorder V
    /// heads, etc. — see [`crate::convert::arch::bake::BakeOp`].
    DirectWithBake {
        gguf_name: String,
        bake: BakeOp,
    },
    /// One HF tensor splits into multiple GGUF tensors (fan-out). Used
    /// by per-arch mappers handling pre-fused safetensors layouts like
    /// the Qwen 3.5 multimodal `mlp.experts.gate_up_proj` that needs
    /// to be split into `ffn_gate_exps` + `ffn_up_exps`, or the SSM
    /// `in_proj_qkvz` that splits into `attn_qkv` + `attn_gate`. Each
    /// output carries its own GGUF name + GGUF-order shape + bake
    /// (typically a slice). Outputs are emitted in the order given;
    /// the plan-build code expands the vec into N separate
    /// `PlanStep::Direct` entries, all referencing the same HF source.
    SplitInto(Vec<SplitOutput>),
    Expert {
        gguf_name: String,
        layer: usize,
        expert_index: usize,
        kind: ExpertKind,
    },
    Drop,
    Unmapped,
}

/// One output of a [`MapOutcome::SplitInto`] fan-out. The plan-build
/// code emits a `PlanStep::Direct` for each `SplitOutput` carrying
/// `bake` (typically a [`BakeOp::Slice`] picking out this output's
/// portion of the shared HF tensor) and the GGUF-order `gguf_shape`.
#[derive(Debug, Clone)]
pub struct SplitOutput {
    pub gguf_name: String,
    pub gguf_shape: Vec<usize>,
    pub bake: BakeOp,
}

/// Per-arch context for Llama 3 Q/K RoPE-halves permute.
/// Built once at convert start from `config.json`; threaded through
/// `map_tensor` so the per-tensor mapper can attach a BakeOp without
/// re-parsing config on every tensor.
#[derive(Debug, Clone)]
struct Llama3Ctx {
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    inner: usize,
}

/// Build Llama 3 RoPE permute context from `config.json`. Returns
/// `None` if any required key is absent or if `hidden_size` isn't
/// divisible by `num_attention_heads` (would indicate a malformed
/// config; callers should surface that as `UnmappedTensor`).
///
/// Mirrors canonical `/opt/llama.cpp/conversion/llama.py:131-141` —
/// `n_head = find_hparam(["n_heads","num_attention_heads"])`;
/// `n_kv_head = find_hparam(["n_kv_heads","num_key_value_heads"])`.
fn build_llama3_ctx(config: &serde_json::Value) -> Option<Llama3Ctx> {
    let text = effective_config(config);
    let hidden_size = text.get("hidden_size")?.as_u64()? as usize;
    let n_head = text
        .get("num_attention_heads")
        .or_else(|| text.get("n_heads"))?
        .as_u64()? as usize;
    let n_kv_head = text
        .get("num_key_value_heads")
        .or_else(|| text.get("n_kv_heads"))
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(n_head);
    if n_head == 0 || hidden_size % n_head != 0 {
        return None;
    }
    Some(Llama3Ctx {
        n_head,
        n_kv_head,
        head_dim: hidden_size / n_head,
        inner: hidden_size,
    })
}

/// Attach the RoPE-halves permute bake to Llama 3 Q/K projections.
/// Mirrors canonical `llama.py:137-141`:
///   `if name.endswith("q_proj.weight"|"q_proj.bias"): permute(.., n_head, n_head)`
///   `if name.endswith("k_proj.weight"|"k_proj.bias"): permute(.., n_head, n_kv_head)`
/// (Note: `permute` overrides `n_head` to `n_kv_head` for K — see
/// `LlamaModel.permute:99-100`.)
fn llama3_attach_bake(
    gguf_name: &str,
    hf_name: &str,
    ctx: &Llama3Ctx,
) -> Option<crate::convert::arch::bake::BakeOp> {
    use crate::convert::arch::bake::BakeOp;
    let is_weight = hf_name.ends_with(".weight");
    let is_bias = hf_name.ends_with(".bias");
    let inner = if is_weight {
        ctx.inner
    } else if is_bias {
        1
    } else {
        return None;
    };
    // gguf_name strips the layer prefix; match on suffix.
    if gguf_name.ends_with("attn_q.weight") || gguf_name.ends_with("attn_q.bias") {
        Some(BakeOp::PermuteRopeHalves {
            n_head: ctx.n_head,
            head_dim: ctx.head_dim,
            inner,
        })
    } else if gguf_name.ends_with("attn_k.weight") || gguf_name.ends_with("attn_k.bias") {
        Some(BakeOp::PermuteRopeHalves {
            n_head: ctx.n_kv_head,
            head_dim: ctx.head_dim,
            inner,
        })
    } else {
        None
    }
}

fn map_tensor(
    arch: ArchName,
    hf_name: &str,
    hf_shape: &[usize],
    qwen35moe_full_ctx: Option<&Qwen35MoeFullCtx>,
    llama3_ctx: Option<&Llama3Ctx>,
    nomic_bert_ctx: &nomic_bert::NomicBertCtx,
) -> MapOutcome {
    match arch {
        ArchName::Llama3 => match llama3::map_tensor_name(hf_name) {
            Some(gguf_name) => match llama3_ctx.and_then(|c| llama3_attach_bake(&gguf_name, hf_name, c)) {
                Some(bake) => MapOutcome::DirectWithBake { gguf_name, bake },
                None => MapOutcome::Direct(gguf_name),
            },
            None => MapOutcome::Unmapped,
        },
        ArchName::Gemma4 => lift_gemma4_mapped(gemma4::map_tensor_name(hf_name)),
        ArchName::Gemma4Mmproj => match gemma4_mmproj::map_tensor_name(hf_name) {
            Some(s) => MapOutcome::Direct(s),
            // In mmproj mode we silently DROP non-vision tensors
            // (text decoder, audio, embed_audio). The arch is only
            // routed here when `--mmproj` is set (see run_convert
            // override), so the unmapped names are expected drops —
            // mirrors canonical's MMPROJ_MODEL_MAP behavior.
            None => MapOutcome::Drop,
        },
        ArchName::Gemma4VisionMmproj => {
            match crate::convert::arch::gemma4_vision_mmproj::map_tensor_name(hf_name) {
                Some(gguf_name) => {
                    // Patch embedder reshape: HF stores
                    // `patch_embedder.input_proj.weight` as 2-D
                    // `[out_features, patch_h*patch_w*channels]`; GGUF
                    // wants 4-D `[out_features, channels, patch_h, patch_w]`.
                    // Per canonical gemma.py:834-838 + ADR-033 task #73.
                    if gguf_name == "v.patch_embd.weight" {
                        // hf_shape is the SAFETENSORS shape (PyTorch
                        // order, outer-first): `[out_features, inner]`
                        // where inner = patch_h*patch_w*channels.
                        // For Gemma 4 26B-A4B-IT: [1152, 768] (= 16² × 3).
                        if hf_shape.len() == 2 {
                            let out_features = hf_shape[0];
                            let inner = hf_shape[1];
                            // Standard Gemma 4 vision: 3 RGB channels.
                            let channels = 3;
                            if inner % channels == 0 {
                                let patch_sq = inner / channels;
                                let patch_size = (patch_sq as f64).sqrt() as usize;
                                if patch_size * patch_size == patch_sq {
                                    return MapOutcome::DirectWithBake {
                                        gguf_name,
                                        bake: BakeOp::PatchEmbedderReshape {
                                            out_features,
                                            patch_h: patch_size,
                                            patch_w: patch_size,
                                            channels,
                                        },
                                    };
                                }
                            }
                        }
                    }
                    MapOutcome::Direct(gguf_name)
                }
                // Drop non-vision tensors (text decoder, etc.) silently —
                // same convention as Gemma4Mmproj.
                None => MapOutcome::Drop,
            }
        }
        ArchName::Bert => match bert::map_tensor_name(hf_name) {
            Some(s) => MapOutcome::Direct(s),
            None => MapOutcome::Unmapped,
        },
        ArchName::NomicBert => match nomic_bert::map_tensor_name(hf_name, hf_shape, nomic_bert_ctx) {
            Some(nomic_bert::MappedTensor::Direct(s)) => MapOutcome::Direct(s),
            Some(nomic_bert::MappedTensor::DirectWithBake { gguf_name, bake }) => {
                MapOutcome::DirectWithBake { gguf_name, bake }
            }
            Some(nomic_bert::MappedTensor::Drop) => MapOutcome::Drop,
            None => MapOutcome::Unmapped,
        },
        ArchName::Qwen3VlText => match qwen3vl_text::map_tensor_name(hf_name) {
            Some(s) => MapOutcome::Direct(s),
            None => {
                // Mirror canonical TextModel.filter_tensors at
                // `/opt/llama.cpp/conversion/base.py:1064-1078` which
                // SILENTLY DROPS multimodal-side tensors (visual,
                // audio, vision-projector) rather than erroring. The
                // mmproj sidecar is written by a separate `--mmproj`
                // run. Unmapped genuinely-unknown names still surface
                // as Unmapped (typed error).
                if hf_name.contains("visual.")
                    || hf_name.contains("vision.")
                    || hf_name.contains("audio.")
                    || hf_name.contains("audio_tower.")
                    || hf_name.starts_with("mtp.")
                    || hf_name.contains("patch_embed")
                    || hf_name.contains("patch_embedding")
                    || hf_name.contains("patch_merger.")
                    || hf_name.contains("merger.")
                    || hf_name.contains("vit.")
                {
                    MapOutcome::Drop
                } else {
                    MapOutcome::Unmapped
                }
            }
        },
        ArchName::Qwen35Moe => lift_qwen_mapped(qwen35moe::map_tensor_name(hf_name)),
        ArchName::Qwen35MoeFull => match qwen35moe_full_ctx {
            Some(ctx) => lift_qwen35moe_full_mapped(qwen35moe_full::map_tensor_name(
                hf_name, hf_shape, ctx,
            )),
            None => MapOutcome::Unmapped,
        },
        ArchName::MiniMaxM2 => lift_minimax_mapped(minimax_m2::map_tensor_name(hf_name)),
        ArchName::Deepseek4 => lift_qwen_mapped(deepseek4::map_tensor_name(hf_name)),
        ArchName::Falcon => MapOutcome::Unmapped,
    }
}

/// Adapt the Qwen 3.5 MoE-full mapper's `MappedTensor` shape (Direct /
/// DirectWithBake / SplitInto / ExpertGroup / Drop) to the driver-level
/// [`MapOutcome`].
fn lift_qwen35moe_full_mapped(m: Option<qwen35moe_full::MappedTensor>) -> MapOutcome {
    match m {
        Some(qwen35moe_full::MappedTensor::Direct(s)) => MapOutcome::Direct(s),
        Some(qwen35moe_full::MappedTensor::DirectWithBake { gguf_name, bake }) => {
            MapOutcome::DirectWithBake { gguf_name, bake }
        }
        Some(qwen35moe_full::MappedTensor::SplitInto(outputs)) => MapOutcome::SplitInto(outputs),
        Some(qwen35moe_full::MappedTensor::ExpertGroup {
            gguf_name,
            layer,
            expert_index,
            kind,
        }) => MapOutcome::Expert {
            gguf_name,
            layer,
            expert_index,
            kind,
        },
        Some(qwen35moe_full::MappedTensor::Drop) => MapOutcome::Drop,
        None => MapOutcome::Unmapped,
    }
}

/// Build a [`Qwen35MoeFullCtx`] from the HF `config.json` for arches
/// dispatched to [`qwen35moe_full::map_tensor_name`]. Returns `None`
/// if any required hparam is missing — that surfaces as
/// `MapOutcome::Unmapped` for every tensor, producing a typed
/// `UnmappedTensor` error per the no-fallback rule.
///
/// Reads from the effective text-config (top-level OR nested
/// `text_config` for multimodal-wrapping `ConditionalGeneration`
/// variants). Detects multimodal wrapping from
/// `architectures` containing `ForConditionalGeneration`.
/// Compute `general.size_label` for arches that emit it. Returns
/// `None` when the arch doesn't (yet) participate in the size_label
/// metadata field — currently only `ArchName::NomicBert` on the
/// v2-moe path. Mirrors canonical's
/// `gguf-py/gguf/utility.py::size_label` formula.
///
/// For nomic v2-moe expert tensors are detected by HF tensor-name
/// pattern `.mlp.experts.mlp.w` (suffix `w1` or `w2`). All other
/// tensors are "shared". `num_experts` from config drives the
/// per-expert division.
fn compute_size_label_for_arch(
    arch: ArchName,
    src: &HfModelSource,
    config: &serde_json::Value,
) -> Option<String> {
    use crate::convert::model_card::compute_size_label;
    match arch {
        ArchName::MiniMaxM2 => {
            // MiniMax-M2 MoE: experts under
            // `model.layers.<N>.block_sparse_moe.experts.<E>.w[1-3]`.
            // Canonical `Metadata.set_size_label` (utility.py:44-52)
            // formats as `{expert_count}x{pretty_size}`.
            let n_experts = config
                .get("num_local_experts")
                .or_else(|| config.get("num_experts"))
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)?;
            let iter = src.tensor_metas().map(|m| {
                let is_expert = m.name.contains(".block_sparse_moe.experts.");
                (m.numel() as u64, is_expert)
            });
            Some(compute_size_label(iter, n_experts))
        }
        ArchName::NomicBert => {
            let nomic_ctx = build_nomic_bert_ctx(config);
            let n_experts = nomic_ctx.num_experts? as u32;
            // Drop `mlp.experts.bias` from the walk (canonical's
            // `NomicBertModel.filter_tensors` at `bert.py:366-369`
            // discards it; including it inflates `shared_params`).
            let iter = src.tensor_metas().filter_map(|m| {
                let stripped = m.name.strip_prefix("bert.").unwrap_or(&m.name);
                if stripped.contains("mlp.experts.bias") {
                    return None;
                }
                let is_expert = stripped.contains(".mlp.experts.mlp.w");
                Some((m.numel() as u64, is_expert))
            });
            Some(compute_size_label(iter, n_experts))
        }
        _ => None,
    }
}

/// Build the `NomicBertCtx` from the HF `config.json`. v1.5 returns a
/// ctx with `num_experts = None`; v2-moe carries
/// `num_experts` / `num_local_experts` per canonical
/// `/opt/llama.cpp/conversion/bert.py:372`. Missing-on-v2-moe surfaces
/// to the convert pipeline as `UnmappedTensor` on the expert
/// tensors — typed error per the no-fallback rule.
fn build_nomic_bert_ctx(config: &serde_json::Value) -> nomic_bert::NomicBertCtx {
    let num_experts = config
        .get("num_experts")
        .or_else(|| config.get("num_local_experts"))
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    nomic_bert::NomicBertCtx { num_experts }
}

fn build_qwen35moe_full_ctx(config: &serde_json::Value) -> Option<Qwen35MoeFullCtx> {
    let text = effective_config(config);
    let num_hidden_layers = text.get("num_hidden_layers")?.as_u64()? as usize;
    let num_experts = text
        .get("num_experts")
        .or_else(|| text.get("num_local_experts"))?
        .as_u64()? as usize;
    let moe_intermediate_size = text.get("moe_intermediate_size")?.as_u64()? as usize;
    let hidden_size = text.get("hidden_size")?.as_u64()? as usize;
    let linear_num_key_heads = text.get("linear_num_key_heads")?.as_u64()? as usize;
    let linear_num_value_heads = text.get("linear_num_value_heads")?.as_u64()? as usize;
    let linear_key_head_dim = text.get("linear_key_head_dim")?.as_u64()? as usize;
    let linear_value_head_dim = text.get("linear_value_head_dim")?.as_u64()? as usize;

    let multimodal_wrapping = config
        .get("architectures")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str())
                .any(|s| s.ends_with("ForConditionalGeneration"))
        })
        .unwrap_or(false);

    // HF2Q_QWEN35_DROP_MTP=1 (or =true, case-insensitive) — convert-time
    // workaround for stock llama.cpp's lack of a qwen35 MTP loader. Strips
    // the MTP block from the emitted GGUF (loses MTP inference). See
    // `Qwen35MoeFullCtx::drop_mtp`.
    let drop_mtp = matches!(
        std::env::var("HF2Q_QWEN35_DROP_MTP").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("True")
    );

    // Ergonomics: stock llama.cpp (current stable) lacks a qwen35 MTP
    // loader — it expects all blocks to share the linear-attention slot
    // layout, so the MTP block triggers `missing tensor 'blk.<N>.ssm_conv1d.weight'`
    // on load. Without `HF2Q_QWEN35_DROP_MTP=1` the produced GGUF is
    // unloadable by current llama.cpp / llama-cli / llama-perplexity.
    // hf2q's own inference path DOES handle MTP, so the strip is only
    // necessary for llama.cpp interop. Warn so operators discover this
    // at convert time rather than at load time. See src/arch/smoke.rs:464-479.
    if !drop_mtp {
        // Inline lookup since `effective_text_config` is per-arch private
        // (lives in qwen35moe_full.rs). Mirror its shape: prefer the nested
        // `text_config` for multimodal `ConditionalGeneration`, else root.
        let text_for_mtp_check = config.get("text_config").unwrap_or(config);
        let n_mtp_raw = text_for_mtp_check
            .get("mtp_num_hidden_layers")
            .and_then(|v: &serde_json::Value| v.as_u64())
            .unwrap_or(0u64);
        if n_mtp_raw > 0 {
            eprintln!(
                "[hf2q convert] note: qwen35moe model has {n_mtp_raw} MTP block(s); \
                 the resulting GGUF will NOT load in stock llama.cpp (current stable \
                 lacks a qwen35 MTP loader). Set HF2Q_QWEN35_DROP_MTP=1 to strip \
                 the MTP block(s) for llama.cpp interop (loses MTP inference, but \
                 hf2q's own inference path handles MTP separately)."
            );
        }
    }

    Some(Qwen35MoeFullCtx {
        num_hidden_layers,
        num_experts,
        moe_intermediate_size,
        hidden_size,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_key_head_dim,
        linear_value_head_dim,
        multimodal_wrapping,
        drop_mtp,
    })
}

/// Adapt the Gemma 4 mapper's `MappedTensor` shape (`Direct` /
/// `Drop`) to the unified driver `MapOutcome`. Gemma 4 needs the `Drop`
/// variant because its safetensors contain vision/audio sidecar
/// tensors (`model.vision_tower.*`, `model.embed_vision.*`, etc.) that
/// the per-arch mapper signs off as off-path for the text-decoder GGUF;
/// the dense `Option<String>` shape can't express that distinction
/// without conflating it with "unmapped tensor = bug".
///
/// Surfaced 2026-05-18 by the real-model finding at
/// `docs/adr-033-real-model-findings/2026-05-18-gemma4-arch-mismatch.md`
/// — the operator's google-gemma-4-26b-a4b-it ships 220+ vision-tower
/// tensors alongside the text decoder, and the convert-v2 driver must
/// silently route those to the mmproj sidecar instead of erroring.
fn lift_gemma4_mapped(m: Option<Gemma4Mapped>) -> MapOutcome {
    match m {
        Some(Gemma4Mapped::Direct(s)) => MapOutcome::Direct(s),
        Some(Gemma4Mapped::Drop) => MapOutcome::Drop,
        None => MapOutcome::Unmapped,
    }
}

/// Adapt Qwen35MoE's `MappedTensor` shape (Direct / ExpertGroup / Drop)
/// to the unified driver `MapOutcome`. Per the qwen35moe.rs module-level
/// comment, this is the "first MoE arch" enum shape — its `ExpertKind`
/// flavors (Gate / Up / Down) map directly onto the driver's
/// accumulator key.
fn lift_qwen_mapped(m: Option<QwenMapped>) -> MapOutcome {
    match m {
        Some(QwenMapped::Direct(s)) => MapOutcome::Direct(s),
        Some(QwenMapped::ExpertGroup {
            gguf_name,
            layer,
            expert_index,
            kind,
        }) => MapOutcome::Expert {
            gguf_name,
            layer,
            expert_index,
            kind,
        },
        Some(QwenMapped::Drop) => MapOutcome::Drop,
        None => MapOutcome::Unmapped,
    }
}

/// Adapt MiniMax-M2's distinct `MappedTensor` shape (Dense / Router /
/// ExpertWeight) to the unified driver `MapOutcome`.
///
/// Mapping rationale:
///  - `Dense { gguf, .. }` and `Router { gguf, .. }` collapse to
///    `Direct(gguf)` — both surface as 1:1 renames at the driver layer.
///    The Dense-vs-Router distinction matters for QUANT policy
///    selection inside the orchestrator (Router tensors might want a
///    different policy in the future), but the driver does not gate
///    on it.
///  - `ExpertWeight { layer, expert, role, gguf_stacked, .. }` carries
///    the same load-bearing info as Qwen's `ExpertGroup` but in a
///    distinct enum shape. We translate `role` (Gate/Up/Down) onto
///    Qwen's `ExpertKind` so the driver's MoE accumulator has one
///    canonical key type.
fn lift_minimax_mapped(m: Option<MiniMaxMapped>) -> MapOutcome {
    match m {
        Some(MiniMaxMapped::Dense { gguf, .. }) => MapOutcome::Direct(gguf),
        Some(MiniMaxMapped::Router { gguf, .. }) => MapOutcome::Direct(gguf),
        Some(MiniMaxMapped::ExpertWeight {
            layer,
            expert,
            role,
            gguf_stacked,
            ..
        }) => MapOutcome::Expert {
            gguf_name: gguf_stacked,
            layer: layer as usize,
            expert_index: expert as usize,
            kind: expert_role_to_kind(role),
        },
        None => MapOutcome::Unmapped,
    }
}

fn expert_role_to_kind(r: ExpertRole) -> ExpertKind {
    match r {
        ExpertRole::Gate => ExpertKind::Gate,
        ExpertRole::Up => ExpertKind::Up,
        ExpertRole::Down => ExpertKind::Down,
    }
}

// ============================================================================
// Convert plan + streaming MoE expert fusion
// ============================================================================
//
// Per ADR-033 §"Open Issues / Real-Model Findings" 2026-05-18: the
// previous staging path collected every HF tensor's F32 payload into a
// `Vec<HfTensor>` and then handed the full vector to the orchestrator
// (which copied it again into its internal `Vec<StagedTensor>`). For
// Gemma 4 26B this peaked at ~104 GB RSS on a 48 GB safetensors source
// and got SIGKILL'd on a 64 GB Mac. The new staging path runs in two
// metadata-only passes (plan + streaming-iteration index) and a single
// data pass (one tensor's F32 buffer alive at a time).
//
// The MoE expert-fusion case consumes multiple HF tensors into one GGUF
// tensor. It streams complete expert slices in expert-index order, so
// its input bound is ONE expert rather than the full fused group.

/// One step of the convert plan: either a direct 1:1 HF→GGUF mapping,
/// an MoE expert fusion that consumes N HF tensors, or a synthesized
/// tensor (produced by `synthesized_tensors_for_arch`, not on disk).
#[derive(Debug, Clone)]
enum PlanStep {
    /// One HF safetensors tensor → one GGUF tensor (1:1 rename + shape
    /// reverse). Carries the canonical GGUF name + shape + source dtype
    /// + (optional) layer index.
    Direct {
        hf_name: String,
        gguf_name: String,
        /// GGUF-order shape (PyTorch order reversed). When `bake` is a
        /// [`BakeOp::Slice`] this shape reflects the SLICED output
        /// (i.e. the GGUF shape after the slice is applied), not the
        /// raw HF tensor shape; the slice itself is the source-side
        /// selector applied during `materialize`.
        gguf_shape: Vec<usize>,
        source_dtype: SourceDtype,
        layer_index: Option<usize>,
        /// Post-load data transform applied in `materialize` after the
        /// F32 buffer is read from safetensors. `None` is a pure
        /// rename (the prior behavior pre-ADR-034-P2). `Some(...)`
        /// triggers a Qwen 3.5/3.6 transform per the
        /// [`crate::convert::arch::bake::BakeOp`] vocabulary.
        bake: Option<BakeOp>,
    },
    /// N MoE expert slices fuse into one 3-D GGUF tensor of shape
    /// `[in, out, n_experts]`. `member_hf_names` is in expert_index
    /// order (sorted at plan build time).
    Fused {
        gguf_name: String,
        /// GGUF-order shape = `[per_expert_shape.reverse(), n_experts]`.
        gguf_shape_fused: Vec<usize>,
        /// HF tensor names of every expert slice, sorted by
        /// expert_index so the stream-time concatenation produces the
        /// `torch.stack(slices, dim=0)` byte layout.
        member_hf_names: Vec<String>,
        /// PyTorch-order shape of ONE expert slice. Used for the
        /// per-slice length-check at stream time.
        per_expert_py_shape: Vec<usize>,
        source_dtype: SourceDtype,
        layer_index: Option<usize>,
    },
    /// A synthesized tensor (currently only Gemma 4's `rope_freqs.weight`).
    /// `synth_idx` indexes into the synthesized tensor list passed to
    /// `PlanStep::materialize`.
    Synthesized {
        gguf_name: String,
        gguf_shape: Vec<usize>,
        source_dtype: SourceDtype,
        layer_index: Option<usize>,
        synth_idx: usize,
    },
}

impl PlanStep {
    /// Cheap projection to an orchestrator `PlanEntry` (no payload).
    fn plan_entry(&self) -> PlanEntry {
        match self {
            PlanStep::Direct {
                gguf_name,
                gguf_shape,
                source_dtype,
                layer_index,
                ..
            } => PlanEntry {
                name: gguf_name.clone(),
                shape: gguf_shape.clone(),
                source_dtype: *source_dtype,
                layer_index: *layer_index,
            },
            PlanStep::Fused {
                gguf_name,
                gguf_shape_fused,
                source_dtype,
                layer_index,
                ..
            } => PlanEntry {
                name: gguf_name.clone(),
                shape: gguf_shape_fused.clone(),
                source_dtype: *source_dtype,
                layer_index: *layer_index,
            },
            PlanStep::Synthesized {
                gguf_name,
                gguf_shape,
                source_dtype,
                layer_index,
                ..
            } => PlanEntry {
                name: gguf_name.clone(),
                shape: gguf_shape.clone(),
                source_dtype: *source_dtype,
                layer_index: *layer_index,
            },
        }
    }

    /// Pull the F32 data for a non-fused step. Fused tensors use the
    /// begin/chunk/finish writer path and must never materialize here.
    fn materialize(
        &self,
        src: &HfModelSource,
        synthesized: &[HfTensor],
    ) -> Result<Vec<f32>, ConvertError> {
        match self {
            PlanStep::Direct { hf_name, bake, .. } => {
                let ht = src.materialize_tensor(hf_name)?;
                match bake {
                    None => Ok(ht.data),
                    Some(op) => bake::apply_bake_op(ht.data, op).map_err(|e| {
                        ConvertError::Source(SourceError::Safetensors(format!(
                            "bake op failed on `{hf_name}`: {e}"
                        )))
                    }),
                }
            }
            PlanStep::Fused { gguf_name, .. } => Err(ConvertError::Source(
                SourceError::Safetensors(format!(
                    "fused tensor `{gguf_name}` must use bounded chunk streaming"
                )),
            )),
            PlanStep::Synthesized { synth_idx, .. } => {
                let t = synthesized.get(*synth_idx).ok_or_else(|| {
                    ConvertError::Source(SourceError::Safetensors(format!(
                        "synthesized tensor index {synth_idx} out of range"
                    )))
                })?;
                Ok(t.data.clone())
            }
        }
    }
}

/// A complete convert plan: every step in deterministic emission order.
/// Built once from the source's tensor metadata + the synthesized
/// tensor list; consumed twice — once by the orchestrator's plan-phase
/// (metadata only) and once by the streaming-write phase. Direct steps
/// materialize once; fused steps materialize and drop one expert slice
/// per chunk.
struct ConvertPlan {
    steps: Vec<PlanStep>,
}

/// Build the convert plan from the source's tensor metadata + the
/// synthesized tensor list. **No payload bytes touched.**
///
/// The plan walks the source's metadata once, classifying each tensor
/// via `map_tensor`:
///   - `Direct` → push a `PlanStep::Direct` entry in source order.
///   - `Expert` → buffer into a `(layer, kind)` accumulator (just the
///     HF name + expert index + per-expert PyTorch shape — no data).
///   - `Drop` → trace and skip.
///   - `Unmapped` → typed error.
///
/// The accumulator is drained at the end into `PlanStep::Fused` entries
/// in `(layer, kind)` order (deterministic — matches the previous
/// buffered staging behavior). Synthesized tensors are appended in
/// their original order.
///
/// Per [[feedback-no-loop-suppression-2026-05-17]]: incomplete /
/// duplicate / non-contiguous expert groups surface as typed errors
/// here, before any GGUF bytes are written.
fn build_convert_plan(
    arch: ArchName,
    src: &HfModelSource,
    synthesized: &[HfTensor],
) -> Result<ConvertPlan, ConvertError> {
    // Multimodal-wrapping configs nest the text-decoder hparams under
    // `text_config` (Qwen3_5MoeForConditionalGeneration / Gemma 4 omni).
    // effective_config returns text_config when present, otherwise the
    // root — same convention as build_metadata.
    let n_experts_cfg = effective_config(&src.config);
    let n_experts = n_experts_cfg
        .get("num_experts")
        .or_else(|| n_experts_cfg.get("num_local_experts"))
        .or_else(|| n_experts_cfg.get("n_routed_experts"))
        .and_then(|v| v.as_u64())
        .map(|x| x as usize);

    // Build per-arch context (e.g. Qwen 3.5 MoE-full needs hparams
    // for V-head reorder, MTP layer shift, expert split). For arches
    // that don't need a ctx this is None; for arches that DO need
    // one but the config is missing required fields it's also None
    // and every tensor surfaces as UnmappedTensor — per the
    // no-fallback rule.
    let qwen35moe_full_ctx: Option<Qwen35MoeFullCtx> = match arch {
        ArchName::Qwen35MoeFull => build_qwen35moe_full_ctx(&src.config),
        _ => None,
    };
    let llama3_ctx: Option<Llama3Ctx> = match arch {
        ArchName::Llama3 => build_llama3_ctx(&src.config),
        _ => None,
    };
    let nomic_bert_ctx: nomic_bert::NomicBertCtx = match arch {
        ArchName::NomicBert => build_nomic_bert_ctx(&src.config),
        _ => nomic_bert::NomicBertCtx { num_experts: None },
    };

    let mut direct_steps: Vec<PlanStep> = Vec::new();
    let mut moe_accum: HashMap<(usize, ExpertKindKey), MoePlanGroup> = HashMap::new();

    for meta in src.tensor_metas() {
        match map_tensor(arch, &meta.name, &meta.shape, qwen35moe_full_ctx.as_ref(), llama3_ctx.as_ref(), &nomic_bert_ctx) {
            MapOutcome::Direct(gguf_name) => {
                let gguf_shape: Vec<usize> = meta.shape.iter().rev().copied().collect();
                let layer_index = gguf_name
                    .strip_prefix("blk.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok());
                direct_steps.push(PlanStep::Direct {
                    hf_name: meta.name.clone(),
                    gguf_name,
                    gguf_shape,
                    source_dtype: meta.source_dtype,
                    layer_index,
                    bake: None,
                });
            }
            MapOutcome::DirectWithBake { gguf_name, bake } => {
                let mut gguf_shape: Vec<usize> = meta.shape.iter().rev().copied().collect();
                // Squeeze drops every singleton dim from gguf_shape —
                // the safetensors stores Qwen 3.5/3.6 linear_attn
                // `conv1d.weight` as `[hidden, 1, kernel]` (3-D) but
                // GGUF + llama.cpp's ggml_ssm_conv expect `[hidden,
                // kernel]` (2-D matrix). The element data is preserved
                // bit-exact; only the shape vector changes.
                // Also handle Squeeze nested inside Sequence (the
                // canonical conv1d composite: Squeeze + V-portion
                // ReorderVHeads).
                fn contains_squeeze(op: &BakeOp) -> bool {
                    match op {
                        BakeOp::Squeeze => true,
                        BakeOp::Sequence(inner) => inner.iter().any(contains_squeeze),
                        _ => false,
                    }
                }
                if contains_squeeze(&bake) {
                    gguf_shape.retain(|d| *d != 1);
                }
                // MoE expert weight bakes: replace the 2-D safetensors
                // shape `[E*F, H]` with the canonical 3-D layout. GGUF
                // order is fastest-varying first, so:
                //   MoeExpertReshape  → `[H, F, E]` (HF→ no data move)
                //   MoeExpertTranspose → `[F, H, E]` (per-expert transpose)
                match &bake {
                    BakeOp::MoeExpertReshape {
                        n_experts,
                        n_inner,
                        n_embd,
                    } => {
                        gguf_shape = vec![*n_embd, *n_inner, *n_experts];
                    }
                    BakeOp::MoeExpertTranspose {
                        n_experts,
                        n_inner,
                        n_embd,
                    } => {
                        gguf_shape = vec![*n_inner, *n_embd, *n_experts];
                    }
                    BakeOp::PatchEmbedderReshape {
                        out_features,
                        patch_h,
                        patch_w,
                        channels,
                    } => {
                        // 2-D HF `[out, h*w*c]` → 4-D logical
                        // `[out, c, h, w]`. GGUF stores innermost-first,
                        // so the dump shows `[w, h, c, out]`.
                        gguf_shape = vec![*patch_w, *patch_h, *channels, *out_features];
                    }
                    _ => {}
                }
                let layer_index = gguf_name
                    .strip_prefix("blk.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok());
                direct_steps.push(PlanStep::Direct {
                    hf_name: meta.name.clone(),
                    gguf_name,
                    gguf_shape,
                    source_dtype: meta.source_dtype,
                    layer_index,
                    bake: Some(bake),
                });
            }
            MapOutcome::SplitInto(outputs) => {
                if outputs.is_empty() {
                    return Err(ConvertError::UnmappedTensor {
                        hf_name: meta.name.clone(),
                        arch: arch.name().to_string(),
                    });
                }
                for out in outputs {
                    let layer_index = out
                        .gguf_name
                        .strip_prefix("blk.")
                        .and_then(|s| s.split('.').next())
                        .and_then(|s| s.parse::<usize>().ok());
                    direct_steps.push(PlanStep::Direct {
                        hf_name: meta.name.clone(),
                        gguf_name: out.gguf_name,
                        gguf_shape: out.gguf_shape,
                        source_dtype: meta.source_dtype,
                        layer_index,
                        bake: Some(out.bake),
                    });
                }
            }
            MapOutcome::Expert {
                gguf_name,
                layer,
                expert_index,
                kind,
            } => {
                let key = (layer, ExpertKindKey::from(kind));
                let group = moe_accum.entry(key).or_insert_with(|| MoePlanGroup {
                    gguf_name: gguf_name.clone(),
                    kind,
                    members: Vec::with_capacity(n_experts.unwrap_or(0)),
                    per_expert_py_shape: meta.shape.clone(),
                    source_dtype: meta.source_dtype,
                });
                // Detect duplicate expert indices (mapper bug or
                // corrupt checkpoint). Per no-loop-suppression: surface
                // instead of silent overwrite.
                if group.members.iter().any(|m| m.expert_index == expert_index) {
                    return Err(ConvertError::DuplicateExpertIndex {
                        gguf_name,
                        layer,
                        kind_label: expert_kind_label(kind),
                        expert_index,
                    });
                }
                group.members.push(MoePlanMember {
                    hf_name: meta.name.clone(),
                    expert_index,
                });
            }
            MapOutcome::Drop => {
                // The arch mapper signed off on dropping this name;
                // tracing it lets operators audit what was discarded
                // without changing behavior.
                tracing::debug!(
                    target: "convert",
                    arch = arch.name(),
                    tensor = %meta.name,
                    "convert: explicit drop per arch mapper"
                );
            }
            MapOutcome::Unmapped => {
                return Err(ConvertError::UnmappedTensor {
                    hf_name: meta.name.clone(),
                    arch: arch.name().to_string(),
                });
            }
        }
    }

    // ----- Drain MoE accumulator into fused plan steps --------------------
    let expected_n_experts = n_experts.unwrap_or(0);
    let mut groups: Vec<((usize, ExpertKindKey), MoePlanGroup)> =
        moe_accum.into_iter().collect();
    // Deterministic emission order: by (layer, kind). Two convert-v2
    // runs on the same input produce identical plan orders.
    groups.sort_by_key(|(k, _)| (k.0, k.1 as u8));

    let mut fused_steps: Vec<PlanStep> = Vec::with_capacity(groups.len());
    for ((layer, _kind_key), group) in groups {
        let MoePlanGroup {
            gguf_name,
            kind,
            mut members,
            per_expert_py_shape,
            source_dtype,
        } = group;
        if expected_n_experts == 0 {
            return Err(ConvertError::IncompleteExpertGroup {
                gguf_name,
                layer,
                kind_label: expert_kind_label(kind),
                present_count: members.len(),
                n_experts_config: 0,
            });
        }
        if members.len() != expected_n_experts {
            return Err(ConvertError::IncompleteExpertGroup {
                gguf_name,
                layer,
                kind_label: expert_kind_label(kind),
                present_count: members.len(),
                n_experts_config: expected_n_experts,
            });
        }
        members.sort_by_key(|m| m.expert_index);

        // Sanity: expert indices are contiguous [0, n_experts).
        for (i, m) in members.iter().enumerate() {
            if m.expert_index != i {
                return Err(ConvertError::IncompleteExpertGroup {
                    gguf_name,
                    layer,
                    kind_label: expert_kind_label(kind),
                    present_count: members.len(),
                    n_experts_config: expected_n_experts,
                });
            }
        }

        // Per qwen35moe.rs module-level docs (§"MoE expert FUSION"):
        // each per-expert PyTorch shape `[out, in]` reversed to GGUF
        // `[in, out]`, then an outer `n_experts` slot appended →
        // fused GGUF shape `[in, out, n_experts]` (innermost-first).
        let mut gguf_shape_fused: Vec<usize> =
            per_expert_py_shape.iter().rev().copied().collect();
        gguf_shape_fused.push(expected_n_experts);

        let member_hf_names: Vec<String> = members.into_iter().map(|m| m.hf_name).collect();
        fused_steps.push(PlanStep::Fused {
            gguf_name,
            gguf_shape_fused,
            member_hf_names,
            per_expert_py_shape,
            source_dtype,
            layer_index: Some(layer),
        });
    }

    // ----- Append synthesized tensors -------------------------------------
    // Currently only Gemma 4's `rope_freqs.weight`; routed through
    // `map_tensor` to get the canonical GGUF name (`Direct` outcome).
    // The driver's old code path appended them to a `Vec<HfTensor>`
    // BEFORE staging; we mirror the same insertion order here so the
    // GGUF layout matches byte-for-byte.
    let mut synth_steps: Vec<PlanStep> = Vec::new();
    for (synth_idx, t) in synthesized.iter().enumerate() {
        match map_tensor(arch, &t.name, &t.shape, qwen35moe_full_ctx.as_ref(), llama3_ctx.as_ref(), &nomic_bert_ctx) {
            MapOutcome::Direct(gguf_name) => {
                let gguf_shape: Vec<usize> = t.shape.iter().rev().copied().collect();
                let layer_index = gguf_name
                    .strip_prefix("blk.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok());
                synth_steps.push(PlanStep::Synthesized {
                    gguf_name,
                    gguf_shape,
                    source_dtype: t.source_dtype,
                    layer_index,
                    synth_idx,
                });
            }
            MapOutcome::Drop => {
                tracing::debug!(
                    target: "convert",
                    arch = arch.name(),
                    tensor = %t.name,
                    "convert: synthesized tensor explicit-drop per arch mapper"
                );
            }
            MapOutcome::DirectWithBake { .. } | MapOutcome::SplitInto(_) => {
                // BakeOp transforms and SplitInto fan-out are NOT
                // supported for synthesized tensors. Synthesized
                // tensors are produced internally (e.g. Gemma 4's
                // rope_freqs) at known shape with no further transform
                // needed; if a future arch's synthesized tensor needs
                // a bake, route it through `PlanStep::Synthesized`
                // with a new `bake` field — that surface change is
                // deferred until an actual caller appears. Per
                // [[feedback-no-loop-suppression-2026-05-17]]: surface
                // as a hard error, no silent fallback.
                return Err(ConvertError::UnmappedTensor {
                    hf_name: t.name.clone(),
                    arch: arch.name().to_string(),
                });
            }
            MapOutcome::Expert { .. } => {
                // Synthesized MoE experts are not currently produced by
                // any arch. If a future arch needs them, route through
                // the same fusion accumulator above. For now this is a
                // hard error rather than a silent skip.
                return Err(ConvertError::UnmappedTensor {
                    hf_name: t.name.clone(),
                    arch: arch.name().to_string(),
                });
            }
            MapOutcome::Unmapped => {
                return Err(ConvertError::UnmappedTensor {
                    hf_name: t.name.clone(),
                    arch: arch.name().to_string(),
                });
            }
        }
    }

    // Final plan order: direct → fused → synthesized. Matches the
    // previous buffered staging path (direct tensors emitted in source
    // order, fused tensors emitted after all directs, synthesized
    // appended last per the old `hf_tensors.extend(synthesized_*)` ordering).
    let mut steps: Vec<PlanStep> =
        Vec::with_capacity(direct_steps.len() + fused_steps.len() + synth_steps.len());
    steps.extend(direct_steps);
    steps.extend(fused_steps);
    steps.extend(synth_steps);

    // Canonical-equivalent sort: llama-quantize stores tensors in a
    // `std::map<string, ..., weight_name_comparer>` (see
    // `/opt/llama.cpp/src/llama-model-loader.h:53-64`), so the output
    // GGUF emits tensors in the comparator's order:
    //   - non-`blk.N.*` tensors first (layer = -1 from failed sscanf),
    //     sorted alphabetically among themselves
    //   - then `blk.N.*` tensors with numeric N order, within each
    //     layer alphabetical by GGUF name
    //
    // hf2q's convert+quantize is a single pipeline (no separate
    // llama-quantize step), so we apply the same sort here before
    // emitting plan steps. Applied to ALL arches as of 2026-05-20
    // evening — the earlier per-arch gating was based on stale
    // byte-cmp claims (ADR-033 §10 re-validation note at commit
    // bbc9ab8e). Canonical sorts every model the same way; we mirror.
    if matches!(arch, ArchName::Gemma4Mmproj | ArchName::Gemma4VisionMmproj) {
        // Canonical mmproj output preserves HF safetensors iteration
        // order (which is alphabetical by HF tensor name in the
        // model.safetensors.index.json that
        // `convert_hf_to_gguf.py` iterates). Sorting by GGUF name
        // (canonical_tensor_name_cmp) would shuffle the per-block
        // tensor order vs canonical — verified against
        // /tmp/gemma_canon_mmproj_f16.gguf dump: canonical layer-0
        // order is ln1, ffn_down, ffn_gate, ffn_up, attn_post_norm,
        // ffn_post_norm, ln2, attn_k_norm, attn_k, attn_out,
        // attn_q_norm, attn_q, attn_v — which matches the HF index's
        // alphabetical-by-HF-name order.
        steps.sort_by(|a, b| {
            // mmproj doesn't fuse MoE experts (Fused) — only Direct +
            // Synthesized. For Fused (shouldn't appear), fall back to
            // the gguf_name. For Synthesized, use gguf_name. For Direct,
            // sort by HF source name (the canonical iteration key).
            let key_of = |step: &PlanStep| -> String {
                match step {
                    PlanStep::Direct { hf_name, .. } => hf_name.clone(),
                    PlanStep::Fused { gguf_name, .. } => gguf_name.clone(),
                    PlanStep::Synthesized { gguf_name, .. } => gguf_name.clone(),
                }
            };
            key_of(a).cmp(&key_of(b))
        });
    } else {
        steps.sort_by(|a, b| {
            canonical_tensor_name_cmp(
                a.plan_entry().name.as_str(),
                b.plan_entry().name.as_str(),
            )
        });
    }
    Ok(ConvertPlan { steps })
}

/// Port of canonical's `weight_name_comparer` at
/// `/opt/llama.cpp/src/llama-model-loader.h:53-64`. Sorts tensor names
/// so that non-`blk.N.` names come first (alphabetically), then
/// `blk.N.` names with numeric N order, then alphabetical within
/// each layer.
fn canonical_tensor_name_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let a_layer = parse_blk_layer(a);
    let b_layer = parse_blk_layer(b);
    if a_layer != b_layer {
        return a_layer.cmp(&b_layer);
    }
    a.cmp(b)
}

/// Extract the layer index from a `blk.<N>.` prefixed name, mirroring
/// canonical's `sscanf(a.c_str(), "blk.%d.", &a_layer)` initialized
/// to `-1`. Returns `-1` when the prefix doesn't match (so non-`blk.`
/// names sort BEFORE `blk.N.` names per the i32 ordering).
fn parse_blk_layer(name: &str) -> i32 {
    let Some(rest) = name.strip_prefix("blk.") else {
        return -1;
    };
    let mut end = 0;
    for (i, c) in rest.char_indices() {
        if c.is_ascii_digit() {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return -1;
    }
    // Require a `.` immediately after the digits (matches `blk.%d.`
    // literal in the canonical sscanf).
    if !rest[end..].starts_with('.') {
        return -1;
    }
    rest[..end].parse::<i32>().unwrap_or(-1)
}

/// Inside-MoE-accumulator membership record. One per `(layer, kind,
/// expert_index)` triple. **No payload bytes** — the F32 data is loaded
/// lazily during the streaming-write phase from the HF name.
struct MoePlanMember {
    hf_name: String,
    expert_index: usize,
}

/// Per-(layer, kind) MoE accumulator entry used during plan building.
struct MoePlanGroup {
    gguf_name: String,
    kind: ExpertKind,
    members: Vec<MoePlanMember>,
    /// PyTorch-order shape of ONE expert slice. The fused GGUF shape is
    /// `per_expert_py_shape.reverse() ++ [n_experts]`.
    per_expert_py_shape: Vec<usize>,
    source_dtype: SourceDtype,
}

// ----- ExpertKind helpers ----------------------------------------------------
// `ExpertKind` is `Copy`, but `HashMap` keys need `Eq + Hash` which the
// upstream type already implements. We re-key as a tiny `repr(u8)` enum
// so we can both Eq-compare and serialize the sort order without
// touching the upstream type.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
enum ExpertKindKey {
    Gate = 0,
    Up = 1,
    Down = 2,
}

impl From<ExpertKind> for ExpertKindKey {
    fn from(k: ExpertKind) -> Self {
        match k {
            ExpertKind::Gate => ExpertKindKey::Gate,
            ExpertKind::Up => ExpertKindKey::Up,
            ExpertKind::Down => ExpertKindKey::Down,
        }
    }
}

fn expert_kind_label(k: ExpertKind) -> &'static str {
    match k {
        ExpertKind::Gate => "gate",
        ExpertKind::Up => "up",
        ExpertKind::Down => "down",
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::ggml_quants::apex::ApexTier;
    use crate::quantize::ggml_quants::LlamaFtype;
    use serde_json::json;

    /// model_type=llama → ArchName::Llama3.
    #[test]
    fn detect_arch_llama_from_model_type() {
        let cfg = json!({ "model_type": "llama" });
        assert_eq!(detect_arch(&cfg).unwrap(), ArchName::Llama3);
    }

    /// model_type=gemma3 → ArchName::Gemma4. Also `gemma` (Gemma 2 / older).
    #[test]
    fn detect_arch_gemma_from_model_type() {
        assert_eq!(
            detect_arch(&json!({ "model_type": "gemma3" })).unwrap(),
            ArchName::Gemma4
        );
        assert_eq!(
            detect_arch(&json!({ "model_type": "gemma" })).unwrap(),
            ArchName::Gemma4
        );
    }

    /// Real Gemma 4 release strings: model_type="gemma4" / "gemma4_text",
    /// architectures=["Gemma4ForConditionalGeneration"]. Surfaced 2026-05-18
    /// by real-model convert smoke test against
    /// /opt/hf2q/models/google-gemma-4-26b-a4b-it.
    #[test]
    fn detect_arch_gemma4_release_variants_real_model_2026_05_18() {
        for mt in ["gemma4", "gemma4_text"] {
            assert_eq!(
                detect_arch(&json!({ "model_type": mt })).unwrap(),
                ArchName::Gemma4,
                "model_type={mt} should resolve to Gemma4"
            );
        }
        for cls in ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"] {
            assert_eq!(
                detect_arch(&json!({ "architectures": [cls] })).unwrap(),
                ArchName::Gemma4,
                "architectures=[{cls}] should resolve to Gemma4"
            );
        }
    }

    /// model_type=qwen3_moe → ArchName::Qwen35Moe.
    #[test]
    fn detect_arch_qwen3moe() {
        assert_eq!(
            detect_arch(&json!({ "model_type": "qwen3_moe" })).unwrap(),
            ArchName::Qwen35Moe
        );
    }

    /// Codex 3b478164 review locked in: operator-released variants
    /// `qwen3_5_moe_text` and `qwen3_6_moe_text` also resolve to
    /// ArchName::Qwen35Moe.
    #[test]
    fn detect_arch_qwen35moe_release_variants_codex_3b478164() {
        // Qwen 3.5 variants with linear-attn + MTP route to
        // ArchName::Qwen35MoeFull (the qwen35moe canonical arch) per
        // the new handler at src/convert/arch/qwen35moe_full.rs.
        // The older qwen3_moe canonical (no linear-attn, no MTP)
        // remains on ArchName::Qwen35Moe.
        //
        // Note: "Qwen 3.6" is a model VERSION name; all locally-
        // available qwen3.6-* models use Qwen3_5* arch strings
        // (canonical /opt/llama.cpp/conversion/qwen.py:626 only
        // registers Qwen3_5Moe*).
        for mt in ["qwen3_5_moe", "qwen3_5_moe_text"] {
            assert_eq!(
                detect_arch(&json!({ "model_type": mt })).unwrap(),
                ArchName::Qwen35MoeFull,
                "model_type={mt} should resolve to Qwen35MoeFull"
            );
        }
        for cls in [
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
        ] {
            assert_eq!(
                detect_arch(&json!({ "architectures": [cls] })).unwrap(),
                ArchName::Qwen35MoeFull,
                "architectures=[{cls}] should resolve to Qwen35MoeFull"
            );
        }
        // Older Qwen 3 dense MoE (no linear-attn, no MTP) keeps
        // routing to ArchName::Qwen35Moe.
        assert_eq!(
            detect_arch(&json!({ "model_type": "qwen3_moe" })).unwrap(),
            ArchName::Qwen35Moe
        );
        assert_eq!(
            detect_arch(&json!({ "architectures": ["Qwen3MoeForCausalLM"] })).unwrap(),
            ArchName::Qwen35Moe
        );
    }

    /// All three qwen3_vl flavors land on Qwen3VlText.
    #[test]
    fn detect_arch_qwen3vl_flavors() {
        for mt in ["qwen3_vl", "qwen3_vl_moe", "qwen3_vl_text"] {
            assert_eq!(
                detect_arch(&json!({ "model_type": mt })).unwrap(),
                ArchName::Qwen3VlText,
                "model_type={mt}"
            );
        }
    }

    /// model_type=bert + architectures=["BertForMaskedLM"] both detect Bert.
    #[test]
    fn detect_arch_bert() {
        assert_eq!(
            detect_arch(&json!({ "model_type": "bert" })).unwrap(),
            ArchName::Bert
        );
        assert_eq!(
            detect_arch(&json!({ "architectures": ["BertForMaskedLM"] })).unwrap(),
            ArchName::Bert
        );
        assert_eq!(
            detect_arch(&json!({ "architectures": ["BertModel"] })).unwrap(),
            ArchName::Bert
        );
    }

    /// model_type=nomic_bert detects NomicBert.
    #[test]
    fn detect_arch_nomic_bert() {
        assert_eq!(
            detect_arch(&json!({ "model_type": "nomic_bert" })).unwrap(),
            ArchName::NomicBert
        );
    }

    /// model_type=minimax_m2 detects MiniMaxM2.
    #[test]
    fn detect_arch_minimax() {
        assert_eq!(
            detect_arch(&json!({ "model_type": "minimax_m2" })).unwrap(),
            ArchName::MiniMaxM2
        );
        assert_eq!(
            detect_arch(&json!({ "architectures": ["MiniMaxM2ForCausalLM"] })).unwrap(),
            ArchName::MiniMaxM2
        );
    }

    /// architectures[] fallback when model_type is absent.
    #[test]
    fn detect_arch_via_architectures_fallback() {
        assert_eq!(
            detect_arch(&json!({ "architectures": ["LlamaForCausalLM"] })).unwrap(),
            ArchName::Llama3
        );
        assert_eq!(
            detect_arch(&json!({ "architectures": ["Qwen3MoeForCausalLM"] })).unwrap(),
            ArchName::Qwen35Moe
        );
    }

    /// model_type/architectures disagreement: model_type wins. (No
    /// silent fallback per the no-loop-suppression rule — but the
    /// loader specifically uses model_type as the primary signal.)
    #[test]
    fn detect_arch_model_type_wins_over_architectures() {
        let cfg = json!({
            "model_type": "llama",
            "architectures": ["Qwen3MoeForCausalLM"]
        });
        assert_eq!(detect_arch(&cfg).unwrap(), ArchName::Llama3);
    }

    /// Unsupported arch surfaces typed error.
    #[test]
    fn detect_arch_unsupported_errors() {
        let cfg = json!({ "model_type": "mamba" });
        match detect_arch(&cfg).expect_err("must error") {
            ConvertError::UnsupportedArch { arch_name } => {
                assert_eq!(arch_name, "mamba");
            }
            other => panic!("expected UnsupportedArch, got {other:?}"),
        }
    }

    /// Missing both model_type and architectures → typed error with a
    /// diagnostic placeholder.
    #[test]
    fn detect_arch_completely_missing_errors() {
        let cfg = json!({});
        match detect_arch(&cfg).expect_err("must error") {
            ConvertError::UnsupportedArch { arch_name } => {
                assert!(arch_name.contains("missing"));
            }
            other => panic!("expected UnsupportedArch, got {other:?}"),
        }
    }

    /// HParams: num_key_value_heads defaults to num_attention_heads.
    #[test]
    fn build_hparams_defaults_kv_heads_to_head_count() {
        let cfg = json!({ "num_attention_heads": 8, "num_hidden_layers": 16 });
        let hp = build_hparams(&cfg).unwrap();
        assert_eq!(hp.n_head, 8);
        assert_eq!(hp.n_head_kv, 8);
        assert_eq!(hp.n_expert, 0);
        assert_eq!(hp.n_layer, 16);
    }

    /// HParams: num_experts (Qwen3MoE) and num_local_experts (MiniMax)
    /// both populate n_expert.
    #[test]
    fn build_hparams_picks_up_moe_expert_count() {
        let cfg_qwen = json!({
            "num_attention_heads": 32,
            "num_experts": 128,
            "num_hidden_layers": 30,
        });
        let cfg_minimax = json!({
            "num_attention_heads": 32,
            "num_local_experts": 32,
            "num_hidden_layers": 40,
        });
        assert_eq!(build_hparams(&cfg_qwen).unwrap().n_expert, 128);
        assert_eq!(build_hparams(&cfg_minimax).unwrap().n_expert, 32);
        assert_eq!(build_hparams(&cfg_qwen).unwrap().n_layer, 30);
    }

    /// HParams missing num_hidden_layers → typed error (per the canonical
    /// `init_quantize_state_counters` dependency on `hparams.n_layer`).
    #[test]
    fn build_hparams_missing_n_layer_errors() {
        let cfg = json!({ "num_attention_heads": 8 });
        match build_hparams(&cfg).expect_err("must error") {
            ConvertError::MissingHparam { key } => {
                assert_eq!(key, "num_hidden_layers");
            }
            other => panic!("expected MissingHparam, got {other:?}"),
        }
    }

    /// HParams missing num_attention_heads → typed error.
    #[test]
    fn build_hparams_missing_head_count_errors() {
        let cfg = json!({});
        match build_hparams(&cfg).expect_err("must error") {
            ConvertError::MissingHparam { key } => {
                assert_eq!(key, "num_attention_heads");
            }
            other => panic!("expected MissingHparam, got {other:?}"),
        }
    }

    // ========================================================================
    // QuantSelector parse tests (mission-spec required ~6 new tests)
    // ========================================================================

    /// "q5_k_m" → QuantSelector::Standard(MostlyQ5_K_M).
    #[test]
    fn parse_quant_selector_standard_round_trip() {
        let sel = QuantSelector::from_name("q5_k_m").expect("must parse");
        assert_eq!(sel, QuantSelector::Standard(LlamaFtype::MostlyQ5_K_M));
    }

    /// "apex-balanced" → QuantSelector::Apex(ApexTier::Balanced).
    #[test]
    fn parse_quant_selector_apex_round_trip() {
        let sel = QuantSelector::from_name("apex-balanced").expect("must parse");
        assert_eq!(sel, QuantSelector::Apex(ApexTier::Balanced));
    }

    /// "apex-i-quality" → QuantSelector::Apex(ApexTier::IQuality). Covers
    /// the I-prefix imatrix tier surface.
    #[test]
    fn parse_quant_selector_apex_i_variant() {
        let sel = QuantSelector::from_name("apex-i-quality").expect("must parse");
        assert_eq!(sel, QuantSelector::Apex(ApexTier::IQuality));
    }

    /// "apex-custom" → Err(ApexCustomRequiresTensorTypeFile).
    #[test]
    fn parse_quant_selector_apex_custom_errors() {
        use crate::convert::quant_selector::QuantSelectorError;
        let err = QuantSelector::from_name("apex-custom").expect_err("must error");
        assert!(matches!(
            err,
            QuantSelectorError::ApexCustomRequiresTensorTypeFile
        ));
    }

    /// "dwq" → Err(DwqReserved). Reserved name per ADR Decision §6.
    #[test]
    fn parse_quant_selector_dwq_reserved() {
        use crate::convert::quant_selector::QuantSelectorError;
        let err = QuantSelector::from_name("dwq").expect_err("must error");
        assert!(matches!(err, QuantSelectorError::DwqReserved));
    }

    /// "apex-nano" → Err(ApexTierOutOfScope). Mudler's experimental
    /// tiers were dropped from v1's surface.
    #[test]
    fn parse_quant_selector_apex_nano_out_of_scope() {
        use crate::convert::quant_selector::QuantSelectorError;
        let err = QuantSelector::from_name("apex-nano").expect_err("must error");
        match err {
            QuantSelectorError::ApexTierOutOfScope { tier } => assert_eq!(tier, "nano"),
            other => panic!("expected ApexTierOutOfScope, got {other:?}"),
        }
    }

    // ============================================================================
    // ADR-033 §Pi imatrix-resolution tests
    // ============================================================================
    //
    // Cover the routes through `resolve_imatrix_input`:
    //   1. `--imatrix <missing-path>`            → Imatrix(ImatrixError::Io)
    //   2. `--imatrix-corpus cdv3` + bogus hf_dir → ConvertFailed (driver fired)
    //   3. `--imatrix-corpus cdv3` + Qwen35Moe  → UnsupportedArchForDriver (Stage 3.0)
    //   3. I-tier without imatrix data          → ImatrixRequiredForITier
    //   4. Non-I tier without imatrix data      → Ok(None)
    //
    // The "happy path" (loading a valid `.imatrix.gguf`) is round-trip
    // tested in `quantize::imatrix::tests::imatrix_data_round_trip_is_byte_stable`.
    //
    // `ApexTier` is brought in at the top of the test module already.

    /// ADR-033 §Pi: an I-tier with no imatrix flags surfaces the typed
    /// `ImatrixRequiredForITier` error (no silent fallback to non-I sibling).
    /// Test-only sentinel hf_dir for paths that don't actually
    /// reach the driver (imatrix-path or no-imatrix branches). The
    /// driver path is exercised separately by
    /// `imatrix_corpus_drives_in_tree_and_errors_typed`.
    fn dummy_hf_dir() -> std::path::PathBuf {
        std::path::PathBuf::from("/tmp/imatrix-test-unused")
    }

    #[test]
    fn imatrix_required_for_i_tier_without_data() {
        let err = super::resolve_imatrix_input(
            &ApexTier::IBalanced,
            None,
            None,
            &dummy_hf_dir(),
            crate::quantize::ggml_quants::ArchName::Gemma4,
            512,
        )
        .unwrap_err();
        match err {
            ConvertError::ImatrixRequiredForITier { tier } => {
                assert_eq!(tier, "i-balanced")
            }
            other => panic!("expected ImatrixRequiredForITier, got {other:?}"),
        }
    }

    /// Non-I tier + no flags → `Ok(None)` (the convert run proceeds
    /// imatrix-less). Mini is non-I per `ApexTier::requires_imatrix`.
    #[test]
    fn no_imatrix_required_for_non_i_tiers() {
        for tier in [
            ApexTier::Quality,
            ApexTier::Balanced,
            ApexTier::Compact,
            ApexTier::Mini,
        ] {
            let res = super::resolve_imatrix_input(
                &tier,
                None,
                None,
                &dummy_hf_dir(),
                crate::quantize::ggml_quants::ArchName::Gemma4,
                512,
            )
            .unwrap();
            assert!(
                res.is_none(),
                "non-I tier {tier:?} should not require imatrix data"
            );
        }
    }

    /// **Stage 3c.2 — `--imatrix-corpus cdv3` drives the in-tree
    /// driver.** With a missing hf_dir the driver fails fast at the
    /// validate step → `ConvertFailed`. (Stage 3.0 has no end-to-end
    /// CI test on a real 26B HF model — that's operator-time per
    /// `compute_imatrix`'s doc.)
    #[test]
    fn imatrix_corpus_drives_in_tree_and_errors_typed() {
        let bogus_hf = std::path::PathBuf::from(
            "/tmp/imatrix-corpus-driver-test-nonexistent",
        );
        let err = super::resolve_imatrix_input(
            &ApexTier::IBalanced,
            None,
            Some("cdv3"),
            &bogus_hf,
            crate::quantize::ggml_quants::ArchName::Gemma4,
            512,
        )
        .unwrap_err();
        match err {
            ConvertError::Imatrix(
                crate::quantize::imatrix::ImatrixError::ConvertFailed { detail },
            ) => {
                assert!(
                    detail.contains("does not exist")
                        || detail.contains("not a directory"),
                    "detail should describe missing hf_dir, got: {detail}"
                );
            }
            other => panic!("expected ConvertFailed, got {other:?}"),
        }
    }

    /// **Stage 3c.2 — `--imatrix-corpus` on an unsupported arch
    /// surfaces `UnsupportedArchForDriver`** BEFORE attempting any
    /// convert/load. Stage 3.0 (Gemma 4) + Stage 3b.4 (Qwen 3.5/3.6
    /// MoE) are the supported driver arches; MiniMax-M2 is the
    /// canonical out-of-scope MoE used here.
    #[test]
    fn imatrix_corpus_unsupported_arch_errors_typed() {
        let err = super::resolve_imatrix_input(
            &ApexTier::IBalanced,
            None,
            Some("cdv3"),
            // /tmp always exists so the hf_dir check passes; the
            // UnsupportedArchForDriver check fires next.
            &std::path::PathBuf::from("/tmp"),
            crate::quantize::ggml_quants::ArchName::MiniMaxM2,
            512,
        )
        .unwrap_err();
        match err {
            ConvertError::Imatrix(
                crate::quantize::imatrix::ImatrixError::UnsupportedArchForDriver {
                    arch,
                    supported,
                },
            ) => {
                assert_eq!(arch, "minimax-m2");
                assert_eq!(supported, &["gemma4", "qwen35moe"]);
            }
            other => panic!("expected UnsupportedArchForDriver, got {other:?}"),
        }
    }

    /// **Stage 3b.4 SHIPPED 2026-05-22 — `--imatrix-corpus` on
    /// Qwen35Moe now passes the arch gate** and reaches the inner
    /// convert step. The hf_dir is bogus so the next error mode is
    /// `ConvertFailed`, which proves we passed the arch gate.
    #[test]
    fn imatrix_corpus_qwen35moe_passes_arch_gate() {
        let err = super::resolve_imatrix_input(
            &ApexTier::IQuality,
            None,
            Some("cdv3"),
            &std::path::PathBuf::from("/tmp/non-existent-fixture-qwen35moe-cli"),
            crate::quantize::ggml_quants::ArchName::Qwen35Moe,
            512,
        )
        .unwrap_err();
        match err {
            ConvertError::Imatrix(
                crate::quantize::imatrix::ImatrixError::UnsupportedArchForDriver { arch, .. },
            ) => panic!(
                "Stage 3b.4 regression: Qwen35Moe should pass arch gate but got \
                 UnsupportedArchForDriver(arch={arch:?})"
            ),
            // Past arch gate. Any other error variant is fine — what
            // matters is that the typed gate is now lifted.
            _ => {}
        }
    }

    /// Bad corpus selector → typed `UnknownBakedCorpus` (caught at
    /// parse time, before the in-tree driver runs).
    #[test]
    fn imatrix_corpus_unknown_name_errors_typed() {
        let err = super::resolve_imatrix_input(
            &ApexTier::IBalanced,
            None,
            Some("wikitext-9000"),
            &dummy_hf_dir(),
            crate::quantize::ggml_quants::ArchName::Gemma4,
            512,
        )
        .unwrap_err();
        match err {
            ConvertError::Imatrix(
                crate::quantize::imatrix::ImatrixError::UnknownBakedCorpus { name, .. },
            ) => assert_eq!(name, "wikitext-9000"),
            other => panic!("expected UnknownBakedCorpus, got {other:?}"),
        }
    }

    /// `--imatrix <missing-path>` errors loudly (typed I/O), not
    /// silent fallback to the corpus path or to the non-I sibling.
    #[test]
    fn imatrix_missing_file_errors_typed() {
        let bogus = std::path::PathBuf::from("/nonexistent/path/imatrix.gguf");
        let err = super::resolve_imatrix_input(
            &ApexTier::IBalanced,
            Some(bogus.as_path()),
            None,
            &dummy_hf_dir(),
            crate::quantize::ggml_quants::ArchName::Gemma4,
            512,
        )
        .unwrap_err();
        match err {
            ConvertError::Imatrix(_) => { /* OK: surfaced from loader */ }
            other => panic!("expected ConvertError::Imatrix, got {other:?}"),
        }
    }

    /// **ADR-033 §Pi closure: `--imatrix-n-ctx 0` surfaces typed
    /// `ImatrixNCtxInvalid`** — refuses to silently default. Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: when the operator
    /// passes an explicit invalid value, refuse rather than mask.
    /// Closes the deferred sub-task at ADR-033 §Pi Stage 3.
    #[test]
    fn imatrix_n_ctx_zero_errors_typed() {
        // Pick the corpus path (the only path that consults n_ctx);
        // n_ctx=0 must error BEFORE attempting any tokenization or
        // forward pass. `/tmp` is a valid dir so we'd otherwise reach
        // the unsupported-arch check on Qwen35Moe.
        let err = super::resolve_imatrix_input(
            &ApexTier::IBalanced,
            None,
            Some("cdv3"),
            &std::path::PathBuf::from("/tmp"),
            crate::quantize::ggml_quants::ArchName::Gemma4,
            0,
        )
        .unwrap_err();
        match err {
            ConvertError::ImatrixNCtxInvalid { n_ctx } => {
                assert_eq!(n_ctx, 0);
                let msg = err.to_string();
                assert!(
                    msg.contains("must be > 0"),
                    "msg should explain the constraint: {msg}",
                );
                assert!(
                    msg.contains("512"),
                    "msg should mention the default for operator hint: {msg}",
                );
            }
            other => panic!("expected ImatrixNCtxInvalid, got {other:?}"),
        }
    }

    /// **`--imatrix-n-ctx` is plumbed through to ComputeImatrixParams.**
    /// We can't run the full forward pass in unit tests (operator-time),
    /// but we CAN verify the n_ctx value reaches the driver: passing a
    /// non-default value with a bogus hf_dir reaches the same
    /// `ConvertFailed` path as the default-n_ctx test
    /// (`imatrix_corpus_drives_in_tree_and_errors_typed`), but with the
    /// validation gate consulted at the requested n_ctx. This pins
    /// the plumbing without a real forward pass.
    #[test]
    fn imatrix_n_ctx_non_default_plumbs_through() {
        let bogus_hf = std::path::PathBuf::from(
            "/tmp/imatrix-n-ctx-plumbing-test-nonexistent",
        );
        let err = super::resolve_imatrix_input(
            &ApexTier::IBalanced,
            None,
            Some("cdv3"),
            &bogus_hf,
            crate::quantize::ggml_quants::ArchName::Gemma4,
            1024,
        )
        .unwrap_err();
        // The error path is the same as the default-n_ctx test —
        // n_ctx=1024 doesn't bypass the hf_dir existence check.
        // The plumbing-correctness assertion is that NO panic /
        // wrong-variant fires before reaching ConvertFailed.
        match err {
            ConvertError::Imatrix(
                crate::quantize::imatrix::ImatrixError::ConvertFailed { .. },
            ) => { /* OK: n_ctx=1024 reached the driver, then failed at convert step */ }
            other => panic!(
                "expected ConvertFailed (n_ctx=1024 plumbed through), got {other:?}"
            ),
        }
    }

    /// `--imatrix <file>` (when loadable) returns `Some(ImatrixData)`,
    /// regardless of tier (non-I tiers can still consume imatrix data).
    #[test]
    fn imatrix_file_loads_for_any_tier() {
        use crate::quantize::imatrix::{write_imatrix_to_path, AccumulatorRegistry};

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut reg = AccumulatorRegistry::new();
        let acc = reg.register("blk.0.attn_q.weight", 4, 1).unwrap();
        acc.absorb_dense(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        write_imatrix_to_path(tmp.path(), &reg, &["cdv3".to_string()], 1, 512).unwrap();

        // I-tier: returns Some with 1 tensor pair.
        let data = super::resolve_imatrix_input(
            &ApexTier::IBalanced,
            Some(tmp.path()),
            None,
            &dummy_hf_dir(),
            crate::quantize::ggml_quants::ArchName::Gemma4,
            512,
        )
        .unwrap()
        .unwrap();
        assert_eq!(data.tensor_pair_count(), 1);

        // Non-I tier: also returns Some (optional imatrix is honored).
        let data = super::resolve_imatrix_input(
            &ApexTier::Balanced,
            Some(tmp.path()),
            None,
            &dummy_hf_dir(),
            crate::quantize::ggml_quants::ArchName::Gemma4,
            512,
        )
        .unwrap()
        .unwrap();
        assert_eq!(data.tensor_pair_count(), 1);
    }

    #[test]
    fn deepseek_detection_and_incomplete_expert_group_fail_closed() {
        assert_eq!(
            detect_arch(&json!({"model_type":"deepseek_v4"})).unwrap(),
            ArchName::Deepseek4
        );
        assert_eq!(
            detect_arch(&json!({"architectures":["DeepseekV4ForCausalLM"]})).unwrap(),
            ArchName::Deepseek4
        );

        use safetensors::tensor::{Dtype, TensorView};
        let dir = tempfile::tempdir().unwrap();
        let expert_bytes = vec![0_u8; 16];
        let scale_bytes = vec![127_u8];
        let expert = TensorView::new(Dtype::I8, vec![1, 16], &expert_bytes).unwrap();
        let scale = TensorView::new(Dtype::F8_E8M0, vec![1, 1], &scale_bytes).unwrap();
        let tensors = vec![
            ("layers.0.ffn.experts.0.w1.weight".to_string(), &expert),
            ("layers.0.ffn.experts.0.w1.scale".to_string(), &scale),
        ];
        std::fs::write(
            dir.path().join("model.safetensors"),
            safetensors::tensor::serialize(tensors, None).unwrap(),
        ).unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&json!({
                "model_type":"deepseek_v4", "n_routed_experts":2,
                "quantization_config":{"quant_method":"fp8", "weight_block":[128,128]}
            })).unwrap(),
        ).unwrap();
        let source = HfModelSource::open(dir.path()).unwrap();
        let err = match build_convert_plan(ArchName::Deepseek4, &source, &[]) {
            Ok(_) => panic!("one of two experts must not form a complete group"),
            Err(err) => err,
        };
        assert!(matches!(err, ConvertError::IncompleteExpertGroup {
            present_count: 1, n_experts_config: 2, ..
        }));
    }

    #[test]
    fn deepseek_tiny_official_layout_converts_to_q2_k_s_end_to_end() {
        use safetensors::tensor::{Dtype, TensorView};
        let dir = tempfile::tempdir().unwrap();
        let f16 = vec![0_u8; 32 * 256 * 2];
        let packed = vec![0x21_u8; 256 * 128];
        let scales = vec![127_u8; 256 * 8];
        let mut owned: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = vec![
            ("embed.weight".into(), Dtype::F16, vec![32, 256], f16.clone()),
            ("head.weight".into(), Dtype::F16, vec![32, 256], f16),
            ("norm.weight".into(), Dtype::F32, vec![256], vec![0; 256 * 4]),
        ];
        for expert in 0..2 {
            for proj in ["w1", "w2", "w3"] {
                owned.push((
                    format!("layers.0.ffn.experts.{expert}.{proj}.weight"),
                    Dtype::I8,
                    vec![256, 128],
                    packed.clone(),
                ));
                owned.push((
                    format!("layers.0.ffn.experts.{expert}.{proj}.scale"),
                    Dtype::U8,
                    vec![256, 8],
                    scales.clone(),
                ));
            }
        }
        let views: Vec<(String, TensorView<'_>)> = owned
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (name.clone(), TensorView::new(*dtype, shape.clone(), bytes).unwrap())
            })
            .collect();
        let refs: Vec<(String, &TensorView<'_>)> =
            views.iter().map(|(name, view)| (name.clone(), view)).collect();
        std::fs::write(
            dir.path().join("model.safetensors"),
            safetensors::tensor::serialize(refs, None).unwrap(),
        ).unwrap();

        let cfg = json!({
            "_name_or_path":"deepseek-ai/DeepSeek-V4-Flash-0731", "model_type":"deepseek_v4",
            "architectures":["DeepseekV4ForCausalLM"], "hidden_size":256,
            "num_hidden_layers":1, "num_attention_heads":4, "num_key_value_heads":1, "head_dim":64,
            "max_position_embeddings":1024, "rms_norm_eps":1e-6, "vocab_size":32,
            "n_routed_experts":2, "num_experts_per_tok":1, "n_shared_experts":1,
            "moe_intermediate_size":256, "routed_scaling_factor":1.5,
            "norm_topk_prob":true, "scoring_func":"sqrtsoftplus", "swiglu_limit":10.0,
            "qk_rope_head_dim":64, "q_lora_rank":64, "sliding_window":128,
            "index_n_heads":4, "index_head_dim":32, "index_topk":16,
            "o_groups":2, "o_lora_rank":64, "compress_ratios":[0],
            "compress_rope_theta":160000.0, "hc_mult":4, "hc_sinkhorn_iters":20,
            "hc_eps":1e-6, "num_hash_layers":0,
            "quantization_config":{"quant_method":"fp8", "weight_block":[128,128]}
        });
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();

        let mut vocab = serde_json::Map::new();
        for i in 0..28 {
            vocab.insert(format!("tok{i}"), json!(i));
        }
        std::fs::write(
            dir.path().join("tokenizer.json"),
            serde_json::to_vec(&json!({
                "model":{"type":"BPE", "byte_fallback":true, "vocab":vocab, "merges":[]},
                "added_tokens":[
                    {"id":28,"content":"<bos>","special":true},
                    {"id":29,"content":"<eos>","special":true},
                    {"id":30,"content":"<pad>","special":true},
                    {"id":31,"content":"<unk>","special":true}
                ]
            })).unwrap(),
        ).unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            serde_json::to_vec(&json!({
                "bos_token":"<bos>", "eos_token":"<eos>", "pad_token":"<pad>",
                "unk_token":"<unk>", "add_bos_token":true, "add_eos_token":false
            })).unwrap(),
        ).unwrap();

        let output = dir.path().join("tiny-q2-k-s.gguf");
        run_convert(ConvertArgs {
            hf_dir: dir.path().to_path_buf(),
            selector: QuantSelector::Standard(LlamaFtype::MostlyQ2_K_S),
            output: output.clone(),
            imatrix: None,
            imatrix_corpus: None,
            imatrix_out: None,
            imatrix_n_ctx: None,
            mmproj: false,
            remote_source: Some(RemoteConversionSource {
                repo: "deepseek-ai/DeepSeek-V4-Flash-0731".into(),
                revision: "a".repeat(40),
                source_sha256: "b".repeat(64),
                files: Vec::new(),
            }),
        }).unwrap();
        let bytes = std::fs::read(&output).unwrap();
        assert_eq!(&bytes[..4], b"GGUF");
        assert!(bytes.windows(b"blk.0.ffn_gate_exps.weight".len())
            .any(|w| w == b"blk.0.ffn_gate_exps.weight"));
        assert!(bytes.len() > 32 * 1024);
        let producer = concat!("hf2q ", env!("CARGO_PKG_VERSION")).as_bytes();
        assert!(bytes.windows(producer.len()).any(|window| window == producer));
        let expected_source_sha = "b".repeat(64);
        assert!(bytes
            .windows(expected_source_sha.len())
            .any(|window| window == expected_source_sha.as_bytes()));
        let receipt: crate::convert::receipt::ConversionReceipt = serde_json::from_slice(
            &std::fs::read(crate::convert::receipt::receipt_path(&output)).unwrap(),
        )
        .unwrap();
        assert_eq!(receipt.source.revision, "a".repeat(40));
        assert_eq!(receipt.quant_selector, "q2_k_s");
        assert_eq!(receipt.output.size, bytes.len() as u64);
    }
}
