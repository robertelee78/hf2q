//! `hf2q convert-v2 <hf-dir> --quant <name> -o <out.gguf>` driver.
//!
//! First operator-facing entry point for the new ADR-033 convert
//! pipeline. Composes [`HfModelSource::open`] → per-arch
//! `map_tensor_name` + `build_metadata` → [`ConvertOrchestrator`] into a
//! single end-to-end run. Streaming throughout: the source reader
//! mmaps each safetensors shard and yields F32 tensors lazily; the
//! orchestrator quantizes + writes one tensor at a time. Per ADR-033
//! §"Open Issues / Real-Model Findings" 2026-05-18, this fixed the
//! 4× SIGKILL-137 on a 26B-param real-model convert.
//!
//! Per ADR-033 §P0-§P3: this driver does NOT introduce any new
//! quantization or write logic — every byte emitted comes from the
//! orchestrator. Per [[feedback-no-loop-suppression-2026-05-17]]: an
//! unsupported arch / unmapped tensor / missing expert surfaces as a
//! typed [`ConvertV2Error`]; the orchestrator already rejects shape
//! misalignments at the policy/quantizer layer.
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: no migration shims,
//! no `--quant` aliases for legacy names — `LlamaFtype::from_name` is
//! the single source of truth.
//!
//! The legacy two-pass pipeline at `src/main.rs::cmd_convert` is
//! intentionally untouched; this lives alongside it until P6 deletes the
//! old path.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use crate::backends::gguf::types::MetaValue;
use crate::convert::arch::{
    bert, gemma4, gemma4_mmproj, llama3, minimax_m2, nomic_bert, qwen35moe, qwen3vl_text,
};
use crate::convert::arch::gemma4::MappedTensor as Gemma4Mapped;
use crate::convert::arch::minimax_m2::{ExpertRole, MappedTensor as MiniMaxMapped};
use crate::convert::arch::qwen35moe::{ExpertKind, MappedTensor as QwenMapped};
use crate::convert::quant_selector::{approximate_for_apex, QuantSelector};
use crate::convert::orchestrator::PlanEntry;
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

// ============================================================================
// Public API
// ============================================================================

/// Arguments for [`run_convert_v2`]. Mirrors the
/// `hf2q convert-v2 <hf-dir> --quant <name> -o <out.gguf>` CLI surface
/// but is constructible directly from integration tests (the `--quant`
/// string is already resolved to a [`QuantSelector`] here).
///
/// `selector` is the unified `--quant <name>` parse result — either a
/// standard llama.cpp ftype, an Apex algorithmic tier, or (out of v1
/// scope) an `apex-custom` tensor-type-file path. See
/// [`crate::convert::quant_selector::QuantSelector`].
#[derive(Debug, Clone)]
pub struct ConvertV2Args {
    /// HuggingFace model directory — must contain `config.json` plus
    /// either `model.safetensors` or `model.safetensors.index.json` +
    /// shards.
    pub hf_dir: PathBuf,
    /// Resolved `--quant <name>` selector. Standard ftypes route through
    /// `StandardPolicy`; Apex tiers route through `ApexPolicy`.
    pub selector: QuantSelector,
    /// Destination GGUF path. Existing files are overwritten.
    pub output: PathBuf,
}

/// Errors raised by [`run_convert_v2`]. Wraps the typed errors from the
/// source reader + orchestrator + filesystem layers, and adds two
/// driver-only variants:
///
/// - [`ConvertV2Error::UnsupportedArch`] — `config.json::model_type` /
///   `architectures` did not match any of the 8 supported arches.
/// - [`ConvertV2Error::UnmappedTensor`] — a safetensors tensor name was
///   not recognized by the selected arch's `map_tensor_name`.
///
/// Per [[feedback-no-loop-suppression-2026-05-17]] both surface as
/// typed errors — never silently skipped.
#[derive(Debug)]
pub enum ConvertV2Error {
    /// `HfModelSource::open` / `iter_tensors` / `materialize_tensor`
    /// failure (missing config, malformed safetensors, unsupported
    /// source dtype, missing FP8 sibling-scale, etc.).
    Source(SourceError),
    /// `ConvertOrchestrator::write` failure (policy reject, quantizer
    /// reject, writer I/O failure).
    Orchestrator(OrchestratorError),
    /// Filesystem I/O failed (e.g. could not create the output file).
    Io(std::io::Error),
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
}

impl std::fmt::Display for ConvertV2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertV2Error::Source(e) => write!(f, "convert-v2/source: {e}"),
            ConvertV2Error::Orchestrator(e) => write!(f, "convert-v2/orchestrator: {e}"),
            ConvertV2Error::Io(e) => write!(f, "convert-v2/io: {e}"),
            ConvertV2Error::UnsupportedArch { arch_name } => {
                write!(
                    f,
                    "convert-v2: unsupported architecture `{arch_name}` \
                     (supported: llama, gemma3, bert, nomic_bert, qwen3_moe, \
                     qwen3_vl, minimax_m2)"
                )
            }
            ConvertV2Error::UnmappedTensor { hf_name, arch } => write!(
                f,
                "convert-v2: tensor `{hf_name}` not recognized by `{arch}` mapper"
            ),
            ConvertV2Error::IncompleteExpertGroup {
                gguf_name,
                layer,
                kind_label,
                present_count,
                n_experts_config,
            } => write!(
                f,
                "convert-v2: expert group `{gguf_name}` (layer={layer}, kind={kind_label}) \
                 only saw {present_count}/{n_experts_config} experts"
            ),
            ConvertV2Error::DuplicateExpertIndex {
                gguf_name,
                layer,
                kind_label,
                expert_index,
            } => write!(
                f,
                "convert-v2: duplicate expert index {expert_index} for \
                 `{gguf_name}` (layer={layer}, kind={kind_label})"
            ),
            ConvertV2Error::MissingHparam { key } => write!(
                f,
                "convert-v2: config.json is missing required hparam `{key}`"
            ),
            ConvertV2Error::ApexMissingLayerCount => write!(
                f,
                "convert-v2: --quant apex-<tier> requires `num_hidden_layers` in config.json"
            ),
            ConvertV2Error::ApexCustomOutOfScope { path } => write!(
                f,
                "convert-v2: --quant apex-custom --tensor-type-file `{}` is reserved \
                 (out of v1 scope)",
                path.display()
            ),
            ConvertV2Error::Apex(e) => write!(f, "convert-v2/apex: {e}"),
            ConvertV2Error::Tokenizer(e) => write!(f, "convert-v2/tokenizer: {e}"),
        }
    }
}

impl std::error::Error for ConvertV2Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConvertV2Error::Source(e) => Some(e),
            ConvertV2Error::Orchestrator(e) => Some(e),
            ConvertV2Error::Io(e) => Some(e),
            ConvertV2Error::Apex(e) => Some(e),
            ConvertV2Error::Tokenizer(e) => Some(e),
            _ => None,
        }
    }
}

impl From<SourceError> for ConvertV2Error {
    fn from(e: SourceError) -> Self {
        ConvertV2Error::Source(e)
    }
}

impl From<OrchestratorError> for ConvertV2Error {
    fn from(e: OrchestratorError) -> Self {
        ConvertV2Error::Orchestrator(e)
    }
}

impl From<std::io::Error> for ConvertV2Error {
    fn from(e: std::io::Error) -> Self {
        ConvertV2Error::Io(e)
    }
}

impl From<ApexError> for ConvertV2Error {
    fn from(e: ApexError) -> Self {
        ConvertV2Error::Apex(e)
    }
}

impl From<TokenizerError> for ConvertV2Error {
    fn from(e: TokenizerError) -> Self {
        ConvertV2Error::Tokenizer(e)
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
///    - `None` → typed [`ConvertV2Error::UnmappedTensor`].
/// 5. Drain the expert accumulator: assert every group has exactly
///    `n_experts` slices, sort by `expert_index`, flatten into the
///    fused 3-D shape `[in, out, n_experts]`, push to the orchestrator.
/// 6. Emit metadata via the arch's `build_metadata`.
/// 7. [`ConvertOrchestrator::write`] → BufWriter over the output file.
pub fn run_convert_v2(args: ConvertV2Args) -> Result<(), ConvertV2Error> {
    // ----- 1. Open source (mmap, metadata-only) ---------------------------
    // Per ADR-033 §"Open Issues / Real-Model Findings" 2026-05-18: the
    // source reader does NOT load every safetensors shard into RAM. It
    // mmaps each shard and records a flat tensor index. Payload bytes are
    // read one tensor at a time in the streaming stage below.
    let src = HfModelSource::open(&args.hf_dir)?;

    // ----- 2. Detect arch ---------------------------------------------------
    let arch = detect_arch(&src.config)?;

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
                .ok_or(ConvertV2Error::ApexMissingLayerCount)?;
            let n_expert = hparams.n_expert;
            let mut apex_policy = ApexPolicy::new(*tier, arch, n_layers, n_expert)?;
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
            let orch = ConvertOrchestrator::new_with_apex(ftype, arch, hparams, apex_policy);
            (orch, ftype)
        }
        QuantSelector::ApexCustom(path) => {
            return Err(ConvertV2Error::ApexCustomOutOfScope {
                path: path.clone(),
            });
        }
    };

    // ----- 4. Emit metadata (orchestrator buffers it for begin_write) ----
    let ftype_u32 = ftype_for_metadata as u32;
    for (k, v) in build_metadata_for_arch(arch, &src.config, ftype_u32) {
        orch.add_metadata(k, v);
    }

    // ----- 4b. Emit tokenizer metadata --------------------------------
    // llama.cpp's vocab loader rejects any GGUF that is missing
    // `tokenizer.ggml.model` — failure mode reported 2026-05-18 by the
    // real-model convert-v2 smoke test on
    // /opt/hf2q/models/google-gemma-4-26b-a4b-it. Per
    // [[feedback-no-loop-suppression-2026-05-17]] we surface every
    // tokenizer-parse failure as a typed `ConvertV2Error::Tokenizer`
    // variant rather than skipping silently — that exact silent skip
    // is what produced the bug.
    for (k, v) in build_tokenizer_metadata(&args.hf_dir, arch)? {
        orch.add_metadata(k, v);
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
    let f = File::create(&args.output)?;
    let bw = BufWriter::new(f);
    let mut sw = orch.begin_write(bw)?;

    // 5c. Stream every tensor's data in plan order. MoE fusion happens
    // inline: each Fused step loads N expert slices in expert_index
    // order, concatenates their F32 buffers, and pushes the fused
    // payload to the writer. Per-call peak memory is bounded by the
    // single largest tensor (Direct) or the largest fused group (Fused).
    for (idx, step) in plan.steps.iter().enumerate() {
        let data: Vec<f32> = step.materialize(&src, &synthesized)?;
        sw.stream_tensor(idx, &data)?;
        // `data` drops here — the next iteration's allocation reuses
        // the freed pages.
    }

    // 5d. Finalize — seek-back to fill tensor offsets, flush.
    sw.finalize()?;
    Ok(())
}

/// Extract `num_hidden_layers` from the HF config. Required by
/// `ApexPolicy::new` for the per-layer EDGE/NEAR/MID gradient. Returns
/// `None` if the field is missing or non-positive — surfaces as
/// [`ConvertV2Error::ApexMissingLayerCount`] at the caller.
fn config_n_layers(config: &serde_json::Value) -> Option<u32> {
    effective_config(config)
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .filter(|&x| x > 0)
        .map(|x| x as u32)
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
/// other arch surfaces as [`ConvertV2Error::UnsupportedArch`].
fn detect_arch(config: &serde_json::Value) -> Result<ArchName, ConvertV2Error> {
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
            // qwen3_moe (canonical) + qwen3_5_moe_text and qwen3_6_moe_text
            // variants operator's models use (per codex 3b478164 review).
            "qwen3_moe" | "qwen3_5_moe_text" | "qwen3_6_moe_text" => return Ok(ArchName::Qwen35Moe),
            "qwen3_vl" | "qwen3_vl_moe" | "qwen3_vl_text" => return Ok(ArchName::Qwen3VlText),
            "minimax_m2" => return Ok(ArchName::MiniMaxM2),
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
            // Qwen3MoeForCausalLM (canonical) + Qwen3_5MoeForCausalLM /
            // Qwen3_6MoeForCausalLM operator-released variants (per codex
            // 3b478164 review).
            "Qwen3MoeForCausalLM" | "Qwen3_5MoeForCausalLM" | "Qwen3_6MoeForCausalLM" => {
                return Ok(ArchName::Qwen35Moe);
            }
            "Qwen3VLForConditionalGeneration"
            | "Qwen3VLMoeForConditionalGeneration"
            | "Qwen3VLTextForCausalLM" => {
                return Ok(ArchName::Qwen3VlText);
            }
            "MiniMaxM2ForCausalLM" => return Ok(ArchName::MiniMaxM2),
            _ => {}
        }
    }

    // Nothing matched — typed error per the no-fallback rule. We carry
    // the most-specific name we observed for diagnostics.
    let observed = model_type
        .map(|s| s.to_string())
        .or_else(|| architectures.first().map(|s| s.to_string()))
        .unwrap_or_else(|| "<missing model_type and architectures>".into());
    Err(ConvertV2Error::UnsupportedArch {
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
fn build_hparams(config: &serde_json::Value) -> Result<HParams, ConvertV2Error> {
    let config = effective_config(config);
    let n_head = config
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .ok_or(ConvertV2Error::MissingHparam {
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
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(0);

    Ok(HParams {
        n_expert,
        n_head,
        n_head_kv,
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
) -> Vec<(String, MetaValue)> {
    // Multimodal-wrapper flatten: text-decoder hparams live in
    // config["text_config"] for Gemma 4 / Qwen3-VL omni-shape configs.
    // Per-arch mappers don't each have to handle this; we resolve at the
    // driver boundary. Surfaced 2026-05-18 by the operator's
    // google-gemma-4-26b-a4b-it real-model convert smoke test.
    let config = effective_config(config);
    match arch {
        ArchName::Llama3 => llama3::build_metadata(config, ftype),
        ArchName::Gemma4 => gemma4::build_metadata(config, ftype),
        ArchName::Gemma4Mmproj => gemma4_mmproj::build_metadata(config, ftype),
        ArchName::Bert => bert::build_metadata(config, ftype),
        ArchName::NomicBert => nomic_bert::build_metadata(config, ftype),
        ArchName::Qwen35Moe => qwen35moe::build_metadata(config, ftype),
        ArchName::Qwen3VlText => qwen3vl_text::build_metadata(config, ftype),
        ArchName::MiniMaxM2 => minimax_m2::build_metadata(config, ftype),
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
    Expert {
        gguf_name: String,
        layer: usize,
        expert_index: usize,
        kind: ExpertKind,
    },
    Drop,
    Unmapped,
}

fn map_tensor(arch: ArchName, hf_name: &str) -> MapOutcome {
    match arch {
        ArchName::Llama3 => match llama3::map_tensor_name(hf_name) {
            Some(s) => MapOutcome::Direct(s),
            None => MapOutcome::Unmapped,
        },
        ArchName::Gemma4 => lift_gemma4_mapped(gemma4::map_tensor_name(hf_name)),
        ArchName::Gemma4Mmproj => match gemma4_mmproj::map_tensor_name(hf_name) {
            Some(s) => MapOutcome::Direct(s),
            None => MapOutcome::Unmapped,
        },
        ArchName::Bert => match bert::map_tensor_name(hf_name) {
            Some(s) => MapOutcome::Direct(s),
            None => MapOutcome::Unmapped,
        },
        ArchName::NomicBert => match nomic_bert::map_tensor_name(hf_name) {
            Some(s) => MapOutcome::Direct(s),
            None => MapOutcome::Unmapped,
        },
        ArchName::Qwen3VlText => match qwen3vl_text::map_tensor_name(hf_name) {
            Some(s) => MapOutcome::Direct(s),
            None => MapOutcome::Unmapped,
        },
        ArchName::Qwen35Moe => lift_qwen_mapped(qwen35moe::map_tensor_name(hf_name)),
        ArchName::MiniMaxM2 => lift_minimax_mapped(minimax_m2::map_tensor_name(hf_name)),
        ArchName::Falcon => MapOutcome::Unmapped,
    }
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
// The MoE expert-fusion case is the only one where multiple HF tensors
// fuse into one GGUF tensor, so the streaming data pass holds at most
// one fused group's experts in memory simultaneously (e.g. Gemma 4 26B
// with 128 experts at ~10 MB each = ~1.3 GB per group — fits easily).

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
        /// GGUF-order shape (PyTorch order reversed).
        gguf_shape: Vec<usize>,
        source_dtype: SourceDtype,
        layer_index: Option<usize>,
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

    /// Pull the F32 data for this step from the source / synthesized
    /// list. For `Direct` and `Synthesized` this is one allocation; for
    /// `Fused` this is a single concatenated allocation containing
    /// every expert slice's F32 data in expert_index order.
    fn materialize(
        &self,
        src: &HfModelSource,
        synthesized: &[HfTensor],
    ) -> Result<Vec<f32>, ConvertV2Error> {
        match self {
            PlanStep::Direct { hf_name, .. } => {
                let ht = src.materialize_tensor(hf_name)?;
                Ok(ht.data)
            }
            PlanStep::Fused {
                member_hf_names,
                per_expert_py_shape,
                ..
            } => {
                let per_expert_elems: usize = per_expert_py_shape.iter().product();
                let mut fused: Vec<f32> =
                    Vec::with_capacity(per_expert_elems * member_hf_names.len());
                for name in member_hf_names {
                    let ht = src.materialize_tensor(name)?;
                    if ht.data.len() != per_expert_elems {
                        return Err(ConvertV2Error::Source(SourceError::Safetensors(format!(
                            "fused expert slice `{name}`: data len {} != expected per-expert {}",
                            ht.data.len(),
                            per_expert_elems
                        ))));
                    }
                    fused.extend_from_slice(&ht.data);
                    // `ht` drops here — only `fused` (which already
                    // copied the bytes) stays live.
                }
                Ok(fused)
            }
            PlanStep::Synthesized { synth_idx, .. } => {
                let t = synthesized.get(*synth_idx).ok_or_else(|| {
                    ConvertV2Error::Source(SourceError::Safetensors(format!(
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
/// (metadata only) and once by the streaming-write phase (one
/// `materialize` call per step, F32 bytes allocated and dropped within
/// one iteration).
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
) -> Result<ConvertPlan, ConvertV2Error> {
    let n_experts = src
        .config
        .get("num_experts")
        .or_else(|| src.config.get("num_local_experts"))
        .and_then(|v| v.as_u64())
        .map(|x| x as usize);

    let mut direct_steps: Vec<PlanStep> = Vec::new();
    let mut moe_accum: HashMap<(usize, ExpertKindKey), MoePlanGroup> = HashMap::new();

    for meta in src.tensor_metas() {
        match map_tensor(arch, &meta.name) {
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
                });
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
                    return Err(ConvertV2Error::DuplicateExpertIndex {
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
                    target: "convert_v2",
                    arch = arch.name(),
                    tensor = %meta.name,
                    "convert-v2: explicit drop per arch mapper"
                );
            }
            MapOutcome::Unmapped => {
                return Err(ConvertV2Error::UnmappedTensor {
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
            return Err(ConvertV2Error::IncompleteExpertGroup {
                gguf_name,
                layer,
                kind_label: expert_kind_label(kind),
                present_count: members.len(),
                n_experts_config: 0,
            });
        }
        if members.len() != expected_n_experts {
            return Err(ConvertV2Error::IncompleteExpertGroup {
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
                return Err(ConvertV2Error::IncompleteExpertGroup {
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
        match map_tensor(arch, &t.name) {
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
                    target: "convert_v2",
                    arch = arch.name(),
                    tensor = %t.name,
                    "convert-v2: synthesized tensor explicit-drop per arch mapper"
                );
            }
            MapOutcome::Expert { .. } => {
                // Synthesized MoE experts are not currently produced by
                // any arch. If a future arch needs them, route through
                // the same fusion accumulator above. For now this is a
                // hard error rather than a silent skip.
                return Err(ConvertV2Error::UnmappedTensor {
                    hf_name: t.name.clone(),
                    arch: arch.name().to_string(),
                });
            }
            MapOutcome::Unmapped => {
                return Err(ConvertV2Error::UnmappedTensor {
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
    Ok(ConvertPlan { steps })
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
        for mt in ["qwen3_5_moe_text", "qwen3_6_moe_text"] {
            assert_eq!(
                detect_arch(&json!({ "model_type": mt })).unwrap(),
                ArchName::Qwen35Moe,
                "model_type={mt} should resolve to Qwen35Moe"
            );
        }
        for cls in ["Qwen3_5MoeForCausalLM", "Qwen3_6MoeForCausalLM"] {
            assert_eq!(
                detect_arch(&json!({ "architectures": [cls] })).unwrap(),
                ArchName::Qwen35Moe,
                "architectures=[{cls}] should resolve to Qwen35Moe"
            );
        }
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
            ConvertV2Error::UnsupportedArch { arch_name } => {
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
            ConvertV2Error::UnsupportedArch { arch_name } => {
                assert!(arch_name.contains("missing"));
            }
            other => panic!("expected UnsupportedArch, got {other:?}"),
        }
    }

    /// HParams: num_key_value_heads defaults to num_attention_heads.
    #[test]
    fn build_hparams_defaults_kv_heads_to_head_count() {
        let cfg = json!({ "num_attention_heads": 8 });
        let hp = build_hparams(&cfg).unwrap();
        assert_eq!(hp.n_head, 8);
        assert_eq!(hp.n_head_kv, 8);
        assert_eq!(hp.n_expert, 0);
    }

    /// HParams: num_experts (Qwen3MoE) and num_local_experts (MiniMax)
    /// both populate n_expert.
    #[test]
    fn build_hparams_picks_up_moe_expert_count() {
        let cfg_qwen = json!({ "num_attention_heads": 32, "num_experts": 128 });
        let cfg_minimax =
            json!({ "num_attention_heads": 32, "num_local_experts": 32 });
        assert_eq!(build_hparams(&cfg_qwen).unwrap().n_expert, 128);
        assert_eq!(build_hparams(&cfg_minimax).unwrap().n_expert, 32);
    }

    /// HParams missing num_attention_heads → typed error.
    #[test]
    fn build_hparams_missing_head_count_errors() {
        let cfg = json!({});
        match build_hparams(&cfg).expect_err("must error") {
            ConvertV2Error::MissingHparam { key } => {
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
}
