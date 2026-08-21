//! `ConvertOrchestrator` — end-to-end driver wiring the new
//! `StandardPolicy` → `GgmlQuantizer` → `GgufWriter` pipeline.
//!
//! Per ADR-033 §P3 + the real-model OOM finding 2026-05-18: the
//! orchestrator was refactored from a buffered API (collect every
//! tensor's F32 payload into `Vec<StagedTensor>` before quantize +
//! collect every quantized payload into `Vec<Prepared>` before write)
//! into a **streaming** API that quantizes and writes one tensor at a
//! time. Peak resident set during convert is now bounded by the largest
//! single tensor's F32 working buffer + its quantized payload, instead
//! of the whole model. See ADR-033 §"Open Issues / Real-Model Findings"
//! for the full triage.
//!
//! Lifecycle (single-shot):
//!
//! 1. [`ConvertOrchestrator::new`] / [`ConvertOrchestrator::new_with_apex`]
//!    — pin to one `ftype` / `arch` / shape.
//! 2. [`add_metadata`] — stage GGUF KV pairs (cheap, metadata-only).
//! 3. [`plan_tensors`] — provide the FULL list of tensors as
//!    [`PlanEntry`] (name + shape + source_dtype + layer_index — no
//!    payload bytes). The orchestrator runs the policy pre-pass + per-
//!    tensor `target_for` decisions, recording each tensor's ggml_type
//!    and payload size in a compact `Vec<PlannedTensor>`.
//! 4. [`begin_write`] — emit the GGUF header, every KV, every
//!    tensor-info reservation, and pad-to-alignment. Returns a
//!    [`StreamingWriter`] handle that owns the underlying sink.
//! 5. [`StreamingWriter::stream_tensor`] — one call per ordinary tensor,
//!    or `begin_tensor_chunks` → repeated row-aligned
//!    `stream_tensor_chunk` → `finish_tensor_chunks` for a fused tensor.
//!    Chunk mode bounds working memory to one expert/row batch.
//! 6. [`StreamingWriter::finalize`] — seek-back to fill tensor offsets,
//!    flush, return the underlying writer.
//!
//! No silent F16 demotion outside the vision/audio gate — any other
//! quantization / shape failure surfaces as [`OrchestratorError`].

use std::collections::HashMap;
use std::io::{Seek, Write};

use half::{bf16, f16};
use sha2::{Digest, Sha256};

use crate::backends::gguf::types::MetaValue;
use crate::backends::gguf::writer::{GgufWriter, WriterError};
use crate::core::provenance::tensor_execution::LogicalF32Hasher;
use crate::quantize::ggml_quants::apex::{ApexError, ApexPolicy};
use crate::quantize::ggml_quants::quantizer::Quantizer;
use crate::quantize::ggml_quants::standard_policy::{
    HParams, LlmType, QsState, StandardPolicy, TensorCategory, tensor_type_fallback,
};
use crate::quantize::ggml_quants::{
    ArchName, Deepseek4AgenticQ2Policy, GgmlType, GgufFtype, QuantizeError, SourceDtype,
    TensorRef, is_audio_tensor_pattern, is_vision_tensor_pattern, quantizer_for,
};

/// Errors raised by [`ConvertOrchestrator::plan_tensors`] /
/// [`ConvertOrchestrator::begin_write`] /
/// [`StreamingWriter::stream_tensor`] / [`StreamingWriter::finalize`].
/// Wraps the typed errors from the policy / quantizer / writer layers —
/// no silent demotion paths exist anywhere inside the orchestrator.
#[derive(Debug)]
pub enum OrchestratorError {
    /// `StandardPolicy::target_for` or `GgmlQuantizer::quantize`
    /// rejected a tensor (shape misalignment, no Quantizer impl,
    /// etc.). Propagated unmodified per the no-fallback rule.
    Quantize(QuantizeError),

    /// `ApexPolicy::target_for` rejected a tensor (unsupported arch,
    /// dense model, missing layer index, etc.). Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: surfaced as a typed
    /// error, never silently demoted to F16 or a dense-policy fallback.
    Apex(ApexError),

    /// Underlying `GgufWriter` failure (I/O, payload-size mismatch,
    /// duplicate / missing tensor payload). Propagated unmodified.
    Writer(WriterError),

    /// Caller violated the streaming protocol — e.g. called
    /// `stream_tensor` with an out-of-bounds plan index, or in the
    /// wrong plan order, or with F32 data whose `len()` does not match
    /// the plan's `shape.iter().product()`. Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: hard error, never
    /// silent skip.
    StreamProtocol(String),

    /// ADR-033 §P4b apply-time imatrix mismatch. Surfaces when the
    /// attached imatrix has an entry for the tensor being quantized,
    /// but the recorded `n_per_row` doesn't match the model's. Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: hard error rather
    /// than a silent downgrade to the no-imatrix path.
    Imatrix(crate::quantize::imatrix::ImatrixError),
}

impl std::fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrchestratorError::Quantize(e) => write!(f, "convert/quantize: {e}"),
            OrchestratorError::Apex(e) => write!(f, "convert/apex: {e}"),
            OrchestratorError::Writer(e) => write!(f, "convert/writer: {e}"),
            OrchestratorError::StreamProtocol(s) => write!(f, "convert/stream-protocol: {s}"),
            OrchestratorError::Imatrix(e) => write!(f, "convert/imatrix: {e}"),
        }
    }
}

impl std::error::Error for OrchestratorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OrchestratorError::Quantize(e) => Some(e),
            OrchestratorError::Apex(e) => Some(e),
            OrchestratorError::Writer(e) => Some(e),
            OrchestratorError::StreamProtocol(_) => None,
            OrchestratorError::Imatrix(e) => Some(e),
        }
    }
}

impl From<crate::quantize::imatrix::ImatrixError> for OrchestratorError {
    fn from(e: crate::quantize::imatrix::ImatrixError) -> Self {
        OrchestratorError::Imatrix(e)
    }
}

impl From<QuantizeError> for OrchestratorError {
    fn from(e: QuantizeError) -> Self {
        OrchestratorError::Quantize(e)
    }
}

impl From<ApexError> for OrchestratorError {
    fn from(e: ApexError) -> Self {
        OrchestratorError::Apex(e)
    }
}

impl From<WriterError> for OrchestratorError {
    fn from(e: WriterError) -> Self {
        OrchestratorError::Writer(e)
    }
}

/// One tensor in the convert plan — name + GGUF-order shape +
/// `source_dtype` + optional layer index. **No payload bytes.**
///
/// `shape` is in GGUF order (innermost dim first; see the
/// [`GgufWriter::reserve_tensor_info`] doc). For a PyTorch-shape weight
/// `[out_dim, in_dim]`, callers reverse to `[in_dim, out_dim]` once at
/// the safetensors → orchestrator boundary; the orchestrator does NOT
/// re-reverse internally. Per ADR-033 §P2 codex-0d28ae3f review.
#[derive(Debug, Clone)]
pub struct PlanEntry {
    pub name: String,
    /// GGUF-order shape (innermost-first). `shape[0]` is `n_per_row`.
    pub shape: Vec<usize>,
    pub source_dtype: SourceDtype,
    pub layer_index: Option<usize>,
}

/// Internal representation of one tensor after the policy has decided
/// its ggml_type. Carries the GGUF-order dims + ggml_type + the expected
/// f32 element count for stream-time validation. **No payload bytes.**
#[derive(Debug, Clone)]
struct PlannedTensor {
    name: String,
    dims_gguf: Vec<u64>,
    ggml_type: GgmlType,
    /// `shape.iter().product()` — used to validate the F32 buffer the
    /// caller hands to `stream_tensor` matches the plan.
    expected_numel: usize,
    /// Innermost dim — `quantizer.quantize(..., n_per_row, ..)` consumes
    /// this. Stored to avoid recomputing from dims_gguf at stream time.
    n_per_row: usize,
}

fn encode_planned_tensor_payload(
    planned: &PlannedTensor,
    data: &[f32],
    imatrix: Option<&[f32]>,
) -> Result<(Vec<u8>, Option<Vec<f32>>), OrchestratorError> {
    Ok(match planned.ggml_type {
        GgmlType::F16 => {
            let mut payload = Vec::with_capacity(data.len() * 2);
            for &value in data {
                payload.extend_from_slice(&f16::from_f32(value).to_le_bytes());
            }
            (payload, None)
        }
        GgmlType::BF16 => {
            let mut payload = Vec::with_capacity(data.len() * 2);
            for &value in data {
                payload.extend_from_slice(&bf16::from_f32(value).to_le_bytes());
            }
            (payload, None)
        }
        GgmlType::F32 => {
            let mut payload = Vec::with_capacity(data.len() * 4);
            for &value in data {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            (payload, None)
        }
        GgmlType::I32 => {
            let mut payload = Vec::with_capacity(data.len() * 4);
            for &value in data {
                if !value.is_finite()
                    || value.fract() != 0.0
                    || value < i32::MIN as f32
                    || value > i32::MAX as f32
                {
                    return Err(OrchestratorError::StreamProtocol(format!(
                        "tensor `{}` contains non-I32 routing value {value}",
                        planned.name
                    )));
                }
                payload.extend_from_slice(&(value as i32).to_le_bytes());
            }
            (payload, None)
        }
        _ => {
            let quantizer = quantizer_for(planned.ggml_type)?;
            // Canonical first writes an F16 intermediate and reads it back
            // before quantizing. This helper is the single byte-authority
            // used by both the streaming writer and persisted replay.
            let f16_roundtrip: Vec<f32> = data
                .iter()
                .map(|&value| f16::from_f32(value).to_f32())
                .collect();
            let payload = quantizer.quantize(&f16_roundtrip, planned.n_per_row, imatrix)?;
            (payload, Some(f16_roundtrip))
        }
    })
}

/// One GGML storage type's contribution to a metadata-only convert plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedTypeSize {
    pub ggml_type: GgmlType,
    pub tensor_count: usize,
    pub payload_bytes: u64,
}

/// Exact tensor-payload estimate available before any source weight is
/// materialized or output file is created. `aligned_payload_bytes` includes
/// the GGUF-required 32-byte padding after every tensor; the small metadata
/// header is intentionally reported separately by the caller as overhead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedSizeSummary {
    pub tensor_count: usize,
    pub payload_bytes: u64,
    pub aligned_payload_bytes: u64,
    pub by_type: Vec<PlannedTypeSize>,
}

/// Pipeline driver for the new ADR-033 convert path. See module-level
/// docs for the lifecycle contract.
///
/// Policy selection: by default the orchestrator routes per-tensor type
/// decisions through [`StandardPolicy::target_for`] (mirroring the
/// peer's canonical per-tensor type selection). When constructed via
/// [`new_with_apex`] it routes through [`ApexPolicy::target_for`]
/// instead — used by `--quant apex-<tier>` on the convert-v2 CLI per
/// ADR-033 §"Plan" / Pa. The shape-misalignment fallback
/// ([`tensor_type_fallback`]) runs for both policies; only the
/// type-pick algorithm changes.
pub struct ConvertOrchestrator {
    ftype: GgufFtype,
    arch: ArchName,
    hparams: HParams,
    /// `Some` when an Apex tier was selected; mutually exclusive with
    /// the StandardPolicy path. The two policies cannot run in the same
    /// convert invocation.
    apex_policy: Option<ApexPolicy>,
    /// Architecture-specific overlay for the DeepSeek-V4 agent profile.
    /// Mutually exclusive with `apex_policy`; its base policy is standard
    /// `MostlyQ2_K`, so expert down projections receive Q3_K.
    deepseek4_agentic_q2: bool,
    metadata: Vec<(String, MetaValue)>,
    /// Populated by `plan_tensors`. Empty before planning, in plan order
    /// during/after planning. Drained slot-by-slot during streaming.
    planned: Vec<PlannedTensor>,
    /// ADR-033 §P4b — per-tensor row-importance vectors used by the
    /// imatrix-aware quantize codepath (Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS,
    /// ...). `None` for runs without `--imatrix <file>` (or for non-I tiers
    /// where the policy gate doesn't load one). The orchestrator threads
    /// the per-tensor slice into `quantizer.quantize(..., Some(imatrix))`
    /// at `StreamingWriter::stream_tensor`. Built once at convert init by
    /// `with_imatrix`.
    imatrix: Option<crate::quantize::imatrix::ImatrixData>,
}

impl ConvertOrchestrator {
    /// Construct an orchestrator pinned to one `ftype` / `arch` / shape.
    ///
    /// `hparams` is the per-model `n_expert` / `n_head` / `n_head_kv`
    /// snapshot consumed by [`StandardPolicy::target_for`] for the
    /// counter-walk + `n_gqa` branches. Real callers populate this
    /// from the safetensors-side config; tests pass synthetic values.
    pub fn new(ftype: GgufFtype, arch: ArchName, hparams: HParams) -> Self {
        Self {
            ftype,
            arch,
            hparams,
            apex_policy: None,
            deepseek4_agentic_q2: false,
            metadata: Vec::new(),
            planned: Vec::new(),
            imatrix: None,
        }
    }

    /// Construct an orchestrator that routes per-tensor type decisions
    /// through `apex_policy` instead of [`StandardPolicy`]. `ftype` is
    /// the closest-standard approximation for the GGUF
    /// `general.file_type` byte (see `quant_selector::approximate_for_apex`);
    /// every tensor's recorded ggml_type comes from `apex_policy`
    /// regardless.
    ///
    /// The `arch` argument MUST match `apex_policy.arch` — the convert
    /// dispatcher already does this when it builds the policy from the
    /// detected arch.
    pub fn new_with_apex(
        ftype: GgufFtype,
        arch: ArchName,
        hparams: HParams,
        apex_policy: ApexPolicy,
    ) -> Self {
        Self {
            ftype,
            arch,
            hparams,
            apex_policy: Some(apex_policy),
            deepseek4_agentic_q2: false,
            metadata: Vec::new(),
            planned: Vec::new(),
            imatrix: None,
        }
    }

    /// Construct the DeepSeek-V4 agent profile. This is deliberately a
    /// distinct constructor so other architectures' standard Q2_K output is
    /// byte-stable and cannot inherit V4-specific Q8 pins accidentally.
    pub fn new_deepseek4_agentic_q2(arch: ArchName, hparams: HParams) -> Self {
        assert_eq!(
            arch,
            ArchName::Deepseek4,
            "DeepSeek-V4 agentic quantization cannot be applied to another architecture"
        );
        Self {
            ftype: GgufFtype::MostlyQ2_K,
            arch,
            hparams,
            apex_policy: None,
            deepseek4_agentic_q2: true,
            metadata: Vec::new(),
            planned: Vec::new(),
            imatrix: None,
        }
    }

    /// ADR-033 §P4b — attach an imatrix to this convert run so the
    /// streaming writer threads per-tensor row-importance vectors into
    /// `quantizer.quantize(..., Some(imatrix))`. Idempotent setter — call
    /// once at convert init before `plan_tensors`. `None` overrides any
    /// previously-set imatrix back to the no-calibration path.
    ///
    /// Used by the convert dispatcher when `--imatrix <file>` resolves
    /// (loaded `ImatrixData`) or when `--imatrix-corpus` computes one
    /// in-tree. Without P4b's wiring (pre-2026-05-22), `--imatrix` was
    /// silently dropped at `cli_driver.rs:518` — apex-i-quality produced
    /// byte-identical output to apex-quality (SHA256 verified). See
    /// `[[project-adr033-p4b-unimplemented-2026-05-22]]`.
    pub fn with_imatrix(mut self, imatrix: Option<crate::quantize::imatrix::ImatrixData>) -> Self {
        self.imatrix = imatrix;
        self
    }

    /// Stage one GGUF metadata KV pair. Written in insertion order
    /// during [`begin_write`].
    pub fn add_metadata(&mut self, key: String, value: MetaValue) {
        self.metadata.push((key, value));
    }

    /// Plan every tensor in one batch.
    ///
    /// Runs the policy pre-pass (counts `n_attention_wv` / `n_ffn_down`
    /// / `n_ffn_gate` / `n_ffn_up` so QsState has them populated before
    /// per-tensor `target_for`), then walks `entries` in order, picking
    /// a `ggml_type` per tensor and recording it.
    ///
    /// **Metadata-only** — no F32 / payload bytes touched. After this
    /// returns, the caller can begin writing; payload bytes are then
    /// pulled in via `stream_tensor` one tensor at a time.
    ///
    /// Per [[feedback-no-loop-suppression-2026-05-17]]: every typed
    /// policy / quantizer error surfaces here, before any GGUF bytes
    /// are emitted. Clean failure mode — no partial files.
    pub fn plan_tensors(&mut self, entries: Vec<PlanEntry>) -> Result<(), OrchestratorError> {
        if !self.planned.is_empty() {
            return Err(OrchestratorError::StreamProtocol(format!(
                "plan_tensors called twice (already planned {} tensors)",
                self.planned.len()
            )));
        }

        // -----------------------------------------------------------------
        // Compute a canonical visit-order permutation for the policy walk
        // (does NOT reorder `entries` itself — the caller's `stream_tensor`
        // protocol uses input-order indices and we preserve that).
        //
        // The canonical quantizer visits tensors in GGUF storage order
        // which `convert_hf_to_gguf.py` emits as: globals → blk.0.* → blk.1.*
        // → ... → blk.<n_layer-1>.* (numeric layer, name-sorted within each).
        // hf2q's HfModelSource iterates HF safetensors v0.3 in **lexical name
        // order**: blk.0, blk.1, blk.10, blk.11, ..., blk.19, blk.2, ...
        //
        // The mismatch is benign for the WRITE side (the peer loads by name),
        // but `target_for` advances `qs.i_attention_wv` per visit and the
        // Q5_K_M attn_v branch at `standard_policy.rs:556` uses that counter
        // DIRECTLY (no `layer_info` parsing).
        // So a lexical visit order makes `use_more_bits(visit_count, n_layer)`
        // fire on the wrong layer indices — measured as 12 attn_v Q5_K↔Q6_K
        // swaps vs canonical on Gemma 4 26B Q5_K_M (docs §10.2).
        //
        // Fix: compute the canonical-order permutation, run the policy walk
        // in that order so qs counters advance canonically, then un-permute
        // `self.planned` back to input order. Result: `planned[i]` still
        // matches `entries[i]` (stream protocol intact), but the ggml_type
        // assignments byte-match canonical for every tensor.
        // -----------------------------------------------------------------
        fn canonical_sort_key(name: &str) -> (u32, u32, &str) {
            if let Some(rest) = name.strip_prefix("blk.") {
                if let Some(dot) = rest.find('.') {
                    if let Ok(n) = rest[..dot].parse::<u32>() {
                        return (1, n, name);
                    }
                }
            }
            (0, 0, name)
        }
        let mut canonical_order: Vec<usize> = (0..entries.len()).collect();
        canonical_order.sort_by(|&a, &b| {
            canonical_sort_key(&entries[a].name).cmp(&canonical_sort_key(&entries[b].name))
        });

        // -----------------------------------------------------------------
        // Pre-pass: count attn_v tensors and hardcode n_ffn_{down,gate,up}
        // to hparams.n_layer.
        //
        // Canonical quantize-state counter seeding:
        //   - attn_v counter is INCREMENTED per visited attn_v-like tensor
        //   - ffn_{down,gate,up} counters are HARDCODED to `n_layer`
        //
        // The hardcode matters for MoE arches where both `<L>.ffn_down.weight`
        // AND `<L>.ffn_down_exps.weight` classify as `TensorCategory::FfnDown`.
        // Counting tensors would double the denominator (e.g., 60 instead of
        // 30 for a Gemma 4 26B MoE with 30 layers), causing `use_more_bits`
        // to land on the wrong layer indices — measured 1.188× perplexity
        // regression vs canonical on Gemma 4 26B Q5_K_M (see
        // `docs/adr-033-real-model-findings/2026-05-19-quality-equivalence-gemma4-26b.md`
        // §8 for the diagnosis trace).
        //
        // The counters do NOT count vision/audio tensors — those skip
        // the policy entirely and don't increment any `i_*` counter.
        // -----------------------------------------------------------------
        let mut n_attention_wv: i32 = 0;
        for e in &entries {
            if is_vision_tensor_pattern(&e.name) || is_audio_tensor_pattern(&e.name) {
                continue;
            }
            if TensorCategory::classify(&e.name).is_attn_v() {
                n_attention_wv += 1;
            }
        }

        let mut qs = QsState::new(self.ftype, self.arch, LlmType::Other, self.hparams);
        qs.n_attention_wv = n_attention_wv;
        qs.n_ffn_down = self.hparams.n_layer as i32;
        qs.n_ffn_gate = self.hparams.n_layer as i32;
        qs.n_ffn_up = self.hparams.n_layer as i32;
        // Canonical seeding: has_tied_embeddings starts true and is cleared when
        // `output.weight` is observed in the model. Without this clear,
        // non-tied models (those with a real output.weight tensor)
        // incorrectly promote token_embd.weight via the Output/tied
        // branch in StandardPolicy::target_for at standard_policy.rs:411.
        // Detection: scan the plan's entries for an "output.weight" name.
        if entries.iter().any(|e| e.name == "output.weight") {
            qs.has_tied_embeddings = false;
        }

        let policy = StandardPolicy::new();

        // Per-tensor: pick ggml_type. Payload size is also recorded for
        // stream-time validation. No payload bytes consumed here.
        //
        // Policy walk runs in CANONICAL order (so qs counters advance to
        // match the peer), but results are stored in `planned[]` indexed
        // by the input position so `stream_tensor(i, data)` semantics are
        // unchanged. See top-of-function note about `canonical_order`.
        let mut planned: Vec<Option<PlannedTensor>> = (0..entries.len()).map(|_| None).collect();
        for &orig_idx in &canonical_order {
            let e = &entries[orig_idx];
            let dims_gguf: Vec<u64> = e.shape.iter().map(|&d| d as u64).collect();
            let expected_numel: usize = e.shape.iter().product();
            let n_per_row = e.shape[0];

            // MTP-layer override: canonical `convert_hf_to_gguf.py:base.py:821-851`
            // applies F32-keep for ffn_gate_inp/ffn_gate_inp_shexp via
            // `match_model_tensor_name(name, FFN_GATE_INP, bid)` which returns
            // False when `bid >= n_text_layers` (MTP layer is not in the arch's
            // per-layer FFN_GATE_INP tensor map). The F32 override therefore
            // doesn't apply on MTP layers, and the tensor falls through to F16
            // default storage (base.py:875). Mirror this here.
            let is_mtp_layer = self.hparams.n_mtp_layers > 0
                && e.layer_index
                    .map(|li| li >= (self.hparams.n_layer - self.hparams.n_mtp_layers) as usize)
                    .unwrap_or(false);
            let mtp_ffn_gate_inp_demote = is_mtp_layer
                && (e.name.contains("ffn_gate_inp.weight")
                    || e.name.contains("ffn_gate_inp_shexp.weight"));

            let ggml_type = if matches!(e.source_dtype, SourceDtype::I32 | SourceDtype::I64) {
                // Hash routing is a lookup table, not a float weight.
                GgmlType::I32
            } else if is_vision_tensor_pattern(&e.name)
                || is_audio_tensor_pattern(&e.name)
                || mtp_ffn_gate_inp_demote
            {
                // Vision / audio modality gate. Canonical's
                // MmprojModel.tensor_force_quant (base.py) returns:
                //   - F16 if ftype == MOSTLY_F16 and name in
                //     {.patch_embd.weight, .patch_merger.weight}
                //   - F32 otherwise for patch_embd/patch_merger
                //   - default False (no force) for other tensors → then
                //     n_dims<2 / substring rules apply per
                //     `tensor_allows_quantization` (llama-quant.cpp:285+)
                //
                // hf2q's rule: vision tensors that match F32-keep
                // patterns (1-D scalars, *_norm.weight, .position_embd,
                // etc.) STAY F32. The exception is .patch_embd which
                // gets F16 when the current ftype is f16 (matches
                // canonical's tensor_force_quant intent).
                // mtp_ffn_gate_inp_demote stays F16 unconditionally
                // (per canonical base.py:875).
                if mtp_ffn_gate_inp_demote {
                    GgmlType::F16
                } else if is_f32_keep_tensor(&e.name, e.shape.len())
                    && !e.name.contains(".patch_embd")
                {
                    GgmlType::F32
                } else {
                    GgmlType::F16
                }
            } else if is_f32_keep_tensor(&e.name, e.shape.len()) {
                // F32-keep gate — emit the F32 row-major payload as-is.
                // See `is_f32_keep_tensor` doc for the rule list.
                GgmlType::F32
            } else {
                let tref = TensorRef {
                    name: &e.name,
                    shape: &e.shape,
                    source_dtype: e.source_dtype,
                    arch: self.arch,
                    layer_index: e.layer_index,
                };
                let category = TensorCategory::classify(&e.name);
                // Branch on policy: ApexPolicy if `--quant apex-<tier>`,
                // else StandardPolicy. Both feed through
                // `tensor_type_fallback` for shape misalignment.
                match &self.apex_policy {
                    Some(ap) => {
                        let picked = ap.target_for(&tref)?;
                        tensor_type_fallback(picked, tref.n_per_row())?
                    }
                    None => {
                        let picked = policy.target_for(&mut qs, &tref, category)?;
                        if self.deepseek4_agentic_q2 {
                            let promoted =
                                Deepseek4AgenticQ2Policy::new().target_for(&tref, picked);
                            tensor_type_fallback(promoted, tref.n_per_row())?
                        } else {
                            picked
                        }
                    }
                }
            };

            planned[orig_idx] = Some(PlannedTensor {
                name: e.name.clone(),
                dims_gguf,
                ggml_type,
                expected_numel,
                n_per_row,
            });
        }

        // Every orig_idx in 0..entries.len() appears in canonical_order
        // exactly once (it's a permutation), so every slot is Some.
        self.planned = planned
            .into_iter()
            .map(|p| p.expect("permutation covers all indices"))
            .collect();
        Ok(())
    }

    /// Number of planned tensors. Zero before `plan_tensors` runs.
    pub fn planned_count(&self) -> usize {
        self.planned.len()
    }

    /// Re-run the exact planned storage encoder for one complete tensor and
    /// derive the same evidence that the streaming writer records. Persisted
    /// receipt verification uses this method so codec selection, the F16
    /// roundtrip, and payload bytes come from current production policy/code,
    /// never from the untrusted sidecar.
    #[allow(dead_code)] // persisted Dynamic evidence replay is staged behind D2b
    pub(crate) fn reproduce_tensor_write_evidence(
        &self,
        tensor_idx: usize,
        data: &[f32],
        relative_payload_offset: u64,
    ) -> Result<TensorWriteEvidence, OrchestratorError> {
        let planned = self.planned.get(tensor_idx).ok_or_else(|| {
            OrchestratorError::StreamProtocol(format!(
                "reproduce_tensor_write_evidence: idx {tensor_idx} out of range"
            ))
        })?;
        if data.len() != planned.expected_numel {
            return Err(OrchestratorError::StreamProtocol(format!(
                "reproduce_tensor_write_evidence: tensor `{}` data length {} != planned numel {}",
                planned.name,
                data.len(),
                planned.expected_numel
            )));
        }
        if self.imatrix.is_some() {
            return Err(OrchestratorError::StreamProtocol(
                "stored-evidence v1 replay does not admit imatrix state".into(),
            ));
        }
        let (payload, f16_roundtrip) = encode_planned_tensor_payload(planned, data, None)?;
        let mut converted_hasher = Sha256::new();
        for value in data {
            converted_hasher.update(value.to_bits().to_le_bytes());
        }
        let mut logical_shape = planned.dims_gguf.clone();
        logical_shape.reverse();
        let converted_logical_f32_sha256 =
            crate::core::provenance::tensor_execution::logical_f32_sha256(&logical_shape, data)
                .map_err(|error| OrchestratorError::StreamProtocol(error.to_string()))?;
        let (f16_roundtrip_f32_bytes_sha256, f16_roundtrip_logical_f32_sha256) =
            if let Some(roundtrip) = f16_roundtrip.as_ref() {
                let mut bytes_hasher = Sha256::new();
                for value in roundtrip {
                    bytes_hasher.update(value.to_bits().to_le_bytes());
                }
                (
                    Some(hex::encode(bytes_hasher.finalize())),
                    Some(
                        crate::core::provenance::tensor_execution::logical_f32_sha256(
                            &logical_shape,
                            roundtrip,
                        )
                        .map_err(|error| OrchestratorError::StreamProtocol(error.to_string()))?,
                    ),
                )
            } else {
                (None, None)
            };
        Ok(TensorWriteEvidence {
            plan_index: tensor_idx,
            tensor_name: planned.name.clone(),
            dims_gguf_innermost_first: planned.dims_gguf.clone(),
            ggml_type: planned.ggml_type,
            converted_f32_bytes_sha256: hex::encode(converted_hasher.finalize()),
            converted_logical_f32_sha256,
            f16_roundtrip_f32_bytes_sha256,
            f16_roundtrip_logical_f32_sha256,
            payload_sha256: hex::encode(Sha256::digest(&payload)),
            payload_bytes: u64::try_from(payload.len()).map_err(|_| {
                OrchestratorError::StreamProtocol(format!(
                    "tensor `{}` payload length is not representable",
                    planned.name
                ))
            })?,
            relative_payload_offset,
        })
    }

    /// Calculate exact payload bytes for the current metadata-only plan.
    pub fn planned_size_summary(&self) -> Result<PlannedSizeSummary, OrchestratorError> {
        let mut payload_bytes = 0u64;
        let mut aligned_payload_bytes = 0u64;
        let mut by_type: HashMap<GgmlType, (usize, u64)> = HashMap::new();

        for tensor in &self.planned {
            if tensor.n_per_row == 0 || tensor.expected_numel % tensor.n_per_row != 0 {
                return Err(OrchestratorError::StreamProtocol(format!(
                    "planned tensor `{}` has invalid numel/row shape {}/{}",
                    tensor.name, tensor.expected_numel, tensor.n_per_row
                )));
            }
            let rows = tensor.expected_numel / tensor.n_per_row;
            let bytes = rows
                .checked_mul(tensor.ggml_type.row_size(tensor.n_per_row))
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| {
                    OrchestratorError::StreamProtocol(format!(
                        "planned payload size overflow for `{}`",
                        tensor.name
                    ))
                })?;
            payload_bytes = payload_bytes.checked_add(bytes).ok_or_else(|| {
                OrchestratorError::StreamProtocol("planned payload total overflow".into())
            })?;
            let aligned = bytes
                .checked_add(31)
                .map(|value| value & !31)
                .ok_or_else(|| {
                    OrchestratorError::StreamProtocol("planned aligned payload overflow".into())
                })?;
            aligned_payload_bytes =
                aligned_payload_bytes.checked_add(aligned).ok_or_else(|| {
                    OrchestratorError::StreamProtocol("planned aligned total overflow".into())
                })?;
            let entry = by_type.entry(tensor.ggml_type).or_insert((0, 0));
            entry.0 += 1;
            entry.1 = entry.1.checked_add(bytes).ok_or_else(|| {
                OrchestratorError::StreamProtocol("planned per-type total overflow".into())
            })?;
        }

        let mut by_type: Vec<_> = by_type
            .into_iter()
            .map(
                |(ggml_type, (tensor_count, payload_bytes))| PlannedTypeSize {
                    ggml_type,
                    tensor_count,
                    payload_bytes,
                },
            )
            .collect();
        by_type.sort_by_key(|entry| entry.ggml_type.name());

        Ok(PlannedSizeSummary {
            tensor_count: self.planned.len(),
            payload_bytes,
            aligned_payload_bytes,
            by_type,
        })
    }

    /// Open the GGUF writer in streaming mode.
    ///
    /// Writes the GGUF header, every staged KV pair, every tensor-info
    /// reservation (with placeholder offsets), and pads to alignment.
    /// Returns a [`StreamingWriter`] that holds the underlying sink +
    /// the plan; callers then push one tensor's F32 data at a time via
    /// [`StreamingWriter::stream_tensor`] in plan order.
    ///
    /// Per the lifecycle contract: `plan_tensors` MUST be called first.
    /// Calling `begin_write` with zero planned tensors writes a
    /// header-only GGUF (acceptance test 4).
    pub fn begin_write<W: Write + Seek>(
        self,
        writer: W,
    ) -> Result<StreamingWriter<W>, OrchestratorError> {
        self.begin_write_internal(writer, false)
    }

    /// Evidence-producing writer. Ordinary conversion uses [`begin_write`]
    /// and pays no hashing cost; only the explicit D2b path enables these
    /// incremental provenance hashes.
    pub(crate) fn begin_write_with_evidence<W: Write + Seek>(
        self,
        writer: W,
    ) -> Result<StreamingWriter<W>, OrchestratorError> {
        self.begin_write_internal(writer, true)
    }

    fn begin_write_internal<W: Write + Seek>(
        self,
        writer: W,
        capture_evidence: bool,
    ) -> Result<StreamingWriter<W>, OrchestratorError> {
        let Self {
            metadata,
            planned,
            imatrix,
            ..
        } = self;

        let mut w = GgufWriter::new(writer);
        w.write_header(planned.len() as u64, metadata.len() as u64)?;

        for (k, v) in &metadata {
            w.write_metadata_kv(k, v)?;
        }

        // Reserve every tensor-info entry (placeholder offsets — filled
        // by `finalize` via seek-back). Per ADR-033 §P2: this is the
        // exact ordering the seek-back writer requires (all info entries
        // BEFORE pad_to_alignment BEFORE the first payload).
        for p in &planned {
            w.reserve_tensor_info(&p.name, &p.dims_gguf, p.ggml_type)?;
        }

        w.pad_to_alignment()?;

        Ok(StreamingWriter {
            writer: w,
            planned,
            next_idx: 0,
            active_chunks: None,
            completed_tensor_evidence: capture_evidence.then(Vec::new),
            imatrix,
            coverage_quantized: 0,
            coverage_with_imatrix: 0,
            coverage_missing: Vec::new(),
        })
    }
}

/// Streaming GGUF writer returned by [`ConvertOrchestrator::begin_write`].
///
/// Owns the underlying sink + the plan. Ordinary tensors may use one
/// `stream_tensor` call. Fused tensors use row-aligned chunks so peak
/// working memory is `max_chunk_elements × 4` for the caller buffer,
/// the same-sized F16-roundtrip F32 buffer, and one chunk payload.
pub struct StreamingWriter<W: Write + Seek> {
    writer: GgufWriter<W>,
    planned: Vec<PlannedTensor>,
    next_idx: usize,
    active_chunks: Option<ActiveTensorChunks>,
    completed_tensor_evidence: Option<Vec<TensorWriteEvidence>>,
    /// ADR-033 §P4b — optional imatrix data for per-tensor row-importance
    /// weighting at `quantizer.quantize`. Looked up per tensor by name in
    /// `tensor_imatrix`. None for non-i-tier convert runs.
    imatrix: Option<crate::quantize::imatrix::ImatrixData>,
    /// ADR-033 §P4b coverage tracker — accumulated across `stream_tensor`
    /// calls when an imatrix is attached. Emitted at `finalize` so the
    /// operator can audit how completely the imatrix covered the
    /// quantized tensors (a partial imatrix-collection run silently
    /// quantizes some tensors without calibration; this surfaces that).
    coverage_quantized: usize,
    coverage_with_imatrix: usize,
    coverage_missing: Vec<String>,
}

#[derive(Debug)]
struct ActiveTensorChunks {
    tensor_idx: usize,
    total_elements: usize,
    chunk_count: usize,
    max_chunk_elements: usize,
    max_input_f32_bytes: usize,
    max_f16_roundtrip_f32_bytes: usize,
    max_quantized_payload_bytes: usize,
    max_working_vec_bytes: usize,
    converted_f32_bytes_hasher: Option<Sha256>,
    converted_logical_f32_hasher: Option<LogicalF32Hasher>,
    f16_roundtrip_f32_bytes_hasher: Option<Sha256>,
    f16_roundtrip_logical_f32_hasher: Option<LogicalF32Hasher>,
    payload_hasher: Option<Sha256>,
    payload_bytes: usize,
    imatrix: Option<Vec<f32>>,
}

/// Exact byte evidence captured at the authoritative conversion/write seam.
///
/// These hashes are accumulated incrementally, preserving the streaming
/// memory bound. Raw-byte hashes cover the little-endian F32 stream, while
/// logical hashes additionally frame the exact outermost-first shape.
/// Quantized tensors bind the canonical F16-to-F32 roundtrip consumed by the
/// quantizer. The payload hash covers exactly the bytes handed to the GGUF
/// writer, before alignment padding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorWriteEvidence {
    pub(crate) plan_index: usize,
    pub(crate) tensor_name: String,
    pub(crate) dims_gguf_innermost_first: Vec<u64>,
    pub(crate) ggml_type: GgmlType,
    pub(crate) converted_f32_bytes_sha256: String,
    pub(crate) converted_logical_f32_sha256: String,
    pub(crate) f16_roundtrip_f32_bytes_sha256: Option<String>,
    pub(crate) f16_roundtrip_logical_f32_sha256: Option<String>,
    pub(crate) payload_sha256: String,
    pub(crate) payload_bytes: u64,
    pub(crate) relative_payload_offset: u64,
}

/// Observable bound receipt for one chunk-streamed tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorChunkStats {
    pub total_elements: usize,
    pub chunk_count: usize,
    pub max_chunk_elements: usize,
    pub max_input_f32_bytes: usize,
    pub max_f16_roundtrip_f32_bytes: usize,
    pub max_quantized_payload_bytes: usize,
    pub max_working_vec_bytes: usize,
}

impl<W: Write + Seek> StreamingWriter<W> {
    fn update_f32_hasher(hasher: &mut Sha256, values: &[f32]) {
        for value in values {
            hasher.update(value.to_le_bytes());
        }
    }

    /// ADR-033 §P4b — look up the per-tensor row-importance vector for
    /// `tensor_name` from the attached imatrix.
    ///
    /// Returns:
    /// - `Ok(Some(weights))` — imatrix has an entry for this tensor and
    ///   shapes match; the slice is the per-tensor importance vector
    ///   ready to feed `quantizer.quantize(..., Some(&weights))`.
    /// - `Ok(None)` — either no imatrix attached, or the tensor wasn't
    ///   intercepted during collection (legitimate gap; the quantizer
    ///   runs without calibration for this tensor). The caller may log
    ///   this as a coverage warning at finalize time.
    /// - `Err(ImatrixError::ApplyShapeMismatch)` — imatrix has an entry
    ///   for this tensor BUT the recorded `n_per_row` doesn't match the
    ///   model's. Hard error per the no-silent-fallback rule — wrong
    ///   imatrix file is worse than no imatrix because mis-calibration
    ///   biases the quantization in the wrong direction.
    ///
    /// **Aggregation policy** (matches what most quantizers consume —
    /// `quantize_row_X(src, dst, n_per_row, imatrix)` reuses the same
    /// `imatrix` pointer for every row of the tensor):
    ///
    /// - **Dense** (`n_mat == 1`): return `Accumulator.values[..n_per_row]`
    ///   as-is. The accumulator stores sum-of-squared-activations per
    ///   column; the canonical quantizer passes these raw to the K-quant kernel,
    ///   which then combines them with `sqrt(sigma2 + x_l²)` per-block.
    /// - **MoE** (`n_mat > 1`, e.g. `ffn_*_exps` for Qwen MoE): sum
    ///   per-column across all experts, divide by total token count, to
    ///   produce a single n_per_row importance vector that's then reused
    ///   per row. This loses per-expert specificity vs. the peer's
    ///   per-expert imatrix path but is the simplest correct first-cut for
    ///   the row-uniform quantize() signature; a future iter could thread
    ///   per-mat slices through if measurement justifies it.
    fn tensor_imatrix(
        &self,
        tensor_name: &str,
        n_per_row: usize,
    ) -> Result<Option<Vec<f32>>, crate::quantize::imatrix::ImatrixError> {
        let Some(data) = self.imatrix.as_ref() else {
            return Ok(None);
        };
        let Some(acc) = data.loaded.registry.get(tensor_name) else {
            return Ok(None);
        };
        if acc.n_per_row != n_per_row {
            // Hard error — wrong imatrix file.
            return Err(crate::quantize::imatrix::ImatrixError::ApplyShapeMismatch {
                tensor: tensor_name.to_string(),
                imatrix_n_per_row: acc.n_per_row,
                model_n_per_row: n_per_row,
            });
        }
        if acc.n_mat == 0 || acc.values.is_empty() {
            return Ok(None);
        }
        if acc.n_mat == 1 {
            // Dense — `values` is already shaped [n_per_row].
            return Ok(Some(acc.values[..n_per_row].to_vec()));
        }
        // MoE — `values` is laid out as `[n_per_row * n_mat]` per
        // `accumulator.rs:54-58` (mat_id-major). Aggregate per column.
        let total_counts: i64 = acc.counts.iter().copied().sum();
        if total_counts <= 0 {
            return Ok(None);
        }
        let mut agg = vec![0.0_f32; n_per_row];
        for mat in 0..acc.n_mat {
            let base = mat * n_per_row;
            for j in 0..n_per_row {
                agg[j] += acc.values[base + j];
            }
        }
        let inv_total = 1.0_f32 / (total_counts as f32);
        for v in agg.iter_mut() {
            *v *= inv_total;
        }
        Ok(Some(agg))
    }

    /// Number of tensors remaining to stream.
    pub fn tensors_remaining(&self) -> usize {
        self.planned.len() - self.next_idx
    }

    /// Total number of planned tensors (constant for the lifetime of
    /// the writer).
    pub fn planned_count(&self) -> usize {
        self.planned.len()
    }

    fn validate_next_tensor(
        &self,
        tensor_idx: usize,
        caller: &str,
    ) -> Result<(), OrchestratorError> {
        if tensor_idx >= self.planned.len() {
            return Err(OrchestratorError::StreamProtocol(format!(
                "{caller}: idx {tensor_idx} out of range (planned {})",
                self.planned.len()
            )));
        }
        if tensor_idx != self.next_idx {
            return Err(OrchestratorError::StreamProtocol(format!(
                "{caller}: out-of-order call (got idx {tensor_idx}, expected {})",
                self.next_idx
            )));
        }
        Ok(())
    }

    /// Begin streaming one planned tensor as row-aligned F32 chunks.
    /// At most one tensor may be active and tensors remain strictly in
    /// plan order.
    pub fn begin_tensor_chunks(&mut self, tensor_idx: usize) -> Result<(), OrchestratorError> {
        self.validate_next_tensor(tensor_idx, "begin_tensor_chunks")?;
        if let Some(active) = self.active_chunks.as_ref() {
            return Err(OrchestratorError::StreamProtocol(format!(
                "begin_tensor_chunks: tensor {} is already active",
                active.tensor_idx
            )));
        }

        let p = &self.planned[tensor_idx];
        let quantized = !matches!(
            p.ggml_type,
            GgmlType::F16 | GgmlType::BF16 | GgmlType::F32 | GgmlType::I32
        );
        let capture_evidence = self.completed_tensor_evidence.is_some();
        let mut logical_shape = p.dims_gguf.clone();
        logical_shape.reverse();
        let converted_logical_f32_hasher = capture_evidence
            .then(|| LogicalF32Hasher::new(&logical_shape))
            .transpose()
            .map_err(|error| {
                OrchestratorError::StreamProtocol(format!(
                    "begin_tensor_chunks: tensor `{}` has invalid logical shape: {error}",
                    p.name
                ))
            })?;
        let f16_roundtrip_logical_f32_hasher = (quantized && capture_evidence)
            .then(|| LogicalF32Hasher::new(&logical_shape))
            .transpose()
            .map_err(|error| {
                OrchestratorError::StreamProtocol(format!(
                    "begin_tensor_chunks: tensor `{}` has invalid roundtrip shape: {error}",
                    p.name
                ))
            })?;
        let imatrix = if quantized {
            self.tensor_imatrix(&p.name, p.n_per_row)?
        } else {
            None
        };
        self.writer.begin_tensor_payload(tensor_idx)?;
        if quantized {
            self.coverage_quantized += 1;
            if imatrix.is_some() {
                self.coverage_with_imatrix += 1;
            } else if self.imatrix.is_some() {
                self.coverage_missing.push(p.name.clone());
            }
        }
        self.active_chunks = Some(ActiveTensorChunks {
            tensor_idx,
            total_elements: 0,
            chunk_count: 0,
            max_chunk_elements: 0,
            max_input_f32_bytes: 0,
            max_f16_roundtrip_f32_bytes: 0,
            max_quantized_payload_bytes: 0,
            max_working_vec_bytes: 0,
            converted_f32_bytes_hasher: capture_evidence.then(Sha256::new),
            converted_logical_f32_hasher,
            f16_roundtrip_f32_bytes_hasher: (quantized && capture_evidence).then(Sha256::new),
            f16_roundtrip_logical_f32_hasher,
            payload_hasher: capture_evidence.then(Sha256::new),
            payload_bytes: 0,
            imatrix,
        });
        Ok(())
    }

    /// Quantize and write one complete-row chunk of the active tensor.
    /// The temporary F16-roundtrip and quantized buffers are bounded by
    /// this chunk, then dropped before the caller materializes the next.
    pub fn stream_tensor_chunk(
        &mut self,
        tensor_idx: usize,
        data: &[f32],
    ) -> Result<(), OrchestratorError> {
        let active = self.active_chunks.as_ref().ok_or_else(|| {
            OrchestratorError::StreamProtocol(format!(
                "stream_tensor_chunk: tensor {tensor_idx} was not begun"
            ))
        })?;
        if active.tensor_idx != tensor_idx {
            return Err(OrchestratorError::StreamProtocol(format!(
                "stream_tensor_chunk: tensor {tensor_idx} does not match active tensor {}",
                active.tensor_idx
            )));
        }
        if data.is_empty() {
            return Err(OrchestratorError::StreamProtocol(format!(
                "stream_tensor_chunk: tensor `{}` received an empty chunk",
                self.planned[tensor_idx].name
            )));
        }

        let p = &self.planned[tensor_idx];
        if data.len() % p.n_per_row != 0 {
            return Err(OrchestratorError::StreamProtocol(format!(
                "stream_tensor_chunk: tensor `{}` chunk length {} is not row-aligned to {}",
                p.name,
                data.len(),
                p.n_per_row
            )));
        }
        let total_elements = active
            .total_elements
            .checked_add(data.len())
            .ok_or_else(|| {
                OrchestratorError::StreamProtocol(format!(
                    "stream_tensor_chunk: tensor `{}` element count overflow",
                    p.name
                ))
            })?;
        if total_elements > p.expected_numel {
            return Err(OrchestratorError::StreamProtocol(format!(
                "stream_tensor_chunk: tensor `{}` cumulative length {} exceeds planned numel {}",
                p.name, total_elements, p.expected_numel
            )));
        }

        let (payload, f16_roundtrip) =
            encode_planned_tensor_payload(p, data, active.imatrix.as_deref())?;

        let input_f32_bytes = data
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                OrchestratorError::StreamProtocol(format!(
                    "stream_tensor_chunk: tensor `{}` input byte count overflow",
                    p.name
                ))
            })?;
        let quantized = !matches!(
            p.ggml_type,
            GgmlType::F16 | GgmlType::BF16 | GgmlType::F32 | GgmlType::I32
        );
        let f16_roundtrip_f32_bytes = if quantized { input_f32_bytes } else { 0 };
        let quantized_payload_bytes = if quantized { payload.len() } else { 0 };
        let working_vec_bytes = input_f32_bytes
            .checked_add(f16_roundtrip_f32_bytes)
            .and_then(|bytes| bytes.checked_add(payload.len()))
            .ok_or_else(|| {
                OrchestratorError::StreamProtocol(format!(
                    "stream_tensor_chunk: tensor `{}` working byte count overflow",
                    p.name
                ))
            })?;

        self.writer
            .write_tensor_payload_chunk(tensor_idx, &payload)?;
        let active = self
            .active_chunks
            .as_mut()
            .expect("active tensor validated above");
        if let Some(hasher) = active.converted_f32_bytes_hasher.as_mut() {
            Self::update_f32_hasher(hasher, data);
        }
        if let Some(hasher) = active.converted_logical_f32_hasher.as_mut() {
            hasher.update(data).map_err(|error| {
                OrchestratorError::StreamProtocol(format!(
                    "stream_tensor_chunk: tensor `{}` logical hash failed: {error}",
                    p.name
                ))
            })?;
        }
        if let (Some(raw_hasher), Some(logical_hasher), Some(values)) = (
            active.f16_roundtrip_f32_bytes_hasher.as_mut(),
            active.f16_roundtrip_logical_f32_hasher.as_mut(),
            f16_roundtrip.as_deref(),
        ) {
            Self::update_f32_hasher(raw_hasher, values);
            logical_hasher.update(values).map_err(|error| {
                OrchestratorError::StreamProtocol(format!(
                    "stream_tensor_chunk: tensor `{}` roundtrip logical hash failed: {error}",
                    p.name
                ))
            })?;
        }
        if let Some(hasher) = active.payload_hasher.as_mut() {
            hasher.update(&payload);
        }
        active.payload_bytes =
            active
                .payload_bytes
                .checked_add(payload.len())
                .ok_or_else(|| {
                    OrchestratorError::StreamProtocol(format!(
                        "stream_tensor_chunk: tensor `{}` payload byte count overflow",
                        p.name
                    ))
                })?;
        active.total_elements = total_elements;
        active.chunk_count += 1;
        active.max_chunk_elements = active.max_chunk_elements.max(data.len());
        active.max_input_f32_bytes = active.max_input_f32_bytes.max(input_f32_bytes);
        active.max_f16_roundtrip_f32_bytes = active
            .max_f16_roundtrip_f32_bytes
            .max(f16_roundtrip_f32_bytes);
        active.max_quantized_payload_bytes = active
            .max_quantized_payload_bytes
            .max(quantized_payload_bytes);
        active.max_working_vec_bytes = active.max_working_vec_bytes.max(working_vec_bytes);
        Ok(())
    }

    /// Finish the active tensor after validating its exact planned
    /// element and payload lengths.
    pub fn finish_tensor_chunks(
        &mut self,
        tensor_idx: usize,
    ) -> Result<TensorChunkStats, OrchestratorError> {
        let active = self.active_chunks.as_ref().ok_or_else(|| {
            OrchestratorError::StreamProtocol(format!(
                "finish_tensor_chunks: tensor {tensor_idx} was not begun"
            ))
        })?;
        if active.tensor_idx != tensor_idx {
            return Err(OrchestratorError::StreamProtocol(format!(
                "finish_tensor_chunks: tensor {tensor_idx} does not match active tensor {}",
                active.tensor_idx
            )));
        }
        let expected = self.planned[tensor_idx].expected_numel;
        if active.total_elements != expected {
            return Err(OrchestratorError::StreamProtocol(format!(
                "finish_tensor_chunks: tensor `{}` received {} elements, expected {}",
                self.planned[tensor_idx].name, active.total_elements, expected
            )));
        }

        self.writer.finish_tensor_payload(tensor_idx)?;
        let relative_payload_offset =
            self.writer
                .tensor_payload_offset(tensor_idx)
                .ok_or_else(|| {
                    OrchestratorError::StreamProtocol(format!(
                        "finish_tensor_chunks: tensor `{}` has no committed payload offset",
                        self.planned[tensor_idx].name
                    ))
                })?;
        let mut active = self.active_chunks.take().expect("validated active tensor");
        let planned = &self.planned[tensor_idx];
        let payload_bytes = u64::try_from(active.payload_bytes).map_err(|_| {
            OrchestratorError::StreamProtocol(format!(
                "finish_tensor_chunks: tensor `{}` payload byte count is not representable",
                planned.name
            ))
        })?;
        if let Some(completed) = self.completed_tensor_evidence.as_mut() {
            let converted_raw = active.converted_f32_bytes_hasher.take().ok_or_else(|| {
                OrchestratorError::StreamProtocol(format!(
                    "finish_tensor_chunks: tensor `{}` is missing its raw evidence hasher",
                    planned.name
                ))
            })?;
            let converted_logical =
                active.converted_logical_f32_hasher.take().ok_or_else(|| {
                    OrchestratorError::StreamProtocol(format!(
                        "finish_tensor_chunks: tensor `{}` is missing its logical evidence hasher",
                        planned.name
                    ))
                })?;
            let payload = active.payload_hasher.take().ok_or_else(|| {
                OrchestratorError::StreamProtocol(format!(
                    "finish_tensor_chunks: tensor `{}` is missing its payload evidence hasher",
                    planned.name
                ))
            })?;
            completed.push(TensorWriteEvidence {
                plan_index: tensor_idx,
                tensor_name: planned.name.clone(),
                dims_gguf_innermost_first: planned.dims_gguf.clone(),
                ggml_type: planned.ggml_type,
                converted_f32_bytes_sha256: hex::encode(converted_raw.finalize()),
                converted_logical_f32_sha256: converted_logical.finalize().map_err(|error| {
                    OrchestratorError::StreamProtocol(format!(
                        "finish_tensor_chunks: tensor `{}` logical hash failed: {error}",
                        planned.name
                    ))
                })?,
                f16_roundtrip_f32_bytes_sha256: active
                    .f16_roundtrip_f32_bytes_hasher
                    .take()
                    .map(|hasher| hex::encode(hasher.finalize())),
                f16_roundtrip_logical_f32_sha256: active
                    .f16_roundtrip_logical_f32_hasher
                    .take()
                    .map(LogicalF32Hasher::finalize)
                    .transpose()
                    .map_err(|error| {
                        OrchestratorError::StreamProtocol(format!(
                            "finish_tensor_chunks: tensor `{}` roundtrip logical hash failed: {error}",
                            planned.name
                        ))
                    })?,
                payload_sha256: hex::encode(payload.finalize()),
                payload_bytes,
                relative_payload_offset,
            });
        }
        self.next_idx += 1;
        Ok(TensorChunkStats {
            total_elements: active.total_elements,
            chunk_count: active.chunk_count,
            max_chunk_elements: active.max_chunk_elements,
            max_input_f32_bytes: active.max_input_f32_bytes,
            max_f16_roundtrip_f32_bytes: active.max_f16_roundtrip_f32_bytes,
            max_quantized_payload_bytes: active.max_quantized_payload_bytes,
            max_working_vec_bytes: active.max_working_vec_bytes,
        })
    }

    /// Stream one complete tensor. This compatibility wrapper uses the
    /// same begin/chunk/finish protocol as bounded fused-expert writes.
    pub fn stream_tensor(
        &mut self,
        tensor_idx: usize,
        data: &[f32],
    ) -> Result<TensorChunkStats, OrchestratorError> {
        self.validate_next_tensor(tensor_idx, "stream_tensor")?;

        let p = &self.planned[tensor_idx];
        if data.len() != p.expected_numel {
            return Err(OrchestratorError::StreamProtocol(format!(
                "stream_tensor: tensor `{}` data length {} != planned numel {}",
                p.name,
                data.len(),
                p.expected_numel
            )));
        }
        self.begin_tensor_chunks(tensor_idx)?;
        if !data.is_empty() {
            self.stream_tensor_chunk(tensor_idx, data)?;
        }
        self.finish_tensor_chunks(tensor_idx)
    }

    /// Seek-back to fill tensor offsets and flush. Must be called after
    /// every planned tensor has been streamed; otherwise the writer
    /// surfaces `WriterError::MissingTensorPayloads` per the existing
    /// `GgufWriter::finalize` contract.
    pub fn finalize(self) -> Result<(), OrchestratorError> {
        self.finalize_inner().map(|_| ())
    }

    /// Finalize the GGUF and return exact tensor evidence only after every
    /// directory offset has been committed successfully.
    pub(crate) fn finalize_with_evidence(
        self,
    ) -> Result<Vec<TensorWriteEvidence>, OrchestratorError> {
        self.finalize_inner()?.ok_or_else(|| {
            OrchestratorError::StreamProtocol(
                "finalize_with_evidence called on an ordinary writer".into(),
            )
        })
    }

    fn finalize_inner(mut self) -> Result<Option<Vec<TensorWriteEvidence>>, OrchestratorError> {
        if let Some(active) = self.active_chunks.as_ref() {
            return Err(OrchestratorError::StreamProtocol(format!(
                "finalize: tensor {} still has an active chunk stream",
                active.tensor_idx
            )));
        }
        if self.next_idx != self.planned.len() {
            return Err(OrchestratorError::StreamProtocol(format!(
                "finalize: only {} of {} planned tensors streamed",
                self.next_idx,
                self.planned.len()
            )));
        }
        self.writer.finalize()?;

        // ADR-033 §P4b coverage report — only emit when an imatrix was
        // attached. Tells the operator how many quantized tensors actually
        // received imatrix calibration. A 0% coverage indicates the
        // imatrix file mapped to no quantized tensors (likely wrong file
        // or wrong arch). Partial coverage is legitimate (e.g. dense
        // tensors using Q6_K which are imatrix-insensitive), but the
        // first 10 missing names are listed so the operator can spot
        // unexpected gaps.
        if self.imatrix.is_some() && self.coverage_quantized > 0 {
            let pct = (self.coverage_with_imatrix as f64 / self.coverage_quantized as f64) * 100.0;
            eprintln!(
                "[hf2q imatrix coverage] {}/{} quantized tensors used imatrix calibration ({:.1}%)",
                self.coverage_with_imatrix, self.coverage_quantized, pct
            );
            if !self.coverage_missing.is_empty() {
                let preview_n = self.coverage_missing.len().min(10);
                eprintln!(
                    "[hf2q imatrix coverage] {} quantized tensor(s) had no matching imatrix entry; first {}:",
                    self.coverage_missing.len(),
                    preview_n
                );
                for name in self.coverage_missing.iter().take(preview_n) {
                    eprintln!("[hf2q imatrix coverage]   - {name}");
                }
                if self.coverage_missing.len() > preview_n {
                    eprintln!(
                        "[hf2q imatrix coverage]   … and {} more",
                        self.coverage_missing.len() - preview_n
                    );
                }
            }
        }

        Ok(self.completed_tensor_evidence)
    }
}

/// Predicate: should this tensor be emitted as F32-raw, skipping the
/// policy / quantizer entirely?
///
/// Mirrors the peer's canonical quantization-eligibility predicate
/// (eligibility false → the source dtype is written unchanged, which
/// for our F32 in-memory representation means F32 on disk).
///
/// **Rules** (inverted from the canonical predicate; we return `true`
/// to mean "keep as F32"):
///
/// 1. `n_dims < 2` — scalars and 1-D vectors are never quantized.
///    Catches `router.scale`, `router.per_expert_scale`,
///    `layer_scalar`, all `*_norm.weight` that happen to be 1-D, etc.
/// 2. Name does NOT end in `.weight` — the "ends with 'weight'" gate.
///    Catches `.scale` sub-name extensions Gemma 4 uses for router
///    scales.
/// 3. Name contains `_norm.weight`.
/// 4. Name contains `ffn_gate_inp.weight` — the router-gate
///    projection is small and stays F32.
/// 5. Name contains `altup` / `laurel` / `per_layer_model_proj` —
///    Gemma3n patterns; benign for arches that don't carry them.
/// 6. Name contains `ssm_conv1d` / `shortconv.conv.weight` /
///    `time_mix_*` / `attn_rel_b.weight` / `.position_embd` /
///    `sam.pos_embd` / `sam.neck.` / `sam.net_` / `.rel_pos` /
///    `.patch_embd` / `.patch_merger`.
/// 7. Gemma 4 synthesized `rope_freqs.weight` — the table carries
///    exact `1.0` / `1e30` magic values; quantizing would saturate
///    `1e30` to inf (F16) or zero (Q4_0). Already covered by rule
///    (3) `_norm.weight` ? No — `rope_freqs.weight` doesn't contain
///    `_norm`, but it IS 1-D so rule (1) catches it.  Keep the
///    explicit rule too as a load-bearing comment anchor.
///
/// **NOT included** (intentionally — the peer quantizes these):
///   - `output.weight` is quantized by default (only kept F32 when
///     `--quantize-output-tensor 0`).
///   - `token_embd.weight` is quantized normally.
///   - Per-layer dense `mlp.{gate,up,down}_proj.weight` always quantized.
fn is_f32_keep_tensor(name: &str, n_dims: usize) -> bool {
    // Rule (1): scalars + 1-D vectors.
    if n_dims < 2 {
        return true;
    }
    // Rule (2): names not ending in `.weight` (Gemma 4 emits
    // `.scale` sub-names that lack the `.weight` suffix).
    if !name.ends_with(".weight") {
        return true;
    }
    // Rules (3)-(7): substring patterns. Same order as the canonical
    // predicate for readability.
    //
    // BERT positional + token-type embeddings — the canonical rule is
    // an exact match on the bare `position_embd.weight` /
    // `token_types.weight` names for BERT-family arches. The later
    // `.position_embd` substring rule catches multimodal SAM-style
    // names with a leading dot (e.g. `v.position_embd`), so it does
    // NOT cover this case.
    name == "position_embd.weight"       // BERT
        || name == "token_types.weight"  // BERT
        || name.contains("_norm.weight")        // (3)
        || name.contains("ffn_gate_inp.weight") // (4)
        || name.contains("altup")        // (5)
        || name.contains("laurel")       // (5)
        || name.contains("per_layer_model_proj") // (5)
        || name.contains("ssm_conv1d")   // (6)
        || name.contains("shortconv.conv.weight") // (6)
        || name.contains("time_mix_first.weight")
        || name.contains("time_mix_w0.weight")
        || name.contains("time_mix_w1.weight")
        || name.contains("time_mix_w2.weight")
        || name.contains("time_mix_v0.weight")
        || name.contains("time_mix_v1.weight")
        || name.contains("time_mix_v2.weight")
        || name.contains("time_mix_a0.weight")
        || name.contains("time_mix_a1.weight")
        || name.contains("time_mix_a2.weight")
        || name.contains("time_mix_g1.weight")
        || name.contains("time_mix_g2.weight")
        || name.contains("time_mix_decay_w1.weight")
        || name.contains("time_mix_decay_w2.weight")
        || name.contains("time_mix_lerp_fused.weight")
        || name.contains("attn_rel_b.weight")  // (6)
        || name.contains(".position_embd") // (6)
        || name.contains("sam.pos_embd")   // (6)
        || name.contains("sam.neck.")      // (6)
        || name.contains("sam.net_")       // (6)
        || name.contains(".rel_pos")       // (6)
        || name.contains(".patch_embd")    // (6)
        || name.contains(".patch_merger")  // (6)
        || name == "rope_freqs.weight" // (7) Gemma 4 synthesized
}

// -----------------------------------------------------------------------------
// Synthetic-end-to-end driver — usable from integration tests + adhoc probes.
// -----------------------------------------------------------------------------

/// A staged-tensor record for the synthetic driver. Carries the full
/// F32 payload so tests can plumb everything in one buffer (the streaming
/// driver pulls data on-demand from `HfModelSource::iter_tensors`).
#[derive(Debug, Clone)]
pub struct StagedTensor {
    pub name: String,
    /// GGUF-order shape (innermost-first).
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub source_dtype: SourceDtype,
    pub layer_index: Option<usize>,
}

/// Run a self-contained synthetic conversion: plan all tensors, then
/// stream their payloads in plan order. Equivalent to driving the
/// orchestrator directly; exists so callers (P3 integration tests,
/// downstream P4 driver tests) have one stable entry point.
pub fn convert_synthetic<W: Write + Seek>(
    ftype: GgufFtype,
    arch: ArchName,
    hparams: HParams,
    metadata: Vec<(String, MetaValue)>,
    tensors: Vec<StagedTensor>,
    writer: W,
) -> Result<(), OrchestratorError> {
    let mut orch = ConvertOrchestrator::new(ftype, arch, hparams);
    for (k, v) in metadata {
        orch.add_metadata(k, v);
    }
    let entries: Vec<PlanEntry> = tensors
        .iter()
        .map(|t| PlanEntry {
            name: t.name.clone(),
            shape: t.shape.clone(),
            source_dtype: t.source_dtype,
            layer_index: t.layer_index,
        })
        .collect();
    orch.plan_tensors(entries)?;
    let mut sw = orch.begin_write(writer)?;
    for (idx, t) in tensors.iter().enumerate() {
        sw.stream_tensor(idx, &t.data)?;
    }
    sw.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;

    fn deterministic_data(n: usize, seed: u32) -> Vec<f32> {
        // Cheap deterministic generator — avoids pulling in `rand` and
        // keeps fixtures reproducible across host architectures.
        (0..n)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as i32;
                (x as f32) / (i32::MAX as f32)
            })
            .collect()
    }

    fn default_hparams() -> HParams {
        // Llama-3-8B-shaped synthetic: 32 heads, 8 KV heads → n_gqa = 4.
        HParams {
            n_expert: 0,
            n_head: 32,
            n_head_kv: 8,
            n_layer: 32,
            n_mtp_layers: 0,
        }
    }

    /// Acceptance test 1 — smoke. Four hand-crafted F32 tensors run
    /// through the orchestrator at Q5_K_M, then re-parsed via the
    /// existing `mlx_native::gguf::GgufFile` reader. Asserts:
    ///   - tensor count / metadata count round-trip
    ///   - every tensor name + ggml_type round-trips
    ///   - the policy-picked types match expectations for Q5_K_M.
    #[test]
    fn smoke_q5_k_m_round_trip_via_reader() {
        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ5_K_M,
            ArchName::Llama3,
            default_hparams(),
        );

        orch.add_metadata(
            "general.architecture".to_string(),
            MetaValue::String("llama".into()),
        );
        orch.add_metadata("general.alignment".to_string(), MetaValue::U32(32));

        let n_per_row = 256usize;
        let shape = vec![n_per_row, 1];
        let entries = vec![
            PlanEntry {
                name: "token_embd.weight".into(),
                shape: shape.clone(),
                source_dtype: SourceDtype::F32,
                layer_index: None,
            },
            PlanEntry {
                name: "output.weight".into(),
                shape: shape.clone(),
                source_dtype: SourceDtype::F32,
                layer_index: None,
            },
            PlanEntry {
                name: "blk.0.attn_q.weight".into(),
                shape: shape.clone(),
                source_dtype: SourceDtype::F32,
                layer_index: Some(0),
            },
            PlanEntry {
                name: "blk.0.ffn_down.weight".into(),
                shape: shape.clone(),
                source_dtype: SourceDtype::F32,
                layer_index: Some(0),
            },
        ];
        let datas = vec![
            deterministic_data(n_per_row, 1),
            deterministic_data(n_per_row, 2),
            deterministic_data(n_per_row, 3),
            deterministic_data(n_per_row, 4),
        ];
        orch.plan_tensors(entries).expect("plan");

        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let f = std::fs::File::create(tmp.path()).unwrap();
            let mut sw = orch.begin_write(f).expect("begin_write");
            for (idx, d) in datas.iter().enumerate() {
                sw.stream_tensor(idx, d).expect("stream");
            }
            sw.finalize().expect("finalize");
        }

        // Round-trip via the canonical reader (mlx-native).
        let gguf =
            mlx_native::gguf::GgufFile::open(tmp.path()).expect("mlx_native parses our GGUF");

        assert_eq!(gguf.metadata_count(), 2);
        assert_eq!(gguf.metadata_string("general.architecture"), Some("llama"));
        assert_eq!(gguf.metadata_u32("general.alignment"), Some(32));

        assert_eq!(gguf.tensor_count(), 4);

        let token = gguf
            .tensor_info("token_embd.weight")
            .expect("token_embd present");
        let output = gguf.tensor_info("output.weight").expect("output present");
        let attn_q = gguf
            .tensor_info("blk.0.attn_q.weight")
            .expect("attn_q present");
        let ffn_down = gguf
            .tensor_info("blk.0.ffn_down.weight")
            .expect("ffn_down present");

        // Shape round-trip — mlx_native reverses on parse, returns PyTorch order.
        assert_eq!(token.shape, vec![1, 256]);
        assert_eq!(output.shape, vec![1, 256]);
        assert_eq!(attn_q.shape, vec![1, 256]);
        assert_eq!(ffn_down.shape, vec![1, 256]);

        // Per canonical llama-quant.cpp:181 + the has_tied_embeddings
        // detection in plan_tensors: when output.weight IS present in
        // entries, the model is NOT tied, so token_embd hits the
        // non-tied TOKEN_EMBD branch (standard_policy.rs:452) which
        // for Q5_K_M ftype falls through to the default base type Q5_K.
        // Pre-tied-detection-fix this test asserted token=Q6_K (the
        // BROKEN promote-as-tied behavior).
        assert_eq!(
            token.ggml_type,
            mlx_native::GgmlType::Q5_K,
            "token_embd (non-tied) → Q5_K"
        );
        assert_eq!(
            output.ggml_type,
            mlx_native::GgmlType::Q6_K,
            "output → Q6_K"
        );
        assert_eq!(
            attn_q.ggml_type,
            mlx_native::GgmlType::Q5_K,
            "attn_q → Q5_K"
        );
        assert_eq!(
            ffn_down.ggml_type,
            mlx_native::GgmlType::Q6_K,
            "ffn_down (i=0) → Q6_K"
        );

        assert_eq!(token.offset % 32, 0);
        assert_eq!(output.offset % 32, 0);
        assert_eq!(attn_q.offset % 32, 0);
        assert_eq!(ffn_down.offset % 32, 0);
    }

    /// Acceptance test 2 — vision pattern dispatch. A vision-named
    /// tensor must skip [`StandardPolicy::target_for`] entirely and
    /// emit F16 directly.
    #[test]
    fn vision_pattern_emits_f16_skipping_policy() {
        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ5_K_M,
            ArchName::Gemma4Mmproj,
            default_hparams(),
        );
        orch.add_metadata(
            "general.architecture".to_string(),
            MetaValue::String("gemma4_mmproj".into()),
        );

        let n_per_row = 15usize;
        let shape = vec![n_per_row, 2];
        let data_vis = deterministic_data(n_per_row * 2, 7);

        let entries = vec![
            PlanEntry {
                name: "model.visual.patch_embd.weight".into(),
                shape: shape.clone(),
                source_dtype: SourceDtype::F32,
                layer_index: None,
            },
            PlanEntry {
                name: "blk.0.attn_q.weight".into(),
                shape: vec![256, 1],
                source_dtype: SourceDtype::F32,
                layer_index: Some(0),
            },
        ];
        let data_attn = deterministic_data(256, 8);
        orch.plan_tensors(entries).expect("plan");

        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let f = std::fs::File::create(tmp.path()).unwrap();
            let mut sw = orch.begin_write(f).expect("begin_write");
            sw.stream_tensor(0, &data_vis).expect("stream vis");
            sw.stream_tensor(1, &data_attn).expect("stream attn");
            sw.finalize().expect("finalize");
        }

        let gguf =
            mlx_native::gguf::GgufFile::open(tmp.path()).expect("mlx_native parses our GGUF");

        let visual = gguf
            .tensor_info("model.visual.patch_embd.weight")
            .expect("vision tensor present");
        assert_eq!(
            visual.ggml_type,
            mlx_native::GgmlType::F16,
            "vision tensor must emit F16, got {:?}",
            visual.ggml_type
        );
        assert_eq!(visual.byte_len, 60);

        let attn_q = gguf
            .tensor_info("blk.0.attn_q.weight")
            .expect("policy tensor present");
        assert_eq!(
            attn_q.ggml_type,
            mlx_native::GgmlType::Q5_K,
            "non-vision sibling must still route through policy → Q5_K"
        );
    }

    #[test]
    fn bf16_storage_is_dense_and_has_no_quantizer_roundtrip() {
        let mut orch =
            ConvertOrchestrator::new(GgufFtype::BF16, ArchName::Llama3, default_hparams());
        orch.add_metadata(
            "general.architecture".to_string(),
            MetaValue::String("llama".into()),
        );
        let values = vec![1.0_f32, -2.5, 0.25, 7.0];
        orch.plan_tensors(vec![PlanEntry {
            name: "blk.0.attn_q.weight".into(),
            shape: vec![4, 1],
            source_dtype: SourceDtype::BF16,
            layer_index: Some(0),
        }])
        .unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let evidence;
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut writer = orch.begin_write_with_evidence(file).unwrap();
            writer.stream_tensor(0, &values).unwrap();
            evidence = writer.finalize_with_evidence().unwrap();
        }
        assert_eq!(evidence[0].ggml_type, GgmlType::BF16);
        assert_eq!(evidence[0].payload_bytes, 8);
        assert!(evidence[0].f16_roundtrip_f32_bytes_sha256.is_none());
        assert!(evidence[0].f16_roundtrip_logical_f32_sha256.is_none());
    }

    #[test]
    fn ordinary_writer_does_not_allocate_or_update_evidence_hashers() {
        let mut orch =
            ConvertOrchestrator::new(GgufFtype::AllF32, ArchName::Llama3, default_hparams());
        orch.plan_tensors(vec![PlanEntry {
            name: "output_norm.weight".into(),
            shape: vec![4],
            source_dtype: SourceDtype::F32,
            layer_index: None,
        }])
        .unwrap();
        let mut cursor = std::io::Cursor::new(Vec::new());
        let mut writer = orch.begin_write(&mut cursor).unwrap();
        assert!(writer.completed_tensor_evidence.is_none());
        writer.begin_tensor_chunks(0).unwrap();
        let active = writer.active_chunks.as_ref().unwrap();
        assert!(active.converted_f32_bytes_hasher.is_none());
        assert!(active.converted_logical_f32_hasher.is_none());
        assert!(active.payload_hasher.is_none());
        writer
            .stream_tensor_chunk(0, &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        writer.finish_tensor_chunks(0).unwrap();
        writer.finalize().unwrap();
    }

    /// Acceptance test 3 — no-fallback typed error. A non-vision /
    /// non-audio tensor with `n_per_row = 15` at a K-quant ftype must
    /// surface `QuantizeError::NotBlockAligned` instead of silently
    /// demoting to F16.
    #[test]
    fn unquantizable_row_surfaces_typed_error() {
        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ5_K_M,
            ArchName::Llama3,
            default_hparams(),
        );
        orch.add_metadata(
            "general.architecture".to_string(),
            MetaValue::String("llama".into()),
        );

        let n_per_row = 15usize;
        let entries = vec![PlanEntry {
            name: "blk.0.attn_q.weight".into(),
            shape: vec![n_per_row, 1],
            source_dtype: SourceDtype::F32,
            layer_index: Some(0),
        }];

        // The plan-time policy reject is the failure point — no bytes
        // are committed to the sink (begin_write never runs).
        let err = orch.plan_tensors(entries).expect_err("must error");
        match err {
            OrchestratorError::Quantize(QuantizeError::NotBlockAligned {
                n_per_row: 15, ..
            }) => {}
            other => panic!(
                "expected OrchestratorError::Quantize(NotBlockAligned {{ n_per_row: 15, .. }}), got {other:?}"
            ),
        }
    }

    /// `stream_tensor` rejects out-of-order calls.
    #[test]
    fn stream_tensor_rejects_out_of_order() {
        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ5_K_M,
            ArchName::Llama3,
            default_hparams(),
        );
        orch.add_metadata(
            "general.architecture".to_string(),
            MetaValue::String("llama".into()),
        );
        let n_per_row = 256usize;
        let entries = vec![
            PlanEntry {
                name: "blk.0.attn_q.weight".into(),
                shape: vec![n_per_row, 1],
                source_dtype: SourceDtype::F32,
                layer_index: Some(0),
            },
            PlanEntry {
                name: "blk.1.attn_q.weight".into(),
                shape: vec![n_per_row, 1],
                source_dtype: SourceDtype::F32,
                layer_index: Some(1),
            },
        ];
        orch.plan_tensors(entries).expect("plan");
        let mut buf = std::io::Cursor::new(Vec::<u8>::new());
        let mut sw = orch.begin_write(&mut buf).expect("begin_write");
        let data = deterministic_data(n_per_row, 5);
        // Try to stream idx 1 before idx 0 — protocol violation.
        let err = sw.stream_tensor(1, &data).expect_err("must error");
        assert!(
            matches!(err, OrchestratorError::StreamProtocol(_)),
            "got {err:?}"
        );
    }

    /// `stream_tensor` rejects wrong data length.
    #[test]
    fn stream_tensor_rejects_wrong_length() {
        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ5_K_M,
            ArchName::Llama3,
            default_hparams(),
        );
        orch.add_metadata(
            "general.architecture".to_string(),
            MetaValue::String("llama".into()),
        );
        orch.plan_tensors(vec![PlanEntry {
            name: "blk.0.attn_q.weight".into(),
            shape: vec![256, 1],
            source_dtype: SourceDtype::F32,
            layer_index: Some(0),
        }])
        .expect("plan");
        let mut buf = std::io::Cursor::new(Vec::<u8>::new());
        let mut sw = orch.begin_write(&mut buf).expect("begin_write");
        let bogus = deterministic_data(128, 5); // wrong length
        let err = sw.stream_tensor(0, &bogus).expect_err("must error");
        assert!(
            matches!(err, OrchestratorError::StreamProtocol(_)),
            "got {err:?}"
        );
    }

    fn one_deepseek_q2_tensor(shape: Vec<usize>) -> ConvertOrchestrator {
        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ2_K_S,
            ArchName::Deepseek4,
            HParams {
                n_expert: shape.last().copied().unwrap_or(0) as u32,
                n_head: 4,
                n_head_kv: 1,
                n_layer: 1,
                n_mtp_layers: 0,
            },
        );
        orch.plan_tensors(vec![PlanEntry {
            name: "blk.0.ffn_gate_exps.weight".into(),
            shape,
            source_dtype: SourceDtype::Mxfp4E2M1,
            layer_index: Some(0),
        }])
        .expect("plan DeepSeek expert tensor");
        orch
    }

    fn one_deepseek_agentic_tensor(name: &str, shape: Vec<usize>) -> ConvertOrchestrator {
        let mut orch = ConvertOrchestrator::new_deepseek4_agentic_q2(
            ArchName::Deepseek4,
            HParams {
                n_expert: shape.last().copied().unwrap_or(0) as u32,
                n_head: 4,
                n_head_kv: 1,
                n_layer: 1,
                n_mtp_layers: 0,
            },
        );
        orch.plan_tensors(vec![PlanEntry {
            name: name.into(),
            shape,
            source_dtype: SourceDtype::Mxfp4E2M1,
            layer_index: Some(0),
        }])
        .expect("plan DeepSeek agentic tensor");
        orch
    }

    #[test]
    fn deepseek4_agentic_q2_plan_pins_context_path_and_keeps_mixed_experts() {
        let mut orch = ConvertOrchestrator::new_deepseek4_agentic_q2(
            ArchName::Deepseek4,
            HParams {
                n_expert: 256,
                n_head: 64,
                n_head_kv: 1,
                n_layer: 43,
                n_mtp_layers: 0,
            },
        );
        let entry = |name: &str| PlanEntry {
            name: name.into(),
            shape: vec![4096, 256],
            source_dtype: SourceDtype::F32,
            layer_index: name
                .strip_prefix("blk.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|layer| layer.parse().ok()),
        };
        orch.plan_tensors(vec![
            entry("output.weight"),
            entry("token_embd.weight"),
            entry("blk.2.attn_compressor_gate.weight"),
            entry("blk.2.attn_q_b.weight"),
            entry("blk.2.hc_attn_fn.weight"),
            entry("blk.2.indexer.attn_q_b.weight"),
            entry("blk.2.ffn_gate_exps.weight"),
            entry("blk.2.ffn_up_exps.weight"),
            entry("blk.2.ffn_down_exps.weight"),
            entry("blk.2.ffn_down_shexp.weight"),
        ])
        .unwrap();

        let planned: std::collections::HashMap<_, _> = orch
            .planned
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.ggml_type))
            .collect();
        for name in [
            "output.weight",
            "token_embd.weight",
            "blk.2.attn_compressor_gate.weight",
            "blk.2.attn_q_b.weight",
            "blk.2.hc_attn_fn.weight",
            "blk.2.indexer.attn_q_b.weight",
        ] {
            assert_eq!(planned[name], GgmlType::Q8_0, "{name}");
        }
        assert_eq!(planned["blk.2.ffn_gate_exps.weight"], GgmlType::Q2_K);
        assert_eq!(planned["blk.2.ffn_up_exps.weight"], GgmlType::Q2_K);
        assert_eq!(planned["blk.2.ffn_down_exps.weight"], GgmlType::Q3_K);
        assert_eq!(planned["blk.2.ffn_down_shexp.weight"], GgmlType::Q3_K);
    }

    #[test]
    fn q2_k_s_chunked_rows_are_byte_identical_to_whole_tensor() {
        let n_per_row = 256;
        let rows_per_expert = 4;
        let experts = 2;
        let per_expert = n_per_row * rows_per_expert;
        let data = deterministic_data(per_expert * experts, 0x5eed);

        let mut whole = std::io::Cursor::new(Vec::new());
        let whole_evidence;
        {
            let mut sw = one_deepseek_q2_tensor(vec![n_per_row, rows_per_expert, experts])
                .begin_write_with_evidence(&mut whole)
                .unwrap();
            sw.stream_tensor(0, &data).unwrap();
            whole_evidence = sw.finalize_with_evidence().unwrap();
        }

        let mut chunked = std::io::Cursor::new(Vec::new());
        let chunked_evidence;
        {
            let mut sw = one_deepseek_q2_tensor(vec![n_per_row, rows_per_expert, experts])
                .begin_write_with_evidence(&mut chunked)
                .unwrap();
            sw.begin_tensor_chunks(0).unwrap();
            sw.stream_tensor_chunk(0, &data[..per_expert]).unwrap();
            sw.stream_tensor_chunk(0, &data[per_expert..]).unwrap();
            let stats = sw.finish_tensor_chunks(0).unwrap();
            assert_eq!(stats.chunk_count, experts);
            assert_eq!(stats.max_chunk_elements, per_expert);
            assert_eq!(stats.max_input_f32_bytes, per_expert * 4);
            assert_eq!(stats.max_f16_roundtrip_f32_bytes, per_expert * 4);
            assert_eq!(
                stats.max_quantized_payload_bytes,
                rows_per_expert * GgmlType::Q2_K.row_size(n_per_row)
            );
            assert_eq!(
                stats.max_working_vec_bytes,
                stats.max_input_f32_bytes
                    + stats.max_f16_roundtrip_f32_bytes
                    + stats.max_quantized_payload_bytes
            );
            chunked_evidence = sw.finalize_with_evidence().unwrap();
        }

        assert_eq!(chunked.into_inner(), whole.into_inner());
        assert_eq!(chunked_evidence, whole_evidence);
        let evidence = &chunked_evidence[0];
        let mut converted = Sha256::new();
        StreamingWriter::<std::io::Cursor<Vec<u8>>>::update_f32_hasher(&mut converted, &data);
        assert_eq!(
            evidence.converted_f32_bytes_sha256,
            hex::encode(converted.finalize())
        );
        let mut logical_shape: Vec<u64> = vec![n_per_row, rows_per_expert, experts]
            .into_iter()
            .map(|dimension| dimension as u64)
            .collect();
        logical_shape.reverse();
        assert_eq!(
            evidence.converted_logical_f32_sha256,
            crate::core::provenance::tensor_execution::logical_f32_sha256(&logical_shape, &data)
                .unwrap()
        );
        assert!(evidence.f16_roundtrip_f32_bytes_sha256.is_some());
        assert!(evidence.f16_roundtrip_logical_f32_sha256.is_some());
        assert_eq!(
            evidence.payload_bytes,
            u64::try_from(rows_per_expert * experts * GgmlType::Q2_K.row_size(n_per_row)).unwrap()
        );
        assert_eq!(evidence.relative_payload_offset, 0);
    }

    #[test]
    fn agentic_q3_k_expert_down_chunks_are_byte_identical_to_whole_tensor() {
        let n_per_row = 256;
        let rows_per_expert = 4;
        let experts = 3;
        let per_expert = n_per_row * rows_per_expert;
        let shape = vec![n_per_row, rows_per_expert, experts];
        let data = deterministic_data(per_expert * experts, 0x43d0_0003);

        let make = || one_deepseek_agentic_tensor("blk.0.ffn_down_exps.weight", shape.clone());
        let mut whole = std::io::Cursor::new(Vec::new());
        {
            let mut stream = make().begin_write(&mut whole).unwrap();
            stream.stream_tensor(0, &data).unwrap();
            stream.finalize().unwrap();
        }

        let mut chunked = std::io::Cursor::new(Vec::new());
        {
            let mut stream = make().begin_write(&mut chunked).unwrap();
            stream.begin_tensor_chunks(0).unwrap();
            for expert in data.chunks_exact(per_expert) {
                stream.stream_tensor_chunk(0, expert).unwrap();
            }
            let stats = stream.finish_tensor_chunks(0).unwrap();
            assert_eq!(stats.chunk_count, experts);
            assert_eq!(stats.max_chunk_elements, per_expert);
            assert_eq!(
                stats.max_quantized_payload_bytes,
                rows_per_expert * GgmlType::Q3_K.row_size(n_per_row)
            );
            stream.finalize().unwrap();
        }

        assert_eq!(chunked.into_inner(), whole.into_inner());
    }

    #[test]
    fn chunked_stream_preserves_256_expert_order_with_one_expert_live() {
        const EXPERTS: usize = 256;
        const N_PER_ROW: usize = 256;
        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ2_K_S,
            ArchName::Deepseek4,
            HParams {
                n_expert: EXPERTS as u32,
                n_head: 4,
                n_head_kv: 1,
                n_layer: 1,
                n_mtp_layers: 0,
            },
        );
        let name = "blk.0.ffn_gate_tid2eid.weight";
        orch.plan_tensors(vec![PlanEntry {
            name: name.into(),
            shape: vec![N_PER_ROW, 1, EXPERTS],
            source_dtype: SourceDtype::I64,
            layer_index: Some(0),
        }])
        .unwrap();

        let mut out = std::io::Cursor::new(Vec::new());
        {
            let mut sw = orch.begin_write(&mut out).unwrap();
            sw.begin_tensor_chunks(0).unwrap();
            for expert in 0..EXPERTS {
                let one_expert = vec![expert as f32; N_PER_ROW];
                sw.stream_tensor_chunk(0, &one_expert).unwrap();
            }
            let stats = sw.finish_tensor_chunks(0).unwrap();
            assert_eq!(stats.chunk_count, EXPERTS);
            assert_eq!(stats.total_elements, EXPERTS * N_PER_ROW);
            assert_eq!(stats.max_chunk_elements, N_PER_ROW);
            sw.finalize().unwrap();
        }

        let bytes = out.into_inner();
        let header_bytes = 24 + 8 + name.len() + 4 + 3 * 8 + 4 + 8;
        let data_start = (header_bytes + 31) / 32 * 32;
        let payload = &bytes[data_start..data_start + EXPERTS * N_PER_ROW * 4];
        for (index, word) in payload.chunks_exact(4).enumerate() {
            assert_eq!(
                i32::from_le_bytes(word.try_into().unwrap()),
                (index / N_PER_ROW) as i32,
                "expert-major ordering changed at element {index}"
            );
        }
    }

    #[test]
    fn official_deepseek_expert_chunk_has_a_66_625_mib_working_set_bound() {
        const EXPERTS: usize = 256;
        const ROWS: usize = 2048;
        const N_PER_ROW: usize = 4096;
        let decoded_f32 = ROWS * N_PER_ROW * std::mem::size_of::<f32>();
        let f16_roundtrip_f32 = decoded_f32;
        let q2_payload = ROWS * GgmlType::Q2_K.row_size(N_PER_ROW);
        let bounded_peak = decoded_f32 + f16_roundtrip_f32 + q2_payload;
        let former_whole_tensor_peak = bounded_peak * EXPERTS;

        assert_eq!(decoded_f32, 32 * 1024 * 1024);
        assert_eq!(q2_payload, 2_752_512);
        assert_eq!(bounded_peak, 69_861_376); // 66.625 MiB
        assert_eq!(former_whole_tensor_peak / bounded_peak, EXPERTS);
        assert_eq!(former_whole_tensor_peak, 17_884_512_256);
    }

    #[test]
    fn chunked_stream_rejects_misaligned_and_incomplete_input() {
        let mut out = std::io::Cursor::new(Vec::new());
        let mut sw = one_deepseek_q2_tensor(vec![256, 2, 2])
            .begin_write(&mut out)
            .unwrap();
        sw.begin_tensor_chunks(0).unwrap();
        let err = sw.stream_tensor_chunk(0, &[0.0; 1280]).unwrap_err();
        assert!(matches!(err, OrchestratorError::StreamProtocol(_)));
        let err = sw.stream_tensor_chunk(0, &[0.0; 255]).unwrap_err();
        assert!(matches!(err, OrchestratorError::StreamProtocol(_)));
        sw.stream_tensor_chunk(0, &[0.0; 256]).unwrap();
        let err = sw.finish_tensor_chunks(0).unwrap_err();
        assert!(matches!(err, OrchestratorError::StreamProtocol(_)));
    }

    /// `finalize` rejects incomplete streaming.
    #[test]
    fn finalize_rejects_incomplete_stream() {
        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ5_K_M,
            ArchName::Llama3,
            default_hparams(),
        );
        orch.add_metadata(
            "general.architecture".to_string(),
            MetaValue::String("llama".into()),
        );
        orch.plan_tensors(vec![
            PlanEntry {
                name: "blk.0.attn_q.weight".into(),
                shape: vec![256, 1],
                source_dtype: SourceDtype::F32,
                layer_index: Some(0),
            },
            PlanEntry {
                name: "blk.1.attn_q.weight".into(),
                shape: vec![256, 1],
                source_dtype: SourceDtype::F32,
                layer_index: Some(1),
            },
        ])
        .expect("plan");
        let mut buf = std::io::Cursor::new(Vec::<u8>::new());
        let sw = orch.begin_write(&mut buf).expect("begin_write");
        // Drop without streaming anything — finalize should reject.
        let err = sw.finalize().expect_err("must error");
        assert!(
            matches!(err, OrchestratorError::StreamProtocol(_)),
            "got {err:?}"
        );
    }

    // ----- adjacent unit-level sanity tests (cheap; cover internals) -----

    #[test]
    fn convert_synthetic_entry_point_works() {
        let metadata = vec![(
            "general.architecture".to_string(),
            MetaValue::String("llama".into()),
        )];
        let tensors = vec![StagedTensor {
            name: "blk.0.attn_q.weight".into(),
            shape: vec![256, 1],
            data: deterministic_data(256, 11),
            source_dtype: SourceDtype::F32,
            layer_index: Some(0),
        }];

        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let f = std::fs::File::create(tmp.path()).unwrap();
            convert_synthetic(
                GgufFtype::MostlyQ5_K_M,
                ArchName::Llama3,
                default_hparams(),
                metadata,
                tensors,
                f,
            )
            .expect("convert_synthetic");
        }
        let gguf = mlx_native::gguf::GgufFile::open(tmp.path()).expect("parse");
        assert_eq!(gguf.tensor_count(), 1);
        let t = gguf.tensor_info("blk.0.attn_q.weight").unwrap();
        assert_eq!(t.ggml_type, mlx_native::GgmlType::Q5_K);
    }

    /// Regression test for ADR-033 §P1 quality-equivalence gate failure
    /// (2026-05-19, findings doc §8). Pre-fix, the pre-pass counted every
    /// tensor classified as `FfnDown` toward `n_ffn_down`, which for a MoE
    /// architecture inflates the denominator threefold (per layer:
    /// `<L>.ffn_down.weight` + `<L>.ffn_down_exps.weight` +
    /// `<L>.ffn_down_exps.scale` — all match the substring "ffn_down").
    ///
    /// The canonical quantizer hardcodes
    /// `n_ffn_down = n_ffn_gate = n_ffn_up = hparams.n_layer` precisely
    /// to side-step this. Post-fix, hf2q does the same: the denominator
    /// must equal `n_layer` regardless of how many tensors classify
    /// as FfnDown.
    ///
    /// Construct a 30-layer 128-expert MoE entry list with the three
    /// per-layer FfnDown-matching tensors and assert that the resulting
    /// `use_more_bits` boundary lands on the canonical layer set
    /// {0,1,2,5,8,11,14,17,20,23,26,27,28,29} (14 layers TRUE for n=30),
    /// not the broken {0..10, 13, 16, 19, 22, 25, 28} (17 layers TRUE
    /// for the inflated n=90 case).
    #[test]
    fn moe_ffn_down_use_more_bits_uses_n_layer_not_counted_tensors() {
        const N_LAYER: u32 = 30;
        let hparams = HParams {
            n_expert: 128,
            n_head: 8,
            n_head_kv: 1,
            n_layer: N_LAYER,
            n_mtp_layers: 0,
        };
        let mut orch =
            ConvertOrchestrator::new(GgufFtype::MostlyQ5_K_M, ArchName::Gemma4, hparams);
        orch.add_metadata(
            "general.architecture".to_string(),
            MetaValue::String("gemma4".into()),
        );

        // Build entries that mirror Gemma 4's per-layer FfnDown matches.
        // Use n_per_row=256 to avoid shape_fallback (Q5_K block_size=256),
        // so the type we get out is the raw `target_for` decision — no
        // legacy-quant downshift to confuse the assertion.
        let n_per_row = 256usize;
        let shape = vec![n_per_row, 64];
        let mut entries: Vec<PlanEntry> = Vec::new();
        for li in 0..N_LAYER as usize {
            // The .scale tensor is the silent extra match — it triples
            // the pre-pass count if the fix is reverted.
            entries.push(PlanEntry {
                name: format!("blk.{li}.ffn_down.weight"),
                shape: shape.clone(),
                source_dtype: SourceDtype::F32,
                layer_index: Some(li),
            });
            entries.push(PlanEntry {
                name: format!("blk.{li}.ffn_down_exps.scale"),
                shape: vec![128usize, 1],
                source_dtype: SourceDtype::F32,
                layer_index: Some(li),
            });
            entries.push(PlanEntry {
                name: format!("blk.{li}.ffn_down_exps.weight"),
                shape: shape.clone(),
                source_dtype: SourceDtype::F32,
                layer_index: Some(li),
            });
        }
        orch.plan_tensors(entries).expect("plan");

        // Canonical use_more_bits(i, 30) TRUE-set for the Q5_K_M Q6_K
        // promotion on ffn_down. See
        // `docs/adr-033-real-model-findings/2026-05-19-quality-equivalence-gemma4-26b.md`
        // §8.2 for the bartowski/canonical agreement on this set.
        let canonical_promoted: std::collections::HashSet<usize> =
            [0, 1, 2, 5, 8, 11, 14, 17, 20, 23, 26, 27, 28, 29]
                .into_iter()
                .collect();

        // Walk planned tensors and assert: for each blk.<i>.ffn_down.weight
        // the picked type is Q6_K iff i ∈ canonical_promoted.
        let mut q6k_layers: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for pt in orch.planned.iter() {
            if pt.name.ends_with(".ffn_down.weight") && pt.name.starts_with("blk.") {
                let layer: usize = pt
                    .name
                    .strip_prefix("blk.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse().ok())
                    .expect("layer parse");
                if matches!(pt.ggml_type, crate::quantize::ggml_quants::GgmlType::Q6_K) {
                    q6k_layers.insert(layer);
                }
            }
        }

        assert_eq!(
            q6k_layers, canonical_promoted,
            "ffn_down Q6_K promotion must match canonical use_more_bits(i, n_layer=30); \
             pre-fix bug had n_ffn_down=90 and would produce 17 layers including {{3,4,6,7,9,10,13,16,19,22,25}}."
        );
    }

    /// Regression test for bug #2 (docs §10.2): attn_v use_more_bits counter
    /// advances on visit order, but hf2q's HfModelSource iterates safetensors
    /// in lexical name order (blk.0, blk.1, blk.10, blk.11, ..., blk.2, ...).
    /// Canonical visits in numeric layer order. Result: 12 attn_v Q5_K↔Q6_K
    /// swaps vs canonical on a 30-layer MoE pre-fix.
    ///
    /// Fix: orchestrator sorts entries by canonical (group, layer, name)
    /// before the policy walk. This test simulates the buggy input order
    /// (lexical) and asserts the result matches canonical's numeric-order
    /// promotion set {0, 1, 2, 9, 12, 15, 18, 21, 24, 26, 27, 28, 29} for
    /// 30-layer use_more_bits over attn_v. (Note: this is the use_more_bits
    /// canonical TRUE-set with n_attention_wv=30; differs from ffn_down's
    /// {0,1,2,5,8,11,14,17,20,23,26,27,28,29} because the boundary computes
    /// differently for incremented-vs-parsed indices.)
    #[test]
    fn attn_v_visit_order_sorts_to_canonical_numeric_layer_order() {
        const N_LAYER: u32 = 30;
        let hparams = HParams {
            n_expert: 128,
            n_head: 8,
            n_head_kv: 1,
            n_layer: N_LAYER,
            n_mtp_layers: 0,
        };
        let mut orch =
            ConvertOrchestrator::new(GgufFtype::MostlyQ5_K_M, ArchName::Gemma4, hparams);
        orch.add_metadata(
            "general.architecture".to_string(),
            MetaValue::String("gemma4".into()),
        );

        // n_per_row=256 so Q5_K and Q6_K are block-aligned (no fallback)
        let n_per_row = 256usize;
        let shape = vec![n_per_row, 64];

        // Build entries in LEXICAL order — the bug input pattern from
        // HfModelSource on real Gemma 4 safetensors.
        let layers_lex: Vec<u32> = {
            let mut v: Vec<u32> = (0..N_LAYER).collect();
            v.sort_by_key(|n| format!("{n}"));
            v
        };
        let mut entries: Vec<PlanEntry> = Vec::new();
        for li in layers_lex {
            entries.push(PlanEntry {
                name: format!("blk.{li}.attn_v.weight"),
                shape: shape.clone(),
                source_dtype: SourceDtype::F32,
                layer_index: Some(li as usize),
            });
        }
        orch.plan_tensors(entries).expect("plan");

        // Walk planned tensors. For each blk.<N>.attn_v.weight assert the
        // type matches use_more_bits(N, 30) → Q6_K, else Q5_K.
        fn use_more_bits(i: u32, n: u32) -> bool {
            i < n / 8 || i >= 7 * n / 8 || (i.saturating_sub(n / 8)) % 3 == 2
        }
        for pt in orch.planned.iter() {
            let layer: u32 = pt
                .name
                .strip_prefix("blk.")
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse().ok())
                .expect("layer parse");
            let expected = if use_more_bits(layer, N_LAYER) {
                crate::quantize::ggml_quants::GgmlType::Q6_K
            } else {
                crate::quantize::ggml_quants::GgmlType::Q5_K
            };
            assert_eq!(
                pt.ggml_type, expected,
                "blk.{layer}.attn_v.weight: visit-order-sorted plan must produce \
                 the canonical use_more_bits(layer, 30) type. Pre-fix bug would \
                 produce Q5_K↔Q6_K swaps due to lexical iteration order."
            );
        }
    }

    #[test]
    fn empty_conversion_writes_header_only_gguf() {
        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ5_K_M,
            ArchName::Llama3,
            default_hparams(),
        );
        orch.plan_tensors(Vec::new()).expect("plan empty");
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let mut f = std::fs::File::create(tmp.path()).unwrap();
            let sw = orch.begin_write(&mut f).expect("begin_write");
            sw.finalize().expect("finalize empty");
            f.flush().unwrap();
        }
        let gguf = mlx_native::gguf::GgufFile::open(tmp.path()).expect("parse");
        assert_eq!(gguf.tensor_count(), 0);
        assert_eq!(gguf.metadata_count(), 0);
    }

    // ----------------------------------------------------------------
    // ADR-033 §P4b regression tests
    //
    // These tests pin the invariant that the convert orchestrator
    // actually applies attached `ImatrixData` to the quantizer dispatch.
    // The bug discovered 2026-05-22 — apex-i-quality producing
    // byte-identical output to apex-quality — would have been caught
    // pre-ship if any test asserted that an `--imatrix`-driven convert
    // byte-DIFFERS from a no-imatrix convert.
    //
    // The fix landed in this file (line 602 + `tensor_imatrix` helper)
    // and in `cli_driver.rs` (line 518 region). See
    // `[[project-adr033-p4b-shipped-2026-05-22]]`.
    // ----------------------------------------------------------------

    /// Build a synthetic `ImatrixData` for a single dense weight tensor.
    /// Avoids GGUF round-trip — just constructs the in-memory shape that
    /// `StreamingWriter::tensor_imatrix` expects.
    fn make_dense_imatrix(
        tensor_name: &str,
        n_per_row: usize,
        seed: u32,
    ) -> crate::quantize::imatrix::ImatrixData {
        let mut registry = crate::quantize::imatrix::AccumulatorRegistry::new();
        let acc = registry
            .register(tensor_name, n_per_row, 1)
            .expect("register accumulator");
        // Two absorbed rows so counts[0] = 2 — exercises the `total_counts`
        // path even though dense aggregation is just pass-through.
        let row1 = deterministic_data(n_per_row, seed);
        let row2 = deterministic_data(n_per_row, seed.wrapping_add(1));
        acc.absorb_dense(&row1).expect("absorb 1");
        acc.absorb_dense(&row2).expect("absorb 2");

        let loaded = crate::quantize::imatrix::LoadedImatrix {
            source_path: "<synthetic>".into(),
            datasets: vec!["test".into()],
            chunk_count: 1,
            chunk_size: 512,
            registry,
        };
        crate::quantize::imatrix::ImatrixData {
            loaded,
            provenance: crate::quantize::imatrix::ImatrixProvenance::Computed {
                corpus_label: "test".into(),
                n_ctx: 512,
            },
        }
    }

    /// Build a synthetic MoE `ImatrixData` — accumulator with n_mat > 1.
    /// Exercises `tensor_imatrix`'s MoE aggregation branch (sum-across-mats,
    /// divide by total counts) instead of the dense pass-through.
    fn make_moe_imatrix(
        tensor_name: &str,
        n_per_row: usize,
        n_experts: usize,
        seed: u32,
    ) -> crate::quantize::imatrix::ImatrixData {
        let mut registry = crate::quantize::imatrix::AccumulatorRegistry::new();
        let acc = registry
            .register(tensor_name, n_per_row, n_experts)
            .expect("register MoE accumulator");
        // Absorb one row per expert so counts[expert] = 1 for all experts.
        for expert_id in 0..n_experts {
            let row = deterministic_data(n_per_row, seed.wrapping_add(expert_id as u32));
            acc.absorb_moe(expert_id, &row).expect("absorb moe");
        }
        let loaded = crate::quantize::imatrix::LoadedImatrix {
            source_path: "<synthetic-moe>".into(),
            datasets: vec!["test-moe".into()],
            chunk_count: 1,
            chunk_size: 512,
            registry,
        };
        crate::quantize::imatrix::ImatrixData {
            loaded,
            provenance: crate::quantize::imatrix::ImatrixProvenance::Computed {
                corpus_label: "test-moe".into(),
                n_ctx: 512,
            },
        }
    }

    /// Direct quantizer A/B at Q4_K: same input, same n_per_row, two
    /// `quantize()` calls differing only in whether imatrix is passed.
    /// Establishes that the underlying quantizer kernel actually
    /// produces different output bytes when imatrix is `Some` — i.e.
    /// the imatrix-aware path exists and is observable.
    ///
    /// Pre-P4b the orchestrator's `stream_tensor` short-circuited this
    /// distinction (always passed `None`), so this kernel-level
    /// difference was never reachable at the convert-API level. With
    /// P4b wired, the orchestrator threads the imatrix through to
    /// `quantizer.quantize` — making the kernel-level diff visible
    /// end-to-end.
    ///
    /// Uses realistic activation magnitudes (large positive sum-of-squares
    /// values) to ensure the K-quant's `sqrt(sigma2 + x²) * qw` weighting
    /// produces a different scale-selection vs. the no-imatrix path.
    /// Small magnitudes can mask the difference because the weighting
    /// degenerates toward uniform.
    #[test]
    fn p4b_q4_k_quantize_differs_with_vs_without_imatrix() {
        use crate::quantize::ggml_quants::ggml_type::GgmlType;
        use crate::quantize::ggml_quants::quantizer::quantizer_for;

        // Q4_K's QK_K = 256 → use n_per_row = 256 (one super-block per row).
        let n_per_row = 256usize;
        // Weights mimicking realistic LLM activations: a mix of small
        // and large magnitudes so the per-block scale search is non-trivial.
        let weights: Vec<f32> = (0..n_per_row)
            .map(|i| {
                let phase = (i as f32) * 0.371;
                phase.sin() * 0.1 + ((i as f32) * 1.7e-3).cos() * 0.05
            })
            .collect();
        // Importance vector with strong variation across columns — a real
        // imatrix's sum-of-squared-activations has heavy tails (some columns
        // ≫ others), so the weighting `sqrt(sigma2 + x²) * qw` reorders
        // the per-block scale optimum.
        let imatrix: Vec<f32> = (0..n_per_row)
            .map(|i| {
                let bucket = i % 16;
                match bucket {
                    0 => 100.0,
                    1..=3 => 10.0,
                    _ => 1.0,
                }
            })
            .collect();

        let q4k = quantizer_for(GgmlType::Q4_K).expect("Q4_K quantizer");

        let bytes_no_imatrix = q4k
            .quantize(&weights, n_per_row, None)
            .expect("quantize no-imatrix");
        let bytes_with_imatrix = q4k
            .quantize(&weights, n_per_row, Some(&imatrix))
            .expect("quantize with-imatrix");

        assert_eq!(
            bytes_no_imatrix.len(),
            bytes_with_imatrix.len(),
            "Q4_K block size invariant"
        );
        assert_ne!(
            bytes_no_imatrix, bytes_with_imatrix,
            "ADR-033 §P4b smoke: Q4_K quantizer's imatrix-aware path \
             must produce different output bytes when imatrix is Some(non-trivial-values). \
             If this test fails, the K-quant kernel itself isn't honoring the imatrix \
             argument (independent of orchestrator wiring)."
        );
    }

    /// End-to-end regression: orchestrator's `stream_tensor` reaches the
    /// quantizer with `Some(imatrix)` when one is attached via
    /// `with_imatrix`. Constructs two convert runs over the SAME tensor
    /// data + ftype, differing only in imatrix attachment, and verifies
    /// the produced GGUF bytes differ.
    ///
    /// Pre-P4b (2026-05-22 vaporware) these were byte-identical because
    /// `orchestrator.rs:602` hardcoded `imatrix=None`. Post-P4b they
    /// MUST differ. If this test ever fails, P4b's wiring has regressed.
    #[test]
    fn p4b_orchestrator_threads_imatrix_through_to_quantizer() {
        let n_per_row = 256usize;
        let tensor_name = "blk.0.ffn_down.weight";

        // Realistic activation magnitudes — see
        // `p4b_q4_k_quantize_differs_with_vs_without_imatrix` for why.
        let data: Vec<f32> = (0..n_per_row)
            .map(|i| {
                let phase = (i as f32) * 0.371;
                phase.sin() * 0.1 + ((i as f32) * 1.7e-3).cos() * 0.05
            })
            .collect();

        // Build an imatrix with heavy-tailed importance to force a
        // non-trivial scale-selection diff at the K-quant layer. Direct
        // construction (not via `make_dense_imatrix`) because we want
        // imatrix values that are NOT just sum-of-squares of `data`.
        let mut registry = crate::quantize::imatrix::AccumulatorRegistry::new();
        let acc = registry
            .register(tensor_name, n_per_row, 1)
            .expect("register");
        // Inject a synthetic heavy-tailed activation pattern.
        let synthetic_row: Vec<f32> = (0..n_per_row)
            .map(|i| {
                let bucket = i % 16;
                match bucket {
                    0 => 10.0,
                    1..=3 => 3.16,
                    _ => 1.0,
                }
            })
            .collect();
        acc.absorb_dense(&synthetic_row).expect("absorb");

        let imatrix = crate::quantize::imatrix::ImatrixData {
            loaded: crate::quantize::imatrix::LoadedImatrix {
                source_path: "<synthetic>".into(),
                datasets: vec!["test".into()],
                chunk_count: 1,
                chunk_size: 512,
                registry,
            },
            provenance: crate::quantize::imatrix::ImatrixProvenance::Computed {
                corpus_label: "test".into(),
                n_ctx: 512,
            },
        };

        let hparams = HParams {
            n_expert: 0,
            n_head: 32,
            n_head_kv: 8,
            n_layer: 32,
            n_mtp_layers: 0,
        };
        let entries = || {
            vec![PlanEntry {
                name: tensor_name.into(),
                shape: vec![n_per_row, 1],
                source_dtype: SourceDtype::F32,
                layer_index: Some(0),
            }]
        };

        // ---- Convert A: no imatrix ----
        let bytes_no_imatrix = {
            let mut orch =
                ConvertOrchestrator::new(GgufFtype::MostlyQ4_K_M, ArchName::Llama3, hparams);
            orch.plan_tensors(entries()).expect("plan A");
            let mut buf = Vec::<u8>::new();
            {
                let mut sw = orch
                    .begin_write(std::io::Cursor::new(&mut buf))
                    .expect("begin_write A");
                sw.stream_tensor(0, &data).expect("stream A");
                sw.finalize().expect("finalize A");
            }
            buf
        };

        // ---- Convert B: same tensor + data, but with imatrix attached ----
        let bytes_with_imatrix = {
            let mut orch =
                ConvertOrchestrator::new(GgufFtype::MostlyQ4_K_M, ArchName::Llama3, hparams)
                    .with_imatrix(Some(imatrix));
            orch.plan_tensors(entries()).expect("plan B");
            let mut buf = Vec::<u8>::new();
            {
                let mut sw = orch
                    .begin_write(std::io::Cursor::new(&mut buf))
                    .expect("begin_write B");
                sw.stream_tensor(0, &data).expect("stream B");
                sw.finalize().expect("finalize B");
            }
            buf
        };

        assert_eq!(
            bytes_no_imatrix.len(),
            bytes_with_imatrix.len(),
            "P4b byte lengths should match (same ftype, same shape)"
        );
        assert_ne!(
            bytes_no_imatrix, bytes_with_imatrix,
            "ADR-033 §P4b regression: orchestrator produced byte-identical \
             output with and without --imatrix attached. The imatrix is \
             not reaching `quantizer.quantize(..., Some(imatrix))`. Check \
             `orchestrator.rs:602` and `cli_driver.rs:518` region. \
             Also verify the policy routed `blk.0.ffn_down.weight` to a \
             K-quant type that consumes imatrix (Q4_K/Q5_K/Q6_K/IQ4_*)."
        );
    }

    /// Pin the `tensor_imatrix` aggregation policy for dense tensors:
    /// the result must equal the raw `Accumulator.values[..n_per_row]`
    /// (sum-of-squared activations), since dense `n_mat == 1` and the
    /// helper returns the slice directly.
    #[test]
    fn p4b_tensor_imatrix_dense_returns_raw_values() {
        let n_per_row = 8usize;
        let tensor_name = "blk.0.attn_q.weight";
        let imatrix = make_dense_imatrix(tensor_name, n_per_row, 100);

        // Expected: sum of squares of the two rows we absorbed.
        let row1 = deterministic_data(n_per_row, 100);
        let row2 = deterministic_data(n_per_row, 101);
        let expected: Vec<f32> = (0..n_per_row)
            .map(|j| row1[j] * row1[j] + row2[j] * row2[j])
            .collect();

        // Construct an empty StreamingWriter just to exercise the helper.
        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ4_K_M,
            ArchName::Llama3,
            default_hparams(),
        )
        .with_imatrix(Some(imatrix));
        orch.plan_tensors(Vec::new()).expect("plan empty");
        let mut buf = Vec::<u8>::new();
        let sw = orch
            .begin_write(std::io::Cursor::new(&mut buf))
            .expect("begin_write");

        let got = sw
            .tensor_imatrix(tensor_name, n_per_row)
            .expect("Result Ok")
            .expect("imatrix slice present");

        assert_eq!(got.len(), n_per_row);
        for j in 0..n_per_row {
            assert!(
                (got[j] - expected[j]).abs() < 1e-6,
                "dense imatrix col {j}: got {} expected {}",
                got[j],
                expected[j]
            );
        }
    }

    /// Pin the `tensor_imatrix` aggregation policy for MoE tensors:
    /// the result must equal `(sum over experts of values[expert*npr + j]) / total_counts`.
    /// This is the row-uniform aggregate that quantize() consumes — losing
    /// per-expert specificity vs. the peer's per-expert imatrix path, but
    /// the first-cut correct behavior for the current quantize() signature.
    #[test]
    fn p4b_tensor_imatrix_moe_aggregates_across_experts() {
        let n_per_row = 4usize;
        let n_experts = 3usize;
        let tensor_name = "blk.0.ffn_gate_exps.weight";
        let imatrix = make_moe_imatrix(tensor_name, n_per_row, n_experts, 200);

        // Reconstruct expected aggregate. Each expert absorbed exactly
        // one row (`absorb_moe` increments counts[expert] by 1), so
        // total_counts = n_experts, and per-column sum of squares.
        let mut expected = vec![0.0_f32; n_per_row];
        for expert_id in 0..n_experts {
            let row = deterministic_data(n_per_row, 200u32.wrapping_add(expert_id as u32));
            for j in 0..n_per_row {
                expected[j] += row[j] * row[j];
            }
        }
        let inv_total = 1.0_f32 / (n_experts as f32);
        for v in expected.iter_mut() {
            *v *= inv_total;
        }

        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ4_K_M,
            ArchName::Llama3,
            default_hparams(),
        )
        .with_imatrix(Some(imatrix));
        orch.plan_tensors(Vec::new()).expect("plan empty");
        let mut buf = Vec::<u8>::new();
        let sw = orch
            .begin_write(std::io::Cursor::new(&mut buf))
            .expect("begin_write");

        let got = sw
            .tensor_imatrix(tensor_name, n_per_row)
            .expect("Result Ok")
            .expect("imatrix slice present");

        assert_eq!(got.len(), n_per_row);
        for j in 0..n_per_row {
            assert!(
                (got[j] - expected[j]).abs() < 1e-6,
                "MoE imatrix col {j}: got {} expected {}",
                got[j],
                expected[j]
            );
        }
    }

    /// Missing-tensor returns `Ok(None)` (legitimate gap, recorded as a
    /// coverage miss at finalize). Wrong-n_per_row returns `Err`
    /// (hard error per the no-silent-fallback rule — mis-calibrating
    /// with the wrong importance vector is worse than no calibration).
    #[test]
    fn p4b_tensor_imatrix_missing_returns_none_mismatch_returns_err() {
        let imatrix = make_dense_imatrix("blk.0.attn_q.weight", 16, 7);
        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ4_K_M,
            ArchName::Llama3,
            default_hparams(),
        )
        .with_imatrix(Some(imatrix));
        orch.plan_tensors(Vec::new()).expect("plan empty");
        let mut buf = Vec::<u8>::new();
        let sw = orch
            .begin_write(std::io::Cursor::new(&mut buf))
            .expect("begin_write");

        // Wrong tensor name → Ok(None) (legitimate gap).
        let res = sw.tensor_imatrix("blk.99.attn_q.weight", 16);
        assert!(matches!(res, Ok(None)));

        // Right tensor, wrong n_per_row → Err(ApplyShapeMismatch). Hard fail
        // — silently downgrading would mis-calibrate the quantizer.
        let res = sw.tensor_imatrix("blk.0.attn_q.weight", 32);
        match res {
            Err(crate::quantize::imatrix::ImatrixError::ApplyShapeMismatch {
                tensor,
                imatrix_n_per_row,
                model_n_per_row,
            }) => {
                assert_eq!(tensor, "blk.0.attn_q.weight");
                assert_eq!(imatrix_n_per_row, 16);
                assert_eq!(model_n_per_row, 32);
            }
            other => panic!("expected ApplyShapeMismatch, got {other:?}"),
        }

        // Sanity: right name + n_per_row returns Ok(Some(_)).
        let res = sw.tensor_imatrix("blk.0.attn_q.weight", 16);
        assert!(matches!(res, Ok(Some(_))));
    }

    /// `with_imatrix(None)` (or omitted altogether) yields `Ok(None)` for
    /// every lookup — proves the no-imatrix path is intact.
    #[test]
    fn p4b_no_imatrix_attached_returns_none_for_everything() {
        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ4_K_M,
            ArchName::Llama3,
            default_hparams(),
        );
        orch.plan_tensors(Vec::new()).expect("plan empty");
        let mut buf = Vec::<u8>::new();
        let sw = orch
            .begin_write(std::io::Cursor::new(&mut buf))
            .expect("begin_write");

        assert!(matches!(
            sw.tensor_imatrix("blk.0.attn_q.weight", 16),
            Ok(None)
        ));
        assert!(matches!(
            sw.tensor_imatrix("blk.0.ffn_down.weight", 256),
            Ok(None)
        ));
    }

    /// End-to-end: a `stream_tensor` call on a tensor whose imatrix entry
    /// has the wrong `n_per_row` propagates the typed `ImatrixError`
    /// through `OrchestratorError::Imatrix` rather than silently
    /// downgrading to a no-imatrix quantize. Matches the canonical
    /// no-silent-fallback discipline from §P7.
    #[test]
    fn p4b_stream_tensor_propagates_apply_shape_mismatch() {
        let model_n_per_row = 256usize;
        let imatrix_n_per_row = 128usize; // intentionally wrong
        let tensor_name = "blk.0.ffn_down.weight";

        // Build an imatrix with the WRONG n_per_row.
        let mut registry = crate::quantize::imatrix::AccumulatorRegistry::new();
        let acc = registry
            .register(tensor_name, imatrix_n_per_row, 1)
            .expect("register");
        acc.absorb_dense(&vec![1.0_f32; imatrix_n_per_row])
            .expect("absorb");
        let imatrix = crate::quantize::imatrix::ImatrixData {
            loaded: crate::quantize::imatrix::LoadedImatrix {
                source_path: "<bad>".into(),
                datasets: vec!["test".into()],
                chunk_count: 1,
                chunk_size: 512,
                registry,
            },
            provenance: crate::quantize::imatrix::ImatrixProvenance::Computed {
                corpus_label: "test".into(),
                n_ctx: 512,
            },
        };

        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ4_K_M,
            ArchName::Llama3,
            default_hparams(),
        )
        .with_imatrix(Some(imatrix));
        orch.plan_tensors(vec![PlanEntry {
            name: tensor_name.into(),
            shape: vec![model_n_per_row, 1],
            source_dtype: SourceDtype::F32,
            layer_index: Some(0),
        }])
        .expect("plan");

        let mut buf = Vec::<u8>::new();
        let mut sw = orch
            .begin_write(std::io::Cursor::new(&mut buf))
            .expect("begin_write");
        let data: Vec<f32> = (0..model_n_per_row).map(|i| (i as f32) * 1e-3).collect();
        let res = sw.stream_tensor(0, &data);
        match res {
            Err(OrchestratorError::Imatrix(
                crate::quantize::imatrix::ImatrixError::ApplyShapeMismatch {
                    tensor,
                    imatrix_n_per_row: ipr,
                    model_n_per_row: mpr,
                },
            )) => {
                assert_eq!(tensor, tensor_name);
                assert_eq!(ipr, imatrix_n_per_row);
                assert_eq!(mpr, model_n_per_row);
            }
            other => {
                panic!("expected OrchestratorError::Imatrix(ApplyShapeMismatch), got {other:?}")
            }
        }
    }

    /// Coverage accounting: stream a tensor that IS in the imatrix and a
    /// tensor that ISN'T, then check the missing-tensor list at finalize.
    /// Validates that the operator-facing coverage report tracks the
    /// expected counts.
    #[test]
    fn p4b_coverage_tracks_missing_tensors() {
        let n_per_row = 256usize;
        let covered = "blk.0.attn_q.weight";
        let uncovered = "blk.0.ffn_down.weight";

        let imatrix = make_dense_imatrix(covered, n_per_row, 123);

        let mut orch = ConvertOrchestrator::new(
            GgufFtype::MostlyQ4_K_M,
            ArchName::Llama3,
            default_hparams(),
        )
        .with_imatrix(Some(imatrix));
        orch.plan_tensors(vec![
            PlanEntry {
                name: covered.into(),
                shape: vec![n_per_row, 1],
                source_dtype: SourceDtype::F32,
                layer_index: Some(0),
            },
            PlanEntry {
                name: uncovered.into(),
                shape: vec![n_per_row, 1],
                source_dtype: SourceDtype::F32,
                layer_index: Some(0),
            },
        ])
        .expect("plan");

        let mut buf = Vec::<u8>::new();
        let mut sw = orch
            .begin_write(std::io::Cursor::new(&mut buf))
            .expect("begin_write");
        let data: Vec<f32> = (0..n_per_row).map(|i| (i as f32) * 1e-3).collect();
        sw.stream_tensor(0, &data).expect("stream covered");
        sw.stream_tensor(1, &data).expect("stream uncovered");

        // Coverage state should reflect: 2 total quantized, 1 with imatrix,
        // 1 missing — `uncovered`.
        assert_eq!(sw.coverage_quantized, 2);
        assert_eq!(sw.coverage_with_imatrix, 1);
        assert_eq!(sw.coverage_missing, vec![uncovered.to_string()]);

        sw.finalize().expect("finalize");
    }
}
