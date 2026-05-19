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
//! 5. [`StreamingWriter::stream_tensor`] — one call per planned tensor,
//!    in plan order. Caller hands over the F32 row-major data; the
//!    orchestrator quantizes inline, streams the payload, and discards
//!    both the F32 + quantized buffers before returning.
//! 6. [`StreamingWriter::finalize`] — seek-back to fill tensor offsets,
//!    flush, return the underlying writer.
//!
//! No silent F16 demotion outside the vision/audio gate — any other
//! quantization / shape failure surfaces as [`OrchestratorError`].

use std::io::{Seek, Write};

use half::f16;

use crate::backends::gguf::types::MetaValue;
use crate::backends::gguf::writer::{GgufWriter, WriterError};
use crate::quantize::ggml_quants::apex::{ApexError, ApexPolicy};
use crate::quantize::ggml_quants::quantizer::Quantizer;
use crate::quantize::ggml_quants::standard_policy::{
    tensor_type_fallback, HParams, LlmType, QsState, StandardPolicy, TensorCategory,
};
use crate::quantize::ggml_quants::{
    is_audio_tensor_pattern, is_vision_tensor_pattern, quantizer_for, ArchName, GgmlType,
    LlamaFtype, QuantizeError, SourceDtype, TensorRef,
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
}

impl std::fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrchestratorError::Quantize(e) => write!(f, "convert/quantize: {e}"),
            OrchestratorError::Apex(e) => write!(f, "convert/apex: {e}"),
            OrchestratorError::Writer(e) => write!(f, "convert/writer: {e}"),
            OrchestratorError::StreamProtocol(s) => write!(f, "convert/stream-protocol: {s}"),
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
        }
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

/// Pipeline driver for the new ADR-033 convert path. See module-level
/// docs for the lifecycle contract.
///
/// Policy selection: by default the orchestrator routes per-tensor type
/// decisions through [`StandardPolicy::target_for`] (mirroring
/// llama.cpp's `llama_tensor_get_type_impl`). When constructed via
/// [`new_with_apex`] it routes through [`ApexPolicy::target_for`]
/// instead — used by `--quant apex-<tier>` on the convert-v2 CLI per
/// ADR-033 §"Plan" / Pa. The shape-misalignment fallback
/// ([`tensor_type_fallback`]) runs for both policies; only the
/// type-pick algorithm changes.
pub struct ConvertOrchestrator {
    ftype: LlamaFtype,
    arch: ArchName,
    hparams: HParams,
    /// `Some` when an Apex tier was selected; mutually exclusive with
    /// the StandardPolicy path. The two policies cannot run in the same
    /// convert invocation.
    apex_policy: Option<ApexPolicy>,
    metadata: Vec<(String, MetaValue)>,
    /// Populated by `plan_tensors`. Empty before planning, in plan order
    /// during/after planning. Drained slot-by-slot during streaming.
    planned: Vec<PlannedTensor>,
}

impl ConvertOrchestrator {
    /// Construct an orchestrator pinned to one `ftype` / `arch` / shape.
    ///
    /// `hparams` is the per-model `n_expert` / `n_head` / `n_head_kv`
    /// snapshot consumed by [`StandardPolicy::target_for`] for the
    /// counter-walk + `n_gqa` branches. Real callers populate this
    /// from the safetensors-side config; tests pass synthetic values.
    pub fn new(ftype: LlamaFtype, arch: ArchName, hparams: HParams) -> Self {
        Self {
            ftype,
            arch,
            hparams,
            apex_policy: None,
            metadata: Vec::new(),
            planned: Vec::new(),
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
        ftype: LlamaFtype,
        arch: ArchName,
        hparams: HParams,
        apex_policy: ApexPolicy,
    ) -> Self {
        Self {
            ftype,
            arch,
            hparams,
            apex_policy: Some(apex_policy),
            metadata: Vec::new(),
            planned: Vec::new(),
        }
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
    pub fn plan_tensors(
        &mut self,
        entries: Vec<PlanEntry>,
    ) -> Result<(), OrchestratorError> {
        if !self.planned.is_empty() {
            return Err(OrchestratorError::StreamProtocol(format!(
                "plan_tensors called twice (already planned {} tensors)",
                self.planned.len()
            )));
        }

        // -----------------------------------------------------------------
        // Pre-pass: count attn_v tensors and hardcode n_ffn_{down,gate,up}
        // to hparams.n_layer.
        //
        // Mirrors `init_quantize_state_counters` at
        // `/opt/llama.cpp/src/llama-quant.cpp:837-852`:
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

        let policy = StandardPolicy::new();

        // Per-tensor: pick ggml_type. Payload size is also recorded for
        // stream-time validation. No payload bytes consumed here.
        let mut planned: Vec<PlannedTensor> = Vec::with_capacity(entries.len());
        for e in &entries {
            let dims_gguf: Vec<u64> = e.shape.iter().map(|&d| d as u64).collect();
            let expected_numel: usize = e.shape.iter().product();
            let n_per_row = e.shape[0];

            let ggml_type = if is_vision_tensor_pattern(&e.name)
                || is_audio_tensor_pattern(&e.name)
            {
                // Vision / audio modality gate — emit F16 directly.
                // Per ADR-033 Decision §"Vision / audio tensor patterns",
                // this is the ONLY place outside the policy where a
                // ggml_type is chosen, and the ONLY place where F16
                // demotion is permitted.
                GgmlType::F16
            } else if is_f32_keep_tensor(&e.name, e.shape.len()) {
                // F32-keep gate — emit the F32 row-major payload as-is.
                // Mirrors llama.cpp's `tensor_allows_quantization`
                // predicate at `llama-quant.cpp:285-353`. See
                // `is_f32_keep_tensor` doc for the rule list.
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
                    None => policy.target_for(&mut qs, &tref, category)?,
                }
            };

            planned.push(PlannedTensor {
                name: e.name.clone(),
                dims_gguf,
                ggml_type,
                expected_numel,
                n_per_row,
            });
        }

        self.planned = planned;
        Ok(())
    }

    /// Number of planned tensors. Zero before `plan_tensors` runs.
    pub fn planned_count(&self) -> usize {
        self.planned.len()
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
        let Self {
            metadata,
            planned,
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
        })
    }
}

/// Streaming GGUF writer returned by [`ConvertOrchestrator::begin_write`].
///
/// Owns the underlying sink + the plan. Each call to `stream_tensor`
/// quantizes one tensor's F32 data, emits its payload, and discards
/// both buffers before returning. Peak resident set per call:
/// `expected_numel × 4 bytes` (F32) + the quantized payload.
pub struct StreamingWriter<W: Write + Seek> {
    writer: GgufWriter<W>,
    planned: Vec<PlannedTensor>,
    next_idx: usize,
}

impl<W: Write + Seek> StreamingWriter<W> {
    /// Number of tensors remaining to stream.
    pub fn tensors_remaining(&self) -> usize {
        self.planned.len() - self.next_idx
    }

    /// Total number of planned tensors (constant for the lifetime of
    /// the writer).
    pub fn planned_count(&self) -> usize {
        self.planned.len()
    }

    /// Stream one tensor's F32 row-major data. The orchestrator
    /// quantizes inline, writes the payload, and drops the quantized
    /// buffer before returning. The caller's F32 buffer is borrowed
    /// (not consumed) and may be dropped after this returns.
    ///
    /// Must be called in plan order — the orchestrator validates this
    /// via `tensor_idx == self.next_idx` and rejects out-of-order calls
    /// per the no-loop-suppression rule.
    ///
    /// `data.len()` must equal the planned tensor's
    /// `shape.iter().product()`; otherwise this returns
    /// [`OrchestratorError::StreamProtocol`].
    pub fn stream_tensor(
        &mut self,
        tensor_idx: usize,
        data: &[f32],
    ) -> Result<(), OrchestratorError> {
        if tensor_idx >= self.planned.len() {
            return Err(OrchestratorError::StreamProtocol(format!(
                "stream_tensor: idx {tensor_idx} out of range (planned {})",
                self.planned.len()
            )));
        }
        if tensor_idx != self.next_idx {
            return Err(OrchestratorError::StreamProtocol(format!(
                "stream_tensor: out-of-order call (got idx {tensor_idx}, expected {})",
                self.next_idx
            )));
        }

        let p = &self.planned[tensor_idx];
        if data.len() != p.expected_numel {
            return Err(OrchestratorError::StreamProtocol(format!(
                "stream_tensor: tensor `{}` data length {} != planned numel {}",
                p.name,
                data.len(),
                p.expected_numel
            )));
        }

        // Build payload inline — three branches mirror the policy
        // decisions made in `plan_tensors`:
        //   - F16: vision/audio gate, fixed F16 cast (2 bytes/elem).
        //   - F32: rope_freqs.weight & co. (4 bytes/elem, identity).
        //   - Other: `quantizer_for(ggml_type).quantize(...)`.
        let payload: Vec<u8> = match p.ggml_type {
            GgmlType::F16 => {
                let mut v = Vec::with_capacity(data.len() * 2);
                for &x in data {
                    v.extend_from_slice(&f16::from_f32(x).to_le_bytes());
                }
                v
            }
            GgmlType::F32 => {
                let mut v = Vec::with_capacity(data.len() * 4);
                for &x in data {
                    v.extend_from_slice(&x.to_le_bytes());
                }
                v
            }
            _ => {
                let quantizer = quantizer_for(p.ggml_type)?;
                quantizer.quantize(data, p.n_per_row, None)?
            }
        };

        self.writer.stream_tensor_payload(tensor_idx, &payload)?;
        self.next_idx += 1;
        Ok(())
    }

    /// Seek-back to fill tensor offsets and flush. Must be called after
    /// every planned tensor has been streamed; otherwise the writer
    /// surfaces `WriterError::MissingTensorPayloads` per the existing
    /// `GgufWriter::finalize` contract.
    pub fn finalize(mut self) -> Result<(), OrchestratorError> {
        if self.next_idx != self.planned.len() {
            return Err(OrchestratorError::StreamProtocol(format!(
                "finalize: only {} of {} planned tensors streamed",
                self.next_idx,
                self.planned.len()
            )));
        }
        self.writer.finalize()?;
        Ok(())
    }
}

/// Predicate: should this tensor be emitted as F32-raw, skipping the
/// policy / quantizer entirely?
///
/// Mirrors llama.cpp's canonical `tensor_allows_quantization` at
/// `/opt/llama.cpp/src/llama-quant.cpp:285-353` (`quantize` returning
/// false → caller writes the source dtype unchanged, which for our F32
/// in-memory representation means F32 on disk).
///
/// **Rules** (inverted from llama-quant.cpp; we return `true` to mean
/// "keep as F32"):
///
/// 1. `n_dims < 2` — scalars and 1-D vectors are never quantized
///    (per `llama-quant.cpp:293`).  Catches `router.scale`,
///    `router.per_expert_scale`, `layer_scalar`, all `*_norm.weight`
///    that happen to be 1-D, etc.
/// 2. Name does NOT end in `.weight` — `llama-quant.cpp:298`
///    "ends with 'weight'" gate.  Catches `.scale` sub-name extensions
///    Gemma 4 uses for router scales.
/// 3. Name contains `_norm.weight` — `llama-quant.cpp:301`.
/// 4. Name contains `ffn_gate_inp.weight` — `llama-quant.cpp:307`,
///    the router-gate projection is small and stays F32.
/// 5. Name contains `altup` / `laurel` / `per_layer_model_proj` —
///    `llama-quant.cpp:310-314` (Gemma3n; benign for arches that
///    don't carry these patterns).
/// 6. Name contains `ssm_conv1d` / `shortconv.conv.weight` /
///    `time_mix_*` / `attn_rel_b.weight` / `.position_embd` /
///    `sam.pos_embd` / `sam.neck.` / `sam.net_` / `.rel_pos` /
///    `.patch_embd` / `.patch_merger` — `llama-quant.cpp:322-352`.
/// 7. Gemma 4 synthesized `rope_freqs.weight` — the table carries
///    exact `1.0` / `1e30` magic values; quantizing would saturate
///    `1e30` to inf (F16) or zero (Q4_0). Already covered by rule
///    (3) `_norm.weight` ? No — `rope_freqs.weight` doesn't contain
///    `_norm`, but it IS 1-D so rule (1) catches it.  Keep the
///    explicit rule too as a load-bearing comment anchor.
///
/// **NOT included** (intentionally — these ARE quantized in llama.cpp):
///   - `output.weight` is quantized by default (only kept F32 when
///     `--quantize-output-tensor 0` per `llama-quant.cpp:303`).
///   - `token_embd.weight` is quantized normally.
///   - Per-layer dense `mlp.{gate,up,down}_proj.weight` always quantized.
fn is_f32_keep_tensor(name: &str, n_dims: usize) -> bool {
    // Rule (1): scalars + 1-D vectors. llama-quant.cpp:293.
    if n_dims < 2 {
        return true;
    }
    // Rule (2): names not ending in `.weight` (Gemma 4 emits
    // `.scale` sub-names that lack the `.weight` suffix per
    // `gemma.py::format_tensor_name` `suffix=".scale"`). llama-quant.cpp:298.
    if !name.ends_with(".weight") {
        return true;
    }
    // Rules (3)-(7): substring patterns. Same order as llama-quant.cpp
    // for readability.
    name.contains("_norm.weight")        // (3) llama-quant.cpp:301
        || name.contains("ffn_gate_inp.weight") // (4) llama-quant.cpp:307
        || name.contains("altup")        // (5) llama-quant.cpp:310
        || name.contains("laurel")       // (5) llama-quant.cpp:311
        || name.contains("per_layer_model_proj") // (5) llama-quant.cpp:314
        || name.contains("ssm_conv1d")   // (6) llama-quant.cpp:322
        || name.contains("shortconv.conv.weight") // (6) llama-quant.cpp:323
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
        || name.contains("attn_rel_b.weight")  // (6) llama-quant.cpp:343
        || name.contains(".position_embd") // (6) llama-quant.cpp:346
        || name.contains("sam.pos_embd")   // (6) llama-quant.cpp:347
        || name.contains("sam.neck.")      // (6) llama-quant.cpp:348
        || name.contains("sam.net_")       // (6) llama-quant.cpp:349
        || name.contains(".rel_pos")       // (6) llama-quant.cpp:350
        || name.contains(".patch_embd")    // (6) llama-quant.cpp:351
        || name.contains(".patch_merger")  // (6) llama-quant.cpp:352
        || name == "rope_freqs.weight"     // (7) Gemma 4 synthesized
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
    ftype: LlamaFtype,
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
            LlamaFtype::MostlyQ5_K_M,
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

        assert_eq!(token.ggml_type as u32, 6, "token_embd → Q6_K");
        assert_eq!(output.ggml_type as u32, 6, "output → Q6_K");
        assert_eq!(attn_q.ggml_type as u32, 5, "attn_q → Q5_K");
        assert_eq!(ffn_down.ggml_type as u32, 6, "ffn_down (i=0) → Q6_K");

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
            LlamaFtype::MostlyQ5_K_M,
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
            visual.ggml_type as u32, 1,
            "vision tensor must emit F16 (positional code 1), got {}",
            visual.ggml_type as u32
        );
        assert_eq!(visual.byte_len, 60);

        let attn_q = gguf
            .tensor_info("blk.0.attn_q.weight")
            .expect("policy tensor present");
        assert_eq!(
            attn_q.ggml_type as u32, 5,
            "non-vision sibling must still route through policy → Q5_K"
        );
    }

    /// Acceptance test 3 — no-fallback typed error. A non-vision /
    /// non-audio tensor with `n_per_row = 15` at a K-quant ftype must
    /// surface `QuantizeError::NotBlockAligned` instead of silently
    /// demoting to F16.
    #[test]
    fn unquantizable_row_surfaces_typed_error() {
        let mut orch = ConvertOrchestrator::new(
            LlamaFtype::MostlyQ5_K_M,
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
                n_per_row: 15,
                ..
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
            LlamaFtype::MostlyQ5_K_M,
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
            LlamaFtype::MostlyQ5_K_M,
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

    /// `finalize` rejects incomplete streaming.
    #[test]
    fn finalize_rejects_incomplete_stream() {
        let mut orch = ConvertOrchestrator::new(
            LlamaFtype::MostlyQ5_K_M,
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
                LlamaFtype::MostlyQ5_K_M,
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
        assert_eq!(t.ggml_type as u32, 5);
    }

    /// Regression test for ADR-033 §P1 quality-equivalence gate failure
    /// (2026-05-19, findings doc §8). Pre-fix, the pre-pass counted every
    /// tensor classified as `FfnDown` toward `n_ffn_down`, which for a MoE
    /// architecture inflates the denominator threefold (per layer:
    /// `<L>.ffn_down.weight` + `<L>.ffn_down_exps.weight` +
    /// `<L>.ffn_down_exps.scale` — all match the substring "ffn_down").
    ///
    /// Canonical's `init_quantize_state_counters`
    /// (`/opt/llama.cpp/src/llama-quant.cpp:837-852`) hardcodes
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
        };
        let mut orch = ConvertOrchestrator::new(
            LlamaFtype::MostlyQ5_K_M,
            ArchName::Gemma4,
            hparams,
        );
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
        let canonical_promoted: std::collections::HashSet<usize> = [
            0, 1, 2, 5, 8, 11, 14, 17, 20, 23, 26, 27, 28, 29,
        ].into_iter().collect();

        // Walk planned tensors and assert: for each blk.<i>.ffn_down.weight
        // the picked type is Q6_K iff i ∈ canonical_promoted.
        let mut q6k_layers: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for pt in orch.planned.iter() {
            if pt.name.ends_with(".ffn_down.weight") && pt.name.starts_with("blk.") {
                let layer: usize = pt.name
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

    #[test]
    fn empty_conversion_writes_header_only_gguf() {
        let mut orch = ConvertOrchestrator::new(
            LlamaFtype::MostlyQ5_K_M,
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
}
