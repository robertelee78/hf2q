//! `ConvertOrchestrator` — end-to-end driver wiring the new
//! `StandardPolicy` → `GgmlQuantizer` → `GgufWriter` pipeline.
//!
//! Per ADR-033 §P3. The orchestrator owns the buffering / sequencing
//! around the three composable pieces; it does NOT introduce any new
//! quantization or write logic — every byte emitted comes from one of:
//!
//! 1. [`GgufWriter::write_metadata_kv`] for KV pairs,
//! 2. [`GgmlQuantizer::quantize`] for non-vision tensor payloads,
//! 3. `half::f16::from_f32(...).to_le_bytes()` for vision/audio
//!    tensors that match [`is_vision_tensor_pattern`] /
//!    [`is_audio_tensor_pattern`] (the ADR-mandated dispatcher gate).
//!
//! No silent F16 demotion outside the vision/audio gate — any other
//! quantization/shape failure surfaces as [`OrchestratorError`].

use std::io::{Seek, Write};

use half::f16;

use crate::backends::gguf::types::MetaValue;
use crate::backends::gguf::writer::{GgufWriter, WriterError};
use crate::quantize::ggml_quants::apex::{ApexError, ApexPolicy};
use crate::quantize::ggml_quants::standard_policy::{
    tensor_type_fallback, HParams, LlmType, QsState, StandardPolicy, TensorCategory,
};
use crate::quantize::ggml_quants::quantizer::Quantizer;
use crate::quantize::ggml_quants::{
    is_audio_tensor_pattern, is_vision_tensor_pattern, quantizer_for, ArchName, GgmlType,
    LlamaFtype, QuantizeError, SourceDtype, TensorRef,
};

/// Errors raised by [`ConvertOrchestrator::write`]. Wraps the typed
/// errors from the policy / quantizer / writer layers — no silent
/// demotion paths exist anywhere inside `write`.
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
}

impl std::fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrchestratorError::Quantize(e) => write!(f, "convert/quantize: {e}"),
            OrchestratorError::Apex(e) => write!(f, "convert/apex: {e}"),
            OrchestratorError::Writer(e) => write!(f, "convert/writer: {e}"),
        }
    }
}

impl std::error::Error for OrchestratorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OrchestratorError::Quantize(e) => Some(e),
            OrchestratorError::Apex(e) => Some(e),
            OrchestratorError::Writer(e) => Some(e),
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

/// A tensor staged inside the orchestrator before [`write`] runs.
///
/// `shape` is **GGUF order** (innermost dim first; see the
/// [`GgufWriter::reserve_tensor_info`] doc). For a PyTorch-shape weight
/// `[out_dim, in_dim]`, callers reverse to `[in_dim, out_dim]` once at
/// the safetensors → orchestrator boundary; the orchestrator does NOT
/// re-reverse internally. Per ADR-033 §P2 codex-0d28ae3f review.
#[derive(Debug, Clone)]
pub struct StagedTensor {
    pub name: String,
    /// GGUF-order shape (innermost-first). `shape[0]` is `n_per_row`.
    pub shape: Vec<usize>,
    /// F32 row-major data, `shape.iter().product()` elements.
    pub data: Vec<f32>,
    pub source_dtype: SourceDtype,
    pub layer_index: Option<usize>,
}

/// Pipeline driver for the new ADR-033 convert path.
///
/// Lifecycle: [`new`] → repeated [`add_metadata`] + [`add_tensor`] →
/// single [`write`] that drains all staged state into the sink.
///
/// `[`write`] consumes `self`; the orchestrator is single-shot.
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
    tensors: Vec<StagedTensor>,
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
            tensors: Vec::new(),
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
            tensors: Vec::new(),
        }
    }

    /// Stage one GGUF metadata KV pair. Written in insertion order
    /// during [`write`].
    pub fn add_metadata(&mut self, key: String, value: MetaValue) {
        self.metadata.push((key, value));
    }

    /// Stage one tensor. `shape` is GGUF order (innermost first);
    /// `data` is row-major F32 with `shape.iter().product()` elements.
    /// `layer_index` is `Some(i)` for `blk.<i>.*` tensors, `None` for
    /// globals (`token_embd`, `output`, etc.).
    pub fn add_tensor(
        &mut self,
        name: String,
        shape: Vec<usize>,
        data: Vec<f32>,
        source_dtype: SourceDtype,
        layer_index: Option<usize>,
    ) {
        self.tensors.push(StagedTensor {
            name,
            shape,
            data,
            source_dtype,
            layer_index,
        });
    }

    /// Drive the full pipeline:
    ///
    /// 1. Write the GGUF header.
    /// 2. Write every staged metadata KV pair.
    /// 3. For each staged tensor — decide target type (vision/audio
    ///    pattern → F16 directly, else [`StandardPolicy::target_for`]),
    ///    quantize / pack the payload, [`reserve_tensor_info`] the
    ///    GGUF tensor-info entry (with placeholder offset).
    /// 4. Pad to ALIGNMENT and stream every payload.
    /// 5. Finalize (seek-back to fill offsets).
    ///
    /// The split point between steps 3 and 4 is intentional: the writer
    /// needs ALL `reserve_tensor_info` calls before `pad_to_alignment`
    /// (the header layout is dims-info-first, payload-after), so we
    /// produce all payloads up-front into a `Vec<Vec<u8>>` and stream
    /// them after padding.
    pub fn write<W: Write + Seek>(self, writer: W) -> Result<(), OrchestratorError> {
        let Self {
            ftype,
            arch,
            hparams,
            apex_policy,
            metadata,
            tensors,
        } = self;

        // -----------------------------------------------------------------
        // Pre-pass: count attn_v / ffn_down / ffn_gate / ffn_up tensors
        // so QsState has the n_* counters populated before target_for is
        // called per-tensor. Mirrors llama.cpp's count-pass at
        // `llama-quant.cpp:llama_model_quantize_impl` immediately before
        // the per-tensor loop (`llama-quant.cpp:1010-1054`).
        //
        // The counters do NOT count vision/audio tensors — those skip
        // the policy entirely and don't increment any `i_*` counter.
        // -----------------------------------------------------------------
        let mut n_attention_wv: i32 = 0;
        let mut n_ffn_down: i32 = 0;
        let mut n_ffn_gate: i32 = 0;
        let mut n_ffn_up: i32 = 0;
        for t in &tensors {
            if is_vision_tensor_pattern(&t.name) || is_audio_tensor_pattern(&t.name) {
                continue;
            }
            match TensorCategory::classify(&t.name) {
                cat if cat.is_attn_v() => n_attention_wv += 1,
                TensorCategory::FfnDown => n_ffn_down += 1,
                TensorCategory::FfnGate => n_ffn_gate += 1,
                TensorCategory::FfnUp => n_ffn_up += 1,
                _ => {}
            }
        }

        let mut qs = QsState::new(ftype, arch, LlmType::Other, hparams);
        qs.n_attention_wv = n_attention_wv;
        qs.n_ffn_down = n_ffn_down;
        qs.n_ffn_gate = n_ffn_gate;
        qs.n_ffn_up = n_ffn_up;

        let policy = StandardPolicy::new();

        // -----------------------------------------------------------------
        // First pass over tensors: decide types, build payloads. We do
        // this BEFORE touching the writer so that any typed error from
        // the policy / quantizer surfaces before any bytes are written
        // (clean failure mode).
        // -----------------------------------------------------------------
        struct Prepared {
            name: String,
            dims_gguf: Vec<u64>,
            ggml_type: GgmlType,
            payload: Vec<u8>,
        }
        let mut prepared: Vec<Prepared> = Vec::with_capacity(tensors.len());

        for t in &tensors {
            let dims_gguf: Vec<u64> = t.shape.iter().map(|&d| d as u64).collect();

            if is_vision_tensor_pattern(&t.name) || is_audio_tensor_pattern(&t.name) {
                // Vision / audio modality gate — emit F16 directly.
                // Per ADR-033 Decision §"Vision / audio tensor patterns",
                // this is the ONLY place outside the policy where a
                // ggml_type is chosen, and the ONLY place where F16
                // demotion is permitted.
                let mut payload = Vec::with_capacity(t.data.len() * 2);
                for &x in &t.data {
                    payload.extend_from_slice(&f16::from_f32(x).to_le_bytes());
                }
                prepared.push(Prepared {
                    name: t.name.clone(),
                    dims_gguf,
                    ggml_type: GgmlType::F16,
                    payload,
                });
                continue;
            }

            if is_f32_keep_tensor(&t.name) {
                // F32-keep gate — emit the F32 row-major payload as-is.
                //
                // Mirrors llama.cpp's quantize-decision predicate at
                // `/opt/llama.cpp/src/llama-quant.cpp:293-355`:
                //
                //   1. 1-D tensors are never quantized
                //      (`ggml_n_dims(tensor) < 2 → return false`).
                //   2. `rope_freqs.weight` is a 1-D freq-factors table
                //      for Gemma 4's proportional RoPE
                //      (`/opt/llama.cpp/conversion/gemma.py:702-718`);
                //      it carries small/exact magic values like `1e30`
                //      that DO NOT survive quantization or F16 cast.
                //
                // Per [[feedback-no-loop-suppression-2026-05-17]] the
                // gate is a positive-list (canonical name match) rather
                // than a silent shape-based fallthrough — if a different
                // synthesized tensor needs F32-keep, it must be added
                // here explicitly.
                let mut payload = Vec::with_capacity(t.data.len() * 4);
                for &x in &t.data {
                    payload.extend_from_slice(&x.to_le_bytes());
                }
                prepared.push(Prepared {
                    name: t.name.clone(),
                    dims_gguf,
                    ggml_type: GgmlType::F32,
                    payload,
                });
                continue;
            }

            // Build a TensorRef for the policy. Shape is already
            // GGUF-order (innermost first), so we pass `&t.shape`
            // directly; `TensorRef::n_per_row()` reads `shape[0]`.
            let tref = TensorRef {
                name: &t.name,
                shape: &t.shape,
                source_dtype: t.source_dtype,
                arch,
                layer_index: t.layer_index,
            };
            let category = TensorCategory::classify(&t.name);
            // Branch on policy: ApexPolicy if `--quant apex-<tier>`,
            // else StandardPolicy. Both feed through
            // `tensor_type_fallback` for shape misalignment — the Apex
            // policy doesn't apply the fallback internally (its
            // `target_for` returns the unfallback'd algorithmic pick).
            // Per ADR §"shape_fallback contract" the second-misalignment
            // case still surfaces as `QuantizeError::NotBlockAligned`.
            let ggml_type = match &apex_policy {
                Some(ap) => {
                    let picked = ap.target_for(&tref)?;
                    tensor_type_fallback(picked, tref.n_per_row())?
                }
                None => policy.target_for(&mut qs, &tref, category)?,
            };

            // n_per_row = innermost dim per GGUF convention.
            let n_per_row = tref.n_per_row();
            let quantizer = quantizer_for(ggml_type)?;
            let payload = quantizer.quantize(&t.data, n_per_row, None)?;

            prepared.push(Prepared {
                name: t.name.clone(),
                dims_gguf,
                ggml_type,
                payload,
            });
        }

        // -----------------------------------------------------------------
        // Drive the seek-back writer end-to-end.
        // -----------------------------------------------------------------
        let mut w = GgufWriter::new(writer);
        w.write_header(prepared.len() as u64, metadata.len() as u64)?;

        for (k, v) in &metadata {
            w.write_metadata_kv(k, v)?;
        }

        // Reserve tensor-info entries (placeholders for offsets).
        for p in &prepared {
            w.reserve_tensor_info(&p.name, &p.dims_gguf, p.ggml_type)?;
        }

        // Pad to ALIGNMENT, then stream payloads in order.
        w.pad_to_alignment()?;
        for (idx, p) in prepared.iter().enumerate() {
            w.stream_tensor_payload(idx, &p.payload)?;
        }

        w.finalize()?;
        Ok(())
    }
}

/// Predicate: should this tensor be emitted as F32-raw, skipping the
/// policy / quantizer entirely?
///
/// Currently matches the Gemma 4 synthesized `rope_freqs.weight` table
/// (`crate::convert::arch::gemma4::build_synthesized_tensors`). The table
/// is 1-D and carries exact `1.0` / `1e30` magic values; quantizing or
/// even F16-casting it would lose the `1e30` masks (saturates to F16
/// inf or quantizes to zero) and break the proportional-RoPE collapse.
///
/// Future synthesized small-1-D tensors (e.g. `altup` / `laurel` / other
/// `rope_*` tables) should be added by exact-name match — never by a
/// broad `shape.len() == 1` heuristic, which would also swallow norm
/// tensors and corrupt them.
fn is_f32_keep_tensor(name: &str) -> bool {
    // Gemma 4 ROPE_FREQS table — see
    // `src/convert/arch/gemma4.rs::GEMMA4_ROPE_FREQS_TENSOR_NAME`.
    name == "rope_freqs.weight"
}

// -----------------------------------------------------------------------------
// Synthetic-end-to-end driver — usable from integration tests + adhoc probes.
// -----------------------------------------------------------------------------

/// Run a self-contained synthetic conversion: build an orchestrator,
/// seed it with `metadata` + `tensors`, and write to the supplied sink.
/// Equivalent to using [`ConvertOrchestrator`] directly; exists so
/// callers (P3 integration tests, downstream P4 driver tests) have one
/// stable entry point.
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
    for t in tensors {
        orch.add_tensor(t.name, t.shape, t.data, t.source_dtype, t.layer_index);
    }
    orch.write(writer)
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
        }
    }

    /// Acceptance test 1 — smoke. Four hand-crafted F32 tensors run
    /// through the orchestrator at Q5_K_M, then re-parsed via the
    /// existing `mlx_native::gguf::GgufFile` reader. Asserts:
    ///   - tensor count / metadata count round-trip
    ///   - every tensor name + ggml_type round-trips
    ///   - the policy-picked types match expectations for Q5_K_M:
    ///       token_embd.weight (tied) → Q6_K (output branch bump)
    ///       output.weight             → Q6_K (output branch)
    ///       blk.0.attn_q.weight       → Q5_K (passthrough)
    ///       blk.0.ffn_down.weight     → Q6_K (use_more_bits @ i=0)
    #[test]
    fn smoke_q5_k_m_round_trip_via_reader() {
        let mut orch = ConvertOrchestrator::new(
            LlamaFtype::MostlyQ5_K_M,
            ArchName::Llama3,
            default_hparams(),
        );

        // Required GGUF metadata keys for the reader to accept the
        // file. `general.architecture` is the only one the reader
        // strictly needs; we add a second for the count round-trip.
        orch.add_metadata(
            "general.architecture".to_string(),
            MetaValue::String("llama".into()),
        );
        orch.add_metadata("general.alignment".to_string(), MetaValue::U32(32));

        // Tensor shapes are GGUF order (innermost-first). 256-aligned
        // so Q5_K / Q6_K stay on their primary types (no shape
        // fallback). One-row tensors so payloads stay small.
        let n_per_row = 256usize;
        let shape = vec![n_per_row, 1];

        orch.add_tensor(
            "token_embd.weight".into(),
            shape.clone(),
            deterministic_data(n_per_row, 1),
            SourceDtype::F32,
            None,
        );
        orch.add_tensor(
            "output.weight".into(),
            shape.clone(),
            deterministic_data(n_per_row, 2),
            SourceDtype::F32,
            None,
        );
        orch.add_tensor(
            "blk.0.attn_q.weight".into(),
            shape.clone(),
            deterministic_data(n_per_row, 3),
            SourceDtype::F32,
            Some(0),
        );
        orch.add_tensor(
            "blk.0.ffn_down.weight".into(),
            shape.clone(),
            deterministic_data(n_per_row, 4),
            SourceDtype::F32,
            Some(0),
        );

        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let f = std::fs::File::create(tmp.path()).unwrap();
            orch.write(f).expect("orchestrator write");
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

        // Shape round-trip. We wrote GGUF order [256, 1]; the
        // mlx_native reader REVERSES on parse (gguf/mod.rs:1008
        // `shape.reverse()`), so info.shape comes back in PyTorch
        // order [1, 256]. This is an existing reader convention we
        // assert against, not a writer requirement.
        assert_eq!(token.shape, vec![1, 256]);
        assert_eq!(output.shape, vec![1, 256]);
        assert_eq!(attn_q.shape, vec![1, 256]);
        assert_eq!(ffn_down.shape, vec![1, 256]);

        // Type round-trip per StandardPolicy at Q5_K_M:
        // - token_embd (tied) routes to OUTPUT branch → Q6_K
        // - output.weight                              → Q6_K
        // - blk.0.attn_q                               → Q5_K (passthrough)
        // - blk.0.ffn_down (use_more_bits @ i=0)       → Q6_K
        //
        // mlx_native's GgmlType is a position-indexed enum (not
        // repr(wire_code)) — F32=0 F16=1 Q4_0=2 Q8_0=3 Q4_K=4 Q5_K=5
        // Q6_K=6. We compare against those positional codes.
        assert_eq!(token.ggml_type as u32, 6, "token_embd → Q6_K");
        assert_eq!(output.ggml_type as u32, 6, "output → Q6_K");
        assert_eq!(attn_q.ggml_type as u32, 5, "attn_q → Q5_K");
        assert_eq!(ffn_down.ggml_type as u32, 6, "ffn_down (i=0) → Q6_K");

        // Offsets must be ALIGNMENT-aligned per spec.
        assert_eq!(token.offset % 32, 0);
        assert_eq!(output.offset % 32, 0);
        assert_eq!(attn_q.offset % 32, 0);
        assert_eq!(ffn_down.offset % 32, 0);
    }

    /// Acceptance test 2 — vision pattern dispatch. A vision-named
    /// tensor must skip [`StandardPolicy::target_for`] entirely and
    /// emit F16 directly. We assert:
    ///   - the on-disk ggml_type for `model.visual.patch_embd.weight`
    ///     is `F16` (wire code 1), NOT the Q5_K_M policy result.
    ///   - a same-shape non-vision tensor on the same conversion DOES
    ///     go through the policy (sanity that we didn't disable the
    ///     policy entirely).
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

        // Shape with n_per_row=15 — deliberately NOT block-aligned for
        // any K-quant. If the orchestrator wrongly routed this through
        // the policy it would return `NotBlockAligned`. The vision
        // gate must rescue it (F16 has no block-size constraint).
        let n_per_row = 15usize;
        let shape = vec![n_per_row, 2];
        let data = deterministic_data(n_per_row * 2, 7);

        orch.add_tensor(
            "model.visual.patch_embd.weight".into(),
            shape.clone(),
            data,
            SourceDtype::F32,
            None,
        );

        // Sanity sibling — a non-vision tensor that DOES round through
        // the policy. Use a 256-aligned shape so Q5_K is reachable.
        orch.add_tensor(
            "blk.0.attn_q.weight".into(),
            vec![256, 1],
            deterministic_data(256, 8),
            SourceDtype::F32,
            Some(0),
        );

        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let f = std::fs::File::create(tmp.path()).unwrap();
            orch.write(f).expect("orchestrator write");
        }

        let gguf =
            mlx_native::gguf::GgufFile::open(tmp.path()).expect("mlx_native parses our GGUF");

        let visual = gguf
            .tensor_info("model.visual.patch_embd.weight")
            .expect("vision tensor present");
        // mlx_native's GgmlType is positional: F16 is enum position 1.
        assert_eq!(
            visual.ggml_type as u32, 1,
            "vision tensor must emit F16 (positional code 1), got {}",
            visual.ggml_type as u32
        );
        // F16 byte length sanity: 15*2 elems * 2 bytes = 60 bytes.
        assert_eq!(visual.byte_len, 60);

        let attn_q = gguf
            .tensor_info("blk.0.attn_q.weight")
            .expect("policy tensor present");
        // Q5_K is enum position 5 in mlx_native's positional enum.
        assert_eq!(
            attn_q.ggml_type as u32, 5,
            "non-vision sibling must still route through policy → Q5_K"
        );
    }

    /// Acceptance test 3 — no-fallback typed error. A non-vision /
    /// non-audio tensor with `n_per_row = 15` at a K-quant ftype must
    /// surface `QuantizeError::NotBlockAligned` instead of silently
    /// demoting to F16. Per ADR §"shape_fallback contract" +
    /// [[feedback-no-loop-suppression-2026-05-17]].
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

        // n_per_row = 15 — not divisible by 256 (Q5_K), not by 32
        // (Q5_1 first-downshift). The shape_fallback contract demands
        // an `Err(NotBlockAligned)` here rather than silent F16 demotion.
        let n_per_row = 15usize;
        let shape = vec![n_per_row, 1];

        orch.add_tensor(
            "blk.0.attn_q.weight".into(),
            shape,
            deterministic_data(n_per_row, 9),
            SourceDtype::F32,
            Some(0),
        );

        // Write to an in-memory sink so we don't litter the FS on the
        // error path.
        let mut buf = std::io::Cursor::new(Vec::<u8>::new());
        let err = orch.write(&mut buf).expect_err("must error");

        match err {
            OrchestratorError::Quantize(QuantizeError::NotBlockAligned {
                n_per_row: 15,
                ..
            }) => {}
            other => panic!(
                "expected OrchestratorError::Quantize(NotBlockAligned {{ n_per_row: 15, .. }}), got {other:?}"
            ),
        }

        // And the failure mode must be clean: no bytes were committed
        // to the sink (the early prepare-pass error fires BEFORE
        // `write_header`). Sanity that the orchestrator doesn't leave
        // a partial / corrupt GGUF behind for callers to clean up.
        assert!(
            buf.get_ref().is_empty(),
            "expected zero bytes on error path, got {}",
            buf.get_ref().len()
        );
    }

    // ----- adjacent unit-level sanity tests (cheap; cover internals) -----

    #[test]
    fn convert_synthetic_entry_point_works() {
        // Exercises the `convert_synthetic` helper to make sure it
        // wires identically to direct ConvertOrchestrator use.
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
        // mlx_native's GgmlType is positional: Q5_K = 5.
        assert_eq!(t.ggml_type as u32, 5);
    }

    #[test]
    fn empty_conversion_writes_header_only_gguf() {
        // Zero tensors / zero metadata — sanity for the writer
        // edge case (header alignment only).
        let orch = ConvertOrchestrator::new(
            LlamaFtype::MostlyQ5_K_M,
            ArchName::Llama3,
            default_hparams(),
        );
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let mut f = std::fs::File::create(tmp.path()).unwrap();
            orch.write(&mut f).expect("empty write");
            f.flush().unwrap();
        }
        let gguf = mlx_native::gguf::GgufFile::open(tmp.path()).expect("parse");
        assert_eq!(gguf.tensor_count(), 0);
        assert_eq!(gguf.metadata_count(), 0);
    }
}
