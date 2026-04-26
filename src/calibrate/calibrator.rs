//! `Calibrator` trait — orthogonal axis of weight-quantization
//! calibrators (ADR-014 P7 Decision 9).
//!
//! ## Why a trait
//!
//! Pre-ADR-014, calibration logic was tangled with the static
//! quantizer hierarchy (`StaticQuantizer`, `MixedBitQuantizer`,
//! `DwqQuantizer`, `ApexQuantizer` in `src/quantize/`). Each
//! quantizer carried its own optional calibration step bundled with
//! its codebook search. ADR-014 P7 splits these axes:
//!
//! - **Calibrator** (this trait): produces per-tensor calibration
//!   data from a model + corpus. Examples: `None` (no calibration),
//!   `Imatrix` (llama.cpp's importance matrix), `Dwq` (Apple/MLX's
//!   distilled weight quantization).
//! - **OutputFormat** (`src/quantize/output_format.rs`, future iter):
//!   the on-disk codebook (Flat / BitPair / KQuant / KQuantAdaptive).
//!
//! The split lets `(Calibrator, OutputFormat)` compose orthogonally
//! — `Imatrix × KQuant` and `Dwq × BitPair` are both reachable, and
//! the CLI (`--quant imatrix-q4_k_m`, `--quant dwq-4-6`) picks the
//! validated cells.
//!
//! ## This iter — minimal Calibrator API
//!
//! P7 iter-1 lands the trait + [`CalibrationData`] enum +
//! [`NoneCalibrator`] impl. The Imatrix and DWQ calibrator impls land
//! in subsequent P7 iters when forward-pass orchestration is wired
//! (uses ADR-013's `RealActivationCapture::run_calibration_prompt`
//! for qwen35/qwen35moe and `gemma4/forward_cpu.rs` for Gemma-4).
//! The `Layout A` path migration of `dwq.rs` / `dwq_activation.rs`
//! / `sensitivity.rs` / `apex.rs` from `src/quantize/` into
//! `src/calibrate/` lands alongside the DwqCalibrator impl.

use std::collections::HashMap;

use thiserror::Error;

use super::imatrix::Stats as ImatrixStats;

/// Errors from calibration. Wraps the algorithm-specific error types
/// (e.g. [`super::imatrix::ImatrixError`]) under a uniform surface for
/// the calibrator dispatch.
#[derive(Error, Debug)]
pub enum CalibrationError {
    #[error("calibration: forward-pass infrastructure unavailable for arch '{arch}'")]
    ForwardPassUnavailable { arch: String },

    #[error("calibration: corpus is empty")]
    EmptyCorpus,

    #[error("calibration: imatrix algorithm error: {0}")]
    Imatrix(#[from] super::imatrix::ImatrixError),

    #[error("calibration: I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("calibration: {message}")]
    Other { message: String },
}

/// Calibration corpus — token sequences fed through the model's
/// forward pass to produce activation captures.
///
/// **Minimal shape this iter**: token sequences only. Full corpus
/// metadata (tokenizer, sampling parameters, batch size, chunk
/// boundaries) lands when the Imatrix/Dwq calibrators wire forward
/// pass — until then, `Calibrator::calibrate(...)` impls treat the
/// corpus as opaque and the actual forward-pass driver lives in the
/// arch-specific path.
#[derive(Debug, Clone)]
pub struct CalibrationCorpus {
    /// Tokenised calibration text. Each inner `Vec<u32>` is one
    /// "chunk" — typically 512 tokens. `chunks.len()` is the number
    /// of forward-pass invocations the calibrator will perform.
    pub chunks: Vec<Vec<u32>>,
    /// Optional human-readable name (e.g. "wikitext-2 test split").
    /// Saved into [`super::imatrix::ImatrixCollector`]'s `.imatrix`
    /// dataset field on emit.
    pub name: String,
}

impl CalibrationCorpus {
    /// Total token count across all chunks. Used by progress
    /// reporting and the `imatrix.chunk_count` GGUF metadata field.
    pub fn total_tokens(&self) -> usize {
        self.chunks.iter().map(|c| c.len()).sum()
    }

    /// Number of chunks (forward-pass invocations).
    pub fn n_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Whether the corpus has zero usable tokens.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty() || self.chunks.iter().all(|c| c.is_empty())
    }
}

/// Calibration data — the per-tensor output of a calibrator. Consumed
/// by the [`crate::quantize::Quantizer`] codebook search at quantize
/// time.
#[derive(Debug, Clone)]
pub enum CalibrationData {
    /// No calibration was performed — the quantizer falls back to
    /// unweighted MSE / round-to-nearest.
    None,

    /// Per-tensor importance vectors (output of
    /// [`super::imatrix::ImatrixCollector::finalise`]). Each `Vec<f32>`
    /// has length `row_size` (dense) or `row_size × n_experts` (MoE,
    /// expert-major).
    Imatrix(HashMap<String, Vec<f32>>),

    /// Per-tensor importance + raw stats (preserves `Stats { values,
    /// counts }` for round-trip through the GGUF format which encodes
    /// per-expert counts exactly, unlike the legacy format's lossy
    /// collapse).
    ImatrixWithStats(HashMap<String, ImatrixStats>),

    /// DWQ sensitivity scores — per-layer scalar importance that
    /// drives the bit-pair allocation in
    /// [`crate::quantize::dwq::DwqQuantizer`]. Each entry's `Vec<f32>`
    /// is a single value (the layer's sensitivity score) wrapped for
    /// uniform shape with the imatrix variant.
    Dwq(HashMap<String, Vec<f32>>),
}

impl CalibrationData {
    /// Whether this data carries any useful information for the
    /// quantizer (vs. None).
    pub fn is_some(&self) -> bool {
        !matches!(self, CalibrationData::None)
    }

    /// Number of per-tensor entries.
    pub fn len(&self) -> usize {
        match self {
            CalibrationData::None => 0,
            CalibrationData::Imatrix(m) => m.len(),
            CalibrationData::ImatrixWithStats(m) => m.len(),
            CalibrationData::Dwq(m) => m.len(),
        }
    }

    /// Whether the calibration data has zero entries (compatible
    /// with `Self::None` semantics).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Calibrator trait — produces [`CalibrationData`] from a model +
/// calibration corpus.
///
/// **Lifecycle (intended)**:
///
/// 1. Caller builds the calibrator (e.g. `NoneCalibrator::new()`
///    or a future `ImatrixCalibrator::new()`).
/// 2. Caller invokes `calibrate(model, meta, corpus, progress)`.
/// 3. Implementation runs forward pass through the corpus (or
///    no-ops, for `None`), accumulates per-tensor data, and returns
///    [`CalibrationData`].
///
/// **Send + Sync**: calibrators may be invoked from rayon worker
/// threads under a future P3-style parallel calibration loop. The
/// trait bound documents that contract upfront.
///
/// **Forward-pass infrastructure dependency**: implementations that
/// `requires_forward_pass()` use ADR-013's
/// `RealActivationCapture::run_calibration_prompt` for qwen35
/// /qwen35moe and `gemma4/forward_cpu.rs` for Gemma-4. The arch
/// dispatch happens inside the trait impl; the orchestration shell
/// (this trait) is arch-agnostic.
pub trait Calibrator: Send + Sync {
    /// Human-readable calibrator name. Used by logs, the `--quant`
    /// CLI variant resolver (P8), and the imatrix sidecar's
    /// dataset metadata field.
    fn name(&self) -> &'static str;

    /// Whether this calibrator needs a forward pass through the
    /// model. `false` → no-op calibrator (`None`); `true` → reads
    /// activations during forward (`Imatrix`, `Dwq`).
    fn requires_forward_pass(&self) -> bool;

    /// Run the calibration. The exact behaviour depends on the
    /// implementor.
    fn calibrate(
        &mut self,
        model: &crate::ir::lazy::LazyTensorMap,
        meta: &crate::ir::ModelMetadata,
        corpus: &CalibrationCorpus,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<CalibrationData, CalibrationError>;
}

/// `NoneCalibrator` — the no-op calibrator. Returns
/// [`CalibrationData::None`] without touching the model or corpus.
/// Used for uncalibrated quantization paths (`--quant q4_k_m` without
/// imatrix, all the static-quant variants).
#[derive(Debug, Default, Clone, Copy)]
pub struct NoneCalibrator;

impl NoneCalibrator {
    /// Construct.
    pub fn new() -> Self {
        Self
    }
}

impl Calibrator for NoneCalibrator {
    fn name(&self) -> &'static str {
        "none"
    }

    fn requires_forward_pass(&self) -> bool {
        false
    }

    fn calibrate(
        &mut self,
        _model: &crate::ir::lazy::LazyTensorMap,
        _meta: &crate::ir::ModelMetadata,
        _corpus: &CalibrationCorpus,
        _progress: &crate::progress::ProgressReporter,
    ) -> Result<CalibrationData, CalibrationError> {
        // Truly no-op. Doesn't even validate the corpus is non-empty
        // — `None` calibration is the path explicitly chosen when
        // calibration would be unwanted overhead (e.g. running a
        // smoke convert with q4 on a small fixture).
        Ok(CalibrationData::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::lazy::LazyTensorMap;

    fn dummy_metadata() -> crate::ir::ModelMetadata {
        crate::ir::ModelMetadata {
            architecture: "Test".into(),
            model_type: "test".into(),
            param_count: 0,
            hidden_size: 4,
            num_layers: 1,
            layer_types: vec![],
            num_attention_heads: 1,
            num_kv_heads: Some(1),
            vocab_size: 16,
            dtype: "float16".into(),
            shard_count: 1,
            num_experts: None,
            top_k_experts: None,
            intermediate_size: None,
            raw_config: serde_json::Value::Null,
            explicit_layer_types: None,
            full_attention_interval: None,
            attn_output_gate: None,
            head_dim: None,
            partial_rotary_factor: None,
            rope_parameters: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_num_key_heads: None,
            linear_value_head_dim: None,
            linear_num_value_heads: None,
            mamba_ssm_dtype: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            output_router_logits: None,
            router_aux_loss_coef: None,
        }
    }

    #[test]
    fn none_calibrator_is_no_op() {
        let mut calibrator = NoneCalibrator::new();
        assert_eq!(calibrator.name(), "none");
        assert!(!calibrator.requires_forward_pass());

        let lazy_map = LazyTensorMap::new();
        let metadata = dummy_metadata();
        let corpus = CalibrationCorpus {
            chunks: vec![vec![1, 2, 3, 4]],
            name: "test".into(),
        };
        let progress = crate::progress::ProgressReporter::new();

        let data = calibrator
            .calibrate(&lazy_map, &metadata, &corpus, &progress)
            .unwrap();
        match data {
            CalibrationData::None => {}
            other => panic!("expected CalibrationData::None, got {other:?}"),
        }
    }

    /// `NoneCalibrator` ignores the corpus entirely — even an empty
    /// corpus is accepted (no `EmptyCorpus` error). This matches the
    /// "truly no-op" contract.
    #[test]
    fn none_calibrator_accepts_empty_corpus() {
        let mut calibrator = NoneCalibrator::new();
        let lazy_map = LazyTensorMap::new();
        let metadata = dummy_metadata();
        let corpus = CalibrationCorpus {
            chunks: vec![],
            name: "empty".into(),
        };
        let progress = crate::progress::ProgressReporter::new();
        assert!(calibrator
            .calibrate(&lazy_map, &metadata, &corpus, &progress)
            .is_ok());
    }

    #[test]
    fn calibration_data_is_some_predicate() {
        assert!(!CalibrationData::None.is_some());
        assert!(CalibrationData::Imatrix(HashMap::new()).is_some());
        assert!(CalibrationData::ImatrixWithStats(HashMap::new()).is_some());
        assert!(CalibrationData::Dwq(HashMap::new()).is_some());
    }

    #[test]
    fn calibration_data_len_and_empty() {
        assert_eq!(CalibrationData::None.len(), 0);
        assert!(CalibrationData::None.is_empty());

        let mut m = HashMap::new();
        m.insert("a".to_string(), vec![1.0_f32]);
        m.insert("b".to_string(), vec![2.0_f32]);
        let data = CalibrationData::Imatrix(m);
        assert_eq!(data.len(), 2);
        assert!(!data.is_empty());
    }

    #[test]
    fn calibration_corpus_introspection() {
        let c = CalibrationCorpus {
            chunks: vec![vec![1, 2, 3], vec![4, 5]],
            name: "test".into(),
        };
        assert_eq!(c.total_tokens(), 5);
        assert_eq!(c.n_chunks(), 2);
        assert!(!c.is_empty());

        let empty = CalibrationCorpus {
            chunks: vec![],
            name: "empty".into(),
        };
        assert!(empty.is_empty());
        assert_eq!(empty.total_tokens(), 0);
        assert_eq!(empty.n_chunks(), 0);

        // Non-empty chunks vec but all chunks empty — also is_empty.
        let all_empty_chunks = CalibrationCorpus {
            chunks: vec![vec![], vec![]],
            name: "x".into(),
        };
        assert!(all_empty_chunks.is_empty());
        assert_eq!(all_empty_chunks.total_tokens(), 0);
        assert_eq!(all_empty_chunks.n_chunks(), 2);
    }

    /// Trait is object-safe (can be used as `&dyn Calibrator` /
    /// `Box<dyn Calibrator>`) — required for runtime dispatch from
    /// the CLI variant resolver.
    #[test]
    fn calibrator_is_object_safe() {
        let _: Box<dyn Calibrator> = Box::new(NoneCalibrator::new());
    }

    /// Trait + impls are Send + Sync — required for rayon
    /// parallelisation and shared `Arc<dyn Calibrator>` patterns.
    #[test]
    fn calibrator_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NoneCalibrator>();
        assert_send_sync::<Box<dyn Calibrator>>();
    }
}
